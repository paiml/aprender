//! A THIN, stateless MCP `forecast` server over `aprender_forecast::chronos` — the
//! ZERO-SHOT forecaster (D-03).
//!
//! One immutable Chronos-Bolt model behind one tool: no fit, no knobs, no RNG, nothing to
//! store. The weights are compiled into the binary (`CHRONOS_EMBED_DIR` at build time,
//! D-13) or read once at startup from `CHRONOS_MODEL_DIR`; they are NEVER downloaded at
//! cold start. Every bound and every refusal lives in `aprender_forecast::chronos` — this
//! crate re-checks nothing (OPS-03).
//!
//! Same shape as `aprender-mcp-forecast` (D-06/D-08), minus the router pool: a Chronos
//! call is ~18 ms against a shared `Arc<Model>`, so pmcp's per-router server mutex is not
//! the bottleneck it is for a seconds-long Prophet fit (D-12).
//!
//! Ported from `sources/007-chronos-mcp-thin-server/src/lib.rs`.

// schemars' JsonSchema derive (reached through ChronosArgs) and serde_json::json! both
// expand to .unwrap() internally, at file scope where a narrower allow cannot reach. Same
// precedent as aprender-mcp-forecast/src/lib.rs:13.
#![allow(clippy::disallowed_methods)]

use std::sync::Arc;

use aprender_forecast::chronos::{load_model_from_bytes, load_model_from_dir, ChronosArgs};
pub use aprender_forecast::chronos::{Model, ModelLoadError};
use aprender_forecast::ForecastError;
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;

/// The Chronos-Bolt weights staged by `build.rs`.
///
/// Non-empty when the build set `CHRONOS_EMBED_DIR`; EMPTY in a plain build, where
/// [`resolve_model`] falls back to `CHRONOS_MODEL_DIR` read as a runtime path.
pub static EMBEDDED_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.safetensors"));

/// The checkpoint `config.json` staged alongside [`EMBEDDED_WEIGHTS`], and empty exactly
/// when it is (`embedded_markers_are_consistent`).
pub static EMBEDDED_CONFIG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/config.json"));

/// The single tool this server advertises.
pub const TOOL_NAME: &str = "forecast";

/// The tool description, lifted into `tools/list`.
///
/// The coverage sentence is the D-16 honesty requirement: the band is reported at its
/// MEASURED empirical coverage on rolling origins, never at the nominal 80 %.
pub const TOOL_DESCRIPTION: &str = "Zero-shot time-series forecast with Chronos-Bolt: no training, no model to store — send the series, \
 get quantile forecasts back in one call. Pass parallel arrays `ds` (dates, YYYY-MM-DD) and `y` (numbers; `null` for a \
 missing value), `horizon` (future periods) and optionally `freq` (D, W or MS — daily by default). Returns future `ds`, \
 `yhat` (median), `yhat_lower`/`yhat_upper` (10th/90th percentiles) and all nine quantiles (0.1 … 0.9). The model looks at \
 the last 2,048 points. Accepts 4 to 20,000 points. Horizons 1 to 64 are one direct forecast; longer horizons need \
 `allow_long_horizon: true` because the model then rolls its own forecast forward and accuracy degrades past step 64 \
 (max 1,024), and the response then carries a warning naming the rollout. \
 Band honesty: the nominal q10–q90 band covered 0.65 of held-out points across 17 rolling-origin windows (spike 006) — \
 treat `yhat_lower`/`yhat_upper` as a ~65% band, not an 80% one.";

/// Load the model this process serves: embedded bytes if the build staged them, else
/// `CHRONOS_MODEL_DIR` read as a runtime path.
///
/// Read ONCE, at startup, never from a request (T-06-06 / V12).
///
/// # Errors
///
/// [`ModelLoadError`] when no model is available either way, or when the bytes at hand
/// are not a decodable Chronos-Bolt checkpoint.
pub fn resolve_model() -> Result<Model, ModelLoadError> {
    if !EMBEDDED_WEIGHTS.is_empty() {
        return load_model_from_bytes(EMBEDDED_WEIGHTS, EMBEDDED_CONFIG, "embedded");
    }
    let dir = std::env::var_os("CHRONOS_MODEL_DIR").ok_or_else(|| {
        ModelLoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no embedded model in this build and CHRONOS_MODEL_DIR is unset",
        ))
    })?;
    load_model_from_dir(std::path::Path::new(&dir))
}

/// Map the library's refusals onto MCP errors: the caller's fault stays the caller's fault.
#[must_use]
pub fn map_error(e: ForecastError) -> pmcp::Error {
    match e {
        ForecastError::Validation(s) => pmcp::Error::validation(s),
        ForecastError::Internal(s) => pmcp::Error::internal(s),
    }
}

/// One stateless tool over one immutable model.
///
/// The `Arc` is shared, never cloned-into-a-pool: nothing in the forward mutates the
/// model, so concurrent calls simply read it. The forward runs on a blocking thread so
/// the protocol loop is never stalled by an 18 ms — or, at horizon 365, a 0.85 s — CPU
/// burst.
///
/// # Errors
///
/// `pmcp::Error` if the server builder refuses the configuration.
pub fn build_server(model: Arc<Model>, name: &str, version: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ChronosArgs, _, _>(
            TOOL_NAME,
            TOOL_DESCRIPTION,
            move |args, _extra| {
                let model = Arc::clone(&model);
                async move {
                    let response = tokio::task::spawn_blocking(move || {
                        aprender_forecast::chronos::forecast(&model, &args)
                    })
                    .await
                    .map_err(|e| pmcp::Error::internal(format!("forecast task join: {e}")))?
                    .map_err(map_error)?;
                    serde_json::to_value(&response)
                        .map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
                }
            },
        )
        .build()
}

/// The HTTP app: pmcp's streamable-http MCP router at `/mcp` (stateless, JSON responses,
/// localhost-locked CORS) plus a same-origin demo page and two sample datasets.
///
/// The demo page is itself an MCP client — it speaks `initialize` -> `tools/list` ->
/// `tools/call` against `/mcp`, so serving it proves the same door the tests drive.
pub fn http_app(server: Server) -> axum::Router {
    use axum::response::{Html, IntoResponse};
    use axum::routing::get;
    use pmcp::server::streamable_http_server::StreamableHttpServerConfig;

    let server = Arc::new(tokio::sync::Mutex::new(server));
    let config = pmcp::axum::RouterConfig {
        server_config: StreamableHttpServerConfig::stateless(),
        allowed_origins: Some(pmcp::axum::AllowedOrigins::localhost()),
        ..Default::default()
    };
    let mcp = pmcp::axum::router_with_config(server, config);
    axum::Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../static/index.html")) }),
        )
        .route(
            "/sample/peyton",
            get(|| async {
                (
                    [("content-type", "text/csv")],
                    include_str!("../fixtures/peyton_manning.csv"),
                )
                    .into_response()
            }),
        )
        .route(
            "/sample/air",
            get(|| async {
                (
                    [("content-type", "text/csv")],
                    include_str!("../fixtures/air_passengers.csv"),
                )
                    .into_response()
            }),
        )
        .nest("/mcp", mcp)
}

/// The server end to end: the shared-shape invariant and the bounds on every run, the
/// oracle parity, the horizon gate and the f16 bar when the weights are armed.
///
/// # Armed vs unarmed (D-18)
///
/// The four weight-dependent tests carry
/// `#[cfg_attr(not(chronos_weights), ignore = "…")]`, so an unarmed run reports them as
/// `N ignored` WITH the arming reason printed. That is the whole point: the
/// print-and-return style of skip reports `0 ignored` — a silent green — which D-18
/// forbids. The five weights-free tests run on EVERY invocation, armed or not.
///
/// Arming (`CHRONOS_MODEL_DIR`) and embedding (`CHRONOS_EMBED_DIR`) are independent env
/// vars on purpose: embedding f16 weights into the binary does not arm the parity suite,
/// because the oracle comparison needs the f32 weights DIRECTORY.
#[cfg(test)]
mod e2e {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use aprender_forecast::bolt::Config;
    use aprender_forecast::chronos::{
        load_csv, load_model_from_dir, ChronosArgs, ChronosResponse, CHRONOS_MAX_HORIZON,
        CHRONOS_MIN_POINTS,
    };
    use aprender_forecast::{ForecastArgs, ForecastResponse};

    use super::{build_server, http_app, resolve_model, EMBEDDED_CONFIG, EMBEDDED_WEIGHTS};

    /// The JSON-RPC error code pmcp 2.19.3 emits for a refusal, pinned by 06-06 after
    /// being READ OFF a live reply. It is `-32603` (JSON-RPC *internal error*) for BOTH
    /// classes, because `pmcp::Error::error_code()` returns `None` for `Validation` and
    /// `Internal` alike — so the code cannot discriminate and [`VALIDATION_PREFIX`] is
    /// what actually does.
    const REFUSAL_CODE: i64 = -32603;

    /// The class marker that separates "you sent something I cannot use" from "I broke".
    /// `map_error` routes `ForecastError::Validation` here; an `"Internal error: "` on any
    /// refusal below is a defect, not a wording choice.
    const VALIDATION_PREFIX: &str = "Validation error: ";

    /// The Peyton Manning series std, from the oracle's own `scale` field. The f16 bar is
    /// RELATIVE to it (`equations.f16_rel_std` = 2 % of the series std).
    const PEYTON_STD: f64 = 0.8718;

    // ---------------------------------------------------------------------------------
    // Contract and fixture readers.
    //
    // `aprender_forecast::test_support` is `pub(crate)`, so this crate carries its own
    // small reader over the same parser rather than widening that API. Tolerances and
    // bounds are READ from the contract, never written as literals: a test that hardcodes
    // one can be loosened without the contract ever noticing (D-15).
    // ---------------------------------------------------------------------------------

    /// Read a dotted path (e.g. `equations.f16_rel_std.float_tolerance`) out of
    /// `contracts/<contract>.yaml` as an `f64`.
    fn contract_f64(contract: &str, path: &str) -> f64 {
        let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts")
            .join(format!("{contract}.yaml"));
        let raw = std::fs::read_to_string(&file).unwrap_or_else(|e| {
            panic!("contract {contract} must exist at {}: {e}", file.display())
        });
        let doc: serde_yaml::Value = serde_yaml::from_str(&raw)
            .unwrap_or_else(|e| panic!("contract {contract} is not valid YAML: {e}"));
        let mut node = &doc;
        for key in path.split('.') {
            node = node
                .get(key)
                .unwrap_or_else(|| panic!("contract {contract} must define {path}"));
        }
        node.as_f64()
            .unwrap_or_else(|| panic!("contract {contract}.{path} must be a number"))
    }

    /// A committed fixture from the sibling `aprender-forecast` crate, parsed as JSON.
    ///
    /// Absence is a DEFECT, never a reason to skip: a parity test that silently does not
    /// run proves nothing (CLAUDE.md Verification Discipline #5).
    fn forecast_fixture(name: &str) -> serde_json::Value {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-forecast/tests/fixtures")
            .join(name);
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("fixture {name} is committed; absence is a defect ({e})"));
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture {name} is not JSON: {e}"))
    }

    /// The f32 quantile bar, ARCHITECTURE-KEYED (REVIEW-06-02), printed with the ARCH it
    /// chose and the measurement it is being applied to.
    ///
    /// aarch64 asserts the SC4 literal 1.0e-6, measured 9.54e-7 through the spike-007
    /// server. Anywhere else — every CI job in this repository — asserts the PROVISIONAL,
    /// UNMEASURED `quantiles_abs_f32_nonaarch64`, whose contract entry obliges the first
    /// such run to record the number printed here and tighten the bar to it.
    fn f32_quantile_bar(measured: f64) -> f64 {
        let equation = if cfg!(target_arch = "aarch64") {
            "quantiles_abs_f32"
        } else {
            "quantiles_abs_f32_nonaarch64"
        };
        let bar = contract_f64(
            "chronos-bolt-parity-v1",
            &format!("equations.{equation}.float_tolerance"),
        );
        println!(
            "f32 quantile bar THROUGH THE SERVER: chronos-bolt-parity-v1.{equation} = {bar:e} \
             (ARCH={}); measured max|delta| = {measured:.4e}",
            std::env::consts::ARCH
        );
        bar
    }

    // ---------------------------------------------------------------------------------
    // The MCP client harness — the sibling's, verbatim in behaviour.
    // ---------------------------------------------------------------------------------

    struct Client {
        http: reqwest::Client,
        url: String,
        next: u64,
    }

    impl Client {
        async fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
            self.next += 1;
            let body = serde_json::json!({
                "jsonrpc": "2.0", "id": self.next, "method": method, "params": params
            });
            let r = self
                .http
                .post(&self.url)
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .body(body.to_string())
                .send()
                .await
                .expect("send");
            let status = r.status();
            let text = r.text().await.expect("text");
            assert!(status.is_success(), "{method}: HTTP {status}\n{text}");
            let payload = text
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap_or(&text);
            serde_json::from_str(payload)
                .unwrap_or_else(|e| panic!("{method}: non-JSON: {e}\n{text}"))
        }

        async fn forecast(&mut self, arguments: serde_json::Value) -> serde_json::Value {
            self.call(
                "tools/call",
                serde_json::json!({"name": "forecast", "arguments": arguments}),
            )
            .await
        }
    }

    fn tool_output(v: &serde_json::Value) -> serde_json::Value {
        if let Some(s) = v["result"].get("structuredContent") {
            return s.clone();
        }
        let text = v["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("text content in {v}"));
        serde_json::from_str(text).expect("tool JSON")
    }

    fn f64s(v: &serde_json::Value) -> Vec<f64> {
        v.as_array()
            .unwrap_or_else(|| panic!("array, got {v}"))
            .iter()
            .map(|x| x.as_f64().expect("f64"))
            .collect()
    }

    /// The armed weights directory. `CHRONOS_MODEL_DIR` is what `cfg(chronos_weights)` was
    /// derived from, so its absence inside a gated test is a DEFECT, never a skip.
    fn model_dir() -> PathBuf {
        PathBuf::from(std::env::var_os("CHRONOS_MODEL_DIR").expect(
            "CHRONOS_MODEL_DIR is set whenever cfg(chronos_weights) is armed; absence is a \
             defect, never a reason to skip",
        ))
    }

    /// `<dir>/../<name>` — the `just fetch-chronos-tiny` layout puts `f32` and `f16` side
    /// by side, so the f16 test finds its weights from the armed f32 directory.
    fn sibling(dir: &Path, name: &str) -> PathBuf {
        dir.parent()
            .unwrap_or_else(|| panic!("{} must have a parent", dir.display()))
            .join(name)
    }

    /// Bring up a fresh in-process server on the weights in `dir`, complete `initialize`,
    /// and hand back the client plus the base URL (the demo page lives there).
    async fn serve(dir: &Path) -> (Client, String) {
        let model = load_model_from_dir(dir)
            .unwrap_or_else(|e| panic!("the armed weights at {} must load: {e}", dir.display()));
        let server =
            build_server(Arc::new(model), "aprender-chronos-test", "0.0.0").expect("server builds");
        let app = http_app(server);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });
        let mut c = Client {
            http: reqwest::Client::new(),
            url: format!("http://{addr}/mcp"),
            next: 0,
        };
        let init = c
            .call(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "e2e", "version": "0"}
                }),
            )
            .await;
        assert!(init.get("error").is_none(), "initialize: {init}");
        (c, format!("http://{addr}"))
    }

    /// The Peyton Manning daily series as the tool takes it (nullable `y`).
    fn peyton() -> (Vec<String>, Vec<Option<f64>>) {
        load_csv(include_str!("../fixtures/peyton_manning.csv")).expect("peyton csv")
    }

    /// Worst pointwise disagreement across the nine quantile paths, ours vs the oracle's
    /// `[quantile][step]` matrix.
    fn worst_vs_oracle(out: &serde_json::Value, oracle_matrix: &serde_json::Value) -> f64 {
        let levels = [
            "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
        ];
        let mut worst = 0.0f64;
        for (qi, lvl) in levels.iter().enumerate() {
            let ours = f64s(&out["quantiles"][*lvl]);
            let theirs = f64s(&oracle_matrix[qi]);
            assert_eq!(
                ours.len(),
                theirs.len(),
                "quantile {lvl}: {} steps vs the oracle's {}",
                ours.len(),
                theirs.len()
            );
            for (a, b) in ours.iter().zip(&theirs) {
                worst = worst.max((a - b).abs());
            }
        }
        worst
    }

    /// Assert the server REFUSED, that the refusal is validation-class, and that its
    /// message names the fix.
    fn assert_refused(reply: &serde_json::Value, needle: &str) {
        let is_error =
            reply.get("error").is_some() || reply["result"]["isError"] == serde_json::json!(true);
        assert!(is_error, "must be REFUSED, never defaulted; got: {reply}");
        assert!(
            reply.to_string().contains(needle),
            "the refusal must name the fix ({needle:?}); got: {reply}"
        );
        let message = if let Some(err) = reply.get("error") {
            assert_eq!(
                err["code"].as_i64(),
                Some(REFUSAL_CODE),
                "pinned refusal code; got: {reply}"
            );
            err["message"].as_str().expect("error.message").to_string()
        } else {
            reply["result"]["content"][0]["text"]
                .as_str()
                .expect("isError reply carries text content")
                .to_string()
        };
        assert!(
            message.starts_with(VALIDATION_PREFIX),
            "a caller-fixable input must reach the client as {VALIDATION_PREFIX:?}, never as \
             \"Internal error: \"; got: {message}"
        );
    }

    // =================================================================================
    // GATED — real weights required (D-18 counted skips when unarmed).
    // =================================================================================

    /// The tracer: ONE 64-step Chronos forecast crossing JSON-RPC -> tool schema ->
    /// validation door -> the Bolt forward -> response, over live streamable-HTTP, checked
    /// against the chronos-forecasting 2.3.1 oracle on the ARCH-selected contract bar.
    #[tokio::test]
    // rustfmt::skip keeps the gate on ONE line. `attr_fn_like_width` (70) would otherwise
    // split it, and a split attribute is harder to grep for than the D-18 audit deserves.
    #[rustfmt::skip]
    #[cfg_attr(not(chronos_weights), ignore = "CHRONOS_MODEL_DIR unset; just fetch-chronos-tiny")]
    async fn forecast_tool_over_streamable_http_matches_oracle() {
        let dir = sibling(&model_dir(), "f32");
        let (mut c, base) = serve(&dir).await;

        // The demo page is same-origin and is itself an MCP client.
        let page = reqwest::get(format!("{base}/"))
            .await
            .expect("page")
            .text()
            .await
            .expect("html");
        assert!(
            page.contains("tools/call") && page.contains("forecast"),
            "demo page must be an MCP client"
        );

        let tools = c.call("tools/list", serde_json::json!({})).await;
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|t| t["name"].as_str().expect("name"))
            .collect();
        assert_eq!(names, vec!["forecast"], "exactly one tool");
        let schema = &tools["result"]["tools"][0]["inputSchema"];
        assert!(
            schema["properties"].get("allow_long_horizon").is_some(),
            "the horizon gate must be ADVERTISED, not just enforced: {schema}"
        );
        assert_eq!(
            schema["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must surface in the advertised schema: {schema}"
        );

        let (ds, y) = peyton();
        let r = c
            .forecast(serde_json::json!({"ds": ds, "y": y, "horizon": 64}))
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "forecast failed: {r}"
        );
        let out = tool_output(&r);
        assert_eq!(out["ds"].as_array().expect("ds").len(), 64);
        assert_eq!(out["ds"][0], "2016-01-21");
        assert_eq!(out["context_used"], 2048, "the model's context cap");
        assert!(
            out.get("warning").is_none(),
            "no warning at the native horizon: {out}"
        );
        assert_eq!(
            out["diagnostics"]["weights_dtype"], "F32",
            "the armed directory must be the f32 one: {}",
            out["diagnostics"]
        );

        // yhat/band ARE the quantile map, not a second computation.
        assert_eq!(f64s(&out["yhat"]), f64s(&out["quantiles"]["0.5"]));
        assert_eq!(f64s(&out["yhat_lower"]), f64s(&out["quantiles"]["0.1"]));
        assert_eq!(f64s(&out["yhat_upper"]), f64s(&out["quantiles"]["0.9"]));

        let oracle = forecast_fixture("peyton_tiny_oracle.json");
        let worst = worst_vs_oracle(&out, &oracle["quantiles_64"]);
        let bar = f32_quantile_bar(worst);
        assert!(
            worst <= bar,
            "peyton quantiles_64 THROUGH THE SERVER: max|delta| {worst:e} over the contract bar \
             {bar:e} (ARCH={}) — record this number in the contract if this is the first run on \
             this architecture",
            std::env::consts::ARCH
        );
    }

    /// The D-13 horizon gate, both sides: refused without the flag, warned with it, and
    /// the rollout arithmetic (365 steps = 46 forwards) reported back.
    #[tokio::test]
    // rustfmt::skip keeps the gate on ONE line. `attr_fn_like_width` (70) would otherwise
    // split it, and a split attribute is harder to grep for than the D-18 audit deserves.
    #[rustfmt::skip]
    #[cfg_attr(not(chronos_weights), ignore = "CHRONOS_MODEL_DIR unset; just fetch-chronos-tiny")]
    async fn long_horizon_refused_without_flag_and_warned_with_it() {
        let dir = sibling(&model_dir(), "f32");
        let (mut c, _) = serve(&dir).await;
        let (ds, y) = peyton();

        let refused = c
            .forecast(serde_json::json!({"ds": ds, "y": y, "horizon": 365}))
            .await;
        assert_refused(&refused, "allow_long_horizon");

        let r = c
            .forecast(serde_json::json!({
                "ds": ds, "y": y, "horizon": 365, "allow_long_horizon": true
            }))
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "the flag must ACCEPT: {r}"
        );
        let out = tool_output(&r);
        assert_eq!(out["ds"].as_array().expect("ds").len(), 365);
        assert!(
            out["warning"]
                .as_str()
                .expect("a long horizon must carry a warning")
                .contains("rollout"),
            "the warning must name the rollout: {out}"
        );
        assert_eq!(
            out["diagnostics"]["forwards"], 46,
            "365 steps is 1 direct block + 5 nine-path rollouts = 46 forwards: {}",
            out["diagnostics"]
        );

        let oracle = forecast_fixture("peyton_tiny_oracle.json");
        let worst = worst_vs_oracle(&out, &oracle["quantiles_365"]);
        let bar = contract_f64(
            "chronos-bolt-parity-v1",
            "equations.rollout_365_abs.float_tolerance",
        );
        println!("rollout 365 THROUGH THE SERVER: max|delta| {worst:.4e} against bar {bar:e}");
        assert!(
            worst <= bar,
            "365-step rollout max|delta| {worst:e} over the contract bar {bar:e}"
        );
    }

    /// The refusal set through the server (T-06-01): five perturbations, each one field,
    /// each answered validation-class with the fix named.
    #[tokio::test]
    // rustfmt::skip keeps the gate on ONE line. `attr_fn_like_width` (70) would otherwise
    // split it, and a split attribute is harder to grep for than the D-18 audit deserves.
    #[rustfmt::skip]
    #[cfg_attr(not(chronos_weights), ignore = "CHRONOS_MODEL_DIR unset; just fetch-chronos-tiny")]
    async fn refusals_through_the_server() {
        let dir = sibling(&model_dir(), "f32");
        let (mut c, _) = serve(&dir).await;
        let (ds, y) = peyton();

        // 1. An unknown key is named back, never ignored (deny_unknown_fields).
        assert_refused(
            &c.forecast(serde_json::json!({
                "ds": ds, "y": y, "horizon": 10, "model": "prophet"
            }))
            .await,
            "model",
        );
        // 2. Below CHRONOS_MIN_POINTS.
        assert_refused(
            &c.forecast(serde_json::json!({"ds": ds[..3], "y": y[..3], "horizon": 10}))
                .await,
            "at least",
        );
        // 3. A date that parses as digits but is not a calendar date.
        let mut bad = ds.clone();
        bad[3] = "2008-02-30".into();
        assert_refused(
            &c.forecast(serde_json::json!({"ds": bad, "y": y, "horizon": 10}))
                .await,
            "calendar",
        );
        // 4. Horizon 0.
        assert_refused(
            &c.forecast(serde_json::json!({"ds": ds, "y": y, "horizon": 0}))
                .await,
            "at least 1",
        );
        // 5. Every value missing: nullable `y` is a gap facility, not a way to send nothing.
        let nulls: Vec<Option<f64>> = vec![None; ds.len()];
        assert_refused(
            &c.forecast(serde_json::json!({"ds": ds, "y": nulls, "horizon": 10}))
                .await,
            "non-null",
        );
    }

    /// What the f16 weights COST, measured through the server on the contract's relative
    /// bar — and proof the two runs really read different files.
    #[tokio::test]
    // rustfmt::skip keeps the gate on ONE line. `attr_fn_like_width` (70) would otherwise
    // split it, and a split attribute is harder to grep for than the D-18 audit deserves.
    #[rustfmt::skip]
    #[cfg_attr(not(chronos_weights), ignore = "CHRONOS_MODEL_DIR unset; just fetch-chronos-tiny")]
    async fn f16_weights_within_two_percent_of_std_through_the_server() {
        let armed = model_dir();
        let (ds, y) = peyton();
        let args = serde_json::json!({"ds": ds, "y": y, "horizon": 64});

        let (mut c32, _) = serve(&sibling(&armed, "f32")).await;
        let a = tool_output(&c32.forecast(args.clone()).await);
        let (mut c16, _) = serve(&sibling(&armed, "f16")).await;
        let b = tool_output(&c16.forecast(args).await);

        // A test that accidentally loaded the same file twice would pass trivially.
        assert_eq!(a["diagnostics"]["weights_dtype"], "F32");
        assert_eq!(
            b["diagnostics"]["weights_dtype"], "F16",
            "the f16 sibling must actually be f16: {}",
            b["diagnostics"]
        );

        let mut worst = 0.0f64;
        for lvl in [
            "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
        ] {
            for (x, z) in f64s(&a["quantiles"][lvl])
                .iter()
                .zip(f64s(&b["quantiles"][lvl]))
            {
                worst = worst.max((x - z).abs());
            }
        }
        let rel = contract_f64(
            "chronos-bolt-parity-v1",
            "equations.f16_rel_std.float_tolerance",
        );
        let bar = rel * PEYTON_STD;
        println!(
            "F16 vs F32 THROUGH THE SERVER (Peyton, 64 steps): max|delta| {worst:.4e} = {:.3}% of \
             the series std {PEYTON_STD}; bar {rel} x std = {bar:.4e}",
            worst / PEYTON_STD * 100.0
        );
        assert!(
            worst <= bar,
            "f16 weights moved a quantile by {worst:e}, past {rel} of the series std ({bar:e})"
        );
    }

    // =================================================================================
    // UNGATED — no weights, so these run on EVERY invocation.
    // =================================================================================

    /// The strictness D-11 relies on must be ADVERTISED, not merely enforced.
    #[test]
    fn chronos_args_schema_is_strict() {
        let schema =
            serde_json::to_value(schemars::schema_for!(ChronosArgs)).expect("schema serializes");
        assert_eq!(
            schema["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must surface as additionalProperties: false: {schema}"
        );
        let mut required: Vec<&str> = schema["required"]
            .as_array()
            .expect("required array")
            .iter()
            .map(|v| v.as_str().expect("required entry is a string"))
            .collect();
        required.sort_unstable();
        assert_eq!(
            required,
            vec!["ds", "horizon", "y"],
            "required must be EXACTLY ds, y, horizon"
        );
        let props = schema["properties"].as_object().expect("properties object");
        for optional in ["freq", "allow_long_horizon"] {
            assert!(
                props.contains_key(optional),
                "{optional} must be advertised as an optional property: {schema}"
            );
        }
        // The knobs this server deliberately does NOT take (assumption delta): no fit, no
        // RNG, no decomposition. Advertising them would promise behaviour it has not got.
        for absent in [
            "model",
            "growth",
            "cap",
            "seasonality_mode",
            "interval_width",
            "holidays",
            "n_lags",
            "seed",
        ] {
            assert!(
                !props.contains_key(absent),
                "{absent} belongs to the FIT server, not the zero-shot one: {schema}"
            );
        }
    }

    /// The bounds this server enforces are EQUAL to the boundary contract, not merely
    /// similar (D-15): a bound written twice can be loosened in one place.
    #[test]
    fn chronos_bounds_match_contract() {
        let min = contract_f64("forecast-tool-boundary-v1", "constants.chronos_min_points");
        let max_h = contract_f64("forecast-tool-boundary-v1", "constants.chronos_max_horizon");
        let native = contract_f64(
            "forecast-tool-boundary-v1",
            "constants.chronos_native_horizon",
        );
        assert!(
            (CHRONOS_MIN_POINTS as f64 - min).abs() < f64::EPSILON,
            "chronos_min_points"
        );
        assert!(
            (CHRONOS_MAX_HORIZON as f64 - max_h).abs() < f64::EPSILON,
            "chronos_max_horizon"
        );

        // The native horizon is not a Rust constant at all — it comes off the checkpoint.
        // Asserting it against the contract is what stops a different checkpoint from
        // silently moving the gate the tool description advertises.
        let cfg = Config::from_json(&forecast_fixture("chronos_bolt_tiny_config.json"));
        assert!(
            (cfg.prediction_length as f64 - native).abs() < f64::EPSILON,
            "the checkpoint's prediction_length {} must equal constants.chronos_native_horizon {native}",
            cfg.prediction_length
        );
    }

    /// The assumption-delta companion: every forecast server, whatever it varies, keeps
    /// the SAME door and the same response core.
    ///
    /// The Chronos server opts INTO nullable `y` and `allow_long_horizon` and OUT of every
    /// fit knob; the fit server does the reverse. What must NOT vary is the required
    /// triple and the common response fields — that is what makes "the same `forecast`
    /// tool" a fact rather than a naming convention.
    #[test]
    fn shared_forecast_shape_holds_for_every_server() {
        let required = |schema: &serde_json::Value| -> Vec<String> {
            let mut r: Vec<String> = schema["required"]
                .as_array()
                .expect("required array")
                .iter()
                .map(|v| v.as_str().expect("string").to_string())
                .collect();
            r.sort();
            r
        };
        let fit = serde_json::to_value(schemars::schema_for!(ForecastArgs)).expect("fit schema");
        let zero =
            serde_json::to_value(schemars::schema_for!(ChronosArgs)).expect("chronos schema");
        let want = vec!["ds".to_string(), "horizon".to_string(), "y".to_string()];
        assert_eq!(required(&fit), want, "fit server required set");
        assert_eq!(required(&zero), want, "zero-shot server required set");
        for schema in [&fit, &zero] {
            assert!(
                schema["properties"].get("freq").is_some(),
                "every server advertises `freq`: {schema}"
            );
        }

        // Hand-built responses: no fit, no weights, no I/O — just the serialized shapes.
        let fit_response = ForecastResponse {
            model: "prophet".into(),
            freq: "D".into(),
            n_history: 3,
            fit_seconds: 0.0,
            predict_seconds: 0.0,
            ds: vec!["2020-01-01".into()],
            yhat: vec![1.0],
            yhat_lower: vec![0.0],
            yhat_upper: vec![2.0],
            trend: vec![1.0],
            components: serde_json::Map::new(),
            diagnostics: serde_json::json!({}),
        };
        let zero_response = ChronosResponse {
            model: "chronos-bolt-tiny".into(),
            freq: "D".into(),
            n_history: 3,
            context_used: 3,
            predict_seconds: 0.0,
            ds: vec!["2020-01-01".into()],
            yhat: vec![1.0],
            yhat_lower: vec![0.0],
            yhat_upper: vec![2.0],
            quantiles: serde_json::Map::new(),
            warning: None,
            diagnostics: serde_json::json!({}),
        };
        let keys = |v: &serde_json::Value| -> std::collections::HashSet<String> {
            v.as_object()
                .expect("response serializes to an object")
                .keys()
                .cloned()
                .collect()
        };
        let fit_keys = keys(&serde_json::to_value(&fit_response).expect("fit response"));
        let zero_keys = keys(&serde_json::to_value(&zero_response).expect("chronos response"));
        for field in [
            "model",
            "freq",
            "n_history",
            "predict_seconds",
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "diagnostics",
        ] {
            assert!(
                fit_keys.contains(field),
                "the fit response must carry `{field}`: {fit_keys:?}"
            );
            assert!(
                zero_keys.contains(field),
                "the zero-shot response must carry `{field}`: {zero_keys:?}"
            );
        }
    }

    /// `resolve_model` must take the branch the BUILD chose, and say which one it took.
    ///
    /// Three cases, selected by what this binary was built and run with — every build
    /// exercises exactly one, and Task 3 runs the crate twice to cover the embedded one.
    #[test]
    fn resolve_model_source_matches_build() {
        if EMBEDDED_WEIGHTS.is_empty() {
            match std::env::var_os("CHRONOS_MODEL_DIR") {
                Some(dir) => {
                    let model = resolve_model().unwrap_or_else(|e| {
                        panic!("CHRONOS_MODEL_DIR={dir:?} must resolve as a path: {e}")
                    });
                    assert_eq!(
                        model.source,
                        Path::new(&dir).display().to_string(),
                        "the runtime branch reports the PATH it read"
                    );
                    println!(
                        "resolve_model: runtime path branch — source {}, dtype {}",
                        model.source, model.dtype
                    );
                }
                None => {
                    // `Model` is not `Debug` (it carries the whole weight tensor set), so
                    // `expect_err` is not available — match instead.
                    let Err(e) = resolve_model() else {
                        panic!(
                            "with no embedded weights and no CHRONOS_MODEL_DIR there is nothing \
                             to serve — it must REFUSE, never default"
                        )
                    };
                    assert!(
                        e.to_string().contains("CHRONOS_MODEL_DIR"),
                        "the error must name the variable to set: {e}"
                    );
                    println!("resolve_model: no-model branch — {e}");
                }
            }
        } else {
            let model = resolve_model().expect("the embedded bytes must decode");
            // The embedded branch must win outright: source == "embedded" even when a
            // CHRONOS_MODEL_DIR is also set (Task 3 runs exactly that combination).
            assert_eq!(
                model.source, "embedded",
                "embedded bytes must be preferred over any CHRONOS_MODEL_DIR"
            );
            println!(
                "resolve_model: embedded branch — source {}, dtype {}, {} params, {} weight \
                 bytes + {} config bytes",
                model.source,
                model.dtype,
                model.n_params,
                EMBEDDED_WEIGHTS.len(),
                EMBEDDED_CONFIG.len()
            );
        }
    }

    /// A half-staged embed is a BUILD defect, not a runtime surprise: `resolve_model`
    /// keys on the weights alone, so weights-without-config would reach
    /// `load_model_from_bytes` and fail there with a confusing decode error instead.
    #[test]
    fn embedded_markers_are_consistent() {
        assert_eq!(
            EMBEDDED_WEIGHTS.is_empty(),
            EMBEDDED_CONFIG.is_empty(),
            "build.rs stages model.safetensors and config.json together or neither; got \
             {} weight bytes and {} config bytes",
            EMBEDDED_WEIGHTS.len(),
            EMBEDDED_CONFIG.len()
        );
    }
}
