//! A THIN, stateless MCP `forecast` server over `aprender-forecast`.
//!
//! One tool, no model artifact: the request carries the series, the server fits and
//! forecasts inside the call (D-01). Same shape as `aprender-mcp-setfit` (D-06). Every
//! bound and every refusal lives in `aprender_forecast::forecast` — this crate re-checks
//! nothing (OPS-03).
//!
//! Ported from `sources/004-forecast-mcp-thin-server/src/lib.rs:278-307` (D-08).

// schemars' JsonSchema derive (re-exported through ForecastArgs) and serde_json::json!
// both expand to .unwrap() internally, at file scope where a narrower allow cannot
// reach. Same precedent as aprender-mcp-setfit/src/lib.rs:29-32.
#![allow(clippy::disallowed_methods)]

use aprender_forecast::{ForecastArgs, ForecastError};
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;

/// The single tool this server advertises.
pub const TOOL_NAME: &str = "forecast";

/// The tool description, lifted into `tools/list`.
///
/// The last sentence is the D-16 honesty requirement: the bands are reported at their
/// MEASURED empirical coverage, never at the nominal 80 %.
pub const TOOL_DESCRIPTION: &str = "Fit a time-series forecaster to the series you pass in and return a forecast — in one call, \
 no model to train or store first. Send parallel arrays `ds` (dates, YYYY-MM-DD) and `y` (numbers) plus \
 `horizon` (number of future periods) and optionally `freq` (D, W or MS — daily by default). Default model \
 is `prophet` (trend with changepoints + Fourier seasonality, with 80% uncertainty bands and named \
 components); `neuralprophet` adds an AR-Net over the last `n_lags` values for short-horizon \
 nowcasting and requires freq D. Both models accept `holidays` (named events with optional windows) \
 and `regressors` (numeric drivers: one value per HISTORY row AND per HORIZON row, so \
 `len(values)` = `len(ds)` + `horizon`, with optional `mode`, `prior_scale` and `standardize`). \
 Switch `model` and send the same arguments: on `neuralprophet`, events are published as one \
 component per event name plus a `holidays` roll-up. Arm restrictions, refused rather than ignored: \
 on `neuralprophet`, `holidays` require `n_lags` = 0, multiplicative `mode` and an explicit \
 `prior_scale` are prophet-only, and a regressor on a series with missing days is refused when \
 `n_lags` > 0. Caution: at `n_lags` > 0 the AR term competes with a continuous driver and can \
 ABSORB it, so a regressor coefficient from `neuralprophet` is not a clean effect estimate \
 (measured: the price weight collapsed to -0.0048 from -0.0493 lag-free — illustrative of the \
 magnitude, NOT a bound); no VIF or condition number is computed on that arm because AR absorption \
 is a training dynamic rather than column collinearity. On `prophet`, regressor requests get \
 `diagnostics.regressors` with per-column VIF and a design condition number. \
 Returns future `ds`, `yhat`, `yhat_lower`, `yhat_upper`, `trend`, \
 components and timing. Bounded at 20,000 points, a 3,650-period horizon and 200 regressors. \
 Band honesty: the nominal 80% interval covered 0.60 (Prophet) and 0.66 (NeuralProphet-lite) of held-out \
 points across 17 rolling-origin windows — treat `yhat_lower`/`yhat_upper` as a ~60-66% band, not an 80% one.";

/// Map the library's refusals onto MCP errors: the caller's fault stays the caller's fault.
#[must_use]
pub fn map_error(e: ForecastError) -> pmcp::Error {
    match e {
        ForecastError::Validation(s) => pmcp::Error::validation(s),
        ForecastError::Internal(s) => pmcp::Error::internal(s),
    }
}

/// One stateless tool. The fit runs on a blocking thread: seconds of CPU must not stall
/// the protocol loop.
///
/// # Errors
///
/// `pmcp::Error` if the server builder refuses the configuration.
pub fn build_server(name: &str, version: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ForecastArgs, _, _>(
            TOOL_NAME,
            TOOL_DESCRIPTION,
            move |args, _extra| async move {
                let response =
                    tokio::task::spawn_blocking(move || aprender_forecast::forecast(&args))
                        .await
                        .map_err(|e| pmcp::Error::internal(format!("forecast task join: {e}")))?
                        .map_err(map_error)?;
                serde_json::to_value(&response)
                    .map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
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

    let server = std::sync::Arc::new(tokio::sync::Mutex::new(server));
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

/// K independent MCP routers behind a round-robin front handler (D-12).
///
/// # Why this exists — a MEASURED serialisation, not a hunch
///
/// pmcp 2.19.3's streamable-HTTP router holds ONE
/// `Arc<tokio::sync::Mutex<Server>>` across the whole tool future
/// (`pmcp-2.19.3/src/server/streamable_http_server.rs:2094` and `:2122`). A tool that
/// spends seconds inside `spawn_blocking` therefore holds that lock for the whole fit,
/// and the next request waits — not for CPU, for the mutex. Spike 010 measured eight
/// concurrent fits against ONE router at **1.0x** the sequential wall (i.e. no
/// concurrency at all) and the same eight against **eight** routers at **3.9x**.
///
/// Each router owns its own `Server`, so K tool calls can be in flight at once. Nothing
/// is shared between them: the door is stateless, the seed is per request, and
/// `pool_equality` asserts that a response computed under load is bit-identical to the
/// same response computed alone.
///
/// `pool <= 1` returns the single-router [`http_app`] unchanged, so the pool is opt-out
/// as well as opt-in.
///
/// # Errors
///
/// `pmcp::Error` if any of the K server builders refuses the configuration.
pub fn pooled_app(pool: usize, name: &str, version: &str) -> pmcp::Result<axum::Router> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tower::ServiceExt;

    if pool <= 1 {
        return Ok(http_app(build_server(name, version)?));
    }
    let mut built = Vec::with_capacity(pool);
    for _ in 0..pool {
        built.push(http_app(build_server(name, version)?));
    }
    let routers = Arc::new(built);
    let next = Arc::new(AtomicUsize::new(0));
    Ok(
        axum::Router::new().fallback(move |req: axum::extract::Request| {
            let routers = Arc::clone(&routers);
            let next = Arc::clone(&next);
            async move {
                let i = next.fetch_add(1, Ordering::Relaxed) % routers.len();
                // `axum::Router`'s service error type is `Infallible`, so the `match e {}`
                // is an uninhabited-type coercion, not `.unwrap()` — it passes .clippy.toml.
                routers[i]
                    .clone()
                    .oneshot(req)
                    .await
                    .unwrap_or_else(|e| match e {})
            }
        }),
    )
}

#[cfg(test)]
mod e2e {
    use super::{build_server, http_app};
    use aprender_forecast::dates::{days_from_civil, format_ymd};

    pub(super) struct Client {
        http: reqwest::Client,
        url: String,
        next: u64,
    }

    impl Client {
        /// The `&self` form, so a burst of concurrent tasks can share ONE client behind
        /// an `Arc`. The caller supplies the JSON-RPC id.
        pub(super) async fn call_id(
            &self,
            id: u64,
            method: &str,
            params: serde_json::Value,
        ) -> serde_json::Value {
            let body = serde_json::json!({
                "jsonrpc": "2.0", "id": id, "method": method, "params": params
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

        /// Sequential convenience form: allocates the next id itself.
        pub(super) async fn call(
            &mut self,
            method: &str,
            params: serde_json::Value,
        ) -> serde_json::Value {
            self.next += 1;
            let id = self.next;
            self.call_id(id, method, params).await
        }
    }

    pub(super) fn tool_output(v: &serde_json::Value) -> serde_json::Value {
        if let Some(s) = v["result"].get("structuredContent") {
            return s.clone();
        }
        let text = v["result"]["content"][0]["text"]
            .as_str()
            .expect("text content");
        serde_json::from_str(text).expect("tool JSON")
    }

    pub(super) fn csv(raw: &str) -> (Vec<String>, Vec<f64>) {
        let mut ds = Vec::new();
        let mut y = Vec::new();
        for line in raw.lines().skip(1) {
            let mut it = line.split(',');
            let Some(d) = it.next() else { continue };
            let Some(v) = it.next() else { continue };
            ds.push(d.trim().trim_matches('"').to_string());
            y.push(
                v.trim()
                    .trim_matches('"')
                    .parse()
                    .expect("second CSV column is a number"),
            );
        }
        (ds, y)
    }

    /// The tracer: ONE Prophet forecast crossing JSON-RPC -> tool schema -> validation
    /// door -> design -> L-BFGS -> predict -> response, over live streamable-HTTP,
    /// in-process.
    #[tokio::test]
    async fn forecast_prophet_happy_path_over_streamable_http() {
        let server = build_server("aprender-forecast-test", "0.0.0").expect("server");
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

        // The demo page is same-origin and is itself an MCP client.
        let page = reqwest::get(format!("http://{addr}/"))
            .await
            .expect("page")
            .text()
            .await
            .expect("html");
        assert!(
            page.contains("tools/call") && page.contains("forecast"),
            "demo page must be an MCP client"
        );

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
            schema["properties"].get("horizon").is_some(),
            "schema advertises horizon: {schema}"
        );
        assert_eq!(
            schema["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must surface in the advertised schema: {schema}"
        );

        // Peyton Manning, 365 days ahead — one real fit inside one call.
        let (ds, y) = csv(include_str!("../fixtures/peyton_manning.csv"));
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {"ds": ds, "y": y, "horizon": 365}
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "forecast failed: {r}"
        );
        let out = tool_output(&r);
        assert_eq!(out["ds"].as_array().expect("ds").len(), 365);
        assert_eq!(out["ds"][0], "2016-01-21");
        let yl = out["yhat_lower"].as_array().expect("yhat_lower");
        let yh = out["yhat"].as_array().expect("yhat");
        let yu = out["yhat_upper"].as_array().expect("yhat_upper");
        assert!(
            yl.iter()
                .zip(yh)
                .zip(yu)
                .all(|((l, h), u)| l.as_f64() <= h.as_f64() && h.as_f64() <= u.as_f64()),
            "band must bracket yhat"
        );
        assert!(
            out["components"].get("yearly").is_some() && out["components"].get("weekly").is_some(),
            "named components: {}",
            out["components"]
        );
        assert_eq!(out["trend"].as_array().expect("trend").len(), 365);
        assert!(
            out["fit_seconds"]
                .as_f64()
                .expect("fit_seconds")
                .is_finite()
                && out["predict_seconds"]
                    .as_f64()
                    .expect("predict_seconds")
                    .is_finite(),
            "timings are finite: {out}"
        );
        assert!(
            out["diagnostics"]["lbfgs"]["rounds"]
                .as_u64()
                .expect("lbfgs.rounds")
                >= 1,
            "the fit actually ran: {}",
            out["diagnostics"]
        );
    }

    // ---------------------------------------------------------------------------------
    // The D-11 refusal set (SC1), driven through a LIVE streamable-HTTP server.
    //
    // One `#[tokio::test]` per case on purpose: a failing case names ONE cause. Each case
    // perturbs `valid_args()` by exactly one field, so nothing else can be the reason.
    // ---------------------------------------------------------------------------------

    /// The JSON-RPC error code pmcp 2.19.3 emits for a refusal, READ OFF a live reply
    /// (not from the SDK docs) and pinned here.
    ///
    /// MEASURED, AND NOT WHAT IT LOOKS LIKE. This is `-32603`, the JSON-RPC *internal
    /// error* code. `pmcp::Error::error_code()` returns `None` for BOTH `Validation` and
    /// `Internal` (only the `Protocol` variant carries an explicit code), so the transport
    /// falls back to `-32603` for either. The numeric code alone therefore cannot tell a
    /// caller's fault from a server fault. The discriminator pmcp actually ships is the
    /// message PREFIX `thiserror` renders from the variant — [`VALIDATION_PREFIX`] below
    /// versus `"Internal error: "` — so both are pinned and a pmcp upgrade that changes
    /// either turns this suite red instead of silently reclassifying every refusal.
    const REFUSAL_CODE: i64 = -32603;

    /// The class marker that separates "you sent something I cannot use" from "I broke".
    /// `map_error` routes `ForecastError::Validation` here and `Internal` to the other
    /// prefix; an `"Internal error: "` on any case below is a defect, not a wording choice.
    const VALIDATION_PREFIX: &str = "Validation error: ";

    /// Bring up a fresh in-process server and complete `initialize`.
    async fn serve() -> Client {
        let server = build_server("aprender-forecast-test", "0.0.0").expect("server");
        client_for(http_app(server)).await
    }

    /// Serve ANY axum app on an ephemeral loopback port and return an initialized client.
    ///
    /// Shared with `pool_equality`, which passes a [`crate::pooled_app`] instead of the
    /// single-router `http_app`.
    pub(super) async fn client_for(app: axum::Router) -> Client {
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
        c
    }

    /// The Peyton Manning daily series (2 905 points, ends 2016-01-20).
    fn peyton() -> (Vec<String>, Vec<f64>) {
        csv(include_str!("../fixtures/peyton_manning.csv"))
    }

    /// The Air Passengers MONTHLY series (144 points, 1949-01 .. 1960-12).
    fn air() -> (Vec<String>, Vec<f64>) {
        csv(include_str!("../fixtures/air_passengers.csv"))
    }

    /// A synthetic daily series: trend + a weekly term, strictly ascending from 2020-01-01.
    ///
    /// The refusal cases use this rather than Peyton so each request is small — every
    /// refusal fires at the door, before any design is built, so the series never matters
    /// beyond being valid in the dimension the case is not perturbing.
    fn synth(n: usize) -> (Vec<String>, Vec<f64>) {
        let t0 = days_from_civil(2020, 1, 1);
        let mut ds = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            ds.push(format_ymd(t0 + i as i64));
            let t = i as f64;
            y.push(10.0 + 0.05 * t + (2.0 * std::f64::consts::PI * t / 7.0).sin());
        }
        (ds, y)
    }

    /// Arguments the door ACCEPTS. Every refusal case below perturbs exactly one field.
    fn valid_args() -> serde_json::Value {
        let (ds, y) = synth(60);
        serde_json::json!({"ds": ds, "y": y, "horizon": 7})
    }

    /// Assert the server REFUSED, that the refusal is validation-class, and that its
    /// message names the fix.
    async fn refused(c: &mut Client, args: serde_json::Value, needle: &str) {
        let reply = c
            .call(
                "tools/call",
                serde_json::json!({"name": "forecast", "arguments": args}),
            )
            .await;
        let is_error =
            reply.get("error").is_some() || reply["result"]["isError"] == serde_json::json!(true);
        assert!(is_error, "must be REFUSED, never defaulted; got: {reply}");
        assert!(
            reply.to_string().contains(needle),
            "the refusal must name the fix ({needle:?}); got: {reply}"
        );
        // Validation-class, not internal: the caller's fault stays the caller's fault.
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
            "a caller-fixable input must reach the client as {VALIDATION_PREFIX:?}, \
             never as \"Internal error: \"; got: {message}"
        );
    }

    /// Merge `extra` into the valid argument object (one perturbation per case).
    fn with(extra: serde_json::Value) -> serde_json::Value {
        let mut args = valid_args();
        let obj = args.as_object_mut().expect("args object");
        for (k, v) in extra.as_object().expect("extra object") {
            obj.insert(k.clone(), v.clone());
        }
        args
    }

    #[tokio::test]
    async fn refuses_unknown_field() {
        // deny_unknown_fields is structural: the key is named back to the caller.
        refused(
            &mut serve().await,
            with(serde_json::json!({"bogus": 1})),
            "bogus",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_length_mismatch() {
        let (ds, y) = synth(60);
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": &y[..59], "horizon": 7}),
            "ds has",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_too_few_points() {
        let (ds, y) = synth(5);
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "at least",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_too_many_points() {
        // MAX_POINTS + 1. This is the T-06-02 bound: it is what actually caps the work
        // one request can buy, so it is asserted over the wire, not just in a unit test.
        let (ds, y) = synth(20_001);
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "exceeds max_points",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_unsorted_ds() {
        let (mut ds, y) = synth(60);
        ds.swap(10, 20);
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "strictly ascending",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_duplicate_ds() {
        let (mut ds, y) = synth(60);
        ds[1] = ds[0].clone();
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "strictly ascending",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_impossible_date() {
        // Shape-valid, calendar-invalid: caught by the civil-date ROUND TRIP, so the
        // message is `not a calendar date` rather than the YYYY-MM-DD shape refusal.
        let (mut ds, y) = synth(60);
        ds[3] = "2008-02-30".into();
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "calendar",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_constant_y() {
        let (ds, _) = synth(60);
        let y = vec![5.0_f64; ds.len()];
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "is constant",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_horizon_zero() {
        refused(
            &mut serve().await,
            with(serde_json::json!({"horizon": 0})),
            "horizon must be",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_horizon_above_max() {
        // MAX_HORIZON + 1.
        refused(
            &mut serve().await,
            with(serde_json::json!({"horizon": 3651})),
            "horizon must be",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_unknown_freq() {
        // The D-11 case that matters most: a silent daily fallback for "H" would be a
        // WRONG answer, not a degraded one.
        refused(
            &mut serve().await,
            with(serde_json::json!({"freq": "H"})),
            "use D, W or MS",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_interval_width_out_of_range() {
        // The interval is OPEN at both ends: a 100% band is not a number this model
        // produces.
        refused(
            &mut serve().await,
            with(serde_json::json!({"interval_width": 1.0})),
            "interval_width must be in (0, 1)",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_logistic_without_cap() {
        refused(
            &mut serve().await,
            with(serde_json::json!({"growth": "logistic"})),
            "needs cap",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_logistic_cap_below_max_y() {
        // synth(60) reaches ~13; a cap of 1.0 is below the data it is supposed to bound.
        refused(
            &mut serve().await,
            with(serde_json::json!({"growth": "logistic", "cap": 1.0})),
            "must exceed max(y)",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_cap_without_logistic_growth() {
        // The BARE cap on the DEFAULTED growth arm — the shape 06-VERIFICATION gap 1
        // probed, where the server accepted `cap` and returned a byte-identical forecast.
        // synth(60) reaches ~13, so 100.0 is a cap a caller might believe is honoured.
        refused(
            &mut serve().await,
            with(serde_json::json!({"cap": 100.0})),
            "logistic-only",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_cap_with_explicit_linear_growth() {
        // The EXPLICIT arm. Two cases, not one: `refuses_cap_without_logistic_growth`
        // alone would still pass if the check were keyed on the ABSENCE of a `growth`
        // key, and one failing input is an anecdote (CLAUDE.md rule 6).
        refused(
            &mut serve().await,
            with(serde_json::json!({"growth": "linear", "cap": 100.0})),
            "logistic-only",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_unknown_model() {
        refused(
            &mut serve().await,
            with(serde_json::json!({"model": "arima"})),
            "prophet or neuralprophet",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_unknown_seasonality_mode() {
        refused(
            &mut serve().await,
            with(serde_json::json!({"seasonality_mode": "quadratic"})),
            "additive or multiplicative",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_unknown_growth() {
        refused(
            &mut serve().await,
            with(serde_json::json!({"growth": "exponential"})),
            "linear, logistic or flat",
        )
        .await;
    }

    // The four malformed-date SHAPES (REVIEW-06-U2). The spike's `s.get(..10)` prefix take
    // accepted the first of these silently; each is now its own case at the wire.

    #[tokio::test]
    async fn refuses_date_with_trailing_content() {
        let (mut ds, y) = synth(60);
        // The exact string the ported prefix-take used to accept as 2008-02-01.
        ds[0] = "2008-02-01garbage".into();
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "YYYY-MM-DD",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_empty_date() {
        let (mut ds, y) = synth(60);
        ds[0] = String::new();
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "YYYY-MM-DD",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_date_with_whitespace() {
        let (mut ds, y) = synth(60);
        ds[0] = " 2008-02-01".into();
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "YYYY-MM-DD",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_non_ascii_date() {
        let (mut ds, y) = synth(60);
        // Full-width digits: ten CHARS, not ten BYTES, and not ASCII either way.
        ds[0] = "２００８-02-01".into();
        refused(
            &mut serve().await,
            serde_json::json!({"ds": ds, "y": y, "horizon": 7}),
            "YYYY-MM-DD",
        )
        .await;
    }

    // ---------------------------------------------------------------------------------
    // The PRODUCT bound (06-11 / 06-12), driven through the SAME live server.
    //
    // Every bound above is a single factor. These two cases are the pair that proves the
    // door bounds their PRODUCT: they differ by ONE history point and straddle
    // `MAX_HOLIDAY_DESIGN_COST` (50 000) — 50 439 cells versus 49 708 — while EVERY
    // individual bound stays satisfied in both. A control an order of magnitude away from
    // the bound would prove only that a huge request is refused; this pair proves the bound
    // is where the contract says it is.
    // ---------------------------------------------------------------------------------

    /// One holiday spanning the widest legal window: `±MAX_HOLIDAY_WINDOW` (365) is
    /// `365 - (-365) + 1` = **731** design columns, from a single date and a ~200-byte
    /// request. That is the multiplier `MAX_HOLIDAY_DESIGN_COST` exists to bound.
    const WIDE_HOLIDAY_COLUMNS: usize = 731;

    /// A single holiday, one date, windows at exactly `±MAX_HOLIDAY_WINDOW`.
    fn wide_holiday() -> serde_json::Value {
        serde_json::json!([{
            "name": "anchor",
            "dates": ["2020-01-15"],
            "lower_window": -365,
            "upper_window": 365
        }])
    }

    #[tokio::test]
    async fn refuses_holiday_design_cost_over_bound() {
        // 62 points + horizon 7 = 69 rows x 731 columns = 50 439 > 50 000.
        // Individually in bounds on EVERY factor: 62 <= MAX_POINTS (20 000), 7 <=
        // MAX_HORIZON (3 650), a 61-day span <= MAX_SPAN_DAYS (20 000), |±365| <=
        // MAX_HOLIDAY_WINDOW, 731 <= MAX_HOLIDAY_COLUMNS (1 000), 1 date <=
        // MAX_HOLIDAY_DATES (1 000) and <= MAX_HOLIDAY_DATES_TOTAL (10 000). Only the
        // PRODUCT is over — which is the whole point: a case that trips a neighbouring
        // bound would prove nothing about this one.
        let (ds, y) = synth(62);
        assert!(
            (ds.len() + 7) * WIDE_HOLIDAY_COLUMNS
                > aprender_forecast::types::MAX_HOLIDAY_DESIGN_COST,
            "the OVER geometry must exceed the bound it is testing"
        );
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 7, "holidays": wide_holiday()
            }),
            "max_holiday_design_cost",
        )
        .await;
    }

    #[tokio::test]
    async fn accepts_holiday_design_cost_just_under_bound() {
        // ONE point fewer: 61 + 7 = 68 rows x 731 columns = 49 708 <= 50 000. The two-sided
        // control at the boundary — the bound must refuse only what it claims to refuse,
        // and a near-miss that merely "does not error" is not a control, so the full
        // promised response shape is asserted here.
        let (ds, y) = synth(61);
        let horizon = 7usize;
        assert!(
            (ds.len() + horizon) * WIDE_HOLIDAY_COLUMNS
                <= aprender_forecast::types::MAX_HOLIDAY_DESIGN_COST,
            "the UNDER geometry must sit under the bound it is testing"
        );
        let mut c = serve().await;
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": horizon, "holidays": wide_holiday()
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "a request one step UNDER the design-cost bound must be accepted: {r}"
        );
        let out = tool_output(&r);
        // ds / yhat / yhat_lower / yhat_upper / trend all present, horizon-length, banded.
        assert_shared_shape(&out, horizon);
        assert!(
            out.get("components").is_some_and(|c| c.is_object()),
            "the accepted near-miss must carry components: {out}"
        );
        assert!(
            out.get("diagnostics").is_some_and(|d| d.is_object()),
            "the accepted near-miss must carry diagnostics: {out}"
        );
    }

    // ------------------------------ the logistic changepoint lambda bound, over the wire ---
    //
    // 06-REVIEW.md CR-01. The `arguments` object below measures **1 138 bytes** of JSON on
    // this synth series (the review's own probe was 1 132 bytes — the same shape with
    // slightly different y values; both are ~1.1 KB and the point is the ORDER, not the
    // digit). It is accepted by
    // every other bound this door has: 33 points <= fit_max_points, a 32-day span <=
    // fit_max_span_days, horizon 3650 <= fit_max_horizon, a cap above max(y), no holidays at
    // all. It bought 2.334 s of CPU against a default pool of K=8. The cost is the Poisson
    // mean of predict's logistic uncertainty arm — `changepoint_count(len(ds)) * (t_max - 1)`
    // — and nothing bounded it, because fit_max_horizon bounds the COUNT of future steps
    // while dates::future_days multiplies that count by ~30.44 for "MS".

    /// The tightest legal history that still earns all 25 changepoints: 33 daily points.
    /// Below 33, `floor(n * 0.8) - 1 < 25` and the count — hence lambda — drops.
    const LAMBDA_POINTS: usize = 33;

    /// The lambda a `(points, horizon, freq)` request makes `predict` draw, computed through
    /// `aprender_forecast::prophet::changepoint_count` — the SAME function the door uses — so
    /// a test named "over" cannot quietly become a test of something under the bound.
    fn lambda_for(ds: &[String], horizon: usize, freq: &str) -> f64 {
        use aprender_forecast::prophet::{auto_seasonalities, changepoint_count, Mode, Spec};
        let days: Vec<i64> = ds
            .iter()
            .map(|s| {
                aprender_forecast::dates::parse_date(s).expect("the helper builds valid dates")
            })
            .collect();
        let fut = aprender_forecast::dates::future_days(days[days.len() - 1], horizon, freq)
            .expect("the helper builds a valid freq");
        let t_scale = (days[days.len() - 1] - days[0]) as f64;
        let t_max = (fut[fut.len() - 1] - days[0]) as f64 / t_scale;
        let spec = Spec::default_linear(auto_seasonalities(&days, 10.0, Mode::Additive));
        changepoint_count(days.len(), &spec) as f64 * (t_max - 1.0)
    }

    #[tokio::test]
    async fn refuses_logistic_changepoint_lambda_over_bound() {
        // The review's measured request, verbatim in shape: 33 points at 1-day spacing,
        // horizon 3650, freq MS, growth logistic, cap 50. lambda = 25 x (3472.594 - 1)
        // = 86 789.8, which is also the STRUCTURAL MAXIMUM of this axis.
        let (ds, y) = synth(LAMBDA_POINTS);
        let lambda = lambda_for(&ds, 3650, "MS");
        assert!(
            lambda > aprender_forecast::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA,
            "the OVER geometry must exceed the bound it is testing: lambda={lambda:.1} vs {}",
            aprender_forecast::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA
        );
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 3650, "freq": "MS",
                "growth": "logistic", "cap": 50.0
            }),
            "max_logistic_changepoint_lambda",
        )
        .await;
    }

    #[tokio::test]
    async fn accepts_logistic_changepoint_lambda_just_under_bound() {
        // The SAME request with ONE field changed — the horizon, which is the review's own
        // control axis (its four measured rows differ in horizon span and nothing else). At
        // 840 MS steps lambda is 19 974.2, inside the bound and within 0.2% of it, so this
        // pair straddles the bound rather than sitting an order of magnitude away from it.
        let horizon = 840usize;
        let (ds, y) = synth(LAMBDA_POINTS);
        let lambda = lambda_for(&ds, horizon, "MS");
        let bound = aprender_forecast::types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA;
        assert!(
            lambda <= bound,
            "the UNDER geometry must sit under the bound: lambda={lambda:.1} vs {bound}"
        );
        assert!(
            lambda > 0.9 * bound,
            "the near miss must be NEAR: lambda={lambda:.1} is not within 10% of {bound}, so \
             this control would pass for a bound an order of magnitude away"
        );
        let mut c = serve().await;
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": horizon, "freq": "MS",
                        "growth": "logistic", "cap": 50.0
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "a request just under the changepoint-lambda bound must be accepted: {r}"
        );
        let out = tool_output(&r);
        assert_shared_shape(&out, horizon);
        assert!(
            out.get("components").is_some_and(|c| c.is_object()),
            "the accepted near-miss must carry components: {out}"
        );
        assert!(
            out.get("diagnostics").is_some_and(|d| d.is_object()),
            "the accepted near-miss must carry diagnostics: {out}"
        );
    }

    #[tokio::test]
    async fn refuses_holiday_dates_total_over_bound() {
        // The third factor, which no e2e case reached before: eleven holidays of 1 000
        // dates each. Every holiday is individually legal (1 000 == MAX_HOLIDAY_DATES) and
        // the columns sum to 11 (<= 1 000), so only the AGGREGATE 11 000 > 10 000 is over.
        let (ds, y) = synth(60);
        let base = days_from_civil(1990, 1, 1);
        let mut holidays = Vec::new();
        for h in 0..11i64 {
            let dates: Vec<String> = (0..1000i64)
                .map(|d| format_ymd(base + h * 1000 + d))
                .collect();
            holidays.push(serde_json::json!({
                "name": format!("h{h}"),
                "dates": dates,
                "lower_window": 0,
                "upper_window": 0
            }));
        }
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 7, "holidays": holidays
            }),
            "max_holiday_dates_total",
        )
        .await;
    }

    #[tokio::test]
    async fn refuses_holiday_dates_total_before_parsing_the_rest() {
        // WR-03 through the transport. The running sum crosses the aggregate ceiling at the
        // eleventh holiday; four LATER holidays each carry `"2020-1-01"` — nine bytes, which
        // `parse_date`'s SHAPE gate rejects. A date-shape refusal here would prove the door
        // kept parsing work it had already decided to discard.
        let (ds, y) = synth(60);
        let base = days_from_civil(1990, 1, 1);
        let mut holidays = Vec::new();
        for h in 0..11i64 {
            let dates: Vec<String> = (0..1000i64)
                .map(|d| format_ymd(base + h * 1000 + d))
                .collect();
            holidays.push(serde_json::json!({
                "name": format!("h{h}"),
                "dates": dates,
                "lower_window": 0,
                "upper_window": 0
            }));
        }
        for h in 11..15i64 {
            holidays.push(serde_json::json!({
                "name": format!("late{h}"),
                "dates": ["2020-1-01"],
                "lower_window": 0,
                "upper_window": 0
            }));
        }
        let args = serde_json::json!({
            "ds": ds, "y": y, "horizon": 7, "holidays": holidays
        });
        let mut c = serve().await;
        refused(&mut c, args.clone(), "max_holiday_dates_total").await;
        let reply = c
            .call(
                "tools/call",
                serde_json::json!({"name": "forecast", "arguments": args}),
            )
            .await;
        assert!(
            !reply.to_string().contains("want YYYY-MM-DD"),
            "the aggregate refusal must fire BEFORE the later holidays are parsed; the \
             date-shape message means it still fires after the loop: {reply}"
        );
    }

    /// C-07 through the transport: one byte over the byte bound is refused.
    #[tokio::test]
    async fn refuses_holiday_name_over_length_bound() {
        let (ds, y) = synth(60);
        let name = "n".repeat(aprender_forecast::types::MAX_HOLIDAY_NAME_LEN + 1);
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 7,
                "holidays": [{
                    "name": name, "dates": ["2020-02-01"],
                    "lower_window": 0, "upper_window": 0
                }]
            }),
            "max_holiday_name_len",
        )
        .await;
    }

    /// The straddling partner: exactly AT the bound, accepted through the same transport,
    /// returning the full shared response shape. A pair that only ever refuses proves the
    /// door is closed, never that it is closed on the right side.
    #[tokio::test]
    async fn accepts_holiday_name_at_length_bound() {
        let horizon = 7usize;
        let (ds, y) = synth(60);
        let name = "n".repeat(aprender_forecast::types::MAX_HOLIDAY_NAME_LEN);
        assert_eq!(
            name.len(),
            aprender_forecast::types::MAX_HOLIDAY_NAME_LEN,
            "the near miss must sit exactly ON the bound"
        );
        let mut c = serve().await;
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": horizon,
                        "holidays": [{
                            "name": name, "dates": ["2020-02-01"],
                            "lower_window": 0, "upper_window": 0
                        }]
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "a holiday name exactly at the bound must be accepted: {r}"
        );
        let out = tool_output(&r);
        assert_shared_shape(&out, horizon);
        assert!(
            out.get("components").is_some_and(|c| c.is_object()),
            "the accepted near-miss must carry components: {out}"
        );
    }

    /// C-08 through the transport: a NeuralProphet request whose training work exceeds
    /// `max_np_train_cost` is refused at the door, before any training runs.
    #[tokio::test]
    async fn refuses_np_train_cost_over_bound() {
        // 2 000 contiguous daily points with n_lags at its own ceiling of 365 prices far
        // over the bound. The refusal fires before np::train, so this case is instant even
        // though the request it refuses would not be.
        let (ds, y) = synth(2_000);
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 7,
                "model": "neuralprophet", "freq": "D", "n_lags": 365
            }),
            "max_np_train_cost",
        )
        .await;
    }

    /// The straddling partner, on the SAME arm with lags on: a request priced under the
    /// bound is accepted and returns the full shared response shape.
    ///
    /// It is not AT the bound, deliberately and with a measured reason: at the bound one
    /// request costs 45.619 s on a debug profile, and this suite runs in debug. The
    /// at-the-bound positive controls are `np::wall::np_train_wall` under
    /// `NP_WALL_MODE=at_bound_*` — three release compositions at 93-99% of the bound,
    /// measured at 1.402-1.642 s.
    #[tokio::test]
    async fn accepts_np_train_cost_under_bound() {
        let horizon = 7usize;
        let (ds, y) = synth(120);
        let mut c = serve().await;
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": horizon,
                        "model": "neuralprophet", "freq": "D", "n_lags": 7
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "a neuralprophet request under the training-cost bound must be accepted: {r}"
        );
        let out = tool_output(&r);
        assert_shared_shape(&out, horizon);
        assert_eq!(out["model"], serde_json::json!("neuralprophet"));
    }

    // ---------------------------------------------------------------------------------
    // Happy paths: the SHARED response shape (D-03) across both models and across the
    // option combinations the refusal cases only ever exercise negatively.
    // ---------------------------------------------------------------------------------

    /// Assert the shared shape: horizon-length parallel arrays with a bracketing band.
    fn assert_shared_shape(out: &serde_json::Value, horizon: usize) {
        for key in ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"] {
            assert_eq!(
                out[key]
                    .as_array()
                    .unwrap_or_else(|| panic!("{key}: {out}"))
                    .len(),
                horizon,
                "{key} must have one row per horizon step: {out}"
            );
        }
        let l = out["yhat_lower"].as_array().expect("yhat_lower");
        let h = out["yhat"].as_array().expect("yhat");
        let u = out["yhat_upper"].as_array().expect("yhat_upper");
        for i in 0..horizon {
            let (lo, mid, hi) = (
                l[i].as_f64().expect("lower"),
                h[i].as_f64().expect("yhat"),
                u[i].as_f64().expect("upper"),
            );
            assert!(mid.is_finite(), "row {i}: yhat must be finite");
            assert!(
                lo <= mid && mid <= hi,
                "row {i}: band must bracket yhat ({lo} <= {mid} <= {hi})"
            );
        }
        assert!(
            out["predict_seconds"]
                .as_f64()
                .expect("predict_seconds")
                .is_finite(),
            "timings must be finite: {out}"
        );
    }

    #[tokio::test]
    async fn neuralprophet_lag_free_happy_path() {
        // The arm plan 06-04 filled in. 06-01 shipped it as a REFUSING stub, so this is
        // the first e2e proof that `model: neuralprophet` dispatches over the wire.
        let mut c = serve().await;
        let (ds, y) = peyton();
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {"ds": ds, "y": y, "horizon": 30, "model": "neuralprophet"}
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "neuralprophet forecast failed: {r}"
        );
        let out = tool_output(&r);
        assert_eq!(out["model"], "neuralprophet");
        assert_shared_shape(&out, 30);
        assert!(
            out["diagnostics"]["selected_lr"].as_f64().is_some(),
            "the lr sweep must be reported: {}",
            out["diagnostics"]
        );
        assert!(
            out["diagnostics"]["band"]
                .as_str()
                .expect("diagnostics.band")
                .contains("residual"),
            "the band must SAY it is residual-sd based, not quantile regression: {}",
            out["diagnostics"]
        );
    }

    #[tokio::test]
    async fn prophet_ms_multiplicative_happy_path() {
        // Monthly data, month-start future grid, multiplicative seasonality — the three
        // knobs the refusal cases only ever set WRONG.
        let mut c = serve().await;
        let (ds, y) = air();
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": 12,
                        "freq": "MS", "seasonality_mode": "multiplicative"
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "MS + multiplicative forecast failed: {r}"
        );
        let out = tool_output(&r);
        assert_eq!(out["freq"], "MS");
        assert_shared_shape(&out, 12);
        assert!(
            out["components"].get("multiplicative_terms").is_some(),
            "multiplicative seasonality must produce multiplicative_terms: {}",
            out["components"]
        );
        for d in out["ds"].as_array().expect("ds") {
            assert!(
                d.as_str().expect("ds entry").ends_with("-01"),
                "MS steps land on the first of a month: {d}"
            );
        }
    }

    #[tokio::test]
    async fn prophet_logistic_with_holidays_happy_path() {
        // Logistic growth WITH a valid cap, plus a named holiday — the positive twin of
        // refuses_logistic_without_cap / refuses_logistic_cap_below_max_y.
        let mut c = serve().await;
        let (ds, y) = peyton();
        let cap = y.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v)) + 2.0;
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": 30,
                        "growth": "logistic", "cap": cap,
                        "holidays": [{
                            "name": "superbowl",
                            "dates": [
                                "2010-02-07", "2011-02-06", "2012-02-05", "2013-02-03",
                                "2014-02-02", "2015-02-01", "2016-02-07"
                            ],
                            "lower_window": 0,
                            "upper_window": 1
                        }]
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "logistic + holidays forecast failed: {r}"
        );
        let out = tool_output(&r);
        assert_shared_shape(&out, 30);
        assert!(
            out["components"].get("superbowl").is_some(),
            "a named holiday must become a named component: {}",
            out["components"]
        );
        assert_eq!(
            out["diagnostics"]["growth"], "Logistic",
            "the diagnostics must report the growth actually fitted: {}",
            out["diagnostics"]
        );
    }

    // ---- D-29 / D-31 over the wire: `holidays` on the neuralprophet arm ----
    //
    // The TRANSPORT needs no change for any of these, and that is the property they prove.
    // Every bound, every refusal and every default lives in the door (D-11), so opening a
    // combination there is visible over the wire with nothing edited outside this test
    // module. Plan 06.1-06 Task 3's placement check asserts exactly that.

    /// Two holidays with non-zero windows on `model: neuralprophet`, over the wire.
    ///
    /// Until this plan the door refused this combination outright. The response must carry
    /// one component per event NAME plus a `holidays` roll-up, matching the prophet arm's
    /// keys for the same argument (D-31).
    #[tokio::test]
    async fn neuralprophet_with_holidays_publishes_per_event_components() {
        let mut c = serve().await;
        let (ds, y) = synth(240);
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": 20, "model": "neuralprophet", "seed": 42,
                        "holidays": [
                            {"name": "promo", "dates": ["2020-03-15", "2020-06-15", "2020-09-15"],
                             "lower_window": -1, "upper_window": 1},
                            {"name": "blackfriday", "dates": ["2020-04-01", "2020-07-01"],
                             "lower_window": 0, "upper_window": 2}
                        ]
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "neuralprophet + holidays must be ACCEPTED over the wire: {r}"
        );
        let out = tool_output(&r);
        assert_eq!(out["model"], "neuralprophet");
        assert_shared_shape(&out, 20);
        for key in ["trend", "promo", "blackfriday", "holidays"] {
            assert!(
                out["components"].get(key).is_some(),
                "the wire response must carry a {key:?} component: {}",
                out["components"]
            );
        }
    }

    /// The CONTROL for the case above: the same request WITHOUT holidays succeeds and
    /// forecasts DIFFERENTLY, so acceptance over the wire is not acceptance-and-ignore.
    #[tokio::test]
    async fn neuralprophet_without_holidays_forecasts_differently_over_the_wire() {
        let (ds, y) = synth(240);
        let base = serde_json::json!({
            "ds": ds, "y": y, "horizon": 20, "model": "neuralprophet", "seed": 42
        });
        let mut with = base.clone();
        with["holidays"] = serde_json::json!([
            {"name": "promo", "dates": ["2020-03-15", "2020-06-15", "2020-09-15"],
             "lower_window": -1, "upper_window": 1}
        ]);

        let mut c = serve().await;
        let a = c
            .call(
                "tools/call",
                serde_json::json!({"name": "forecast", "arguments": base}),
            )
            .await;
        let b = c
            .call(
                "tools/call",
                serde_json::json!({"name": "forecast", "arguments": with}),
            )
            .await;
        for (label, r) in [("without", &a), ("with", &b)] {
            assert!(
                r.get("error").is_none() && r["result"]["isError"] != true,
                "the {label}-holidays request must succeed: {r}"
            );
        }
        let (off, on) = (tool_output(&a), tool_output(&b));
        assert!(
            off["components"].get("holidays").is_none(),
            "an event-free neuralprophet response must publish NO event components: {}",
            off["components"]
        );
        assert_ne!(
            off["yhat"], on["yhat"],
            "holidays were accepted over the wire but changed nothing"
        );
    }

    /// The SCOPED half of the opening, over the wire: `holidays` with `n_lags > 0` is
    /// refused, and the refusal text names the configuration that DOES work.
    ///
    /// MEASURED (06.1-05 FINDING 1): per-column recovery runs 0.2191-0.4660 against the
    /// 0.10 bar at the epoch budget the door configures, one run negative. The combination
    /// is refused rather than shipped behind a disclaimer.
    #[tokio::test]
    async fn neuralprophet_refuses_holidays_with_lags_over_the_wire() {
        let (ds, y) = synth(240);
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 20, "model": "neuralprophet", "n_lags": 7,
                "holidays": [
                    {"name": "promo", "dates": ["2020-03-15"], "lower_window": -1,
                     "upper_window": 1}
                ]
            }),
            "n_lags = 0",
        )
        .await;
    }

    /// The hoisted COLUMN ceiling is paid on the neuralprophet arm over the wire.
    ///
    /// Before this plan the arm could not reach this bound at all, because `holidays` was
    /// refused before dispatch. The bound now sits above the model dispatch and both arms
    /// pay it (D-33).
    #[tokio::test]
    async fn neuralprophet_refuses_holidays_over_the_column_ceiling_over_the_wire() {
        let (ds, y) = synth(60);
        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 7, "model": "neuralprophet",
                "holidays": [
                    {"name": "a", "dates": ["2020-02-01"], "lower_window": -250,
                     "upper_window": 250},
                    {"name": "b", "dates": ["2020-02-02"], "lower_window": -250,
                     "upper_window": 250}
                ]
            }),
            "max_holiday_columns",
        )
        .await;
    }

    /// D-33's reason for existing, over the wire: a GAPPY neuralprophet series whose POINT
    /// count clears the design-cost ceiling but whose imputed-grid SPAN does not, refused
    /// with the SPAN operand named.
    ///
    /// The same request is ACCEPTED on the prophet arm, whose design really is built per
    /// supplied row — which is what proves the two operands differ rather than that the
    /// request is simply too big for anything.
    #[tokio::test]
    async fn neuralprophet_refuses_a_gappy_series_on_the_span_operand_over_the_wire() {
        let t0 = days_from_civil(2020, 1, 1);
        let ds: Vec<String> = (0..40).map(|i| format_ymd(t0 + i * 100)).collect();
        let y: Vec<f64> = (0..40).map(|i| 10.0 + 0.05 * f64::from(i)).collect();
        let holidays = serde_json::json!([
            {"name": "promo", "dates": ["2020-01-06"], "lower_window": -15,
             "upper_window": 14}
        ]);

        refused(
            &mut serve().await,
            serde_json::json!({
                "ds": ds, "y": y, "horizon": 10, "model": "neuralprophet",
                "holidays": holidays
            }),
            "span_days",
        )
        .await;

        // The CONTROL: the very same gappy request on the PROPHET arm is accepted.
        let mut c = serve().await;
        let r = c
            .call(
                "tools/call",
                serde_json::json!({
                    "name": "forecast",
                    "arguments": {
                        "ds": ds, "y": y, "horizon": 10, "holidays": holidays
                    }
                }),
            )
            .await;
        assert!(
            r.get("error").is_none() && r["result"]["isError"] != true,
            "the same gappy request must be ACCEPTED on the prophet arm, whose design is \
             built per supplied row: {r}"
        );
    }
}

#[cfg(test)]
mod tests {
    use aprender_forecast::ForecastArgs;

    /// The strictness D-11 relies on must be ADVERTISED, not merely enforced: a client
    /// reads `tools/list` and has to be able to see that an unknown key will be refused
    /// and that exactly three fields are required.
    #[test]
    fn the_tool_schema_is_strict_and_requires_ds_y_horizon() {
        let schema =
            serde_json::to_value(schemars::schema_for!(ForecastArgs)).expect("schema serializes");
        assert_eq!(
            schema["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must surface as additionalProperties: false: {schema}"
        );
        let required: Vec<&str> = schema["required"]
            .as_array()
            .expect("required array")
            .iter()
            .map(|v| v.as_str().expect("required entry is a string"))
            .collect();
        let mut sorted = required.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec!["ds", "horizon", "y"],
            "required must be EXACTLY ds, y, horizon; got {required:?}"
        );
        let props = schema["properties"].as_object().expect("properties object");
        for optional in ["freq", "model", "n_lags", "seed", "holidays"] {
            assert!(
                props.contains_key(optional),
                "{optional} must be advertised as an optional property: {schema}"
            );
        }
    }
}

/// Equality under load (D-12, SC5) — the CORRECTNESS half of the pool claim.
///
/// # What is asserted here, and what is deliberately NOT (REVIEW-06-04)
///
/// ASSERTED: a response computed while eight (then sixteen) requests are in flight is
/// **bit-identical** to the same response computed alone. That is a correctness claim, it
/// holds on every host, and it is what makes a forecast reproducible from its request
/// (threat T-06-07).
///
/// NOT ASSERTED: the wall-clock speed-up. Both cross-AI reviewers reached this
/// independently. With four Tokio workers, eight heterogeneous `spawn_blocking` fits and
/// BLAS underneath, the ratio moves with CPU throttling and unrelated background load
/// *independently of the router serialisation the pool exists to remove* — so a hard
/// `assert!(speedup >= 2.0)` in the correctness suite fails for reasons the pool does not
/// control, and a suite that cries wolf gets its real failures ignored. The ratio is
/// instead PRINTED, once, in a machine-parsable line carrying its own provenance, and
/// `just forecast-pool-ratio` (plan 06-08) is what asserts `>= 2.0`, best-of-3, on the
/// aarch64 release host. There is no timing assertion and no `cfg!(target_arch)` gate
/// anywhere in this module.
#[cfg(test)]
mod pool_equality {
    use std::sync::Arc;
    use std::time::Instant;

    use aprender_forecast::dates::{days_from_civil, format_ymd};
    use aprender_forecast::prophet::Rng;

    use super::e2e::{client_for, csv, tool_output};
    use super::pooled_app;

    /// The default pool size, ASSERTED equal to `constants.pool_default` by
    /// `aprender_forecast::types::tests::cost_bounds_match_contract`.
    use aprender_forecast::types::DEFAULT_POOL as POOL;

    /// The fields whose bits must not depend on what else the server was doing.
    fn signature(out: &serde_json::Value) -> String {
        serde_json::json!({
            "ds": out["ds"], "yhat": out["yhat"],
            "yhat_lower": out["yhat_lower"], "yhat_upper": out["yhat_upper"],
            "trend": out["trend"], "components": out["components"]
        })
        .to_string()
    }

    /// Worst pointwise disagreement in `yhat`, so a mismatch reports a MAGNITUDE rather
    /// than only "the strings differ".
    fn max_abs_diff(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
        let col = |v: &serde_json::Value| -> Vec<f64> {
            v["yhat"]
                .as_array()
                .expect("yhat array")
                .iter()
                .map(|x| x.as_f64().expect("yhat entry"))
                .collect()
        };
        col(a)
            .iter()
            .zip(col(b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// A deterministic synthetic daily series: trend + yearly + weekly + seeded noise.
    fn synth(n: usize, seed: u64) -> (Vec<String>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let start = days_from_civil(2012, 1, 1);
        let mut ds = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut level = 10.0 + seed as f64;
        for i in 0..n {
            let day = start + i as i64;
            level += 0.002;
            let f = day as f64;
            y.push(
                level
                    + 0.8 * (2.0 * std::f64::consts::PI * f / 365.25).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * f / 7.0).cos()
                    + 0.2 * rng.normal(),
            );
            ds.push(format_ymd(day));
        }
        (ds, y)
    }

    /// The eight distinct requests the burst issues: four Prophet fits on a 1 000-point
    /// daily series with different seeds and horizons, two lag-free NeuralProphet fits and
    /// two `n_lags = 30` NeuralProphet fits (the autograd-tape-heaviest path) on 500
    /// points.
    ///
    /// `FORECAST_POOL_SERIES=peyton` swaps in the real 2 905-point Peyton Manning series —
    /// the set `just forecast-pool-ratio` (06-08) measures on a release build. The default
    /// is sized to stay inside a debug-profile feedback loop.
    fn requests() -> Vec<serde_json::Value> {
        let peyton = std::env::var("FORECAST_POOL_SERIES").is_ok_and(|v| v == "peyton");
        let (long, short) = if peyton {
            let p = csv(include_str!("../fixtures/peyton_manning.csv"));
            (p.clone(), p)
        } else {
            (synth(1_000, 1), synth(500, 2))
        };
        let (ld, ly) = long;
        let (sd, sy) = short;
        vec![
            serde_json::json!({"ds": ld, "y": ly, "horizon": 30, "model": "prophet", "seed": 1}),
            serde_json::json!({"ds": ld, "y": ly, "horizon": 60, "model": "prophet", "seed": 2}),
            serde_json::json!({
                "ds": ld, "y": ly, "horizon": 30, "model": "prophet", "seed": 3,
                "seasonality_mode": "multiplicative"
            }),
            serde_json::json!({"ds": ld, "y": ly, "horizon": 90, "model": "prophet", "seed": 4}),
            serde_json::json!({
                "ds": sd, "y": sy, "horizon": 30, "model": "neuralprophet", "seed": 5
            }),
            serde_json::json!({
                "ds": sd, "y": sy, "horizon": 14, "model": "neuralprophet", "seed": 6
            }),
            serde_json::json!({
                "ds": sd, "y": sy, "horizon": 30, "model": "neuralprophet",
                "n_lags": 30, "seed": 7
            }),
            serde_json::json!({
                "ds": sd, "y": sy, "horizon": 14, "model": "neuralprophet",
                "n_lags": 30, "seed": 8
            }),
        ]
    }

    fn call_body(args: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({"name": "forecast", "arguments": args})
    }

    fn ok_output(reply: &serde_json::Value, what: &str) -> serde_json::Value {
        assert!(
            reply.get("error").is_none() && reply["result"]["isError"] != true,
            "{what} failed: {reply}"
        );
        tool_output(reply)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn eight_concurrent_requests_are_bit_identical_to_sequential() {
        let app = pooled_app(POOL, "aprender-forecast-pool-test", "0.0.0").expect("pooled app");
        let client = Arc::new(client_for(app).await);
        let reqs = requests();

        // 1. Sequential baseline.
        let t0 = Instant::now();
        let mut base = Vec::with_capacity(reqs.len());
        for (i, args) in reqs.iter().enumerate() {
            let reply = client
                .call_id(100 + i as u64, "tools/call", call_body(args))
                .await;
            base.push(ok_output(&reply, &format!("sequential request {i}")));
        }
        let seq = t0.elapsed();

        // 2. The same requests, all at once, against the pooled app.
        let t1 = Instant::now();
        let handles: Vec<_> = reqs
            .iter()
            .enumerate()
            .map(|(i, args)| {
                let client = Arc::clone(&client);
                let body = call_body(args);
                tokio::spawn(
                    async move { client.call_id(1_000 + i as u64, "tools/call", body).await },
                )
            })
            .collect();
        let mut outs = Vec::with_capacity(reqs.len());
        for (i, h) in handles.into_iter().enumerate() {
            let reply = h.await.expect("concurrent task join");
            outs.push(ok_output(&reply, &format!("concurrent request {i}")));
        }
        let conc = t1.elapsed();

        // 3. THE ONLY ASSERTION: equality. Bit-identical, not close.
        let identical = outs
            .iter()
            .zip(&base)
            .filter(|(o, b)| signature(o) == signature(b))
            .count();
        let worst = outs
            .iter()
            .zip(&base)
            .map(|(o, b)| max_abs_diff(o, b))
            .fold(0.0, f64::max);
        assert_eq!(
            identical,
            reqs.len(),
            "every concurrent response must be bit-identical to its sequential result; \
             {identical}/{} matched, worst |d yhat| {worst:e}",
            reqs.len()
        );
        assert_eq!(
            worst, 0.0,
            "a per-request-seeded deterministic pipeline has no legitimate source of \
             variation under concurrency; worst |d yhat| was {worst:e}"
        );

        // 4. The ratio is REPORTED, never asserted (REVIEW-06-04). `just
        //    forecast-pool-ratio` (06-08) greps this exact line and is what applies the
        //    >= 2.0 bar, best-of-3, on the aarch64 release host. The arch/profile/cpus
        //    fields are what make a later comparison mean anything: a ratio without the
        //    machine it was measured on is a number, not a measurement.
        let speedup = seq.as_secs_f64() / conc.as_secs_f64();
        let cpus = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
        let profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        println!(
            "POOL SPEEDUP: {speedup:.3}x seq={}ms conc={}ms arch={} profile={profile} \
             workers=4 cpus={cpus}",
            seq.as_millis(),
            conc.as_millis(),
            std::env::consts::ARCH
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sixteen_concurrent_neuralprophet_fits_stay_identical() {
        // The tape-heaviest path (`n_lags = 30` drives the AR-Net through autograd), two
        // alternating series, 16 at once against 8 routers — so each router also has to
        // serve two calls back to back without leaking state between them. No timing
        // assertion here either, and this test prints no POOL SPEEDUP line: exactly one
        // such line per run is the contract with 06-08.
        let app = pooled_app(POOL, "aprender-forecast-pool-test", "0.0.0").expect("pooled app");
        let client = Arc::new(client_for(app).await);
        let series = [synth(500, 11), synth(500, 12)];
        let stress: Vec<serde_json::Value> = (0..16)
            .map(|i| {
                let (ds, y) = &series[i % 2];
                serde_json::json!({
                    "ds": ds, "y": y, "horizon": 14, "model": "neuralprophet",
                    "n_lags": 30, "seed": 42
                })
            })
            .collect();

        let mut expected = Vec::with_capacity(2);
        for (i, args) in stress.iter().take(2).enumerate() {
            let reply = client
                .call_id(5_000 + i as u64, "tools/call", call_body(args))
                .await;
            expected.push(signature(&ok_output(&reply, &format!("baseline {i}"))));
        }

        let handles: Vec<_> = stress
            .iter()
            .enumerate()
            .map(|(i, args)| {
                let client = Arc::clone(&client);
                let body = call_body(args);
                tokio::spawn(
                    async move { client.call_id(6_000 + i as u64, "tools/call", body).await },
                )
            })
            .collect();

        let mut identical = 0;
        let mut errors = 0;
        for (i, h) in handles.into_iter().enumerate() {
            let reply = h.await.expect("stress task join");
            if reply.get("error").is_some() || reply["result"]["isError"] == true {
                errors += 1;
                continue;
            }
            if signature(&tool_output(&reply)) == expected[i % 2] {
                identical += 1;
            }
        }
        assert_eq!(errors, 0, "no request may fail under a 16-wide burst");
        assert_eq!(
            identical, 16,
            "all 16 concurrent NeuralProphet fits must equal their sequential result"
        );
    }
}
