//! Spike 007 — a THIN, stateless MCP `forecast` tool over the spike-005 Chronos-Bolt port. One
//! zero-shot model compiled into the binary (or resolved from `CHRONOS_MODEL_DIR`), one tool,
//! native quantiles, nothing to fit or store. Same shape as `aprender-mcp-setfit` and spike 004.
#![allow(clippy::disallowed_methods)]
pub mod bolt;
pub mod dates;
pub mod safetensors;

use bolt::{Bolt, Config};
use dates::*;
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

pub const TOOL_NAME: &str = "forecast";
pub const TOOL_DESCRIPTION: &str = "Zero-shot time-series forecast with Chronos-Bolt: no training, no model to store — send the series, \
 get quantile forecasts back in one call. Pass parallel arrays `ds` (dates, YYYY-MM-DD) and `y` (numbers; `null` for a \
 missing value), `horizon` (future periods) and optionally `freq` (D, W or MS — daily by default). Returns future `ds`, \
 `yhat` (median), `yhat_lower`/`yhat_upper` (10th/90th percentiles) and all nine quantiles (0.1 … 0.9). The model looks at \
 the last 2,048 points. Horizons up to 64 are one direct forecast; longer horizons need `allow_long_horizon: true` \
 because the model then rolls its own forecast forward and accuracy degrades past step 64 (max 1,024).";
pub const MAX_POINTS: usize = 20_000;
pub const MIN_POINTS: usize = 4;
pub const MAX_HORIZON: usize = 1_024;

/// Weights and config staged by `build.rs` (empty when the build had no `CHRONOS_EMBED_DIR`).
pub static EMBEDDED_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.safetensors"));
pub static EMBEDDED_CONFIG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/config.json"));

pub struct Model { pub bolt: Bolt, pub name: String, pub n_params: usize, pub dtype: String, pub source: String, pub load_seconds: f64 }

pub fn load_model_from_bytes(weights: &[u8], config: &[u8], source: &str) -> Result<Model, String> {
    let t0 = Instant::now();
    let cfg_json: serde_json::Value = serde_json::from_slice(config).map_err(|e| format!("config.json: {e}"))?;
    let name = cfg_json["_name_or_path"].as_str().unwrap_or("chronos-bolt").rsplit('/').next().unwrap_or("chronos-bolt").to_string();
    let (w, dtype) = safetensors::load_bytes(weights)?;
    let n_params = w.values().map(|t| t.data.len()).sum();
    let bolt = Bolt::load(&w, Config::from_json(&cfg_json));
    Ok(Model { bolt, name, n_params, dtype, source: source.to_string(), load_seconds: t0.elapsed().as_secs_f64() })
}

pub fn load_model_from_dir(dir: &std::path::Path) -> Result<Model, String> {
    let w = std::fs::read(dir.join("model.safetensors")).map_err(|e| format!("read {}/model.safetensors: {e}", dir.display()))?;
    let c = std::fs::read(dir.join("config.json")).map_err(|e| format!("read {}/config.json: {e}", dir.display()))?;
    load_model_from_bytes(&w, &c, &dir.display().to_string())
}

/// Embedded bytes if the build staged them, else `CHRONOS_MODEL_DIR` read as a runtime path.
pub fn resolve_model() -> Result<Model, String> {
    if !EMBEDDED_WEIGHTS.is_empty() { return load_model_from_bytes(EMBEDDED_WEIGHTS, EMBEDDED_CONFIG, "embedded"); }
    let dir = std::env::var_os("CHRONOS_MODEL_DIR").ok_or_else(|| "no embedded model in this build and CHRONOS_MODEL_DIR is unset".to_string())?;
    load_model_from_dir(std::path::Path::new(&dir))
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForecastArgs {
    /// Timestamps, YYYY-MM-DD (a time part is ignored), ascending, unique.
    pub ds: Vec<String>,
    /// Observed values, same length as `ds`; `null` marks a missing value (the model handles gaps).
    pub y: Vec<Option<f64>>,
    /// Number of future periods: 1 … 64 by default, up to 1024 with `allow_long_horizon`.
    pub horizon: usize,
    /// Period of the future steps: "D" (default), "W", or "MS" (month start).
    #[serde(default)]
    pub freq: Option<String>,
    /// Accept a horizon beyond the model's native 64 steps. The model then rolls its own forecast
    /// forward; accuracy past step 64 degrades (measured), and the response carries a warning.
    #[serde(default)]
    pub allow_long_horizon: bool,
}

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub model: String,
    pub freq: String,
    pub n_history: usize,
    pub context_used: usize,
    pub predict_seconds: f64,
    pub ds: Vec<String>,
    pub yhat: Vec<f64>,
    pub yhat_lower: Vec<f64>,
    pub yhat_upper: Vec<f64>,
    pub quantiles: serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    pub diagnostics: serde_json::Value,
}

#[derive(Debug)]
pub enum ForecastError { Validation(String), Internal(String) }
impl std::fmt::Display for ForecastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { match self { Self::Validation(s) | Self::Internal(s) => f.write_str(s) } }
}
fn bad(s: String) -> ForecastError { ForecastError::Validation(s) }

pub fn forecast(model: &Model, args: &ForecastArgs) -> Result<ForecastResponse, ForecastError> {
    // ---- validation (the transport re-checks nothing; this is THE door) ----
    let cfg = &model.bolt.cfg;
    if args.ds.len() != args.y.len() { return Err(bad(format!("ds has {} entries but y has {}", args.ds.len(), args.y.len()))); }
    if args.ds.len() < MIN_POINTS { return Err(bad(format!("need at least {MIN_POINTS} points, got {}", args.ds.len()))); }
    if args.ds.len() > MAX_POINTS { return Err(bad(format!("{} points exceeds max_points {MAX_POINTS}", args.ds.len()))); }
    if args.horizon == 0 { return Err(bad("horizon must be at least 1".into())); }
    if args.horizon > MAX_HORIZON { return Err(bad(format!("horizon {} exceeds the maximum {MAX_HORIZON}", args.horizon))); }
    if args.horizon > cfg.prediction_length && !args.allow_long_horizon {
        return Err(bad(format!("horizon {} exceeds the model's native {} steps; beyond that the model rolls its own forecast forward and accuracy degrades — pass allow_long_horizon: true to accept (max {MAX_HORIZON})", args.horizon, cfg.prediction_length)));
    }
    let n_missing = args.y.iter().filter(|v| v.is_none()).count();
    if args.y.iter().flatten().any(|v| !v.is_finite()) { return Err(bad("y contains a non-finite value".into())); }
    if args.y.len() - n_missing < 2 { return Err(bad("y needs at least 2 observed (non-null) values".into())); }
    let ds: Vec<i64> = args.ds.iter().map(|s| parse_date(s).map_err(bad)).collect::<Result<_, _>>()?;
    if !ds.windows(2).all(|w| w[0] < w[1]) { return Err(bad("ds must be strictly ascending with no duplicates".into())); }
    let freq = args.freq.clone().unwrap_or_else(|| "D".into());
    let fut = future_days(ds[ds.len() - 1], args.horizon, &freq).map_err(bad)?;

    // ---- one zero-shot pass ----
    let context: Vec<f32> = args.y.iter().map(|v| v.map_or(f32::NAN, |x| x as f32)).collect();
    let context_used = context.len().min(cfg.context_length);
    let t0 = Instant::now();
    let (q, forwards) = model.bolt.predict(&context, args.horizon);
    let predict_seconds = t0.elapsed().as_secs_f64();
    let level = |want: f32| cfg.quantiles.iter().position(|&l| (l - want).abs() < 1e-6).ok_or_else(|| ForecastError::Internal(format!("model has no {want} quantile")));
    let (lo, med, hi) = (level(0.1)?, level(0.5)?, level(0.9)?);
    let to64 = |v: &Vec<f32>| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
    let mut quantiles = serde_json::Map::new();
    for (i, lvl) in cfg.quantiles.iter().enumerate() { quantiles.insert(format!("{lvl:.1}"), serde_json::json!(to64(&q[i]))); }
    let rollouts = forwards.saturating_sub(1) / cfg.quantiles.len();
    let warning = (args.horizon > cfg.prediction_length).then(|| format!("horizon {} exceeds the model's native {} steps: steps {}+ come from {} autoregressive rollout(s) of the model's own forecast; spike 006 measured MASE 1.7 past step 64 vs 1.1 within it on daily data", args.horizon, cfg.prediction_length, cfg.prediction_length + 1, rollouts));
    Ok(ForecastResponse {
        model: model.name.clone(), freq, n_history: ds.len(), context_used, predict_seconds,
        ds: fut.iter().map(|d| format_ymd(*d)).collect(),
        yhat: to64(&q[med]), yhat_lower: to64(&q[lo]), yhat_upper: to64(&q[hi]), quantiles, warning,
        diagnostics: serde_json::json!({"model": model.name, "params": model.n_params, "weights_dtype": model.dtype, "weights_source": model.source, "context_length": cfg.context_length, "patch": cfg.patch, "native_horizon": cfg.prediction_length, "rollouts": rollouts, "forwards": forwards, "missing_values": n_missing, "quantile_levels": cfg.quantiles.iter().map(|q| (*q as f64 * 10.0).round() / 10.0).collect::<Vec<f64>>()}),
    })
}

fn map_error(e: ForecastError) -> pmcp::Error { match e { ForecastError::Validation(s) => pmcp::Error::validation(s), ForecastError::Internal(s) => pmcp::Error::internal(s) } }

/// One stateless tool over one immutable model (`Arc`; no per-thread state, so concurrent calls
/// simply share it). The forward runs on a blocking thread so the protocol loop is never stalled.
pub fn build_server(model: Arc<Model>, name: &str, version: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ForecastArgs, _, _>(TOOL_NAME, TOOL_DESCRIPTION, move |args, _extra| {
            let model = model.clone();
            async move {
                let response = tokio::task::spawn_blocking(move || forecast(&model, &args)).await.map_err(|e| pmcp::Error::internal(format!("forecast task join: {e}")))?.map_err(map_error)?;
                serde_json::to_value(&response).map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
            }
        })
        .build()
}

/// The HTTP app: pmcp's streamable-http MCP router at `/mcp` (stateless, JSON responses,
/// localhost-locked CORS) plus a same-origin demo page and two sample datasets.
pub fn http_app(server: Server) -> axum::Router {
    use axum::response::{Html, IntoResponse};
    use axum::routing::get;
    let server = Arc::new(tokio::sync::Mutex::new(server));
    let config = pmcp::axum::RouterConfig { server_config: pmcp::server::streamable_http_server::StreamableHttpServerConfig::stateless(), allowed_origins: Some(pmcp::axum::AllowedOrigins::localhost()), ..Default::default() };
    let mcp = pmcp::axum::router_with_config(server, config);
    axum::Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/sample/peyton", get(|| async { ([("content-type", "text/csv")], include_str!("../fixtures/peyton_manning.csv")).into_response() }))
        .route("/sample/air", get(|| async { ([("content-type", "text/csv")], include_str!("../fixtures/air_passengers.csv")).into_response() }))
        .nest("/mcp", mcp)
}

/// CSV `ds,y` loader, sorted by date and de-duplicated (keep last) — the spike-006 convention.
pub fn load_csv(path: &str) -> Result<(Vec<String>, Vec<Option<f64>>), String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut rows: Vec<(String, Option<f64>)> = Vec::new();
    for line in s.lines().skip(1) {
        let mut it = line.split(',');
        let (Some(d), Some(v)) = (it.next(), it.next()) else { continue };
        let d = d.trim_matches('"').trim().get(..10).unwrap_or("").to_string();
        if d.len() != 10 { continue; }
        let v = v.trim_matches('"').trim();
        rows.push((d, if v.is_empty() || v.eq_ignore_ascii_case("nan") { None } else { v.parse().ok() }));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows.dedup_by(|later, earlier| { if later.0 == earlier.0 { *earlier = later.clone(); true } else { false } });
    Ok(rows.into_iter().unzip())
}
