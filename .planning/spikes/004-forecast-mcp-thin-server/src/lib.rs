//! Spike 004 — a THIN, stateless MCP `forecast` tool over the spike-001/003 Prophet port and the
//! spike-002 NeuralProphet-lite. One tool, no model artifact: the request carries the series, the
//! server fits and forecasts inside the call. Same shape as `aprender-mcp-setfit`.
#![allow(clippy::disallowed_methods)]
pub mod np;
pub mod prophet;

use aprender::optim::{ConvergenceStatus, LbfgsF64};
use aprender::primitives::Vector;
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;
use prophet::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::time::Instant;

pub const TOOL_NAME: &str = "forecast";
pub const TOOL_DESCRIPTION: &str = "Fit a time-series forecaster to the series you pass in and return a forecast — in one call, \
 no model to train or store first. Send parallel arrays `ds` (dates, YYYY-MM-DD) and `y` (numbers) plus \
 `horizon` (number of future periods) and optionally `freq` (D, W or MS — daily by default). Default model \
 is `prophet` (trend with changepoints + Fourier seasonality + optional holidays, with 80% uncertainty \
 bands and named components); `neuralprophet` adds an AR-Net over the last `n_lags` values for short-horizon \
 nowcasting. Returns future `ds`, `yhat`, `yhat_lower`, `yhat_upper`, `trend`, components and timing. \
 Bounded at 20,000 points and a 3,650-period horizon.";
pub const MAX_POINTS: usize = 20_000;
pub const MAX_HORIZON: usize = 3_650;
pub const MIN_POINTS: usize = 10;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HolidayArg {
    /// Holiday name (becomes a component).
    pub name: String,
    /// Dates the holiday occurs on, YYYY-MM-DD (past AND future occurrences).
    pub dates: Vec<String>,
    /// Days before the date to include (≤ 0). Default 0.
    #[serde(default)]
    pub lower_window: i64,
    /// Days after the date to include (≥ 0). Default 0.
    #[serde(default)]
    pub upper_window: i64,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForecastArgs {
    /// Timestamps, YYYY-MM-DD (a time part is ignored), ascending, unique.
    pub ds: Vec<String>,
    /// Observed values, same length as `ds`.
    pub y: Vec<f64>,
    /// Number of future periods to forecast (1 … 3650).
    pub horizon: usize,
    /// Period of the future steps: "D" (default), "W", or "MS" (month start).
    #[serde(default)]
    pub freq: Option<String>,
    /// "prophet" (default) or "neuralprophet".
    #[serde(default)]
    pub model: Option<String>,
    /// Prophet growth: "linear" (default), "logistic" (needs `cap`) or "flat".
    #[serde(default)]
    pub growth: Option<String>,
    /// Carrying capacity for logistic growth (original units).
    #[serde(default)]
    pub cap: Option<f64>,
    /// "additive" (default) or "multiplicative" seasonality.
    #[serde(default)]
    pub seasonality_mode: Option<String>,
    /// Width of the uncertainty band, default 0.8.
    #[serde(default)]
    pub interval_width: Option<f64>,
    /// Holidays / events with optional windows (Prophet only).
    #[serde(default)]
    pub holidays: Option<Vec<HolidayArg>>,
    /// NeuralProphet only: number of autoregressive lags (0 = trend + seasonality only).
    #[serde(default)]
    pub n_lags: Option<usize>,
    /// Random seed for the uncertainty simulation / training (default 42).
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub model: String,
    pub freq: String,
    pub n_history: usize,
    pub fit_seconds: f64,
    pub predict_seconds: f64,
    pub ds: Vec<String>,
    pub yhat: Vec<f64>,
    pub yhat_lower: Vec<f64>,
    pub yhat_upper: Vec<f64>,
    pub trend: Vec<f64>,
    pub components: serde_json::Map<String, serde_json::Value>,
    pub diagnostics: serde_json::Value,
}

#[derive(Debug)]
pub enum ForecastError { Validation(String), Internal(String) }
impl std::fmt::Display for ForecastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { match self { Self::Validation(s) | Self::Internal(s) => f.write_str(s) } }
}

fn parse_date(s: &str) -> Result<i64, ForecastError> {
    let date = s.get(..10).ok_or_else(|| ForecastError::Validation(format!("bad date {s:?}: want YYYY-MM-DD")))?;
    let mut it = date.split('-');
    let mut next = |what: &str| -> Result<i64, ForecastError> { it.next().and_then(|p| p.parse::<i64>().ok()).ok_or_else(|| ForecastError::Validation(format!("bad date {s:?}: cannot read {what}"))) };
    let (y, m, d) = (next("year")?, next("month")?, next("day")?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) { return Err(ForecastError::Validation(format!("bad date {s:?}: month/day out of range"))); }
    let days = days_from_civil(y, m as u32, d as u32);
    let (yy, mm, dd) = civil_from_days(days);
    if (yy, mm as i64, dd as i64) != (y, m, d) { return Err(ForecastError::Validation(format!("bad date {s:?}: not a calendar date"))); }
    Ok(days)
}

/// `make_future_dataframe` for D / W / MS.
pub fn future_days(last: i64, horizon: usize, freq: &str) -> Result<Vec<i64>, ForecastError> {
    Ok(match freq {
        "D" => (1..=horizon as i64).map(|i| last + i).collect(),
        "W" => (1..=horizon as i64).map(|i| last + 7 * i).collect(),
        "MS" => {
            let (y, m, _) = civil_from_days(last);
            (1..=horizon as i64).map(|i| { let total = (y * 12 + (m as i64 - 1)) + i; days_from_civil(total.div_euclid(12), (total.rem_euclid(12) + 1) as u32, 1) }).collect()
        }
        other => return Err(ForecastError::Validation(format!("unsupported freq {other:?}: use D, W or MS"))),
    })
}

/// Objective + gradient evaluated ONCE per distinct point; L-BFGS's line search asks for both
/// at the same x, and re-asks at the accepted point next iteration.
struct Cached<'a> { model: Model<'a>, last: RefCell<Option<(Vec<f64>, f64, Vec<f64>)>>, evals: RefCell<usize> }
impl<'a> Cached<'a> {
    fn ensure(&self, x: &[f64]) {
        let hit = self.last.borrow().as_ref().map_or(false, |(k, _, _)| k.as_slice() == x);
        if !hit {
            *self.evals.borrow_mut() += 1;
            let (f, g) = self.model.value_and_grad(x);
            *self.last.borrow_mut() = Some((x.to_vec(), f, g));
        }
    }
    fn f(&self, x: &[f64]) -> f64 { self.ensure(x); self.last.borrow().as_ref().expect("cached").1 }
    fn g(&self, x: &[f64]) -> Vec<f64> { self.ensure(x); self.last.borrow().as_ref().expect("cached").2.clone() }
}

pub struct FitInfo { pub rounds: usize, pub iterations: usize, pub evals: usize, pub objective: f64, pub status: String, pub budget_hit: bool }
/// Per-round L-BFGS iteration cap and the wall-clock budget for the whole fit.
pub const MAX_ITERS_PER_ROUND: usize = 2_000;
pub const FIT_BUDGET_SECS: f64 = 15.0;

/// Spike 001 config + spike 003 restarts.
pub fn fit_prophet(design: &Design, max_rounds: usize) -> (Params, FitInfo) {
    let model = Model::new(design);
    let init = model.init();
    let mut x = Vector::from_vec(model.pack(&init));
    let cache = Cached { model, last: RefCell::new(None), evals: RefCell::new(0) };
    let mut best_f = cache.f(x.as_slice());
    let mut opt = LbfgsF64::new(MAX_ITERS_PER_ROUND, 1e-7, 20);
    let (mut rounds, mut iters, mut status, mut budget_hit) = (0, 0, String::new(), false);
    let t0 = Instant::now();
    for _ in 0..max_rounds {
        let r = opt.minimize(|v: &Vector<f64>| cache.f(v.as_slice()), |v: &Vector<f64>| Vector::from_vec(cache.g(v.as_slice())), &x);
        rounds += 1; iters += r.iterations; status = format!("{:?}", r.status);
        let improved = r.objective_value < best_f - 1e-6 * best_f.abs().max(1.0);
        if improved { best_f = r.objective_value; x = r.solution; }
        if !improved || r.status == ConvergenceStatus::Converged { break; }
        if t0.elapsed().as_secs_f64() > FIT_BUDGET_SECS { budget_hit = true; break; }
    }
    let p = cache.model.unpack(x.as_slice());
    let evals = *cache.evals.borrow();
    (p, FitInfo { rounds, iterations: iters, evals, objective: best_f / cache.model.scale, status, budget_hit })
}

pub fn forecast(args: &ForecastArgs) -> Result<ForecastResponse, ForecastError> {
    // ---- validation (the transport re-checks nothing; this is THE door) ----
    if args.ds.len() != args.y.len() { return Err(ForecastError::Validation(format!("ds has {} entries but y has {}", args.ds.len(), args.y.len()))); }
    if args.ds.len() < MIN_POINTS { return Err(ForecastError::Validation(format!("need at least {MIN_POINTS} points, got {}", args.ds.len()))); }
    if args.ds.len() > MAX_POINTS { return Err(ForecastError::Validation(format!("{} points exceeds max_points {MAX_POINTS}", args.ds.len()))); }
    if args.horizon == 0 || args.horizon > MAX_HORIZON { return Err(ForecastError::Validation(format!("horizon must be 1..={MAX_HORIZON}, got {}", args.horizon))); }
    if args.y.iter().any(|v| !v.is_finite()) { return Err(ForecastError::Validation("y contains a non-finite value".into())); }
    let ds: Vec<i64> = args.ds.iter().map(|s| parse_date(s)).collect::<Result<_, _>>()?;
    if !ds.windows(2).all(|w| w[0] < w[1]) { return Err(ForecastError::Validation("ds must be strictly ascending with no duplicates".into())); }
    let freq = args.freq.clone().unwrap_or_else(|| "D".into());
    let fut = future_days(ds[ds.len() - 1], args.horizon, &freq)?;
    let model_name = args.model.clone().unwrap_or_else(|| "prophet".into());
    let interval_width = args.interval_width.unwrap_or(0.8);
    if !(0.0 < interval_width && interval_width < 1.0) { return Err(ForecastError::Validation("interval_width must be in (0, 1)".into())); }
    let seed = args.seed.unwrap_or(42);
    let (y_min, y_max) = args.y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
    if y_min == y_max { return Err(ForecastError::Validation("y is constant; nothing to fit (Prophet special-cases this too)".into())); }

    match model_name.as_str() {
        "prophet" => {
            let mode = match args.seasonality_mode.as_deref() { None | Some("additive") => Mode::Additive, Some("multiplicative") => Mode::Multiplicative, Some(o) => return Err(ForecastError::Validation(format!("seasonality_mode {o:?}: additive or multiplicative"))) };
            let growth = match args.growth.as_deref() { None | Some("linear") => Growth::Linear, Some("logistic") => Growth::Logistic, Some("flat") => Growth::Flat, Some(o) => return Err(ForecastError::Validation(format!("growth {o:?}: linear, logistic or flat"))) };
            if growth == Growth::Logistic {
                let cap = args.cap.ok_or_else(|| ForecastError::Validation("logistic growth needs cap".into()))?;
                if cap <= y_max { return Err(ForecastError::Validation(format!("cap {cap} must exceed max(y) = {y_max}"))); }
            }
            let mut holidays = Vec::new();
            for h in args.holidays.as_deref().unwrap_or(&[]) {
                if h.lower_window > 0 || h.upper_window < 0 { return Err(ForecastError::Validation(format!("holiday {:?}: lower_window ≤ 0 ≤ upper_window", h.name))); }
                let days: Vec<i64> = h.dates.iter().map(|s| parse_date(s)).collect::<Result<_, _>>()?;
                holidays.push(Holiday { name: h.name.clone(), days, lower_window: h.lower_window, upper_window: h.upper_window, prior_scale: 10.0 });
            }
            let mut spec = Spec::default_linear(auto_seasonalities(&ds, 10.0, mode));
            spec.growth = growth; spec.cap = args.cap; spec.holidays = holidays; spec.holidays_mode = mode; spec.interval_width = interval_width;
            if spec.seasonalities.is_empty() && spec.holidays.is_empty() {
                // Prophet fits a 'zeros' column here; the design needs K ≥ 1 — give it a harmless weekly term with a tiny prior.
                spec.seasonalities.push(Seasonality { name: "weekly".into(), period: 7.0, order: 1, prior_scale: 1e-3, mode });
            }
            let design = make_design(&ds, &args.y, &spec);
            let t0 = Instant::now();
            let (p, info) = fit_prophet(&design, 8);
            let fit_seconds = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let fc = predict(&design, &p, &fut, seed);
            let predict_seconds = t1.elapsed().as_secs_f64();
            let mut components = serde_json::Map::new();
            for (n, v) in &fc.components { components.insert(n.clone(), serde_json::json!(v)); }
            Ok(ForecastResponse { model: "prophet".into(), freq, n_history: ds.len(), fit_seconds, predict_seconds, ds: fut.iter().map(|d| format_ymd(*d)).collect(), yhat: fc.yhat, yhat_lower: fc.yhat_lower, yhat_upper: fc.yhat_upper, trend: fc.trend, components,
                diagnostics: serde_json::json!({"growth": format!("{growth:?}"), "seasonality_mode": format!("{mode:?}"), "seasonalities": spec.seasonalities.iter().map(|s| format!("{} (order {})", s.name, s.order)).collect::<Vec<_>>(), "n_changepoints": design.changepoints_t.len(), "active_changepoints": p.delta.iter().filter(|d| d.abs() > 1e-3).count(), "sigma_obs": p.sigma_obs, "lbfgs": {"rounds": info.rounds, "iterations": info.iterations, "evaluations": info.evals, "objective": info.objective, "last_status": info.status, "budget_hit": info.budget_hit, "iters_per_round_cap": MAX_ITERS_PER_ROUND, "budget_secs": FIT_BUDGET_SECS}, "interval_width": interval_width, "uncertainty_samples": spec.uncertainty_samples}) })
        }
        "neuralprophet" => {
            if freq != "D" { return Err(ForecastError::Validation("neuralprophet in this spike supports freq D only".into())); }
            let n_lags = args.n_lags.unwrap_or(0);
            if n_lags > 365 { return Err(ForecastError::Validation("n_lags ≤ 365".into())); }
            let n_train = ds.len();
            let d = np::NpData::new(&ds, &args.y, n_train, 10, 0.8);
            if n_lags >= d.n_train_grid { return Err(ForecastError::Validation("n_lags must be smaller than the series span in days".into())); }
            let t0 = Instant::now();
            // spike-002 lesson: a short lr sweep selected by train loss stands in for NP's range test;
            // 4× the auto epochs when lags are on (the linear AR case needs the budget).
            let mut best: Option<(f64, f64, np::NpModel, np::TrainLog)> = None;
            let lrs: &[f64] = if n_lags > 0 { &[0.03, 0.1] } else { &[0.01, 0.03, 0.1] };
            for &lr in lrs {
                let cfg = np::TrainConfig { n_lags, ar_layers: if n_lags > 0 { vec![32] } else { vec![] }, max_lr: lr, epochs: if n_lags > 0 { Some(np::auto_epochs(n_train).min(320)) } else { None }, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed };
                let (m, log) = np::train(&d, &cfg, false);
                let fl = *log.epoch_loss.last().unwrap_or(&f64::INFINITY);
                if !fl.is_finite() { continue; }
                if best.as_ref().map_or(true, |b| fl < b.0) { best = Some((fl, lr, m, log)); }
            }
            let (train_loss, selected_lr, m, log) = best.ok_or_else(|| ForecastError::Internal("training diverged for every learning rate".into()))?;
            let fit_seconds = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let (yhat, trend) = if n_lags == 0 {
                (np::predict_ts(&d, &m, &fut), np::predict_trend(&d, &m, &fut))
            } else {
                (np::predict_ar_recursive(&d, &m, &fut), np::predict_trend(&d, &m, &fut))
            };
            // residual-based band (NeuralProphet itself would use quantile regression — not spiked)
            let fitted = if n_lags == 0 { np::predict_ts(&d, &m, &ds) } else { let idx: Vec<usize> = (0..ds.len()).filter(|&i| (ds[i] - d.t0) as usize >= n_lags).map(|i| (ds[i] - d.t0) as usize).collect(); let pr = np::predict_ar_1step(&d, &m, &idx); let mut out = vec![f64::NAN; ds.len()]; let mut k = 0; for i in 0..ds.len() { if (ds[i] - d.t0) as usize >= n_lags { out[i] = pr[k]; k += 1; } } out };
            let resid: Vec<f64> = fitted.iter().zip(&args.y).filter(|(f, _)| f.is_finite()).map(|(f, y)| y - f).collect();
            let sd = (resid.iter().map(|r| r * r).sum::<f64>() / resid.len().max(1) as f64).sqrt();
            let z = normal_quantile((1.0 + interval_width) / 2.0);
            let predict_seconds = t1.elapsed().as_secs_f64();
            let mut components = serde_json::Map::new();
            components.insert("trend".into(), serde_json::json!(trend));
            Ok(ForecastResponse { model: "neuralprophet".into(), freq, n_history: ds.len(), fit_seconds, predict_seconds, ds: fut.iter().map(|d| format_ymd(*d)).collect(), yhat_lower: yhat.iter().map(|v| v - z * sd).collect(), yhat_upper: yhat.iter().map(|v| v + z * sd).collect(), yhat, trend, components,
                diagnostics: serde_json::json!({"n_lags": n_lags, "ar_layers": if n_lags > 0 { vec![32] } else { vec![] }, "epochs": log.epochs, "batch": log.batch, "steps": log.steps, "params": log.n_params, "selected_lr": selected_lr, "final_train_loss": train_loss, "residual_sd": sd, "band": "residual-sd based, not NeuralProphet's quantile regression", "seasonalities": d.seasons.iter().map(|s| format!("{} (order {})", s.name, s.order)).collect::<Vec<_>>()}) })
        }
        other => Err(ForecastError::Validation(format!("model {other:?}: prophet or neuralprophet"))),
    }
}

/// Acklam's inverse normal CDF (enough for band z-scores).
fn normal_quantile(p: f64) -> f64 {
    let a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    let b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    let c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    let d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let pl = 0.02425;
    if p < pl { let q = (-2.0 * p.ln()).sqrt(); (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0) }
    else if p <= 1.0 - pl { let q = p - 0.5; let r = q * q; (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0) }
    else { let q = (-2.0 * (1.0 - p).ln()).sqrt(); -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0) }
}

fn map_error(e: ForecastError) -> pmcp::Error { match e { ForecastError::Validation(s) => pmcp::Error::validation(s), ForecastError::Internal(s) => pmcp::Error::internal(s) } }

/// One stateless tool. The fit runs on a blocking thread: seconds of CPU must not stall the
/// protocol loop, and the autograd tape is thread-local, so two fits never share a thread's tape.
pub fn build_server(name: &str, version: &str) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ForecastArgs, _, _>(TOOL_NAME, TOOL_DESCRIPTION, move |args, _extra| async move {
            let response = tokio::task::spawn_blocking(move || forecast(&args)).await.map_err(|e| pmcp::Error::internal(format!("forecast task join: {e}")))?.map_err(map_error)?;
            serde_json::to_value(&response).map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
        })
        .build()
}

/// The HTTP app: pmcp's streamable-http MCP router at `/mcp` (stateless, JSON responses,
/// localhost-locked CORS) plus a same-origin demo page and two sample datasets.
pub fn http_app(server: Server) -> axum::Router {
    use axum::response::{Html, IntoResponse};
    use axum::routing::get;
    let server = std::sync::Arc::new(tokio::sync::Mutex::new(server));
    let config = pmcp::axum::RouterConfig { server_config: pmcp::server::streamable_http_server::StreamableHttpServerConfig::stateless(), allowed_origins: Some(pmcp::axum::AllowedOrigins::localhost()), ..Default::default() };
    let mcp = pmcp::axum::router_with_config(server, config);
    axum::Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/sample/peyton", get(|| async { ([("content-type", "text/csv")], include_str!("../fixtures/peyton_manning.csv")).into_response() }))
        .route("/sample/air", get(|| async { ([("content-type", "text/csv")], include_str!("../fixtures/air_passengers.csv")).into_response() }))
        .nest("/mcp", mcp)
}
