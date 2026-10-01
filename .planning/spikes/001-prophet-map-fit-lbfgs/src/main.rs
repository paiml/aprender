//! Spike 001 — does aprender's `LbfgsF64` reproduce Stan's MAP fit of Prophet?
//!
//! Run from the spike directory (fixtures default to the three checked in):
//!   CARGO_TARGET_DIR=../../../target cargo run --release [-- fixtures/*.json]
mod prophet;
mod report;

use aprender::optim::{ConvergenceStatus, LbfgsF64};
use aprender::primitives::Vector;
use prophet::*;
use serde::Deserialize;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Deserialize)]
struct FixSeasonality { name: String, period: f64, fourier_order: usize, prior_scale: f64, mode: String }
#[derive(Deserialize)]
struct FixParams { k: f64, m: f64, delta: Vec<f64>, beta: Vec<f64>, sigma_obs: f64 }
#[derive(Deserialize)]
struct FixHistory { ds: Vec<String>, t: Vec<f64>, y: Vec<f64>, y_scaled: Vec<f64> }
#[derive(Deserialize)]
struct FixForecast { ds: Vec<String>, trend: Vec<f64>, yhat: Vec<f64>, components: BTreeMap<String, Vec<f64>> }
#[derive(Deserialize)]
struct Fixture {
    dataset: String,
    prophet_version: String,
    fit_seconds: f64,
    y_scale: f64,
    t_scale_days: f64,
    changepoints_t: Vec<f64>,
    changepoint_prior_scale: f64,
    seasonality_columns: Vec<String>,
    prior_scales: Vec<f64>,
    seasonalities: Vec<FixSeasonality>,
    params: FixParams,
    log_posterior_at_map_unnormalized: f64,
    history: FixHistory,
    #[serde(rename = "X_first3")] x_first3: Vec<Vec<f64>>,
    #[serde(rename = "X_last3")] x_last3: Vec<Vec<f64>>,
    forecast: FixForecast,
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Central finite-difference check of `gradient` against `objective`.
fn fd_check(model: &Model, theta: &[f64], h: f64) -> (f64, f64) {
    let g = model.gradient(theta);
    let (mut worst_abs, mut worst_rel) = (0.0_f64, 0.0_f64);
    for i in 0..theta.len() {
        let (mut tp, mut tm) = (theta.to_vec(), theta.to_vec());
        tp[i] += h;
        tm[i] -= h;
        let fd = (model.objective(&tp) - model.objective(&tm)) / (2.0 * h);
        let abs = (fd - g[i]).abs();
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(abs / fd.abs().max(g[i].abs()).max(1e-8));
    }
    (worst_abs, worst_rel)
}

#[derive(Clone, Copy)]
struct Cfg { l1: L1Mode, scaled: bool, guard: bool, hist: usize, tol: f64 }
impl Cfg {
    fn model<'a>(&self, d: &'a Design) -> Model<'a> {
        let mut m = Model::new(d, self.l1);
        if self.scaled { m = m.scaled(); }
        if self.guard { m = m.guarded(); }
        m
    }
    fn name(&self) -> String {
        format!("{}{}{}, m={}, tol {:.0e}",
            match self.l1 { L1Mode::Exact => "exact".to_string(), L1Mode::Smooth(e) => format!("smooth {e:.0e}") },
            if self.scaled { ", f/T" } else { ", raw" }, if self.guard { ", guard" } else { "" }, self.hist, self.tol)
    }
}
/// The configuration this spike recommends for the build.
const RECOMMENDED: Cfg = Cfg { l1: L1Mode::Exact, scaled: true, guard: true, hist: 20, tol: 1e-7 };
const BASELINE: Cfg = Cfg { l1: L1Mode::Exact, scaled: false, guard: false, hist: 5, tol: 1e-5 };
const STAN_HISTORY: Cfg = Cfg { l1: L1Mode::Exact, scaled: true, guard: true, hist: 5, tol: 1e-7 };
const SMOOTH_BOUND: Cfg = Cfg { l1: L1Mode::Smooth(1e-4), scaled: false, guard: true, hist: 5, tol: 1e-5 };

struct FitRun {
    label: String,
    status: ConvergenceStatus,
    iterations: usize,
    evals: usize,
    seconds: f64,
    /// Objective under the EXACT, unscaled L1 (comparable across variants and to Python).
    objective_exact: f64,
    grad_norm: f64,
    params: Params,
}

fn fit(design: &Design, config: Cfg, init: &Params, max_iter: usize) -> FitRun {
    let model = config.model(design);
    let evals = Cell::new(0usize);
    let obj = |x: &Vector<f64>| { evals.set(evals.get() + 1); model.objective(x.as_slice()) };
    let grad = |x: &Vector<f64>| Vector::from_vec(model.gradient(x.as_slice()));
    let x0 = Vector::from_vec(model.pack(init));
    let mut opt = LbfgsF64::new(max_iter, config.tol, config.hist);
    let t0 = Instant::now();
    let r = opt.minimize(obj, grad, &x0);
    let seconds = t0.elapsed().as_secs_f64();
    let params = model.unpack(r.solution.as_slice());
    let exact = Model::new(design, L1Mode::Exact);
    FitRun { label: config.name(), status: r.status, iterations: r.iterations, evals: evals.get(), seconds,
        objective_exact: exact.objective(&exact.pack(&params)), grad_norm: r.gradient_norm, params }
}

fn active(delta: &[f64]) -> usize { delta.iter().filter(|d| d.abs() > 1e-3).count() }

struct DatasetOutcome { name: String, n: usize, py_secs: f64, rec: FitRun, rec_yhat_hist: f64, rec_yhat_fut: f64, y_range: f64, rows: Vec<serde_json::Value> }

fn run_dataset(path: &str, deep: bool) -> DatasetOutcome {
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(path).expect("read fixture")).expect("parse fixture");
    println!("\n---\n\n# Dataset `{}` — {} rows, Python Prophet {} fit {:.2}s", fx.dataset, fx.history.ds.len(), fx.prophet_version, fx.fit_seconds);

    // ---- 1. Rebuild Stan's data from raw (ds, y) ----
    let ds_days: Vec<i64> = fx.history.ds.iter().map(|s| parse_ymd(s)).collect();
    let seasonalities = auto_seasonalities(&ds_days, 10.0);
    let fx_seas: Vec<(String, f64, usize)> = fx.seasonalities.iter().map(|s| { assert_eq!(s.mode, "additive"); (s.name.clone(), s.period, s.fourier_order) }).collect();
    let our_seas: Vec<(String, f64, usize)> = seasonalities.iter().map(|s| (s.name.clone(), s.period, s.order)).collect();
    println!("Auto seasonalities: Rust {our_seas:?} vs Python {fx_seas:?}");
    assert_eq!(our_seas, fx_seas, "auto seasonality mismatch");
    let design = make_design(&ds_days, &fx.history.y, 25, 0.8, fx.changepoint_prior_scale, &seasonalities);
    let t_diff = max_abs_diff(&design.t, &fx.history.t);
    let ys_diff = max_abs_diff(&design.y_scaled, &fx.history.y_scaled);
    let cp_diff = max_abs_diff(&design.changepoints_t, &fx.changepoints_t);
    let x_first: Vec<f64> = design.x[..3 * design.k].to_vec();
    let x_last: Vec<f64> = design.x[(design.t.len() - 3) * design.k..].to_vec();
    let x_diff = max_abs_diff(&x_first, &fx.x_first3.concat()).max(max_abs_diff(&x_last, &fx.x_last3.concat()));
    println!("\n## 1. Data preparation parity (Rust from raw ds/y vs Prophet's Stan data)\n\n| quantity | max abs diff |\n|---|---|");
    println!("| t (scaled time) | {t_diff:.2e} |\n| y_scaled (y_scale {} vs {}) | {ys_diff:.2e} |\n| changepoints_t ({}) | {cp_diff:.2e} |\n| Fourier X first/last 3 rows (K={}) | {x_diff:.2e} |",
        design.y_scale, fx.y_scale, design.changepoints_t.len(), design.k);
    assert_eq!(design.k, fx.seasonality_columns.len());
    assert_eq!(design.prior_scales, fx.prior_scales);
    assert!((design.t_scale_days - fx.t_scale_days).abs() < 1e-9);
    assert!(t_diff < 1e-12 && ys_diff < 1e-12 && cp_diff < 1e-12 && x_diff < 1e-9, "design mismatch");

    // ---- 2. Objective parity at Python's MAP ----
    let py = Params { k: fx.params.k, m: fx.params.m, delta: fx.params.delta.clone(), beta: fx.params.beta.clone(), sigma_obs: fx.params.sigma_obs };
    let exact = Model::new(&design, L1Mode::Exact);
    let theta_py = exact.pack(&py);
    let f_py = exact.objective(&theta_py);
    let g_py = exact.gradient(&theta_py);
    let g_norm = g_py.iter().map(|v| v * v).sum::<f64>().sqrt();
    let s = design.changepoints_t.len();
    let g_norm_no_delta: f64 = g_py.iter().enumerate().filter(|(i, _)| *i < 2 || *i >= 2 + s).map(|(_, v)| v * v).sum::<f64>().sqrt();
    println!("\n## 2. Objective parity at Python's MAP\n\nRust f(θ_py) = {f_py:.6}, Python −lp = {:.6}, abs diff {:.1e}. Gradient norm where Stan stopped: {g_norm:.3} (without the {s} δ subgradients: {g_norm_no_delta:.3}). Python δ active: {} of {s}.",
        -fx.log_posterior_at_map_unnormalized, (f_py + fx.log_posterior_at_map_unnormalized).abs(), active(&fx.params.delta));
    assert!((f_py + fx.log_posterior_at_map_unnormalized).abs() < 1e-6, "objective mismatch");

    // ---- 3. Gradient check ----
    let init = exact.init();
    let theta0 = exact.pack(&init);
    if deep {
        let smooth = Model::new(&design, L1Mode::Smooth(1e-4));
        println!("\n## 3. Analytic gradient vs central finite differences (h = 1e-6)\n\n| point | L1 mode | worst abs | worst rel |\n|---|---|---|---|");
        for (name, th) in [("init", &theta0), ("python MAP", &theta_py)] {
            let (a, r) = fd_check(&exact, th, 1e-6);
            println!("| {name} | exact | {a:.2e} | {r:.2e} |");
            let (a, r) = fd_check(&smooth, th, 1e-6);
            println!("| {name} | smooth 1e-4 | {a:.2e} | {r:.2e} |");
        }
        let mut theta_p = theta_py.clone();
        for (i, v) in theta_p.iter_mut().enumerate() { *v += 0.01 * ((i as f64 * 0.37).sin()); }
        let (a, r) = fd_check(&exact, &theta_p, 1e-6);
        println!("| perturbed MAP (no δ at 0) | exact | {a:.2e} | {r:.2e} |");
        assert!(r < 1e-4, "gradient check failed at perturbed point");
    }

    // ---- 4. Fits ----
    println!("\n## 4. L-BFGS fits from Prophet's init (k, m through the endpoints; δ = β = 0; σ = 1)\n\nPython f* = {f_py:.4}\n");
    println!("| variant | status | iters | f evals | secs | f (exact L1) | Δf vs Python | grad norm | σ | active δ |\n|---|---|---|---|---|---|---|---|---|---|");
    let cfgs: Vec<Cfg> = if deep { vec![BASELINE, RECOMMENDED, STAN_HISTORY, SMOOTH_BOUND] } else { vec![BASELINE, RECOMMENDED, STAN_HISTORY, SMOOTH_BOUND] };
    let mut runs = Vec::new();
    for c in &cfgs {
        let r = fit(&design, *c, &init, 10_000);
        println!("| {} | {:?} | {} | {} | {:.3} | {:.4} | {:+.4} | {:.3} | {:.5} | {} |", r.label, r.status, r.iterations, r.evals, r.seconds, r.objective_exact, r.objective_exact - f_py, r.grad_norm, r.params.sigma_obs, active(&r.params.delta));
        runs.push(r);
    }

    // ---- 5. Forecast parity ----
    let fc_days: Vec<i64> = fx.forecast.ds.iter().map(|s| parse_ymd(s)).collect();
    let last = *ds_days.last().expect("nonempty");
    let mut own_future = ds_days.clone();
    own_future.extend((1..=365).map(|i| last + i));
    assert_eq!(own_future, fc_days, "make_future_dataframe parity");
    let n_hist = ds_days.len();
    let y_range = fx.history.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - fx.history.y.iter().cloned().fold(f64::INFINITY, f64::min);
    let comp_names: Vec<&String> = fx.forecast.components.keys().collect();
    println!("\n## 5. Forecast parity vs Python (history {n_hist} rows + 365 daily future; y range {y_range:.3})\n");
    println!("| variant | yhat hist max abs | yhat hist RMSE | yhat future max abs | yhat future RMSE | trend future max abs | {} |\n|---|---|---|---|---|---|{}|",
        comp_names.iter().map(|c| format!("{c} max abs")).collect::<Vec<_>>().join(" | "), "---|".repeat(comp_names.len()));
    let mut rows = Vec::new();
    let mut rec: Option<(FitRun, Prediction, f64, f64)> = None;
    for (i, r) in runs.into_iter().enumerate() {
        let p = predict(&design, &r.params, &fc_days);
        let yh_h = max_abs_diff(&p.yhat[..n_hist], &fx.forecast.yhat[..n_hist]);
        let yh_hr = rmse(&p.yhat[..n_hist], &fx.forecast.yhat[..n_hist]);
        let yh_f = max_abs_diff(&p.yhat[n_hist..], &fx.forecast.yhat[n_hist..]);
        let yh_fr = rmse(&p.yhat[n_hist..], &fx.forecast.yhat[n_hist..]);
        let tr_f = max_abs_diff(&p.trend[n_hist..], &fx.forecast.trend[n_hist..]);
        let comps: Vec<String> = comp_names.iter().map(|c| {
            let ours = &p.components.iter().find(|(n, _)| n == *c).expect("component").1;
            format!("{:.4}", max_abs_diff(ours, &fx.forecast.components[*c]))
        }).collect();
        println!("| {} | {yh_h:.4} | {yh_hr:.5} | {yh_f:.4} | {yh_fr:.5} | {tr_f:.4} | {} |", r.label, comps.join(" | "));
        rows.push(serde_json::json!({"label": r.label, "status": format!("{:?}", r.status), "iterations": r.iterations, "evals": r.evals, "seconds": r.seconds,
            "objective_exact": r.objective_exact, "python_objective": f_py, "yhat_hist_max_abs": yh_h, "yhat_future_max_abs": yh_f, "yhat_future_rmse": yh_fr}));
        if i == 1 { rec = Some((r, p, yh_h, yh_f)); }
    }
    let (rec_run, rec_pred, rec_h, rec_f) = rec.expect("recommended run");
    let p_py = predict(&design, &py, &fc_days);
    println!("\nPython params through Rust predict(): yhat max abs diff {:.1e}, trend {:.1e} (predict-path parity).", max_abs_diff(&p_py.yhat, &fx.forecast.yhat), max_abs_diff(&p_py.trend, &fx.forecast.trend));

    // ---- 6. Robustness probes (deep only) ----
    if deep {
        println!("\n## 6. Robustness probes (recommended config)\n");
        let mut pert = rec_run.params.clone();
        for (i, b) in pert.beta.iter_mut().enumerate() { *b += 0.05 * ((i as f64) * 1.3).cos(); }
        for (i, d) in pert.delta.iter_mut().enumerate() { *d += 0.2 * ((i as f64) * 0.7).sin(); }
        pert.k += 0.3; pert.m -= 0.2; pert.sigma_obs = 0.5;
        let r2 = fit(&design, RECOMMENDED, &pert, 10_000);
        let p2 = predict(&design, &r2.params, &fc_days);
        println!("- (a) restart from a perturbed point: {:?}, {} iters, f = {:.4} (Δ vs recommended {:+.1e}); yhat max abs diff vs recommended {:.1e}", r2.status, r2.iterations, r2.objective_exact, r2.objective_exact - rec_run.objective_exact, max_abs_diff(&p2.yhat, &rec_pred.yhat));
        let y_big: Vec<f64> = fx.history.y.iter().map(|v| v * 1e6).collect();
        let d_big = make_design(&ds_days, &y_big, 25, 0.8, fx.changepoint_prior_scale, &seasonalities);
        let r3 = fit(&d_big, RECOMMENDED, &RECOMMENDED.model(&d_big).init(), 10_000);
        let p3: Vec<f64> = predict(&d_big, &r3.params, &fc_days).yhat.iter().map(|v| v / 1e6).collect();
        println!("- (b) y × 1e6: {:?}, yhat/1e6 max abs diff vs unscaled {:.1e}", r3.status, max_abs_diff(&p3, &rec_pred.yhat));
        for n_short in [60usize, 30] {
            let d_s = make_design(&ds_days[..n_short], &fx.history.y[..n_short], 25, 0.8, fx.changepoint_prior_scale, &auto_seasonalities(&ds_days[..n_short], 10.0));
            let r = fit(&d_s, RECOMMENDED, &RECOMMENDED.model(&d_s).init(), 10_000);
            println!("- (c) first {n_short} rows only (Python would use Newton; seasonalities {:?}; {} changepoints): {:?}, {} iters, {:.3}s, σ = {:.4}, active δ {}", d_s.seasonalities.iter().map(|s| s.name.clone()).collect::<Vec<_>>(), d_s.changepoints_t.len(), r.status, r.iterations, r.seconds, r.params.sigma_obs, active(&r.params.delta));
        }
        let d_nocp = make_design(&ds_days, &fx.history.y, 0, 0.8, fx.changepoint_prior_scale, &seasonalities);
        let r5 = fit(&d_nocp, RECOMMENDED, &RECOMMENDED.model(&d_nocp).init(), 10_000);
        println!("- (d) n_changepoints = 0: {:?}, {} iters, k = {:.4}, σ = {:.4}", r5.status, r5.iterations, r5.params.k, r5.params.sigma_obs);
        let y_const = vec![7.0; 200];
        let d_const = make_design(&ds_days[..200], &y_const, 25, 0.8, fx.changepoint_prior_scale, &seasonalities);
        let r6 = fit(&d_const, RECOMMENDED, &RECOMMENDED.model(&d_const).init(), 10_000);
        println!("- (e) constant y (200 rows): {:?}, {} iters, σ = {:.1e}, f = {:.1} — unbounded below as σ → 0; Prophet special-cases y.min()==y.max() (σ = 1e-9, no fit) and so must we", r6.status, r6.iterations, r6.params.sigma_obs, r6.objective_exact);
        let per_eval = rec_run.seconds / rec_run.evals as f64;
        println!("- (f) cost model: {} objective evals in {:.3}s = {:.0} µs per eval at T×K = {}×{}", rec_run.evals, rec_run.seconds, per_eval * 1e6, n_hist, design.k);
    }

    let html = report::render(&fx.forecast.ds, &fx.history.y, &rec_pred.yhat, &fx.forecast.yhat, &rec_pred.trend, &fx.forecast.trend, n_hist, &format!("{} ({})", rec_run.label, fx.dataset), rec_run.seconds, fx.fit_seconds, &fx.params.delta, &rec_run.params.delta);
    std::fs::write(format!("report-{}.html", fx.dataset), html).expect("write report");
    DatasetOutcome { name: fx.dataset, n: n_hist, py_secs: fx.fit_seconds, rec: rec_run, rec_yhat_hist: rec_h, rec_yhat_fut: rec_f, y_range, rows }
}

fn main() {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        paths = ["peyton_manning", "air_passengers", "retail_sales"].iter().map(|n| format!("fixtures/{n}_prophet140.json")).collect();
    }
    println!("# Spike 001 — Prophet MAP via aprender `LbfgsF64` vs Python Prophet (Stan L-BFGS)");
    let mut outcomes = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        outcomes.push(run_dataset(p, i == 0));
    }
    println!("\n---\n\n# Summary (recommended config: {})\n", RECOMMENDED.name());
    println!("| dataset | rows | Rust fit secs | Python fit secs | status | Δf vs Python | yhat hist max abs | yhat future max abs | y range |\n|---|---|---|---|---|---|---|---|---|");
    for o in &outcomes {
        println!("| {} | {} | {:.3} | {:.2} | {:?} | {:+.4} | {:.4} | {:.4} | {:.1} |", o.name, o.n, o.rec.seconds, o.py_secs, o.rec.status, o.rec.objective_exact - o.rows[1]["python_objective"].as_f64().expect("f"), o.rec_yhat_hist, o.rec_yhat_fut, o.y_range);
    }
    let results = serde_json::json!({ "recommended": RECOMMENDED.name(), "datasets": outcomes.iter().map(|o| serde_json::json!({"dataset": o.name, "rows": o.n, "runs": o.rows})).collect::<Vec<_>>() });
    std::fs::write("results.json", serde_json::to_string_pretty(&results).expect("json")).expect("write results");
    println!("\nWrote results.json and report-<dataset>.html (Rust vs Python yhat overlay, residual panel, δ table).");
}
