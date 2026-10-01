//! Spike 003 — intervals, components, holidays, logistic growth, multiplicative seasonality.
//! Run from the spike directory: CARGO_TARGET_DIR=../../../target cargo run --release
mod prophet3;
mod report;

use aprender::optim::LbfgsF64;
use aprender::primitives::Vector;
use prophet3::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Deserialize)] struct FixSeason { name: String, period: f64, fourier_order: usize, prior_scale: f64, mode: String }
#[derive(Deserialize)] struct FixHoliday { holiday: String, ds: String, lower_window: i64, upper_window: i64 }
#[derive(Deserialize)] struct FixParams { k: f64, m: f64, delta: Vec<f64>, beta: Vec<f64>, sigma_obs: f64 }
#[derive(Deserialize)] struct FixHistory { ds: Vec<String>, t: Vec<f64>, y: Vec<f64>, y_scaled: Vec<f64> }
#[derive(Deserialize)] struct FixForecast { ds: Vec<String>, trend: Vec<f64>, yhat: Vec<f64>, yhat_lower: Vec<f64>, yhat_upper: Vec<f64>, trend_lower: Vec<f64>, trend_upper: Vec<f64>, components: BTreeMap<String, Vec<f64>> }
#[derive(Deserialize)] struct FixUnc { interval_width: f64, uncertainty_samples: usize, future_band_mean_width: f64, hist_band_mean_width: f64, future_trend_band_mean_width: f64, future_band_last30_mean_width: f64 }
#[derive(Deserialize)] struct FixHoldout { n_train: usize, coverage_by_seed: Vec<f64>, mean_width_by_seed: Vec<f64>, mae: f64 }
#[derive(Deserialize)]
struct Fixture {
    dataset: String, mode: String, growth: String, seasonality_mode: String, fit_seconds: f64,
    y_scale: f64, changepoints_t: Vec<f64>, changepoint_prior_scale: f64, holidays_prior_scale: f64,
    cap: Option<f64>, holidays: Vec<FixHoliday>, seasonalities: Vec<FixSeason>, columns: Vec<String>, prior_scales: Vec<f64>,
    s_a: Vec<f64>, s_m: Vec<f64>, params: FixParams, history: FixHistory,
    #[serde(rename = "X_first3")] x_first3: Vec<Vec<f64>>, #[serde(rename = "X_last3")] x_last3: Vec<Vec<f64>>,
    forecast: FixForecast, uncertainty: FixUnc, holdout: Option<FixHoldout>,
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 { assert_eq!(a.len(), b.len()); a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }
fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

fn fd_check(model: &Model, theta: &[f64], h: f64) -> (f64, f64, usize) {
    let g = model.gradient(theta);
    let (mut wa, mut wr, mut wi) = (0.0_f64, 0.0_f64, 0);
    for i in 0..theta.len() {
        let (mut tp, mut tm) = (theta.to_vec(), theta.to_vec());
        tp[i] += h; tm[i] -= h;
        let fd = (model.objective(&tp) - model.objective(&tm)) / (2.0 * h);
        let abs = (fd - g[i]).abs();
        let rel = abs / fd.abs().max(g[i].abs()).max(1e-8);
        if rel > wr { wr = rel; wi = i; }
        wa = wa.max(abs);
    }
    (wa, wr, wi)
}

/// Spike 001's config plus RESTARTS: exact-L1 L-BFGS ends `Stalled` at a kink; restarting from
/// the stall point with fresh curvature history keeps descending until it genuinely cannot.
fn fit(design: &Design) -> (Params, f64, String, f64) {
    let model = Model::new(design);
    let init = model.init();
    let mut x = Vector::from_vec(model.pack(&init));
    let mut opt = LbfgsF64::new(10_000, 1e-7, 20);
    let t0 = Instant::now();
    let mut best_f = model.objective(x.as_slice());
    let mut trail = Vec::new();
    let mut total_iters = 0;
    for round in 0..8 {
        let r = opt.minimize(|v: &Vector<f64>| model.objective(v.as_slice()), |v: &Vector<f64>| Vector::from_vec(model.gradient(v.as_slice())), &x);
        total_iters += r.iterations;
        let improved = r.objective_value < best_f - 1e-12 * best_f.abs().max(1.0);
        trail.push(format!("r{round}:{:?}/{}→{:.6}", r.status, r.iterations, r.objective_value / model.scale));
        if !improved { break; }
        best_f = r.objective_value;
        x = r.solution;
        if r.status == aprender::optim::ConvergenceStatus::Converged { break; }
    }
    let p = model.unpack(x.as_slice());
    let exact = Model { d: design, scale: 1.0, guard: false };
    let f_exact = exact.objective(&exact.pack(&p));
    (p, t0.elapsed().as_secs_f64(), format!("{} restarts, {total_iters} iters [{}]", trail.len(), trail.join(" ")), f_exact)
}

fn spec_from(fx: &Fixture, ds_days: &[i64]) -> Spec {
    let mode = if fx.seasonality_mode == "multiplicative" { Mode::Multiplicative } else { Mode::Additive };
    let mut seas = auto_seasonalities(ds_days, 10.0, mode);
    seas.sort_by_key(|s| match s.name.as_str() { "yearly" => 0, "weekly" => 1, _ => 2 });
    let mut hol: Vec<Holiday> = Vec::new();
    for h in &fx.holidays {
        match hol.iter_mut().find(|x| x.name == h.holiday) {
            Some(x) => x.days.push(parse_ymd(&h.ds)),
            None => hol.push(Holiday { name: h.holiday.clone(), days: vec![parse_ymd(&h.ds)], lower_window: h.lower_window, upper_window: h.upper_window, prior_scale: fx.holidays_prior_scale }),
        }
    }
    let growth = match fx.growth.as_str() { "logistic" => Growth::Logistic, "flat" => Growth::Flat, _ => Growth::Linear };
    let mut spec = Spec::default_linear(seas);
    spec.growth = growth; spec.cap = fx.cap; spec.holidays = hol; spec.holidays_mode = mode; spec.changepoint_prior_scale = fx.changepoint_prior_scale;
    spec.interval_width = fx.uncertainty.interval_width; spec.uncertainty_samples = fx.uncertainty.uncertainty_samples;
    spec
}

fn main() {
    let names = ["peyton_default", "peyton_holidays", "wp_log_R_logistic", "air_multiplicative"];
    println!("# Spike 003 — Prophet intervals, components, holidays, logistic growth, multiplicative seasonality\n");
    let mut panels = Vec::new();
    let mut summary = Vec::new();
    for name in names {
        let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/{name}_prophet140.json")).expect("fixture")).expect("parse");
        let ds_days: Vec<i64> = fx.history.ds.iter().map(|s| parse_ymd(s)).collect();
        let spec = spec_from(&fx, &ds_days);
        let d = make_design(&ds_days, &fx.history.y, &spec);
        println!("\n---\n\n## `{}` — mode **{}**, growth {}, seasonality {} ({} rows, K = {})\n", fx.dataset, fx.mode, fx.growth, fx.seasonality_mode, ds_days.len(), d.k);
        // 1. design parity
        let our_cols: Vec<String> = d.cols.iter().map(|c| c.name.clone()).collect();
        assert_eq!(our_cols, fx.columns, "column names/order");
        assert_eq!(d.s_a, fx.s_a); assert_eq!(d.s_m, fx.s_m); assert_eq!(d.prior_scales, fx.prior_scales);
        let x_first: Vec<f64> = d.x[..3 * d.k].to_vec(); let x_last: Vec<f64> = d.x[(d.t.len() - 3) * d.k..].to_vec();
        let xd = max_abs_diff(&x_first, &fx.x_first3.concat()).max(max_abs_diff(&x_last, &fx.x_last3.concat()));
        // holiday columns: compare full X for holiday columns against fixture forecast? (not in fixture) — check column sums instead
        let hol_hits: usize = d.cols.iter().enumerate().filter(|(_, c)| c.holiday.is_some()).map(|(ci, _)| (0..d.t.len()).filter(|&i| d.x[i * d.k + ci] == 1.0).count()).sum();
        println!("1. Design parity: columns {:?} identical ({} cols), s_a/s_m/prior_scales identical, t/y_scaled/changepoints max diff {:.1e}, X first/last rows {xd:.1e}{}", if d.k <= 4 { format!("{our_cols:?}") } else { format!("[{} … {}]", our_cols[0], our_cols[d.k - 1]) }, d.k,
            max_abs_diff(&d.t, &fx.history.t).max(max_abs_diff(&d.y_scaled, &fx.history.y_scaled)).max(max_abs_diff(&d.changepoints_t, &fx.changepoints_t)), if hol_hits > 0 { format!(", {hol_hits} holiday-indicator ones in X") } else { String::new() });
        assert!(xd < 1e-9);
        // 2. objective + gradient at Python MAP
        let py = Params { k: fx.params.k, m: fx.params.m, delta: fx.params.delta.clone(), beta: fx.params.beta.clone(), sigma_obs: fx.params.sigma_obs };
        let exact = Model { d: &d, scale: 1.0, guard: false };
        let th = exact.pack(&py);
        let f_py = exact.objective(&th);
        let mut thp = th.clone(); for (i, v) in thp.iter_mut().enumerate() { *v += 0.01 * ((i as f64 * 0.37).sin()) + 0.005; }
        let (wa, wr, wi) = fd_check(&exact, &thp, 1e-6);
        println!("2. Gradient vs central FD at a perturbed MAP (all {} params incl. {}): worst rel {wr:.1e} (param {wi}), worst abs {wa:.1e}", th.len(), match spec.growth { Growth::Logistic => "logistic γ-recursion", _ => "linear trend" });
        assert!(wr < 1e-4, "gradient check failed");
        // 3. fit + forecast parity
        let (p, secs, status, f_rs) = fit(&d);
        let fc_days: Vec<i64> = fx.forecast.ds.iter().map(|s| parse_ymd(s)).collect();
        let f = predict(&d, &p, &fc_days, 42);
        let f_py_params = predict(&d, &py, &fc_days, 42);
        let n_hist = ds_days.len();
        let yr = fx.history.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - fx.history.y.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("3. Fit: {status}, {secs:.3}s, f = {f_rs:.4} vs Python {f_py:.4} (Δ {:+.3}); Python fit {:.2}s", f_rs - f_py, fx.fit_seconds);
        println!("   Forecast vs Python (y range {yr:.2}): yhat hist max|Δ| {:.4}, future max|Δ| {:.4}, trend future {:.4}; Python params through Rust predict(): yhat {:.1e}, trend {:.1e}",
            max_abs_diff(&f.yhat[..n_hist], &fx.forecast.yhat[..n_hist]), max_abs_diff(&f.yhat[n_hist..], &fx.forecast.yhat[n_hist..]), max_abs_diff(&f.trend[n_hist..], &fx.forecast.trend[n_hist..]),
            max_abs_diff(&f_py_params.yhat, &fx.forecast.yhat), max_abs_diff(&f_py_params.trend, &fx.forecast.trend));
        // 4. components
        let mut comp_lines = Vec::new();
        for (nm, v) in &f_py_params.components {
            if let Some(pyv) = fx.forecast.components.get(nm) { comp_lines.push(format!("{nm} {:.1e}", max_abs_diff(v, pyv))); }
        }
        let add = &f.components.iter().find(|(n, _)| n == "additive_terms").expect("add").1;
        let mul = &f.components.iter().find(|(n, _)| n == "multiplicative_terms").expect("mul").1;
        let recon: Vec<f64> = (0..f.yhat.len()).map(|i| f.trend[i] * (1.0 + mul[i]) + add[i]).collect();
        println!("4. Components (Python params through Rust): max|Δ| per component vs Python — {}; trend·(1+multiplicative_terms)+additive_terms reconstructs yhat to {:.1e}", comp_lines.join(", "), max_abs_diff(&recon, &f.yhat));
        // 5. intervals
        let fut = n_hist..f.yhat.len();
        let width = |lo: &[f64], hi: &[f64]| -> f64 { mean(&lo.iter().zip(hi).map(|(a, b)| b - a).collect::<Vec<_>>()) };
        let (w_fut, w_hist) = (width(&f.yhat_lower[fut.clone()], &f.yhat_upper[fut.clone()]), width(&f.yhat_lower[..n_hist], &f.yhat_upper[..n_hist]));
        let w_tr = width(&f.trend_lower[fut.clone()], &f.trend_upper[fut.clone()]);
        let w_last30 = width(&f.yhat_lower[f.yhat.len() - 30..], &f.yhat_upper[f.yhat.len() - 30..]);
        // seed sensitivity of OUR band and of Python params through our simulator
        let f2 = predict(&d, &p, &fc_days, 7);
        let w_fut2 = width(&f2.yhat_lower[fut.clone()], &f2.yhat_upper[fut.clone()]);
        println!("5. 80% intervals ({} draws) — mean band width: history Rust {w_hist:.4} vs Python {:.4}; future Rust {w_fut:.4} (seed 7: {w_fut2:.4}) vs Python {:.4}; last 30 days Rust {w_last30:.4} vs Python {:.4}; future trend band Rust {w_tr:.4} vs Python {:.4}",
            spec.uncertainty_samples, fx.uncertainty.hist_band_mean_width, fx.uncertainty.future_band_mean_width, fx.uncertainty.future_band_last30_mean_width, fx.uncertainty.future_trend_band_mean_width);
        let inside_hist = (0..n_hist).filter(|&i| fx.history.y[i] >= f.yhat_lower[i] && fx.history.y[i] <= f.yhat_upper[i]).count() as f64 / n_hist as f64;
        println!("   In-sample coverage of the 80% band: Rust {:.3}", inside_hist);
        // 6. holdout coverage (default fixture only)
        let mut holdout_line = String::new();
        if let Some(h) = &fx.holdout {
            let (tr_days, tr_y) = (&ds_days[..h.n_train], &fx.history.y[..h.n_train]);
            let (te_days, te_y) = (&ds_days[h.n_train..], &fx.history.y[h.n_train..]);
            let spec_tr = spec_from(&fx, tr_days);
            let d_tr = make_design(tr_days, tr_y, &spec_tr);
            let (p_tr, _, _, _) = fit(&d_tr);
            let mut covs = Vec::new(); let mut widths = Vec::new(); let mut maes = Vec::new();
            for seed in [1u64, 2, 3] {
                let ft = predict(&d_tr, &p_tr, te_days, seed);
                covs.push((0..te_y.len()).filter(|&i| te_y[i] >= ft.yhat_lower[i] && te_y[i] <= ft.yhat_upper[i]).count() as f64 / te_y.len() as f64);
                widths.push(width(&ft.yhat_lower, &ft.yhat_upper));
                maes.push(mean(&ft.yhat.iter().zip(te_y).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>()));
            }
            holdout_line = format!("6. Holdout (train {} / test {}): Rust coverage {:?} (mean width {:?}, MAE {:.4}) vs Python coverage {:?} (mean width {:?}, MAE {:.4})",
                h.n_train, te_y.len(), covs.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(), widths.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(), maes[0], h.coverage_by_seed.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(), h.mean_width_by_seed.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(), h.mae);
            println!("{holdout_line}");
        }
        summary.push(serde_json::json!({"fixture": name, "fit_secs": secs, "status": status, "delta_f_vs_python": f_rs - f_py, "yhat_future_max_abs": max_abs_diff(&f.yhat[n_hist..], &fx.forecast.yhat[n_hist..]), "band_future_rust": w_fut, "band_future_python": fx.uncertainty.future_band_mean_width, "band_hist_rust": w_hist, "band_hist_python": fx.uncertainty.hist_band_mean_width, "coverage_in_sample": inside_hist, "holdout": holdout_line}));
        let xs: Vec<f64> = fc_days.iter().map(|&x| x as f64).collect();
        let x_from = fc_days[n_hist.saturating_sub(730)] as f64;
        let keep: Vec<usize> = (0..xs.len()).filter(|&i| xs[i] >= x_from).collect();
        let sel = |v: &[f64]| -> Vec<f64> { keep.iter().map(|&i| v[i]).collect() };
        let keep_h: Vec<usize> = (0..n_hist).filter(|&i| ds_days[i] as f64 >= x_from).collect();
        panels.push(report::Panel { title: format!("{} — {} (growth {}, seasonality {})", fx.dataset, fx.mode, fx.growth, fx.seasonality_mode),
            xs_hist: keep_h.iter().map(|&i| ds_days[i] as f64).collect(), y: keep_h.iter().map(|&i| fx.history.y[i]).collect(),
            xs: sel(&xs), yhat: sel(&f.yhat), lower: sel(&f.yhat_lower), upper: sel(&f.yhat_upper), py_lower: sel(&fx.forecast.yhat_lower), py_upper: sel(&fx.forecast.yhat_upper), py_yhat: sel(&fx.forecast.yhat),
            note: format!("Rust fit {secs:.3}s ({status}); future band width Rust {w_fut:.3} vs Python {:.3}; yhat future max|Δ| {:.3} on y range {yr:.2}", fx.uncertainty.future_band_mean_width, max_abs_diff(&f.yhat[n_hist..], &fx.forecast.yhat[n_hist..])) });
    }
    std::fs::write("results.json", serde_json::to_string_pretty(&summary).expect("json")).expect("write");
    std::fs::write("report.html", report::render("Spike 003 — Prophet intervals, components, holidays, logistic, multiplicative", &panels)).expect("write");
    println!("\nWrote results.json and report.html (four panels: y, Rust yhat, Rust 80% band, Python band edges).");
}
