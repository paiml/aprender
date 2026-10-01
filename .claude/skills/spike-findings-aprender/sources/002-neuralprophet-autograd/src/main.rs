//! Spike 002 — can NeuralProphet's model be trained on aprender's autograd with existing ops?
//! Run from the spike directory: CARGO_TARGET_DIR=../../../target cargo run --release
mod np;
mod prophet;
mod report;

use aprender::autograd::{clear_graph, get_grad, Tensor};
use aprender::nn::loss::{MSELoss, SmoothL1Loss};
use aprender::optim::LbfgsF64;
use aprender::primitives::Vector;
use np::*;
use prophet::*;
use std::time::Instant;

const H: usize = 365;

fn mae(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64 }
fn rmse(a: &[f64], b: &[f64]) -> f64 { (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt() }

/// Spike 001's recommended Prophet fit (exact L1, objective ÷ T, finite guard, m = 20, tol 1e-7).
fn fit_prophet(design: &Design) -> (Params, f64, String) {
    let model = Model::new(design, L1Mode::Exact).scaled().guarded();
    let init = model.init();
    let x0 = Vector::from_vec(model.pack(&init));
    let mut opt = LbfgsF64::new(10_000, 1e-7, 20);
    let t0 = Instant::now();
    let r = opt.minimize(|x: &Vector<f64>| model.objective(x.as_slice()), |x: &Vector<f64>| Vector::from_vec(model.gradient(x.as_slice())), &x0);
    (model.unpack(r.solution.as_slice()), t0.elapsed().as_secs_f64(), format!("{:?} after {} iters", r.status, r.iterations))
}

fn main() {
    let csv = std::fs::read_to_string("fixtures/peyton_manning.csv").expect("csv");
    let mut ds_days = Vec::new();
    let mut y = Vec::new();
    for line in csv.lines().skip(1) {
        let mut it = line.split(',');
        let d = it.next().expect("ds").trim_matches('"');
        let v: f64 = it.next().expect("y").trim_matches('"').parse().expect("y f64");
        ds_days.push(parse_ymd(d));
        y.push(v);
    }
    let n = y.len();
    let n_train = n - H;
    let (test_days, test_y) = (&ds_days[n_train..], &y[n_train..]);
    println!("# Spike 002 — NeuralProphet-lite on aprender's f32 autograd\n\nPeyton Manning: {n} rows, train {n_train}, test = last {H} rows ({} … {}).", format_ymd(test_days[0]), format_ymd(test_days[H - 1]));

    // ---- 0. Does core's SmoothL1Loss backpropagate? ----
    println!("\n## 0. Graph connectivity of core's losses (a Huber loss that cannot backprop cannot train NeuralProphet)\n");
    println!("| loss | param grad after backward() |\n|---|---|");
    for (name, is_huber) in [("MSELoss", false), ("SmoothL1Loss (Huber)", true)] {
        clear_graph();
        let p = Tensor::from_slice(&[0.5, 1.5]).requires_grad();
        let pred = p.pow(2.0);
        let target = Tensor::from_slice(&[1.0, 1.0]);
        let loss = if is_huber { SmoothL1Loss::new().forward(&pred, &target) } else { MSELoss::new().forward(&pred, &target) };
        loss.backward();
        println!("| {name} | {} |", match get_grad(p.id()) { Some(g) => format!("Some({:?})", g.data()), None => "**None — detached** (builds output with `Tensor::new` from raw data)".into() });
        clear_graph();
    }
    println!("\nThis spike therefore builds Huber from graph-connected ops (`sub`, `abs`, `pow`, `mul` by a constant 0/1 mask) — exact value and exact gradient.");

    // ---- 1. Data prep parity with NeuralProphet ----
    let oracle: Option<serde_json::Value> = std::fs::read_to_string("fixtures/np_oracle_peyton.json").ok().and_then(|s| serde_json::from_str(&s).ok());
    let d = NpData::new(&ds_days, &y, n_train, 10, 0.8);
    println!("\n## 1. Data preparation (NeuralProphet defaults)\n");
    println!("- soft normalisation: shift (min) = {:.6}, scale (q95 − min) = {:.6}{}", d.shift, d.scale,
        oracle.as_ref().map(|o| format!("  — NP: shift {}, scale {}", o["trend_seasonality"]["data_params"]["y"]["shift"], o["trend_seasonality"]["data_params"]["y"]["scale"])).unwrap_or_default());
    println!("- time: t = (ds − {}) / {} days{}", format_ymd(d.t0), d.t_span, oracle.as_ref().map(|o| format!("  — NP: {} / {}", o["trend_seasonality"]["data_params"]["ds"]["shift"], o["trend_seasonality"]["data_params"]["ds"]["scale"])).unwrap_or_default());
    println!("- changepoints_t (11 segments): {:?}", d.cps.iter().map(|c| format!("{c:.4}")).collect::<Vec<_>>());
    if let Some(o) = &oracle { let np_cps: Vec<f64> = o["trend_seasonality"]["changepoints_t"].as_array().expect("cps").iter().map(|v| v.as_f64().expect("f")).collect(); println!("  NP changepoints max abs diff: {:.1e}", d.cps.iter().zip(&np_cps).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)); }
    println!("- seasonalities: {:?}", d.seasons.iter().map(|s| format!("{} P={} R={}", s.name, s.period, s.order)).collect::<Vec<_>>());
    println!("- daily grid {} days ({} observed in train, {} imputed by linear interpolation); train grid {} days", d.grid_days.len(), d.grid_observed[..d.n_train_grid].iter().filter(|o| **o).count(), d.n_train_grid - d.grid_observed[..d.n_train_grid].iter().filter(|o| **o).count(), d.n_train_grid);
    println!("- auto batch {} / epochs {} for {} lag-free samples (NP: batch 64, epochs 80)", auto_batch(n_train), auto_epochs(n_train), n_train);

    // ---- 2. Trend + seasonality model: learning-rate sweep ----
    println!("\n## 2. Trend + seasonality (n_lags = 0): AdamW + one-cycle, learning-rate sweep (NP picks lr by a range test)\n");
    println!("| max_lr | epochs | batch | steps | params | train secs | tape len/step | final train loss (Huber) | test MAE (365-ahead) | test RMSE |\n|---|---|---|---|---|---|---|---|---|---|");
    let mut best: Option<(f64, f64, Vec<f64>, TrainLog)> = None;
    let mut sweep_rows = Vec::new();
    for &lr in &[0.003, 0.01, 0.03, 0.1, 0.3, 1.0] {
        let cfg = TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: lr, epochs: None, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
        let (m, log) = train(&d, &cfg, false);
        let yhat = predict_ts(&d, &m, test_days);
        let (e, r) = (mae(&yhat, test_y), rmse(&yhat, test_y));
        let fl = *log.epoch_loss.last().expect("loss");
        println!("| {lr} | {} | {} | {} | {} | {:.2} | {} | {:.5} | {:.4} | {:.4} |", log.epochs, log.batch, log.steps, log.n_params, log.seconds, log.tape_len_per_step, fl, e, r);
        sweep_rows.push(serde_json::json!({"lr": lr, "final_train_loss": fl, "test_mae": e, "test_rmse": r, "secs": log.seconds, "epochs": log.epochs, "batch": log.batch}));
        if best.as_ref().map_or(true, |b| fl < b.1) { best = Some((lr, fl, yhat, log)); }
    }
    let (best_lr, best_loss, best_yhat, best_log) = best.expect("sweep");
    let best_mae = mae(&best_yhat, test_y);
    println!("\nSelected by lowest final TRAIN loss (never by test): max_lr = {best_lr} (loss {best_loss:.5}) → test MAE **{best_mae:.4}**.");

    // ---- 3. Baselines on the same split ----
    println!("\n## 3. Same split, other forecasters\n");
    let seas = auto_seasonalities(&ds_days[..n_train], 10.0);
    let design = make_design(&ds_days[..n_train], &y[..n_train], 25, 0.8, 0.05, &seas);
    let (pp, p_secs, p_status) = fit_prophet(&design);
    let p_pred = predict(&design, &pp, test_days);
    let p_mae = mae(&p_pred.yhat, test_y);
    let mean_fc = y[n_train - 365..n_train].iter().sum::<f64>() / 365.0;
    let mean_mae = mae(&vec![mean_fc; H], test_y);
    let np_ts_mae = oracle.as_ref().and_then(|o| o["trend_seasonality"]["mae_365ahead_on_test_rows"].as_f64());
    let np_ts_secs = oracle.as_ref().and_then(|o| o["trend_seasonality"]["secs"].as_f64());
    println!("| forecaster | fit secs | test MAE (365-ahead) | test RMSE |\n|---|---|---|---|");
    println!("| Rust NeuralProphet-lite (best train-loss lr {best_lr}) | {:.2} | **{best_mae:.4}** | {:.4} |", best_log.seconds, rmse(&best_yhat, test_y));
    if let (Some(m), Some(s)) = (np_ts_mae, np_ts_secs) { println!("| Python NeuralProphet 0.9.0 (torch, auto lr-finder; 363 test rows covered) | {s:.1} | **{m:.4}** | – |"); }
    println!("| Rust Prophet (spike 001 config; {p_status}) | {p_secs:.2} | **{p_mae:.4}** | {:.4} |", rmse(&p_pred.yhat, test_y));
    println!("| last-year mean | 0 | {mean_mae:.4} | – |");

    // ---- 4. AR-Net ----
    println!("\n## 4. AR-Net on 30 stationarised lags, one-step-ahead on the test year (true lags)\n");
    let test_idx: Vec<usize> = test_days.iter().map(|&day| (day - d.t0) as usize).collect();
    let naive: Vec<f64> = (n_train..n).map(|i| y[i - 1]).collect();
    println!("| model | epochs | steps | params | train secs | final train loss | test MAE (1-step) |\n|---|---|---|---|---|---|---|");
    let mut ar_rows = Vec::new();
    let mut ar_lin_pred: Option<Vec<f64>> = None;
    for (label, layers, np_key) in [("AR-Net linear (30 → 1)", vec![], "ar30_linear"), ("AR-Net 30 → 32 → 1 (ReLU)", vec![32usize], "ar30_hidden32")] {
        // NP picks the lr per model with its range test; mirror that with a sweep selected by TRAIN loss.
        let mut picked: Option<(f64, TrainLog, Vec<f64>)> = None;
        let mut sweep = Vec::new();
        for &lr in &[0.003, 0.01, 0.03, 0.1, 0.3, 1.0] {
            let cfg = TrainConfig { n_lags: 30, ar_layers: layers.clone(), max_lr: lr, epochs: None, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
            let (m, log) = train(&d, &cfg, false);
            let yhat = predict_ar_1step(&d, &m, &test_idx);
            let fl = *log.epoch_loss.last().expect("l");
            sweep.push(format!("lr {lr}: loss {fl:.5} / test MAE {:.4}", mae(&yhat, test_y)));
            if picked.as_ref().map_or(true, |p| fl < *p.1.epoch_loss.last().expect("l")) { picked = Some((lr, log, yhat)); }
        }
        let (lr, log, yhat) = picked.expect("sweep");
        let e = mae(&yhat, test_y);
        println!("| Rust {label} (lr {lr} by train loss) | {} | {} | {} | {:.2} | {:.5} | **{e:.4}** |", log.epochs, log.steps, log.n_params, log.seconds, log.epoch_loss.last().expect("l"));
        if let Some(o) = &oracle { println!("| Python NP {label} | {} | – | – | {:.1} | {:.5} | **{:.4}** |", o[np_key]["epochs"], o[np_key]["secs"].as_f64().unwrap_or(0.0), o[np_key]["final_train_loss"].as_f64().unwrap_or(0.0), o[np_key]["mae_1step_test"].as_f64().unwrap_or(0.0)); }
        eprintln!("{label} sweep: {}", sweep.join("; "));
        ar_rows.push(serde_json::json!({"label": label, "lr": lr, "test_mae_1step": e, "secs": log.seconds, "final_train_loss": log.epoch_loss.last(), "sweep": sweep}));
        if layers.is_empty() { ar_lin_pred = Some(yhat); }
    }
    // Is the linear AR gap vs NP an optimisation-budget effect? (convex model → unique optimum)
    eprintln!("AR-linear budget probe:");
    for (label, lr, epochs) in [("lr 0.05, 80 ep", 0.05, None), ("lr 0.15, 80 ep", 0.15, None), ("lr 0.1, 320 ep", 0.1, Some(320usize)), ("lr 0.05, 320 ep", 0.05, Some(320usize)), ("lr 0.1, 800 ep", 0.1, Some(800usize))] {
        let cfg = TrainConfig { n_lags: 30, ar_layers: vec![], max_lr: lr, epochs, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
        let (m, log) = train(&d, &cfg, false);
        let e = mae(&predict_ar_1step(&d, &m, &test_idx), test_y);
        eprintln!("  {label}: final train loss {:.5}, test MAE {e:.4}, {:.2}s", log.epoch_loss.last().expect("l"), log.seconds);
    }
    println!("| naive (previous observed row) | – | – | – | 0 | – | {:.4} |", mae(&naive, test_y));

    // ---- 5. Robustness probes ----
    println!("\n## 5. Robustness probes (trend + seasonality, lr {best_lr})\n");
    let mut seed_maes = Vec::new();
    for seed in [1u64, 2, 3] {
        let cfg = TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: best_lr, epochs: None, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed };
        let (m, log) = train(&d, &cfg, false);
        let e = mae(&predict_ts(&d, &m, test_days), test_y);
        seed_maes.push(e);
        println!("- seed {seed}: final train loss {:.5}, test MAE {e:.4}, all epoch losses finite: {}", log.epoch_loss.last().expect("l"), log.epoch_loss.iter().all(|v| v.is_finite()));
    }
    for (label, epochs) in [("half the epochs", auto_epochs(n_train) / 2), ("double the epochs", auto_epochs(n_train) * 2)] {
        let cfg = TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: best_lr, epochs: Some(epochs), batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
        let (m, log) = train(&d, &cfg, false);
        println!("- {label} ({epochs}): final train loss {:.5}, test MAE {:.4}, {:.2}s", log.epoch_loss.last().expect("l"), mae(&predict_ts(&d, &m, test_days), test_y), log.seconds);
    }
    {
        let cfg = TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: best_lr, epochs: None, batch: Some(n_train), weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
        let (m, log) = train(&d, &cfg, false);
        println!("- full-batch ({} rows, {} steps): final train loss {:.5}, test MAE {:.4}, {:.2}s", n_train, log.steps, log.epoch_loss.last().expect("l"), mae(&predict_ts(&d, &m, test_days), test_y), log.seconds);
    }
    {
        let cfg = TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: best_lr, epochs: None, batch: None, weight_decay: 0.0, huber_beta: 0.3, newer_w: 1.0, seed: 7 };
        let (m, log) = train(&d, &cfg, false);
        println!("- no weight decay, no newer-sample weighting: final train loss {:.5}, test MAE {:.4}", log.epoch_loss.last().expect("l"), mae(&predict_ts(&d, &m, test_days), test_y));
    }
    let short = 60;
    let d_short = NpData::new(&ds_days[..short + 30], &y[..short + 30], short, 10, 0.8);
    let cfg = TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: best_lr, epochs: None, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
    let (m, log) = train(&d_short, &cfg, false);
    println!("- {short}-row series: auto batch {} / epochs {}, {:.2}s, seasonalities {:?}, test MAE (next 30 rows) {:.4}", log.batch, log.epochs, log.seconds, d_short.seasons.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), mae(&predict_ts(&d_short, &m, &ds_days[short..short + 30]), &y[short..short + 30]));

    // ---- 6. Ops audit ----
    println!("\n## 6. Autograd ops this model needed\n\n`matmul` (2-D), `transpose`, `broadcast_add` (bias), `add`, `sub`, `mul` (same shape), `mul_scalar`, `abs`, `pow`, `mean`, `relu`, `view`, `backward`, `no_grad`, `clear_graph`. Missing but worked around: a graph-connected Huber loss (core's `SmoothL1Loss` is detached), a `where`/`clamp` op (constant mask instead), `cat` (avoided by reshaping the lag-time seasonality through `view`).");

    // ---- 7. Artifacts ----
    let results = serde_json::json!({
        "split": {"n": n, "n_train": n_train, "h": H},
        "normalisation": {"shift": d.shift, "scale": d.scale},
        "lr_sweep": sweep_rows, "selected_lr": best_lr,
        "test_mae_365ahead": {"rust_np_lite": best_mae, "python_np": np_ts_mae, "rust_prophet": p_mae, "last_year_mean": mean_mae},
        "ar": ar_rows, "naive_1step_mae": mae(&naive, test_y), "seed_maes": seed_maes,
    });
    std::fs::write("results.json", serde_json::to_string_pretty(&results).expect("json")).expect("write");
    let xs_all: Vec<f64> = ds_days.iter().map(|&d| d as f64).collect();
    let xs_test: Vec<f64> = test_days.iter().map(|&d| d as f64).collect();
    let np_lite_label = format!("Rust NeuralProphet-lite trend+seasonality (MAE {best_mae:.3})");
    let prophet_label = format!("Rust Prophet (MAE {p_mae:.3})");
    let mut series = vec![
        report::Series { name: "y (observed)", color: "#999", dash: false, x: xs_all.clone(), y: y.clone(), width: 1.0 },
        report::Series { name: &np_lite_label, color: "#d62728", dash: false, x: xs_test.clone(), y: best_yhat.clone(), width: 1.8 },
        report::Series { name: &prophet_label, color: "#1f77b4", dash: true, x: xs_test.clone(), y: p_pred.yhat.clone(), width: 1.8 },
    ];
    let np_yhat: Option<(Vec<f64>, Vec<f64>)> = oracle.as_ref().and_then(|o| o["trend_seasonality"]["yhat_by_ds"].as_object()).map(|m| {
        let mut v: Vec<(f64, f64)> = m.iter().map(|(k, v)| (parse_ymd(k) as f64, v.as_f64().unwrap_or(f64::NAN))).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("f"));
        v.into_iter().unzip()
    });
    let np_label = format!("Python NeuralProphet (MAE {})", np_ts_mae.map(|v| format!("{v:.3}")).unwrap_or("n/a".into()));
    if let Some((xs, ys)) = &np_yhat { series.push(report::Series { name: &np_label, color: "#2ca02c", dash: true, x: xs.clone(), y: ys.clone(), width: 1.5 }); }
    let ar_label = format!("Rust AR-Net linear, 1-step-ahead (MAE {:.3})", ar_rows[0]["test_mae_1step"].as_f64().unwrap_or(0.0));
    if let Some(p) = &ar_lin_pred { series.push(report::Series { name: &ar_label, color: "#ff7f0e", dash: false, x: xs_test.clone(), y: p.clone(), width: 1.0 }); }
    let x_min = ds_days[n_train - 730] as f64;
    let table = format!("<b>Test year, 365-day-ahead MAE</b> (lower is better): Rust NP-lite <b>{best_mae:.4}</b> · Python NeuralProphet <b>{}</b> · Rust Prophet <b>{p_mae:.4}</b> · last-year mean {mean_mae:.4}<br><b>One-step-ahead MAE</b>: Rust AR-Net linear <b>{:.4}</b> · naive {:.4}", np_ts_mae.map(|v| format!("{v:.4}")).unwrap_or("n/a".into()), ar_rows[0]["test_mae_1step"].as_f64().unwrap_or(0.0), mae(&naive, test_y));
    let html = report::render("Spike 002 — NeuralProphet-lite on aprender autograd", "Peyton Manning, last two training years + held-out test year. Forecasts start at the shaded region.", &series, ds_days[n_train] as f64, x_min, ds_days[n - 1] as f64, &table);
    std::fs::write("report.html", html).expect("write report");
    println!("\nWrote results.json and report.html.");
}
