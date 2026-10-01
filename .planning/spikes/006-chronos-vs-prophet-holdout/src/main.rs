//! Spike 006 — zero-shot Chronos-Bolt (tiny, small) vs fitted Prophet / NeuralProphet-lite on
//! rolling-origin holdouts. Run from the spike dir: CARGO_TARGET_DIR=../../../target cargo run --release
mod bolt;
mod fit;
mod np;
mod prophet;
mod safetensors;

use bolt::Bolt;
use prophet::*;
use std::time::Instant;

struct Series { name: &'static str, ds: Vec<i64>, y: Vec<f64>, period: usize, splits: Vec<(usize, usize)> }

fn load(path: &str) -> (Vec<i64>, Vec<f64>) {
    let s = std::fs::read_to_string(path).expect("csv");
    let (mut ds, mut y) = (Vec::new(), Vec::new());
    for l in s.lines().skip(1) { let mut it = l.split(','); let d = it.next().expect("ds").trim_matches('"'); let v: f64 = it.next().expect("y").trim_matches('"').parse().expect("y"); ds.push(parse_ymd(d)); y.push(v); }
    // sort by date and keep the LAST value of a duplicated date (wp_log_R has one)
    let mut idx: Vec<usize> = (0..ds.len()).collect(); idx.sort_by_key(|&i| (ds[i], i));
    let (mut ds2, mut y2): (Vec<i64>, Vec<f64>) = (Vec::new(), Vec::new());
    for i in idx { if ds2.last() == Some(&ds[i]) { *y2.last_mut().expect("y") = y[i]; } else { ds2.push(ds[i]); y2.push(y[i]); } }
    (ds2, y2)
}

#[derive(Clone)]
struct Fc { yhat: Vec<f64>, q10: Option<Vec<f64>>, q90: Option<Vec<f64>>, secs: f64 }

fn mae(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64 }
fn rmse(a: &[f64], b: &[f64]) -> f64 { (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt() }
fn std(v: &[f64]) -> f64 { let m = v.iter().sum::<f64>() / v.len() as f64; (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt() }
fn pinball(y: &[f64], q: &[f64], level: f64) -> f64 { y.iter().zip(q).map(|(yv, qv)| { let u = yv - qv; if u >= 0.0 { level * u } else { (level - 1.0) * u } }).sum::<f64>() }
/// Weighted quantile loss over the 0.1/0.5/0.9 levels (the three every model here can produce).
fn wql3(y: &[f64], f: &Fc) -> Option<f64> {
    let (q10, q90) = (f.q10.as_ref()?, f.q90.as_ref()?);
    let denom = y.iter().map(|v| v.abs()).sum::<f64>() * 3.0;
    Some(2.0 * (pinball(y, q10, 0.1) + pinball(y, &f.yhat, 0.5) + pinball(y, q90, 0.9)) / denom)
}

struct Row { series: &'static str, n_train: usize, h: usize, model: String, mae: f64, mase: f64, mase_first64: f64, mase_beyond64: Option<f64>, rmse: f64, cov: Option<f64>, width_rel: Option<f64>, wql3: Option<f64>, secs: f64 }

fn main() {
    let t_all = Instant::now();
    let mut series = vec![
        Series { name: "peyton", ds: vec![], y: vec![], period: 7, splits: vec![(2540, 365), (2200, 90), (2400, 90), (2600, 90), (2815, 90)] },
        Series { name: "wp_log_r", ds: vec![], y: vec![], period: 7, splits: vec![(2498, 365), (2100, 90), (2400, 90), (2773, 90)] },
        Series { name: "air", ds: vec![], y: vec![], period: 12, splits: vec![(120, 24), (96, 12), (108, 12), (132, 12)] },
        Series { name: "retail", ds: vec![], y: vec![], period: 12, splits: vec![(269, 24), (221, 12), (245, 12), (281, 12)] },
    ];
    for s in series.iter_mut() { let (ds, y) = load(&format!("fixtures/{}.csv", match s.name { "peyton" => "peyton_manning", "air" => "air_passengers", "retail" => "retail_sales", _ => "wp_log_R" })); s.ds = ds; s.y = y; }
    let oracle: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("fixtures/chronos_holdout_oracle.json").expect("oracle")).expect("json");
    let mut bolts: Vec<(String, Bolt)> = Vec::new();
    for (label, dir) in [("chronos-bolt-tiny", "models/tiny"), ("chronos-bolt-small", "models/small")] {
        if let Ok(w) = safetensors::load(&format!("{dir}/model.safetensors")) {
            let cfg = bolt::Config::from_json(&serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).expect("cfg")).expect("json"));
            let n: usize = w.values().map(|t| t.data.len()).sum();
            let b = Bolt::load(&w, cfg);
            eprintln!("loaded {label}: {n} params");
            bolts.push((label.into(), b));
        }
    }
    println!("# Spike 006 — zero-shot Chronos-Bolt vs fitted Prophet / NeuralProphet-lite, rolling-origin holdouts\n");
    // ---- small-model parity on the full Peyton context (tiny was proven in spike 005) ----
    if let Some((_, small)) = bolts.iter().find(|(l, _)| l.ends_with("small")) {
        let s = &series[0];
        let ctx: Vec<f32> = s.y.iter().map(|v| *v as f32).collect();
        let q = small.forward(&ctx).quantiles;
        let py: Vec<Vec<f32>> = oracle["models"]["amazon/chronos-bolt-small"]["peyton_full_q64"].as_array().expect("q").iter().map(|r| r.as_array().expect("r").iter().map(|v| v.as_f64().expect("f") as f32).collect()).collect();
        let d = q.iter().flatten().zip(py.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("Bolt-small parity on Peyton (same code as spike 005, bigger config): quantiles_64 max abs diff vs Python **{d:.1e}**.\n");
    }

    println!("## Per-split results\n\nMASE = MAE / in-sample seasonal-naive MAE (period 7 daily, 12 monthly). cov80 / width: 80 % band (Prophet: simulated; Chronos: q10–q90; NP-lite: residual sd). WQL3: weighted quantile loss over q10/q50/q90.\n");
    println!("| series | train → h | model | MAE | MASE | RMSE | cov80 | width/σ | WQL3 | secs |\n|---|---|---|---|---|---|---|---|---|---|");
    let mut rows: Vec<Row> = Vec::new();
    let mut cross = Vec::new();
    let mut panels = Vec::new();
    for s in &series {
        for &(n_train, h) in &s.splits {
            let (tr_ds, tr_y) = (&s.ds[..n_train], &s.y[..n_train]);
            let (te_ds, te_y) = (&s.ds[n_train..n_train + h], &s.y[n_train..n_train + h]);
            let denom = (s.period..n_train).map(|i| (tr_y[i] - tr_y[i - s.period]).abs()).sum::<f64>() / (n_train - s.period) as f64;
            let sigma = std(tr_y);
            let mut fcs: Vec<(String, Fc)> = Vec::new();
            // baselines
            fcs.push(("naive (last value)".into(), Fc { yhat: vec![tr_y[n_train - 1]; h], q10: None, q90: None, secs: 0.0 }));
            fcs.push(("seasonal naive".into(), Fc { yhat: (0..h).map(|t| tr_y[n_train - s.period + t % s.period]).collect(), q10: None, q90: None, secs: 0.0 }));
            // Prophet (spike 001/003/004)
            {
                let spec = Spec::default_linear(auto_seasonalities(tr_ds, 10.0, Mode::Additive));
                let design = make_design(tr_ds, tr_y, &spec);
                let t0 = Instant::now();
                let (p, _) = fit::fit_prophet(&design);
                let f = predict(&design, &p, te_ds, 42);
                fcs.push(("Prophet (Rust port)".into(), Fc { yhat: f.yhat, q10: Some(f.yhat_lower), q90: Some(f.yhat_upper), secs: t0.elapsed().as_secs_f64() }));
            }
            // NeuralProphet-lite trend+seasonality (spike 002/004)
            {
                let t0 = Instant::now();
                let d = np::NpData::new(tr_ds, tr_y, n_train, 10, 0.8);
                let mut best: Option<(f64, np::NpModel)> = None;
                for &lr in &[0.01, 0.03, 0.1] {
                    let cfg = np::TrainConfig { n_lags: 0, ar_layers: vec![], max_lr: lr, epochs: None, batch: None, weight_decay: 1e-3, huber_beta: 0.3, newer_w: 2.0, seed: 7 };
                    let (m, log) = np::train(&d, &cfg, false);
                    let fl = *log.epoch_loss.last().expect("l");
                    if fl.is_finite() && best.as_ref().map_or(true, |b| fl < b.0) { best = Some((fl, m)); }
                }
                let (_, m) = best.expect("np");
                let yhat = np::predict_ts(&d, &m, te_ds);
                let fitted = np::predict_ts(&d, &m, tr_ds);
                let sd = (fitted.iter().zip(tr_y).map(|(f, y)| (y - f) * (y - f)).sum::<f64>() / n_train as f64).sqrt();
                fcs.push(("NeuralProphet-lite (Rust)".into(), Fc { q10: Some(yhat.iter().map(|v| v - 1.2816 * sd).collect()), q90: Some(yhat.iter().map(|v| v + 1.2816 * sd).collect()), yhat, secs: t0.elapsed().as_secs_f64() }));
            }
            // Chronos-Bolt zero-shot
            for (label, b) in &bolts {
                let ctx: Vec<f32> = tr_y.iter().map(|v| *v as f32).collect();
                let t0 = Instant::now();
                let q = b.predict(&ctx, h);
                let secs = t0.elapsed().as_secs_f64();
                let fc = Fc { yhat: q[4].iter().map(|v| *v as f64).collect(), q10: Some(q[0].iter().map(|v| *v as f64).collect()), q90: Some(q[8].iter().map(|v| *v as f64).collect()), secs };
                let key = format!("{}:{n_train}:{h}", s.name);
                let o = &oracle["models"][format!("amazon/{label}")]["forecasts"][&key];
                cross.push((label.clone(), key, (mae(&fc.yhat, te_y) - o["mae_median"].as_f64().expect("mae")).abs(), o["secs"].as_f64().expect("s"), secs));
                fcs.push((format!("{label} (zero-shot)"), fc));
            }
            for (model, f) in &fcs {
                let cov = f.q10.as_ref().zip(f.q90.as_ref()).map(|(lo, hi)| te_y.iter().enumerate().filter(|(i, v)| **v >= lo[*i] && **v <= hi[*i]).count() as f64 / h as f64);
                let width = f.q10.as_ref().zip(f.q90.as_ref()).map(|(lo, hi)| lo.iter().zip(hi).map(|(a, b)| b - a).sum::<f64>() / h as f64 / sigma);
                let k = h.min(64);
                let r = Row { series: s.name, n_train, h, model: model.clone(), mae: mae(&f.yhat, te_y), mase: mae(&f.yhat, te_y) / denom, mase_first64: mae(&f.yhat[..k], &te_y[..k]) / denom, mase_beyond64: if h > 64 { Some(mae(&f.yhat[64..], &te_y[64..]) / denom) } else { None }, rmse: rmse(&f.yhat, te_y), cov, width_rel: width, wql3: wql3(te_y, f), secs: f.secs };
                println!("| {} | {n_train} → {h} | {} | {:.4} | {:.3} | {:.4} | {} | {} | {} | {:.2} |", s.name, r.model, r.mae, r.mase, r.rmse, r.cov.map_or("–".into(), |v| format!("{v:.2}")), r.width_rel.map_or("–".into(), |v| format!("{v:.2}")), r.wql3.map_or("–".into(), |v| format!("{v:.4}")), r.secs);
                rows.push(r);
            }
            if h > 24 || (s.period == 12 && h == 24) { panels.push((s.name, tr_ds.to_vec(), tr_y.to_vec(), te_ds.to_vec(), te_y.to_vec(), fcs.clone())); }
        }
    }
    // ---- summary: mean MASE per series & model, and rank ----
    println!("\n## Summary — mean MASE over the rolling origins (lower is better; 1.0 = seasonal naive in-sample)\n");
    let models: Vec<String> = { let mut v: Vec<String> = Vec::new(); for r in &rows { if !v.contains(&r.model) { v.push(r.model.clone()); } } v };
    println!("| model | {} | mean | MASE steps 1–64 | MASE steps 65+ (daily 90/365 windows) | mean cov80 | mean width/σ | mean WQL3 | total secs |\n|---|{}---|---|---|---|---|---|---|", series.iter().map(|s| s.name).collect::<Vec<_>>().join(" | "), "---|".repeat(series.len()));
    let mut summary = Vec::new();
    for m in &models {
        let per: Vec<f64> = series.iter().map(|s| { let v: Vec<f64> = rows.iter().filter(|r| r.model == *m && r.series == s.name).map(|r| r.mase).collect(); v.iter().sum::<f64>() / v.len() as f64 }).collect();
        let mean = per.iter().sum::<f64>() / per.len() as f64;
        let rs: Vec<&Row> = rows.iter().filter(|r| r.model == *m).collect();
        let avg = |f: &dyn Fn(&Row) -> Option<f64>| { let v: Vec<f64> = rs.iter().filter_map(|r| f(r)).collect(); if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) } };
        let (cov, wid, wq) = (avg(&|r| r.cov), avg(&|r| r.width_rel), avg(&|r| r.wql3));
        let (m64, mb) = (avg(&|r| Some(r.mase_first64)).expect("m64"), avg(&|r| r.mase_beyond64));
        let secs: f64 = rs.iter().map(|r| r.secs).sum();
        println!("| {m} | {} | **{mean:.3}** | {m64:.3} | {} | {} | {} | {} | {secs:.1} |", per.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" | "), mb.map_or("–".into(), |v| format!("{v:.3}")), cov.map_or("–".into(), |v| format!("{v:.2}")), wid.map_or("–".into(), |v| format!("{v:.2}")), wq.map_or("–".into(), |v| format!("{v:.4}")));
        summary.push(serde_json::json!({"model": m, "mase_per_series": per, "mase_mean": mean, "mase_first64": m64, "mase_beyond64": mb, "cov80": cov, "width_rel": wid, "wql3": wq, "secs": secs}));
    }
    println!("\n## Cross-check against the Python Chronos oracle (same splits)\n");
    let worst = cross.iter().map(|c| c.2).fold(0.0, f64::max);
    let (rs_t, py_t): (f64, f64) = cross.iter().fold((0.0, 0.0), |a, c| (a.0 + c.4, a.1 + c.3));
    println!("- {} Chronos forecasts compared: max |MAE(Rust) − MAE(Python)| = **{worst:.1e}** (the metric plumbing and the port agree); total Chronos time Rust {rs_t:.1} s vs torch {py_t:.1} s", cross.len());
    println!("\nTotal wall time {:.0} s.", t_all.elapsed().as_secs_f64());
    std::fs::write("results.json", serde_json::to_string_pretty(&serde_json::json!({"rows": rows.iter().map(|r| serde_json::json!({"series": r.series, "n_train": r.n_train, "h": r.h, "model": r.model, "mae": r.mae, "mase": r.mase, "mase_first64": r.mase_first64, "mase_beyond64": r.mase_beyond64, "rmse": r.rmse, "cov80": r.cov, "width_rel": r.width_rel, "wql3": r.wql3, "secs": r.secs})).collect::<Vec<_>>(), "summary": summary})).expect("json")).expect("write");

    // ---- report ----
    let mut html = String::from(r##"<!doctype html><html><head><meta charset="utf-8"><title>Spike 006 — Chronos vs Prophet holdouts</title><style>body{font:14px system-ui;margin:24px;background:#fafafa;color:#222}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:14px}.legend span{display:inline-block;margin-right:14px;font-size:12px}.sw{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:5px}</style></head><body><h2>Spike 006 — zero-shot Chronos-Bolt vs fitted Prophet on the long holdouts</h2>"##);
    for (name, tr_ds, tr_y, te_ds, te_y, fcs) in &panels {
        let keep = tr_ds.len().saturating_sub(2 * te_ds.len().max(60));
        let xs: Vec<f64> = tr_ds[keep..].iter().chain(te_ds).map(|d| *d as f64).collect();
        let (x0, x1) = (xs[0], xs[xs.len() - 1]);
        let get = |label: &str| fcs.iter().find(|(l, _)| l.starts_with(label)).map(|(_, f)| f.clone());
        let (pro, small, npl) = (get("Prophet"), get("chronos-bolt-small").or_else(|| get("chronos-bolt-tiny")), get("NeuralProphet"));
        let mut all: Vec<f64> = tr_y[keep..].iter().chain(te_y).cloned().collect();
        for f in [&pro, &small, &npl].into_iter().flatten() { all.extend(f.yhat.iter()); if let Some(q) = &f.q10 { all.extend(q.iter()); } if let Some(q) = &f.q90 { all.extend(q.iter()); } }
        let (mut y0, mut y1) = (all.iter().cloned().fold(f64::INFINITY, f64::min), all.iter().cloned().fold(f64::NEG_INFINITY, f64::max)); let pad = 0.05 * (y1 - y0); y0 -= pad; y1 += pad;
        let (w, h, p) = (1100.0, 360.0, 40.0);
        let px = |x: f64| p + (x - x0) / (x1 - x0) * (w - 2.0 * p); let py = |v: f64| h - p - (v - y0) / (y1 - y0) * (h - 2.0 * p);
        let poly = |xs: &[f64], ys: &[f64]| xs.iter().zip(ys).map(|(x, y)| format!("{:.1},{:.1}", px(*x), py(*y))).collect::<Vec<_>>().join(" ");
        let txs: Vec<f64> = te_ds.iter().map(|d| *d as f64).collect();
        let band = |f: &Fc| { let hi = poly(&txs, f.q90.as_ref().expect("q90")); let rx: Vec<f64> = txs.iter().rev().cloned().collect(); let rl: Vec<f64> = f.q10.as_ref().expect("q10").iter().rev().cloned().collect(); format!("{hi} {}", poly(&rx, &rl)) };
        let hist_x: Vec<f64> = tr_ds[keep..].iter().map(|d| *d as f64).collect();
        let mut svg = format!(r##"<rect x="{:.1}" y="{p}" width="{:.1}" height="{:.1}" fill="#f0f4ff"/>"##, px(txs[0]), w - p - px(txs[0]), h - 2.0 * p);
        if let Some(f) = &small { svg.push_str(&format!(r##"<polygon points="{}" fill="#c6e5c6" opacity="0.7"/>"##, band(f))); }
        if let Some(f) = &pro { svg.push_str(&format!(r##"<polygon points="{}" fill="#f4b6b6" opacity="0.5"/>"##, band(f))); }
        svg.push_str(&format!(r##"<polyline points="{}" fill="none" stroke="#999" stroke-width="1"/><polyline points="{}" fill="none" stroke="#222" stroke-width="1.2"/>"##, poly(&hist_x, &tr_y[keep..]), poly(&txs, te_y)));
        if let Some(f) = &npl { svg.push_str(&format!(r##"<polyline points="{}" fill="none" stroke="#ff7f0e" stroke-width="1.2" stroke-dasharray="4,3"/>"##, poly(&txs, &f.yhat))); }
        if let Some(f) = &pro { svg.push_str(&format!(r##"<polyline points="{}" fill="none" stroke="#d62728" stroke-width="1.6"/>"##, poly(&txs, &f.yhat))); }
        if let Some(f) = &small { svg.push_str(&format!(r##"<polyline points="{}" fill="none" stroke="#2ca02c" stroke-width="1.6"/>"##, poly(&txs, &f.yhat))); }
        let mae_line = fcs.iter().map(|(l, f)| format!("{}: {:.3}", l.split(' ').next().unwrap_or(l), mae(&f.yhat, te_y))).collect::<Vec<_>>().join(" · ");
        html.push_str(&format!(r##"<div class="card"><b>{name}</b> — train {} → test {} ({} … {}). MAE: {mae_line}<div class="legend"><span><i class="sw" style="background:#222"></i>test y</span><span><i class="sw" style="background:#d62728"></i>Prophet</span><span><i class="sw" style="background:#f4b6b6;height:10px"></i>Prophet 80 %</span><span><i class="sw" style="background:#2ca02c"></i>Chronos-Bolt-small q50</span><span><i class="sw" style="background:#c6e5c6;height:10px"></i>Chronos q10–q90</span><span><i class="sw" style="background:#ff7f0e"></i>NeuralProphet-lite</span></div><svg width="{w}" height="{h}">{svg}</svg></div>"##, tr_ds.len(), te_ds.len(), format_ymd(te_ds[0]), format_ymd(te_ds[te_ds.len() - 1])));
    }
    html.push_str("</body></html>");
    std::fs::write("report.html", html).expect("report");
    println!("Wrote results.json and report.html.");
}
