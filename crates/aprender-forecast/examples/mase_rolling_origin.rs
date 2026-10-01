//! Rolling-origin holdout harness: MASE, coverage and WQL3 for every forecaster this crate
//! ships, against `naive` and `seasonal naive` baselines (D-16).
//!
//! # Why this is an EXAMPLE and not a test
//!
//! D-16: the harness ships as a compiled example, not a per-commit test. It fits 17 windows
//! across 4 series and takes ~15 s in release — too slow for the per-commit suite, and its
//! output is a *judgement* (which model wins where), not an assertion. CI still compiles it:
//! `cargo build --examples --workspace --keep-going` is on the Integration-tests line of
//! `.github/workflows/ci.yml`, so it can never rot silently.
//!
//! # What it measures
//!
//! Four series x rolling origins (5 + 4 + 4 + 4 = 17 windows). Per window and model:
//! MAE, **MASE** (MAE / in-sample seasonal-naive MAE, period 7 daily / 12 monthly — 1.0 means
//! "as good as repeating last week/year"), RMSE, 80 % coverage and band width / sigma, and
//! **WQL3** (weighted quantile loss over q10/q50/q90, the three levels every model here
//! produces). Every window carries a `naive (last value)` and a `seasonal naive` row, and the
//! summary slices the horizon at 64 steps — a single split flatters every model, and the
//! Chronos loss on daily data is almost entirely a past-64-steps effect.
//!
//! # The routing rule this evidence supports — DOCUMENTATION ONLY
//!
//! `model: auto` routing is DEFERRED (06-CONTEXT). This example does not implement a router
//! and nothing in the shipped servers reads it; the rule is recorded here beside the numbers
//! that produced it so that a future `model: auto` starts from measured evidence:
//!
//! - Monthly series, or horizon <= 64 steps -> Chronos-Bolt (zero-shot, no fit).
//! - Otherwise (daily data, long horizon) -> NeuralProphet-lite (best mean MASE) or Prophet
//!   (when bands and named components are wanted).
//! - Chronos past 64 steps only behind `allow_long_horizon`, with a warning.
//!
//! # Running it
//!
//! ```text
//! just mase-rolling-origin
//! # or, directly:
//! CHRONOS_MODEL_DIR=$PWD/models/chronos-bolt-tiny/f32 \
//!     cargo run --release -p aprender-forecast --example mase_rolling_origin
//! ```
//!
//! With `CHRONOS_MODEL_DIR` unset the Prophet / NeuralProphet / baseline rows still run and
//! the Chronos rows are announced as skipped. That print is honest for an EXAMPLE — a test
//! that silently dropped its subject would be a vacuous green, which is why this is not one.
//! When the variable IS set, every Chronos MAE is cross-checked per window against the Python
//! oracle in `tests/fixtures/chronos_holdout_oracle.json` (`chronos-forecasting` 2.3.1).

use std::path::{Path, PathBuf};
use std::time::Instant;

use aprender_forecast::chronos::{load_model_from_dir, Model};
use aprender_forecast::dates::parse_ymd;
use aprender_forecast::fit::fit_prophet;
use aprender_forecast::np;
use aprender_forecast::prophet::{auto_seasonalities, make_design, predict, Mode, Spec};

/// L-BFGS restart budget — the same value `aprender_forecast::forecast` passes.
const PROPHET_ROUNDS: usize = 8;
/// Where the summary slices the horizon (steps 1..=64 vs 65+).
const HORIZON_SLICE: usize = 64;
/// z for a two-sided 80 % interval, used for NeuralProphet-lite's residual-sd band.
const Z80: f64 = 1.2816;

/// One evaluated series and the rolling `(n_train, horizon)` origins taken from it.
struct Series {
    name: &'static str,
    file: &'static str,
    /// Seasonal period for the MASE denominator: 7 for daily, 12 for monthly.
    period: usize,
    splits: &'static [(usize, usize)],
    ds: Vec<i64>,
    y: Vec<f64>,
}

/// One model's forecast for one window.
#[derive(Clone)]
struct Fc {
    yhat: Vec<f64>,
    q10: Option<Vec<f64>>,
    q90: Option<Vec<f64>>,
    secs: f64,
}

/// One row of the per-split table.
struct Row {
    series: &'static str,
    model: String,
    mae: f64,
    mase: f64,
    mase_first: f64,
    mase_beyond: Option<f64>,
    rmse: f64,
    cov80: Option<f64>,
    width_rel: Option<f64>,
    wql3: Option<f64>,
    secs: f64,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Read a two-column `ds,y` CSV, then sort and de-duplicate by `ds`, keeping the LAST value.
///
/// Both steps are load-bearing: `wp_log_R.csv` is not chronological and carries a duplicated
/// date. The Python oracle sorts the same way; when it did not, the cross-check reported an
/// 8.5e-2 disagreement on exactly the eight unsorted forecasts.
fn load_csv(path: &Path) -> (Vec<i64>, Vec<f64>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut ds = Vec::new();
    let mut y = Vec::new();
    for line in text.lines().skip(1) {
        let mut cols = line.split(',');
        let (Some(raw_ds), Some(raw_y)) = (cols.next(), cols.next()) else {
            continue;
        };
        let raw_ds = raw_ds.trim().trim_matches('"');
        let raw_y = raw_y.trim().trim_matches('"');
        if raw_ds.is_empty() || raw_y.is_empty() {
            continue;
        }
        ds.push(parse_ymd(raw_ds));
        y.push(raw_y.parse::<f64>().expect("the y column is a number"));
    }
    let mut order: Vec<usize> = (0..ds.len()).collect();
    order.sort_by_key(|&i| (ds[i], i));
    let mut out_ds: Vec<i64> = Vec::with_capacity(ds.len());
    let mut out_y: Vec<f64> = Vec::with_capacity(y.len());
    for i in order {
        if out_ds.last() == Some(&ds[i]) {
            *out_y.last_mut().expect("a value was just pushed") = y[i];
        } else {
            out_ds.push(ds[i]);
            out_y.push(y[i]);
        }
    }
    (out_ds, out_y)
}

fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

fn stddev(v: &[f64]) -> f64 {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Pinball (quantile) loss at `level`, summed over the horizon.
fn pinball(y: &[f64], q: &[f64], level: f64) -> f64 {
    y.iter()
        .zip(q)
        .map(|(actual, quantile)| {
            let u = actual - quantile;
            if u >= 0.0 {
                level * u
            } else {
                (level - 1.0) * u
            }
        })
        .sum()
}

/// Weighted quantile loss over q10 / q50 / q90 — `None` for a point forecast with no band.
fn wql3(y: &[f64], f: &Fc) -> Option<f64> {
    let (q10, q90) = (f.q10.as_ref()?, f.q90.as_ref()?);
    let denom = y.iter().map(|v| v.abs()).sum::<f64>() * 3.0;
    Some(2.0 * (pinball(y, q10, 0.1) + pinball(y, &f.yhat, 0.5) + pinball(y, q90, 0.9)) / denom)
}

fn fmt_opt(v: Option<f64>, places: usize) -> String {
    v.map_or_else(|| "–".to_string(), |x| format!("{x:.places$}"))
}

/// The Chronos model, or `None` with a printed reason. `None` is a legitimate outcome here:
/// the weights are gitignored and this is an example, not a gate.
fn load_chronos() -> Option<Model> {
    let Some(dir) = std::env::var_os("CHRONOS_MODEL_DIR") else {
        println!(
            "chronos rows skipped: CHRONOS_MODEL_DIR unset (point it at \
             models/chronos-bolt-tiny/f32, e.g. via `just fetch-chronos-tiny`)\n"
        );
        return None;
    };
    let dir = PathBuf::from(dir);
    match load_model_from_dir(&dir) {
        Ok(model) => {
            println!(
                "loaded {} — {} params, weights {}, from {} in {:.0} ms\n",
                model.name,
                model.n_params,
                model.dtype,
                dir.display(),
                model.load_seconds * 1e3
            );
            Some(model)
        }
        Err(error) => {
            println!("chronos rows skipped: {} refused: {error}\n", dir.display());
            None
        }
    }
}

/// The `naive` and `seasonal naive` baselines. Every window carries both: an accuracy claim
/// without them is unfalsifiable, because MASE 1.0 IS the in-sample seasonal naive.
fn baselines(tr_y: &[f64], period: usize, h: usize) -> Vec<(String, Fc)> {
    let n = tr_y.len();
    let last = tr_y[n - 1];
    vec![
        (
            "naive (last value)".to_string(),
            Fc {
                yhat: vec![last; h],
                q10: None,
                q90: None,
                secs: 0.0,
            },
        ),
        (
            "seasonal naive".to_string(),
            Fc {
                yhat: (0..h).map(|t| tr_y[n - period + t % period]).collect(),
                q10: None,
                q90: None,
                secs: 0.0,
            },
        ),
    ]
}

fn prophet_row(tr_ds: &[i64], tr_y: &[f64], te_ds: &[i64]) -> (String, Fc) {
    let spec = Spec::default_linear(auto_seasonalities(tr_ds, 10.0, Mode::Additive));
    let design = make_design(tr_ds, tr_y, &spec);
    let t0 = Instant::now();
    let (params, _info) = fit_prophet(&design, PROPHET_ROUNDS);
    let fc = predict(
        &design,
        &params,
        te_ds,
        42,
        &aprender_forecast::regressors::RegressorChannel::NONE,
    )
    .expect("this example passes an inert regressor channel");
    (
        "Prophet (Rust port)".to_string(),
        Fc {
            yhat: fc.yhat,
            q10: Some(fc.yhat_lower),
            q90: Some(fc.yhat_upper),
            secs: t0.elapsed().as_secs_f64(),
        },
    )
}

/// NeuralProphet-lite: trend + seasonality, no AR lags, best of three learning rates by final
/// epoch loss. The band is a residual-sd band, not a sampled one.
fn np_lite_row(tr_ds: &[i64], tr_y: &[f64], te_ds: &[i64]) -> (String, Fc) {
    let t0 = Instant::now();
    let data = np::NpData::new(tr_ds, tr_y, tr_ds.len(), 10, 0.8);
    let mut best: Option<(f64, np::NpModel)> = None;
    for &lr in &[0.01, 0.03, 0.1] {
        let cfg = np::TrainConfig {
            n_lags: 0,
            ar_layers: vec![],
            max_lr: lr,
            epochs: None,
            batch: None,
            weight_decay: 1e-3,
            huber_beta: 0.3,
            newer_w: 2.0,
            seed: 7,
            // This example is event-free; the block exists but nothing here builds one.
            event_design: None,
            regressors: None,
        };
        let (model, log) = np::train(&data, &cfg, false);
        let loss = *log.epoch_loss.last().expect("at least one epoch");
        if loss.is_finite() && best.as_ref().is_none_or(|(b, _)| loss < *b) {
            best = Some((loss, model));
        }
    }
    let (_, model) = best.expect("at least one finite NeuralProphet-lite fit");
    let yhat = np::predict_ts(&data, &model, te_ds, None, None);
    let fitted = np::predict_ts(&data, &model, tr_ds, None, None);
    let sd = (fitted
        .iter()
        .zip(tr_y)
        .map(|(f, y)| (y - f) * (y - f))
        .sum::<f64>()
        / tr_y.len() as f64)
        .sqrt();
    (
        "NeuralProphet-lite (Rust)".to_string(),
        Fc {
            q10: Some(yhat.iter().map(|v| v - Z80 * sd).collect()),
            q90: Some(yhat.iter().map(|v| v + Z80 * sd).collect()),
            yhat,
            secs: t0.elapsed().as_secs_f64(),
        },
    )
}

/// Zero-shot Chronos-Bolt: no fit, the training window is the context.
fn chronos_row(model: &Model, tr_y: &[f64], h: usize) -> (String, Fc) {
    let ctx: Vec<f32> = tr_y.iter().map(|v| *v as f32).collect();
    let t0 = Instant::now();
    let (q, _forwards) = model.bolt.predict(&ctx, h);
    let secs = t0.elapsed().as_secs_f64();
    let col = |i: usize| -> Vec<f64> { q[i].iter().map(|v| f64::from(*v)).collect() };
    (
        format!("{} (zero-shot)", model.name),
        Fc {
            yhat: col(4),
            q10: Some(col(0)),
            q90: Some(col(8)),
            secs,
        },
    )
}

fn main() {
    let t_all = Instant::now();
    let dir = fixtures();
    let mut series = vec![
        Series {
            name: "peyton",
            file: "peyton_manning.csv",
            period: 7,
            splits: &[(2540, 365), (2200, 90), (2400, 90), (2600, 90), (2815, 90)],
            ds: vec![],
            y: vec![],
        },
        Series {
            name: "wp_log_r",
            file: "wp_log_R.csv",
            period: 7,
            splits: &[(2498, 365), (2100, 90), (2400, 90), (2773, 90)],
            ds: vec![],
            y: vec![],
        },
        Series {
            name: "air",
            file: "air_passengers.csv",
            period: 12,
            splits: &[(120, 24), (96, 12), (108, 12), (132, 12)],
            ds: vec![],
            y: vec![],
        },
        Series {
            name: "retail",
            file: "retail_sales.csv",
            period: 12,
            splits: &[(269, 24), (221, 12), (245, 12), (281, 12)],
            ds: vec![],
            y: vec![],
        },
    ];
    for s in &mut series {
        let (ds, y) = load_csv(&dir.join(s.file));
        s.ds = ds;
        s.y = y;
    }

    println!("# Rolling-origin holdouts — Chronos-Bolt (zero-shot) vs fitted Prophet / NeuralProphet-lite\n");
    let chronos = load_chronos();
    let oracle: Option<serde_json::Value> = chronos.as_ref().map(|_| {
        let path = dir.join("chronos_holdout_oracle.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        serde_json::from_str(&text).expect("chronos_holdout_oracle.json is JSON")
    });

    println!("## Per-split results\n");
    println!(
        "MASE = MAE / in-sample seasonal-naive MAE (period 7 daily, 12 monthly). cov80 / width: \
         80 % band (Prophet: simulated; Chronos: q10–q90; NP-lite: residual sd). WQL3: weighted \
         quantile loss over q10/q50/q90.\n"
    );
    println!("| series | train → h | model | MAE | MASE | RMSE | cov80 | width/σ | WQL3 | secs |");
    println!("|---|---|---|---|---|---|---|---|---|---|");

    let mut rows: Vec<Row> = Vec::new();
    let mut cross: Vec<(String, f64, f64, f64)> = Vec::new();
    for s in &series {
        for &(n_train, h) in s.splits {
            let (tr_ds, tr_y) = (&s.ds[..n_train], &s.y[..n_train]);
            let (te_ds, te_y) = (&s.ds[n_train..n_train + h], &s.y[n_train..n_train + h]);
            // MASE denominator: in-sample seasonal-naive MAE on the TRAINING window only.
            let denom = (s.period..n_train)
                .map(|i| (tr_y[i] - tr_y[i - s.period]).abs())
                .sum::<f64>()
                / (n_train - s.period) as f64;
            let sigma = stddev(tr_y);

            let mut fcs = baselines(tr_y, s.period, h);
            fcs.push(prophet_row(tr_ds, tr_y, te_ds));
            fcs.push(np_lite_row(tr_ds, tr_y, te_ds));
            if let (Some(model), Some(oracle)) = (chronos.as_ref(), oracle.as_ref()) {
                let (label, fc) = chronos_row(model, tr_y, h);
                let key = format!("{}:{n_train}:{h}", s.name);
                let entry = &oracle["models"][format!("amazon/{}", model.name)]["forecasts"][&key];
                if let Some(py_mae) = entry["mae_median"].as_f64() {
                    cross.push((
                        key,
                        (mae(&fc.yhat, te_y) - py_mae).abs(),
                        entry["secs"].as_f64().unwrap_or(f64::NAN),
                        fc.secs,
                    ));
                }
                fcs.push((label, fc));
            }

            for (model, f) in &fcs {
                let cov80 = f.q10.as_ref().zip(f.q90.as_ref()).map(|(lo, hi)| {
                    te_y.iter()
                        .enumerate()
                        .filter(|(i, v)| **v >= lo[*i] && **v <= hi[*i])
                        .count() as f64
                        / h as f64
                });
                let width_rel = f.q10.as_ref().zip(f.q90.as_ref()).map(|(lo, hi)| {
                    lo.iter().zip(hi).map(|(a, b)| b - a).sum::<f64>() / h as f64 / sigma
                });
                let k = h.min(HORIZON_SLICE);
                let row = Row {
                    series: s.name,
                    model: model.clone(),
                    mae: mae(&f.yhat, te_y),
                    mase: mae(&f.yhat, te_y) / denom,
                    mase_first: mae(&f.yhat[..k], &te_y[..k]) / denom,
                    mase_beyond: if h > HORIZON_SLICE {
                        Some(mae(&f.yhat[HORIZON_SLICE..], &te_y[HORIZON_SLICE..]) / denom)
                    } else {
                        None
                    },
                    rmse: rmse(&f.yhat, te_y),
                    cov80,
                    width_rel,
                    wql3: wql3(te_y, f),
                    secs: f.secs,
                };
                println!(
                    "| {} | {n_train} → {h} | {} | {:.4} | {:.3} | {:.4} | {} | {} | {} | {:.2} |",
                    row.series,
                    row.model,
                    row.mae,
                    row.mase,
                    row.rmse,
                    fmt_opt(row.cov80, 2),
                    fmt_opt(row.width_rel, 2),
                    fmt_opt(row.wql3, 4),
                    row.secs
                );
                rows.push(row);
            }
        }
    }

    println!("\n## Summary — mean MASE over the rolling origins (lower is better; 1.0 = seasonal naive in-sample)\n");
    let mut models: Vec<String> = Vec::new();
    for r in &rows {
        if !models.contains(&r.model) {
            models.push(r.model.clone());
        }
    }
    println!(
        "| model | {} | mean | MASE steps 1–{HORIZON_SLICE} | MASE steps {}+ | mean cov80 | \
         mean width/σ | mean WQL3 | total secs |",
        series
            .iter()
            .map(|s| s.name)
            .collect::<Vec<_>>()
            .join(" | "),
        HORIZON_SLICE + 1
    );
    println!(
        "|---|{}---|---|---|---|---|---|---|",
        "---|".repeat(series.len())
    );
    for m in &models {
        let per: Vec<f64> = series
            .iter()
            .map(|s| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.model == *m && r.series == s.name)
                    .map(|r| r.mase)
                    .collect();
                v.iter().sum::<f64>() / v.len() as f64
            })
            .collect();
        let mean = per.iter().sum::<f64>() / per.len() as f64;
        let mine: Vec<&Row> = rows.iter().filter(|r| r.model == *m).collect();
        let avg = |pick: &dyn Fn(&Row) -> Option<f64>| -> Option<f64> {
            let v: Vec<f64> = mine.iter().filter_map(|r| pick(r)).collect();
            if v.is_empty() {
                None
            } else {
                Some(v.iter().sum::<f64>() / v.len() as f64)
            }
        };
        let first = avg(&|r| Some(r.mase_first)).expect("every row has a 1..64 slice");
        let beyond = avg(&|r| r.mase_beyond);
        let secs: f64 = mine.iter().map(|r| r.secs).sum();
        println!(
            "| {m} | {} | **{mean:.3}** | {first:.3} | {} | {} | {} | {} | {secs:.1} |",
            per.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(" | "),
            fmt_opt(beyond, 3),
            fmt_opt(avg(&|r| r.cov80), 2),
            fmt_opt(avg(&|r| r.width_rel), 2),
            fmt_opt(avg(&|r| r.wql3), 4)
        );
    }

    if cross.is_empty() {
        println!("\n(no Chronos rows, so no cross-check against the Python oracle)");
    } else {
        println!("\n## Cross-check against the Python Chronos oracle (same splits)\n");
        let worst = cross.iter().map(|c| c.1).fold(0.0f64, f64::max);
        let rust_s: f64 = cross.iter().map(|c| c.3).sum();
        let torch_s: f64 = cross.iter().map(|c| c.2).sum();
        println!(
            "- {} Chronos forecasts compared against `tests/fixtures/chronos_holdout_oracle.json`: \
             max |MAE(Rust) − MAE(Python)| = **{worst:.1e}**; total Chronos time Rust {rust_s:.1} s \
             vs torch {torch_s:.1} s",
            cross.len()
        );
    }

    println!(
        "\n## The routing rule this table supports (`model: auto` is DEFERRED — documentation only)\n"
    );
    println!("- Monthly series, or horizon <= {HORIZON_SLICE} steps → Chronos-Bolt zero-shot.");
    println!("- Daily series with a long horizon → NeuralProphet-lite (best mean MASE), or Prophet when bands and named components matter.");
    println!(
        "- Chronos past {HORIZON_SLICE} steps only behind `allow_long_horizon`, with a warning."
    );
    println!("\nNo router is implemented here or in either MCP server; this is the evidence a future one would start from.");
    println!("\nTotal wall time {:.0} s.", t_all.elapsed().as_secs_f64());
}
