//! BEAT-SKLEARN-MULTINOMIALNB-SPEED — Pillar-1 speed cascade (PMAT-728). **NIGHTLY ONLY.**
//!
//! cargo test -p aprender-core --release --test beat_sklearn_multinomialnb_speed -- --ignored --nocapture
//!
//! Time apr AND scikit-learn MultinomialNB fit+predict on the SAME count data, SAME host, SAME run;
//! gate the RATIO apr_ms/sklearn_ms. MultinomialNB is COMPUTE-bound (per-class log-prob matvec over
//! counts + argmax) with NO LAPACK/BLAS — apr precomputes feature_log_prob in fit and takes argmax
//! of the log-posterior directly, so this is the robust (cross-platform) kind of beat. MEASURED.

#![cfg(test)]

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aprender::classification::MultinomialNB;
use aprender::datasets::make_classification;
use aprender::prelude::*;

const N_SAMPLES: usize = 50_000;
const N_FEATURES: usize = 30;
const N_INFORMATIVE: usize = 20;
const N_CLASSES: usize = 8;
const SEED: u64 = 42;
const RUNS: usize = 5;
/// apr must be at least this fast (ratio = apr/sklearn). Tightened once measured.
const RATIO_CEILING: f64 = 0.90; // loosened for CI-host (OpenBLAS) variance; worst observed ~0.75 (PMAT-733)

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// MultinomialNB needs non-negative count features. Map make_classification's continuous features
/// to integer-ish counts via `round(|v| * 4)` — deterministic, class-correlated, same for both sides.
fn to_counts(x: &aprender::Matrix<f32>) -> aprender::Matrix<f32> {
    let (r, c) = x.shape();
    let mut data = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            data.push((x.get(i, j).abs() * 4.0).round());
        }
    }
    aprender::Matrix::from_vec(r, c, data).expect("counts matrix")
}

fn time_apr(x: &aprender::Matrix<f32>, y: &[usize]) -> f64 {
    {
        let mut m = MultinomialNB::new();
        m.fit(x, y).expect("warmup fit");
        let _ = m.predict(x);
    }
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut m = MultinomialNB::new();
        m.fit(x, y).expect("fit");
        let _p = m.predict(x);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    median(&times)
}

fn write_csv(x: &aprender::Matrix<f32>, y: &[usize]) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("tempfile");
    for i in 0..x.n_rows() {
        let mut line = String::new();
        for j in 0..x.n_cols() {
            line.push_str(&x.get(i, j).to_string());
            line.push(',');
        }
        line.push_str(&y[i].to_string());
        writeln!(f, "{line}").expect("row");
    }
    f.flush().expect("flush");
    f
}

fn time_sklearn(csv: &std::path::Path) -> f64 {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.naive_bayes import MultinomialNB
D = np.loadtxt(r"{csv}", delimiter=",")
X, y = D[:, :-1], D[:, -1].astype(np.int64)
ts = []
m = MultinomialNB(); m.fit(X, y); _ = m.predict(X)
for _ in range({runs}):
    t = time.perf_counter()
    m = MultinomialNB(); m.fit(X, y); _ = m.predict(X)
    ts.append((time.perf_counter() - t) * 1000.0)
ts.sort()
print("SKLEARN_MS=%f" % ts[len(ts)//2])
"#,
        csv = csv.display(),
        runs = RUNS
    );
    let out = Command::new("uv")
        .args([
            "run",
            "--with",
            "scikit-learn",
            "--with",
            "numpy",
            "python3",
            "-c",
            &py,
        ])
        .output()
        .expect("uv");
    assert!(
        out.status.success(),
        "sklearn failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("SKLEARN_MS="))
        .unwrap_or_else(|| panic!("no SKLEARN_MS: {stdout}"))
        .trim()
        .parse::<f64>()
        .expect("parse")
}

#[test]
#[ignore = "nightly-only: needs uv + scikit-learn (beat-speed-nightly.yml)"]
fn beat_sklearn_multinomialnb_speed() {
    let (x_raw, y) = make_classification(N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_CLASSES, SEED);
    let x = to_counts(&x_raw);

    let apr_ms = time_apr(&x, &y);
    let csv = write_csv(&x, &y);
    let sklearn_ms = time_sklearn(csv.path());

    let ratio = apr_ms / sklearn_ms;
    eprintln!(
        "BEAT-SKLEARN-MULTINOMIALNB-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms \
         ratio={ratio:.3} (apr {:.2}x faster) on {N_SAMPLES}x{N_FEATURES} classes={N_CLASSES}, median of {RUNS}",
        sklearn_ms / apr_ms
    );
    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-SKLEARN-MULTINOMIALNB-SPEED: ratio {ratio:.3} > {RATIO_CEILING:.2} (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms)"
    );
}
