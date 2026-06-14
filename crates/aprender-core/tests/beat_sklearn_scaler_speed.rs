//! BEAT-SKLEARN-STANDARDSCALER-SPEED — Pillar-1 speed cascade (PMAT-726). **NIGHTLY ONLY.**
//!
//! ```text
//! cargo test -p aprender-core --release --test beat_sklearn_scaler_speed -- --ignored --nocapture
//! ```
//! Time apr AND scikit-learn StandardScaler fit_transform on the SAME data, SAME host, SAME run;
//! gate the RATIO apr_ms/sklearn_ms. Pure O(n·d) mean/std — LAPACK-free, apr's SIMD turf. MEASURED.

#![cfg(test)]

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aprender::datasets::make_regression;
use aprender::prelude::*;
use aprender::preprocessing::StandardScaler;

const N_SAMPLES: usize = 200_000;
const N_FEATURES: usize = 50;
const SEED: u64 = 42;
const RUNS: usize = 5;
/// apr must be at least this fast (ratio = apr/sklearn). Measured ~0.52 (apr ~1.94x faster on a
/// 16-core box); gate at 0.80 (apr >= 1.25x faster) for margin against nightly-host variance.
const RATIO_CEILING: f64 = 0.80;

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

fn time_apr(x: &aprender::Matrix<f32>) -> f64 {
    {
        let mut m = StandardScaler::new();
        let _ = m.fit_transform(x).expect("apr warmup");
    }
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut m = StandardScaler::new();
        let _z = m.fit_transform(x).expect("apr fit_transform");
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    median(&times)
}

fn write_csv(x: &aprender::Matrix<f32>) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("tempfile");
    for i in 0..x.n_rows() {
        let mut line = String::new();
        for j in 0..x.n_cols() {
            if j > 0 {
                line.push(',');
            }
            line.push_str(&x.get(i, j).to_string());
        }
        writeln!(f, "{line}").expect("write csv row");
    }
    f.flush().expect("flush csv");
    f
}

fn time_sklearn(csv: &std::path::Path) -> f64 {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.preprocessing import StandardScaler
X = np.loadtxt(r"{csv}", delimiter=",").astype(np.float32)
ts = []
m = StandardScaler(); _ = m.fit_transform(X)  # warmup
for _ in range({runs}):
    t = time.perf_counter()
    m = StandardScaler(); _ = m.fit_transform(X)
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
        .expect("run uv");
    assert!(
        out.status.success(),
        "sklearn timing failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("SKLEARN_MS="))
        .unwrap_or_else(|| panic!("no SKLEARN_MS in: {stdout}"))
        .trim()
        .parse::<f64>()
        .expect("parse")
}

#[test]
#[ignore = "nightly-only: needs uv + scikit-learn (beat-speed-nightly.yml)"]
fn beat_sklearn_scaler_speed() {
    let (x, _y) = make_regression(N_SAMPLES, N_FEATURES, 0.1, SEED);
    let apr_ms = time_apr(&x);
    let csv = write_csv(&x);
    let sklearn_ms = time_sklearn(csv.path());
    let ratio = apr_ms / sklearn_ms;
    eprintln!(
        "BEAT-SKLEARN-STANDARDSCALER-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms \
         ratio={ratio:.3} (apr {:.2}x faster) on {N_SAMPLES}x{N_FEATURES}, median of {RUNS}",
        sklearn_ms / apr_ms
    );
    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY: apr/sklearn ratio {ratio:.3} > {RATIO_CEILING:.2} (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms)"
    );
}
