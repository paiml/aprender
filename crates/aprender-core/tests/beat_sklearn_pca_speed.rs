//! BEAT-SKLEARN-PCA-SPEED — Pillar-1 re-measurement of the PCA row (#3148). **NIGHTLY ONLY.**
//!
//! `#[ignore]`d: needs `uv` + scikit-learn. Run by `.github/workflows/beat-speed-nightly.yml`:
//!
//! ```text
//! cargo test -p aprender-core --release --test beat_sklearn_pca_speed -- --ignored --nocapture
//! ```
//!
//! apr `PCA::new(K)` against `sklearn.decomposition.PCA(n_components=K)`, both on their
//! default `svd_solver="auto"`, fit+transform on the same data, same host, same run, on
//! two shapes: TALL 10_000x100 and WIDE 2_000x1_000 (the regime where the pre-#3148
//! covariance route paid O(d³) for a d x d eigendecomposition). At K=10 both libraries
//! resolve "auto" to the randomized solver (max(n,d) > 500 and K < 0.8·min(n,d)) with the
//! same oversampling (10) and power iterations (4 tall, 7 wide), so the comparison is
//! algorithm-for-algorithm; what differs is LAPACK/BLAS vs apr's matmuls.
//!
//! Measured 2026-09-26 on lambda (load ~50, sklearn 1.9.1): ratio 0.54 tall, 0.28 wide.
//! The pre-#3148 covariance code measured 0.50 tall but 24.4 wide, so the WIDE shape is
//! the one that catches a return to the d x d route. The ceiling (0.80, apr >= 1.25x
//! faster on BOTH shapes) leaves margin for runner noise above the measured ratios.

#![cfg(test)]

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aprender::datasets::make_classification;
use aprender::prelude::*;
use aprender::preprocessing::PCA;

/// (label, n_samples, n_features)
const SHAPES: [(&str, usize, usize); 2] = [("tall", 10_000, 100), ("wide", 2_000, 1_000)];
const K: usize = 10;
const SEED: u64 = 42;
const RUNS: usize = 5;
/// Matches contracts/beat-sklearn-pca-speed-v1.yaml beat_threshold.
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
    let run = || {
        let mut p = PCA::new(K);
        let z = p.fit_transform(x).expect("fit_transform");
        assert_eq!(z.shape(), (x.n_rows(), K));
    };
    run(); // warmup
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        run();
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
        let row: Vec<String> = (0..x.n_cols()).map(|j| x.get(i, j).to_string()).collect();
        writeln!(f, "{}", row.join(",")).expect("row");
    }
    f.flush().expect("flush");
    f
}

fn time_sklearn(csv: &std::path::Path) -> f64 {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.decomposition import PCA
X = np.loadtxt(r"{csv}", delimiter=",").astype(np.float64)
def run():
    z = PCA(n_components={k}).fit_transform(X)
    assert z.shape == (X.shape[0], {k})
run()  # warmup
ts = []
for _ in range({runs}):
    t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1000.0)
ts.sort()
print("SKLEARN_MS=%f" % ts[len(ts)//2])
"#,
        csv = csv.display(),
        k = K,
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
fn beat_sklearn_pca_speed() {
    let mut over = Vec::new();
    for (label, n, d) in SHAPES {
        let (x, _y) = make_classification(n, d, 20, 5, SEED);
        let apr_ms = time_apr(&x);
        let csv = write_csv(&x);
        let sklearn_ms = time_sklearn(csv.path());
        let ratio = apr_ms / sklearn_ms;
        eprintln!(
            "BEAT-SKLEARN-PCA-SPEED[{label}]: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms ratio={ratio:.3} \
             on {n}x{d} k={K} (svd_solver=auto -> randomized), median of {RUNS}"
        );
        if ratio > RATIO_CEILING {
            over.push(format!(
                "{label} {ratio:.3} (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms)"
            ));
        }
    }
    assert!(
        over.is_empty(),
        "FALSIFY-BEAT-SKLEARN-PCA-SPEED: ratio > {RATIO_CEILING:.2}: {over:?}"
    );
}
