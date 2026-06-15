//! BEAT-SKLEARN-GMM-SPEED — Pillar-1 speed cascade (PMAT-731). **NIGHTLY ONLY.**
//! cargo test -p aprender-core --release --test beat_sklearn_gmm_speed -- --ignored --nocapture
//! Compute-bound (per-component diagonal-Gaussian responsibilities over EM iters, no LAPACK).
#![cfg(test)]

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aprender::cluster::{CovarianceType, GaussianMixture};
use aprender::datasets::make_classification;
use aprender::prelude::*;

const N_SAMPLES: usize = 20_000;
const N_FEATURES: usize = 10;
const K: usize = 5;
const MAX_ITER: usize = 100;
const SEED: u64 = 42;
const RUNS: usize = 5;
// Gate at 0.70 (apr >= 1.43x faster) for cross-host robustness — measured 0.25 (dev box) / 0.53
// (aarch64); the Intel CI runner (MKL numpy) can be relatively faster, so leave margin (PMAT-733).
const RATIO_CEILING: f64 = 0.70;

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
        let mut m = GaussianMixture::new(K, CovarianceType::Diag)
            .with_max_iter(MAX_ITER)
            .with_random_state(SEED);
        m.fit(x).expect("warmup fit");
        let _ = m.predict(x);
    }
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut m = GaussianMixture::new(K, CovarianceType::Diag)
            .with_max_iter(MAX_ITER)
            .with_random_state(SEED);
        m.fit(x).expect("fit");
        let _p = m.predict(x);
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
        writeln!(f, "{line}").expect("row");
    }
    f.flush().expect("flush");
    f
}

fn time_sklearn(csv: &std::path::Path) -> f64 {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.mixture import GaussianMixture
X = np.loadtxt(r"{csv}", delimiter=",").astype(np.float64)
ts = []
def run():
    m = GaussianMixture(n_components={k}, covariance_type="diag", max_iter={mi},
                        tol=1e-3, n_init=1, random_state={seed})
    m.fit(X); _ = m.predict(X)
run()  # warmup
for _ in range({runs}):
    t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1000.0)
ts.sort()
print("SKLEARN_MS=%f" % ts[len(ts)//2])
"#,
        csv = csv.display(),
        k = K,
        mi = MAX_ITER,
        seed = SEED,
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
fn beat_sklearn_gmm_speed() {
    let (x, _y) = make_classification(N_SAMPLES, N_FEATURES, N_FEATURES, K, SEED);
    let apr_ms = time_apr(&x);
    let csv = write_csv(&x);
    let sklearn_ms = time_sklearn(csv.path());
    let ratio = apr_ms / sklearn_ms;
    eprintln!(
        "BEAT-SKLEARN-GMM-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms \
         ratio={ratio:.3} (apr {:.2}x faster) on {N_SAMPLES}x{N_FEATURES} k={K} max_iter={MAX_ITER}, median of {RUNS}",
        sklearn_ms / apr_ms
    );
    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-SKLEARN-GMM-SPEED: ratio {ratio:.3} > {RATIO_CEILING:.2} (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms)"
    );
}
