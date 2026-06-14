//! BEAT-SKLEARN-GAUSSIANNB-SPEED — Pillar-1 speed cascade (PMAT-723, Day-7). **NIGHTLY ONLY.**
//!
//! `#[ignore]`d: needs `uv` + scikit-learn at runtime. Run by
//! `.github/workflows/beat-speed-nightly.yml` and locally where `uv` exists:
//!
//! ```text
//! cargo test -p aprender-core --release --test beat_sklearn_gaussiannb_speed -- --ignored --nocapture
//! ```
//!
//! Same methodology as beat_sklearn_linreg_speed: time apr AND scikit-learn GaussianNB fit+predict
//! on the SAME data, SAME host, SAME run, gate the RATIO `apr_ms / sklearn_ms`. GaussianNB is pure
//! O(n·d·classes) elementwise arithmetic (per-class mean/var, Gaussian log-likelihood) with NO
//! LAPACK/BLAS — apr's SIMD turf vs numpy's per-op overhead. Still MEASURED, not assumed.

#![cfg(test)]

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aprender::classification::GaussianNB;
use aprender::datasets::make_classification;
use aprender::prelude::*;

const N_SAMPLES: usize = 50_000;
const N_FEATURES: usize = 30;
const N_INFORMATIVE: usize = 20;
const N_CLASSES: usize = 8;
const SEED: u64 = 42;
const RUNS: usize = 5;
/// apr must be at least this fast (ratio = apr/sklearn). Measured ~0.20 (apr ~4.9x faster on a
/// 16-core box after hoisting the per-class `ln` constant out of the predict hot loop); gate at
/// 0.50 (apr >= 2x faster) for large margin against nightly-host variance.
const RATIO_CEILING: f64 = 0.50;

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

fn time_apr(x: &aprender::Matrix<f32>, y: &[usize]) -> f64 {
    {
        let mut m = GaussianNB::new();
        m.fit(x, y).expect("apr warmup fit");
        let _ = m.predict(x).expect("apr warmup predict");
    }
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut m = GaussianNB::new();
        m.fit(x, y).expect("apr fit");
        let _p = m.predict(x).expect("apr predict");
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    median(&times)
}

/// Writes features + label-as-last-column to CSV for the sklearn subprocess.
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
        writeln!(f, "{line}").expect("write csv row");
    }
    f.flush().expect("flush csv");
    f
}

fn time_sklearn(csv: &std::path::Path) -> f64 {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.naive_bayes import GaussianNB
D = np.loadtxt(r"{csv}", delimiter=",")
X, y = D[:, :-1], D[:, -1].astype(np.int64)
ts = []
m = GaussianNB(); m.fit(X, y); _ = m.predict(X)  # warmup
for _ in range({runs}):
    t = time.perf_counter()
    m = GaussianNB(); m.fit(X, y); _ = m.predict(X)
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
        .expect("run uv (is `uv` installed? this test is nightly-only)");
    assert!(
        out.status.success(),
        "sklearn timing failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("SKLEARN_MS="))
        .unwrap_or_else(|| panic!("no SKLEARN_MS in: {stdout}"));
    line.trim().parse::<f64>().expect("parse sklearn ms")
}

#[test]
#[ignore = "nightly-only: needs uv + scikit-learn (beat-speed-nightly.yml)"]
fn beat_sklearn_gaussiannb_speed() {
    let (x, y) = make_classification(N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_CLASSES, SEED);

    let apr_ms = time_apr(&x, &y);
    let csv = write_csv(&x, &y);
    let sklearn_ms = time_sklearn(csv.path());

    let ratio = apr_ms / sklearn_ms;
    let speedup = sklearn_ms / apr_ms;
    eprintln!(
        "BEAT-SKLEARN-GAUSSIANNB-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms \
         ratio={ratio:.3} (apr {speedup:.2}x faster) on {N_SAMPLES}x{N_FEATURES} \
         classes={N_CLASSES}, median of {RUNS}"
    );

    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-SKLEARN-GAUSSIANNB-SPEED: apr/sklearn ratio {ratio:.3} > {RATIO_CEILING:.2} \
         — apr GaussianNB is not faster than scikit-learn (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms)"
    );
}
