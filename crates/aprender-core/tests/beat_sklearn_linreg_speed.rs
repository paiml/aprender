//! BEAT-SKLEARN-LINREG-SPEED — Pillar-1 speed beat (PMAT-722). **NIGHTLY ONLY.**
//!
//! `#[ignore]`d because it needs `uv` + scikit-learn at runtime, which the normal
//! per-PR CI image does not have. It is run by
//! `.github/workflows/beat-speed-nightly.yml` (and locally where `uv` exists):
//!
//! ```text
//! cargo test -p aprender-core --test beat_sklearn_linreg_speed -- --ignored --nocapture
//! ```
//!
//! ## Why a ratio, and why the workload is large
//! This beat times apr AND scikit-learn `LinearRegression` fit+predict on the
//! SAME generated data, on the SAME host, in the SAME run, and gates the
//! **ratio** `apr_ms / sklearn_ms`.
//!
//! This file used to claim that "a relative comparison cancels machine-speed
//! variance, so a slow runner slows both sides proportionally and the ratio
//! holds". **That is false, and the nightly falsified it.** Two runs an hour
//! apart on the same host at 10_000x20:
//!
//! ```text
//!   apr=2.289ms sklearn=9.770ms ratio=0.234   PASS
//!   apr=5.798ms sklearn=5.556ms ratio=1.044   FAIL   (ceiling 0.90)
//! ```
//!
//! The two sides did not move proportionally - they moved in OPPOSITE
//! directions. At 10_000x20 apr's fit+predict is ~1.2ms, so one scheduler
//! preemption (~1ms) is an ~80% perturbation while sklearn's ~4.5ms absorbs it.
//! The ratio was measuring the host, not the algorithms.
//!
//! The workload is therefore 200_000x50 (~50x the work): apr ~120ms, sklearn
//! ~220ms, where millisecond jitter is ~1% rather than ~80%. Measured on
//! lambda-vector (Threadripper 7960X):
//!
//! ```text
//!   10_000x20   idle 0.255-0.339 | cpu 0.128-0.189 | mem 0.253-0.362 | CI 0.234..1.044
//!   200_000x50  idle 0.494-0.582 | cpu 0.593-0.636
//! ```
//!
//! The large workload spans 0.49-0.64 across every locally inducible
//! condition; the small one spanned 8x on CI.
//!
//! VALIDATED ON THE HOST THAT FAILED. Dispatched twice onto the clean-room
//! pool at 200_000x50:
//!
//! ```text
//!   apr=265.473ms sklearn=337.106ms ratio=0.788
//!   apr=252.463ms sklearn=313.446ms ratio=0.805
//! ```
//!
//! 2.2% spread, where the 10_000x20 configuration swung 0.234 -> 1.044 (346%)
//! on that same pool. The stability goal is met.
//!
//! But read the LEVEL, not just the spread: on that host apr is ~1.25x faster,
//! NOT the ~1.85x measured on lambda-vector, leaving only ~11% headroom under
//! the 0.90 ceiling. apr is penalised more than sklearn by that host (2.3x vs
//! 1.55x slower than lambda-vector). Do NOT "fix" a future failure here by
//! enlarging the workload again - apr's advantage SHRINKS with size
//! (0.26 at 10_000x20 -> 0.54 at 200_000x50 locally), so a bigger problem
//! makes the ratio worse, not better.
//!
//! OPEN, not resolved here: the contract records baseline_floor 0.56, which
//! describes neither this host (0.79-0.81) nor the old size (0.26). Whether
//! that gap is slower CI hardware or a real apr regression cannot be settled
//! without historical CI data at this size, and is deliberately not asserted
//! either way.
//!
//! TWO HONEST CAVEATS. (1) The headline win SHRINKS with size: apr is ~3.6x
//! faster at 10_000x20 but ~1.7x at 200_000x50. The smaller number is the one
//! that can be measured reliably, and a gate that measures reliably is worth
//! more than a gate that flatters. (2) The CI failure mode - apr itself slowing
//! 2.5x - could NOT be reproduced locally under either CPU or memory-bandwidth
//! pressure, so this rests on the stability comparison above, not on a
//! reproduction of the exact mechanism.
//!
//! The
//! gate (`contracts/beat-sklearn-linreg-speed-v1.yaml`, beat_threshold 0.90)
//! requires apr to stay ≥ ~1.11× faster — a large margin below the measured
//! ~1.78× (commit 34d61a608), so CI noise cannot trip it but a real regression
//! (apr losing its speed advantage) fails the gate.

#![cfg(test)]

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aprender::datasets::make_regression;
use aprender::prelude::*;

const N_SAMPLES: usize = 200_000;
const N_FEATURES: usize = 50;
const SEED: u64 = 42;
const RUNS: usize = 9;
/// apr must be at least this much faster than sklearn (ratio = apr/sklearn).
/// Matches contracts/beat-sklearn-linreg-speed-v1.yaml beat_threshold.
const RATIO_CEILING: f64 = 0.90;

/// Median of a slice of f64 (sorts a copy).
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

/// Time apr LinearRegression fit+predict, median wall-clock ms over RUNS (+warmup).
fn time_apr(x: &aprender::Matrix<f32>, y: &aprender::Vector<f32>) -> f64 {
    // Warmup (page-in, branch predictor, allocator).
    {
        let mut m = LinearRegression::new();
        m.fit(x, y).expect("apr warmup fit");
        let _ = m.predict(x);
    }
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut m = LinearRegression::new();
        m.fit(x, y).expect("apr fit");
        let _p = m.predict(x);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    median(&times)
}

/// Write the dataset to a CSV (features..., target) for the sklearn side.
fn write_csv(x: &aprender::Matrix<f32>, y: &aprender::Vector<f32>) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("tempfile");
    let ys = y.as_slice();
    for i in 0..x.n_rows() {
        let mut line = String::new();
        for j in 0..x.n_cols() {
            line.push_str(&x.get(i, j).to_string());
            line.push(',');
        }
        line.push_str(&ys[i].to_string());
        writeln!(f, "{line}").expect("write csv row");
    }
    f.flush().expect("flush csv");
    f
}

/// Time scikit-learn LinearRegression on the same CSV via uv; returns median ms.
fn time_sklearn(csv: &std::path::Path) -> f64 {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.linear_model import LinearRegression
d = np.loadtxt(r"{csv}", delimiter=",")
X, y = d[:, :-1], d[:, -1]
ts = []
m = LinearRegression().fit(X, y); _ = m.predict(X)  # warmup
for _ in range({runs}):
    t = time.perf_counter()
    m = LinearRegression().fit(X, y); _ = m.predict(X)
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
fn beat_sklearn_linreg_speed() {
    let (x, y) = make_regression(N_SAMPLES, N_FEATURES, 0.1, SEED);

    let apr_ms = time_apr(&x, &y);
    let csv = write_csv(&x, &y);
    let sklearn_ms = time_sklearn(csv.path());

    let ratio = apr_ms / sklearn_ms;
    let speedup = sklearn_ms / apr_ms;
    eprintln!(
        "BEAT-SKLEARN-LINREG-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms \
         ratio={ratio:.3} (apr {speedup:.2}x faster) on {N_SAMPLES}x{N_FEATURES}, median of {RUNS}"
    );

    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-SKLEARN-LINREG-SPEED: apr/sklearn ratio {ratio:.3} > {RATIO_CEILING:.2} \
         — apr LinearRegression is no longer comfortably faster than scikit-learn \
         (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms; contract beat-sklearn-linreg-speed-v1.yaml)"
    );
}
