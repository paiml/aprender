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
/// apr must be at least this fast (ratio = apr/sklearn).
///
/// MEASURED, PER HOST (2026-07-28 - the earlier "~0.20 / apr ~4.9x faster on a 16-core box"
/// pinned here was a DEV-BOX number that has never been observed on the CI host, and it made
/// the gate look like it had 2.5x of headroom when it has ~20%):
///
///   lambda-vector (modern 48-thread box, moderate load):
///       apr ~12.0ms  sklearn ~56ms   ratio 0.188 / 0.237 / 0.220   (apr 4.2-5.3x)
///   mac-server = the canonical nightly host (Xeon W-3245, 32 threads, IDLE):
///       apr 27.961ms sklearn 70.308ms ratio 0.398                  (apr 2.51x)
///   mac-server under real nightly load (historical, from run logs):
///       ratio 0.278 .. 0.601 - i.e. it STRADDLES this gate.
///
/// Why the host matters so much: apr's GaussianNB is a scalar Rust loop, while sklearn's is
/// vectorised numpy. The Xeon W-3245 has AVX-512 (avx512f/bw/cd/dq/vl), so numpy closes much
/// of the gap there: going from lambda to mac-server costs apr 2.3x but sklearn only 1.25x.
/// That is why the "4.9x" never reproduced - it is not decay, it is a different machine.
///
/// The gate stays at 0.50 (apr >= 2x). It is NOT weakened to accommodate the breaches: the
/// idle-host reading (0.398) clears it, so a run over 0.50 is real signal - either genuine
/// contention on a 17-runner box or a genuine regression, and the fit/predict split below
/// exists to tell those two apart. See PMAT-GNB-SPEED-DEFENSE-001. The durable fix that would
/// restore real margin is SIMD for the predict hot loop (trueno), not a looser threshold.
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

/// Returns `(total_ms, fit_ms, predict_ms)`, each a median over `RUNS`.
///
/// The phase split is diagnostic, not decorative. When this beat goes red the question is
/// always "contention or regression?", and the two look different: host contention inflates
/// fit and predict roughly together, while an algorithmic regression lands in one phase. With
/// only a fused total (the previous shape) five red nights told you nothing you could act on.
fn time_apr(x: &aprender::Matrix<f32>, y: &[usize]) -> (f64, f64, f64) {
    {
        let mut m = GaussianNB::new();
        m.fit(x, y).expect("apr warmup fit");
        let _ = m.predict(x).expect("apr warmup predict");
    }
    let mut totals = Vec::with_capacity(RUNS);
    let mut fits = Vec::with_capacity(RUNS);
    let mut predicts = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut m = GaussianNB::new();
        m.fit(x, y).expect("apr fit");
        let fit_done = t.elapsed().as_secs_f64() * 1000.0;
        let _p = m.predict(x).expect("apr predict");
        let total = t.elapsed().as_secs_f64() * 1000.0;
        totals.push(total);
        fits.push(fit_done);
        predicts.push(total - fit_done);
    }
    (median(&totals), median(&fits), median(&predicts))
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

/// Returns `(total_ms, fit_ms, predict_ms)`, each a median over `RUNS`. See `time_apr` for why
/// the phase split exists.
fn time_sklearn(csv: &std::path::Path) -> (f64, f64, f64) {
    let py = format!(
        r#"
import time, numpy as np
from sklearn.naive_bayes import GaussianNB
D = np.loadtxt(r"{csv}", delimiter=",")
X, y = D[:, :-1], D[:, -1].astype(np.int64)
ts = []
fs = []
ps = []
m = GaussianNB(); m.fit(X, y); _ = m.predict(X)  # warmup
for _ in range({runs}):
    t = time.perf_counter()
    m = GaussianNB(); m.fit(X, y)
    f = time.perf_counter()
    _ = m.predict(X)
    e = time.perf_counter()
    fs.append((f - t) * 1000.0)
    ps.append((e - f) * 1000.0)
    ts.append((e - t) * 1000.0)
def med(v):
    v = sorted(v)
    return v[len(v)//2]
print("SKLEARN_MS=%f" % med(ts))
print("SKLEARN_FIT_MS=%f" % med(fs))
print("SKLEARN_PREDICT_MS=%f" % med(ps))
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
    let field = |key: &str| -> f64 {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or_else(|| panic!("no {key} in: {stdout}"))
            .trim()
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("parse {key}: {e}"))
    };
    (
        field("SKLEARN_MS="),
        field("SKLEARN_FIT_MS="),
        field("SKLEARN_PREDICT_MS="),
    )
}

#[test]
#[ignore = "nightly-only: needs uv + scikit-learn (beat-speed-nightly.yml)"]
fn beat_sklearn_gaussiannb_speed() {
    let (x, y) = make_classification(N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_CLASSES, SEED);

    let (apr_ms, apr_fit_ms, apr_pred_ms) = time_apr(&x, &y);
    let csv = write_csv(&x, &y);
    let (sklearn_ms, skl_fit_ms, skl_pred_ms) = time_sklearn(csv.path());

    let ratio = apr_ms / sklearn_ms;
    let speedup = sklearn_ms / apr_ms;
    let fit_ratio = apr_fit_ms / skl_fit_ms;
    let pred_ratio = apr_pred_ms / skl_pred_ms;
    eprintln!(
        "BEAT-SKLEARN-GAUSSIANNB-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.3}ms \
         ratio={ratio:.3} (apr {speedup:.2}x faster) on {N_SAMPLES}x{N_FEATURES} \
         classes={N_CLASSES}, median of {RUNS}"
    );
    // Phase split: which half moved? Contention drags fit and predict together; an
    // algorithmic regression shows up in one of them. Printed on every run, pass or
    // fail, so a sequence of nightlies is directly comparable.
    eprintln!(
        "BEAT-SKLEARN-GAUSSIANNB-SPEED-PHASES: fit apr={apr_fit_ms:.3}ms skl={skl_fit_ms:.3}ms \
         ratio={fit_ratio:.3} | predict apr={apr_pred_ms:.3}ms skl={skl_pred_ms:.3}ms \
         ratio={pred_ratio:.3}"
    );

    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-SKLEARN-GAUSSIANNB-SPEED: apr/sklearn ratio {ratio:.3} > {RATIO_CEILING:.2} \
         — apr GaussianNB is not faster than scikit-learn (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.3}ms). \
         Phase split: fit {fit_ratio:.3}, predict {pred_ratio:.3}. Both phases up together on the \
         nightly host usually means contention (mac-server runs 17 runners on 32 threads; idle \
         reference is ratio 0.398 / apr 27.961ms); one phase up alone means a real regression."
    );
}
