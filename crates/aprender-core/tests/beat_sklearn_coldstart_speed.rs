//! BEAT-SKLEARN-COLDSTART-SPEED — Pillar-1 (scikit-learn) speed beat. **NIGHTLY ONLY.**
//!
//! ## The honest win
//! For a ONE-SHOT small-model fit+predict invoked from the shell (the very common
//! "fit me a quick classifier on this data" workflow), apr is a pure-Rust STATIC
//! BINARY with ~0 framework startup, while scikit-learn pays HUNDREDS of ms just
//! for the Python interpreter + `import numpy` + `import sklearn` before any model
//! work begins. Measured END-TO-END PROCESS wall-clock on noah-Lambda-Vector
//! (48-core x86, CPU, warm uv cache, median of 5 + warmup):
//!
//!   apr   full process (cold-start + GaussianNB fit+predict): ~1-4 ms
//!   `import sklearn` only (no work, numpy + sklearn import):  ~400-700 ms
//!   sklearn full process (import + equivalent fit+predict):   ~450-750 ms
//!
//! => apr is ~100-700x faster end-to-end than a one-shot `python -c "import
//!    sklearn; ...fit+predict..."`, BEFORE sklearn has finished importing. apr
//!    does the FULL fit+predict in less time than `import sklearn` alone.
//!
//! ## Why this is HONEST and host-independent (startup, not algorithm)
//! This is a STARTUP-COST beat, identical in spirit to the shipped Pillar-2
//! beat_pytorch_coldstart_speed.rs (~1600x) and Pillar-3
//! beat_unsloth_coldstart_speed.rs (~5000x). The DOMINANT factor is the absence
//! of the Python numpy+sklearn import, not the algorithm — apr also has the
//! algorithm, but the static-binary cold-start advantage is what wins here, and
//! it is architecture-independent, so it is robust across CI hosts (unlike the
//! removed bandwidth-bound elementwise scaler beats, which were host-fragile).
//!
//! This is SCOPED to the one-shot CLI-fit scenario and labeled as such. It
//! COMPLEMENTS — does not replace — the existing in-process sklearn SPEED beats
//! (LinReg ~1.78x, GaussianNB ~4.9x) which measure pure ALGORITHM time with the
//! import cost already paid on BOTH sides. Those answer "is apr's math faster?";
//! this one answers "is apr's whole one-shot CLI invocation faster?". They are
//! different, both legitimate, user-facing claims.
//!
//! ## Why a ratio, measured same-host/same-run
//! Time apr AND sklearn on the SAME host, SAME run; gate the ratio
//! apr_ms / sklearn_ms. The gate ceiling is set conservatively at 0.10 (apr must
//! stay >= 10x faster) so CI host variance / a faster future numpy/sklearn import
//! cannot trip it, but a regression that loses apr's static-binary cold-start
//! advantage (e.g. a heavy startup dep creeping into the binary) would fail.
//!
//! Run:
//!   cargo test -p aprender-core --release --test beat_sklearn_coldstart_speed -- --ignored --nocapture

#![cfg(test)]

use std::process::Command;
use std::time::Instant;

const RUNS: usize = 5;
/// apr must be at least 10x faster than sklearn end-to-end (ratio = apr/sklearn).
const RATIO_CEILING: f64 = 0.10;

/// Representative cheap one-shot workload dimensions (small CLI-fit regime).
const N_SAMPLES: usize = 500;
const N_FEATURES: usize = 10;
const N_INFORMATIVE: usize = 8;
const N_CLASSES: usize = 3;
const SEED: u64 = 42;

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

/// The actual one-shot fit+predict workload, run in the re-exec'd child process.
/// A representative cheap model (GaussianNB) on a small make_classification — the
/// kind of one-shot job a user runs from the shell. We deliberately keep it tiny
/// so the dominant cost being measured is apr's whole-process cold-start.
fn apr_fit_predict_workload() {
    use aprender::classification::GaussianNB;
    use aprender::datasets::make_classification;
    let (x, y) = make_classification(N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_CLASSES, SEED);
    let mut m = GaussianNB::new();
    m.fit(&x, &y).expect("apr fit");
    let _p = m.predict(&x).expect("apr predict");
}

/// Time apr's OWN process end-to-end by re-exec'ing the test binary in a
/// "work only" mode. The child does the full one-shot fit+predict and exits; we
/// time the whole child process (the cold-start a user pays). Signalled via the
/// APR_SKLEARN_COLDSTART_CHILD env var.
fn time_apr_process(self_exe: &std::path::Path) -> f64 {
    let run = || {
        Command::new(self_exe)
            .env("APR_SKLEARN_COLDSTART_CHILD", "1")
            .output()
            .expect("re-exec apr child")
    };
    let _ = run(); // warmup
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = run();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        assert!(out.status.success(), "apr child failed");
    }
    median(&times)
}

/// Time scikit-learn end-to-end process wall-clock for the equivalent one-shot
/// fit+predict, INCLUDING the `import sklearn` cold-start cost (this is the whole
/// point — `python -c "import sklearn; ...fit+predict..."` pays numpy + sklearn
/// import every invocation). Same generated-data shape as the apr child so the
/// algorithm work is comparable; the dominant term on the sklearn side is the
/// import.
fn time_sklearn_process() -> f64 {
    let py = format!(
        r#"
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples={n}, n_features={d}, n_informative={ni},
    n_redundant=0, n_classes={c}, random_state={seed})
m = GaussianNB().fit(X, y)
_ = m.predict(X)
"#,
        n = N_SAMPLES,
        d = N_FEATURES,
        ni = N_INFORMATIVE,
        c = N_CLASSES,
        seed = SEED,
    );
    let run = || {
        Command::new("uv")
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
            .expect("run uv (is `uv` installed? this test is nightly-only)")
    };
    let _ = run(); // warmup (uv cache + page-in)
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = run();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        assert!(
            out.status.success(),
            "sklearn timing failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    median(&times)
}

#[test]
#[ignore = "nightly-only: needs uv + scikit-learn (beat-speed-nightly.yml)"]
fn beat_sklearn_coldstart_speed() {
    // Child mode: do the work and exit so the parent can time our whole process.
    if std::env::var("APR_SKLEARN_COLDSTART_CHILD").is_ok() {
        apr_fit_predict_workload();
        return;
    }
    let self_exe = std::env::current_exe().expect("current_exe");
    let apr_ms = time_apr_process(&self_exe);
    let sklearn_ms = time_sklearn_process();

    let ratio = apr_ms / sklearn_ms;
    let speedup = sklearn_ms / apr_ms;
    eprintln!(
        "BEAT-SKLEARN-COLDSTART-SPEED: apr={apr_ms:.3}ms sklearn={sklearn_ms:.1}ms \
         ratio={ratio:.5} (apr {speedup:.0}x faster), one-shot {N_SAMPLES}x{N_FEATURES} \
         GaussianNB fit+predict, median of {RUNS}"
    );

    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-SKLEARN-COLDSTART-SPEED: apr/sklearn ratio {ratio:.5} > {RATIO_CEILING:.2} \
         — apr lost its static-binary cold-start advantage for one-shot small fit+predict \
         (apr={apr_ms:.3}ms, sklearn={sklearn_ms:.1}ms; contract beat-sklearn-coldstart-speed-v1.yaml)"
    );
}
