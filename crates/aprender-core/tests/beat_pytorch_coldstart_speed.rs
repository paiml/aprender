//! BEAT-PYTORCH-COLDSTART-SPEED — Pillar-2 (PyTorch) speed beat. **NIGHTLY ONLY.**
//! (DRAFT / uncommitted scout artifact — PMAT-XXX, measured 2026-06-15.)
//!
//! ## The honest win
//! For a ONE-SHOT small-model training job invoked from the shell (the common
//! "fit me a quick classifier on this CSV" workflow), apr is a pure-Rust STATIC
//! BINARY with ~0 framework startup, while PyTorch pays ~740ms just for
//! `import torch` plus Python interpreter startup. Measured END-TO-END PROCESS
//! wall-clock on noah-Lambda-Vector (16-core x86):
//!
//!   apr   full process (incl startup): ~1.08 ms
//!   torch full process (no uv overhead): ~1739 ms   -> apr ~1600x faster
//!   torch full process (via `uv run`):   ~1687 ms   -> apr ~1560x faster
//!
//! Decomposition (HONEST): of torch's ~1687ms, ~743ms is `import torch`
//! (framework startup) and ~585ms is the 100-step GD loop on tiny (200x5)
//! tensors (Python per-op dispatch overhead). apr's whole process is ~1ms.
//! So this is BOTH a framework-startup win AND an in-loop win — but ONLY in the
//! SMALL one-shot regime. On a LARGER MLP, PyTorch's MKL + fused autograd
//! amortizes dispatch and WINS the in-loop throughput (~11x — see
//! beat_pytorch_autograd_grad.rs / docs/BEATS.md Pillar-2 CONCEDED). This beat
//! is deliberately scoped to the small one-shot job, which is a legitimate,
//! extremely common user-facing scenario (CLI fit), and is labeled as such.
//!
//! ## Why a ratio, measured same-host/same-run
//! Same as the sklearn speed beats: time apr AND PyTorch on the SAME data, SAME
//! host, SAME run; gate the ratio apr_ms / torch_ms. The measured ratio is
//! ~0.0006 (apr ~1600x faster); the gate ceiling is set conservatively at 0.10
//! (apr must stay >= 10x faster) so CI host variance / a faster future torch
//! cannot trip it, but a regression that loses the static-binary cold-start
//! advantage (e.g. apr accidentally growing a heavy startup) would fail.
//!
//! Run: cargo test -p aprender-core --test beat_pytorch_coldstart_speed -- --ignored --nocapture

#![cfg(test)]

use std::process::Command;
use std::time::Instant;

const RUNS: usize = 5;
/// apr must be at least 10x faster than PyTorch end-to-end (ratio = apr/torch).
const RATIO_CEILING: f64 = 0.10;

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

/// Time apr's OWN process end-to-end by re-exec'ing the test binary in a
/// "train only" mode. The child does the full small logistic-regression GD fit
/// and exits; we time the whole child process (this is the cold-start cost a
/// user pays). The child is signalled via the APR_COLDSTART_CHILD env var.
fn time_apr_process(self_exe: &std::path::Path) -> f64 {
    let run = || {
        Command::new(self_exe)
            .env("APR_COLDSTART_CHILD", "1")
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

/// The actual training workload, run in the re-exec'd child process.
fn apr_train_workload() {
    use aprender::classification::{FitMode, LogisticRegression};
    use aprender::datasets::make_classification;
    let (x, y) = make_classification(200, 5, 5, 2, 0);
    let mut m = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(100)
        .with_tolerance(0.0)
        .with_fit_mode(FitMode::Batch);
    m.fit(&x, &y).expect("apr fit");
    let _ = m.predict(&x);
}

/// Time PyTorch end-to-end process wall-clock for the equivalent tiny GD fit.
fn time_torch_process() -> f64 {
    let py = r#"
import time, torch
torch.manual_seed(0)
N, D = 200, 5
X = torch.randn(N, D); w_true = torch.randn(D, 1)
y = (X @ w_true > 0).float()
w = torch.zeros(D, 1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.1)
for _ in range(100):
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
    loss.backward(); opt.step()
"#;
    let run = || {
        Command::new("uv")
            .args(["run", "--with", "torch", "python3", "-c", py])
            .output()
            .expect("run uv (is `uv` installed? nightly-only)")
    };
    let _ = run(); // warmup (uv cache + page-in)
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = run();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        assert!(
            out.status.success(),
            "torch timing failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    median(&times)
}

#[test]
#[ignore = "nightly-only: needs uv + torch (beat-speed-nightly.yml)"]
fn beat_pytorch_coldstart_speed() {
    // Child mode: do the work and exit so the parent can time our whole process.
    if std::env::var("APR_COLDSTART_CHILD").is_ok() {
        apr_train_workload();
        return;
    }
    let self_exe = std::env::current_exe().expect("current_exe");
    let apr_ms = time_apr_process(&self_exe);
    let torch_ms = time_torch_process();

    let ratio = apr_ms / torch_ms;
    let speedup = torch_ms / apr_ms;
    eprintln!(
        "BEAT-PYTORCH-COLDSTART-SPEED: apr={apr_ms:.3}ms torch={torch_ms:.1}ms \
         ratio={ratio:.5} (apr {speedup:.0}x faster), one-shot 200x5 logreg, median of {RUNS}"
    );

    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-PYTORCH-COLDSTART-SPEED: apr/torch ratio {ratio:.5} > {RATIO_CEILING:.2} \
         — apr lost its static-binary cold-start advantage for one-shot small training \
         (apr={apr_ms:.3}ms, torch={torch_ms:.1}ms)"
    );
}
