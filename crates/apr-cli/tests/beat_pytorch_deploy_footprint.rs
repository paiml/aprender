//! BEAT-PYTORCH-DEPLOY-FOOTPRINT — Pillar-2 (replace+beat PyTorch) DEPLOY-SIZE beat.
//! **PER-PR CPU test** (no network, no `uv`, no torch install at test time).
//!
//! ## The honest win
//! For the INFERENCE-DEPLOYMENT scenario (ship a model to an edge box / container /
//! serverless function and serve it), the *framework runtime* you must put on disk
//! alongside the (identical, separate) model weights is:
//!
//!   * **apr**: a single self-contained pure-Rust STATIC binary. It links only the
//!     host's own libc/libm/libgcc (present on every Linux box) — NO Python, NO
//!     framework runtime, NO native ML libs shipped. The whole inference runtime IS
//!     the binary.
//!   * **PyTorch / HF transformers**: the `torch` wheel (CPU ~700 MB, dominated by
//!     `libtorch_cpu.so` ~420 MB) + `transformers` + their transitive deps
//!     (numpy, sympy, tokenizers, hf-xet, …) + a Python interpreter to run it all.
//!
//! ### Measured (noah-Lambda-Vector, x86-64 Linux, 2026-06-15)
//! Fresh `uv venv` (CPython 3.11.10) + `uv pip install torch --index-url
//! https://download.pytorch.org/whl/cpu` + `uv pip install transformers`, then
//! `du -sb` (see `contracts/beat-pytorch-deploy-footprint-v1.yaml`):
//!
//! | Component                                            | Size        |
//! |-----------------------------------------------------|-------------|
//! | `torch` (CPU 2.12.0+cpu; libtorch_cpu.so = 422 MB)  | 698 MiB     |
//! | `transformers` (5.12.0)                             |  51 MiB     |
//! | numpy + sympy + tokenizers + hf-xet + other deps    | ~104 MiB    |
//! | **site-packages total**                             | **853 MiB** |
//! | + CPython 3.11 interpreter (required to run torch)  |  67 MiB     |
//! | **full CPU inference deploy**                        | **921 MiB** |
//!
//! A CUDA torch wheel is **2.5–3.5 GB** (bundled cuDNN/cuBLAS/NCCL), so the CPU
//! figure above is the *conservative, apr-favorable-but-honest* incumbent.
//!
//! ### apr (this binary), release, same host/day
//!   * as-built:  56,532,392 B  ≈ **53.9 MiB**
//!   * stripped:  47,105,376 B  ≈ **44.9 MiB**
//!
//! ### Ratio (PyTorch deploy bytes / apr binary bytes)
//!   * site-packages (853 MiB) / apr as-built (53.9 MiB) ≈ **15.8×**
//!   * site-packages (853 MiB) / apr stripped (44.9 MiB) ≈ **19.0×**
//!   * full CPU deploy (921 MiB) / apr as-built          ≈ **17.1×**
//!   * full CPU deploy (921 MiB) / apr stripped          ≈ **20.5×**
//!   * (vs a CUDA torch deploy the ratio is **~50×+**.)
//!
//! apr WINS the inference-deployment footprint by 15–20× on CPU (50×+ on CUDA),
//! host-independent. This is the deploy-size analog of the cold-start beats.
//!
//! ## Honest scope label — DEPLOY-SIZE, training throughput CONCEDED
//! This is a DEPLOY-FOOTPRINT / static-binary win for the INFERENCE-deployment
//! scenario. apr CONCEDES training throughput (overhead-bound; see
//! `apr-pytorch-autograd-equivalence-beat-v1` for the correctness win where apr is
//! ~11× slower to *train*). The deploy-size advantage is architecture-independent.
//!
//! ## What this per-PR test gates
//! The apr side is MEASURED here (cargo builds the `apr` bin and hands us its exact
//! path via `CARGO_BIN_EXE_apr`); we assert the RELEASE binary stays below a
//! conservative ceiling AND that the deploy ratio holds, so a regression that bloats
//! the deploy artifact toward framework-runtime size fails the gate. The PyTorch
//! figure is a DOCUMENTED CONSTANT (measured above; cited in the contract) so the
//! test is fast, network-free, and needs no `uv`/torch at test time.
//!
//! ## Profile note — the deploy artifact is the RELEASE binary
//! The shipped inference runtime is the OPTIMIZED, stripped RELEASE binary
//! (~53.9 MiB). A DEBUG build is ~844 MiB (full debug symbols, no opt/strip) and is
//! NEVER deployed — so the gate only asserts the win for the release profile. Under
//! `cargo test` (debug) the test is INFORMATIONAL (prints the figures, no
//! assertion). The enforcing run is the release one named in the contract's
//! falsification command:
//!   `cargo test -p apr-cli --release --test beat_pytorch_deploy_footprint`
//! (this is how the `beat_pytorch_deploy_footprint` CI gate invokes it).

#![cfg(test)]

use std::path::Path;

/// The apr release binary must stay below this on-disk ceiling (bytes).
/// Measured release size 2026-06-15: 56,532,392 B (~53.9 MiB). Ceiling 150 MB
/// gives a large margin for normal growth while still being ~6× below the
/// conservative PyTorch *CPU* deploy figure (921 MiB) — any regression that
/// bloats apr toward framework-runtime size trips the gate.
const APR_BINARY_CEILING_BYTES: u64 = 150 * 1024 * 1024; // 150 MiB

/// Documented, MEASURED PyTorch CPU inference-deploy footprint (bytes), used to
/// compute and assert the deploy-footprint ratio. Sourced 2026-06-15 via fresh
/// `uv venv` + CPU `torch` wheel + `transformers` + transitive deps + CPython
/// (see the module docs and `contracts/beat-pytorch-deploy-footprint-v1.yaml`).
const PYTORCH_SITE_PACKAGES_BYTES: u64 = 894_938_262; // torch CPU + transformers + deps (~853 MiB)
const PYTORCH_FULL_DEPLOY_BYTES: u64 = 965_534_951; // + CPython interpreter (~921 MiB)

/// apr must win the deploy-footprint comparison by at least this factor. The
/// measured CPU ratio is ~15.8× (site-packages) / ~17.1× (full deploy); we gate a
/// conservative 5× so CI host / future-torch variance cannot trip it but losing
/// the static-binary advantage (apr ballooning past ~1/5 of the torch deploy)
/// fails. Matches `beat_threshold` in the contract.
const MIN_DEPLOY_FOOTPRINT_RATIO: f64 = 5.0;

fn apr_binary_path() -> &'static Path {
    // cargo builds the `apr` bin for this crate and exposes its exact path,
    // so we measure the real artifact at its real build location — no guessing
    // a target dir, robust to CARGO_TARGET_DIR redirection.
    Path::new(env!("CARGO_BIN_EXE_apr"))
}

#[test]
fn beat_pytorch_deploy_footprint() {
    let apr = apr_binary_path();
    let meta = std::fs::metadata(apr)
        .unwrap_or_else(|e| panic!("apr binary not found at {}: {e}", apr.display()));
    let apr_bytes = meta.len();
    assert!(apr_bytes > 0, "apr binary at {} is empty", apr.display());

    let ratio_site = PYTORCH_SITE_PACKAGES_BYTES as f64 / apr_bytes as f64;
    let ratio_full = PYTORCH_FULL_DEPLOY_BYTES as f64 / apr_bytes as f64;

    eprintln!(
        "BEAT-PYTORCH-DEPLOY-FOOTPRINT: apr={apr_bytes} B ({:.1} MiB) at {} | \
         pytorch site-packages={PYTORCH_SITE_PACKAGES_BYTES} B ({:.0} MiB), \
         full CPU deploy={PYTORCH_FULL_DEPLOY_BYTES} B ({:.0} MiB) | \
         ratio_site={ratio_site:.1}x ratio_full={ratio_full:.1}x (apr smaller; CUDA torch ~50x+) | \
         release={}",
        apr_bytes as f64 / 1_048_576.0,
        apr.display(),
        PYTORCH_SITE_PACKAGES_BYTES as f64 / 1_048_576.0,
        PYTORCH_FULL_DEPLOY_BYTES as f64 / 1_048_576.0,
        !cfg!(debug_assertions),
    );

    // The deploy artifact is the RELEASE binary. A DEBUG build is ~844 MiB (full
    // debug symbols, no opt/strip) and is NEVER shipped, so we only ASSERT the win
    // for the release profile; under `cargo test` (debug) this test is
    // informational (the figures are printed above for visibility).
    if cfg!(debug_assertions) {
        eprintln!(
            "BEAT-PYTORCH-DEPLOY-FOOTPRINT: debug build (~{:.0} MiB, not a deploy artifact) — \
             gate is INFORMATIONAL; run with --release to enforce. The shipped deploy artifact \
             is the optimized release binary (~53.9 MiB, ratio ~15.8x).",
            apr_bytes as f64 / 1_048_576.0,
        );
        return;
    }

    // ---- RELEASE profile: enforce the deploy-footprint win. ----

    // Headline claim: apr's deploy footprint is >= 5x smaller than the pinned
    // PyTorch/transformers inference site-packages.
    assert!(
        ratio_site >= MIN_DEPLOY_FOOTPRINT_RATIO,
        "FALSIFY-BEAT-PYTORCH-DEPLOY-FOOTPRINT: release apr deploy footprint ratio {ratio_site:.2}x \
         (pytorch site-packages {PYTORCH_SITE_PACKAGES_BYTES} B / apr {apr_bytes} B) \
         < required {MIN_DEPLOY_FOOTPRINT_RATIO:.1}x — apr lost its static-binary deploy-size \
         advantage over the PyTorch/transformers inference runtime."
    );

    // Absolute binary-size ceiling on the deploy artifact: a regression that bloats
    // apr toward framework-runtime size fails the gate.
    assert!(
        apr_bytes <= APR_BINARY_CEILING_BYTES,
        "FALSIFY-BEAT-PYTORCH-DEPLOY-FOOTPRINT: release apr binary {apr_bytes} B \
         ({:.1} MiB) exceeds ceiling {APR_BINARY_CEILING_BYTES} B ({:.0} MiB) — \
         the deploy artifact is bloating toward framework-runtime size. \
         Investigate what dependency added the weight; the PyTorch CPU deploy is \
         {PYTORCH_FULL_DEPLOY_BYTES} B (~921 MiB) and apr's whole pitch is to be \
         ~15-20x smaller.",
        apr_bytes as f64 / 1_048_576.0,
        APR_BINARY_CEILING_BYTES as f64 / 1_048_576.0,
    );
}
