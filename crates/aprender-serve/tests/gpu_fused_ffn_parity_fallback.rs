//! PMAT-798: GPU fused gate+up+SwiGLU FFN parity fallback.
//!
//! Root cause (localized via per-layer CPU-vs-GPU bisection on GB10 sm_121,
//! TinyLlama-1.1B-Chat Q4_K_M): the fused gate+up+SwiGLU HW DP4A Q4K kernel
//! quantizes RMSNorm activations to Q8_1 before the integer dot product. On
//! LLaMA-NORM-family models with massive activations (TinyLlama develops a
//! ~-138 outlier at hidden dim 624 starting in layer-2 FFN), that Q8 step
//! drops the first-token GPU-vs-CPU logit cosine to ~0.969 - below the 0.98
//! parity gate - so the model is wrongly pushed off the GPU. Disabling just
//! the fusion (force_high_precision_ffn) recovers cosine to ~0.99 while
//! keeping HW DP4A for the separate GEMVs.
//!
//! Run: cargo test --test gpu_fused_ffn_parity_fallback --features cuda -- --nocapture

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;

fn cuda_available() -> bool {
    CudaExecutor::is_available()
}

/// `force_high_precision_ffn` must disable the fused gate+up path and report
/// whether it changed anything. The second call must be a no-op (idempotent).
#[test]
fn force_high_precision_ffn_disables_fusion() {
    if !cuda_available() {
        eprintln!("Skipping: CUDA not available");
        return;
    }
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Skipping: CudaExecutor::new failed: {e:?}");
            return;
        },
    };

    // First call returns the *previous* fused flag. After the call fusion MUST
    // be off, so a second call MUST report no change.
    let _was = exec.force_high_precision_ffn();
    let still_changed = exec.force_high_precision_ffn();
    assert!(
        !still_changed,
        "force_high_precision_ffn must be idempotent: 2nd call should report no change"
    );
}
