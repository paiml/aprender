//! FALSIFY-Q6K-ACTIVATION-AMPLIFICATION-002 — does activation distribution
//! non-uniformity amplify the per-matvec CPU↔GPU divergence?
//!
//! ## Why this falsifier exists
//!
//! [FALSIFY-Q6K-FP-ACC-001](./falsify_q6k_fp_accumulator_order_001.rs)
//! (PR #1801) showed that on **uniform synthetic input** the per-matvec
//! divergence between CPU `fused_q6k_parallel_matvec` and GPU
//! `CudaExecutor::q6k_gemv` is ulp-scale (~1e-7). That's nowhere near
//! the 0.94-0.987 cosine drop cited for 7 layers in #1583.
//!
//! The PR #1801 docstring surfaced three remaining hypotheses for the
//! amplifier:
//!
//! 1. ~~Expert-routing differences~~ — **ruled out by code inspection**
//!    in this session: both CPU `moe_ffn_forward_layer` and CUDA
//!    `moe_ffn_forward_layer_cuda` run the SAME host-side Rust code
//!    for router logits / softmax / top-K selection (lines 412-451
//!    in `qwen3_moe_load.rs` and lines 89-119 in
//!    `gguf/cuda/moe_ffn_forward_layer_cuda.rs` respectively — two
//!    copies, byte-equivalent implementations).
//! 2. **Activation distribution non-uniformity** — this falsifier.
//! 3. Accumulator-chain length in the full forward — separate cascade
//!    follow-up; this file does NOT cover it.
//!
//! ## What this falsifier asserts
//!
//! Runs the same Q6_K matvec with activations drawn from progressively
//! more "bursty" distributions (uniform → log-normal → outlier-heavy).
//! Reports the per-rel_diff at each level so a future fix-PR can
//! verify that activation-amplification truly is the dominant amplifier
//! (or rule it out and pivot to hypothesis #3).
//!
//! The assertion is just `rel_diff < 1.0` (sanity floor — output isn't
//! garbage). The TELEMETRY (the `eprintln!`) is the load-bearing part.
//! Per `feedback_falsifier_chain_assert_difference.md` — this isn't
//! the chain's end; it's the **measurement** step that feeds the next
//! cascade decision.
//!
//! ## Empirical result (lambda-vector RTX 4090, 2026-05-19, apr 0.34.0)
//!
//! ```text
//! distribution        rel_diff        cpu_l2        gpu_l2
//! ------------------------------------------------------------
//! uniform             5.976e-7        22.369        22.369
//! log_normal          2.066e-6        10.354        10.354
//! outlier_5x          4.399e-7        33.426        33.426
//! outlier_100x        3.539e-7       584.389       584.389
//! ```
//!
//! **Hypothesis #2 FALSIFIED.** All four distributions produce rel_diff
//! in the 1e-7 to 2e-6 range. Even `outlier_100x` with `cpu_l2=584`
//! (massive activations) gives rel_diff = 3.5e-7 — actually SMALLER than
//! the uniform baseline.
//!
//! ## Implication for #1583
//!
//! With hypothesis #1 ruled out (expert-routing — code-inspection: both
//! paths run identical host-side Rust) and hypothesis #2 ruled out (this
//! file's empirical sweep), the cascade now points squarely at:
//!
//! - **Hypothesis #3: Accumulator-chain length in the full forward.**
//!   The 0.94-cos drop on layers L7/L9/L12/L20/L23/L29/L46 must come from
//!   COMPOSITIONAL round-off across many ops (embedding → RMSNorm → QKV
//!   → RoPE → causal attention → output proj → router → expert FFN
//!   → residual), not from any single primitive's reduction order.
//!
//! Next falsifier in the cascade should isolate one layer's chain at a
//! time — start with the layer-0 to layer-7 prefix (cheapest, since L7
//! is the first reported divergence layer in #1583).
//!
//! ## How to run
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_q6k_activation_amplification_002 \
//!   -- --ignored --nocapture
//! ```
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - Sibling: `falsify_q6k_fp_accumulator_order_001.rs` (#1801) — uniform-input baseline
//! - CPU side: `crates/aprender-serve/src/quantize/fused_q.rs::fused_q6k_parallel_matvec`
//! - GPU side: `crates/aprender-serve/src/cuda/executor/fused.rs::CudaExecutor::q6k_gemv`
//! - Memory: `feedback_falsifier_cascade_decomposes_magnitude.md`,
//!   `feedback_falsifier_chain_assert_difference.md`

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;
use realizar::quantize::fused_q6k_parallel_matvec;
use trueno_quant::quantize_q6_k_matrix;

const IN_DIM: usize = 256;
const OUT_DIM: usize = 16;

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// `kind` = "uniform" → flat in [-1, 1].
/// `kind` = "log_normal" → mild bursty.
/// `kind` = "outlier_5x" → 95% uniform [-1,1], 5% at ±5×.
/// `kind` = "outlier_100x" → 95% uniform [-1,1], 5% at ±100×.
fn synthetic_activations(n: usize, seed: u32, kind: &str) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let bits1 = xorshift32(&mut state);
            let u1 = (bits1 as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            match kind {
                "uniform" => u1,
                "log_normal" => {
                    // Crude log-normal via exp of a centered uniform: spread ~1e0 to ~1e2.
                    let bits2 = xorshift32(&mut state);
                    let u2 = (bits2 as f32) / (u32::MAX as f32) - 0.5; // ∈ [-0.5, 0.5]
                    let mag = (u2 * 4.0).exp(); // ∈ [~0.14, ~7.4]
                    u1.signum() * mag * 0.1 // scale down so most stay <1.0
                }
                "outlier_5x" => {
                    let bits2 = xorshift32(&mut state);
                    let pick = (bits2 as f32) / (u32::MAX as f32); // ∈ [0, 1]
                    if pick < 0.05 {
                        // 5% outliers at ±5×
                        u1 * 5.0
                    } else {
                        u1
                    }
                }
                "outlier_100x" => {
                    let bits2 = xorshift32(&mut state);
                    let pick = (bits2 as f32) / (u32::MAX as f32);
                    if pick < 0.05 {
                        // 5% outliers at ±100×
                        u1 * 100.0
                    } else {
                        u1
                    }
                }
                _ => unreachable!("unknown activation kind: {kind}"),
            }
        })
        .collect()
}

fn synthetic_weights(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let bits = xorshift32(&mut state);
            (bits as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-30);
            (x - y).abs() / denom
        })
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "requires CUDA hardware (RTX 4090); synthetic data sweep, runs in <2s"]
fn falsify_q6k_activation_amplification_sweep() {
    let weights_f32 = synthetic_weights(OUT_DIM * IN_DIM, 0x1583_0002);
    let weights_q6k = quantize_q6_k_matrix(&weights_f32, &[OUT_DIM, IN_DIM]);

    let mut cuda = CudaExecutor::new(0)
        .expect("CudaExecutor::new(0) must succeed per realizar CLAUDE.md CUDA-always rule");

    // Sweep activation distributions from uniform → bursty.
    let distributions = [
        ("uniform     ", "uniform"),
        ("log_normal  ", "log_normal"),
        ("outlier_5x  ", "outlier_5x"),
        ("outlier_100x", "outlier_100x"),
    ];

    eprintln!(
        "FALSIFY-Q6K-AMP-002: per-matvec rel_diff vs activation distribution"
    );
    eprintln!("(if rel_diff scales with non-uniformity → activation-amplification confirmed)");
    eprintln!();
    eprintln!(
        "{:14}  {:>12}  {:>12}  {:>12}",
        "distribution", "rel_diff", "cpu_l2", "gpu_l2"
    );
    eprintln!("{}", "-".repeat(60));

    for (label, kind) in distributions {
        let activations = synthetic_activations(IN_DIM, 0x1583_0003, kind);
        let cpu_out = fused_q6k_parallel_matvec(&weights_q6k, &activations, IN_DIM, OUT_DIM)
            .expect("cpu Q6_K matvec must not error");
        let mut gpu_out = vec![0.0f32; OUT_DIM];
        cuda.q6k_gemv(
            &weights_q6k,
            &activations,
            &mut gpu_out,
            OUT_DIM as u32,
            IN_DIM as u32,
        )
        .expect("CudaExecutor::q6k_gemv must succeed");

        // Sanity: both outputs finite.
        assert!(
            cpu_out.iter().all(|x| x.is_finite()),
            "cpu output non-finite at distribution={label} → {cpu_out:?}"
        );
        assert!(
            gpu_out.iter().all(|x| x.is_finite()),
            "gpu output non-finite at distribution={label} → {gpu_out:?}"
        );

        let cpu_l2: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        let gpu_l2: f32 = gpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel = max_rel_diff(&cpu_out, &gpu_out);

        eprintln!(
            "{:14}  {:>12.3e}  {:>12.3}  {:>12.3}",
            label, rel, cpu_l2, gpu_l2
        );

        // Sanity-floor: output isn't garbage. Per the falsifier-cascade
        // pattern (`feedback_falsifier_cascade_decomposes_magnitude.md`),
        // the load-bearing artifact is the eprintln-telemetry above;
        // this assertion just keeps the test from regressing into a
        // no-op if something completely breaks.
        assert!(
            rel < 1.0,
            "rel_diff = {rel:.3e} ≥ 1.0 at distribution={label} — output is garbage, \
             not a divergence pattern. cpu={cpu_out:?}, gpu={gpu_out:?}"
        );
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn synthetic_activations_uniform_in_range() {
        let v = synthetic_activations(1000, 1, "uniform");
        assert_eq!(v.len(), 1000);
        for x in &v {
            assert!(x.is_finite());
            assert!((-1.0..=1.0).contains(x), "uniform out of range: {x}");
        }
    }

    #[test]
    fn synthetic_activations_outlier_100x_has_some_large_values() {
        // 5% outliers × 1000 samples = ~50 expected; lower bound generous.
        let v = synthetic_activations(1000, 1, "outlier_100x");
        let large_count = v.iter().filter(|&&x| x.abs() > 10.0).count();
        assert!(
            large_count >= 10,
            "outlier_100x should have ≥10 large values in 1000 samples; got {large_count}"
        );
    }

    #[test]
    fn synthetic_activations_uniform_no_outliers() {
        // uniform should have NO values > 1.0 in absolute terms.
        let v = synthetic_activations(1000, 1, "uniform");
        let large_count = v.iter().filter(|&&x| x.abs() > 1.001).count();
        assert_eq!(large_count, 0, "uniform should have no outliers; got {large_count}");
    }

    #[test]
    fn max_rel_diff_zero_for_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(max_rel_diff(&a, &a), 0.0);
    }

    #[test]
    fn max_rel_diff_handles_zero_via_floor() {
        // a is zero, b is small → denom floor prevents div-by-zero blow-up.
        let a = vec![0.0, 0.0];
        let b = vec![1e-10, 1e-10];
        let rel = max_rel_diff(&a, &b);
        assert!(rel.is_finite(), "rel_diff with zero baseline must stay finite, got {rel}");
    }
}
