//! FALSIFY-Q6K-FP-ACCUMULATOR-ORDER-001 — empirical synthetic divergence
//! between CPU `fused_q6k_parallel_matvec` (rayon midi-tile reduction) and
//! GPU `CudaExecutor::q6k_gemv` (warp-shuffle reduction).
//!
//! ## Why this falsifier exists
//!
//! M-GPU-MOE-3 (#1583) claims ~7-8 specific layers (L7/L9/L12/L20/L23/L29/L46)
//! of `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` sit at cos 0.94-0.987 between CPU
//! `forward_qwen3_moe` and GPU `forward_qwen3_moe_cuda`. The existing
//! `qwen3_moe_per_layer_gpu_parity.rs` (FALSIFY-QW3-MOE-PER-LAYER-001) captures
//! that claim on the real model — but the real model is slow (full forward pass,
//! 30B params, minutes per run) and the divergence is layered on top of every
//! other op (RoPE, RMSNorm, attention, router, etc).
//!
//! **This file decomposes the multi-layer parity problem into a single-matvec
//! falsifier.** It uses tiny synthetic Q6_K data (16 rows × 256 cols = 1
//! super-block wide, 16-vector output), runs CPU + GPU sides on the same bytes,
//! and asserts the EXPECTED accumulator-order divergence pattern.
//!
//! Per `feedback_falsifier_cascade_decomposes_magnitude.md` — 1 PR ≈ 1
//! falsifier, ~50-200 LOC. Per `feedback_falsifier_chain_assert_difference.md`
//! — once divergence is empirically pinned, the assertion asserts the EXPECTED
//! non-zero gap (not zero) so a future "fix" that accidentally aligns to the
//! wrong-order reduction would still fail the test.
//!
//! ## What this asserts
//!
//! 1. **Both finite + non-trivial**: neither side produces NaN/Inf, both have
//!    non-zero L2 norm. Sanity floor — if either side is dead, the divergence
//!    measurement is meaningless.
//! 2. **High cosine similarity**: `cos(cpu, gpu) ≥ 0.99`. Same algorithm on
//!    same bytes; the divergence is fp-accumulator-order only, not
//!    quantization or dispatch.
//! 3. **Non-zero element-wise divergence**: at least one of `gpu[i]` differs
//!    from `cpu[i]` (bit-level non-identity). This proves the
//!    accumulator-order claim is real, not just a numerical-precision
//!    coincidence — and pins the divergence at the per-matvec primitive
//!    rather than further up the stack.
//!
//! ## How this decomposes the M-GPU-MOE-3 cascade
//!
//! The real-model parity test reports an aggregated metric across:
//!   `fused_q6k_parallel_matvec` × N_experts × N_tokens × N_layers
//!
//! If this synthetic falsifier shows divergence ε per matvec, then by linearity
//! the cumulative per-layer divergence is ≈ ε × N_experts (per-token, since
//! the activation distribution shifts each layer). The 7-8 layers near
//! cos 0.94-0.987 are layers where the activation distribution amplifies ε
//! by 5-10× due to per-tensor non-uniformity (per the M99/M100 amplifier
//! findings in `feedback_falsifier_cascade_decomposes_magnitude.md`).
//!
//! ## Empirical result (lambda-vector RTX 4090, 2026-05-18, apr 0.34.0)
//!
//! ```text
//! cos = 1.000000      (within f32 print precision)
//! max_abs_diff = 9.537e-7  (single ulp at this magnitude)
//! rel_diff = 1.051e-7
//! cpu_l2 = 18.991, gpu_l2 = 18.991  (identical to 4 sig figs)
//! ```
//!
//! **The per-matvec divergence is REAL but tiny (~1 ulp at f32).** Compounding
//! linearly across 128 experts × 48 layers gives ~1e-3 cumulative — nowhere
//! near the 0.94-0.987 cosine drop cited in #1583 for the 7 problem layers.
//!
//! **Implication**: the M-GPU-MOE-3 root cause is NOT simple per-matvec
//! reduction-order divergence. The real-model parity drop must come from
//! amplification — likely candidates:
//!   1. Expert-routing differences (different top-K experts selected on
//!      CPU vs GPU because router softmax tiebreaks shift)
//!   2. Activation distribution non-uniformity (real Qwen3 layers have
//!      bursty activations that amplify ulp-scale rel_diff by 1000×+)
//!   3. Accumulator-chain length in the full forward (RoPE + softmax +
//!      matvec compose, accumulating round-off across many ops)
//!
//! This falsifier rules out hypothesis (3a) "fp-accumulator-order at the
//! matvec primitive is sufficient" — the gap exists but isn't large enough
//! on synthetic uniform input.
//!
//! Fix path for the underlying real-model issue is **not** in this file —
//! this falsifier merely pins the per-matvec divergence floor so a future
//! cascade can build on top (e.g. "expert-routing parity falsifier",
//! "activation-distribution amplification falsifier") to find the actual
//! amplifier.
//!
//! ## How to run
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_q6k_fp_accumulator_order_001 \
//!   -- --ignored --nocapture
//! ```
//!
//! Skipped on non-CUDA hosts via `#![cfg(feature = "cuda")]`. The `#[ignore]`
//! gate keeps it out of default `cargo test` while still being discoverable
//! via `--ignored` (consistent with the per-layer real-model falsifier).
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - Sibling falsifier: `qwen3_moe_per_layer_gpu_parity.rs` (FALSIFY-QW3-MOE-PER-LAYER-001)
//! - CPU side: `crates/aprender-serve/src/quantize/fused_q.rs::fused_q6k_parallel_matvec`
//! - GPU side: `crates/aprender-serve/src/cuda/executor/fused.rs::CudaExecutor::q6k_gemv`
//! - Memory: `feedback_falsifier_cascade_decomposes_magnitude.md`,
//!   `feedback_falsifier_chain_assert_difference.md`

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;
use realizar::quantize::fused_q6k_parallel_matvec;
use trueno_quant::quantize_q6_k_matrix;

/// Deterministic xorshift32 — keeps this test free of an `rand` dev-dep.
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Generate a deterministic synthetic f32 vector with values in [-1.0, 1.0].
fn synthetic_vec(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let bits = xorshift32(&mut state);
            // Map u32 → [-1.0, 1.0] via division by u32::MAX / 2.
            (bits as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// One Q6_K super-block is 256 elements. Tightest synthetic shape: in_dim = 256
/// (single super-block per row), out_dim = 16 (16 rows = 16 matvec results).
const IN_DIM: usize = 256;
const OUT_DIM: usize = 16;

/// Cosine threshold — same algorithm on same bytes, expect high agreement.
/// Tight (0.99 like the real-model gate) but not impossibly tight; the whole
/// point of this falsifier is that the gap is non-zero.
const COSINE_THRESHOLD: f32 = 0.99;

#[test]
#[ignore = "requires CUDA hardware (RTX 4090); synthetic data, runs in <1s"]
fn falsify_q6k_fp_accumulator_order_cpu_vs_cuda_q6k_gemv() {
    // ── 1. Synthetic inputs (deterministic across runs/hosts) ────────────
    let weights_f32 = synthetic_vec(OUT_DIM * IN_DIM, 0x1583_0001);
    let activations = synthetic_vec(IN_DIM, 0x1583_0002);

    // Quantize the synthetic weights to Q6_K using the SAME quantizer both
    // sides will see. trueno_quant::quantize_q6_k_matrix lays rows out
    // contiguously: row 0 super-blocks, then row 1 super-blocks, etc.
    let weights_q6k = quantize_q6_k_matrix(&weights_f32, &[OUT_DIM, IN_DIM]);

    // ── 2. CPU side: fused_q6k_parallel_matvec (rayon midi-tile) ─────────
    let cpu_out = fused_q6k_parallel_matvec(&weights_q6k, &activations, IN_DIM, OUT_DIM)
        .expect("cpu Q6_K matvec must not error on well-formed inputs");
    assert_eq!(cpu_out.len(), OUT_DIM, "cpu output dim must match out_dim");

    // ── 3. GPU side: CudaExecutor::q6k_gemv (warp-shuffle) ───────────────
    let mut cuda = match CudaExecutor::new(0) {
        Ok(exec) => exec,
        Err(e) => panic!(
            "CudaExecutor::new(0) failed — RTX 4090 should be available per realizar \
             CLAUDE.md's CUDA-always-available rule. Real error: {e:?}"
        ),
    };
    let mut gpu_out = vec![0.0f32; OUT_DIM];
    cuda.q6k_gemv(
        &weights_q6k,
        &activations,
        &mut gpu_out,
        OUT_DIM as u32, // n = output dim
        IN_DIM as u32,  // k = input dim
    )
    .expect("CudaExecutor::q6k_gemv must succeed on well-formed inputs");

    // ── 4. Sanity floor — neither side dead ──────────────────────────────
    assert!(
        cpu_out.iter().all(|x| x.is_finite()),
        "cpu output must be all-finite; got {cpu_out:?}"
    );
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "gpu output must be all-finite; got {gpu_out:?}"
    );
    let cpu_l2: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let gpu_l2: f32 = gpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        cpu_l2 > 1e-6,
        "cpu L2 must be non-trivial; got {cpu_l2} on out={cpu_out:?}"
    );
    assert!(
        gpu_l2 > 1e-6,
        "gpu L2 must be non-trivial; got {gpu_l2} on out={gpu_out:?}"
    );

    // ── 5. High agreement (algorithmic parity) ──────────────────────────
    let cos = cosine_similarity(&cpu_out, &gpu_out);
    assert!(
        cos >= COSINE_THRESHOLD,
        "FALSIFY-Q6K-FP-ACC-001: cos(cpu, gpu) below floor — algorithm-level \
         divergence, not just accumulator order. \
         cos = {cos:.6}, threshold = {COSINE_THRESHOLD}, \
         cpu = {cpu_out:?}, gpu = {gpu_out:?}"
    );

    // ── 6. The empirical claim: element-wise divergence exists ──────────
    // Per #1583, CPU rayon midi-tile reduction and GPU warp-shuffle
    // reduction produce DIFFERENT f32 sums-of-products on the same bytes
    // because float addition is non-associative. This is the bit-level
    // claim that 7-8 layers of the real model cite.
    //
    // If `max_diff == 0`, then either (a) the reductions happen to align
    // on THIS synthetic input (which is itself a finding worth surfacing)
    // OR (b) somebody fixed the GPU side and forgot to update the
    // real-model expectation in #1583. Either way: fail loud.
    let max_diff = max_abs_diff(&cpu_out, &gpu_out);
    let cpu_max_abs = cpu_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_diff = if cpu_max_abs > 0.0 {
        max_diff / cpu_max_abs
    } else {
        0.0
    };

    eprintln!(
        "FALSIFY-Q6K-FP-ACC-001: cos={cos:.6} max_abs_diff={max_diff:.3e} \
         rel_diff={rel_diff:.3e} cpu_l2={cpu_l2:.3} gpu_l2={gpu_l2:.3}"
    );

    assert!(
        max_diff > 0.0,
        "FALSIFY-Q6K-FP-ACC-001: max_abs_diff is ZERO — CPU and GPU produced \
         BIT-IDENTICAL output on synthetic Q6_K. Either #1583's accumulator-order \
         claim is wrong, OR the GPU reduction was silently fixed, OR this \
         synthetic input fails to trigger the divergence (e.g. trivial scale). \
         Investigate before relaxing this assertion. \
         cpu = {cpu_out:?}, gpu = {gpu_out:?}"
    );
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn xorshift32_is_deterministic_across_calls() {
        let mut state_a = 0xdead_beef;
        let mut state_b = 0xdead_beef;
        for _ in 0..100 {
            assert_eq!(xorshift32(&mut state_a), xorshift32(&mut state_b));
        }
    }

    #[test]
    fn synthetic_vec_has_correct_length_and_range() {
        let v = synthetic_vec(1000, 42);
        assert_eq!(v.len(), 1000);
        for x in &v {
            assert!(x.is_finite(), "synthetic_vec produced non-finite: {x}");
            assert!((-1.0..=1.0).contains(x), "synthetic_vec out of range: {x}");
        }
    }

    #[test]
    fn cosine_similarity_handles_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let cos = cosine_similarity(&a, &a);
        assert!(
            (cos - 1.0).abs() < 1e-6,
            "cos(a, a) should be 1.0; got {cos}"
        );
    }

    #[test]
    fn cosine_similarity_returns_zero_for_degenerate_inputs() {
        let a = vec![0.0; 4];
        let b = vec![1.0; 4];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
    }

    #[test]
    fn max_abs_diff_zero_for_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(max_abs_diff(&a, &a), 0.0);
    }

    #[test]
    fn max_abs_diff_picks_largest_element_gap() {
        let a = vec![0.0, 1.0, 5.0];
        let b = vec![0.5, 1.0, 5.001];
        // gaps: 0.5, 0.0, 0.001 → max = 0.5
        assert!((max_abs_diff(&a, &b) - 0.5).abs() < 1e-6);
    }
}
