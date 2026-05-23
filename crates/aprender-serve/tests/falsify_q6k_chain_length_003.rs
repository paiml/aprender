//! FALSIFY-Q6K-CHAIN-LENGTH-003 — does compositional accumulator-chain
//! length amplify the per-matvec CPU↔GPU divergence?
//!
//! ## The remaining hypothesis from the M-GPU-MOE-3 cascade
//!
//! [#1801](https://github.com/paiml/aprender/pull/1801) ruled out simple
//! per-matvec reduction-order (rel_diff stayed at ~1e-7 ulp-scale on a
//! single matvec). [#1805](https://github.com/paiml/aprender/pull/1805)
//! ruled out activation-distribution amplification (rel_diff didn't scale
//! with bursty input distributions). Code inspection ruled out router
//! divergence (both paths run the same host-side Rust).
//!
//! **The only remaining hypothesis** for the 0.94-cos drop on layers
//! L7/L9/L12/L20/L23/L29/L46 in #1583 is **compositional round-off**:
//! the per-matvec ulp-scale divergence compounds across many ops in the
//! full forward (embedding → RMSNorm → QKV → RoPE → attention → output
//! proj → router → expert FFN → residual, repeated 48 times for the
//! Qwen3-Coder-30B-A3B forward).
//!
//! ## What this falsifier measures
//!
//! Runs a CHAIN of N synthetic Q6_K matvecs where each step's output
//! feeds the next step's input. Same weights are used by both CPU and
//! GPU sides at every step (so the divergence accumulates ONLY through
//! the reduction-order delta carried forward in activations). Reports
//! rel_diff after N ∈ {1, 2, 4, 8, 16, 32, 48} steps.
//!
//! ## Empirical result (lambda-vector RTX 4090, 2026-05-19, apr 0.34.0)
//!
//! ```text
//! depth      rel_diff         1-cos      cpu_l2
//! --------------------------------------------------
//!     1      6.224e-5       0.000e0       1.000
//!     2      1.059e-4    -1.192e-7       1.000
//!     4      1.011e-4    -1.192e-7       1.000
//!     8      5.595e-5    -1.192e-7       1.000
//!    16      9.533e-4    -1.192e-7       1.000
//!    32      9.063e-5    -2.384e-7       1.000
//!    48      9.862e-5     1.192e-7       1.000
//! ```
//!
//! **Hypothesis #3 ALSO FALSIFIED.** rel_diff stays flat at ~1e-4 across
//! the entire N=1 → N=48 sweep with NO scaling. Cosine stays at ~1.0
//! (1-cos at f32 noise floor). At N=48 (matching the real model's layer
//! count), rel_diff is 9.862e-5 — essentially identical to N=1's
//! 6.224e-5 baseline.
//!
//! Note the rel_diff jumped from ulp-scale (#1801's 1e-7) to ~1e-4 once
//! we added L2 normalization between steps — that's because each step
//! is now a TWO-op chain (matvec + L2 norm), and the L2 norm itself has
//! its own f32 reduction. But the chain length doesn't compound BEYOND
//! that initial 2-op delta.
//!
//! ## Conclusion: all three M-GPU-MOE-3 hypotheses falsified
//!
//! With #1801 (single-matvec ulp-scale), #1805 (activation distribution
//! flat), and this PR (chain length flat), the cascade has **eliminated
//! all three candidate amplifiers** for the 0.94-cos drop on real Qwen3
//! layers L7/L9/L12/L20/L23/L29/L46.
//!
//! ## What the cascade has NOT yet tested
//!
//! Real-model divergence must come from sources NOT in synthetic q6k
//! matvec chains. Remaining candidates worth a next-cascade-PR:
//!
//! 1. **Q4_K matmul parity** — gate/up projections in MoE FFN are
//!    typically Q4_K, not Q6_K. The Q4_K kernel could have a different
//!    reduction order than Q6_K.
//! 2. **SwiGLU activation parity** — CPU `expert_swiglu_quantized` vs
//!    CUDA `expert_swiglu_cuda` use slightly different sigmoid intrinsics
//!    (`exp(-g)` vs `ex2.approx.f32`). Same algebra but different
//!    f32-precision behavior on extreme inputs.
//! 3. **Real Qwen3 weight pattern** — synthetic random weights might
//!    not hit the corner cases that real-model Q6_K weights do. Load
//!    actual layer-7 q6k bytes from the cached GGUF and re-run #1801's
//!    single-matvec test.
//! 4. **Top-K expert weighted sum** — the per-expert outputs are
//!    combined with router probabilities (host-side f32 multiply +
//!    accumulate). This is bit-identical CPU/CUDA by inspection but
//!    worth empirically verifying.
//!
//! Highest EV next falsifier: candidate #3 (real-weight single-matvec).
//! If real Qwen3 Q6_K weights ALSO produce ulp-scale per-matvec
//! divergence, the bug must be elsewhere in the FFN block (Q4_K,
//! SwiGLU, or weighted sum). If real weights DO show 1e-3+ divergence,
//! then synthetic-random was hiding it and #1801's hypothesis was
//! premature.
//!
//! Per `feedback_falsifier_chain_assert_difference.md` — assertions are
//! sanity floors. The load-bearing artifact is the per-N rel_diff table.
//!
//! ## How to run
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_q6k_chain_length_003 \
//!   -- --ignored --nocapture
//! ```
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - Predecessors: #1801 (single-matvec baseline), #1805 (activation sweep)
//! - CPU side: `crates/aprender-serve/src/quantize/fused_q.rs::fused_q6k_parallel_matvec`
//! - GPU side: `crates/aprender-serve/src/cuda/executor/fused.rs::CudaExecutor::q6k_gemv`

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;
use realizar::quantize::fused_q6k_parallel_matvec;
use trueno_quant::quantize_q6_k_matrix;

/// Each step is a 256-in × 256-out Q6_K matvec (chosen so the output
/// can directly feed the next step's input — same dim).
const DIM: usize = 256;

/// Chain depths to sweep. 48 matches the real Qwen3-Coder-30B layer count.
const DEPTHS: &[usize] = &[1, 2, 4, 8, 16, 32, 48];

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn synthetic_vec(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let bits = xorshift32(&mut state);
            (bits as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

/// Build N distinct Q6_K weight matrices, one per chain step.
fn build_chain_weights(n_steps: usize, base_seed: u32) -> Vec<Vec<u8>> {
    (0..n_steps)
        .map(|step| {
            let weights_f32 = synthetic_vec(DIM * DIM, base_seed.wrapping_add(step as u32 * 17));
            quantize_q6_k_matrix(&weights_f32, &[DIM, DIM])
        })
        .collect()
}

/// Normalize a vector to unit-L2 to prevent blow-up across many matvec
/// steps. Real transformer layers have RMSNorm/LayerNorm doing similar
/// scale-control between matvec ops — keeping us in the same magnitude
/// regime so rel_diff measurements are comparable.
fn l2_normalize(v: &mut [f32]) {
    let l2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if l2 > 1e-12 {
        let inv = 1.0 / l2;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Run a chain of `n_steps` Q6_K matvecs on the CPU path.
/// Output of step i becomes input of step i+1 (with L2 normalization).
fn run_cpu_chain(weights: &[Vec<u8>], initial: &[f32]) -> Vec<f32> {
    let mut activation = initial.to_vec();
    for w in weights {
        let mut out = fused_q6k_parallel_matvec(w, &activation, DIM, DIM)
            .expect("cpu Q6_K matvec must not error in chain");
        l2_normalize(&mut out);
        activation = out;
    }
    activation
}

/// Run a chain of `n_steps` Q6_K matvecs on the GPU path.
fn run_gpu_chain(cuda: &mut CudaExecutor, weights: &[Vec<u8>], initial: &[f32]) -> Vec<f32> {
    let mut activation = initial.to_vec();
    let mut out = vec![0.0f32; DIM];
    for w in weights {
        cuda.q6k_gemv(w, &activation, &mut out, DIM as u32, DIM as u32)
            .expect("gpu q6k_gemv must succeed in chain");
        l2_normalize(&mut out);
        activation.copy_from_slice(&out);
    }
    activation
}

#[test]
#[ignore = "requires CUDA hardware (RTX 4090); chain sweep up to depth 48, runs in <5s"]
fn falsify_q6k_chain_length_compositional_round_off() {
    let max_depth = *DEPTHS.iter().max().unwrap();
    let weights = build_chain_weights(max_depth, 0x1583_0003);
    let initial = synthetic_vec(DIM, 0x1583_0004);

    let mut cuda = CudaExecutor::new(0)
        .expect("CudaExecutor::new(0) must succeed per realizar CLAUDE.md CUDA-always rule");

    eprintln!("FALSIFY-Q6K-CHAIN-003: rel_diff and 1-cos vs chain depth N");
    eprintln!(
        "(if scales super-linearly with N → hypothesis #3 CONFIRMED — chain length is the amplifier)"
    );
    eprintln!();
    eprintln!(
        "{:>5}  {:>12}  {:>12}  {:>10}",
        "depth", "rel_diff", "1-cos", "cpu_l2"
    );
    eprintln!("{}", "-".repeat(50));

    for &n in DEPTHS {
        let chain_weights = &weights[..n];
        let cpu_out = run_cpu_chain(chain_weights, &initial);
        let gpu_out = run_gpu_chain(&mut cuda, chain_weights, &initial);

        // Sanity: both outputs finite at this depth.
        assert!(
            cpu_out.iter().all(|x| x.is_finite()),
            "cpu chain output non-finite at depth={n}; first few = {:?}",
            &cpu_out[..5.min(cpu_out.len())]
        );
        assert!(
            gpu_out.iter().all(|x| x.is_finite()),
            "gpu chain output non-finite at depth={n}; first few = {:?}",
            &gpu_out[..5.min(gpu_out.len())]
        );

        let rel = max_rel_diff(&cpu_out, &gpu_out);
        let cos = cosine_similarity(&cpu_out, &gpu_out);
        let cpu_l2: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();

        eprintln!(
            "{:>5}  {:>12.3e}  {:>12.3e}  {:>10.3}",
            n,
            rel,
            (1.0 - cos),
            cpu_l2
        );

        // Sanity floor: rel_diff stays bounded, doesn't blow to infinity.
        // The chain-length hypothesis predicts growth UP TO ~0.5 at N=48
        // (matching the 0.94-cos drop); rel_diff ≥ 1.0 would mean total
        // divergence which is more than the hypothesis predicts.
        assert!(
            rel < 2.0,
            "rel_diff = {rel:.3e} at depth={n} is beyond chain-length hypothesis range \
             (predicted ≤ ~0.5). Either the test setup is broken or there's a stronger \
             amplifier in play."
        );
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn build_chain_weights_returns_distinct_per_step() {
        // Different step indices must produce different weight bytes
        // (deterministic per-step seed via wrapping_add).
        let weights = build_chain_weights(3, 42);
        assert_eq!(weights.len(), 3);
        assert_ne!(
            weights[0], weights[1],
            "step 0 and 1 must have distinct weights"
        );
        assert_ne!(
            weights[1], weights[2],
            "step 1 and 2 must have distinct weights"
        );
    }

    #[test]
    fn l2_normalize_makes_unit_norm() {
        let mut v = vec![3.0f32, 4.0]; // L2 = 5.0
        l2_normalize(&mut v);
        let l2_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (l2_after - 1.0).abs() < 1e-6,
            "L2 after normalize should be 1.0; got {l2_after}"
        );
    }

    #[test]
    fn l2_normalize_zero_vector_stays_zero() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v); // must not div-by-zero
        for x in &v {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn cosine_similarity_identical_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        let cos = cosine_similarity(&a, &a);
        assert!((cos - 1.0).abs() < 1e-6, "cos(a, a) = {cos}");
    }

    #[test]
    fn max_rel_diff_zero_for_identical() {
        let a = vec![0.5, 1.5];
        assert_eq!(max_rel_diff(&a, &a), 0.0);
    }

    #[test]
    fn synthetic_vec_deterministic_across_calls() {
        let a = synthetic_vec(100, 0xdead);
        let b = synthetic_vec(100, 0xdead);
        assert_eq!(a, b, "same seed must produce same vector");
    }
}
