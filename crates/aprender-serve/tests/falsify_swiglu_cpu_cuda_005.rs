//! FALSIFY-SWIGLU-CPU-CUDA-005 — does the SwiGLU activation kernel
//! produce different f32 round-off between CPU `f32::exp` and CUDA
//! `ex2.approx.f32 * LOG2_E`?
//!
//! ## Why this falsifier exists
//!
//! Track A M-GPU-MOE-3 cascade (#1583) cumulative state after 4 PRs:
//!
//! 1. [#1801](https://github.com/paiml/aprender/pull/1801) — per-matvec
//!    Q6_K reduction-order is ulp-scale on synthetic input.
//! 2. [#1805](https://github.com/paiml/aprender/pull/1805) — activation
//!    distribution amplification is flat (uniform → bursty same floor).
//! 3. [#1811](https://github.com/paiml/aprender/pull/1811) — chain-length
//!    compounding is flat (N=1 to N=48 same floor).
//! 4. [#1816](https://github.com/paiml/aprender/pull/1816) — real Qwen3
//!    Q6_K weights match the synthetic ulp-scale floor + structural
//!    finding that 3 of 7 "problem layers" (L7/L9/L12) use Q4_K, not
//!    Q6_K, ruling out Q6_K-specific root cause.
//!
//! **The remaining root cause must be qtype-agnostic.** #1816's pivot
//! ranked SwiGLU activation parity as the HIGHEST EV candidate
//! because:
//!
//! - CPU `expert_swiglu_quantized` uses `f32::exp(-x)` (natural exp).
//! - CUDA `FusedSwigluKernel::build_ptx` uses `ex2.approx.f32` with a
//!   `LOG2_E` multiplier: `silu(x) = x / (1 + exp(-x))` becomes
//!   `x / (1 + ex2(-x * LOG2_E))`.
//! - Algebraically identical sigmoid.
//! - **But the PTX `ex2.approx.f32` is documented as "approximate" — ~2
//!   ulps — while libm `f32::exp` is typically 1 ulp accurate.**
//!
//! Both Q4_K and Q6_K MoE FFN paths hit this op at the same point in
//! the forward (gate matmul → up matmul → SwiGLU → down matvec).
//! Qtype-agnostic root cause hypothesis fits.
//!
//! ## What this falsifier measures
//!
//! Runs CPU `silu(gate) * up` (the exact formula in
//! `expert_swiglu_quantized`) and GPU `CudaExecutor::fused_swiglu_host`
//! on identical synthetic gate/up vectors across multiple input
//! distributions:
//!
//! 1. **uniform** — flat in [-1, 1] (matches #1801 / #1805 baselines)
//! 2. **moderate** — flat in [-5, 5] (typical activation range)
//! 3. **extreme_neg** — flat in [-20, -10] (sigmoid → 0, exp → ∞)
//! 4. **extreme_pos** — flat in [10, 20] (sigmoid → 1, exp(-x) → 0)
//! 5. **mixed** — flat in [-20, 20] (both extremes in one batch)
//!
//! Reports max_abs_diff and max_rel_diff for each distribution.
//!
//! ## Empirical result (lambda-vector RTX 4090, 2026-05-19)
//!
//! ```text
//! distribution      lo      hi    max_abs     max_rel     cpu_l2
//! ------------------------------------------------------------------
//! uniform        -1.00    1.00   5.960e-8    2.369e-7      11.300
//! moderate       -5.00    5.00   1.907e-6    4.303e-7     362.262
//! extreme_neg   -20.00  -10.00   4.657e-9    9.970e-7       0.107
//! extreme_pos    10.00   20.00    0.000e0     0.000e0   14930.386
//! mixed         -20.00   20.00   7.629e-6    9.803e-7    5998.385
//! ```
//!
//! **Hypothesis FALSIFIED.** SwiGLU parity is FINE across all
//! distributions (rel_diff in 0 to 1e-6 range). Even on `mixed`
//! (the most extreme [-20, 20] range), max_rel stays at 9.8e-7 —
//! ulp-scale floor. The `ex2.approx.f32` vs `f32::exp` discrepancy
//! is NOT visible at the SwiGLU activation level.
//!
//! Notably `extreme_pos` (gate ∈ [10, 20]) gives rel_diff = 0.0
//! exactly — both sides converge to `silu(g) ≈ g` since
//! `exp(-g) ≈ 0` in both intrinsics, and the f32 multiply is bit-
//! identical.
//!
//! ## Cumulative cascade status — 6 hypotheses ruled out
//!
//! 1. ✅ Per-matvec Q6_K reduction-order on synthetic (#1801)
//! 2. ✅ Activation distribution amplification (#1805)
//! 3. ✅ Accumulator-chain length compounding (#1811)
//! 4. ✅ Per-matvec Q6_K on real Qwen3 weights (#1816)
//! 5. ✅ Q6_K-specific root cause — structural qtype-mix (#1816)
//! 6. ✅ SwiGLU activation parity (this PR)
//!
//! ## Remaining candidates
//!
//! 1. **Q4_K real-weight matvec parity** — has not been directly
//!    tested CPU vs CUDA on real Qwen3 Q4_K weights. The Q4_K kernel
//!    path is different from Q6_K (different super-block layout,
//!    different reduction shape).
//! 2. **Compositional FFN-block** — gate matmul × up matmul × SwiGLU
//!    × down matvec × weighted_sum chained together with real Qwen3
//!    weights and real activation distributions (not synthetic + L2
//!    norm between steps). The 7-element FFN-block chain on real
//!    inputs may exhibit a divergence pattern that no single primitive
//!    individually shows.
//! 3. **Top-K weighted-sum** accumulation order — host-side f32
//!    accumulation that combines top-K expert outputs. Bit-identical
//!    in code inspection but worth empirical verification.
//!
//! ## How to run
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_swiglu_cpu_cuda_005 \
//!   -- --ignored --nocapture
//! ```
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - Predecessors: #1801, #1805, #1811, #1816 (all Q6_K-focused)
//! - CPU side: `crates/aprender-serve/src/gguf/qwen3_moe_load.rs::expert_swiglu_quantized`
//! - CUDA side: `crates/aprender-serve/src/cuda/executor/kernel.rs::CudaExecutor::fused_swiglu_host`
//! - PTX kernel: `crates/aprender-gpu/src/kernels/elementwise/swiglu.rs::FusedSwigluKernel`
//! - Memory: `feedback_falsifier_cascade_decomposes_magnitude.md`

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;

const N: usize = 4096;

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Generate synthetic f32 vector in `[lo, hi]` with deterministic seed.
fn synthetic_range(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    let span = hi - lo;
    (0..n)
        .map(|_| {
            let bits = xorshift32(&mut state);
            let u = (bits as f32) / (u32::MAX as f32); // ∈ [0, 1]
            lo + u * span
        })
        .collect()
}

/// CPU SiLU * Up — mirror of the formula in `expert_swiglu_quantized`.
/// `silu(x) = x / (1 + exp(-x))`. Element-wise multiply by `up`.
fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let sigmoid_g = 1.0 / (1.0 + (-g).exp());
            g * sigmoid_g * u
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
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
#[ignore = "requires CUDA hardware (RTX 4090); runs in <2s on n=4096 across 5 distributions"]
fn falsify_swiglu_cpu_vs_cuda_intrinsic_divergence() {
    let mut cuda = CudaExecutor::new(0)
        .expect("CudaExecutor::new(0) must succeed per realizar CLAUDE.md CUDA-always rule");

    eprintln!("FALSIFY-SWIGLU-CPU-CUDA-005: per-distribution SwiGLU parity");
    eprintln!("(if rel_diff blows up at extreme ranges → ex2.approx.f32 is the amplifier)");
    eprintln!();
    eprintln!(
        "{:14}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}",
        "distribution", "lo", "hi", "max_abs", "max_rel", "cpu_l2"
    );
    eprintln!("{}", "-".repeat(78));

    let distributions: [(&str, f32, f32); 5] = [
        ("uniform     ", -1.0, 1.0),
        ("moderate    ", -5.0, 5.0),
        ("extreme_neg ", -20.0, -10.0),
        ("extreme_pos ", 10.0, 20.0),
        ("mixed       ", -20.0, 20.0),
    ];

    for (label, lo, hi) in distributions {
        // Deterministic per-distribution seeds
        let gate = synthetic_range(N, 0x1583_0005_u32.wrapping_add(label.len() as u32), lo, hi);
        let up = synthetic_range(N, 0x1583_0006_u32.wrapping_add(label.len() as u32), lo, hi);

        let cpu_out = cpu_swiglu(&gate, &up);
        let mut gpu_out = vec![0.0f32; N];
        cuda.fused_swiglu_host(&gate, &up, &mut gpu_out)
            .expect("gpu fused_swiglu_host must succeed");

        // Both outputs must be finite for the measurement to be meaningful.
        assert!(
            cpu_out.iter().all(|x| x.is_finite()),
            "cpu_out non-finite at {label}: first few {:?}",
            &cpu_out[..5]
        );
        assert!(
            gpu_out.iter().all(|x| x.is_finite()),
            "gpu_out non-finite at {label}: first few {:?}",
            &gpu_out[..5]
        );

        let abs = max_abs_diff(&cpu_out, &gpu_out);
        let rel = max_rel_diff(&cpu_out, &gpu_out);
        let cpu_l2: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();

        eprintln!(
            "{:14}  {:>10.2}  {:>10.2}  {:>10.3e}  {:>12.3e}  {:>12.3}",
            label, lo, hi, abs, rel, cpu_l2
        );

        // Sanity floor — output isn't garbage. Telemetry above is the
        // load-bearing artifact per
        // `feedback_falsifier_chain_assert_difference.md`.
        assert!(
            rel < 1.0,
            "rel_diff = {rel:.3e} ≥ 1.0 at {label}: total divergence rather than \
             precision differential. cpu={:?}, gpu={:?}",
            &cpu_out[..5],
            &gpu_out[..5]
        );
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn synthetic_range_respects_bounds() {
        let v = synthetic_range(1000, 42, -3.0, 7.0);
        for x in &v {
            assert!(x.is_finite());
            assert!(*x >= -3.0 - 1e-5);
            assert!(*x <= 7.0 + 1e-5);
        }
    }

    #[test]
    fn synthetic_range_deterministic() {
        let a = synthetic_range(100, 12345, -1.0, 1.0);
        let b = synthetic_range(100, 12345, -1.0, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn cpu_swiglu_zero_gate_gives_zero() {
        // SiLU(0) = 0 → output[i] = 0 regardless of up[i].
        let gate = vec![0.0f32; 8];
        let up = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = cpu_swiglu(&gate, &up);
        for x in &out {
            assert!(x.abs() < 1e-7, "silu(0)*up should be ~0; got {x}");
        }
    }

    #[test]
    fn cpu_swiglu_large_positive_gate_approaches_gate_times_up() {
        // For large positive g, sigmoid → 1, so silu(g) → g, and output → g*u.
        let g = 100.0f32;
        let u = 2.0f32;
        let out = cpu_swiglu(&[g], &[u]);
        let expected = g * u;
        assert!(
            (out[0] - expected).abs() / expected.abs() < 1e-6,
            "silu(100)*2 ≈ 200; got {} (rel_err={:.3e})",
            out[0],
            (out[0] - expected).abs() / expected.abs()
        );
    }

    #[test]
    fn cpu_swiglu_large_negative_gate_approaches_zero() {
        // For large negative g, sigmoid → 0, so silu(g) → 0, output → 0.
        let g = -100.0f32;
        let u = 5.0f32;
        let out = cpu_swiglu(&[g], &[u]);
        assert!(
            out[0].abs() < 1e-30,
            "silu(-100)*5 should be ~0; got {}",
            out[0]
        );
    }

    #[test]
    fn max_rel_diff_zero_for_identical() {
        let a = vec![0.5, -1.5, 0.0, 7.0];
        assert_eq!(max_rel_diff(&a, &a), 0.0);
    }
}
