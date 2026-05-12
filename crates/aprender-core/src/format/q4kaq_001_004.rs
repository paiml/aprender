// SHIP-TWO-001 — `cpu-q4k-activation-quant-v1` algorithm-level PARTIAL
// discharge for FALSIFY-AQ-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/cpu-q4k-activation-quant-v1.yaml`.
// Spec: CPU Q4K kernel must pre-quantize activations to Q8_K for
// integer-only inner loop (qwen-coder-deploy bench: apr 9.5 tok/s
// vs llama.cpp 74 tok/s, 7.8× gap).

// ===========================================================================
// AQ-001 — Dot product parity: |dot_q4k_q8k - dot_q4k_f32| / |dot_f32| ≤ 0.001
// ===========================================================================

pub const AC_AQ_001_RELATIVE_TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aq001Verdict { Pass, Fail }

/// Pass iff `|dot_q8k - dot_f32| / max(|dot_f32|, ε)` ≤ 0.001 (per-vector
/// relative tolerance — matches the contract's "0.1%" spec).
#[must_use]
pub fn verdict_from_dot_product_parity(dot_q8k: f32, dot_f32: f32) -> Aq001Verdict {
    if !dot_q8k.is_finite() || !dot_f32.is_finite() { return Aq001Verdict::Fail; }
    let denom = dot_f32.abs().max(1.0e-6_f32);
    let rel = (dot_q8k - dot_f32).abs() / denom;
    if !rel.is_finite() { return Aq001Verdict::Fail; }
    if rel > AC_AQ_001_RELATIVE_TOLERANCE { return Aq001Verdict::Fail; }
    Aq001Verdict::Pass
}

// ===========================================================================
// AQ-002 — Throughput target: apr_cpu_tps ≥ 0.85 × llama_cpp_cpu_tps
// ===========================================================================

pub const AC_AQ_002_MIN_RATIO: f32 = 0.85;
pub const AC_AQ_002_TARGET_TPS: f32 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aq002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_target(
    apr_cpu_tps: f32,
    llama_cpp_cpu_tps: f32,
) -> Aq002Verdict {
    if !apr_cpu_tps.is_finite() || !llama_cpp_cpu_tps.is_finite() {
        return Aq002Verdict::Fail;
    }
    if apr_cpu_tps <= 0.0 || llama_cpp_cpu_tps <= 0.0 { return Aq002Verdict::Fail; }
    if apr_cpu_tps < AC_AQ_002_TARGET_TPS { return Aq002Verdict::Fail; }
    let ratio = apr_cpu_tps / llama_cpp_cpu_tps;
    if !ratio.is_finite() || ratio < AC_AQ_002_MIN_RATIO { return Aq002Verdict::Fail; }
    Aq002Verdict::Pass
}

// ===========================================================================
// AQ-003 — Amortized quantization: count(quantize_row_q8_k) ≤ out_dim per matmul
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aq003Verdict { Pass, Fail }

/// Pass iff `quantize_call_count` ≤ `num_matmuls` (the target: once per
/// matmul, NOT once per dot product). The contract's regression class is
/// "Quantization called per-dot instead of per-matmul" which produces
/// quantize_calls = out_dim × num_matmuls (orders of magnitude more).
#[must_use]
pub const fn verdict_from_amortized_quantization(
    quantize_call_count: u64,
    num_matmuls: u64,
) -> Aq003Verdict {
    if num_matmuls == 0 { return Aq003Verdict::Fail; }
    if quantize_call_count > num_matmuls { return Aq003Verdict::Fail; }
    if quantize_call_count == 0 { return Aq003Verdict::Fail; } // some quant must occur
    Aq003Verdict::Pass
}

// ===========================================================================
// AQ-004 — No regression in output quality: argmax(q8k) == argmax(f32) per step
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aq004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_output_quality_parity(
    q8k_argmax_per_step: &[u32],
    f32_argmax_per_step: &[u32],
) -> Aq004Verdict {
    if q8k_argmax_per_step.is_empty() || f32_argmax_per_step.is_empty() {
        return Aq004Verdict::Fail;
    }
    if q8k_argmax_per_step.len() != f32_argmax_per_step.len() {
        return Aq004Verdict::Fail;
    }
    if q8k_argmax_per_step == f32_argmax_per_step {
        Aq004Verdict::Pass
    } else {
        Aq004Verdict::Fail
    }
}

// ===========================================================================
// Helper: theoretical speedup from activation quantization (per contract)
// ===========================================================================

/// Bandwidth reduction: sizeof(f32) / sizeof(int8) = 4.
/// Compute reduction: fma_latency / maddubs_latency ≈ 3-4.
/// Combined: 4-8× in memory-bound regime.
pub const AQ_BANDWIDTH_SPEEDUP: f32 = 4.0;
pub const AQ_COMPUTE_SPEEDUP_MIN: f32 = 3.0;
pub const AQ_COMPUTE_SPEEDUP_MAX: f32 = 4.0;

#[cfg(test)]
mod tests {
    use super::*;

    // AQ-001 (dot product parity)
    #[test] fn aq001_pass_identical() {
        assert_eq!(verdict_from_dot_product_parity(1234.5, 1234.5), Aq001Verdict::Pass);
    }
    #[test] fn aq001_pass_within_tolerance() {
        // 0.05% drift — well within 0.1%.
        assert_eq!(verdict_from_dot_product_parity(1000.5, 1000.0), Aq001Verdict::Pass);
    }
    #[test] fn aq001_fail_above_tolerance() {
        // 0.5% drift — exceeds 0.1%.
        assert_eq!(verdict_from_dot_product_parity(1005.0, 1000.0), Aq001Verdict::Fail);
    }
    #[test] fn aq001_pass_near_zero() {
        // Both near zero — the |·| ≥ 1e-6 floor prevents division-by-zero.
        assert_eq!(verdict_from_dot_product_parity(1e-7, 1e-7), Aq001Verdict::Pass);
    }
    #[test] fn aq001_fail_nan() {
        assert_eq!(verdict_from_dot_product_parity(f32::NAN, 1000.0), Aq001Verdict::Fail);
    }
    #[test] fn aq001_fail_inf() {
        assert_eq!(verdict_from_dot_product_parity(f32::INFINITY, 1000.0), Aq001Verdict::Fail);
    }

    // AQ-002 (throughput target)
    #[test] fn aq002_pass_canonical() {
        // apr 65 tok/s, llama 74 tok/s → 0.878 ≥ 0.85, AND apr ≥ 60.
        assert_eq!(verdict_from_throughput_target(65.0, 74.0), Aq002Verdict::Pass);
    }
    #[test] fn aq002_fail_below_60_target() {
        // apr 50 tok/s — below target_tps even though ratio is decent.
        assert_eq!(verdict_from_throughput_target(50.0, 55.0), Aq002Verdict::Fail);
    }
    #[test] fn aq002_fail_below_ratio() {
        // apr 60 tok/s, llama 100 tok/s → ratio 0.6 < 0.85.
        assert_eq!(verdict_from_throughput_target(60.0, 100.0), Aq002Verdict::Fail);
    }
    #[test] fn aq002_fail_pre_fix_baseline() {
        // The contract's stated regression: apr 9.5 tok/s vs llama 74 tok/s.
        // Pre-fix state must Fail this gate.
        assert_eq!(verdict_from_throughput_target(9.5, 74.0), Aq002Verdict::Fail);
    }
    #[test] fn aq002_fail_zero_llama() {
        assert_eq!(verdict_from_throughput_target(60.0, 0.0), Aq002Verdict::Fail);
    }
    #[test] fn aq002_fail_nan() {
        assert_eq!(verdict_from_throughput_target(f32::NAN, 74.0), Aq002Verdict::Fail);
    }

    // AQ-003 (amortized quantization)
    #[test] fn aq003_pass_one_per_matmul() {
        // Canonical: 28 layers × 7 matmuls/layer = 196 matmuls = 196 quant calls.
        assert_eq!(verdict_from_amortized_quantization(196, 196), Aq003Verdict::Pass);
    }
    #[test] fn aq003_pass_fewer_quants() {
        // Cache hit: same activation quantized once, reused for multiple matmuls.
        assert_eq!(verdict_from_amortized_quantization(50, 196), Aq003Verdict::Pass);
    }
    #[test] fn aq003_fail_per_dot_regression() {
        // The contract's stated regression: quantize called per-dot instead
        // of per-matmul. For out_dim=4096 × 196 matmuls = 802,816 calls.
        assert_eq!(verdict_from_amortized_quantization(802816, 196), Aq003Verdict::Fail);
    }
    #[test] fn aq003_fail_zero_matmuls() {
        assert_eq!(verdict_from_amortized_quantization(0, 0), Aq003Verdict::Fail);
    }
    #[test] fn aq003_fail_zero_quants() {
        // Some quant must occur (no quant means F32 fallback path).
        assert_eq!(verdict_from_amortized_quantization(0, 196), Aq003Verdict::Fail);
    }

    // AQ-004 (output quality parity)
    #[test] fn aq004_pass_canonical() {
        let q8k = [42_u32, 100, 7, 99];
        let f32 = [42_u32, 100, 7, 99];
        assert_eq!(verdict_from_output_quality_parity(&q8k, &f32), Aq004Verdict::Pass);
    }
    #[test] fn aq004_fail_step_divergence() {
        // The contract's stated regression: "Quantization error flips argmax".
        let q8k = [42_u32, 100, 7, 99];
        let f32_ax = [42_u32, 100, 8, 99]; // step 2 diverged
        assert_eq!(verdict_from_output_quality_parity(&q8k, &f32_ax), Aq004Verdict::Fail);
    }
    #[test] fn aq004_fail_length_mismatch() {
        let q8k = [42_u32];
        let f32 = [42_u32, 100];
        assert_eq!(verdict_from_output_quality_parity(&q8k, &f32), Aq004Verdict::Fail);
    }
    #[test] fn aq004_fail_empty() {
        assert_eq!(verdict_from_output_quality_parity(&[], &[]), Aq004Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_AQ_001_RELATIVE_TOLERANCE - 1e-3).abs() < 1e-9);
        assert!((AC_AQ_002_MIN_RATIO - 0.85).abs() < 1e-9);
        assert!((AC_AQ_002_TARGET_TPS - 60.0).abs() < 1e-9);
        assert!((AQ_BANDWIDTH_SPEEDUP - 4.0).abs() < 1e-9);
    }
}
