// SHIP-TWO-001 — `safetensors-cpu-dispatch-v1` algorithm-level PARTIAL
// discharge for FALSIFY-SD-001..003 (closes 3/3 sweep).
//
// Contract: `contracts/safetensors-cpu-dispatch-v1.yaml`.
// Spec: SafeTensors CPU path must dispatch to quantized kernels after
// runtime Q4K conversion (qwen-coder-deploy bench-results-v2:
// SafeTensors 6.0 vs GGUF 9.5 tok/s = 36% gap regression).

// ===========================================================================
// SD-001 — Format throughput parity: SafeTensors CPU ≥ 0.9 × GGUF CPU
// ===========================================================================

pub const AC_SD_001_MIN_RATIO: f32 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sd001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_parity(safetensors_tps: f32, gguf_tps: f32) -> Sd001Verdict {
    if !safetensors_tps.is_finite() || !gguf_tps.is_finite() { return Sd001Verdict::Fail; }
    if safetensors_tps <= 0.0 || gguf_tps <= 0.0 { return Sd001Verdict::Fail; }
    let ratio = safetensors_tps / gguf_tps;
    if !ratio.is_finite() { return Sd001Verdict::Fail; }
    if ratio < AC_SD_001_MIN_RATIO { return Sd001Verdict::Fail; }
    Sd001Verdict::Pass
}

// ===========================================================================
// SD-002 — Quantized dispatch: ALL weight tensors have type Q4_K post-conversion
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdTensorType { Q4K, Q6K, Q8K, F16, F32, Bf16 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sd002Verdict { Pass, Fail }

/// Pass iff every tensor type in `weight_tensor_types` is Q4K. Empty
/// inventory fails (no tensors to verify). Any non-Q4K entry indicates
/// the conversion is incomplete and dispatch will fall through to F32.
#[must_use]
pub fn verdict_from_quantized_dispatch(weight_tensor_types: &[SdTensorType]) -> Sd002Verdict {
    if weight_tensor_types.is_empty() { return Sd002Verdict::Fail; }
    for &t in weight_tensor_types {
        if t != SdTensorType::Q4K { return Sd002Verdict::Fail; }
    }
    Sd002Verdict::Pass
}

// ===========================================================================
// SD-003 — Output parity: argmax(safetensors logits) == argmax(gguf logits)
//          for every step of greedy generation
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sd003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_output_parity(
    safetensors_argmax: &[u32],
    gguf_argmax: &[u32],
) -> Sd003Verdict {
    if safetensors_argmax.is_empty() || gguf_argmax.is_empty() { return Sd003Verdict::Fail; }
    if safetensors_argmax.len() != gguf_argmax.len() { return Sd003Verdict::Fail; }
    if safetensors_argmax == gguf_argmax { Sd003Verdict::Pass } else { Sd003Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // SD-001 (throughput parity)
    #[test] fn sd001_pass_above_threshold() {
        // 9.0 / 9.5 ≈ 0.947 ≥ 0.9.
        assert_eq!(verdict_from_throughput_parity(9.0, 9.5), Sd001Verdict::Pass);
    }
    #[test] fn sd001_pass_at_threshold() {
        // 9.0 / 10.0 = 0.9 exactly.
        assert_eq!(verdict_from_throughput_parity(9.0, 10.0), Sd001Verdict::Pass);
    }
    #[test] fn sd001_fail_canonical_regression() {
        // The contract's stated baseline: SafeTensors 6.0 / GGUF 9.5 = 0.63.
        // Pre-fix state must Fail.
        assert_eq!(verdict_from_throughput_parity(6.0, 9.5), Sd001Verdict::Fail);
    }
    #[test] fn sd001_fail_severe_regression() {
        // F32 fallback: SafeTensors 2.4 / GGUF 9.5 ≈ 0.25 (4× memory traffic).
        assert_eq!(verdict_from_throughput_parity(2.4, 9.5), Sd001Verdict::Fail);
    }
    #[test] fn sd001_fail_zero_gguf() {
        assert_eq!(verdict_from_throughput_parity(9.0, 0.0), Sd001Verdict::Fail);
    }
    #[test] fn sd001_fail_nan() {
        assert_eq!(verdict_from_throughput_parity(f32::NAN, 9.5), Sd001Verdict::Fail);
    }

    // SD-002 (quantized dispatch)
    #[test] fn sd002_pass_all_q4k() {
        let types = vec![SdTensorType::Q4K; 339];
        assert_eq!(verdict_from_quantized_dispatch(&types), Sd002Verdict::Pass);
    }
    #[test] fn sd002_fail_partial_f32() {
        // The contract's stated regression: some tensors remain F32.
        let mut types = vec![SdTensorType::Q4K; 338];
        types.push(SdTensorType::F32);
        assert_eq!(verdict_from_quantized_dispatch(&types), Sd002Verdict::Fail);
    }
    #[test] fn sd002_fail_partial_f16() {
        let mut types = vec![SdTensorType::Q4K; 338];
        types.push(SdTensorType::F16);
        assert_eq!(verdict_from_quantized_dispatch(&types), Sd002Verdict::Fail);
    }
    #[test] fn sd002_fail_partial_q6k() {
        // Even Q6K (a quantized type) is not Q4K — incomplete conversion.
        let mut types = vec![SdTensorType::Q4K; 338];
        types.push(SdTensorType::Q6K);
        assert_eq!(verdict_from_quantized_dispatch(&types), Sd002Verdict::Fail);
    }
    #[test] fn sd002_fail_empty() {
        assert_eq!(verdict_from_quantized_dispatch(&[]), Sd002Verdict::Fail);
    }

    // SD-003 (output parity)
    #[test] fn sd003_pass_canonical() {
        let st = [42_u32, 100, 7, 99];
        let gg = [42_u32, 100, 7, 99];
        assert_eq!(verdict_from_output_parity(&st, &gg), Sd003Verdict::Pass);
    }
    #[test] fn sd003_fail_step_divergence() {
        // The contract's stated regression: conversion rounding differs
        // from GGUF quantization → step-level divergence in greedy.
        let st = [42_u32, 100, 7, 99];
        let gg = [42_u32, 100, 8, 99]; // step 2 diverged
        assert_eq!(verdict_from_output_parity(&st, &gg), Sd003Verdict::Fail);
    }
    #[test] fn sd003_fail_length_mismatch() {
        let st = [42_u32];
        let gg = [42_u32, 100];
        assert_eq!(verdict_from_output_parity(&st, &gg), Sd003Verdict::Fail);
    }
    #[test] fn sd003_fail_empty() {
        assert_eq!(verdict_from_output_parity(&[], &[]), Sd003Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_SD_001_MIN_RATIO - 0.9).abs() < 1e-9);
    }
}
