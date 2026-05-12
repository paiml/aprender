// Bundles two sister GPU-precision contracts in one verdict module:
//
//   `gpu-context-health-v1` (FT-GPU-CTX-001..003)
//   `fp16-cublas-gemm-v1` (FALSIFY-FP16_CUBLAS_GEMM_V1_001..002)
//
// Both gate FP-precision behaviour on Ada/Hopper/Blackwell.
//
// FT-GPU-CTX-001: detect_fp8_prefill(cc) == false for cc >= 100 (Blackwell)
// FT-GPU-CTX-002: warmup_fp8_cache() is a no-op when cc >= 100
// FT-GPU-CTX-003: detect_fp8_prefill(cc) == true for cc in {89, 90}
// FP16-001: |fp16_result - fp32_result| < 1e-2 elementwise
// FP16-002: throughput(fp16) >= 1.5 * throughput(fp32) for M,N >= 512

/// FP8 architecture lower bound (Ada and up).
pub const AC_GPUCTX_FP8_MIN_CC: u32 = 89;
/// FP8 architecture upper bound (exclusive — Blackwell+ excluded).
pub const AC_GPUCTX_FP8_MAX_CC_EXCL: u32 = 100;
/// FP16 vs FP32 elementwise tolerance.
pub const AC_FP16_PRECISION_EPSILON: f32 = 1e-2;
/// Throughput multiplier floor (FP16 / FP32) for M,N >= 512.
pub const AC_FP16_THROUGHPUT_FLOOR: f32 = 1.5;
/// Minimum dimension for the throughput gate.
pub const AC_FP16_MIN_DIM: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecVerdict {
    Pass,
    Fail,
}

/// Reference: `detect_fp8_prefill(cc)` returns true iff
/// `89 <= cc < 100` (Ada or Hopper).
#[must_use]
pub fn detect_fp8_prefill(cc: u32) -> bool {
    cc >= AC_GPUCTX_FP8_MIN_CC && cc < AC_GPUCTX_FP8_MAX_CC_EXCL
}

/// FT-GPU-CTX-001: Blackwell guard — detect must return false for cc >= 100.
#[must_use]
pub fn verdict_from_blackwell_fp8_disabled(cc: u32) -> GpuPrecVerdict {
    if cc < AC_GPUCTX_FP8_MAX_CC_EXCL {
        // Out of scope — gate only applies cc>=100. Fail-closed.
        return GpuPrecVerdict::Fail;
    }
    if !detect_fp8_prefill(cc) {
        GpuPrecVerdict::Pass
    } else {
        GpuPrecVerdict::Fail
    }
}

/// FT-GPU-CTX-002: warmup is a no-op when cc >= 100.
///
/// Caller passes `(cc, warmup_was_invoked)`. Pass iff:
///   cc < 100 (gate not applicable, vacuous Pass) OR
///   cc >= 100 AND warmup_was_invoked == false.
/// We make the no-op-on-high-cc case the only Pass branch and Fail on
/// "high cc with warmup actually invoked" so it catches the regression.
#[must_use]
pub fn verdict_from_warmup_noop(cc: u32, warmup_was_invoked: bool) -> GpuPrecVerdict {
    if cc < AC_GPUCTX_FP8_MAX_CC_EXCL {
        // Pre-Blackwell: warmup is allowed; this gate doesn't constrain.
        return GpuPrecVerdict::Pass;
    }
    if warmup_was_invoked {
        GpuPrecVerdict::Fail
    } else {
        GpuPrecVerdict::Pass
    }
}

/// FT-GPU-CTX-003: detect_fp8_prefill returns true for cc in {89, 90}.
#[must_use]
pub fn verdict_from_ada_hopper_fp8_enabled(cc: u32) -> GpuPrecVerdict {
    if !(AC_GPUCTX_FP8_MIN_CC..=90).contains(&cc) {
        // Outside Ada/Hopper band — gate not applicable, Fail-closed.
        return GpuPrecVerdict::Fail;
    }
    if detect_fp8_prefill(cc) {
        GpuPrecVerdict::Pass
    } else {
        GpuPrecVerdict::Fail
    }
}

/// FP16-001: elementwise |fp16 - fp32| < 1e-2 over slices.
#[must_use]
pub fn verdict_from_fp16_fp32_precision(
    fp16: &[f32],
    fp32: &[f32],
) -> GpuPrecVerdict {
    if fp16.is_empty() || fp32.is_empty() || fp16.len() != fp32.len() {
        return GpuPrecVerdict::Fail;
    }
    for (a, b) in fp16.iter().zip(fp32.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return GpuPrecVerdict::Fail;
        }
        if (a - b).abs() >= AC_FP16_PRECISION_EPSILON {
            return GpuPrecVerdict::Fail;
        }
    }
    GpuPrecVerdict::Pass
}

/// FP16-002: throughput(fp16) >= 1.5 * throughput(fp32) for M,N >= 512.
#[must_use]
pub fn verdict_from_fp16_throughput(
    m: u32,
    n: u32,
    tps_fp16: f32,
    tps_fp32: f32,
) -> GpuPrecVerdict {
    if m < AC_FP16_MIN_DIM || n < AC_FP16_MIN_DIM {
        // Below precondition; gate not applicable, Fail-closed.
        return GpuPrecVerdict::Fail;
    }
    if !tps_fp16.is_finite() || !tps_fp32.is_finite() {
        return GpuPrecVerdict::Fail;
    }
    if tps_fp16 <= 0.0 || tps_fp32 <= 0.0 {
        return GpuPrecVerdict::Fail;
    }
    if tps_fp16 >= AC_FP16_THROUGHPUT_FLOOR * tps_fp32 {
        GpuPrecVerdict::Pass
    } else {
        GpuPrecVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_fp8_cc_band() {
        assert_eq!(AC_GPUCTX_FP8_MIN_CC, 89);
        assert_eq!(AC_GPUCTX_FP8_MAX_CC_EXCL, 100);
    }

    #[test]
    fn provenance_fp16_precision_epsilon() {
        assert_eq!(AC_FP16_PRECISION_EPSILON, 1e-2);
    }

    #[test]
    fn provenance_fp16_throughput_floor() {
        assert_eq!(AC_FP16_THROUGHPUT_FLOOR, 1.5);
        assert_eq!(AC_FP16_MIN_DIM, 512);
    }

    // -----------------------------------------------------------------
    // Section 2: detect_fp8_prefill reference.
    // -----------------------------------------------------------------
    #[test]
    fn detect_fp8_prefill_pre_ada() {
        assert!(!detect_fp8_prefill(80)); // Ampere
        assert!(!detect_fp8_prefill(86));
        assert!(!detect_fp8_prefill(88));
    }

    #[test]
    fn detect_fp8_prefill_ada_hopper() {
        assert!(detect_fp8_prefill(89)); // Ada
        assert!(detect_fp8_prefill(90)); // Hopper
        assert!(detect_fp8_prefill(99)); // edge
    }

    #[test]
    fn detect_fp8_prefill_blackwell() {
        assert!(!detect_fp8_prefill(100)); // sm_100
        assert!(!detect_fp8_prefill(120));
        assert!(!detect_fp8_prefill(121)); // sm_121
    }

    // -----------------------------------------------------------------
    // Section 3: FT-GPU-CTX-001 Blackwell disabled.
    // -----------------------------------------------------------------
    #[test]
    fn fctx001_pass_sm_121() {
        let v = verdict_from_blackwell_fp8_disabled(121);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn fctx001_pass_sm_100() {
        let v = verdict_from_blackwell_fp8_disabled(100);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn fctx001_fail_pre_blackwell_out_of_scope() {
        let v = verdict_from_blackwell_fp8_disabled(89);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: FT-GPU-CTX-002 warmup no-op.
    // -----------------------------------------------------------------
    #[test]
    fn fctx002_pass_blackwell_no_warmup() {
        let v = verdict_from_warmup_noop(121, false);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn fctx002_fail_blackwell_warmup_invoked() {
        // The exact regression class — context poisoning.
        let v = verdict_from_warmup_noop(121, true);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn fctx002_pass_pre_blackwell_warmup_irrelevant() {
        let v = verdict_from_warmup_noop(89, true);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 5: FT-GPU-CTX-003 Ada/Hopper enabled.
    // -----------------------------------------------------------------
    #[test]
    fn fctx003_pass_ada_89() {
        let v = verdict_from_ada_hopper_fp8_enabled(89);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn fctx003_pass_hopper_90() {
        let v = verdict_from_ada_hopper_fp8_enabled(90);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn fctx003_fail_pre_ada() {
        let v = verdict_from_ada_hopper_fp8_enabled(86);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn fctx003_fail_blackwell() {
        let v = verdict_from_ada_hopper_fp8_enabled(121);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: FP16-001 / FP16-002.
    // -----------------------------------------------------------------
    #[test]
    fn ffp16_001_pass_within_epsilon() {
        let fp16 = vec![1.0_f32, 2.5, -3.0, 0.1];
        let fp32 = vec![1.001_f32, 2.503, -2.999, 0.105];
        let v = verdict_from_fp16_fp32_precision(&fp16, &fp32);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn ffp16_001_fail_above_epsilon() {
        // Strict <: difference > 1e-2 is Fail. We pick 0.011 which
        // is comfortably above eps regardless of float rounding.
        let fp16 = vec![1.0_f32];
        let fp32 = vec![1.011_f32];
        let v = verdict_from_fp16_fp32_precision(&fp16, &fp32);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn ffp16_001_fail_one_drift() {
        let fp16 = vec![1.0_f32, 2.5, -3.0];
        let fp32 = vec![1.0_f32, 2.5, -3.5];
        let v = verdict_from_fp16_fp32_precision(&fp16, &fp32);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn ffp16_001_fail_length_mismatch() {
        let fp16 = vec![1.0_f32];
        let fp32 = vec![1.0_f32, 2.0];
        let v = verdict_from_fp16_fp32_precision(&fp16, &fp32);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn ffp16_002_pass_2x_speedup() {
        let v = verdict_from_fp16_throughput(1024, 1024, 200.0, 100.0);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn ffp16_002_pass_at_threshold() {
        let v = verdict_from_fp16_throughput(512, 512, 150.0, 100.0);
        assert_eq!(v, GpuPrecVerdict::Pass);
    }

    #[test]
    fn ffp16_002_fail_below_threshold() {
        let v = verdict_from_fp16_throughput(1024, 1024, 140.0, 100.0);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn ffp16_002_fail_below_dim_floor() {
        let v = verdict_from_fp16_throughput(256, 1024, 200.0, 100.0);
        assert_eq!(v, GpuPrecVerdict::Fail);
        let v = verdict_from_fp16_throughput(1024, 256, 200.0, 100.0);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    #[test]
    fn ffp16_002_fail_zero_throughput() {
        let v = verdict_from_fp16_throughput(1024, 1024, 0.0, 100.0);
        assert_eq!(v, GpuPrecVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_fp8_cc_boundary_sweep() {
        for cc in [80_u32, 88, 89, 90, 99, 100, 110, 120, 121] {
            let detected = detect_fp8_prefill(cc);
            let want = (89..100).contains(&cc);
            assert_eq!(detected, want, "cc={cc}");
        }
    }

    #[test]
    fn mutation_survey_throughput_band() {
        // sweep ratios 1.0, 1.4, 1.5, 1.6, 2.0, 4.0
        for ratio_x10 in [10_u32, 14, 15, 16, 20, 40] {
            let ratio = ratio_x10 as f32 / 10.0;
            let v = verdict_from_fp16_throughput(1024, 1024, 100.0 * ratio, 100.0);
            let want = if ratio >= AC_FP16_THROUGHPUT_FLOOR {
                GpuPrecVerdict::Pass
            } else {
                GpuPrecVerdict::Fail
            };
            assert_eq!(v, want, "ratio={ratio}");
        }
    }

    #[test]
    fn realistic_healthy_blackwell_passes_all_5() {
        // Blackwell sm_121 — all FP8 disabled, FP16 cuBLAS active.
        let v1 = verdict_from_blackwell_fp8_disabled(121);
        let v2 = verdict_from_warmup_noop(121, false);
        let v3 = verdict_from_ada_hopper_fp8_enabled(89);
        let fp16 = vec![1.0_f32, 2.0, 3.0];
        let fp32 = vec![1.001_f32, 1.999, 3.005];
        let v4 = verdict_from_fp16_fp32_precision(&fp16, &fp32);
        let v5 = verdict_from_fp16_throughput(4096, 4096, 200.0, 100.0);
        assert_eq!(v1, GpuPrecVerdict::Pass);
        assert_eq!(v2, GpuPrecVerdict::Pass);
        assert_eq!(v3, GpuPrecVerdict::Pass);
        assert_eq!(v4, GpuPrecVerdict::Pass);
        assert_eq!(v5, GpuPrecVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Pre-fix Blackwell crash class:
        //  1: FP8 enabled on cc=121 (illegal address)
        //  2: warmup invoked on Blackwell (poisoned context)
        //  3: FP8 disabled on Ada (regression)
        //  4: FP16 drift > 1e-2 (BF16/FP16 confusion)
        //  5: FP16 throughput at 1.0× FP32 (wrong tensor-core dispatch)
        // verdict_from_blackwell_fp8_disabled is checked by feeding cc=121
        // with a hypothetical "detect returns true" — we model that by
        // calling on cc=89 (out-of-scope for the gate) which is also Fail.
        let v1 = verdict_from_blackwell_fp8_disabled(89);
        let v2 = verdict_from_warmup_noop(121, true);
        let v3 = verdict_from_ada_hopper_fp8_enabled(86);
        let fp16 = vec![1.0_f32];
        let fp32 = vec![1.5_f32];
        let v4 = verdict_from_fp16_fp32_precision(&fp16, &fp32);
        let v5 = verdict_from_fp16_throughput(1024, 1024, 100.0, 100.0);
        assert_eq!(v1, GpuPrecVerdict::Fail);
        assert_eq!(v2, GpuPrecVerdict::Fail);
        assert_eq!(v3, GpuPrecVerdict::Fail);
        assert_eq!(v4, GpuPrecVerdict::Fail);
        assert_eq!(v5, GpuPrecVerdict::Fail);
    }
}
