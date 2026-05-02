// Bundles two sister contracts in one verdict module:
//
//   `arch-constraints-v1` (FALSIFY-ARCH-CONSTRAINTS-001..003)
//   `cooperative-matrix-gemm-v1` (FALSIFY-COOP-001..003)
//
// ARCH-CONSTRAINTS-001: every model-family yaml constraints == this contract
// ARCH-CONSTRAINTS-002: enum exhaustiveness — no unknown enum values
// ARCH-CONSTRAINTS-003: DeepSeek eps == 1e-6 (regression guard)
// COOP-001: |coop - tiled| < 1e-3 elementwise
// COOP-002: coop GFLOPS > 2× tiled GFLOPS
// COOP-003: fallback path doesn't crash when coop unavailable

/// ARCH-CONSTRAINTS-003: DeepSeek RmsNorm epsilon exact value.
pub const AC_ARCH_DEEPSEEK_EPS: f32 = 1e-6;
/// COOP-001: parity tolerance for cooperative-matrix vs tiled GEMM.
pub const AC_COOP_PARITY_TOLERANCE: f32 = 1e-3;
/// COOP-002: minimum GFLOPS multiplier (coop must be > 2× tiled).
pub const AC_COOP_GFLOPS_FLOOR: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchCoopVerdict {
    Pass,
    Fail,
}

// ----------------------------------------------------------------
// ARCH-CONSTRAINTS-001..003
// ----------------------------------------------------------------

/// ARCH-CONSTRAINTS-001: model-family yaml consistency.
///
/// `mismatch_count` = number of (family, field) pairs where the
/// model-family YAML doesn't match this contract.
#[must_use]
pub fn verdict_from_family_consistency(mismatch_count: u32) -> ArchCoopVerdict {
    if mismatch_count == 0 {
        ArchCoopVerdict::Pass
    } else {
        ArchCoopVerdict::Fail
    }
}

/// ARCH-CONSTRAINTS-002: enum exhaustiveness — no unknown values.
#[must_use]
pub fn verdict_from_enum_exhaustive(unknown_enum_count: u32) -> ArchCoopVerdict {
    if unknown_enum_count == 0 {
        ArchCoopVerdict::Pass
    } else {
        ArchCoopVerdict::Fail
    }
}

/// ARCH-CONSTRAINTS-003: DeepSeek eps regression guard.
///
/// Pass iff `eps == 1e-6` exactly.
#[must_use]
pub fn verdict_from_deepseek_eps(eps: f32) -> ArchCoopVerdict {
    if !eps.is_finite() {
        return ArchCoopVerdict::Fail;
    }
    if eps == AC_ARCH_DEEPSEEK_EPS {
        ArchCoopVerdict::Pass
    } else {
        ArchCoopVerdict::Fail
    }
}

// ----------------------------------------------------------------
// COOP-001..003
// ----------------------------------------------------------------

/// COOP-001: max|coop - tiled| < AC_COOP_PARITY_TOLERANCE elementwise.
#[must_use]
pub fn verdict_from_coop_parity(coop: &[f32], tiled: &[f32]) -> ArchCoopVerdict {
    if coop.is_empty() || coop.len() != tiled.len() {
        return ArchCoopVerdict::Fail;
    }
    for (a, b) in coop.iter().zip(tiled.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return ArchCoopVerdict::Fail;
        }
        if (a - b).abs() >= AC_COOP_PARITY_TOLERANCE {
            return ArchCoopVerdict::Fail;
        }
    }
    ArchCoopVerdict::Pass
}

/// COOP-002: coop GFLOPS > 2× tiled GFLOPS.
#[must_use]
pub fn verdict_from_coop_throughput(coop_gflops: f32, tiled_gflops: f32) -> ArchCoopVerdict {
    if !coop_gflops.is_finite() || !tiled_gflops.is_finite() {
        return ArchCoopVerdict::Fail;
    }
    if tiled_gflops <= 0.0 || coop_gflops <= 0.0 {
        return ArchCoopVerdict::Fail;
    }
    if coop_gflops > AC_COOP_GFLOPS_FLOOR * tiled_gflops {
        ArchCoopVerdict::Pass
    } else {
        ArchCoopVerdict::Fail
    }
}

/// COOP-003: fallback path takes effect on unsupported hardware.
///
/// Pass iff:
///   - coop_supported == true → used_coop, no crash
///   - coop_supported == false → used_tiled fallback, no crash
#[must_use]
#[allow(clippy::fn_params_excessive_bools)]
pub fn verdict_from_coop_fallback(
    coop_supported: bool,
    used_coop_path: bool,
    used_tiled_fallback: bool,
    crashed: bool,
) -> ArchCoopVerdict {
    if crashed {
        return ArchCoopVerdict::Fail;
    }
    if coop_supported && used_coop_path && !used_tiled_fallback {
        return ArchCoopVerdict::Pass;
    }
    if !coop_supported && !used_coop_path && used_tiled_fallback {
        return ArchCoopVerdict::Pass;
    }
    ArchCoopVerdict::Fail
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_ARCH_DEEPSEEK_EPS, 1e-6);
        assert_eq!(AC_COOP_PARITY_TOLERANCE, 1e-3);
        assert_eq!(AC_COOP_GFLOPS_FLOOR, 2.0);
    }

    // -----------------------------------------------------------------
    // Section 2: ARCH-CONSTRAINTS-001..003.
    // -----------------------------------------------------------------
    #[test]
    fn farch001_pass_no_mismatch() {
        let v = verdict_from_family_consistency(0);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn farch001_fail_one_mismatch() {
        let v = verdict_from_family_consistency(1);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn farch002_pass_no_unknown() {
        let v = verdict_from_enum_exhaustive(0);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn farch002_fail_unknown_present() {
        let v = verdict_from_enum_exhaustive(3);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn farch003_pass_correct_eps() {
        let v = verdict_from_deepseek_eps(1e-6);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn farch003_fail_old_default() {
        let v = verdict_from_deepseek_eps(1e-5);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn farch003_fail_nan() {
        let v = verdict_from_deepseek_eps(f32::NAN);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: COOP-001 parity.
    // -----------------------------------------------------------------
    #[test]
    fn fcoop001_pass_within_tolerance() {
        let coop = vec![1.0_f32, 2.0, 3.0];
        let tiled = vec![1.0001, 2.0001, 2.9999];
        let v = verdict_from_coop_parity(&coop, &tiled);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn fcoop001_fail_drift() {
        let coop = vec![1.0_f32, 2.0];
        let tiled = vec![1.0, 2.5];
        let v = verdict_from_coop_parity(&coop, &tiled);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn fcoop001_fail_length_mismatch() {
        let coop = vec![1.0_f32, 2.0];
        let tiled = vec![1.0_f32];
        let v = verdict_from_coop_parity(&coop, &tiled);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn fcoop001_fail_nan() {
        let coop = vec![1.0_f32, f32::NAN];
        let tiled = vec![1.0_f32, 2.0];
        let v = verdict_from_coop_parity(&coop, &tiled);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: COOP-002 throughput.
    // -----------------------------------------------------------------
    #[test]
    fn fcoop002_pass_3x() {
        let v = verdict_from_coop_throughput(300.0, 100.0);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn fcoop002_fail_at_2x() {
        // strict >, not >=
        let v = verdict_from_coop_throughput(200.0, 100.0);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn fcoop002_fail_below_2x() {
        let v = verdict_from_coop_throughput(150.0, 100.0);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn fcoop002_fail_zero_tiled() {
        let v = verdict_from_coop_throughput(300.0, 0.0);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: COOP-003 fallback.
    // -----------------------------------------------------------------
    #[test]
    fn fcoop003_pass_supported_used_coop() {
        let v = verdict_from_coop_fallback(true, true, false, false);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn fcoop003_pass_unsupported_used_tiled() {
        let v = verdict_from_coop_fallback(false, false, true, false);
        assert_eq!(v, ArchCoopVerdict::Pass);
    }

    #[test]
    fn fcoop003_fail_crash() {
        let v = verdict_from_coop_fallback(false, false, false, true);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn fcoop003_fail_supported_but_used_tiled() {
        let v = verdict_from_coop_fallback(true, false, true, false);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    #[test]
    fn fcoop003_fail_unsupported_used_coop() {
        // The exact regression — would crash on real hardware.
        let v = verdict_from_coop_fallback(false, true, false, false);
        assert_eq!(v, ArchCoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_002_throughput_band() {
        for ratio_x10 in [10_u32, 19, 20, 21, 25, 30] {
            let coop = (ratio_x10 as f32 / 10.0) * 100.0;
            let v = verdict_from_coop_throughput(coop, 100.0);
            let want = if coop > 200.0 {
                ArchCoopVerdict::Pass
            } else {
                ArchCoopVerdict::Fail
            };
            assert_eq!(v, want, "ratio={ratio_x10}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_6() {
        let v1 = verdict_from_family_consistency(0);
        let v2 = verdict_from_enum_exhaustive(0);
        let v3 = verdict_from_deepseek_eps(1e-6);
        let coop = vec![1.0_f32, 2.0];
        let tiled = vec![1.0001, 2.0001];
        let v4 = verdict_from_coop_parity(&coop, &tiled);
        let v5 = verdict_from_coop_throughput(300.0, 100.0);
        let v6 = verdict_from_coop_fallback(true, true, false, false);
        for v in [v1, v2, v3, v4, v5, v6] {
            assert_eq!(v, ArchCoopVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        let v1 = verdict_from_family_consistency(7);
        let v2 = verdict_from_enum_exhaustive(2);
        let v3 = verdict_from_deepseek_eps(1e-5); // pre-fix value
        let coop = vec![1.0_f32, 2.0];
        let tiled = vec![1.0, 2.5];
        let v4 = verdict_from_coop_parity(&coop, &tiled);
        let v5 = verdict_from_coop_throughput(150.0, 100.0); // not >2×
        let v6 = verdict_from_coop_fallback(false, true, false, false); // crash class
        for v in [v1, v2, v3, v4, v5, v6] {
            assert_eq!(v, ArchCoopVerdict::Fail);
        }
    }
}
