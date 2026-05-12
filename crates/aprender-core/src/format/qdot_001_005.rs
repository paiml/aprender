// `quantized-dot-product-v1` algorithm-level PARTIAL discharge for
// FALSIFY-QDOT-001..005.
//
// Contract: `contracts/quantized-dot-product-v1.yaml`.
//
// Pure-Rust verdicts for the 5 falsification gates:
//   QDOT-001: SIMD kernel matches scalar reference within ULP tolerance
//   QDOT-002: cross-format isolation — wrong-format kernel produces garbage
//   QDOT-003: precomputed bsums == on-the-fly bsums (exact integer equality)
//   QDOT-004: format registry / trait-impl symmetry (no orphans, no ghosts)
//   QDOT-005: dispatch table is exhaustive — every format has ≥ scalar entry

use std::collections::HashSet;

/// QDOT-001 numerical tolerance (ULPs of f32).
/// Per Q4_K/Q6_K accumulation budget — 4 ULP * f32::EPSILON per dot.
pub const AC_QDOT_ULP_TOLERANCE: u32 = 4;

/// QDOT-002 garbage-divergence threshold: wrong-format result must
/// differ from correct by >100×.
pub const AC_QDOT_CROSS_FORMAT_GARBAGE_FACTOR: f32 = 100.0;

/// QDOT-005 minimum dispatch key per format ("scalar").
pub const AC_QDOT_DISPATCH_MIN_KEY: &str = "scalar";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QdotVerdict {
    Pass,
    Fail,
}

/// QDOT-001: SIMD vs scalar within `ulp_tol * f32::EPSILON * |scalar|`.
///
/// Pass iff:
///   `|simd - scalar| <= ulp_tol * f32::EPSILON * |scalar|.max(1.0)`
/// Non-finite either side → Fail.
#[must_use]
pub fn verdict_from_simd_vs_scalar(simd: f32, scalar: f32, ulp_tol: u32) -> QdotVerdict {
    if !simd.is_finite() || !scalar.is_finite() {
        return QdotVerdict::Fail;
    }
    let bound = (ulp_tol as f32) * f32::EPSILON * scalar.abs().max(1.0);
    if (simd - scalar).abs() <= bound {
        QdotVerdict::Pass
    } else {
        QdotVerdict::Fail
    }
}

/// QDOT-002: cross-format kernel produces garbage.
///
/// Pass iff `|wrong_kernel_result - correct| > AC_QDOT_CROSS_FORMAT_GARBAGE_FACTOR * |correct|.max(1.0)`.
/// I.e. routing Q6K bytes through a Q4K kernel must NOT accidentally
/// produce a near-correct answer.
#[must_use]
pub fn verdict_from_cross_format_isolation(
    wrong_kernel_result: f32,
    correct: f32,
) -> QdotVerdict {
    if !wrong_kernel_result.is_finite() || !correct.is_finite() {
        return QdotVerdict::Fail;
    }
    let scale = correct.abs().max(1.0);
    let divergence = (wrong_kernel_result - correct).abs();
    if divergence > AC_QDOT_CROSS_FORMAT_GARBAGE_FACTOR * scale {
        QdotVerdict::Pass
    } else {
        QdotVerdict::Fail
    }
}

/// QDOT-003: precomputed bsums equal on-the-fly bsums.
///
/// Both inputs are integer (i32 to preserve sub-block accumulator
/// semantics). Pass iff exact equality AND non-empty AND lengths match.
#[must_use]
pub fn verdict_from_bsum_equality(precomputed: &[i32], on_the_fly: &[i32]) -> QdotVerdict {
    if precomputed.is_empty() || on_the_fly.is_empty() {
        return QdotVerdict::Fail;
    }
    if precomputed.len() != on_the_fly.len() {
        return QdotVerdict::Fail;
    }
    if precomputed == on_the_fly {
        QdotVerdict::Pass
    } else {
        QdotVerdict::Fail
    }
}

/// QDOT-004: format registry and trait-impl set are equal.
///
/// `yaml_formats` and `impl_formats` are the format-id sets. Pass iff
/// they contain exactly the same names (no orphans, no ghosts).
#[must_use]
pub fn verdict_from_registry_symmetry(
    yaml_formats: &[&str],
    impl_formats: &[&str],
) -> QdotVerdict {
    if yaml_formats.is_empty() && impl_formats.is_empty() {
        // Vacuous Pass would mask "registry deleted" regressions.
        return QdotVerdict::Fail;
    }
    let yaml: HashSet<&str> = yaml_formats.iter().copied().collect();
    let imp: HashSet<&str> = impl_formats.iter().copied().collect();
    if yaml == imp {
        QdotVerdict::Pass
    } else {
        QdotVerdict::Fail
    }
}

/// QDOT-005: every format in registry has at least the `scalar` entry
/// in `simd_dispatch`.
///
/// `dispatch_keys_by_format` is `&[(format, &[keys])]`.
/// Pass iff every format's keys contains `AC_QDOT_DISPATCH_MIN_KEY`.
#[must_use]
pub fn verdict_from_dispatch_exhaustive(
    dispatch_keys_by_format: &[(&str, &[&str])],
) -> QdotVerdict {
    if dispatch_keys_by_format.is_empty() {
        return QdotVerdict::Fail;
    }
    for (_format, keys) in dispatch_keys_by_format {
        if !keys.contains(&AC_QDOT_DISPATCH_MIN_KEY) {
            return QdotVerdict::Fail;
        }
    }
    QdotVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_ulp_tolerance_4() {
        assert_eq!(AC_QDOT_ULP_TOLERANCE, 4);
    }

    #[test]
    fn provenance_cross_format_factor_100() {
        assert_eq!(AC_QDOT_CROSS_FORMAT_GARBAGE_FACTOR, 100.0);
    }

    #[test]
    fn provenance_dispatch_min_key_scalar() {
        assert_eq!(AC_QDOT_DISPATCH_MIN_KEY, "scalar");
    }

    // -----------------------------------------------------------------
    // Section 2: QDOT-001 SIMD vs scalar.
    // -----------------------------------------------------------------
    #[test]
    fn fqdot001_pass_exact_match() {
        let v = verdict_from_simd_vs_scalar(1.0, 1.0, AC_QDOT_ULP_TOLERANCE);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot001_pass_within_ulp() {
        // 1.0 + 1 ULP is well within 4-ULP budget.
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let v = verdict_from_simd_vs_scalar(bumped, 1.0, AC_QDOT_ULP_TOLERANCE);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot001_fail_far_off() {
        let v = verdict_from_simd_vs_scalar(1.0, 1.5, AC_QDOT_ULP_TOLERANCE);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot001_fail_nan() {
        let v = verdict_from_simd_vs_scalar(f32::NAN, 1.0, AC_QDOT_ULP_TOLERANCE);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot001_pass_at_zero() {
        let v = verdict_from_simd_vs_scalar(0.0, 0.0, AC_QDOT_ULP_TOLERANCE);
        assert_eq!(v, QdotVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 3: QDOT-002 cross-format isolation.
    // -----------------------------------------------------------------
    #[test]
    fn fqdot002_pass_garbage_difference_200x() {
        let v = verdict_from_cross_format_isolation(200.0, 1.0);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot002_fail_close_match() {
        let v = verdict_from_cross_format_isolation(1.01, 1.0);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot002_at_strict_threshold() {
        // Spec says ">100×". |observed-correct| must STRICTLY exceed
        // 100*|correct|.max(1.0). Equality at boundary is Fail.
        // |101-1|=100, not strictly >100 → Fail.
        let v = verdict_from_cross_format_isolation(101.0, 1.0);
        assert_eq!(v, QdotVerdict::Fail);
        // |202-1|=201 > 100 → Pass.
        let v = verdict_from_cross_format_isolation(202.0, 1.0);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot002_fail_nan() {
        let v = verdict_from_cross_format_isolation(f32::NAN, 1.0);
        assert_eq!(v, QdotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: QDOT-003 bsum equality.
    // -----------------------------------------------------------------
    #[test]
    fn fqdot003_pass_exact_match() {
        let v = verdict_from_bsum_equality(&[10, 20, 30, 40], &[10, 20, 30, 40]);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot003_fail_one_off() {
        let v = verdict_from_bsum_equality(&[10, 20, 30, 41], &[10, 20, 30, 40]);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot003_fail_length_mismatch() {
        let v = verdict_from_bsum_equality(&[10, 20, 30], &[10, 20, 30, 40]);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot003_fail_empty() {
        let v = verdict_from_bsum_equality(&[], &[]);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot003_pass_negative_bsums() {
        let v = verdict_from_bsum_equality(&[-1, -100, 0, 50], &[-1, -100, 0, 50]);
        assert_eq!(v, QdotVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 5: QDOT-004 registry symmetry.
    // -----------------------------------------------------------------
    #[test]
    fn fqdot004_pass_exact_match() {
        let yaml = ["Q4_K", "Q6_K", "Q8_0"];
        let imp = ["Q4_K", "Q6_K", "Q8_0"];
        let v = verdict_from_registry_symmetry(&yaml, &imp);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot004_pass_unordered() {
        let yaml = ["Q4_K", "Q6_K", "Q8_0"];
        let imp = ["Q8_0", "Q4_K", "Q6_K"];
        let v = verdict_from_registry_symmetry(&yaml, &imp);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot004_fail_orphan_impl() {
        // Q4_1 in code but not YAML
        let yaml = ["Q4_K", "Q6_K"];
        let imp = ["Q4_K", "Q6_K", "Q4_1"];
        let v = verdict_from_registry_symmetry(&yaml, &imp);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot004_fail_ghost_yaml() {
        // Q5_K in YAML but not code
        let yaml = ["Q4_K", "Q6_K", "Q5_K"];
        let imp = ["Q4_K", "Q6_K"];
        let v = verdict_from_registry_symmetry(&yaml, &imp);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot004_fail_both_empty() {
        // Vacuous Pass would mask "registry deleted" — Fail.
        let v = verdict_from_registry_symmetry(&[], &[]);
        assert_eq!(v, QdotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: QDOT-005 dispatch exhaustive.
    // -----------------------------------------------------------------
    #[test]
    fn fqdot005_pass_all_have_scalar() {
        let dispatch: &[(&str, &[&str])] = &[
            ("Q4_K", &["scalar", "avx2"]),
            ("Q6_K", &["scalar"]),
            ("Q8_0", &["scalar", "avx512"]),
        ];
        let v = verdict_from_dispatch_exhaustive(dispatch);
        assert_eq!(v, QdotVerdict::Pass);
    }

    #[test]
    fn fqdot005_fail_one_missing_scalar() {
        let dispatch: &[(&str, &[&str])] = &[
            ("Q4_K", &["scalar"]),
            ("Q6_K", &["avx2"]), // missing scalar
        ];
        let v = verdict_from_dispatch_exhaustive(dispatch);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot005_fail_empty() {
        let v = verdict_from_dispatch_exhaustive(&[]);
        assert_eq!(v, QdotVerdict::Fail);
    }

    #[test]
    fn fqdot005_fail_format_with_no_keys() {
        let dispatch: &[(&str, &[&str])] = &[("Q4_K", &[])];
        let v = verdict_from_dispatch_exhaustive(dispatch);
        assert_eq!(v, QdotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation survey + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_ulp_band() {
        // Bump scalar by k ULPs, verify only k <= ulp_tol passes.
        let scalar = 1.5_f32;
        for k in 0_u32..8 {
            let bumped = f32::from_bits(scalar.to_bits() + k);
            let v = verdict_from_simd_vs_scalar(bumped, scalar, AC_QDOT_ULP_TOLERANCE);
            // 1.5 has |scalar|=1.5 → bound = 4 * eps * 1.5 ≈ 7.15e-7.
            // Each ULP at 1.5 ≈ 1.19e-7. So 4 ULPs ≈ 4.77e-7 < 7.15e-7,
            // 6 ULPs ≈ 7.15e-7 ≈ bound (boundary), 7 ULPs > bound.
            // We assert: k<=4 always passes, k>=7 always fails.
            if k <= 4 {
                assert_eq!(v, QdotVerdict::Pass, "k={k} (within ULP budget)");
            }
            if k >= 7 {
                assert_eq!(v, QdotVerdict::Fail, "k={k} (above bound)");
            }
        }
    }

    #[test]
    fn mutation_survey_004_subset_drift() {
        // Add and remove one element; both must Fail.
        let yaml = ["Q4_K", "Q6_K", "Q8_0"];
        let imp_added = ["Q4_K", "Q6_K", "Q8_0", "Q5_K"];
        let imp_removed = ["Q4_K", "Q6_K"];
        assert_eq!(
            verdict_from_registry_symmetry(&yaml, &imp_added),
            QdotVerdict::Fail
        );
        assert_eq!(
            verdict_from_registry_symmetry(&yaml, &imp_removed),
            QdotVerdict::Fail
        );
    }

    #[test]
    fn realistic_healthy_quant_dot_passes_all_5() {
        let v1 = verdict_from_simd_vs_scalar(1.0000001, 1.0, AC_QDOT_ULP_TOLERANCE);
        let v2 = verdict_from_cross_format_isolation(1000.0, 5.0);
        let v3 = verdict_from_bsum_equality(&[10, 20, 30], &[10, 20, 30]);
        let v4 = verdict_from_registry_symmetry(&["Q4_K", "Q6_K", "Q8_0"], &["Q4_K", "Q6_K", "Q8_0"]);
        let v5 = verdict_from_dispatch_exhaustive(&[
            ("Q4_K", &["scalar", "avx2"]),
            ("Q6_K", &["scalar"]),
            ("Q8_0", &["scalar"]),
        ]);
        assert_eq!(v1, QdotVerdict::Pass);
        assert_eq!(v2, QdotVerdict::Pass);
        assert_eq!(v3, QdotVerdict::Pass);
        assert_eq!(v4, QdotVerdict::Pass);
        assert_eq!(v5, QdotVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Regression class:
        //   1: SIMD wrong by 0.5 (sign error in nibble extraction)
        //   2: cross-format kernel produced near-correct (Q4K read as Q6K)
        //   3: bsum off-by-one (sub-block boundary misaligned)
        //   4: orphan in trait-impl (Q5_K added to code, not YAML)
        //   5: dispatch missing scalar entry for Q6_K
        let v1 = verdict_from_simd_vs_scalar(1.5, 1.0, AC_QDOT_ULP_TOLERANCE);
        let v2 = verdict_from_cross_format_isolation(5.05, 5.0);
        let v3 = verdict_from_bsum_equality(&[10, 21, 30], &[10, 20, 30]);
        let v4 = verdict_from_registry_symmetry(&["Q4_K", "Q6_K"], &["Q4_K", "Q6_K", "Q5_K"]);
        let v5 = verdict_from_dispatch_exhaustive(&[("Q4_K", &["scalar"]), ("Q6_K", &["avx2"])]);
        assert_eq!(v1, QdotVerdict::Fail);
        assert_eq!(v2, QdotVerdict::Fail);
        assert_eq!(v3, QdotVerdict::Fail);
        assert_eq!(v4, QdotVerdict::Fail);
        assert_eq!(v5, QdotVerdict::Fail);
    }
}
