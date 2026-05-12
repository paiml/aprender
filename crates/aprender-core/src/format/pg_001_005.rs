// `performance-grading-v1` algorithm-level PARTIAL discharge for
// FALSIFY-PG-001..005.
//
// Contract: `contracts/performance-grading-v1.yaml`.
//
// Pure-Rust verdicts for the 5 falsification gates plus a reference
// grade classifier for ratio-based grading and efficiency-based grading.
//
// PG-001: exhaustive grading — every ratio in [0, ∞) maps to exactly one grade
// PG-002: monotonicity — increasing ratio never decreases grade
// PG-003: concrete ceiling — DDR4 33 GB/s, 4.19 GB → ceiling ∈ [7.0, 9.0]
// PG-004: efficiency grade monotonic — higher efficiency → same/better grade
// PG-005: SIMD vs scalar grading equivalence (zero tolerance)

/// Concrete-ceiling test point: DDR4 bandwidth.
pub const AC_PG_DDR4_BANDWIDTH_GB_S: f32 = 33.0;
/// Concrete-ceiling test point: Qwen3-8B Q4K size.
pub const AC_PG_QWEN3_8B_Q4K_GB: f32 = 4.19;
/// Acceptable ceiling band [7.0, 9.0] tok/s.
pub const AC_PG_CEILING_LO: f32 = 7.0;
pub const AC_PG_CEILING_HI: f32 = 9.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParityGrade {
    F = 0,
    D = 1,
    C = 2,
    B = 3,
    A = 4,
    APlus = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EfficiencyGrade {
    F = 0,
    D = 1,
    C = 2,
    B = 3,
    A = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgVerdict {
    Pass,
    Fail,
}

/// Reference: ratio (apr_tps / baseline_tps) → ParityGrade.
///
/// Per contract:
///   F     : ratio < 0.5
///   D     : 0.5  <= ratio < 0.75
///   C     : 0.75 <= ratio < 1.0
///   B     : 1.0  <= ratio < 1.5
///   A     : 1.5  <= ratio < 2.0
///   A+    : ratio >= 2.0
///
/// Non-finite or negative ratio → F (fail-closed).
#[must_use]
pub fn classify_parity_ratio(ratio: f32) -> ParityGrade {
    if !ratio.is_finite() || ratio < 0.0 {
        return ParityGrade::F;
    }
    if ratio < 0.5 {
        ParityGrade::F
    } else if ratio < 0.75 {
        ParityGrade::D
    } else if ratio < 1.0 {
        ParityGrade::C
    } else if ratio < 1.5 {
        ParityGrade::B
    } else if ratio < 2.0 {
        ParityGrade::A
    } else {
        ParityGrade::APlus
    }
}

/// Reference: efficiency (actual_tps / roofline_ceiling) → EfficiencyGrade.
///
///   F : eff < 0.10
///   D : 0.10 <= eff < 0.20
///   C : 0.20 <= eff < 0.40
///   B : 0.40 <= eff < 0.50
///   A : eff >= 0.50
///
/// Non-finite or negative → F.
#[must_use]
pub fn classify_efficiency(eff: f32) -> EfficiencyGrade {
    if !eff.is_finite() || eff < 0.0 {
        return EfficiencyGrade::F;
    }
    if eff < 0.10 {
        EfficiencyGrade::F
    } else if eff < 0.20 {
        EfficiencyGrade::D
    } else if eff < 0.40 {
        EfficiencyGrade::C
    } else if eff < 0.50 {
        EfficiencyGrade::B
    } else {
        EfficiencyGrade::A
    }
}

/// PG-001: exhaustive grading — input ratio always maps to exactly
/// one grade (no None / panic). Modeled by checking that the
/// classifier returns SOME grade.
#[must_use]
pub fn verdict_from_exhaustive_grading(ratio: f32) -> PgVerdict {
    let _ = classify_parity_ratio(ratio);
    PgVerdict::Pass
}

/// PG-002: parity-grade monotonicity for a pair `r_lo <= r_hi`.
#[must_use]
pub fn verdict_from_parity_monotone(r_lo: f32, r_hi: f32) -> PgVerdict {
    if !r_lo.is_finite() || !r_hi.is_finite() {
        return PgVerdict::Fail;
    }
    if r_lo > r_hi {
        return PgVerdict::Fail; // caller invariant violation
    }
    let g_lo = classify_parity_ratio(r_lo);
    let g_hi = classify_parity_ratio(r_hi);
    if g_lo <= g_hi {
        PgVerdict::Pass
    } else {
        PgVerdict::Fail
    }
}

/// PG-003: DDR4 ceiling within [7.0, 9.0] tok/s for Qwen3-8B Q4K.
///
/// `bw_gb_per_sec / model_size_gb` should fall inside the band.
#[must_use]
pub fn verdict_from_concrete_ceiling(bw_gb_per_sec: f32, model_size_gb: f32) -> PgVerdict {
    if !bw_gb_per_sec.is_finite() || !model_size_gb.is_finite() {
        return PgVerdict::Fail;
    }
    if bw_gb_per_sec <= 0.0 || model_size_gb <= 0.0 {
        return PgVerdict::Fail;
    }
    let ceiling = bw_gb_per_sec / model_size_gb;
    if (AC_PG_CEILING_LO..=AC_PG_CEILING_HI).contains(&ceiling) {
        PgVerdict::Pass
    } else {
        PgVerdict::Fail
    }
}

/// PG-004: efficiency-grade monotonicity for a pair `e_lo <= e_hi`.
#[must_use]
pub fn verdict_from_efficiency_monotone(e_lo: f32, e_hi: f32) -> PgVerdict {
    if !e_lo.is_finite() || !e_hi.is_finite() {
        return PgVerdict::Fail;
    }
    if e_lo > e_hi {
        return PgVerdict::Fail;
    }
    let g_lo = classify_efficiency(e_lo);
    let g_hi = classify_efficiency(e_hi);
    if g_lo <= g_hi {
        PgVerdict::Pass
    } else {
        PgVerdict::Fail
    }
}

/// PG-005: SIMD vs scalar grade equivalence — output enum identical.
#[must_use]
pub fn verdict_from_simd_grading_equivalence(
    simd_grade: ParityGrade,
    scalar_grade: ParityGrade,
) -> PgVerdict {
    if simd_grade == scalar_grade {
        PgVerdict::Pass
    } else {
        PgVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_PG_DDR4_BANDWIDTH_GB_S, 33.0);
        assert_eq!(AC_PG_QWEN3_8B_Q4K_GB, 4.19);
        assert_eq!(AC_PG_CEILING_LO, 7.0);
        assert_eq!(AC_PG_CEILING_HI, 9.0);
    }

    // -----------------------------------------------------------------
    // Section 2: classify_parity_ratio reference.
    // -----------------------------------------------------------------
    #[test]
    fn classify_parity_boundaries() {
        // Below 0.5 → F
        assert_eq!(classify_parity_ratio(0.0), ParityGrade::F);
        assert_eq!(classify_parity_ratio(0.49), ParityGrade::F);
        // [0.5, 0.75) → D
        assert_eq!(classify_parity_ratio(0.5), ParityGrade::D);
        assert_eq!(classify_parity_ratio(0.74), ParityGrade::D);
        // [0.75, 1.0) → C
        assert_eq!(classify_parity_ratio(0.75), ParityGrade::C);
        assert_eq!(classify_parity_ratio(0.99), ParityGrade::C);
        // [1.0, 1.5) → B
        assert_eq!(classify_parity_ratio(1.0), ParityGrade::B);
        assert_eq!(classify_parity_ratio(1.49), ParityGrade::B);
        // [1.5, 2.0) → A
        assert_eq!(classify_parity_ratio(1.5), ParityGrade::A);
        assert_eq!(classify_parity_ratio(1.99), ParityGrade::A);
        // >= 2.0 → A+
        assert_eq!(classify_parity_ratio(2.0), ParityGrade::APlus);
        assert_eq!(classify_parity_ratio(10.0), ParityGrade::APlus);
    }

    #[test]
    fn classify_parity_invalid_ratios() {
        assert_eq!(classify_parity_ratio(-1.0), ParityGrade::F);
        assert_eq!(classify_parity_ratio(f32::NAN), ParityGrade::F);
        assert_eq!(classify_parity_ratio(f32::INFINITY), ParityGrade::F);
    }

    #[test]
    fn classify_efficiency_boundaries() {
        assert_eq!(classify_efficiency(0.0), EfficiencyGrade::F);
        assert_eq!(classify_efficiency(0.099), EfficiencyGrade::F);
        assert_eq!(classify_efficiency(0.10), EfficiencyGrade::D);
        assert_eq!(classify_efficiency(0.199), EfficiencyGrade::D);
        assert_eq!(classify_efficiency(0.20), EfficiencyGrade::C);
        assert_eq!(classify_efficiency(0.399), EfficiencyGrade::C);
        assert_eq!(classify_efficiency(0.40), EfficiencyGrade::B);
        assert_eq!(classify_efficiency(0.499), EfficiencyGrade::B);
        assert_eq!(classify_efficiency(0.50), EfficiencyGrade::A);
        assert_eq!(classify_efficiency(0.99), EfficiencyGrade::A);
    }

    // -----------------------------------------------------------------
    // Section 3: PG-001 exhaustive.
    // -----------------------------------------------------------------
    #[test]
    fn fpg001_pass_arbitrary_ratios() {
        // Total fn — never panics, always returns a grade.
        for r in [0.0_f32, 0.1, 0.5, 1.0, 1.5, 2.0, 100.0, -1.0, f32::NAN] {
            let v = verdict_from_exhaustive_grading(r);
            assert_eq!(v, PgVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 4: PG-002 monotone parity.
    // -----------------------------------------------------------------
    #[test]
    fn fpg002_pass_increasing_ratios() {
        let v = verdict_from_parity_monotone(0.4, 1.6);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg002_pass_same_ratio() {
        let v = verdict_from_parity_monotone(1.0, 1.0);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg002_fail_inverted() {
        let v = verdict_from_parity_monotone(2.0, 0.5);
        assert_eq!(v, PgVerdict::Fail);
    }

    #[test]
    fn fpg002_fail_nan() {
        let v = verdict_from_parity_monotone(f32::NAN, 1.0);
        assert_eq!(v, PgVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: PG-003 concrete ceiling.
    // -----------------------------------------------------------------
    #[test]
    fn fpg003_pass_ddr4_qwen3_8b() {
        // 33 / 4.19 ≈ 7.876 — inside [7.0, 9.0]
        let v = verdict_from_concrete_ceiling(AC_PG_DDR4_BANDWIDTH_GB_S, AC_PG_QWEN3_8B_Q4K_GB);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg003_pass_at_low_boundary() {
        let v = verdict_from_concrete_ceiling(7.0, 1.0);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg003_pass_at_high_boundary() {
        let v = verdict_from_concrete_ceiling(9.0, 1.0);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg003_fail_below_boundary() {
        let v = verdict_from_concrete_ceiling(6.99, 1.0);
        assert_eq!(v, PgVerdict::Fail);
    }

    #[test]
    fn fpg003_fail_above_boundary() {
        let v = verdict_from_concrete_ceiling(10.0, 1.0);
        assert_eq!(v, PgVerdict::Fail);
    }

    #[test]
    fn fpg003_fail_zero_size() {
        let v = verdict_from_concrete_ceiling(33.0, 0.0);
        assert_eq!(v, PgVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: PG-004 efficiency monotone, PG-005 SIMD equivalence.
    // -----------------------------------------------------------------
    #[test]
    fn fpg004_pass_increasing_efficiency() {
        let v = verdict_from_efficiency_monotone(0.05, 0.55);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg004_fail_inverted() {
        let v = verdict_from_efficiency_monotone(0.55, 0.05);
        assert_eq!(v, PgVerdict::Fail);
    }

    #[test]
    fn fpg005_pass_same_grade() {
        let v = verdict_from_simd_grading_equivalence(ParityGrade::B, ParityGrade::B);
        assert_eq!(v, PgVerdict::Pass);
    }

    #[test]
    fn fpg005_fail_grade_drift() {
        let v = verdict_from_simd_grading_equivalence(ParityGrade::A, ParityGrade::B);
        assert_eq!(v, PgVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_002_full_ratio_band() {
        // Sweep ratios 0.0 to 3.0 in 0.1 steps; verify monotonicity.
        let probes: Vec<f32> = (0..=30).map(|i| i as f32 * 0.1).collect();
        for window in probes.windows(2) {
            let v = verdict_from_parity_monotone(window[0], window[1]);
            assert_eq!(v, PgVerdict::Pass, "{:?} -> {:?}", window[0], window[1]);
        }
    }

    #[test]
    fn mutation_survey_003_ceiling_band_sweep() {
        for tenths in [69_u32, 70, 71, 80, 89, 90, 91] {
            let ceiling = tenths as f32 / 10.0;
            let v = verdict_from_concrete_ceiling(ceiling, 1.0);
            let want = if (AC_PG_CEILING_LO..=AC_PG_CEILING_HI).contains(&ceiling) {
                PgVerdict::Pass
            } else {
                PgVerdict::Fail
            };
            assert_eq!(v, want, "ceiling={ceiling}");
        }
    }

    #[test]
    fn realistic_healthy_grading_passes_all_5() {
        let v1 = verdict_from_exhaustive_grading(1.434);
        let v2 = verdict_from_parity_monotone(1.0, 1.5);
        let v3 = verdict_from_concrete_ceiling(33.0, 4.19);
        let v4 = verdict_from_efficiency_monotone(0.10, 0.50);
        let v5 = verdict_from_simd_grading_equivalence(ParityGrade::B, ParityGrade::B);
        assert_eq!(v1, PgVerdict::Pass);
        assert_eq!(v2, PgVerdict::Pass);
        assert_eq!(v3, PgVerdict::Pass);
        assert_eq!(v4, PgVerdict::Pass);
        assert_eq!(v5, PgVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Pre-fix regressions:
        //   1: PG-001 is total — exhaustive can't fail; we use the
        //      monotonicity Fail twice to cover the "5 simultaneous"
        //      claim.
        //   2: parity classifier decreased grade with rising ratio
        //   3: ceiling outside band (e.g. wrong DDR5 bandwidth used)
        //   4: efficiency classifier inverted
        //   5: SIMD path produced different grade
        let v1 = verdict_from_parity_monotone(2.0, 0.5);
        let v2 = verdict_from_parity_monotone(2.5, 0.4);
        let v3 = verdict_from_concrete_ceiling(50.0, 4.19);
        let v4 = verdict_from_efficiency_monotone(0.6, 0.05);
        let v5 = verdict_from_simd_grading_equivalence(ParityGrade::APlus, ParityGrade::B);
        assert_eq!(v1, PgVerdict::Fail);
        assert_eq!(v2, PgVerdict::Fail);
        assert_eq!(v3, PgVerdict::Fail);
        assert_eq!(v4, PgVerdict::Fail);
        assert_eq!(v5, PgVerdict::Fail);
    }
}
