// Bundles two sister contracts in one verdict module:
//
//   `tdg-scoring-v1` (FALSIFY-TDG_SCORING_V1_001..002)
//   `decode-hot-path-zero-syscalls-v1` (FALSIFY-DECODE-HP-001..002)
//
// TDG-001: 0 ≤ TDG ≤ 100 (score range)
// TDG-002: score(A) > score(B) ⟹ grade(A) ≥ grade(B) (monotonicity)
// DECODE-HP-001: zero per-token fs writes in graphed greedy decode
// DECODE-HP-002: post-fix throughput ≥ 440 tok/s (≥ 4% over 420 baseline)

/// TDG score range bounds (inclusive).
pub const AC_TDG_SCORE_MIN: f32 = 0.0;
pub const AC_TDG_SCORE_MAX: f32 = 100.0;
/// DECODE-HP-002 baseline (before fix).
pub const AC_DECODE_HP_BASELINE_TPS: f32 = 420.0;
/// DECODE-HP-002 post-fix floor.
pub const AC_DECODE_HP_POSTFIX_TPS: f32 = 440.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TdgGrade {
    F = 0,
    D = 1,
    C = 2,
    B = 3,
    A = 4,
    APlus = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdgDecodeVerdict {
    Pass,
    Fail,
}

/// Reference: TDG score → grade.
///
///   < 60 → F
///   60-69 → D
///   70-79 → C
///   80-89 → B
///   90-94 → A
///   ≥ 95 → A+
#[must_use]
pub fn classify_tdg_score(score: f32) -> TdgGrade {
    if !score.is_finite() || score < AC_TDG_SCORE_MIN {
        return TdgGrade::F;
    }
    if score < 60.0 {
        TdgGrade::F
    } else if score < 70.0 {
        TdgGrade::D
    } else if score < 80.0 {
        TdgGrade::C
    } else if score < 90.0 {
        TdgGrade::B
    } else if score < 95.0 {
        TdgGrade::A
    } else {
        TdgGrade::APlus
    }
}

/// TDG-001: score is within [0, 100] AND finite.
#[must_use]
pub fn verdict_from_score_in_range(score: f32) -> TdgDecodeVerdict {
    if !score.is_finite() {
        return TdgDecodeVerdict::Fail;
    }
    if (AC_TDG_SCORE_MIN..=AC_TDG_SCORE_MAX).contains(&score) {
        TdgDecodeVerdict::Pass
    } else {
        TdgDecodeVerdict::Fail
    }
}

/// TDG-002: score(A) > score(B) ⟹ grade(A) ≥ grade(B).
#[must_use]
pub fn verdict_from_grade_monotone(score_a: f32, score_b: f32) -> TdgDecodeVerdict {
    if !score_a.is_finite() || !score_b.is_finite() {
        return TdgDecodeVerdict::Fail;
    }
    if score_a <= score_b {
        // Predicate doesn't apply (A is not strictly greater)
        return TdgDecodeVerdict::Pass;
    }
    let g_a = classify_tdg_score(score_a);
    let g_b = classify_tdg_score(score_b);
    if g_a >= g_b {
        TdgDecodeVerdict::Pass
    } else {
        TdgDecodeVerdict::Fail
    }
}

/// DECODE-HP-001: zero per-token fs writes in greedy graphed decode.
///
/// `fs_write_count` is the count of `std::fs::write` calls inside
/// `forward_gpu_resident_to_token_id` and
/// `forward_graphed_replay_to_token_id` collected during a 50-token
/// generation. Pass iff exactly 0.
#[must_use]
pub fn verdict_from_zero_fs_writes(fs_write_count: u32) -> TdgDecodeVerdict {
    if fs_write_count == 0 {
        TdgDecodeVerdict::Pass
    } else {
        TdgDecodeVerdict::Fail
    }
}

/// DECODE-HP-002: post-fix throughput ≥ 440 tok/s.
#[must_use]
pub fn verdict_from_postfix_throughput(observed_tps: f32) -> TdgDecodeVerdict {
    if !observed_tps.is_finite() || observed_tps <= 0.0 {
        return TdgDecodeVerdict::Fail;
    }
    if observed_tps >= AC_DECODE_HP_POSTFIX_TPS {
        TdgDecodeVerdict::Pass
    } else {
        TdgDecodeVerdict::Fail
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
        assert_eq!(AC_TDG_SCORE_MIN, 0.0);
        assert_eq!(AC_TDG_SCORE_MAX, 100.0);
        assert_eq!(AC_DECODE_HP_BASELINE_TPS, 420.0);
        assert_eq!(AC_DECODE_HP_POSTFIX_TPS, 440.0);
    }

    // -----------------------------------------------------------------
    // Section 2: TDG-001 score range.
    // -----------------------------------------------------------------
    #[test]
    fn ftdg001_pass_zero() {
        let v = verdict_from_score_in_range(0.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn ftdg001_pass_hundred() {
        let v = verdict_from_score_in_range(100.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn ftdg001_pass_typical() {
        let v = verdict_from_score_in_range(95.2);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn ftdg001_fail_negative() {
        let v = verdict_from_score_in_range(-1.0);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    #[test]
    fn ftdg001_fail_above_100() {
        let v = verdict_from_score_in_range(101.0);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    #[test]
    fn ftdg001_fail_nan() {
        let v = verdict_from_score_in_range(f32::NAN);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: TDG-002 grade monotone.
    // -----------------------------------------------------------------
    #[test]
    fn ftdg002_pass_strict_increase() {
        // 95 (A+) > 75 (C)
        let v = verdict_from_grade_monotone(95.0, 75.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn ftdg002_pass_within_band() {
        // 75 (C) > 72 (C) — same grade, monotonicity OK
        let v = verdict_from_grade_monotone(75.0, 72.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn ftdg002_pass_a_not_greater() {
        // Pre-condition: score_a > score_b. If not, verdict is vacuous Pass.
        let v = verdict_from_grade_monotone(50.0, 75.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn ftdg002_fail_nan() {
        let v = verdict_from_grade_monotone(f32::NAN, 75.0);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: DECODE-HP-001 zero fs writes.
    // -----------------------------------------------------------------
    #[test]
    fn fdecode001_pass_zero_writes() {
        let v = verdict_from_zero_fs_writes(0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn fdecode001_fail_one_write() {
        let v = verdict_from_zero_fs_writes(1);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    #[test]
    fn fdecode001_fail_per_token_writes() {
        // 50-token gen × 1 write per token = 50 writes
        let v = verdict_from_zero_fs_writes(50);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: DECODE-HP-002 post-fix throughput.
    // -----------------------------------------------------------------
    #[test]
    fn fdecode002_pass_at_threshold() {
        let v = verdict_from_postfix_throughput(440.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn fdecode002_pass_well_above() {
        let v = verdict_from_postfix_throughput(500.0);
        assert_eq!(v, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn fdecode002_fail_below_threshold() {
        let v = verdict_from_postfix_throughput(439.9);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    #[test]
    fn fdecode002_fail_pre_fix_baseline() {
        // 420 tok/s is the pre-fix baseline — must Fail
        let v = verdict_from_postfix_throughput(AC_DECODE_HP_BASELINE_TPS);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    #[test]
    fn fdecode002_fail_zero() {
        let v = verdict_from_postfix_throughput(0.0);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    #[test]
    fn fdecode002_fail_nan() {
        let v = verdict_from_postfix_throughput(f32::NAN);
        assert_eq!(v, TdgDecodeVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: classify_tdg_score reference + mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn classify_tdg_score_boundaries() {
        assert_eq!(classify_tdg_score(0.0), TdgGrade::F);
        assert_eq!(classify_tdg_score(59.9), TdgGrade::F);
        assert_eq!(classify_tdg_score(60.0), TdgGrade::D);
        assert_eq!(classify_tdg_score(69.9), TdgGrade::D);
        assert_eq!(classify_tdg_score(70.0), TdgGrade::C);
        assert_eq!(classify_tdg_score(79.9), TdgGrade::C);
        assert_eq!(classify_tdg_score(80.0), TdgGrade::B);
        assert_eq!(classify_tdg_score(89.9), TdgGrade::B);
        assert_eq!(classify_tdg_score(90.0), TdgGrade::A);
        assert_eq!(classify_tdg_score(94.9), TdgGrade::A);
        assert_eq!(classify_tdg_score(95.0), TdgGrade::APlus);
        assert_eq!(classify_tdg_score(100.0), TdgGrade::APlus);
    }

    #[test]
    fn mutation_survey_002_score_pair_band() {
        // For any score pair in [0, 100] with a > b, grade(a) >= grade(b).
        let probes = [0.0_f32, 30.0, 60.0, 70.0, 75.0, 80.0, 90.0, 95.0, 100.0];
        for &a in &probes {
            for &b in &probes {
                if a > b {
                    let v = verdict_from_grade_monotone(a, b);
                    assert_eq!(v, TdgDecodeVerdict::Pass, "a={a} b={b}");
                }
            }
        }
    }

    #[test]
    fn mutation_survey_001_range_sweep() {
        for s in [-1_f32, 0.0, 50.0, 100.0, 100.0001, 200.0] {
            let v = verdict_from_score_in_range(s);
            let want = if (0.0..=100.0).contains(&s) {
                TdgDecodeVerdict::Pass
            } else {
                TdgDecodeVerdict::Fail
            };
            assert_eq!(v, want, "s={s}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_4() {
        let v1 = verdict_from_score_in_range(95.2);
        let v2 = verdict_from_grade_monotone(95.0, 75.0);
        let v3 = verdict_from_zero_fs_writes(0);
        let v4 = verdict_from_postfix_throughput(440.4); // measured post-fix
        assert_eq!(v1, TdgDecodeVerdict::Pass);
        assert_eq!(v2, TdgDecodeVerdict::Pass);
        assert_eq!(v3, TdgDecodeVerdict::Pass);
        assert_eq!(v4, TdgDecodeVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // Pre-fix regressions:
        //  1: TDG score corrupted to 200 (out of range)
        //  2: NaN score broke monotonicity gate
        //  3: 50 fs writes per 50-token gen (per-token /tmp scratch)
        //  4: 184 tok/s baseline (before HP-001 fix)
        let v1 = verdict_from_score_in_range(200.0);
        let v2 = verdict_from_grade_monotone(f32::NAN, 75.0);
        let v3 = verdict_from_zero_fs_writes(50);
        let v4 = verdict_from_postfix_throughput(184.0);
        assert_eq!(v1, TdgDecodeVerdict::Fail);
        assert_eq!(v2, TdgDecodeVerdict::Fail);
        assert_eq!(v3, TdgDecodeVerdict::Fail);
        assert_eq!(v4, TdgDecodeVerdict::Fail);
    }
}
