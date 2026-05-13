// SHIP-TWO-001 — `eval-harness-humaneval-v1` algorithm-level PARTIAL
// discharge for FALSIFY-HE-001..006.
//
// Contract: `contracts/eval-harness-humaneval-v1.yaml` v1.1.0.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (AC-SHIP1-005).
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Six gates from the HumanEval audit contract:
//
// - HE-001 (teacher reproduces baseline): teacher pass@1 ∈ [84.5, 87.0]
//   (85.98 ± 1.5 cross-run noise band).
// - HE-002 (student meets ship threshold): student pass@1 ≥ 86.0.
// - HE-003 (native vs GGUF parity): |native - gguf| ≤ 0.6.
// - HE-004 (merged is upper bound): merged ≥ student_primary - 0.6.
// - HE-005 (problem-level regression): ≥ 2 teacher-failed problems
//   pass on student.
// - HE-006 (T=0 determinism): two runs at T=0 produce identical
//   per-problem pass/fail flags across all 164 problems.

/// Canonical HumanEval problem count.
pub const AC_HE_PROBLEM_COUNT: usize = 164;

/// Spec AC-SHIP1-005 threshold.
pub const AC_HE_002_SHIP_THRESHOLD_PCT: f32 = 86.0;

/// Teacher reference baseline (2026-03-28).
pub const AC_HE_001_TEACHER_BASELINE_PCT: f32 = 85.98;

/// Teacher cross-run noise band (± 1.5%).
pub const AC_HE_001_TEACHER_NOISE_BAND: f32 = 1.5;

/// Native vs GGUF parity tolerance (per LAYOUT-001).
pub const AC_HE_003_PARITY_TOLERANCE_PCT: f32 = 0.6;

/// Merged upper-bound slack: merged ≥ q4k - 0.6 (quant noise).
pub const AC_HE_004_MERGED_SLACK_PCT: f32 = 0.6;

/// Min teacher-failed problems student must fix.
pub const AC_HE_005_MIN_FIXED_PROBLEMS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: HE-001 — teacher reproduces baseline.
// -----------------------------------------------------------------------------

/// Pass iff teacher pass@1 falls in `[85.98 - 1.5, 85.98 + 1.5]`.
#[must_use]
pub fn verdict_from_teacher_baseline(teacher_pass_at_1_pct: f32) -> HeVerdict {
    if !teacher_pass_at_1_pct.is_finite() {
        return HeVerdict::Fail;
    }
    let lo = AC_HE_001_TEACHER_BASELINE_PCT - AC_HE_001_TEACHER_NOISE_BAND;
    let hi = AC_HE_001_TEACHER_BASELINE_PCT + AC_HE_001_TEACHER_NOISE_BAND;
    if teacher_pass_at_1_pct >= lo && teacher_pass_at_1_pct <= hi {
        HeVerdict::Pass
    } else {
        HeVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: HE-002 — student meets ship threshold.
// -----------------------------------------------------------------------------

/// Pass iff student pass@1 ≥ 86.0% AND total problems == 164.
#[must_use]
pub fn verdict_from_student_ship_threshold(
    student_pass_at_1_pct: f32,
    total_problems: usize,
) -> HeVerdict {
    if !student_pass_at_1_pct.is_finite() {
        return HeVerdict::Fail;
    }
    if total_problems != AC_HE_PROBLEM_COUNT {
        return HeVerdict::Fail;
    }
    if student_pass_at_1_pct >= AC_HE_002_SHIP_THRESHOLD_PCT {
        HeVerdict::Pass
    } else {
        HeVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: HE-003 — native vs GGUF parity.
// -----------------------------------------------------------------------------

/// Pass iff `|native - gguf| ≤ 0.6` percentage points.
#[must_use]
pub fn verdict_from_native_vs_gguf_parity(
    native_pass_at_1_pct: f32,
    gguf_pass_at_1_pct: f32,
) -> HeVerdict {
    if !native_pass_at_1_pct.is_finite() || !gguf_pass_at_1_pct.is_finite() {
        return HeVerdict::Fail;
    }
    let delta = (native_pass_at_1_pct - gguf_pass_at_1_pct).abs();
    if delta <= AC_HE_003_PARITY_TOLERANCE_PCT {
        HeVerdict::Pass
    } else {
        HeVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: HE-004 — merged is upper bound.
// -----------------------------------------------------------------------------

/// Pass iff `merged ≥ student_primary - 0.6` (allow quant slack).
#[must_use]
pub fn verdict_from_merged_upper_bound(
    merged_pass_at_1_pct: f32,
    student_primary_pass_at_1_pct: f32,
) -> HeVerdict {
    if !merged_pass_at_1_pct.is_finite() || !student_primary_pass_at_1_pct.is_finite() {
        return HeVerdict::Fail;
    }
    if merged_pass_at_1_pct >= student_primary_pass_at_1_pct - AC_HE_004_MERGED_SLACK_PCT {
        HeVerdict::Pass
    } else {
        HeVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: HE-005 — problem-level improvement on teacher-failed.
// -----------------------------------------------------------------------------

/// `teacher_failed_task_ids` — list of task ids the teacher failed.
/// `student_pass_status` — for each task in `teacher_failed_task_ids`,
/// `true` iff student passes that problem.
#[must_use]
pub fn verdict_from_problem_level_improvement(
    student_pass_status_on_teacher_failed: &[bool],
) -> HeVerdict {
    let fixed = student_pass_status_on_teacher_failed
        .iter()
        .filter(|&&p| p)
        .count();
    if fixed >= AC_HE_005_MIN_FIXED_PROBLEMS {
        HeVerdict::Pass
    } else {
        HeVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 6: HE-006 — T=0 sampling determinism.
// -----------------------------------------------------------------------------

/// Pass iff every per-problem pass/fail flag is identical across two
/// runs at T=0.0 AND both runs have all 164 problems.
#[must_use]
pub fn verdict_from_t0_determinism(
    run1_per_problem_pass: &[bool],
    run2_per_problem_pass: &[bool],
) -> HeVerdict {
    if run1_per_problem_pass.len() != AC_HE_PROBLEM_COUNT
        || run2_per_problem_pass.len() != AC_HE_PROBLEM_COUNT
    {
        return HeVerdict::Fail;
    }
    if run1_per_problem_pass == run2_per_problem_pass {
        HeVerdict::Pass
    } else {
        HeVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_problem_count_164() {
        assert_eq!(AC_HE_PROBLEM_COUNT, 164);
    }

    #[test]
    fn provenance_ship_threshold_86() {
        assert_eq!(AC_HE_002_SHIP_THRESHOLD_PCT, 86.0);
    }

    #[test]
    fn provenance_teacher_baseline_8598() {
        assert_eq!(AC_HE_001_TEACHER_BASELINE_PCT, 85.98);
    }

    #[test]
    fn provenance_teacher_noise_15() {
        assert_eq!(AC_HE_001_TEACHER_NOISE_BAND, 1.5);
    }

    #[test]
    fn provenance_parity_tolerance_06() {
        assert_eq!(AC_HE_003_PARITY_TOLERANCE_PCT, 0.6);
    }

    #[test]
    fn provenance_merged_slack_06() {
        assert_eq!(AC_HE_004_MERGED_SLACK_PCT, 0.6);
    }

    #[test]
    fn provenance_min_fixed_problems_2() {
        assert_eq!(AC_HE_005_MIN_FIXED_PROBLEMS, 2);
    }

    // -------------------------------------------------------------------------
    // Section 2: HE-001 — teacher baseline.
    // -------------------------------------------------------------------------
    #[test]
    fn he001_pass_at_baseline() {
        assert_eq!(verdict_from_teacher_baseline(85.98), HeVerdict::Pass);
    }

    #[test]
    fn he001_pass_within_band() {
        assert_eq!(verdict_from_teacher_baseline(84.5), HeVerdict::Pass);
        assert_eq!(verdict_from_teacher_baseline(87.0), HeVerdict::Pass);
        assert_eq!(verdict_from_teacher_baseline(86.5), HeVerdict::Pass);
    }

    #[test]
    fn he001_fail_below_band() {
        assert_eq!(verdict_from_teacher_baseline(84.0), HeVerdict::Fail);
        assert_eq!(verdict_from_teacher_baseline(80.0), HeVerdict::Fail);
    }

    #[test]
    fn he001_fail_above_band() {
        assert_eq!(verdict_from_teacher_baseline(87.5), HeVerdict::Fail);
        assert_eq!(verdict_from_teacher_baseline(95.0), HeVerdict::Fail);
    }

    #[test]
    fn he001_fail_nan() {
        assert_eq!(verdict_from_teacher_baseline(f32::NAN), HeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: HE-002 — student ship threshold.
    // -------------------------------------------------------------------------
    #[test]
    fn he002_pass_exactly_86() {
        assert_eq!(
            verdict_from_student_ship_threshold(86.0, 164),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he002_pass_well_above() {
        assert_eq!(
            verdict_from_student_ship_threshold(90.0, 164),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he002_fail_just_below() {
        assert_eq!(
            verdict_from_student_ship_threshold(85.99, 164),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he002_fail_garbage_zero() {
        // The v1.1.0 validation result: pass@1 ~= 0 due to broken weights.
        assert_eq!(
            verdict_from_student_ship_threshold(0.0, 164),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he002_fail_wrong_problem_count() {
        // Subsample invalidates comparison.
        assert_eq!(
            verdict_from_student_ship_threshold(95.0, 100),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he002_fail_nan() {
        assert_eq!(
            verdict_from_student_ship_threshold(f32::NAN, 164),
            HeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: HE-003 — native vs GGUF parity.
    // -------------------------------------------------------------------------
    #[test]
    fn he003_pass_identical() {
        assert_eq!(
            verdict_from_native_vs_gguf_parity(87.0, 87.0),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he003_pass_at_tolerance() {
        // |87.0 - 86.4| = 0.6 == tolerance ⇒ Pass (inclusive).
        assert_eq!(
            verdict_from_native_vs_gguf_parity(87.0, 86.4),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he003_fail_above_tolerance() {
        // LAYOUT-001 transpose bug.
        assert_eq!(
            verdict_from_native_vs_gguf_parity(87.0, 80.0),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he003_fail_just_above_tolerance() {
        // |87.0 - 86.3| = 0.7 > 0.6.
        assert_eq!(
            verdict_from_native_vs_gguf_parity(87.0, 86.3),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he003_fail_nan() {
        assert_eq!(
            verdict_from_native_vs_gguf_parity(f32::NAN, 87.0),
            HeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: HE-004 — merged is upper bound.
    // -------------------------------------------------------------------------
    #[test]
    fn he004_pass_merged_exceeds_q4k() {
        // Expected: merged > q4k since quantization is lossy.
        assert_eq!(
            verdict_from_merged_upper_bound(89.0, 87.0),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he004_pass_merged_within_slack_below() {
        // Merged = q4k - 0.6 (boundary, inclusive).
        assert_eq!(
            verdict_from_merged_upper_bound(86.4, 87.0),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he004_fail_merged_far_below_q4k() {
        // Merged 5pp lower than q4k — quantization-monotonicity bug.
        assert_eq!(
            verdict_from_merged_upper_bound(82.0, 87.0),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he004_fail_merged_just_below_slack() {
        // 87.0 - 0.7 = 86.3 < 86.4 ⇒ Fail.
        assert_eq!(
            verdict_from_merged_upper_bound(86.3, 87.0),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he004_fail_nan() {
        assert_eq!(
            verdict_from_merged_upper_bound(f32::NAN, 87.0),
            HeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: HE-005 — problem-level improvement.
    // -------------------------------------------------------------------------
    #[test]
    fn he005_pass_two_fixed() {
        // 5 teacher-failed; student fixes 2.
        let stat = vec![true, true, false, false, false];
        assert_eq!(
            verdict_from_problem_level_improvement(&stat),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he005_pass_all_fixed() {
        let stat = vec![true; 5];
        assert_eq!(
            verdict_from_problem_level_improvement(&stat),
            HeVerdict::Pass
        );
    }

    #[test]
    fn he005_fail_one_fixed() {
        let stat = vec![true, false, false, false, false];
        assert_eq!(
            verdict_from_problem_level_improvement(&stat),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he005_fail_zero_fixed() {
        let stat = vec![false; 5];
        assert_eq!(
            verdict_from_problem_level_improvement(&stat),
            HeVerdict::Fail
        );
    }

    #[test]
    fn he005_fail_empty_list() {
        let stat: Vec<bool> = vec![];
        assert_eq!(
            verdict_from_problem_level_improvement(&stat),
            HeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: HE-006 — T=0 determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn he006_pass_identical_runs() {
        let r = vec![true; 164];
        assert_eq!(verdict_from_t0_determinism(&r, &r), HeVerdict::Pass);
    }

    #[test]
    fn he006_pass_mixed_but_consistent() {
        let mut r = vec![true; 164];
        for i in 0..40 {
            r[i] = false;
        }
        assert_eq!(verdict_from_t0_determinism(&r, &r), HeVerdict::Pass);
    }

    #[test]
    fn he006_fail_one_problem_differs() {
        let r1 = vec![true; 164];
        let mut r2 = r1.clone();
        r2[42] = false;
        assert_eq!(verdict_from_t0_determinism(&r1, &r2), HeVerdict::Fail);
    }

    #[test]
    fn he006_fail_run_too_short() {
        let r1 = vec![true; 164];
        let r2 = vec![true; 100]; // wrong size
        assert_eq!(verdict_from_t0_determinism(&r1, &r2), HeVerdict::Fail);
    }

    #[test]
    fn he006_fail_both_wrong_size() {
        let r = vec![true; 100];
        assert_eq!(verdict_from_t0_determinism(&r, &r), HeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Sweep — student threshold band.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_student_threshold_around_86() {
        let test_cases = [
            (85.0_f32, HeVerdict::Fail),
            (85.99, HeVerdict::Fail),
            (86.0, HeVerdict::Pass),
            (86.01, HeVerdict::Pass),
            (90.0, HeVerdict::Pass),
            (95.0, HeVerdict::Pass),
        ];
        for (pct, expected) in test_cases {
            let v = verdict_from_student_ship_threshold(pct, 164);
            assert_eq!(v, expected, "pct={pct}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 9: Realistic — contract validation v1.1.0 scenario.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_v1_1_0_falsified_scenario() {
        // The v1.1.0 result: student pass@1 = 0, teacher = 85.98.
        assert_eq!(
            verdict_from_teacher_baseline(85.98),
            HeVerdict::Pass
        );
        assert_eq!(
            verdict_from_student_ship_threshold(0.0, 164),
            HeVerdict::Fail
        );
    }

    #[test]
    fn realistic_layout_001_regression_caught() {
        // HE-003 if_fails: "LAYOUT-001 row-major transpose bug".
        assert_eq!(
            verdict_from_native_vs_gguf_parity(87.0, 50.0),
            HeVerdict::Fail
        );
    }

    #[test]
    fn realistic_quantization_anomaly_caught() {
        // HE-004 if_fails: merged scores LOWER than q4k.
        assert_eq!(
            verdict_from_merged_upper_bound(80.0, 87.0),
            HeVerdict::Fail
        );
    }

    #[test]
    fn realistic_t0_nondeterminism_caught() {
        // HE-006 if_fails: "non-determinism in the decode path".
        let r1 = vec![true; 164];
        let mut r2 = r1.clone();
        r2[100] = false; // single divergence
        assert_eq!(verdict_from_t0_determinism(&r1, &r2), HeVerdict::Fail);
    }

    #[test]
    fn realistic_full_qa_gate_must_pass_set() {
        // QA gate must_pass: HE-002 + HE-001.
        assert_eq!(
            verdict_from_student_ship_threshold(87.20, 164),
            HeVerdict::Pass
        );
        assert_eq!(verdict_from_teacher_baseline(85.98), HeVerdict::Pass);
    }
}
