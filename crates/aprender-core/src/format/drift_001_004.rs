// `drift-detection-v1` algorithm-level PARTIAL discharge for
// FALSIFY-DRIFT-001..004.
//
// Contract: `contracts/drift-detection-v1.yaml`.
//
// Pure-Rust verdicts for the 4 falsification gates:
//   DRIFT-001: drift_score ≥ 0 for all inputs
//   DRIFT-002: higher score → equal-or-higher severity (monotone)
//   DRIFT-003: |data| < min_samples → NoDrift
//   DRIFT-004: identical reference == current → drift_score = 0 → NoDrift

/// `mu_ref == mu_cur` with sigma_ref > 0 ⇒ score is exactly 0.0.
pub const AC_DRIFT_NO_DRIFT_SCORE: f32 = 0.0;
/// `min_samples` strict lower bound — any data shorter than this is
/// reported as `NoDrift` regardless of distribution.
pub const AC_DRIFT_MIN_SAMPLES_FLOOR: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DriftStatus {
    NoDrift = 0,
    Warning = 1,
    Drift = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftVerdict {
    Pass,
    Fail,
}

/// Reference univariate drift score: `|mu_ref - mu_cur| / sigma_ref`.
/// Returns `f32::NAN` if `sigma_ref <= 0` (signals invalid input).
#[must_use]
pub fn univariate_drift_score(mu_ref: f32, mu_cur: f32, sigma_ref: f32) -> f32 {
    if !sigma_ref.is_finite() || sigma_ref <= 0.0 {
        return f32::NAN;
    }
    if !mu_ref.is_finite() || !mu_cur.is_finite() {
        return f32::NAN;
    }
    (mu_ref - mu_cur).abs() / sigma_ref
}

/// Reference threshold-based classifier:
///   score < warn_threshold → NoDrift
///   warn_threshold ≤ score < drift_threshold → Warning
///   drift_threshold ≤ score → Drift
/// `0 < warn_threshold < drift_threshold` precondition.
/// On precondition failure, returns `NoDrift` (defensive — fail-closed).
#[must_use]
pub fn classify_drift(score: f32, warn_threshold: f32, drift_threshold: f32) -> DriftStatus {
    if !score.is_finite() || !warn_threshold.is_finite() || !drift_threshold.is_finite() {
        return DriftStatus::NoDrift;
    }
    if !(0.0 < warn_threshold && warn_threshold < drift_threshold) {
        return DriftStatus::NoDrift;
    }
    if score < warn_threshold {
        DriftStatus::NoDrift
    } else if score < drift_threshold {
        DriftStatus::Warning
    } else {
        DriftStatus::Drift
    }
}

/// DRIFT-001: drift score is non-negative.
///
/// Pass iff `score >= 0` AND `score.is_finite()`.
#[must_use]
pub fn verdict_from_score_nonneg(score: f32) -> DriftVerdict {
    if score.is_finite() && score >= 0.0 {
        DriftVerdict::Pass
    } else {
        DriftVerdict::Fail
    }
}

/// DRIFT-002: classification is monotone in score.
///
/// Given two scores `s_lo <= s_hi` and shared thresholds, the lower
/// score's status must be `<=` the higher score's status.
/// Pass iff for the given pair the monotonicity holds.
#[must_use]
pub fn verdict_from_status_monotone(
    s_lo: f32,
    s_hi: f32,
    warn_threshold: f32,
    drift_threshold: f32,
) -> DriftVerdict {
    // Treat NaN (incomparable) and inverted pairs as caller invariant violations.
    if !matches!(
        s_lo.partial_cmp(&s_hi),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    ) {
        return DriftVerdict::Fail; // caller invariant violated
    }
    let st_lo = classify_drift(s_lo, warn_threshold, drift_threshold);
    let st_hi = classify_drift(s_hi, warn_threshold, drift_threshold);
    if st_lo <= st_hi {
        DriftVerdict::Pass
    } else {
        DriftVerdict::Fail
    }
}

/// DRIFT-003: `|data| < min_samples` ⇒ NoDrift.
///
/// Pass iff `actual_status == NoDrift` whenever `data_len < min_samples`.
/// When `data_len >= min_samples`, the gate is not applicable; we
/// require the caller to gate on `min_samples_applicable` before calling
/// this verdict — but we still validate min_samples >= floor here.
#[must_use]
pub fn verdict_from_min_samples_guard(
    data_len: usize,
    min_samples: usize,
    actual_status: DriftStatus,
) -> DriftVerdict {
    if min_samples < AC_DRIFT_MIN_SAMPLES_FLOOR {
        return DriftVerdict::Fail;
    }
    if data_len < min_samples && actual_status != DriftStatus::NoDrift {
        return DriftVerdict::Fail;
    }
    DriftVerdict::Pass
}

/// DRIFT-004: identical distributions yield NoDrift.
///
/// Pass iff `mu_ref == mu_cur` ⇒ `score == 0.0` AND
/// `classify_drift(0.0, ..) == NoDrift`.
#[must_use]
pub fn verdict_from_identical_no_drift(
    mu_ref: f32,
    sigma_ref: f32,
    warn_threshold: f32,
    drift_threshold: f32,
) -> DriftVerdict {
    let score = univariate_drift_score(mu_ref, mu_ref, sigma_ref);
    if score != AC_DRIFT_NO_DRIFT_SCORE {
        return DriftVerdict::Fail;
    }
    let status = classify_drift(score, warn_threshold, drift_threshold);
    if status == DriftStatus::NoDrift {
        DriftVerdict::Pass
    } else {
        DriftVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_no_drift_score_is_zero() {
        assert_eq!(AC_DRIFT_NO_DRIFT_SCORE, 0.0);
    }

    #[test]
    fn provenance_min_samples_floor_is_1() {
        assert_eq!(AC_DRIFT_MIN_SAMPLES_FLOOR, 1);
    }

    #[test]
    fn provenance_status_ordering() {
        assert!(DriftStatus::NoDrift < DriftStatus::Warning);
        assert!(DriftStatus::Warning < DriftStatus::Drift);
    }

    // -----------------------------------------------------------------
    // Section 2: DRIFT-001 score nonneg.
    // -----------------------------------------------------------------
    #[test]
    fn fdrift001_pass_zero_score() {
        let v = verdict_from_score_nonneg(0.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift001_pass_positive_score() {
        let v = verdict_from_score_nonneg(2.5);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift001_fail_negative_score() {
        let v = verdict_from_score_nonneg(-0.0001);
        assert_eq!(v, DriftVerdict::Fail);
    }

    #[test]
    fn fdrift001_fail_nan() {
        let v = verdict_from_score_nonneg(f32::NAN);
        assert_eq!(v, DriftVerdict::Fail);
    }

    #[test]
    fn fdrift001_fail_neg_infinity() {
        let v = verdict_from_score_nonneg(f32::NEG_INFINITY);
        assert_eq!(v, DriftVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: DRIFT-002 monotone classification.
    // -----------------------------------------------------------------
    #[test]
    fn fdrift002_pass_strictly_increasing() {
        let v = verdict_from_status_monotone(0.1, 1.5, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift002_pass_same_score_same_status() {
        let v = verdict_from_status_monotone(0.5, 0.5, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift002_pass_no_drift_to_warning() {
        let v = verdict_from_status_monotone(0.5, 1.5, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift002_pass_warning_to_drift() {
        let v = verdict_from_status_monotone(1.5, 2.5, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift002_pass_no_drift_to_drift() {
        let v = verdict_from_status_monotone(0.0, 5.0, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift002_fail_inverted_inputs() {
        // Caller invariant: s_lo <= s_hi. Inverted is Fail.
        let v = verdict_from_status_monotone(1.5, 0.5, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: DRIFT-003 min_samples guard.
    // -----------------------------------------------------------------
    #[test]
    fn fdrift003_pass_below_min_with_no_drift() {
        let v = verdict_from_min_samples_guard(5, 30, DriftStatus::NoDrift);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift003_fail_below_min_with_warning() {
        let v = verdict_from_min_samples_guard(5, 30, DriftStatus::Warning);
        assert_eq!(v, DriftVerdict::Fail);
    }

    #[test]
    fn fdrift003_fail_below_min_with_drift() {
        let v = verdict_from_min_samples_guard(5, 30, DriftStatus::Drift);
        assert_eq!(v, DriftVerdict::Fail);
    }

    #[test]
    fn fdrift003_pass_above_min_status_irrelevant() {
        // Once data is sufficient, the guard's predicate vacuously holds.
        let v = verdict_from_min_samples_guard(100, 30, DriftStatus::Drift);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift003_fail_zero_min_samples() {
        // min_samples must be ≥ 1.
        let v = verdict_from_min_samples_guard(0, 0, DriftStatus::NoDrift);
        assert_eq!(v, DriftVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: DRIFT-004 identical distributions.
    // -----------------------------------------------------------------
    #[test]
    fn fdrift004_pass_identical_means() {
        let v = verdict_from_identical_no_drift(5.0, 1.0, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift004_pass_identical_negative_mean() {
        let v = verdict_from_identical_no_drift(-3.0, 0.5, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Pass);
    }

    #[test]
    fn fdrift004_fail_invalid_sigma() {
        let v = verdict_from_identical_no_drift(5.0, 0.0, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Fail);
        let v = verdict_from_identical_no_drift(5.0, -1.0, 1.0, 2.0);
        assert_eq!(v, DriftVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Reference helpers + mutation survey.
    // -----------------------------------------------------------------
    #[test]
    fn univariate_drift_zero_when_means_match() {
        assert_eq!(univariate_drift_score(3.0, 3.0, 1.0), 0.0);
    }

    #[test]
    fn univariate_drift_scales_with_shift() {
        let s1 = univariate_drift_score(0.0, 1.0, 1.0);
        let s2 = univariate_drift_score(0.0, 5.0, 1.0);
        assert!(s2 > s1);
        assert!(s1 > 0.0);
    }

    #[test]
    fn classify_drift_three_regions() {
        // boundary inclusivity: warn-inclusive lower bound
        assert_eq!(classify_drift(0.0, 1.0, 2.0), DriftStatus::NoDrift);
        assert_eq!(classify_drift(0.99, 1.0, 2.0), DriftStatus::NoDrift);
        assert_eq!(classify_drift(1.0, 1.0, 2.0), DriftStatus::Warning);
        assert_eq!(classify_drift(1.99, 1.0, 2.0), DriftStatus::Warning);
        assert_eq!(classify_drift(2.0, 1.0, 2.0), DriftStatus::Drift);
        assert_eq!(classify_drift(100.0, 1.0, 2.0), DriftStatus::Drift);
    }

    #[test]
    fn classify_drift_invalid_thresholds_fail_closed() {
        // warn >= drift: fail-closed → NoDrift
        assert_eq!(classify_drift(5.0, 2.0, 1.0), DriftStatus::NoDrift);
        // warn == drift: fail-closed
        assert_eq!(classify_drift(5.0, 1.0, 1.0), DriftStatus::NoDrift);
        // negative threshold: fail-closed
        assert_eq!(classify_drift(5.0, -1.0, 2.0), DriftStatus::NoDrift);
    }

    #[test]
    fn mutation_survey_002_score_sweep_monotone() {
        // For any pair of scores in [0, 5], classification must be monotone.
        let warn = 1.0_f32;
        let drift = 2.0_f32;
        let probes = [0.0_f32, 0.5, 0.99, 1.0, 1.5, 1.99, 2.0, 3.0, 5.0];
        for &lo in &probes {
            for &hi in &probes {
                if lo <= hi {
                    let v = verdict_from_status_monotone(lo, hi, warn, drift);
                    assert_eq!(v, DriftVerdict::Pass, "lo={lo} hi={hi}");
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_drift_passes_all_4() {
        // 1: positive score
        let v1 = verdict_from_score_nonneg(0.42);
        // 2: monotone across thresholds
        let v2 = verdict_from_status_monotone(0.1, 2.5, 1.0, 2.0);
        // 3: 5 samples below min=30 → NoDrift
        let v3 = verdict_from_min_samples_guard(5, 30, DriftStatus::NoDrift);
        // 4: identical means → NoDrift
        let v4 = verdict_from_identical_no_drift(5.0, 1.0, 1.0, 2.0);
        assert_eq!(v1, DriftVerdict::Pass);
        assert_eq!(v2, DriftVerdict::Pass);
        assert_eq!(v3, DriftVerdict::Pass);
        assert_eq!(v4, DriftVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // Regression class:
        //   1: signed-bug producing negative score
        //   2: caller passed inverted thresholds (fail-closed),
        //      so even a "drift-class" score lo=2.5 hi=0.5 trips
        //   3: alarm fired with only 5 samples below min=30
        //   4: invalid sigma_ref ≤ 0
        let v1 = verdict_from_score_nonneg(-0.5);
        let v2 = verdict_from_status_monotone(2.5, 0.5, 1.0, 2.0);
        let v3 = verdict_from_min_samples_guard(5, 30, DriftStatus::Drift);
        let v4 = verdict_from_identical_no_drift(5.0, -1.0, 1.0, 2.0);
        assert_eq!(v1, DriftVerdict::Fail);
        assert_eq!(v2, DriftVerdict::Fail);
        assert_eq!(v3, DriftVerdict::Fail);
        assert_eq!(v4, DriftVerdict::Fail);
    }
}
