// `linear-models-v1` algorithm-level PARTIAL discharge for
// FALSIFY-LM-001..005.
//
// Contract: `contracts/linear-models-v1.yaml`.
//
// Pure-Rust verdicts for the 5 falsification gates:
//   LM-001: training R² ≥ 0 for OLS LinearRegression
//   LM-002: predict(X) is deterministic across calls
//   LM-003: R² ≈ 1 when y = 2x + 3 + zero-noise
//   LM-004: P(y=1|x) ∈ (0, 1) strictly for finite x (no overflow/underflow)
//   LM-005: P(y=0) + P(y=1) = 1 within float tolerance

/// LM-001: minimum R² floor on training data.
pub const AC_LM_R2_TRAIN_FLOOR: f32 = 0.0;
/// LM-003: tolerance on R² ≈ 1.
pub const AC_LM_R2_PERFECT_TOLERANCE: f32 = 1e-3;
/// LM-005: tolerance on P(y=0) + P(y=1) = 1.
pub const AC_LM_PROB_SUM_TOLERANCE: f32 = 1e-6;
/// LM-004: epsilon for strict-(0,1) bounds — sigmoid must not saturate.
pub const AC_LM_PROB_SATURATION_EPSILON: f32 = 1e-30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmVerdict {
    Pass,
    Fail,
}

/// LM-001: training R² ≥ 0.
#[must_use]
pub fn verdict_from_r2_nonneg_train(r2_train: f32) -> LmVerdict {
    if !r2_train.is_finite() {
        return LmVerdict::Fail;
    }
    if r2_train >= AC_LM_R2_TRAIN_FLOOR {
        LmVerdict::Pass
    } else {
        LmVerdict::Fail
    }
}

/// LM-002: predict(X) deterministic — two calls produce identical output.
#[must_use]
pub fn verdict_from_predict_deterministic(call_a: &[f32], call_b: &[f32]) -> LmVerdict {
    if call_a.is_empty() || call_b.is_empty() || call_a.len() != call_b.len() {
        return LmVerdict::Fail;
    }
    for (x, y) in call_a.iter().zip(call_b.iter()) {
        if x.to_bits() != y.to_bits() {
            return LmVerdict::Fail;
        }
    }
    LmVerdict::Pass
}

/// LM-003: R² ≈ 1 within tolerance.
#[must_use]
pub fn verdict_from_perfect_fit_r2(r2: f32) -> LmVerdict {
    if !r2.is_finite() {
        return LmVerdict::Fail;
    }
    if (r2 - 1.0).abs() <= AC_LM_R2_PERFECT_TOLERANCE {
        LmVerdict::Pass
    } else {
        LmVerdict::Fail
    }
}

/// LM-004: every probability in (0, 1) strictly — no saturation.
///
/// Pass iff every probability is finite AND `0.0 < p < 1.0`. The
/// `AC_LM_PROB_SATURATION_EPSILON` constant exists for documentation
/// purposes — at f32 precision, `1.0 - 1e-30` rounds back to `1.0`,
/// so we use strict `< 1.0` and `> 0.0` comparisons directly.
#[must_use]
pub fn verdict_from_logistic_bounded(probs: &[f32]) -> LmVerdict {
    if probs.is_empty() {
        return LmVerdict::Fail;
    }
    for &p in probs {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return LmVerdict::Fail;
        }
    }
    LmVerdict::Pass
}

/// LM-005: P(y=0) + P(y=1) = 1 within float tolerance.
#[must_use]
pub fn verdict_from_probability_sums_to_one(p0: &[f32], p1: &[f32]) -> LmVerdict {
    if p0.is_empty() || p1.is_empty() || p0.len() != p1.len() {
        return LmVerdict::Fail;
    }
    for (a, b) in p0.iter().zip(p1.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return LmVerdict::Fail;
        }
        let s = a + b;
        if (s - 1.0).abs() > AC_LM_PROB_SUM_TOLERANCE {
            return LmVerdict::Fail;
        }
    }
    LmVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_LM_R2_TRAIN_FLOOR, 0.0);
        assert_eq!(AC_LM_R2_PERFECT_TOLERANCE, 1e-3);
        assert_eq!(AC_LM_PROB_SUM_TOLERANCE, 1e-6);
        assert_eq!(AC_LM_PROB_SATURATION_EPSILON, 1e-30);
    }

    // -----------------------------------------------------------------
    // Section 2: LM-001 training R².
    // -----------------------------------------------------------------
    #[test]
    fn flm001_pass_zero_r2() {
        let v = verdict_from_r2_nonneg_train(0.0);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm001_pass_perfect() {
        let v = verdict_from_r2_nonneg_train(1.0);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm001_fail_negative_r2() {
        // The regression class — solver instability producing R² < 0
        let v = verdict_from_r2_nonneg_train(-0.001);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm001_fail_nan() {
        let v = verdict_from_r2_nonneg_train(f32::NAN);
        assert_eq!(v, LmVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: LM-002 predict deterministic.
    // -----------------------------------------------------------------
    #[test]
    fn flm002_pass_identical_calls() {
        let v = verdict_from_predict_deterministic(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm002_fail_one_ulp_drift() {
        let bumped = f32::from_bits(2.0_f32.to_bits() + 1);
        let v = verdict_from_predict_deterministic(&[1.0, 2.0, 3.0], &[1.0, bumped, 3.0]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm002_fail_length_mismatch() {
        let v = verdict_from_predict_deterministic(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm002_fail_empty() {
        let v = verdict_from_predict_deterministic(&[], &[]);
        assert_eq!(v, LmVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: LM-003 perfect fit.
    // -----------------------------------------------------------------
    #[test]
    fn flm003_pass_exact_one() {
        let v = verdict_from_perfect_fit_r2(1.0);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm003_pass_within_tolerance() {
        let v = verdict_from_perfect_fit_r2(0.9995);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm003_fail_below_tolerance() {
        let v = verdict_from_perfect_fit_r2(0.95);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm003_fail_above_tolerance() {
        // R² > 1 is mathematically impossible for valid OLS — trip
        // the gate as a numerics-corruption guard.
        let v = verdict_from_perfect_fit_r2(1.1);
        assert_eq!(v, LmVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: LM-004 logistic bounded.
    // -----------------------------------------------------------------
    #[test]
    fn flm004_pass_typical_probs() {
        let v = verdict_from_logistic_bounded(&[0.1, 0.5, 0.9, 0.99]);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm004_fail_exact_zero() {
        // Sigmoid underflow — the regression class.
        let v = verdict_from_logistic_bounded(&[0.5, 0.0, 0.7]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm004_fail_exact_one() {
        // Sigmoid overflow.
        let v = verdict_from_logistic_bounded(&[0.5, 1.0, 0.7]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm004_fail_negative() {
        let v = verdict_from_logistic_bounded(&[-0.1]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm004_fail_nan() {
        let v = verdict_from_logistic_bounded(&[f32::NAN]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm004_fail_empty() {
        let v = verdict_from_logistic_bounded(&[]);
        assert_eq!(v, LmVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: LM-005 prob sums to 1.
    // -----------------------------------------------------------------
    #[test]
    fn flm005_pass_typical() {
        let v = verdict_from_probability_sums_to_one(&[0.3, 0.2, 0.7], &[0.7, 0.8, 0.3]);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm005_fail_sum_drift() {
        let v = verdict_from_probability_sums_to_one(&[0.3], &[0.6]); // sums to 0.9
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm005_pass_within_tolerance() {
        // (0.5 + 0.5000001) sums to 1.0000001, |1.0000001 - 1| = 1e-7 < 1e-6
        let v = verdict_from_probability_sums_to_one(&[0.5], &[0.5000001]);
        assert_eq!(v, LmVerdict::Pass);
    }

    #[test]
    fn flm005_fail_length_mismatch() {
        let v = verdict_from_probability_sums_to_one(&[0.3], &[0.7, 0.5]);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn flm005_fail_nan() {
        let v = verdict_from_probability_sums_to_one(&[f32::NAN], &[0.5]);
        assert_eq!(v, LmVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_r2_around_zero() {
        for r2_x100 in [-1_i32, 0, 1, 10, 50, 100] {
            let r2 = r2_x100 as f32 / 100.0;
            let v = verdict_from_r2_nonneg_train(r2);
            let want = if r2 >= 0.0 {
                LmVerdict::Pass
            } else {
                LmVerdict::Fail
            };
            assert_eq!(v, want, "r2={r2}");
        }
    }

    #[test]
    fn mutation_survey_004_probability_band() {
        let probs = [0.001_f32, 0.01, 0.5, 0.99, 0.999];
        let v = verdict_from_logistic_bounded(&probs);
        assert_eq!(v, LmVerdict::Pass);
        // Now adding 0.0 trips the gate.
        let bad = [0.001_f32, 0.0, 0.5, 0.99];
        let v = verdict_from_logistic_bounded(&bad);
        assert_eq!(v, LmVerdict::Fail);
    }

    #[test]
    fn realistic_healthy_passes_all_5() {
        let v1 = verdict_from_r2_nonneg_train(0.85);
        let v2 = verdict_from_predict_deterministic(&[1.5, 2.5, 3.5], &[1.5, 2.5, 3.5]);
        let v3 = verdict_from_perfect_fit_r2(0.99995);
        let v4 = verdict_from_logistic_bounded(&[0.1, 0.5, 0.9]);
        let v5 = verdict_from_probability_sums_to_one(&[0.3, 0.7], &[0.7, 0.3]);
        assert_eq!(v1, LmVerdict::Pass);
        assert_eq!(v2, LmVerdict::Pass);
        assert_eq!(v3, LmVerdict::Pass);
        assert_eq!(v4, LmVerdict::Pass);
        assert_eq!(v5, LmVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // Regression class:
        //  1: solver produced R²=-0.01 (unstable normal equation)
        //  2: predict() returned different bits across calls (state leak)
        //  3: collinear y produced R²=0.85 (wrong slope estimate)
        //  4: sigmoid saturated to 1.0 (overflow at huge logit)
        //  5: probabilities normalized to 0.99 (rounding bug)
        let v1 = verdict_from_r2_nonneg_train(-0.01);
        let bumped = f32::from_bits(2.0_f32.to_bits() + 1);
        let v2 = verdict_from_predict_deterministic(&[1.0, 2.0], &[1.0, bumped]);
        let v3 = verdict_from_perfect_fit_r2(0.85);
        let v4 = verdict_from_logistic_bounded(&[0.5, 1.0]);
        let v5 = verdict_from_probability_sums_to_one(&[0.3], &[0.7 - 0.01]);
        assert_eq!(v1, LmVerdict::Fail);
        assert_eq!(v2, LmVerdict::Fail);
        assert_eq!(v3, LmVerdict::Fail);
        assert_eq!(v4, LmVerdict::Fail);
        assert_eq!(v5, LmVerdict::Fail);
    }
}
