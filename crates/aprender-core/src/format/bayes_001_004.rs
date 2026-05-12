// `bayesian-v1` algorithm-level PARTIAL discharge for the 4 Bayesian-
// inference falsifiers (posterior positivity, prediction finiteness,
// prediction determinism, posterior concentration with more data).
//
// Contract: `contracts/bayesian-v1.yaml`.
// Refs: Gelman et al. (2013) Bayesian Data Analysis, Murphy (2012)
// Machine Learning: A Probabilistic Perspective.

// =============================================================================
// FALSIFY-BAYES-001 — posterior parameters remain positive
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayesPosteriorPositivityVerdict {
    /// alpha' > 0 AND beta' > 0 after conjugate update.
    Pass,
    /// Either parameter went non-positive (zero or negative) — arithmetic
    /// error or underflow.
    Fail,
}

#[must_use]
pub fn verdict_from_bayes_posterior_positivity(alpha_prime: f64, beta_prime: f64) -> BayesPosteriorPositivityVerdict {
    if !alpha_prime.is_finite() || !beta_prime.is_finite() {
        return BayesPosteriorPositivityVerdict::Fail;
    }
    if alpha_prime > 0.0 && beta_prime > 0.0 {
        BayesPosteriorPositivityVerdict::Pass
    } else {
        BayesPosteriorPositivityVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BAYES-002 — BLR predictions finite
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayesPredictionFinitenessVerdict {
    /// All BLR predictions y_hat[i] are finite (no NaN/Inf).
    Pass,
    /// At least one prediction is NaN or Inf — covariance inversion instability.
    Fail,
}

#[must_use]
pub fn verdict_from_bayes_prediction_finiteness(predictions: &[f64]) -> BayesPredictionFinitenessVerdict {
    if predictions.is_empty() {
        return BayesPredictionFinitenessVerdict::Fail;
    }
    for &p in predictions {
        if !p.is_finite() {
            return BayesPredictionFinitenessVerdict::Fail;
        }
    }
    BayesPredictionFinitenessVerdict::Pass
}

// =============================================================================
// FALSIFY-BAYES-003 — prediction is deterministic
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayesDeterminismVerdict {
    /// predict(X) == predict(X) bit-identical across two calls.
    Pass,
    /// Predictions diverge between calls — random state leaked.
    Fail,
}

#[must_use]
pub fn verdict_from_bayes_determinism(call_a: &[f64], call_b: &[f64]) -> BayesDeterminismVerdict {
    if call_a.len() != call_b.len() {
        return BayesDeterminismVerdict::Fail;
    }
    if call_a.is_empty() {
        return BayesDeterminismVerdict::Fail;
    }
    for (a, b) in call_a.iter().zip(call_b.iter()) {
        if a.is_nan() && b.is_nan() {
            // NaN != NaN by IEEE; treat as "both NaN" → divergent prediction
            // Fail (separately from determinism, but the contract treats
            // any NaN output as a Fail per BAYES-002).
            return BayesDeterminismVerdict::Fail;
        }
        if a != b {
            return BayesDeterminismVerdict::Fail;
        }
    }
    BayesDeterminismVerdict::Pass
}

// =============================================================================
// FALSIFY-BAYES-004 — posterior concentration with more data
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayesConcentrationVerdict {
    /// More data ⇒ posterior variance strictly decreases (or stays equal
    /// in degenerate cases).
    Pass,
    /// 2n-sample variance ≥ n-sample variance — evidence not accumulating.
    Fail,
}

#[must_use]
pub fn verdict_from_bayes_concentration(variance_n: f64, variance_2n: f64) -> BayesConcentrationVerdict {
    if !variance_n.is_finite() || !variance_2n.is_finite() {
        return BayesConcentrationVerdict::Fail;
    }
    if variance_n < 0.0 || variance_2n < 0.0 {
        return BayesConcentrationVerdict::Fail;
    }
    if variance_2n < variance_n {
        BayesConcentrationVerdict::Pass
    } else {
        BayesConcentrationVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: BAYES-001 posterior positivity.
    // -------------------------------------------------------------------------
    #[test]
    fn fb001_pass_canonical_beta_binomial() {
        // α' = 5+10, β' = 2+3.
        assert_eq!(
            verdict_from_bayes_posterior_positivity(15.0, 5.0),
            BayesPosteriorPositivityVerdict::Pass
        );
    }

    #[test]
    fn fb001_pass_just_above_zero() {
        assert_eq!(
            verdict_from_bayes_posterior_positivity(1e-10, 1e-10),
            BayesPosteriorPositivityVerdict::Pass
        );
    }

    #[test]
    fn fb001_fail_alpha_zero() {
        assert_eq!(
            verdict_from_bayes_posterior_positivity(0.0, 5.0),
            BayesPosteriorPositivityVerdict::Fail
        );
    }

    #[test]
    fn fb001_fail_beta_negative() {
        assert_eq!(
            verdict_from_bayes_posterior_positivity(5.0, -0.001),
            BayesPosteriorPositivityVerdict::Fail
        );
    }

    #[test]
    fn fb001_fail_nan_alpha() {
        assert_eq!(
            verdict_from_bayes_posterior_positivity(f64::NAN, 5.0),
            BayesPosteriorPositivityVerdict::Fail
        );
    }

    #[test]
    fn fb001_fail_inf_beta() {
        assert_eq!(
            verdict_from_bayes_posterior_positivity(5.0, f64::INFINITY),
            BayesPosteriorPositivityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: BAYES-002 prediction finiteness.
    // -------------------------------------------------------------------------
    #[test]
    fn fb002_pass_all_finite() {
        let p = vec![1.0, -2.5, 3.1415, 0.0];
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&p),
            BayesPredictionFinitenessVerdict::Pass
        );
    }

    #[test]
    fn fb002_fail_nan_in_predictions() {
        let p = vec![1.0, f64::NAN];
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&p),
            BayesPredictionFinitenessVerdict::Fail
        );
    }

    #[test]
    fn fb002_fail_inf_in_predictions() {
        let p = vec![f64::INFINITY];
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&p),
            BayesPredictionFinitenessVerdict::Fail
        );
    }

    #[test]
    fn fb002_fail_neg_inf() {
        let p = vec![1.0, f64::NEG_INFINITY];
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&p),
            BayesPredictionFinitenessVerdict::Fail
        );
    }

    #[test]
    fn fb002_fail_empty() {
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&[]),
            BayesPredictionFinitenessVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BAYES-003 determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn fb003_pass_bit_identical() {
        let p = vec![1.5, 2.5, 3.5];
        assert_eq!(
            verdict_from_bayes_determinism(&p, &p),
            BayesDeterminismVerdict::Pass
        );
    }

    #[test]
    fn fb003_fail_one_element_differs() {
        let a = vec![1.5, 2.5, 3.5];
        let b = vec![1.5, 2.5, 3.50000001];
        assert_eq!(
            verdict_from_bayes_determinism(&a, &b),
            BayesDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fb003_fail_length_mismatch() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        assert_eq!(
            verdict_from_bayes_determinism(&a, &b),
            BayesDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fb003_fail_both_nan() {
        let a = vec![f64::NAN];
        let b = vec![f64::NAN];
        assert_eq!(
            verdict_from_bayes_determinism(&a, &b),
            BayesDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fb003_fail_empty() {
        assert_eq!(
            verdict_from_bayes_determinism(&[], &[]),
            BayesDeterminismVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: BAYES-004 posterior concentration.
    // -------------------------------------------------------------------------
    #[test]
    fn fb004_pass_2n_smaller() {
        // Doubling data halves variance approximately.
        assert_eq!(
            verdict_from_bayes_concentration(1.0, 0.5),
            BayesConcentrationVerdict::Pass
        );
    }

    #[test]
    fn fb004_pass_marginal_decrease() {
        assert_eq!(
            verdict_from_bayes_concentration(1.0, 0.999999),
            BayesConcentrationVerdict::Pass
        );
    }

    #[test]
    fn fb004_fail_variance_unchanged() {
        // Strict decrease required.
        assert_eq!(
            verdict_from_bayes_concentration(1.0, 1.0),
            BayesConcentrationVerdict::Fail
        );
    }

    #[test]
    fn fb004_fail_variance_increased() {
        assert_eq!(
            verdict_from_bayes_concentration(1.0, 1.5),
            BayesConcentrationVerdict::Fail
        );
    }

    #[test]
    fn fb004_fail_negative_variance() {
        // Variance should never go negative.
        assert_eq!(
            verdict_from_bayes_concentration(1.0, -0.1),
            BayesConcentrationVerdict::Fail
        );
    }

    #[test]
    fn fb004_fail_nan_variance() {
        assert_eq!(
            verdict_from_bayes_concentration(1.0, f64::NAN),
            BayesConcentrationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Realistic — full healthy Bayesian inference passes all 4.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_bayes_passes_all_4() {
        // Beta-Binomial conjugate update: α'=15, β'=5.
        assert_eq!(
            verdict_from_bayes_posterior_positivity(15.0, 5.0),
            BayesPosteriorPositivityVerdict::Pass
        );
        // BLR predictions all finite.
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&[1.5, -2.0, 3.7]),
            BayesPredictionFinitenessVerdict::Pass
        );
        // Deterministic predictions.
        let p = vec![1.5, -2.0, 3.7];
        assert_eq!(
            verdict_from_bayes_determinism(&p, &p),
            BayesDeterminismVerdict::Pass
        );
        // Posterior concentrated with more data.
        assert_eq!(
            verdict_from_bayes_concentration(2.0, 1.0),
            BayesConcentrationVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // 001: arithmetic underflow → α' = 0.
        assert_eq!(
            verdict_from_bayes_posterior_positivity(0.0, 5.0),
            BayesPosteriorPositivityVerdict::Fail
        );
        // 002: covariance inversion produced NaN.
        assert_eq!(
            verdict_from_bayes_prediction_finiteness(&[1.0, f64::NAN]),
            BayesPredictionFinitenessVerdict::Fail
        );
        // 003: random state leaked → predictions differ.
        let a = vec![1.0];
        let b = vec![1.5];
        assert_eq!(
            verdict_from_bayes_determinism(&a, &b),
            BayesDeterminismVerdict::Fail
        );
        // 004: evidence not accumulating → variance grew.
        assert_eq!(
            verdict_from_bayes_concentration(1.0, 1.5),
            BayesConcentrationVerdict::Fail
        );
    }
}
