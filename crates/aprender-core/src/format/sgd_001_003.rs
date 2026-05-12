// `apr-stochastic-lr-v1` algorithm-level PARTIAL discharge for the 3
// SGD/mini-batch falsifiers (backward compat default Batch, stochastic
// MCC > 0.3 on imbalanced, mini-batch(n) == full-batch).
//
// Contract: `contracts/apr-stochastic-lr-v1.yaml`.
// Refs: GH-428 (minority class signal dilution), Bottou (2012)
// "Stochastic Gradient Descent Tricks".

/// Minimum MCC required by FALSIFY-SGD-002 on imbalanced classification.
pub const AC_SGD_MIN_MCC: f64 = 0.3;

/// Tolerance for "mini-batch(n) == full-batch" gradient comparison
/// (deterministic identity, allow only floating-point reduction noise).
pub const AC_SGD_MINIBATCH_TOLERANCE: f64 = 1e-6;

/// LogisticRegression fit-mode enum per the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    Batch,
    Stochastic,
    MiniBatch(usize),
}

// =============================================================================
// FALSIFY-SGD-001 — backward compatibility (default == Batch, identical weights)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardCompatVerdict {
    /// Default fit mode is Batch AND weights are bit-identical to legacy.
    Pass,
    /// Default mode changed OR weights drift.
    Fail,
}

#[must_use]
pub fn verdict_from_backward_compat(
    default_mode: FitMode,
    legacy_weights: &[f64],
    new_weights: &[f64],
) -> BackwardCompatVerdict {
    if !matches!(default_mode, FitMode::Batch) {
        return BackwardCompatVerdict::Fail;
    }
    if legacy_weights.len() != new_weights.len() {
        return BackwardCompatVerdict::Fail;
    }
    for (a, b) in legacy_weights.iter().zip(new_weights.iter()) {
        // Bit-identical check, not approximate — backward compat means
        // "no behavior change", not "approximately the same".
        if (a - b).abs() > 0.0 {
            return BackwardCompatVerdict::Fail;
        }
    }
    BackwardCompatVerdict::Pass
}

// =============================================================================
// FALSIFY-SGD-002 — stochastic mode improves imbalanced classification
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticImbalancedVerdict {
    /// MCC > 0.3 on imbalanced dataset under FitMode::Stochastic.
    Pass,
    /// MCC at-or-below threshold — stochastic mode didn't improve over batch.
    Fail,
}

#[must_use]
pub fn verdict_from_stochastic_imbalanced(mcc: f64) -> StochasticImbalancedVerdict {
    if mcc > AC_SGD_MIN_MCC {
        StochasticImbalancedVerdict::Pass
    } else {
        StochasticImbalancedVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-SGD-003 — mini-batch(n) == full-batch (unbiased estimator)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniBatchUnbiasedVerdict {
    /// MiniBatch(n_samples) gradient agrees with Batch gradient
    /// element-wise within 1e-6 tolerance.
    Pass,
    /// At least one component differs — gradient computation diverged.
    Fail,
}

#[must_use]
pub fn verdict_from_minibatch_unbiased(
    full_batch_grad: &[f64],
    minibatch_n_grad: &[f64],
) -> MiniBatchUnbiasedVerdict {
    if full_batch_grad.len() != minibatch_n_grad.len() {
        return MiniBatchUnbiasedVerdict::Fail;
    }
    if full_batch_grad.is_empty() {
        return MiniBatchUnbiasedVerdict::Fail;
    }
    for (a, b) in full_batch_grad.iter().zip(minibatch_n_grad.iter()) {
        if (a - b).abs() > AC_SGD_MINIBATCH_TOLERANCE {
            return MiniBatchUnbiasedVerdict::Fail;
        }
    }
    MiniBatchUnbiasedVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_mcc_03() {
        assert!((AC_SGD_MIN_MCC - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_minibatch_tolerance_1e6() {
        assert!((AC_SGD_MINIBATCH_TOLERANCE - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_minibatch_1_equals_stochastic() {
        // The contract says MiniBatch(1) == Stochastic semantically. Encode
        // that as a comment-pin: callers passing MiniBatch(1) should get
        // the same algorithmic behavior as Stochastic.
        let mb1 = FitMode::MiniBatch(1);
        let stoch = FitMode::Stochastic;
        // They are distinct enum variants, but the contract documents that
        // they're isomorphic; the type system requires the caller to choose.
        assert!(!matches!(mb1, FitMode::Stochastic));
        assert!(!matches!(stoch, FitMode::MiniBatch(_)));
    }

    // -------------------------------------------------------------------------
    // Section 2: SGD-001 backward compat.
    // -------------------------------------------------------------------------
    #[test]
    fn fs001_pass_batch_default_identical() {
        let legacy = vec![0.5, -0.3, 1.2];
        let new = vec![0.5, -0.3, 1.2];
        assert_eq!(
            verdict_from_backward_compat(FitMode::Batch, &legacy, &new),
            BackwardCompatVerdict::Pass
        );
    }

    #[test]
    fn fs001_fail_default_changed_to_stochastic() {
        let legacy = vec![0.5];
        let new = vec![0.5];
        assert_eq!(
            verdict_from_backward_compat(FitMode::Stochastic, &legacy, &new),
            BackwardCompatVerdict::Fail
        );
    }

    #[test]
    fn fs001_fail_default_changed_to_minibatch() {
        let legacy = vec![0.5];
        let new = vec![0.5];
        assert_eq!(
            verdict_from_backward_compat(FitMode::MiniBatch(32), &legacy, &new),
            BackwardCompatVerdict::Fail
        );
    }

    #[test]
    fn fs001_fail_weights_drift() {
        // Even a tiny float drift fails — backward compat is exact.
        let legacy = vec![0.5];
        let new = vec![0.5000001];
        assert_eq!(
            verdict_from_backward_compat(FitMode::Batch, &legacy, &new),
            BackwardCompatVerdict::Fail
        );
    }

    #[test]
    fn fs001_fail_weights_length_mismatch() {
        let legacy = vec![0.5, 0.3];
        let new = vec![0.5];
        assert_eq!(
            verdict_from_backward_compat(FitMode::Batch, &legacy, &new),
            BackwardCompatVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: SGD-002 stochastic imbalanced.
    // -------------------------------------------------------------------------
    #[test]
    fn fs002_pass_high_mcc() {
        assert_eq!(
            verdict_from_stochastic_imbalanced(0.7),
            StochasticImbalancedVerdict::Pass
        );
    }

    #[test]
    fn fs002_pass_just_above_threshold() {
        assert_eq!(
            verdict_from_stochastic_imbalanced(0.31),
            StochasticImbalancedVerdict::Pass
        );
    }

    #[test]
    fn fs002_fail_at_threshold() {
        // Strict greater-than: 0.3 exactly fails per contract test predicate.
        assert_eq!(
            verdict_from_stochastic_imbalanced(0.3),
            StochasticImbalancedVerdict::Fail
        );
    }

    #[test]
    fn fs002_fail_low_mcc() {
        assert_eq!(
            verdict_from_stochastic_imbalanced(0.1),
            StochasticImbalancedVerdict::Fail
        );
    }

    #[test]
    fn fs002_fail_negative_mcc() {
        // MCC can be negative for inverse correlation.
        assert_eq!(
            verdict_from_stochastic_imbalanced(-0.5),
            StochasticImbalancedVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: SGD-003 mini-batch unbiased.
    // -------------------------------------------------------------------------
    #[test]
    fn fs003_pass_identical_gradients() {
        let g = vec![0.123, -0.456, 0.789];
        assert_eq!(
            verdict_from_minibatch_unbiased(&g, &g),
            MiniBatchUnbiasedVerdict::Pass
        );
    }

    #[test]
    fn fs003_pass_within_tolerance() {
        let a = vec![0.123];
        let b = vec![0.123 + 5e-7];
        assert_eq!(
            verdict_from_minibatch_unbiased(&a, &b),
            MiniBatchUnbiasedVerdict::Pass
        );
    }

    #[test]
    fn fs003_fail_outside_tolerance() {
        let a = vec![0.123];
        let b = vec![0.124];
        assert_eq!(
            verdict_from_minibatch_unbiased(&a, &b),
            MiniBatchUnbiasedVerdict::Fail
        );
    }

    #[test]
    fn fs003_fail_length_mismatch() {
        let a = vec![0.1, 0.2];
        let b = vec![0.1];
        assert_eq!(
            verdict_from_minibatch_unbiased(&a, &b),
            MiniBatchUnbiasedVerdict::Fail
        );
    }

    #[test]
    fn fs003_fail_empty() {
        let empty: [f64; 0] = [];
        assert_eq!(
            verdict_from_minibatch_unbiased(&empty, &empty),
            MiniBatchUnbiasedVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Realistic — full SGD rollout passes all 3.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_rollout_passes_all_3() {
        // SGD-001: fit() defaults to Batch, weights byte-identical.
        let w = vec![0.42, -0.17, 0.99];
        assert_eq!(
            verdict_from_backward_compat(FitMode::Batch, &w, &w),
            BackwardCompatVerdict::Pass
        );
        // SGD-002: stochastic mode achieves MCC=0.55 on imbalanced.
        assert_eq!(
            verdict_from_stochastic_imbalanced(0.55),
            StochasticImbalancedVerdict::Pass
        );
        // SGD-003: mini-batch(n) gradient matches full-batch within 1e-6.
        let full = vec![0.123456, 0.789012, -0.345678];
        let mb_n = vec![0.123456, 0.789012, -0.345678];
        assert_eq!(
            verdict_from_minibatch_unbiased(&full, &mb_n),
            MiniBatchUnbiasedVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_3_failures() {
        // SGD-001: someone changed default to Stochastic.
        let w = vec![0.5];
        assert_eq!(
            verdict_from_backward_compat(FitMode::Stochastic, &w, &w),
            BackwardCompatVerdict::Fail
        );
        // SGD-002: imbalanced classification regressed below 0.3.
        assert_eq!(
            verdict_from_stochastic_imbalanced(0.15),
            StochasticImbalancedVerdict::Fail
        );
        // SGD-003: mini-batch averaging diverged from full-batch.
        let full = vec![0.123];
        let mb_n = vec![0.456];
        assert_eq!(
            verdict_from_minibatch_unbiased(&full, &mb_n),
            MiniBatchUnbiasedVerdict::Fail
        );
    }
}
