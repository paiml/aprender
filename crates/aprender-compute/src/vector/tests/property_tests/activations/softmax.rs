use super::super::super::super::*;
use proptest::prelude::*;

// ========================================================================
// Property tests for softmax() - Probability distribution
// ========================================================================

proptest! {
    /// Property test: softmax() produces values that sum to 1
    #[test]
    fn test_softmax_sums_to_one(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let probs = va.softmax().unwrap();
        let sum: f32 = probs.as_slice().iter().sum();

        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {}, expected 1.0",
            sum
        );
    }
}

proptest! {
    /// Property test: softmax() produces values in [0, 1]
    #[test]
    fn test_softmax_in_unit_range(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let probs = va.softmax().unwrap();

        for &p in probs.as_slice() {
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "probability {} not in [0, 1]",
                p
            );
        }
    }
}

proptest! {
    /// Property test: softmax() is translation invariant
    /// softmax(x + c) = softmax(x) for any constant c
    #[test]
    fn test_softmax_translation_invariant(
        a in prop::collection::vec(-20.0f32..20.0, 2..50),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let probs1 = va.softmax().unwrap();

        // Add constant to all elements
        let shifted: Vec<f32> = a.iter().map(|&x| x + c).collect();
        let vb = Vector::from_slice(&shifted);
        let probs2 = vb.softmax().unwrap();

        // Probabilities should be identical
        for i in 0..probs1.len() {
            prop_assert!(
                (probs1.data[i] - probs2.data[i]).abs() < 1e-4,
                "Translation invariance violated at index {}: softmax(x)={}, softmax(x+{})={}",
                i, probs1.data[i], c, probs2.data[i]
            );
        }
    }
}

// ========================================================================
// Property tests for log_softmax() - Log probability distribution
// ========================================================================

proptest! {
    /// Property test: exp(log_softmax(x)) sums to 1
    /// Since log_softmax returns log probabilities, exponentiating should give valid probabilities
    #[test]
    fn test_log_softmax_exp_sums_to_one(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let log_probs = va.log_softmax().unwrap();

        // Exponentiate to get probabilities
        let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();

        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "exp(log_softmax) sum = {}, expected 1.0",
            sum
        );
    }
}

proptest! {
    /// Property test: log_softmax() produces values <= 0
    /// Since probabilities are in [0, 1], log(prob) <= 0
    #[test]
    fn test_log_softmax_non_positive(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let log_probs = va.log_softmax().unwrap();

        for &lp in log_probs.as_slice() {
            prop_assert!(
                lp <= 1e-5,
                "log_probability {} should be <= 0",
                lp
            );
        }
    }
}

proptest! {
    /// Property test: log_softmax() is translation invariant
    /// log_softmax(x + c) = log_softmax(x) for any constant c
    #[test]
    fn test_log_softmax_translation_invariant(
        a in prop::collection::vec(-20.0f32..20.0, 2..50),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let log_probs1 = va.log_softmax().unwrap();

        // Add constant to all elements
        let shifted: Vec<f32> = a.iter().map(|&x| x + c).collect();
        let vb = Vector::from_slice(&shifted);
        let log_probs2 = vb.log_softmax().unwrap();

        // Log probabilities should be identical
        for i in 0..log_probs1.len() {
            prop_assert!(
                (log_probs1.data[i] - log_probs2.data[i]).abs() < 1e-4,
                "Translation invariance violated at index {}: log_softmax(x)={}, log_softmax(x+{})={}",
                i, log_probs1.data[i], c, log_probs2.data[i]
            );
        }
    }
}
