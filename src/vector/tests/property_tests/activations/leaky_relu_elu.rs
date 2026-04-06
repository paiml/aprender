use super::super::super::super::*;
use proptest::prelude::*;

// ========================================================================
// Property tests for leaky_relu() - Leaky Rectified Linear Unit
// ========================================================================

proptest! {
    /// Property test: leaky_relu() preserves positive values exactly
    #[test]
    fn test_leaky_relu_preserves_positive_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.0f32..1.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.leaky_relu(alpha).unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val > 0.0 {
                prop_assert!(
                    (result.data[i] - val).abs() < 1e-6,
                    "Positive value {} should be preserved, got {}",
                    val, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: leaky_relu() scales negative values by alpha
    #[test]
    fn test_leaky_relu_scales_negative_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.01f32..0.5 // Use smaller range to avoid precision issues
    ) {
        let va = Vector::from_slice(&a);
        let result = va.leaky_relu(alpha).unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val < 0.0 {
                let expected = alpha * val;
                prop_assert!(
                    (result.data[i] - expected).abs() < 1e-4,
                    "Negative value {} should be scaled by {}: expected {}, got {}",
                    val, alpha, expected, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: leaky_relu() is monotonically increasing
    /// If x < y, then leaky_relu(x) < leaky_relu(y)
    #[test]
    fn test_leaky_relu_monotonic_property(
        a in prop::collection::vec(-50.0f32..50.0, 2..100),
        alpha_raw in 1u32..50
    ) {
        let alpha = alpha_raw as f32 / 100.0; // 0.01..0.50, avoids proptest float sampler panic
        let va = Vector::from_slice(&a);
        let result = va.leaky_relu(alpha).unwrap();

        // Check all pairs for monotonicity
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] < a[j] {
                    prop_assert!(
                        result.data[i] < result.data[j] + 1e-5,
                        "Monotonicity violated: {} < {} but leaky_relu({}) = {} >= leaky_relu({}) = {}",
                        a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                    );
                }
            }
        }
    }
}

// ========================================================================
// Property tests for elu() - Exponential Linear Unit
// ========================================================================

proptest! {
    /// Property test: elu() preserves positive values exactly
    #[test]
    fn test_elu_preserves_positive_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.1f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.elu(alpha).unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val > 0.0 {
                prop_assert!(
                    (result.data[i] - val).abs() < 1e-6,
                    "Positive value {} should be preserved, got {}",
                    val, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: elu() produces values >= -alpha for negative inputs
    /// ELU saturates to -alpha as x -> -inf
    #[test]
    fn test_elu_bounded_below_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.1f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.elu(alpha).unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val >= -alpha - 0.01,
                "ELU output {} should be >= -alpha = {}",
                val, -alpha
            );
        }
    }
}

proptest! {
    /// Property test: elu() is monotonically increasing
    /// If x < y, then elu(x) < elu(y)
    #[test]
    fn test_elu_monotonic_property(
        a in prop::collection::vec(-20.0f32..20.0, 2..50),
        alpha in 0.5f32..2.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.elu(alpha).unwrap();

        // Check all pairs for monotonicity
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] < a[j] {
                    prop_assert!(
                        result.data[i] < result.data[j] + 1e-5,
                        "Monotonicity violated: {} < {} but elu({}) = {} >= elu({}) = {}",
                        a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                    );
                }
            }
        }
    }
}
