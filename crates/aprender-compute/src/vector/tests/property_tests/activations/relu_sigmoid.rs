use super::super::super::super::*;
use proptest::prelude::*;

// ========================================================================
// Property tests for relu() - Rectified Linear Unit
// ========================================================================

proptest! {
    /// Property test: relu() produces non-negative outputs
    /// All outputs should be >= 0
    #[test]
    fn test_relu_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.relu().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val >= 0.0,
                "ReLU output {} should be non-negative",
                val
            );
        }
    }
}

proptest! {
    /// Property test: relu() preserves positive values
    /// For all x > 0, relu(x) = x
    #[test]
    fn test_relu_preserves_positive(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.relu().unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val > 0.0 {
                prop_assert!(
                    (result.data[i] - val).abs() < 1e-6,
                    "ReLU should preserve positive value: {} became {}",
                    val, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: relu() is idempotent
    /// relu(relu(x)) = relu(x)
    #[test]
    fn test_relu_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let relu1 = va.relu().unwrap();
        let relu2 = relu1.relu().unwrap();

        for (i, &orig_val) in a.iter().enumerate() {
            prop_assert!(
                (relu1.data[i] - relu2.data[i]).abs() < 1e-6,
                "ReLU should be idempotent: relu(relu({})) = {} != relu({}) = {}",
                orig_val, relu2.data[i], orig_val, relu1.data[i]
            );
        }
    }
}

// ========================================================================
// Property tests for sigmoid() - Logistic activation
// ========================================================================

proptest! {
    /// Property test: sigmoid() produces values in [0, 1]
    #[test]
    fn test_sigmoid_bounded(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sigmoid().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                (0.0..=1.0).contains(&val),
                "Sigmoid output {} not in [0, 1]",
                val
            );
        }
    }
}

proptest! {
    /// Property test: sigmoid() symmetry: sigma(-x) = 1 - sigma(x)
    #[test]
    fn test_sigmoid_symmetry_property(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sig_pos = va.sigmoid().unwrap();

        // Create negated vector
        let a_neg: Vec<f32> = a.iter().map(|&x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let sig_neg = va_neg.sigmoid().unwrap();

        // sigma(-x) + sigma(x) should equal 1
        for (i, &val) in a.iter().enumerate() {
            let sum = sig_pos.data[i] + sig_neg.data[i];
            prop_assert!(
                (sum - 1.0).abs() < 1e-5,
                "Symmetry violated: sigma({}) + sigma({}) = {} + {} = {} != 1",
                val, -val, sig_pos.data[i], sig_neg.data[i], sum
            );
        }
    }
}

proptest! {
    /// Property test: sigmoid() is monotonically increasing
    /// If x < y, then sigma(x) < sigma(y)
    #[test]
    fn test_sigmoid_monotonic(
        a in prop::collection::vec(-50.0f32..50.0, 2..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sigmoid().unwrap();

        // Check all pairs for monotonicity
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] < a[j] {
                    prop_assert!(
                        result.data[i] < result.data[j] + 1e-6,
                        "Monotonicity violated: {} < {} but sigma({}) = {} >= sigma({}) = {}",
                        a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                    );
                }
            }
        }
    }
}
