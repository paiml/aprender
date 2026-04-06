use super::super::super::super::*;
use proptest::prelude::*;

// ========================================
// Property Tests: sum_of_squares()
// ========================================

// Property test: non-negativity - sum_of_squares is always >= 0
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_of_squares_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sum_of_squares().unwrap();

        prop_assert!(
            result >= 0.0,
            "sum_of_squares should be non-negative: {} < 0",
            result
        );
    }
}

// Property test: equivalence with dot product - sum_of_squares(v) = dot(v, v)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_of_squares_equals_dot_self(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sum_sq = va.sum_of_squares().unwrap();
        let dot_self = va.dot(&va).unwrap();

        prop_assert!(
            (sum_sq - dot_self).abs() < 1e-4,
            "sum_of_squares should equal dot(self, self): {} != {}",
            sum_sq, dot_self
        );
    }
}

// Property test: scaling - sum_of_squares(k*v) = k^2 * sum_of_squares(v)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_of_squares_scaling(
        a in prop::collection::vec(-10.0f32..10.0, 1..50),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let scaled = va.scale(k).unwrap();

        let sum_sq_original = va.sum_of_squares().unwrap();
        let sum_sq_scaled = scaled.sum_of_squares().unwrap();
        let expected = k * k * sum_sq_original;

        // Use relative tolerance for larger values
        let tolerance = 1e-3 * expected.abs().max(1.0);
        prop_assert!(
            (sum_sq_scaled - expected).abs() < tolerance,
            "sum_of_squares({} * v) = {} != {}^2 * {} = {}",
            k, sum_sq_scaled, k, sum_sq_original, expected
        );
    }

    /// Property test: mean(v) is between min(v) and max(v)
    #[test]
    fn test_mean_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let mean_val = va.mean().unwrap();
        let min_val = va.min().unwrap();
        let max_val = va.max().unwrap();

        prop_assert!(
            mean_val >= min_val && mean_val <= max_val,
            "mean({}) = {} not in range [{}, {}]",
            mean_val, mean_val, min_val, max_val
        );
    }

    /// Property test: mean(v + c) = mean(v) + c (translation property)
    #[test]
    fn test_mean_translation(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let mean_original = va.mean().unwrap();

        // Create translated vector: v + c
        let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
        let vt = Vector::from_slice(&translated);
        let mean_translated = vt.mean().unwrap();

        let expected = mean_original + c;
        let tolerance = 1e-4 * expected.abs().max(1.0);
        prop_assert!(
            (mean_translated - expected).abs() < tolerance,
            "mean(v + {}) = {} != mean(v) + {} = {}",
            c, mean_translated, c, expected
        );
    }

    /// Property test: mean(k*v) = k*mean(v) (scaling property)
    #[test]
    fn test_mean_scaling(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let mean_original = va.mean().unwrap();
        let scaled = va.scale(k).unwrap();
        let mean_scaled = scaled.mean().unwrap();

        let expected = k * mean_original;
        let tolerance = 1e-4 * expected.abs().max(1.0);
        prop_assert!(
            (mean_scaled - expected).abs() < tolerance,
            "mean({} * v) = {} != {} * mean(v) = {}",
            k, mean_scaled, k, expected
        );
    }

    /// Property test: variance(v) >= 0 (non-negativity)
    #[test]
    fn test_variance_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let var = va.variance().unwrap();

        prop_assert!(
            var >= -1e-5, // Allow small numerical error
            "variance = {} should be non-negative",
            var
        );
    }

    /// Property test: variance(k*v) = k^2*variance(v) (scaling property)
    #[test]
    fn test_variance_scaling(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let var_original = va.variance().unwrap();
        let scaled = va.scale(k).unwrap();
        let var_scaled = scaled.variance().unwrap();

        let expected = k * k * var_original;
        // Use absolute tolerance for small values, relative for large
        // Note: f32 has ~7 significant digits; variance involves squaring which
        // doubles relative error, then k^2 multiplies it again. Use 0.5% tolerance.
        let tolerance = if expected.abs() < 1.0 {
            1e-2  // Absolute tolerance for small variance
        } else {
            5e-3 * expected.abs()  // 0.5% relative tolerance for large variance
        };
        prop_assert!(
            (var_scaled - expected).abs() < tolerance,
            "variance({} * v) = {} != {}^2 * variance(v) = {}",
            k, var_scaled, k, expected
        );
    }

    /// Property test: variance(v + c) = variance(v) (translation invariance)
    #[test]
    fn test_variance_translation_invariance(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let var_original = va.variance().unwrap();

        // Create translated vector: v + c
        let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
        let vt = Vector::from_slice(&translated);
        let var_translated = vt.variance().unwrap();

        // Use absolute tolerance for small values, relative for large
        let tolerance = if var_original.abs() < 1.0 {
            1e-2  // Absolute tolerance for small variance
        } else {
            1e-3 * var_original.abs()  // Relative tolerance for large variance
        };
        prop_assert!(
            (var_translated - var_original).abs() < tolerance,
            "variance(v + {}) = {} != variance(v) = {}",
            c, var_translated, var_original
        );
    }

    /// Property test: stddev(v) >= 0 (non-negativity)
    #[test]
    fn test_stddev_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sd = va.stddev().unwrap();

        prop_assert!(
            sd >= -1e-5, // Allow small numerical error
            "stddev = {} should be non-negative",
            sd
        );
    }

    /// Property test: stddev(k*v) = |k|*stddev(v) (linear scaling)
    #[test]
    fn test_stddev_scaling(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let sd_original = va.stddev().unwrap();
        let scaled = va.scale(k).unwrap();
        let sd_scaled = scaled.stddev().unwrap();

        let expected = k.abs() * sd_original;
        // Use absolute tolerance for small values, relative for large
        // Note: stddev is numerically unstable for very similar values due to
        // catastrophic cancellation in variance computation
        let tolerance = if expected.abs() < 1.0 {
            5e-2  // Absolute tolerance for small stddev (increased for numerical stability)
        } else {
            1e-3 * expected.abs()  // Relative tolerance for large stddev
        };
        prop_assert!(
            (sd_scaled - expected).abs() < tolerance,
            "stddev({} * v) = {} != |{}| * stddev(v) = {}",
            k, sd_scaled, k, expected
        );
    }

    /// Property test: stddev(v + c) = stddev(v) (translation invariance)
    #[test]
    fn test_stddev_translation_invariance(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let sd_original = va.stddev().unwrap();

        // Create translated vector: v + c
        let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
        let vt = Vector::from_slice(&translated);
        let sd_translated = vt.stddev().unwrap();

        // Use absolute tolerance for small values, relative for large
        let tolerance = if sd_original.abs() < 1.0 {
            1e-2  // Absolute tolerance for small stddev
        } else {
            1e-3 * sd_original.abs()  // Relative tolerance for large stddev
        };
        prop_assert!(
            (sd_translated - sd_original).abs() < tolerance,
            "stddev(v + {}) = {} != stddev(v) = {}",
            c, sd_translated, sd_original
        );
    }
}
