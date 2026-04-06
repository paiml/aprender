use super::super::super::super::*;
use proptest::prelude::*;

// ========================================================================
// Property tests for gelu() - Gaussian Error Linear Unit
// ========================================================================

proptest! {
    /// Property test: gelu() produces finite values
    #[test]
    fn test_gelu_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.gelu().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val.is_finite(),
                "GELU output {} should be finite",
                val
            );
        }
    }
}

proptest! {
    /// Property test: gelu(0) = 0
    #[test]
    fn test_gelu_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.gelu().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-10,
            "gelu(0) should be 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For large positive x, gelu(x) ≈ x
    #[test]
    fn test_gelu_linear_large_positive(
        a in prop::collection::vec(5.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.gelu().unwrap();

        for (i, &val) in a.iter().enumerate() {
            // For large positive values, gelu(x) should be very close to x
            prop_assert!(
                (result.data[i] - val).abs() < 0.01,
                "For large positive {}, gelu should ≈ x, got {} vs {}",
                val, result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: swish() produces finite values
    #[test]
    fn test_swish_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.swish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "Swish output should be finite");
        }
    }
}

proptest! {
    /// Property test: swish(0) = 0 always
    #[test]
    fn test_swish_zero_property(
        a in prop::collection::vec(-0.001f32..0.001, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.swish().unwrap();

        // For values very close to 0, swish should also be close to 0
        for &val in result.as_slice() {
            prop_assert!(
                val.abs() < 0.001,
                "Swish of near-zero should be near-zero, got {}",
                val
            );
        }
    }
}

proptest! {
    /// Property test: For large positive x, swish(x) ≈ x (linear)
    #[test]
    fn test_swish_linear_large_positive(
        a in prop::collection::vec(10.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.swish().unwrap();

        for (i, &val) in a.iter().enumerate() {
            // For large positive values, swish(x) should be very close to x
            prop_assert!(
                (result.data[i] - val).abs() < 0.01,
                "For large positive {}, swish should ≈ x, got {} vs {}",
                val, result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: hardswish() produces finite values
    #[test]
    fn test_hardswish_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "Hardswish output should be finite");
        }
    }
}

proptest! {
    /// Property test: hardswish(0) = 0 always
    #[test]
    fn test_hardswish_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.hardswish().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-10,
            "hardswish(0) should be 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For x >= 3, hardswish(x) = x (identity)
    #[test]
    fn test_hardswish_identity_large_positive(
        a in prop::collection::vec(3.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for (i, &val) in a.iter().enumerate() {
            prop_assert!(
                (result.data[i] - val).abs() < 1e-5,
                "For x >= 3, hardswish(x) should = x, got {} vs {}",
                result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: For x <= -3, hardswish(x) = 0
    #[test]
    fn test_hardswish_zero_large_negative(
        a in prop::collection::vec(-100.0f32..-3.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val.abs() < 1e-10,
                "For x <= -3, hardswish(x) should = 0, got {}",
                val
            );
        }
    }
}

proptest! {
    /// Property test: hardswish matches formula in transition region
    #[test]
    fn test_hardswish_transition_property(
        a in prop::collection::vec(-2.999f32..2.999, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for (i, &x) in a.iter().enumerate() {
            let expected = x * (x + 3.0) / 6.0;
            prop_assert!(
                (result.data[i] - expected).abs() < 1e-5,
                "hardswish({}) should = {} * ({} + 3) / 6 = {}, got {}",
                x, x, x, expected, result.data[i]
            );
        }
    }
}

proptest! {
    /// Property test: mish() produces finite values
    #[test]
    fn test_mish_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "Mish output should be finite");
        }
    }
}

proptest! {
    /// Property test: mish(0) ≈ 0 (mish(0) = 0 * tanh(softplus(0)) = 0)
    #[test]
    fn test_mish_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.mish().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-5,
            "mish(0) should be ≈ 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For large positive x, mish(x) ≈ x (linear)
    #[test]
    fn test_mish_linear_large_positive(
        a in prop::collection::vec(20.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for (i, &val) in a.iter().enumerate() {
            // For large positive values, mish(x) should be very close to x
            // since softplus(x) → x and tanh(x) → 1
            prop_assert!(
                (result.data[i] - val).abs() < 0.01,
                "For large positive {}, mish should ≈ x, got {} vs {}",
                val, result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: For large negative x, mish(x) → 0
    #[test]
    fn test_mish_zero_large_negative(
        a in prop::collection::vec(-100.0f32..-20.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val.abs() < 1e-5,
                "For large negative x, mish(x) should → 0, got {}",
                val
            );
        }
    }
}

proptest! {
    /// Property test: mish has negative region (unlike ReLU)
    /// mish(x) can be slightly negative for x in (-1.5, 0)
    #[test]
    fn test_mish_negative_region_property(
        a in prop::collection::vec(-1.0f32..-0.1, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for (i, &x) in a.iter().enumerate() {
            // Mish should produce negative values in this range
            // The minimum of mish is approximately -0.31 at x ≈ -1.07
            prop_assert!(
                result.data[i] < 0.0,
                "mish({}) should be negative in (-1, -0.1), got {}",
                x, result.data[i]
            );
        }
    }
}

proptest! {
    /// Property test: selu() produces finite values
    #[test]
    fn test_selu_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.selu().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "SELU output should be finite");
        }
    }
}

proptest! {
    /// Property test: selu(0) = 0 always
    #[test]
    fn test_selu_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.selu().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-10,
            "selu(0) should be 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For positive x, selu(x) = λ * x (linear scaling)
    #[test]
    fn test_selu_linear_positive(
        a in prop::collection::vec(0.001f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.selu().unwrap();

        const LAMBDA: f32 = 1.0507009873554804934193349852946;

        for (i, &val) in a.iter().enumerate() {
            let expected = LAMBDA * val;
            prop_assert!(
                (result.data[i] - expected).abs() < 1e-4,
                "For positive {}, selu should = λ*x = {}, got {}",
                val, expected, result.data[i]
            );
        }
    }
}

proptest! {
    /// Property test: For large negative x, selu(x) → -λ * α ≈ -1.7581
    #[test]
    fn test_selu_asymptote_negative(
        a in prop::collection::vec(-100.0f32..-20.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.selu().unwrap();

        const LAMBDA: f32 = 1.0507009873554804934193349852946;
        const ALPHA: f32 = 1.6732632423543772848170429916717;
        let asymptote = -LAMBDA * ALPHA;

        for &val in result.as_slice() {
            prop_assert!(
                (val - asymptote).abs() < 1e-3,
                "For large negative x, selu should → {}, got {}",
                asymptote, val
            );
        }
    }
}

proptest! {
    /// Property test: selu is monotonically increasing
    #[test]
    fn test_selu_monotonic_property(
        a in prop::collection::vec(-10.0f32..10.0, 2..50)
    ) {
        let mut sorted = a.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let va = Vector::from_slice(&sorted);
        let result = va.selu().unwrap();

        for i in 1..result.data.len() {
            prop_assert!(
                result.data[i] >= result.data[i-1] - 1e-5,
                "selu should be monotonic: selu({}) = {} >= selu({}) = {}",
                sorted[i], result.data[i], sorted[i-1], result.data[i-1]
            );
        }
    }
}
