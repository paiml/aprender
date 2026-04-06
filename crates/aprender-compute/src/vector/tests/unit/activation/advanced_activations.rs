use super::super::super::super::*;

// ========================================================================
// Tests for gelu() - Gaussian Error Linear Unit
// ========================================================================

#[test]
fn test_gelu_basic() {
    // Basic GELU behavior
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.gelu().unwrap();

    // gelu(0) should be exactly 0
    assert_eq!(result.data[2], 0.0);

    // Negative values should give small negative outputs
    assert!(result.data[0] < 0.0 && result.data[0] > -0.1);
    assert!(result.data[1] < 0.0 && result.data[1] > -0.2);

    // Positive values should be positive and approach linear for large x
    assert!(result.data[3] > 0.8);
    assert!(result.data[4] > 1.8);
}

#[test]
fn test_gelu_zero() {
    // gelu(0) should be exactly 0
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.gelu().unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0, "gelu(0) should be 0");
    }
}

#[test]
fn test_gelu_smoothness() {
    // GELU is smooth everywhere - test that it produces reasonable values
    let v = Vector::from_slice(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let result = v.gelu().unwrap();

    // All outputs should be finite
    for &val in result.as_slice() {
        assert!(val.is_finite(), "GELU output should be finite");
    }

    // Verify increasing trend (though not strictly monotonic)
    // Generally gelu increases with x
    assert!(result.data[0] < result.data[3]); // gelu(-3) < gelu(0)
    assert!(result.data[3] < result.data[6]); // gelu(0) < gelu(3)
}

#[test]
fn test_gelu_large_positive() {
    // For large positive x, gelu(x) ≈ x (linear behavior)
    let v = Vector::from_slice(&[5.0, 10.0, 20.0]);
    let result = v.gelu().unwrap();

    for i in 0..v.len() {
        // Should be very close to x for large positive values
        assert!(
            (result.data[i] - v.data[i]).abs() < 0.01,
            "gelu({}) = {} should ≈ {} for large positive x",
            v.data[i],
            result.data[i],
            v.data[i]
        );
    }
}

#[test]
fn test_gelu_large_negative() {
    // For large negative x, gelu(x) ≈ 0
    let v = Vector::from_slice(&[-5.0, -10.0, -20.0]);
    let result = v.gelu().unwrap();

    for &val in result.as_slice() {
        assert!(val.abs() < 0.001, "gelu should approach 0 for large negative inputs, got {}", val);
    }
}

#[test]
fn test_gelu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.gelu();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ============================================================================
// Swish (SiLU) Tests
// ============================================================================

#[test]
fn test_swish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.swish().unwrap();

    // swish(-2) ≈ -0.238, swish(-1) ≈ -0.269, swish(0) = 0
    // swish(1) ≈ 0.731, swish(2) ≈ 1.762
    assert!((result.as_slice()[0] - (-0.238)).abs() < 0.01);
    assert!((result.as_slice()[1] - (-0.269)).abs() < 0.01);
    assert_eq!(result.as_slice()[2], 0.0);
    assert!((result.as_slice()[3] - 0.731).abs() < 0.01);
    assert!((result.as_slice()[4] - 1.762).abs() < 0.01);
}

#[test]
fn test_swish_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.swish().unwrap();
    assert_eq!(result.as_slice()[0], 0.0); // swish(0) = 0
}

#[test]
fn test_swish_minimum() {
    // Swish has a minimum value around x ≈ -1.278, value ≈ -0.278
    let v = Vector::from_slice(&[-2.0, -1.5, -1.278, -1.0, -0.5]);
    let result = v.swish().unwrap();

    // All values should be above the minimum
    for &val in result.as_slice() {
        assert!(val > -0.3, "Swish value {} below minimum", val);
    }

    // The middle value (closest to -1.278) should be near the minimum
    assert!(result.as_slice()[2] < -0.27);
    assert!(result.as_slice()[2] > -0.29);
}

#[test]
fn test_swish_large_positive() {
    // For large positive x, swish(x) ≈ x (linear behavior)
    let v = Vector::from_slice(&[10.0, 20.0, 50.0]);
    let result = v.swish().unwrap();

    assert!((result.as_slice()[0] - 10.0).abs() < 0.01);
    assert!((result.as_slice()[1] - 20.0).abs() < 0.01);
    assert!((result.as_slice()[2] - 50.0).abs() < 0.01);
}

#[test]
fn test_swish_large_negative() {
    // For large negative x, swish(x) ≈ 0
    let v = Vector::from_slice(&[-10.0, -20.0, -50.0]);
    let result = v.swish().unwrap();

    // swish(-10) ≈ -0.000454, swish(-20) ≈ -4.1e-9, swish(-50) ≈ 0
    assert!(result.as_slice()[0].abs() < 1e-3);
    assert!(result.as_slice()[1].abs() < 1e-7);
    assert!(result.as_slice()[2].abs() < 1e-15); // Effectively 0
}

#[test]
fn test_swish_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.swish();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ============================================================================
// Hardswish activation tests
// ============================================================================

#[test]
fn test_hardswish_basic() {
    let v = Vector::from_slice(&[-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]);
    let result = v.hardswish().unwrap();

    // x <= -3: 0
    assert_eq!(result.as_slice()[0], 0.0);
    assert_eq!(result.as_slice()[1], 0.0);

    // -3 < x < 3: x * (x + 3) / 6
    // hardswish(-1.5) = -1.5 * 1.5 / 6 = -0.375
    assert!((result.as_slice()[2] - (-0.375)).abs() < 1e-5);
    // hardswish(0) = 0 * 3 / 6 = 0
    assert_eq!(result.as_slice()[3], 0.0);
    // hardswish(1.5) = 1.5 * 4.5 / 6 = 1.125
    assert!((result.as_slice()[4] - 1.125).abs() < 1e-5);

    // x >= 3: x
    assert_eq!(result.as_slice()[5], 3.0);
    assert_eq!(result.as_slice()[6], 4.0);
}

#[test]
fn test_hardswish_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.hardswish().unwrap();
    assert_eq!(result.as_slice()[0], 0.0);
}

#[test]
fn test_hardswish_boundary_values() {
    // Test exact boundary values
    let v = Vector::from_slice(&[-3.0, 3.0]);
    let result = v.hardswish().unwrap();

    // At x = -3: 0 (boundary)
    assert_eq!(result.as_slice()[0], 0.0);
    // At x = 3: 3 (boundary)
    assert_eq!(result.as_slice()[1], 3.0);
}

#[test]
fn test_hardswish_large_values() {
    let v = Vector::from_slice(&[-100.0, -10.0, 10.0, 100.0]);
    let result = v.hardswish().unwrap();

    // Large negative: 0
    assert_eq!(result.as_slice()[0], 0.0);
    assert_eq!(result.as_slice()[1], 0.0);

    // Large positive: x
    assert_eq!(result.as_slice()[2], 10.0);
    assert_eq!(result.as_slice()[3], 100.0);
}

#[test]
fn test_hardswish_transition_region() {
    // Test values in the transition region (-3, 3)
    let v = Vector::from_slice(&[-2.0, -1.0, 1.0, 2.0]);
    let result = v.hardswish().unwrap();

    // hardswish(-2) = -2 * 1 / 6 = -0.333...
    assert!((result.as_slice()[0] - (-1.0 / 3.0)).abs() < 1e-5);
    // hardswish(-1) = -1 * 2 / 6 = -0.333...
    assert!((result.as_slice()[1] - (-1.0 / 3.0)).abs() < 1e-5);
    // hardswish(1) = 1 * 4 / 6 = 0.666...
    assert!((result.as_slice()[2] - (2.0 / 3.0)).abs() < 1e-5);
    // hardswish(2) = 2 * 5 / 6 = 1.666...
    assert!((result.as_slice()[3] - (5.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_hardswish_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.hardswish();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ============================================================================
// Mish activation tests
// ============================================================================

#[test]
fn test_mish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.mish().unwrap();

    // mish has small negative values for negative inputs
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);

    // mish(0) is a small positive value (0 * tanh(ln(2)) = 0)
    assert!(result.as_slice()[2].abs() < 1e-5);

    // Positive inputs give positive outputs
    assert!(result.as_slice()[3] > 0.0);
    assert!(result.as_slice()[4] > 0.0);
}

#[test]
fn test_mish_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.mish().unwrap();
    // mish(0) = 0 * tanh(ln(2)) = 0
    assert!(result.as_slice()[0].abs() < 1e-10);
}

#[test]
fn test_mish_large_positive() {
    // For large positive x, mish(x) ≈ x
    let v = Vector::from_slice(&[10.0, 20.0, 50.0]);
    let result = v.mish().unwrap();

    // Should be very close to x for large values
    assert!((result.as_slice()[0] - 10.0).abs() < 0.001);
    assert!((result.as_slice()[1] - 20.0).abs() < 0.001);
    assert!((result.as_slice()[2] - 50.0).abs() < 0.001);
}

#[test]
fn test_mish_large_negative() {
    // For large negative x, mish(x) ≈ 0
    let v = Vector::from_slice(&[-10.0, -20.0, -50.0]);
    let result = v.mish().unwrap();

    // Should be very close to 0 for large negative values
    assert!(result.as_slice()[0].abs() < 0.001);
    assert!(result.as_slice()[1].abs() < 1e-6);
    assert!(result.as_slice()[2].abs() < 1e-10);
}

#[test]
fn test_mish_minimum() {
    // Mish has a minimum around x ≈ -1.19 with value ≈ -0.31
    let v = Vector::from_slice(&[-1.19]);
    let result = v.mish().unwrap();

    // Should be close to the minimum value
    assert!(result.as_slice()[0] < -0.2);
    assert!(result.as_slice()[0] > -0.4);
}

#[test]
fn test_mish_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.mish();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ============================================================================
// SELU unit tests
// ============================================================================

#[test]
fn test_selu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.selu().unwrap();
    let data = result.as_slice();

    // SELU constants
    const LAMBDA: f32 = 1.0507009873554804934193349852946;
    const ALPHA: f32 = 1.6732632423543772848170429916717;

    // Positive values: selu(x) = λ * x
    assert!((data[3] - LAMBDA * 1.0).abs() < 1e-5); // selu(1.0) = λ
    assert!((data[4] - LAMBDA * 2.0).abs() < 1e-5); // selu(2.0) = 2λ

    // Zero: selu(0) = 0
    assert!(data[2].abs() < 1e-5);

    // Negative values: selu(x) = λ * α * (exp(x) - 1)
    let expected_neg1 = LAMBDA * ALPHA * ((-1.0_f32).exp() - 1.0);
    assert!((data[1] - expected_neg1).abs() < 1e-5);
}

#[test]
fn test_selu_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.selu().unwrap();
    assert!(result.as_slice()[0].abs() < 1e-10);
}

#[test]
fn test_selu_positive_scaling() {
    // For positive values, selu(x) = λ * x
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 10.0]);
    let result = v.selu().unwrap();
    let data = result.as_slice();

    const LAMBDA: f32 = 1.0507009873554804934193349852946;

    for (i, &x) in [1.0, 2.0, 3.0, 10.0].iter().enumerate() {
        assert!(
            (data[i] - LAMBDA * x).abs() < 1e-5,
            "selu({}) should be {} but got {}",
            x,
            LAMBDA * x,
            data[i]
        );
    }
}

#[test]
fn test_selu_negative_asymptote() {
    // For very negative x, selu(x) → -λ * α ≈ -1.7581
    let v = Vector::from_slice(&[-100.0]);
    let result = v.selu().unwrap();

    const LAMBDA: f32 = 1.0507009873554804934193349852946;
    const ALPHA: f32 = 1.6732632423543772848170429916717;
    let asymptote = -LAMBDA * ALPHA;

    assert!(
        (result.as_slice()[0] - asymptote).abs() < 1e-4,
        "selu(-100) should approach {} but got {}",
        asymptote,
        result.as_slice()[0]
    );
}

#[test]
fn test_selu_continuity_at_zero() {
    // Test values approaching zero from both sides
    let eps = 1e-6;
    let v = Vector::from_slice(&[-eps, 0.0, eps]);
    let result = v.selu().unwrap();
    let data = result.as_slice();

    // All should be very close to zero (continuous at x=0)
    assert!(data[0].abs() < 1e-3);
    assert!(data[1].abs() < 1e-10);
    assert!(data[2].abs() < 1e-3);
}

#[test]
fn test_selu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.selu();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}
