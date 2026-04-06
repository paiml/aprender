use super::super::super::super::super::*;

#[test]
fn test_leaky_relu_basic() {
    // Basic Leaky ReLU with alpha = 0.01
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.01).unwrap();

    assert_eq!(result.as_slice(), &[-0.02, -0.01, 0.0, 1.0, 2.0]);
}

#[test]
fn test_leaky_relu_different_slopes() {
    // Test with different negative slopes
    let v = Vector::from_slice(&[-10.0, 5.0]);

    // alpha = 0.01 (default)
    let result_001 = v.leaky_relu(0.01).unwrap();
    assert!((result_001.data[0] - (-0.1)).abs() < 1e-6); // -10 * 0.01
    assert_eq!(result_001.data[1], 5.0);

    // alpha = 0.1
    let result_01 = v.leaky_relu(0.1).unwrap();
    assert!((result_01.data[0] - (-1.0)).abs() < 1e-6); // -10 * 0.1
    assert_eq!(result_01.data[1], 5.0);

    // alpha = 0.2
    let result_02 = v.leaky_relu(0.2).unwrap();
    assert!((result_02.data[0] - (-2.0)).abs() < 1e-6); // -10 * 0.2
    assert_eq!(result_02.data[1], 5.0);
}

#[test]
fn test_leaky_relu_reduces_to_relu() {
    // With alpha = 0, should behave like standard ReLU
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let leaky = v.leaky_relu(0.0).unwrap();
    let relu = v.relu().unwrap();

    for i in 0..v.len() {
        assert_eq!(leaky.data[i], relu.data[i], "alpha=0 should equal ReLU");
    }
}

#[test]
fn test_leaky_relu_preserves_positive() {
    // Positive values should remain unchanged regardless of alpha
    let v = Vector::from_slice(&[0.5, 1.0, 5.0, 10.0]);
    let result = v.leaky_relu(0.01).unwrap();

    for i in 0..v.len() {
        assert_eq!(result.data[i], v.data[i], "Positive values should be preserved");
    }
}

#[test]
fn test_leaky_relu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.leaky_relu(0.01);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_leaky_relu_invalid_slope() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);

    // Negative slope should fail
    let result = v.leaky_relu(-0.1);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

    // Slope >= 1.0 should fail
    let result = v.leaky_relu(1.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

    let result = v.leaky_relu(1.5);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_elu_basic() {
    // Basic ELU with alpha = 1.0
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.elu(1.0).unwrap();

    // elu(-2, 1) = 1*(e^-2 - 1) ~= -0.8647
    // elu(-1, 1) = 1*(e^-1 - 1) ~= -0.6321
    assert!((result.data[0] - (-0.8647)).abs() < 0.001);
    assert!((result.data[1] - (-0.6321)).abs() < 0.001);
    assert_eq!(result.data[2], 0.0);
    assert_eq!(result.data[3], 1.0);
    assert_eq!(result.data[4], 2.0);
}

#[test]
fn test_elu_different_alphas() {
    // Test with different alpha values
    let v = Vector::from_slice(&[-1.0, 2.0]);

    // alpha = 1.0 (standard)
    let result_1 = v.elu(1.0).unwrap();
    assert!((result_1.data[0] - (-0.6321)).abs() < 0.001);
    assert_eq!(result_1.data[1], 2.0);

    // alpha = 0.5
    let result_05 = v.elu(0.5).unwrap();
    assert!((result_05.data[0] - (-0.3161)).abs() < 0.001); // 0.5 * (e^-1 - 1)
    assert_eq!(result_05.data[1], 2.0);

    // alpha = 2.0
    let result_2 = v.elu(2.0).unwrap();
    assert!((result_2.data[0] - (-1.2642)).abs() < 0.001); // 2.0 * (e^-1 - 1)
    assert_eq!(result_2.data[1], 2.0);
}

#[test]
fn test_elu_saturation() {
    // For very negative values, ELU saturates to -alpha
    let v = Vector::from_slice(&[-10.0, -20.0, -100.0]);
    let result = v.elu(1.0).unwrap();

    // All should be very close to -1.0 (saturation at -alpha)
    for &val in result.as_slice() {
        assert!(
            (val - (-1.0)).abs() < 0.001,
            "ELU should saturate to -alpha for very negative inputs, got {}",
            val
        );
    }
}

#[test]
fn test_elu_preserves_positive() {
    // Positive values should remain unchanged
    let v = Vector::from_slice(&[0.5, 1.0, 5.0, 10.0]);
    let result = v.elu(1.0).unwrap();

    for i in 0..v.len() {
        assert_eq!(result.data[i], v.data[i], "Positive values should be preserved");
    }
}

#[test]
fn test_elu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.elu(1.0);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_elu_invalid_alpha() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);

    // Alpha <= 0 should fail
    let result = v.elu(0.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

    let result = v.elu(-1.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}
