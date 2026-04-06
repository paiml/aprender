use super::super::super::super::super::*;

#[test]
fn test_relu_basic() {
    // Basic ReLU: negative values -> 0, positive values unchanged
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.relu().unwrap();

    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_relu_all_negative() {
    // All negative values should become zero
    let v = Vector::from_slice(&[-5.0, -3.0, -1.0, -0.5]);
    let result = v.relu().unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0, "All negative values should become 0");
    }
}

#[test]
fn test_relu_all_positive() {
    // All positive values should remain unchanged
    let v = Vector::from_slice(&[0.5, 1.0, 3.0, 5.0]);
    let expected = v.clone();
    let result = v.relu().unwrap();

    for i in 0..v.len() {
        assert_eq!(result.data[i], expected.data[i], "Positive values should remain unchanged");
    }
}

#[test]
fn test_relu_zero_boundary() {
    // Zero should remain zero (boundary case)
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.relu().unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0, "Zero should remain zero");
    }
}

#[test]
fn test_relu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.relu();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_relu_sparsity() {
    // ReLU creates sparse activations (zeros for negative inputs)
    let v = Vector::from_slice(&[-10.0, 5.0, -3.0, 8.0, -1.0, 2.0]);
    let result = v.relu().unwrap();

    // Count zeros (should be 3)
    let zero_count = result.as_slice().iter().filter(|&&x| x == 0.0).count();
    assert_eq!(zero_count, 3, "ReLU should produce sparse activations");

    // Verify positive values preserved
    assert_eq!(result.data[1], 5.0);
    assert_eq!(result.data[3], 8.0);
    assert_eq!(result.data[5], 2.0);
}

#[test]
fn test_sigmoid_basic() {
    // Basic sigmoid: negative -> (0, 0.5), zero -> 0.5, positive -> (0.5, 1)
    let v = Vector::from_slice(&[-2.0, 0.0, 2.0]);
    let result = v.sigmoid().unwrap();

    // sigmoid(-2) ~= 0.1192, sigmoid(0) = 0.5, sigmoid(2) ~= 0.8808
    assert!((result.data[0] - 0.1192).abs() < 0.001);
    assert!((result.data[1] - 0.5).abs() < 0.001);
    assert!((result.data[2] - 0.8808).abs() < 0.001);
}

#[test]
fn test_sigmoid_range() {
    // All outputs should be in [0, 1] range (inclusive for numerical stability)
    let v = Vector::from_slice(&[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]);
    let result = v.sigmoid().unwrap();

    for &val in result.as_slice() {
        assert!((0.0..=1.0).contains(&val), "Sigmoid output {} not in [0, 1]", val);
    }
}

#[test]
fn test_sigmoid_symmetry() {
    // Test sigma(-x) = 1 - sigma(x)
    let v = Vector::from_slice(&[-3.0, -1.5, -0.5]);
    let v_neg = Vector::from_slice(&[3.0, 1.5, 0.5]);

    let sig = v.sigmoid().unwrap();
    let sig_neg = v_neg.sigmoid().unwrap();

    for i in 0..v.len() {
        let sum = sig.data[i] + sig_neg.data[i];
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Symmetry violated: sig({}) + sig({}) = {} + {} = {} != 1",
            v.data[i],
            v_neg.data[i],
            sig.data[i],
            sig_neg.data[i],
            sum
        );
    }
}

#[test]
fn test_sigmoid_extreme_values() {
    // Test numerical stability with extreme values
    let v = Vector::from_slice(&[-100.0, -50.0, 50.0, 100.0]);
    let result = v.sigmoid().unwrap();

    // Very negative -> close to 0
    assert!(result.data[0] < 1e-6, "sigmoid(-100) should be ~= 0");
    assert!(result.data[1] < 1e-6, "sigmoid(-50) should be ~= 0");

    // Very positive -> close to 1
    assert!(result.data[2] > 1.0 - 1e-6, "sigmoid(50) should be ~= 1");
    assert!(result.data[3] > 1.0 - 1e-6, "sigmoid(100) should be ~= 1");
}

#[test]
fn test_sigmoid_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.sigmoid();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_sigmoid_zero() {
    // sigmoid(0) should be exactly 0.5
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.sigmoid().unwrap();

    for &val in result.as_slice() {
        assert!((val - 0.5).abs() < 1e-7, "sigmoid(0) = {} != 0.5", val);
    }
}
