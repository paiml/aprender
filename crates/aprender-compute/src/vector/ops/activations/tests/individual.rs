use super::*;

// ========== ReLU ==========

#[test]
fn test_relu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.relu().unwrap();
    let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "relu[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_relu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.relu(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_relu_all_negative() {
    let v = Vector::from_slice(&[-5.0, -3.0, -1.0]);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_relu_all_positive() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.relu().unwrap();
    let expected = [1.0, 2.0, 3.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "relu[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_relu_large() {
    let v = Vector::from_slice(&[-1.0; 1000]);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// ========== Sigmoid ==========

#[test]
fn test_sigmoid_basic() {
    let v = Vector::from_slice(&[-10.0, 0.0, 10.0]);
    let result = v.sigmoid().unwrap();
    assert!(result.as_slice()[0] < 0.001);
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-6);
    assert!(result.as_slice()[2] > 0.999);
}

#[test]
fn test_sigmoid_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.sigmoid(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_sigmoid_range() {
    let v = Vector::from_slice(&[-100.0, -1.0, 0.0, 1.0, 100.0]);
    let result = v.sigmoid().unwrap();
    for &val in result.as_slice() {
        assert!((0.0..=1.0).contains(&val));
    }
}

#[test]
fn test_sigmoid_large() {
    let v = Vector::from_slice(&[0.0; 1000]);
    let result = v.sigmoid().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.5).abs() < 1e-6);
    }
}

// ========== Leaky ReLU ==========

#[test]
fn test_leaky_relu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.01).unwrap();
    let expected = [-0.02, -0.01, 0.0, 1.0, 2.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "leaky_relu[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_leaky_relu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.leaky_relu(0.01), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_leaky_relu_different_slopes() {
    let v = Vector::from_slice(&[-1.0]);
    let r1 = v.leaky_relu(0.1).unwrap();
    assert!((r1.as_slice()[0] - (-0.1)).abs() < 1e-6);
    let r2 = v.leaky_relu(0.2).unwrap();
    assert!((r2.as_slice()[0] - (-0.2)).abs() < 1e-6);
}

#[test]
fn test_leaky_relu_slope_zero_acts_like_relu() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.0).unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_leaky_relu_invalid_slopes() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    assert!(matches!(v.leaky_relu(-0.1), Err(TruenoError::InvalidInput(_))));
    assert!(matches!(v.leaky_relu(1.0), Err(TruenoError::InvalidInput(_))));
    assert!(matches!(v.leaky_relu(1.5), Err(TruenoError::InvalidInput(_))));
}

// ========== ELU ==========

#[test]
fn test_elu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.elu(1.0).unwrap();
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);
}

#[test]
fn test_elu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.elu(1.0), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_elu_invalid_alpha() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    assert!(matches!(v.elu(0.0), Err(TruenoError::InvalidInput(_))));
    assert!(matches!(v.elu(-1.0), Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_elu_different_alpha_values() {
    let v = Vector::from_slice(&[-1.0]);
    let r1 = v.elu(0.5).unwrap();
    let r2 = v.elu(2.0).unwrap();
    assert!(r2.as_slice()[0] < r1.as_slice()[0]);
    assert!((r1.as_slice()[0] - 0.5 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
    assert!((r2.as_slice()[0] - 2.0 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
}

// ========== GELU ==========

#[test]
fn test_gelu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    assert!(result.as_slice()[3] > 0.5);
    assert!(result.as_slice()[4] > 1.5);
    assert!(result.as_slice()[0].abs() < 0.1);
}

#[test]
fn test_gelu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.gelu(), Err(TruenoError::EmptyVector)));
}

// ========== Swish ==========

#[test]
fn test_swish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 0.731).abs() < 0.01);
}

#[test]
fn test_swish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.swish(), Err(TruenoError::EmptyVector)));
}

// ========== Hardswish ==========

#[test]
fn test_hardswish_basic() {
    let v = Vector::from_slice(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
    let result = v.hardswish().unwrap();
    let expected = [0.0, 0.0, 0.0, 3.0, 4.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "hardswish[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_hardswish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.hardswish(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_hardswish_mid_range() {
    let inputs = [-2.0, -1.0, 1.0, 2.0];
    let v = Vector::from_slice(&inputs);
    let result = v.hardswish().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let x = inputs[i];
        let expected = x * (x + 3.0) / 6.0;
        assert!((val - expected).abs() < 1e-5, "hardswish({x}) = {val}, expected {expected}");
    }
}

#[test]
fn test_hardswish_continuity_at_boundaries() {
    let v = Vector::from_slice(&[-3.001, -3.0, -2.999]);
    let result = v.hardswish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-3);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    let v2 = Vector::from_slice(&[2.999, 3.0, 3.001]);
    let result2 = v2.hardswish().unwrap();
    assert!((result2.as_slice()[1] - 3.0).abs() < 1e-5);
    assert!((result2.as_slice()[2] - 3.001).abs() < 1e-5);
}

// ========== Mish ==========

#[test]
fn test_mish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.mish().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!(result.as_slice()[0] < 0.0);
}

#[test]
fn test_mish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.mish(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_mish_extreme_values() {
    // Very negative → 0
    let v = Vector::from_slice(&[-25.0, -30.0, -50.0]);
    let result = v.mish().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6, "mish of very negative should be 0, got {val}");
    }
    // Very positive → x
    let v2 = Vector::from_slice(&[25.0, 30.0, 50.0]);
    let result2 = v2.mish().unwrap();
    for (i, &val) in result2.as_slice().iter().enumerate() {
        let input = [25.0, 30.0, 50.0][i];
        assert!(
            (val - input).abs() < 1e-4,
            "mish of very positive should be x, got {val} for {input}"
        );
    }
    // Boundary values (no panics, finite)
    let v3 = Vector::from_slice(&[-20.0, -19.9, 19.9, 20.0]);
    let result3 = v3.mish().unwrap();
    for &val in result3.as_slice() {
        assert!(val.is_finite(), "mish should produce finite results at boundaries");
    }
}

// ========== SELU ==========

#[test]
fn test_selu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.selu().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    assert!((result.as_slice()[3] - 1.0507).abs() < 0.001);
    assert!((result.as_slice()[4] - 2.1014).abs() < 0.001);
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);
}

#[test]
fn test_selu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.selu(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_selu_specific_values() {
    // lambda * 1.0 = 1.0507
    let v1 = Vector::from_slice(&[1.0]);
    assert!((v1.selu().unwrap().as_slice()[0] - 1.0507).abs() < 0.001);
    // Negative: lambda * alpha * (exp(-1) - 1) ≈ -1.1113
    let v2 = Vector::from_slice(&[-1.0]);
    assert!((v2.selu().unwrap().as_slice()[0] - (-1.1113)).abs() < 0.01);
    // Very negative: ≈ -lambda * alpha ≈ -1.7581
    let v3 = Vector::from_slice(&[-50.0]);
    assert!((v3.selu().unwrap().as_slice()[0] - (-1.7581)).abs() < 0.01);
}
