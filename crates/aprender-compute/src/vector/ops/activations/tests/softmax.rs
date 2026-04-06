use super::*;

#[test]
fn test_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    for &val in result.as_slice() {
        assert!((0.0..=1.0).contains(&val));
    }
    assert!(result.as_slice()[2] > result.as_slice()[1]);
    assert!(result.as_slice()[1] > result.as_slice()[0]);
}

#[test]
fn test_softmax_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.softmax(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_softmax_single() {
    let v = Vector::from_slice(&[5.0]);
    let result = v.softmax().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_uniform() {
    let v = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let result = v.softmax().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.25).abs() < 1e-6);
    }
}

#[test]
fn test_softmax_large_values() {
    let v = Vector::from_slice(&[1000.0, 1001.0, 1002.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_very_negative_values() {
    let v = Vector::from_slice(&[-1000.0, -999.0, -998.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_softmax_large() {
    let v = Vector::from_slice(&[1.0; 100]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    for &val in result.as_slice() {
        assert!((val - 0.01).abs() < 1e-4);
    }
}

#[test]
fn test_softmax_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ========== Log Softmax ==========

#[test]
fn test_log_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0);
    }
}

#[test]
fn test_log_softmax_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.log_softmax(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_log_softmax_single() {
    let v = Vector::from_slice(&[5.0]);
    let result = v.log_softmax().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_log_softmax_consistency_with_softmax() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let softmax = v.softmax().unwrap();
    let log_softmax = v.log_softmax().unwrap();
    for (i, &ls) in log_softmax.as_slice().iter().enumerate() {
        let from_log = ls.exp();
        assert!(
            (from_log - softmax.as_slice()[i]).abs() < 1e-5,
            "Mismatch at {i}: exp(log_softmax)={from_log}, softmax={}",
            softmax.as_slice()[i]
        );
    }
}

#[test]
fn test_log_softmax_uniform() {
    let v = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let result = v.log_softmax().unwrap();
    let expected = -(4.0_f32).ln();
    for &val in result.as_slice() {
        assert!((val - expected).abs() < 1e-5);
    }
}

#[test]
fn test_log_softmax_large_values() {
    let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0, "log_softmax value should be <= 0, got {val}");
    }
}

#[test]
fn test_log_softmax_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0);
    }
}
