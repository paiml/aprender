use super::super::super::super::*;

// sinh() tests
#[test]
fn test_sinh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.sinh().unwrap();
    let expected = [0.0, 1.0_f32.sinh(), (-1.0_f32).sinh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_sinh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.sinh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_sinh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.sinh().unwrap();
    let expected = 2.0_f32.sinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_sinh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.sinh().unwrap();
    let expected = (-2.0_f32).sinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_sinh_odd_function() {
    // sinh(-x) = -sinh(x)
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[-1.5]);
    let sinh_a = a.sinh().unwrap();
    let sinh_b = b.sinh().unwrap();
    assert!(
        (sinh_a.as_slice()[0] + sinh_b.as_slice()[0]).abs() < 1e-5,
        "sinh is an odd function: sinh(-x) = -sinh(x)"
    );
}

#[test]
fn test_sinh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sinh().unwrap();
    assert_eq!(result.len(), 0);
}

// cosh() tests
#[test]
fn test_cosh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.cosh().unwrap();
    let expected = [0.0_f32.cosh(), 1.0_f32.cosh(), (-1.0_f32).cosh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_cosh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.cosh().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.cosh().unwrap();
    let expected = 2.0_f32.cosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_cosh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.cosh().unwrap();
    let expected = (-2.0_f32).cosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_cosh_even_function() {
    // cosh(-x) = cosh(x)
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[-1.5]);
    let cosh_a = a.cosh().unwrap();
    let cosh_b = b.cosh().unwrap();
    assert!(
        (cosh_a.as_slice()[0] - cosh_b.as_slice()[0]).abs() < 1e-5,
        "cosh is an even function: cosh(-x) = cosh(x)"
    );
}

#[test]
fn test_cosh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.cosh().unwrap();
    assert_eq!(result.len(), 0);
}

// tanh() tests
#[test]
fn test_tanh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.tanh().unwrap();
    let expected = [0.0_f32.tanh(), 1.0_f32.tanh(), (-1.0_f32).tanh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_tanh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tanh_range() {
    // tanh(x) is bounded: -1 <= tanh(x) <= 1
    // For very large values, it approaches +/-1 in floating-point
    let a = Vector::from_slice(&[10.0, -10.0, 100.0]);
    let result = a.tanh().unwrap();
    for &val in result.as_slice() {
        assert!((-1.0..=1.0).contains(&val), "tanh value {} out of range [-1, 1]", val);
    }
}

#[test]
fn test_tanh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.tanh().unwrap();
    let expected = (-2.0_f32).tanh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_tanh_sinh_cosh_relation() {
    // tanh(x) = sinh(x) / cosh(x)
    let a = Vector::from_slice(&[1.5]);
    let tanh_result = a.tanh().unwrap();
    let sinh_result = a.sinh().unwrap();
    let cosh_result = a.cosh().unwrap();
    let ratio = sinh_result.as_slice()[0] / cosh_result.as_slice()[0];
    assert!((tanh_result.as_slice()[0] - ratio).abs() < 1e-5, "tanh(x) = sinh(x)/cosh(x)");
}

#[test]
fn test_tanh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.tanh();
    assert!(result.is_err());
}

// asinh() tests
#[test]
fn test_asinh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.asinh().unwrap();
    let expected = [0.0_f32.asinh(), 1.0_f32.asinh(), (-1.0_f32).asinh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_asinh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.asinh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_asinh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.asinh().unwrap();
    let expected = 2.0_f32.asinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_asinh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.asinh().unwrap();
    let expected = (-2.0_f32).asinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_asinh_odd_function() {
    // asinh(-x) = -asinh(x)
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[-1.5]);
    let asinh_a = a.asinh().unwrap();
    let asinh_b = b.asinh().unwrap();
    assert!(
        (asinh_a.as_slice()[0] + asinh_b.as_slice()[0]).abs() < 1e-5,
        "asinh is an odd function: asinh(-x) = -asinh(x)"
    );
}

#[test]
fn test_asinh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.asinh().unwrap();
    assert_eq!(result.len(), 0);
}

// acosh() tests
#[test]
fn test_acosh_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.acosh().unwrap();
    let expected = [1.0_f32.acosh(), 2.0_f32.acosh(), 3.0_f32.acosh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_acosh_one() {
    let a = Vector::from_slice(&[1.0]);
    let result = a.acosh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_acosh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.acosh().unwrap();
    let expected = 2.0_f32.acosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_acosh_large() {
    let a = Vector::from_slice(&[10.0]);
    let result = a.acosh().unwrap();
    let expected = 10.0_f32.acosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_acosh_cosh_relation() {
    // acosh(cosh(x)) = x for x >= 0
    let a = Vector::from_slice(&[1.5]);
    let cosh_result = a.cosh().unwrap();
    let acosh_result = cosh_result.acosh().unwrap();
    assert!((a.as_slice()[0] - acosh_result.as_slice()[0]).abs() < 1e-5, "acosh(cosh(x)) = x");
}

#[test]
fn test_acosh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.acosh().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_atanh_basic() {
    let a = Vector::from_slice(&[0.0, 0.5, -0.5]);
    let result = a.atanh().unwrap();
    let expected: Vec<f32> = vec![0.0_f32.atanh(), 0.5_f32.atanh(), (-0.5_f32).atanh()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "atanh failed at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_atanh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.atanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_atanh_positive() {
    let a = Vector::from_slice(&[0.5]);
    let result = a.atanh().unwrap();
    let expected = 0.5_f32.atanh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_atanh_negative() {
    let a = Vector::from_slice(&[-0.5]);
    let result = a.atanh().unwrap();
    let expected = (-0.5_f32).atanh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_atanh_odd_function() {
    // atanh(-x) = -atanh(x)
    let a = Vector::from_slice(&[0.5]);
    let neg_a = Vector::from_slice(&[-0.5]);
    let result_a = a.atanh().unwrap();
    let result_neg_a = neg_a.atanh().unwrap();
    assert!(
        (result_a.as_slice()[0] + result_neg_a.as_slice()[0]).abs() < 1e-5,
        "atanh(-x) = -atanh(x)"
    );
}

#[test]
fn test_atanh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.atanh().unwrap();
    assert_eq!(result.len(), 0);
}
