use super::super::super::super::super::*;

// ln() operation tests (element-wise natural logarithm: ln(x))
#[test]
fn test_ln_basic() {
    let a = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
    let result = a.ln().unwrap();
    let expected = [0.0, 1.0, 2.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "ln mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_ln_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.ln().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "ln(1) should be 0.0");
    }
}

#[test]
fn test_ln_small_values() {
    let a = Vector::from_slice(&[0.1, 0.5, 0.9]);
    let result = a.ln().unwrap();
    let expected = [0.1f32.ln(), 0.5f32.ln(), 0.9f32.ln()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "ln small values mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_ln_large_values() {
    let a = Vector::from_slice(&[10.0, 100.0, 1000.0]);
    let result = a.ln().unwrap();
    let expected = [10.0f32.ln(), 100.0f32.ln(), 1000.0f32.ln()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "ln large values mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_ln_inverse_exp() {
    // Test that ln(exp(x)) = x
    let a = Vector::from_slice(&[0.5, 1.0, 2.0, 3.0]);
    let exp_result = a.exp().unwrap();
    let ln_result = exp_result.ln().unwrap();
    for (i, (&original, &recovered)) in
        a.as_slice().iter().zip(ln_result.as_slice().iter()).enumerate()
    {
        assert!(
            (original - recovered).abs() < 1e-5,
            "ln(exp(x)) != x at {}: {} != {}",
            i,
            original,
            recovered
        );
    }
}

#[test]
fn test_ln_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.ln().unwrap();
    assert_eq!(result.len(), 0);
}

// log2() operation tests (element-wise base-2 logarithm)

#[test]
fn test_log2_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0, 16.0]);
    let result = a.log2().unwrap();
    let expected = [0.0, 1.0, 2.0, 3.0, 4.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "log2 mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_log2_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.log2().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "log2(1) should be 0.0");
    }
}

#[test]
fn test_log2_fractional() {
    let a = Vector::from_slice(&[0.5, 0.25, 0.125]);
    let result = a.log2().unwrap();
    let expected = [-1.0, -2.0, -3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "log2 fractional mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_log2_non_powers() {
    let a = Vector::from_slice(&[3.0, 5.0, 10.0]);
    let result = a.log2().unwrap();
    let expected = [3.0f32.log2(), 5.0f32.log2(), 10.0f32.log2()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "log2 non-powers mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_log2_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.log2().unwrap();
    assert_eq!(result.len(), 0);
}

// log10() operation tests (element-wise base-10 logarithm)

#[test]
fn test_log10_basic() {
    let a = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
    let result = a.log10().unwrap();
    let expected = [0.0, 1.0, 2.0, 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "log10 mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_log10_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.log10().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "log10(1) should be 0.0");
    }
}

#[test]
fn test_log10_fractional() {
    let a = Vector::from_slice(&[0.1, 0.01, 0.001]);
    let result = a.log10().unwrap();
    let expected = [-1.0, -2.0, -3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "log10 fractional mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_log10_non_powers() {
    let a = Vector::from_slice(&[2.0, 5.0, 50.0]);
    let result = a.log10().unwrap();
    let expected = [2.0f32.log10(), 5.0f32.log10(), 50.0f32.log10()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "log10 non-powers mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_log10_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.log10().unwrap();
    assert_eq!(result.len(), 0);
}
