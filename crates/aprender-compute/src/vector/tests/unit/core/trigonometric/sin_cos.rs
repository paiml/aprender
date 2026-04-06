use super::super::super::super::super::*;

// sin() operation tests (element-wise sine)
#[test]
fn test_sin_basic() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
    let result = a.sin().unwrap();
    let expected = [0.0, 1.0, 0.0, -1.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "sin mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_sin_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.sin().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "sin(0) should be 0.0");
    }
}

#[test]
fn test_sin_quarter_circle() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
    let result = a.sin().unwrap();
    let expected = [0.5, std::f32::consts::FRAC_1_SQRT_2, (3.0f32).sqrt() / 2.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "sin quarter circle mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_sin_negative() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[-PI / 2.0, -PI, -3.0 * PI / 2.0]);
    let result = a.sin().unwrap();
    let expected = [-1.0, 0.0, 1.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "sin negative mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_sin_periodicity() {
    use std::f32::consts::PI;
    // sin(x + 2pi) = sin(x)
    let a = Vector::from_slice(&[0.5, 1.0, 1.5]);
    let b = Vector::from_slice(&[0.5 + 2.0 * PI, 1.0 + 2.0 * PI, 1.5 + 2.0 * PI]);
    let result_a = a.sin().unwrap();
    let result_b = b.sin().unwrap();
    for (i, (&res_a, &res_b)) in
        result_a.as_slice().iter().zip(result_b.as_slice().iter()).enumerate()
    {
        assert!(
            (res_a - res_b).abs() < 1e-5,
            "sin periodicity failed at {}: {} != {}",
            i,
            res_a,
            res_b
        );
    }
}

#[test]
fn test_sin_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sin().unwrap();
    assert_eq!(result.len(), 0);
}

// cos() operation tests (element-wise cosine)
#[test]
fn test_cos_basic() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
    let result = a.cos().unwrap();
    let expected = [1.0, 0.0, -1.0, 0.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "cos mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_cos_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.cos().unwrap();
    for &val in result.as_slice() {
        assert!((val - 1.0).abs() < 1e-5, "cos(0) should be 1.0");
    }
}

#[test]
fn test_cos_quarter_circle() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
    let result = a.cos().unwrap();
    let expected = [(3.0f32).sqrt() / 2.0, std::f32::consts::FRAC_1_SQRT_2, 0.5];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "cos quarter circle mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_cos_negative() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[-PI / 2.0, -PI, -3.0 * PI / 2.0]);
    let result = a.cos().unwrap();
    let expected = [0.0, -1.0, 0.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "cos negative mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_cos_sin_relation() {
    use std::f32::consts::PI;
    // cos(x) = sin(x + pi/2)
    let a = Vector::from_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 3.0]);
    let cos_result = a.cos().unwrap();

    let a_plus_pi_2: Vec<f32> = a.as_slice().iter().map(|&x| x + PI / 2.0).collect();
    let shifted = Vector::from_slice(&a_plus_pi_2);
    let sin_result = shifted.sin().unwrap();

    for (i, (&cos_val, &sin_val)) in
        cos_result.as_slice().iter().zip(sin_result.as_slice().iter()).enumerate()
    {
        assert!(
            (cos_val - sin_val).abs() < 1e-5,
            "cos(x) = sin(x + pi/2) failed at {}: {} != {}",
            i,
            cos_val,
            sin_val
        );
    }
}

#[test]
fn test_cos_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.cos().unwrap();
    assert_eq!(result.len(), 0);
}
