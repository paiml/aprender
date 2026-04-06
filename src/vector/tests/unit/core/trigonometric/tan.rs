use super::super::super::super::super::*;

#[test]
fn test_tan_basic() {
    use std::f32::consts::PI;
    // tan(0) = 0, tan(pi/4) = 1, tan(-pi/4) = -1
    let a = Vector::from_slice(&[0.0, PI / 4.0, -PI / 4.0]);
    let result = a.tan().unwrap();
    let expected = [0.0, 1.0, -1.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "tan basic mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_tan_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.tan().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tan_quarter_circle() {
    use std::f32::consts::PI;
    // tan(pi/4) = 1
    let a = Vector::from_slice(&[PI / 4.0]);
    let result = a.tan().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_tan_negative() {
    use std::f32::consts::PI;
    // tan is odd: tan(-x) = -tan(x)
    let a = Vector::from_slice(&[-PI / 4.0, -PI / 6.0]);
    let result = a.tan().unwrap();
    let expected = [-1.0, -(1.0 / 3.0_f32.sqrt())];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "tan negative mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_tan_sin_cos_relation() {
    use std::f32::consts::PI;
    // tan(x) = sin(x) / cos(x)
    let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
    let tan_result = a.tan().unwrap();
    let sin_result = a.sin().unwrap();
    let cos_result = a.cos().unwrap();

    for (i, ((&tan_val, &sin_val), &cos_val)) in tan_result
        .as_slice()
        .iter()
        .zip(sin_result.as_slice().iter())
        .zip(cos_result.as_slice().iter())
        .enumerate()
    {
        let expected = sin_val / cos_val;
        assert!(
            (tan_val - expected).abs() < 1e-5,
            "tan(x) != sin(x)/cos(x) at {}: {} != {}",
            i,
            tan_val,
            expected
        );
    }
}

#[test]
fn test_tan_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.tan().unwrap();
    assert_eq!(result.len(), 0);
}
