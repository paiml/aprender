use super::super::super::super::super::*;

// asin() tests
#[test]
fn test_asin_basic() {
    use std::f32::consts::PI;
    // asin(0) = 0, asin(1) = pi/2, asin(-1) = -pi/2, asin(0.5) = pi/6
    let a = Vector::from_slice(&[0.0, 1.0, -1.0, 0.5]);
    let result = a.asin().unwrap();
    let expected = [0.0, PI / 2.0, -PI / 2.0, PI / 6.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "asin basic mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_asin_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.asin().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_asin_range() {
    use std::f32::consts::PI;
    // asin domain is [-1, 1], range is [-pi/2, pi/2]
    let a = Vector::from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
    let result = a.asin().unwrap();
    for (i, &res) in result.as_slice().iter().enumerate() {
        assert!(
            (-PI / 2.0..=PI / 2.0).contains(&res),
            "asin range violation at {}: {} not in [-pi/2, pi/2]",
            i,
            res
        );
    }
}

#[test]
fn test_asin_negative() {
    use std::f32::consts::PI;
    // asin is odd: asin(-x) = -asin(x)
    let a = Vector::from_slice(&[-0.5, -0.707]);
    let result = a.asin().unwrap();
    let expected = [-PI / 6.0, -PI / 4.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-3, "asin negative mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_asin_sin_inverse() {
    use std::f32::consts::PI;
    // asin(sin(x)) = x for x in [-pi/2, pi/2]
    let a = Vector::from_slice(&[-PI / 4.0, 0.0, PI / 6.0, PI / 4.0]);
    let sin_result = a.sin().unwrap();
    let asin_result = sin_result.asin().unwrap();

    for (i, (&original, &reconstructed)) in
        a.as_slice().iter().zip(asin_result.as_slice().iter()).enumerate()
    {
        assert!(
            (original - reconstructed).abs() < 1e-5,
            "asin(sin(x)) != x at {}: {} != {}",
            i,
            reconstructed,
            original
        );
    }
}

#[test]
fn test_asin_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.asin().unwrap();
    assert_eq!(result.len(), 0);
}

// acos() tests
#[test]
fn test_acos_basic() {
    use std::f32::consts::PI;
    // acos(0) = pi/2, acos(1) = 0, acos(-1) = pi, acos(0.5) = pi/3
    let a = Vector::from_slice(&[0.0, 1.0, -1.0, 0.5]);
    let result = a.acos().unwrap();
    let expected = [PI / 2.0, 0.0, PI, PI / 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-5, "acos basic mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_acos_zero() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[1.0]);
    let result = a.acos().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);

    // Also test acos(0) = pi/2
    let b = Vector::from_slice(&[0.0]);
    let result_b = b.acos().unwrap();
    assert!((result_b.as_slice()[0] - PI / 2.0).abs() < 1e-5);
}

#[test]
fn test_acos_range() {
    use std::f32::consts::PI;
    // acos domain is [-1, 1], range is [0, pi]
    let a = Vector::from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
    let result = a.acos().unwrap();
    for (i, &res) in result.as_slice().iter().enumerate() {
        assert!((0.0..=PI).contains(&res), "acos range violation at {}: {} not in [0, pi]", i, res);
    }
}

#[test]
fn test_acos_symmetry() {
    use std::f32::consts::PI;
    // acos(-x) = pi - acos(x)
    let a = Vector::from_slice(&[0.5, 0.707]);
    let result_pos = a.acos().unwrap();

    let a_neg = Vector::from_slice(&[-0.5, -0.707]);
    let result_neg = a_neg.acos().unwrap();

    for (i, (&pos, &neg)) in
        result_pos.as_slice().iter().zip(result_neg.as_slice().iter()).enumerate()
    {
        let expected_neg = PI - pos;
        assert!(
            (neg - expected_neg).abs() < 1e-5,
            "acos symmetry failed at {}: acos(-x)={} != pi - acos(x)={}",
            i,
            neg,
            expected_neg
        );
    }
}

#[test]
fn test_acos_cos_inverse() {
    use std::f32::consts::PI;
    // acos(cos(x)) = x for x in [0, pi]
    let a = Vector::from_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI]);
    let cos_result = a.cos().unwrap();
    let acos_result = cos_result.acos().unwrap();

    for (i, (&original, &reconstructed)) in
        a.as_slice().iter().zip(acos_result.as_slice().iter()).enumerate()
    {
        assert!(
            (original - reconstructed).abs() < 1e-5,
            "acos(cos(x)) != x at {}: {} != {}",
            i,
            reconstructed,
            original
        );
    }
}

#[test]
fn test_acos_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.acos().unwrap();
    assert_eq!(result.len(), 0);
}

// atan() tests
#[test]
fn test_atan_basic() {
    use std::f32::consts::PI;
    // atan(0) = 0, atan(1) = pi/4, atan(-1) = -pi/4
    let a = Vector::from_slice(&[0.0, 1.0, -1.0, 1.732]); // 1.732 ~ sqrt(3) for atan(sqrt(3)) = pi/3
    let result = a.atan().unwrap();
    let expected = [0.0, PI / 4.0, -PI / 4.0, PI / 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-3, "atan basic mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_atan_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.atan().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_atan_range() {
    use std::f32::consts::PI;
    // atan range is (-pi/2, pi/2) for all real inputs
    let a = Vector::from_slice(&[-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0]);
    let result = a.atan().unwrap();
    for (i, &res) in result.as_slice().iter().enumerate() {
        assert!(
            (-PI / 2.0..PI / 2.0).contains(&res),
            "atan range violation at {}: {} not in (-pi/2, pi/2)",
            i,
            res
        );
    }
}

#[test]
fn test_atan_negative() {
    use std::f32::consts::PI;
    // atan is odd: atan(-x) = -atan(x)
    let a = Vector::from_slice(&[-1.0, -1.732]);
    let result = a.atan().unwrap();
    let expected = [-PI / 4.0, -PI / 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-3, "atan negative mismatch at {}: {} != {}", i, res, exp);
    }
}

#[test]
fn test_atan_tan_inverse() {
    use std::f32::consts::PI;
    // atan(tan(x)) = x for x in (-pi/2, pi/2)
    let a = Vector::from_slice(&[-PI / 4.0, 0.0, PI / 6.0, PI / 4.0]);
    let tan_result = a.tan().unwrap();
    let atan_result = tan_result.atan().unwrap();

    for (i, (&original, &reconstructed)) in
        a.as_slice().iter().zip(atan_result.as_slice().iter()).enumerate()
    {
        assert!(
            (original - reconstructed).abs() < 1e-5,
            "atan(tan(x)) != x at {}: {} != {}",
            i,
            reconstructed,
            original
        );
    }
}

#[test]
fn test_atan_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.atan().unwrap();
    assert_eq!(result.len(), 0);
}
