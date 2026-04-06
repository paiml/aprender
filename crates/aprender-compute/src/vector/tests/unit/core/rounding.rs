use super::super::super::super::*;

#[test]
fn test_floor_basic() {
    let a = Vector::from_slice(&[3.7, -2.3, 5.0]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[3.0, -3.0, 5.0]);
}

#[test]
fn test_floor_positive() {
    let a = Vector::from_slice(&[1.1, 2.9, 3.5]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_floor_negative() {
    let a = Vector::from_slice(&[-1.1, -2.9, -3.5]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[-2.0, -3.0, -4.0]);
}

#[test]
fn test_floor_integers() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, -4.0]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, -4.0]);
}

#[test]
fn test_floor_zero() {
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0]);
}

#[test]
fn test_floor_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.floor().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ceil_basic() {
    let a = Vector::from_slice(&[3.2, -2.7, 5.0]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[4.0, -2.0, 5.0]);
}

#[test]
fn test_ceil_positive() {
    let a = Vector::from_slice(&[1.1, 2.9, 3.5]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_ceil_negative() {
    let a = Vector::from_slice(&[-1.1, -2.9, -3.5]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -2.0, -3.0]);
}

#[test]
fn test_ceil_integers() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, -4.0]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, -4.0]);
}

#[test]
fn test_ceil_zero() {
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0]);
}

#[test]
fn test_ceil_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.ceil().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_round_basic() {
    let a = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8, 5.0]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, -2.0, -3.0, 5.0]);
}

#[test]
fn test_round_positive() {
    let a = Vector::from_slice(&[1.4, 1.5, 1.6, 2.5]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn test_round_negative() {
    let a = Vector::from_slice(&[-1.4, -1.5, -1.6, -2.5]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -2.0, -2.0, -3.0]);
}

#[test]
fn test_round_halfway() {
    // Rust's round() uses "round half away from zero"
    let a = Vector::from_slice(&[0.5, 1.5, 2.5, 3.5, 4.5]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_round_zero() {
    let a = Vector::from_slice(&[0.0, -0.0, 0.3, -0.3]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0, 0.0, -0.0]);
}

#[test]
fn test_round_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.round().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_trunc_basic() {
    let a = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8, 5.0]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 3.0, -2.0, -2.0, 5.0]);
}

#[test]
fn test_trunc_positive() {
    let a = Vector::from_slice(&[1.1, 1.9, 2.5, 3.99]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_trunc_negative() {
    let a = Vector::from_slice(&[-1.1, -1.9, -2.5, -3.99]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -1.0, -2.0, -3.0]);
}

#[test]
fn test_trunc_toward_zero() {
    // Verify trunc() always moves toward zero
    let a = Vector::from_slice(&[2.7, -2.7, 5.3, -5.3]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[2.0, -2.0, 5.0, -5.0]);
}

#[test]
fn test_trunc_zero() {
    let a = Vector::from_slice(&[0.0, -0.0, 0.9, -0.9]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0, 0.0, -0.0]);
}

#[test]
fn test_trunc_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.trunc().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_fract_basic() {
    let a = Vector::from_slice(&[3.7, -2.3, 5.0]);
    let result = a.fract().unwrap();
    // fract returns fractional part with same sign
    assert!((result.as_slice()[0] - 0.7).abs() < 1e-5);
    assert!((result.as_slice()[1] - (-0.3)).abs() < 1e-5);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_fract_positive() {
    let a = Vector::from_slice(&[1.2, 2.5, 3.9]);
    let result = a.fract().unwrap();
    assert!((result.as_slice()[0] - 0.2).abs() < 1e-5);
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-5);
    assert!((result.as_slice()[2] - 0.9).abs() < 1e-5);
}

#[test]
fn test_fract_negative() {
    let a = Vector::from_slice(&[-1.2, -2.5, -3.9]);
    let result = a.fract().unwrap();
    assert!((result.as_slice()[0] - (-0.2)).abs() < 1e-5);
    assert!((result.as_slice()[1] - (-0.5)).abs() < 1e-5);
    assert!((result.as_slice()[2] - (-0.9)).abs() < 1e-5);
}

#[test]
fn test_fract_integers() {
    let a = Vector::from_slice(&[1.0, 2.0, -3.0, 0.0]);
    let result = a.fract().unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, -0.0, 0.0]);
}

#[test]
fn test_fract_range() {
    // fract() is always in range [0, 1) for positive, (-1, 0] for negative
    let a = Vector::from_slice(&[0.1, 0.5, 0.9, -0.1, -0.5, -0.9]);
    let result = a.fract().unwrap();
    for &val in result.as_slice() {
        assert!(val.abs() < 1.0, "fract value should be in range (-1, 1): {}", val);
    }
}

#[test]
fn test_fract_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.fract().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_signum_basic() {
    let a = Vector::from_slice(&[5.0, -3.0, 0.0, -0.0]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_signum_positive() {
    let a = Vector::from_slice(&[0.1, 1.0, 100.0, f32::INFINITY]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_signum_negative() {
    let a = Vector::from_slice(&[-0.1, -1.0, -100.0, f32::NEG_INFINITY]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -1.0, -1.0, -1.0]);
}

#[test]
fn test_signum_mixed() {
    let a = Vector::from_slice(&[42.5, -17.3, 0.0001, -0.0001]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_signum_zero_handling() {
    // Rust's signum treats +0.0 as positive (1.0) and -0.0 as negative (-1.0)
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, -1.0]);
}

#[test]
fn test_signum_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.signum().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_copysign_basic() {
    let magnitude = Vector::from_slice(&[5.0, 3.0, 2.0, 4.0]);
    let sign = Vector::from_slice(&[-1.0, 1.0, -1.0, 1.0]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[-5.0, 3.0, -2.0, 4.0]);
}

#[test]
fn test_copysign_negative_magnitude() {
    // copysign takes absolute magnitude, so negative magnitude becomes positive first
    let magnitude = Vector::from_slice(&[-5.0, -3.0]);
    let sign = Vector::from_slice(&[1.0, -1.0]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[5.0, -3.0]);
}

#[test]
fn test_copysign_zero() {
    // copysign handles +0.0 and -0.0
    let magnitude = Vector::from_slice(&[3.0, 3.0]);
    let sign = Vector::from_slice(&[0.0, -0.0]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[3.0, -3.0]);
}

#[test]
fn test_copysign_infinity() {
    let magnitude = Vector::from_slice(&[5.0, 5.0]);
    let sign = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[5.0, -5.0]);
}

#[test]
fn test_copysign_size_mismatch() {
    let magnitude = Vector::from_slice(&[1.0, 2.0]);
    let sign = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = magnitude.copysign(&sign);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TruenoError::SizeMismatch { .. }));
}

#[test]
fn test_copysign_empty() {
    let magnitude: Vector<f32> = Vector::from_slice(&[]);
    let sign: Vector<f32> = Vector::from_slice(&[]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.len(), 0);
}
