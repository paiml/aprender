use super::super::super::super::super::*;

// sqrt() operation tests (element-wise square root)
#[test]
fn test_sqrt_basic() {
    let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sqrt_zeros() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_sqrt_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_sqrt_fractional() {
    let a = Vector::from_slice(&[0.25, 0.01, 0.0625]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.1, 0.25]);
}

#[test]
fn test_sqrt_negative() {
    let a = Vector::from_slice(&[-1.0, 4.0, -9.0]);
    let result = a.sqrt().unwrap();
    // Negative values produce NaN
    assert!(result.as_slice()[0].is_nan());
    assert_eq!(result.as_slice()[1], 4.0_f32.sqrt());
    assert!(result.as_slice()[2].is_nan());
}

#[test]
fn test_sqrt_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.len(), 0);
}

// recip() operation tests (element-wise reciprocal: 1/x)
#[test]
fn test_recip_basic() {
    let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
}

#[test]
fn test_recip_ones() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_recip_negatives() {
    let a = Vector::from_slice(&[-2.0, -4.0, -0.5]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[-0.5, -0.25, -2.0]);
}

#[test]
fn test_recip_fractional() {
    let a = Vector::from_slice(&[0.5, 0.25, 0.1]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 4.0, 10.0]);
}

#[test]
fn test_recip_zero() {
    let a = Vector::from_slice(&[0.0, 2.0, 0.0]);
    let result = a.recip().unwrap();
    // 1/0 = infinity
    assert!(result.as_slice()[0].is_infinite());
    assert_eq!(result.as_slice()[1], 0.5);
    assert!(result.as_slice()[2].is_infinite());
}

#[test]
fn test_recip_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.recip().unwrap();
    assert_eq!(result.len(), 0);
}
