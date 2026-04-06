use super::super::super::super::*;

// Add operation tests
#[test]
fn test_add() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_add_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_add_single() {
    let a = Vector::from_slice(&[1.0]);
    let b = Vector::from_slice(&[2.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[3.0]);
}

#[test]
fn test_add_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[3.0]);
    let result = a.add(&b);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TruenoError::SizeMismatch { expected: 2, actual: 1 });
}

// Subtract operation tests
#[test]
fn test_sub() {
    let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_sub_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_sub_single() {
    let a = Vector::from_slice(&[5.0]);
    let b = Vector::from_slice(&[2.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[3.0]);
}

#[test]
fn test_sub_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[3.0]);
    let result = a.sub(&b);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TruenoError::SizeMismatch { expected: 2, actual: 1 });
}

#[test]
fn test_sub_negative_result() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[-4.0, -4.0, -4.0]);
}

// Multiply operation tests
#[test]
fn test_mul() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
}

#[test]
fn test_mul_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_mul_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0]);
    let result = a.mul(&b);
    assert!(result.is_err());
}

// Division operation tests
#[test]
fn test_div() {
    let a = Vector::from_slice(&[10.0, 20.0, 30.0]);
    let b = Vector::from_slice(&[2.0, 4.0, 5.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0]);
}

#[test]
fn test_div_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_div_single() {
    let a = Vector::from_slice(&[10.0]);
    let b = Vector::from_slice(&[2.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0]);
}

#[test]
fn test_div_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0]);
    let result = a.div(&b);
    assert!(result.is_err());
}

#[test]
fn test_div_by_one() {
    let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
}

#[test]
fn test_div_fractional() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[2.0, 4.0, 8.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.5, 0.375]);
}
