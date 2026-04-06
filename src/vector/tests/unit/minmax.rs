use super::super::super::*;

// Unit Tests: minimum()
// ========================================

#[test]
fn test_minimum_basic() {
    let a = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    let b = Vector::from_slice(&[2.0, 3.0, 4.0, 1.0]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 3.0, 3.0, 1.0]);
}

#[test]
fn test_minimum_negative() {
    let a = Vector::from_slice(&[-1.0, -5.0, 3.0]);
    let b = Vector::from_slice(&[-2.0, -3.0, 4.0]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice(), &[-2.0, -5.0, 3.0]);
}

#[test]
fn test_minimum_nan() {
    // NaN handling: NAN.min(x) = x (prefers non-NaN)
    let a = Vector::from_slice(&[f32::NAN, 5.0, f32::NAN]);
    let b = Vector::from_slice(&[3.0, f32::NAN, f32::NAN]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice()[0], 3.0);
    assert_eq!(result.as_slice()[1], 5.0);
    assert!(result.as_slice()[2].is_nan());
}

#[test]
fn test_minimum_infinity() {
    let a = Vector::from_slice(&[f32::INFINITY, 5.0, f32::NEG_INFINITY]);
    let b = Vector::from_slice(&[3.0, f32::INFINITY, -10.0]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice(), &[3.0, 5.0, f32::NEG_INFINITY]);
}

#[test]
fn test_minimum_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.minimum(&b);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_minimum_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
// Unit Tests: maximum()
// ========================================

#[test]
fn test_maximum_basic() {
    let a = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    let b = Vector::from_slice(&[2.0, 3.0, 4.0, 1.0]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 5.0, 4.0, 2.0]);
}

#[test]
fn test_maximum_negative() {
    let a = Vector::from_slice(&[-1.0, -5.0, 3.0]);
    let b = Vector::from_slice(&[-2.0, -3.0, 4.0]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -3.0, 4.0]);
}

#[test]
fn test_maximum_nan() {
    // NaN handling: NAN.max(x) = x (prefers non-NaN)
    let a = Vector::from_slice(&[f32::NAN, 5.0, f32::NAN]);
    let b = Vector::from_slice(&[3.0, f32::NAN, f32::NAN]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice()[0], 3.0);
    assert_eq!(result.as_slice()[1], 5.0);
    assert!(result.as_slice()[2].is_nan());
}

#[test]
fn test_maximum_infinity() {
    let a = Vector::from_slice(&[f32::INFINITY, 5.0, f32::NEG_INFINITY]);
    let b = Vector::from_slice(&[3.0, f32::INFINITY, -10.0]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice(), &[f32::INFINITY, f32::INFINITY, -10.0]);
}

#[test]
fn test_maximum_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.maximum(&b);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_maximum_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
// Unit Tests: neg()
// ========================================

#[test]
fn test_neg_basic() {
    let a = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0]);
    let result = a.neg().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_neg_zero() {
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.neg().unwrap();
    // -0.0 becomes 0.0, 0.0 becomes -0.0
    assert_eq!(result.as_slice()[0], -0.0);
    assert_eq!(result.as_slice()[1], 0.0);
}

#[test]
fn test_neg_double_negation() {
    // Property: -(-x) = x (double negation is identity)
    let a = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0]);
    let neg_once = a.neg().unwrap();
    let neg_twice = neg_once.neg().unwrap();
    for (i, (&original, &double_neg)) in
        a.as_slice().iter().zip(neg_twice.as_slice().iter()).enumerate()
    {
        assert!(
            (original - double_neg).abs() < 1e-6,
            "Double negation failed at {}: -(-{}) = {} != {}",
            i,
            original,
            double_neg,
            original
        );
    }
}

#[test]
fn test_neg_nan() {
    let a = Vector::from_slice(&[f32::NAN, 5.0]);
    let result = a.neg().unwrap();
    assert!(result.as_slice()[0].is_nan());
    assert_eq!(result.as_slice()[1], -5.0);
}

#[test]
fn test_neg_infinity() {
    let a = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY]);
    let result = a.neg().unwrap();
    assert_eq!(result.as_slice(), &[f32::NEG_INFINITY, f32::INFINITY]);
}

#[test]
fn test_neg_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.neg().unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
