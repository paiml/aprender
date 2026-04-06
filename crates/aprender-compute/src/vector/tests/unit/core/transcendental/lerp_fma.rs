use super::super::super::super::super::*;

// Linear interpolation (lerp) tests
#[test]
fn test_lerp_basic() {
    let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
    let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
}

#[test]
fn test_lerp_at_zero() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.lerp(&b, 0.0).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]); // Should return a
}

#[test]
fn test_lerp_at_one() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.lerp(&b, 1.0).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]); // Should return b
}

#[test]
fn test_lerp_extrapolate_above() {
    let a = Vector::from_slice(&[0.0, 10.0]);
    let b = Vector::from_slice(&[10.0, 20.0]);
    let result = a.lerp(&b, 2.0).unwrap();
    assert_eq!(result.as_slice(), &[20.0, 30.0]); // Extrapolation beyond b
}

#[test]
fn test_lerp_extrapolate_below() {
    let a = Vector::from_slice(&[10.0, 20.0]);
    let b = Vector::from_slice(&[20.0, 30.0]);
    let result = a.lerp(&b, -1.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 10.0]); // Extrapolation before a
}

#[test]
fn test_lerp_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.lerp(&b, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_lerp_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.len(), 0);
}

// fma() operation tests (fused multiply-add: a * b + c)
#[test]
fn test_fma_basic() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [2*5+1, 3*6+2, 4*7+3] = [11, 20, 31]
    assert_eq!(result.as_slice(), &[11.0, 20.0, 31.0]);
}

#[test]
fn test_fma_zeros() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [0*5+1, 0*6+2, 0*7+3] = [1, 2, 3]
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_fma_ones() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let c = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [2*1+0, 3*1+0, 4*1+0] = [2, 3, 4]
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_fma_negatives() {
    let a = Vector::from_slice(&[-2.0, 3.0, -4.0]);
    let b = Vector::from_slice(&[5.0, -6.0, 7.0]);
    let c = Vector::from_slice(&[1.0, 2.0, -3.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [-2*5+1, 3*(-6)+2, -4*7+(-3)] = [-9, -16, -31]
    assert_eq!(result.as_slice(), &[-9.0, -16.0, -31.0]);
}

#[test]
fn test_fma_size_mismatch_b() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let c = Vector::from_slice(&[1.0, 2.0]);
    let result = a.fma(&b, &c);
    assert!(result.is_err());
}

#[test]
fn test_fma_size_mismatch_c() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.fma(&b, &c);
    assert!(result.is_err());
}

#[test]
fn test_fma_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let c: Vector<f32> = Vector::from_slice(&[]);
    let result = a.fma(&b, &c).unwrap();
    assert_eq!(result.len(), 0);
}
