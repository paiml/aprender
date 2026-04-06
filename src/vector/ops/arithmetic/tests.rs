use super::*;

// ===== Add Tests =====

#[test]
fn test_add_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_add_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.add(&b);
    assert!(result.is_err());
    match result {
        Err(TruenoError::SizeMismatch { expected, actual }) => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 3);
        }
        _ => panic!("Expected SizeMismatch error"),
    }
}

#[test]
fn test_add_empty() {
    let a = Vector::from_slice(&[]);
    let b = Vector::from_slice(&[]);
    let result = a.add(&b).unwrap();
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_add_single_element() {
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[2.5]);
    let result = a.add(&b).unwrap();
    assert!((result.as_slice()[0] - 4.0).abs() < 1e-6);
}

#[test]
fn test_add_negatives() {
    let a = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_add_large_array() {
    let n = 10000;
    let a = Vector::from_slice(&vec![1.0; n]);
    let b = Vector::from_slice(&vec![2.0; n]);
    let result = a.add(&b).unwrap();
    for val in result.as_slice() {
        assert!((val - 3.0).abs() < 1e-6);
    }
}

// ===== Sub Tests =====

#[test]
fn test_sub_basic() {
    let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_sub_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[1.0]);
    let result = a.sub(&b);
    assert!(result.is_err());
}

#[test]
fn test_sub_empty() {
    let a = Vector::from_slice(&[]);
    let b = Vector::from_slice(&[]);
    let result = a.sub(&b).unwrap();
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_sub_self() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = a.sub(&a).unwrap();
    for val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// ===== Mul Tests =====

#[test]
fn test_mul_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 6.0, 12.0]);
}

#[test]
fn test_mul_size_mismatch() {
    let a = Vector::from_slice(&[1.0]);
    let b = Vector::from_slice(&[1.0, 2.0]);
    let result = a.mul(&b);
    assert!(result.is_err());
}

#[test]
fn test_mul_empty() {
    let a = Vector::from_slice(&[]);
    let b = Vector::from_slice(&[]);
    let result = a.mul(&b).unwrap();
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_mul_by_zero() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_mul_by_one() {
    let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
}

// ===== Div Tests =====

#[test]
fn test_div_basic() {
    let a = Vector::from_slice(&[4.0, 6.0, 8.0]);
    let b = Vector::from_slice(&[2.0, 2.0, 2.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_div_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = a.div(&b);
    assert!(result.is_err());
}

#[test]
fn test_div_empty() {
    let a = Vector::from_slice(&[]);
    let b = Vector::from_slice(&[]);
    let result = a.div(&b).unwrap();
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_div_by_one() {
    let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
}

#[test]
fn test_div_by_zero_produces_inf() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[0.0, 0.0]);
    let result = a.div(&b).unwrap();
    assert!(result.as_slice()[0].is_infinite());
    assert!(result.as_slice()[1].is_infinite());
}

// ===== Scale Tests =====

#[test]
fn test_scale_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.scale(2.0).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_scale_empty() {
    let a = Vector::from_slice(&[]);
    let result = a.scale(5.0).unwrap();
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_scale_by_zero() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.scale(0.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_scale_by_one() {
    let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
    let result = a.scale(1.0).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
}

#[test]
fn test_scale_negative() {
    let a = Vector::from_slice(&[1.0, -2.0, 3.0]);
    let result = a.scale(-1.0).unwrap();
    assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0]);
}

// ===== FMA Tests =====

#[test]
fn test_fma_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[2.0, 2.0, 2.0]);
    let c = Vector::from_slice(&[1.0, 1.0, 1.0]);
    // a * b + c = [2+1, 4+1, 6+1] = [3, 5, 7]
    let result = a.fma(&b, &c).unwrap();
    assert_eq!(result.as_slice(), &[3.0, 5.0, 7.0]);
}

#[test]
fn test_fma_size_mismatch_b() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[2.0]);
    let c = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.fma(&b, &c);
    assert!(result.is_err());
}

#[test]
fn test_fma_size_mismatch_c() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[2.0, 2.0, 2.0]);
    let c = Vector::from_slice(&[1.0]);
    let result = a.fma(&b, &c);
    assert!(result.is_err());
}

#[test]
fn test_fma_empty() {
    let a = Vector::from_slice(&[]);
    let b = Vector::from_slice(&[]);
    let c = Vector::from_slice(&[]);
    let result = a.fma(&b, &c).unwrap();
    assert!(result.as_slice().is_empty());
}

#[test]
fn test_fma_multiply_by_zero() {
    let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
    let b = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    // a * 0 + c = c
    let result = a.fma(&b, &c).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_fma_add_zero() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[3.0, 2.0, 1.0]);
    let c = Vector::from_slice(&[0.0, 0.0, 0.0]);
    // a * b + 0 = a * b
    let result = a.fma(&b, &c).unwrap();
    assert_eq!(result.as_slice(), &[6.0, 6.0, 4.0]);
}

// ===== Backend Tests =====

#[test]
fn test_add_scalar_backend() {
    let a = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let b = Vector::from_slice_with_backend(&[4.0, 5.0, 6.0], Backend::Scalar);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_add_sse2_backend() {
    let a = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0, 4.0], Backend::SSE2);
    let b = Vector::from_slice_with_backend(&[4.0, 5.0, 6.0, 7.0], Backend::SSE2);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0, 11.0]);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_add_avx2_backend() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return; // Skip if AVX2 not available
    }
    let data: Vec<f32> = vec![1.0; 16];
    let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
    let b_data: Vec<f32> = vec![2.0; 16];
    let b = Vector::from_slice_with_backend(&b_data, Backend::AVX2);
    let result = a.add(&b).unwrap();
    for &val in result.as_slice() {
        assert!((val - 3.0).abs() < 1e-6);
    }
}

// ===== Edge Cases =====

#[test]
fn test_add_non_aligned_size() {
    // Test with sizes that don't align to SIMD register widths
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]); // 7 elements
    let b = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_mul_preserves_sign() {
    let a = Vector::from_slice(&[2.0, -2.0, 2.0, -2.0]);
    let b = Vector::from_slice(&[3.0, 3.0, -3.0, -3.0]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[6.0, -6.0, -6.0, 6.0]);
}

#[test]
fn test_operations_with_special_floats() {
    let a = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY, 0.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.add(&b).unwrap();
    assert!(result.as_slice()[0].is_infinite());
    assert!(result.as_slice()[1].is_infinite());
    assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
}
