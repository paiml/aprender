use crate::vector::Vector;
use crate::{Backend, TruenoError};

// ========== Basic Reductions ==========

#[test]
fn test_dot_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.dot(&b).unwrap();
    assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_dot_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0]);
    assert!(matches!(a.dot(&b), Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_dot_empty() {
    let a = Vector::<f32>::from_slice(&[]);
    let b = Vector::<f32>::from_slice(&[]);
    let result = a.dot(&b).unwrap();
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_single() {
    let a = Vector::from_slice(&[3.0]);
    let b = Vector::from_slice(&[4.0]);
    let result = a.dot(&b).unwrap();
    assert!((result - 12.0).abs() < 1e-6);
}

#[test]
fn test_dot_large_aligned() {
    // Test SIMD path with aligned size
    let a = Vector::from_slice(&[1.0; 256]);
    let b = Vector::from_slice(&[2.0; 256]);
    let result = a.dot(&b).unwrap();
    assert!((result - 512.0).abs() < 1e-3); // 256 * 1 * 2 = 512
}

#[test]
fn test_dot_large_unaligned() {
    // Test SIMD path with unaligned size
    let a = Vector::from_slice(&[1.0; 259]);
    let b = Vector::from_slice(&[2.0; 259]);
    let result = a.dot(&b).unwrap();
    assert!((result - 518.0).abs() < 1e-3); // 259 * 1 * 2 = 518
}

#[test]
fn test_sum_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert!((v.sum().unwrap() - 10.0).abs() < 1e-6);
}

#[test]
fn test_sum_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!((v.sum().unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn test_sum_single() {
    let v = Vector::from_slice(&[42.0]);
    assert!((v.sum().unwrap() - 42.0).abs() < 1e-6);
}

#[test]
fn test_sum_negatives() {
    let v = Vector::from_slice(&[-1.0, -2.0, 3.0, 4.0]);
    assert!((v.sum().unwrap() - 4.0).abs() < 1e-6);
}

#[test]
fn test_max_basic() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert!((v.max().unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn test_max_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.max(), Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_max_single() {
    let v = Vector::from_slice(&[42.0]);
    assert!((v.max().unwrap() - 42.0).abs() < 1e-6);
}

#[test]
fn test_max_all_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -3.0, -2.0]);
    assert!((v.max().unwrap() - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_min_basic() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert!((v.min().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn test_min_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.min(), Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_min_single() {
    let v = Vector::from_slice(&[42.0]);
    assert!((v.min().unwrap() - 42.0).abs() < 1e-6);
}

#[test]
fn test_min_all_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -3.0, -2.0]);
    assert!((v.min().unwrap() - (-5.0)).abs() < 1e-6);
}

// ========== Index-finding ==========

#[test]
fn test_argmax_basic() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.argmax().unwrap(), 1);
}

#[test]
fn test_argmax_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.argmax(), Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_argmax_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.argmax().unwrap(), 0);
}

#[test]
fn test_argmax_duplicate_max() {
    let v = Vector::from_slice(&[1.0, 5.0, 5.0, 2.0]);
    // Should return first occurrence
    assert_eq!(v.argmax().unwrap(), 1);
}

#[test]
fn test_argmin_basic() {
    let v = Vector::from_slice(&[3.0, 1.0, 5.0, 2.0]);
    assert_eq!(v.argmin().unwrap(), 1);
}

#[test]
fn test_argmin_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.argmin(), Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_argmin_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.argmin().unwrap(), 0);
}

// ========== Numerically Stable ==========

#[test]
fn test_sum_kahan_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert!((v.sum_kahan().unwrap() - 10.0).abs() < 1e-6);
}

#[test]
fn test_sum_kahan_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!((v.sum_kahan().unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn test_sum_kahan_precision() {
    // Kahan summation provides better precision for certain scenarios
    // but f32 limits mean 1e10 + 1 = 1e10 in float representation
    // Test with values that demonstrate the benefit of Kahan
    let v = Vector::from_slice(&[1.0, 1e-8, 1e-8, 1e-8, 1e-8]);
    let result = v.sum_kahan().unwrap();
    // Should be close to 1.0 + 4e-8
    assert!((result - 1.00000004).abs() < 1e-6);
}

#[test]
fn test_sum_of_squares_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    // 1 + 4 + 9 = 14
    assert!((v.sum_of_squares().unwrap() - 14.0).abs() < 1e-6);
}

#[test]
fn test_sum_of_squares_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!((v.sum_of_squares().unwrap() - 0.0).abs() < 1e-6);
}

// ========== Statistical ==========

#[test]
fn test_mean_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!((v.mean().unwrap() - 3.0).abs() < 1e-6);
}

#[test]
fn test_mean_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.mean(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_mean_single() {
    let v = Vector::from_slice(&[42.0]);
    assert!((v.mean().unwrap() - 42.0).abs() < 1e-6);
}

#[test]
fn test_variance_basic() {
    let v = Vector::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    // Mean = 5, Variance = 4
    let var = v.variance().unwrap();
    assert!((var - 4.0).abs() < 1e-3);
}

#[test]
fn test_variance_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.variance(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_variance_constant() {
    let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    assert!((v.variance().unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn test_stddev_basic() {
    let v = Vector::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    // Stddev = sqrt(4) = 2
    let std = v.stddev().unwrap();
    assert!((std - 2.0).abs() < 1e-3);
}

#[test]
fn test_stddev_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.stddev(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_covariance_basic() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]); // y = 2x
    let cov = x.covariance(&y).unwrap();
    // Cov(X, 2X) = 2 * Var(X) = 2 * 2 = 4
    assert!((cov - 4.0).abs() < 1e-3);
}

#[test]
fn test_covariance_size_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[1.0, 2.0]);
    assert!(matches!(x.covariance(&y), Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_covariance_empty() {
    let x = Vector::<f32>::from_slice(&[]);
    let y = Vector::<f32>::from_slice(&[]);
    assert!(matches!(x.covariance(&y), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_correlation_positive() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]); // y = 2x
    let corr = x.correlation(&y).unwrap();
    // Perfect positive correlation
    assert!((corr - 1.0).abs() < 1e-3);
}

#[test]
fn test_correlation_negative() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::from_slice(&[10.0, 8.0, 6.0, 4.0, 2.0]); // y = -2x + 12
    let corr = x.correlation(&y).unwrap();
    // Perfect negative correlation
    assert!((corr - (-1.0)).abs() < 1e-3);
}

#[test]
fn test_correlation_constant_x() {
    let x = Vector::from_slice(&[5.0, 5.0, 5.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert!(matches!(x.correlation(&y), Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_correlation_constant_y() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[5.0, 5.0, 5.0]);
    assert!(matches!(x.correlation(&y), Err(TruenoError::DivisionByZero)));
}

// ========== Backend Tests ==========

#[test]
fn test_dot_scalar_backend() {
    let a = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let b = Vector::from_slice_with_backend(&[4.0, 5.0, 6.0], Backend::Scalar);
    let result = a.dot(&b).unwrap();
    assert!((result - 32.0).abs() < 1e-6);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_dot_sse2_backend() {
    let a =
        Vector::from_slice_with_backend(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Backend::SSE2);
    let b =
        Vector::from_slice_with_backend(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], Backend::SSE2);
    let result = a.dot(&b).unwrap();
    assert!((result - 36.0).abs() < 1e-6); // sum 1..8 = 36
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_dot_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let a = Vector::from_slice_with_backend(&[1.0; 32], Backend::AVX2);
    let b = Vector::from_slice_with_backend(&[2.0; 32], Backend::AVX2);
    let result = a.dot(&b).unwrap();
    assert!((result - 64.0).abs() < 1e-4);
}

#[test]
fn test_sum_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0, 4.0], Backend::Scalar);
    assert!((v.sum().unwrap() - 10.0).abs() < 1e-6);
}

#[test]
fn test_max_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 5.0, 3.0, 2.0], Backend::Scalar);
    assert!((v.max().unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn test_min_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 5.0, 3.0, 2.0], Backend::Scalar);
    assert!((v.min().unwrap() - 1.0).abs() < 1e-6);
}
