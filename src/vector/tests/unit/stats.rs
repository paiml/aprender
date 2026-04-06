use super::super::super::*;

// Unit Tests: sum_of_squares()
// ========================================

#[test]
fn test_sum_of_squares_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 14.0); // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
}

#[test]
fn test_sum_of_squares_negative() {
    let a = Vector::from_slice(&[-1.0, -2.0, 3.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 14.0); // (-1)^2 + (-2)^2 + 3^2 = 1 + 4 + 9 = 14
}

#[test]
fn test_sum_of_squares_single() {
    let a = Vector::from_slice(&[5.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 25.0);
}

#[test]
fn test_sum_of_squares_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_sum_of_squares_pythagorean() {
    // 3-4-5 Pythagorean triple
    let a = Vector::from_slice(&[3.0, 4.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 25.0); // 3^2 + 4^2 = 9 + 16 = 25
}

#[test]
fn test_sum_of_squares_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 0.0);
}

// ========================================================================
// Tests for mean() - arithmetic average
// ========================================================================

#[test]
fn test_mean_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = a.mean().unwrap();
    assert!((result - 2.5).abs() < 1e-5); // (1+2+3+4)/4 = 2.5
}

#[test]
fn test_mean_negative() {
    let a = Vector::from_slice(&[-2.0, -4.0, -6.0]);
    let result = a.mean().unwrap();
    assert!((result - (-4.0)).abs() < 1e-5); // (-2-4-6)/3 = -4.0
}

#[test]
fn test_mean_mixed() {
    let a = Vector::from_slice(&[-10.0, 0.0, 10.0]);
    let result = a.mean().unwrap();
    assert!(result.abs() < 1e-5); // (-10+0+10)/3 = 0.0
}

#[test]
fn test_mean_single() {
    let a = Vector::from_slice(&[42.0]);
    let result = a.mean().unwrap();
    assert!((result - 42.0).abs() < 1e-5); // 42/1 = 42
}

#[test]
fn test_mean_all_same() {
    let a = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
    let result = a.mean().unwrap();
    assert!((result - 5.0).abs() < 1e-5); // (5+5+5+5+5)/5 = 5
}

#[test]
fn test_mean_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.mean();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for variance() - population variance
// ========================================================================

#[test]
fn test_variance_basic() {
    // Variance of [1,2,3,4,5]: mean=3, var=E[X²]-μ²=11-9=2
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = a.variance().unwrap();
    assert!((result - 2.0).abs() < 1e-5);
}

#[test]
fn test_variance_constant() {
    // Variance of constant vector is 0
    let a = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
    let result = a.variance().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_variance_single() {
    // Variance of single element is 0
    let a = Vector::from_slice(&[42.0]);
    let result = a.variance().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_variance_symmetric() {
    // Variance of [-2, -1, 0, 1, 2]: mean=0, var=E[X²]=2
    let a = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = a.variance().unwrap();
    assert!((result - 2.0).abs() < 1e-5);
}

#[test]
fn test_variance_two_values() {
    // Variance of [1, 5]: mean=3, var=(1-3)²+(5-3)²/2=8/2=4
    let a = Vector::from_slice(&[1.0, 5.0]);
    let result = a.variance().unwrap();
    assert!((result - 4.0).abs() < 1e-5);
}

#[test]
fn test_variance_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.variance();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for stddev() - standard deviation
// ========================================================================

#[test]
fn test_stddev_basic() {
    // stddev of [1,2,3,4,5]: variance=2, stddev=sqrt(2)≈1.414
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = a.stddev().unwrap();
    assert!((result - std::f32::consts::SQRT_2).abs() < 1e-5);
}

#[test]
fn test_stddev_constant() {
    // stddev of constant vector is 0
    let a = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
    let result = a.stddev().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_stddev_single() {
    // stddev of single element is 0
    let a = Vector::from_slice(&[42.0]);
    let result = a.stddev().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_stddev_symmetric() {
    // stddev of [-2,-1,0,1,2]: variance=2, stddev=sqrt(2)
    let a = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = a.stddev().unwrap();
    assert!((result - std::f32::consts::SQRT_2).abs() < 1e-5);
}

#[test]
fn test_stddev_two_values() {
    // stddev of [1,5]: variance=4, stddev=2
    let a = Vector::from_slice(&[1.0, 5.0]);
    let result = a.stddev().unwrap();
    assert!((result - 2.0).abs() < 1e-5);
}

#[test]
fn test_stddev_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.stddev();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for covariance() - population covariance
// ========================================================================

#[test]
fn test_covariance_positive() {
    // Perfect positive linear relationship: y = 2x
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let result = x.covariance(&y).unwrap();
    // Cov(X,2X) = 2*Var(X) = 2*(2/3) = 4/3 ≈ 1.333
    assert!((result - (4.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_covariance_negative() {
    // Negative linear relationship
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[3.0, 2.0, 1.0]);
    let result = x.covariance(&y).unwrap();
    assert!((result - (-2.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_covariance_zero() {
    // No linear relationship
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 3.0, 1.0, 3.0]);
    let result = x.covariance(&y).unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_covariance_self() {
    // Cov(X,X) = Var(X)
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let cov = x.covariance(&x).unwrap();
    let var = x.variance().unwrap();
    assert!((cov - var).abs() < 1e-5);
}

#[test]
fn test_covariance_size_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = x.covariance(&y);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { expected: 2, actual: 3 })));
}

#[test]
fn test_covariance_empty() {
    let x: Vector<f32> = Vector::from_slice(&[]);
    let y: Vector<f32> = Vector::from_slice(&[]);
    let result = x.covariance(&y);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for correlation() - Pearson correlation coefficient
// ========================================================================

#[test]
fn test_correlation_perfect_positive() {
    // Perfect positive linear relationship: y = 2x
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let result = x.correlation(&y).unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_correlation_perfect_negative() {
    // Perfect negative linear relationship
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y = Vector::from_slice(&[4.0, 3.0, 2.0, 1.0]);
    let result = x.correlation(&y).unwrap();
    assert!((result - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_correlation_zero() {
    // No correlation
    let x = Vector::from_slice(&[1.0, 2.0, 1.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 1.0, 2.0, 2.0]);
    let result = x.correlation(&y).unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_correlation_self() {
    // Correlation with self is always 1
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = x.correlation(&x).unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_correlation_constant_vector() {
    // Constant vector has zero std dev → division by zero
    let x = Vector::from_slice(&[5.0, 5.0, 5.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = x.correlation(&y);
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_correlation_size_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = x.correlation(&y);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { expected: 2, actual: 3 })));
}

// ========================================================================
