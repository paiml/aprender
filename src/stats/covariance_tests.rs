use super::*;

#[test]
fn test_cov_positive_relationship() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let covariance = cov(&x, &y).expect("Should compute covariance");
    assert!(
        covariance > 0.0,
        "Positive relationship should have positive covariance"
    );
}

#[test]
fn test_cov_negative_relationship() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::from_slice(&[10.0, 8.0, 6.0, 4.0, 2.0]);

    let covariance = cov(&x, &y).expect("Should compute covariance");
    assert!(
        covariance < 0.0,
        "Negative relationship should have negative covariance"
    );
}

#[test]
fn test_cov_dimension_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[1.0, 2.0]);

    let result = cov(&x, &y);
    assert!(result.is_err());
    let err = result.expect_err("Should be dimension mismatch");
    assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
}

#[test]
fn test_cov_empty() {
    let x = Vector::from_slice(&[]);
    let y = Vector::from_slice(&[]);

    let result = cov(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_cov_matrix_simple() {
    // 3 samples, 2 features
    let data =
        Matrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).expect("Valid matrix");

    let cov_mat = cov_matrix(&data).expect("Should compute covariance matrix");

    assert_eq!(cov_mat.n_rows(), 2);
    assert_eq!(cov_mat.n_cols(), 2);

    // Covariance matrix should be symmetric
    assert!((cov_mat.get(0, 1) - cov_mat.get(1, 0)).abs() < 1e-6);

    // Diagonal should be positive (variances)
    assert!(cov_mat.get(0, 0) > 0.0);
    assert!(cov_mat.get(1, 1) > 0.0);
}

#[test]
fn test_corr_perfect_positive() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let correlation = corr(&x, &y).expect("Should compute correlation");
    assert!(
        (correlation - 1.0).abs() < 1e-6,
        "Perfect positive correlation should be 1.0"
    );
}

#[test]
fn test_corr_perfect_negative() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y = Vector::from_slice(&[8.0, 6.0, 4.0, 2.0]);

    let correlation = corr(&x, &y).expect("Should compute correlation");
    assert!(
        (correlation + 1.0).abs() < 1e-6,
        "Perfect negative correlation should be -1.0"
    );
}

#[test]
fn test_corr_no_relationship() {
    // Orthogonal vectors (mean-centered)
    let x = Vector::from_slice(&[-1.0, 0.0, 1.0]);
    let y = Vector::from_slice(&[1.0, -2.0, 1.0]);

    let correlation = corr(&x, &y).expect("Should compute correlation");
    assert!(
        correlation.abs() < 0.1,
        "Orthogonal vectors should have near-zero correlation, got {correlation}"
    );
}

#[test]
fn test_corr_zero_variance() {
    let x = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let result = corr(&x, &y);
    assert!(result.is_err(), "Should error when variance is zero");
}

#[test]
fn test_corr_matrix_simple() {
    // 4 samples, 2 features
    let data = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0])
        .expect("Valid matrix");

    let corr_mat = corr_matrix(&data).expect("Should compute correlation matrix");

    assert_eq!(corr_mat.n_rows(), 2);
    assert_eq!(corr_mat.n_cols(), 2);

    // Diagonal should be 1.0
    assert!((corr_mat.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((corr_mat.get(1, 1) - 1.0).abs() < 1e-6);

    // Should be symmetric
    assert!((corr_mat.get(0, 1) - corr_mat.get(1, 0)).abs() < 1e-6);

    // Perfect correlation (y = 2*x)
    assert!((corr_mat.get(0, 1) - 1.0).abs() < 1e-6);
}

#[test]
fn test_corr_matrix_independent() {
    // 3 samples, 2 independent features
    let data =
        Matrix::from_vec(3, 2, vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0]).expect("Valid matrix");

    let result = corr_matrix(&data);
    // Second feature has zero variance
    assert!(result.is_err());
}

#[test]
fn test_corr_matrix_three_features() {
    // 3 samples, 3 features
    let data = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])
        .expect("Valid matrix");

    let corr_mat = corr_matrix(&data).expect("Should compute correlation matrix");

    assert_eq!(corr_mat.n_rows(), 3);
    assert_eq!(corr_mat.n_cols(), 3);

    // All diagonals should be 1.0
    for i in 0..3 {
        assert!((corr_mat.get(i, i) - 1.0).abs() < 1e-6);
    }

    // All features perfectly correlated (linear relationship)
    for i in 0..3 {
        for j in 0..3 {
            assert!((corr_mat.get(i, j) - 1.0).abs() < 1e-6);
        }
    }
}

// =========================================================================
// Additional coverage: error paths and edge cases
// =========================================================================

#[test]
fn test_corr_dimension_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[1.0, 2.0]);
    let result = corr(&x, &y);
    assert!(result.is_err());
    let err = result.expect_err("Should be dimension mismatch");
    assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
}

#[test]
fn test_corr_empty() {
    let x = Vector::from_slice(&[]);
    let y = Vector::from_slice(&[]);
    let result = corr(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_corr_both_zero_variance() {
    let x = Vector::from_slice(&[5.0, 5.0, 5.0]);
    let y = Vector::from_slice(&[3.0, 3.0, 3.0]);
    let result = corr(&x, &y);
    assert!(result.is_err(), "Should error when both have zero variance");
}

#[test]
fn test_corr_y_zero_variance() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[5.0, 5.0, 5.0]);
    let result = corr(&x, &y);
    assert!(result.is_err(), "Should error when y has zero variance");
}

#[test]
fn test_cov_matrix_empty_rows() {
    // 0 rows, 2 cols -> empty data
    let data = Matrix::from_vec(0, 0, vec![]);
    assert!(data.is_err() || cov_matrix(&data.expect("empty matrix")).is_err());
}

#[test]
fn test_corr_matrix_empty() {
    // Attempt to create an empty-like matrix. Since from_vec may reject 0x0,
    // test with a valid but single-row matrix where we cannot compute variance
    // meaningfully. Let's trigger the n==0||p==0 path differently.
    // Instead, just test with a constant-feature matrix for the zero-variance path.
    let data =
        Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0]).expect("Valid matrix");
    let result = corr_matrix(&data);
    // Feature 1 (column index 1) has zero variance
    assert!(result.is_err());
}

#[test]
fn test_cov_single_element() {
    let x = Vector::from_slice(&[5.0]);
    let y = Vector::from_slice(&[10.0]);
    let result = cov(&x, &y).expect("Should compute covariance for single element");
    // Cov of single element is 0 (no deviation)
    assert!((result - 0.0).abs() < 1e-10);
}

#[test]
fn test_cov_zero_covariance() {
    // x and y are uncorrelated (mean-centered, orthogonal)
    let x = Vector::from_slice(&[-1.0, 0.0, 1.0]);
    let y = Vector::from_slice(&[1.0, -2.0, 1.0]);
    let result = cov(&x, &y).expect("Should compute covariance");
    assert!(
        result.abs() < 1e-6,
        "Covariance of orthogonal vectors should be ~0"
    );
}

#[test]
fn test_cov_matrix_single_feature() {
    // 3 samples, 1 feature => 1x1 covariance matrix
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let cov_mat = cov_matrix(&data).expect("Should compute 1x1 covariance matrix");
    assert_eq!(cov_mat.n_rows(), 1);
    assert_eq!(cov_mat.n_cols(), 1);
    // Variance of [1,2,3] = 2/3
    assert!((cov_mat.get(0, 0) - 2.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_cov_matrix_single_sample() {
    // 1 sample, 2 features => all zeros (no variance with 1 sample)
    let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).expect("Valid matrix");
    let cov_mat = cov_matrix(&data).expect("Should compute covariance matrix");
    assert_eq!(cov_mat.n_rows(), 2);
    assert_eq!(cov_mat.n_cols(), 2);
    assert!((cov_mat.get(0, 0) - 0.0).abs() < 1e-10);
    assert!((cov_mat.get(1, 1) - 0.0).abs() < 1e-10);
}

#[test]
fn test_corr_matrix_negative_correlation() {
    // Feature 1 increases, feature 2 decreases
    let data = Matrix::from_vec(4, 2, vec![1.0, 8.0, 2.0, 6.0, 3.0, 4.0, 4.0, 2.0])
        .expect("Valid matrix");
    let corr_mat = corr_matrix(&data).expect("Should compute correlation matrix");
    // Off-diagonal should be negative (inverse relationship)
    assert!(
        corr_mat.get(0, 1) < 0.0,
        "Should have negative correlation, got {}",
        corr_mat.get(0, 1)
    );
    // Diagonal should be 1.0
    assert!((corr_mat.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((corr_mat.get(1, 1) - 1.0).abs() < 1e-6);
}

#[test]
fn test_corr_single_pair() {
    // Minimum valid correlation: 2 values each
    let x = Vector::from_slice(&[0.0, 1.0]);
    let y = Vector::from_slice(&[0.0, 1.0]);
    let result = corr(&x, &y).expect("Should compute correlation for 2 elements");
    assert!((result - 1.0).abs() < 1e-6, "Perfect positive correlation");
}

#[test]
fn test_cov_identical_vectors() {
    // Cov(X, X) = Var(X)
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let cov_xx = cov(&x, &x).expect("Should compute self-covariance");
    // Population variance of [1,2,3,4,5] = 2.0
    assert!(
        (cov_xx - 2.0).abs() < 1e-5,
        "Cov(X,X) should equal Var(X), got {cov_xx}"
    );
}
