//! Tests for linear_model module.

pub(crate) use super::*;

#[test]
fn test_new() {
    let model = LinearRegression::new();
    assert!(!model.is_fitted());
    assert!(model.fit_intercept);
}

#[test]
fn test_simple_regression() {
    // y = 2x + 1
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());

    // Check coefficients
    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-4);
    assert!((model.intercept() - 1.0).abs() < 1e-4);

    // Check predictions
    let predictions = model.predict(&x);
    for i in 0..4 {
        assert!((predictions[i] - y[i]).abs() < 1e-4);
    }

    // Check R²
    let r2 = model.score(&x, &y);
    assert!((r2 - 1.0).abs() < 1e-4);
}

#[test]
fn test_multivariate_regression() {
    // y = 1 + 2*x1 + 3*x2
    let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-4);
    assert!((coef[1] - 3.0).abs() < 1e-4);
    assert!((model.intercept() - 1.0).abs() < 1e-4);

    let r2 = model.score(&x, &y);
    assert!((r2 - 1.0).abs() < 1e-4);
}

#[test]
fn test_no_intercept() {
    // y = 2x (no intercept)
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = LinearRegression::new().with_intercept(false);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-4);
    assert!((model.intercept() - 0.0).abs() < 1e-4);
}

#[test]
fn test_predict_new_data() {
    // y = x + 1
    let x_train =
        Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y_train = Vector::from_slice(&[2.0, 3.0, 4.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x_train, &y_train)
        .expect("Fit should succeed with valid test data");

    let x_test = Matrix::from_vec(2, 1, vec![4.0, 5.0]).expect("Valid matrix dimensions for test");
    let predictions = model.predict(&x_test);

    assert!((predictions[0] - 5.0).abs() < 1e-4);
    assert!((predictions[1] - 6.0).abs() < 1e-4);
}

#[test]
fn test_dimension_mismatch_error() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = LinearRegression::new();
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_empty_data_error() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = LinearRegression::new();
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_with_noise() {
    // y ≈ 2x + 1 with some noise
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.1, 4.9, 7.2, 8.8, 11.1]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Should still get approximately correct coefficients
    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 0.2);
    assert!((model.intercept() - 1.0).abs() < 0.5);

    // R² should be high but not perfect
    let r2 = model.score(&x, &y);
    assert!(r2 > 0.95);
    assert!(r2 < 1.0);
}

#[test]
fn test_default() {
    let model = LinearRegression::default();
    assert!(!model.is_fitted());
}

#[test]
fn test_clone() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
    assert!((cloned.intercept() - model.intercept()).abs() < 1e-6);
}

#[test]
fn test_score_range() {
    // R² should be between negative infinity and 1
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let r2 = model.score(&x, &y);
    assert!(r2 <= 1.0);
}

#[test]
fn test_prediction_invariant() {
    // Property: predict(fit(X, y), X) should approximate y
    // Use non-collinear data
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0])
        .expect("Valid matrix dimensions for test");
    // y = 2*x1 + 3*x2 + 1
    let y = Vector::from_slice(&[6.0, 14.0, 13.0, 24.0, 23.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let predictions = model.predict(&x);

    for i in 0..y.len() {
        assert!((predictions[i] - y[i]).abs() < 1e-3);
    }
}

#[test]
fn test_coefficients_length_invariant() {
    // Property: coefficients.len() == n_features
    // Use well-conditioned data with independent columns
    let x = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            1.0,
        ],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 3.0, 5.0, 4.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert_eq!(model.coefficients().len(), 3);
}

#[test]
fn test_larger_dataset() {
    // Test with more samples
    let n = 100;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let x_val = i as f32;
        x_data.push(x_val);
        y_data.push(2.0 * x_val + 3.0); // y = 2x + 3
    }

    let x = Matrix::from_vec(n, 1, x_data).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(y_data);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-3);
    assert!((model.intercept() - 3.0).abs() < 1e-3);
}

#[test]
fn test_single_sample_single_feature() {
    // Edge case: minimum viable data
    let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // y = 2x + 1
    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-4);
    assert!((model.intercept() - 1.0).abs() < 1e-4);
}

#[test]
fn test_negative_values() {
    // Test with negative coefficients and values
    let x = Matrix::from_vec(4, 1, vec![-2.0, -1.0, 0.0, 1.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[5.0, 3.0, 1.0, -1.0]); // y = -2x + 1

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - (-2.0)).abs() < 1e-4);
    assert!((model.intercept() - 1.0).abs() < 1e-4);
}

#[test]
fn test_large_values() {
    // Test numerical stability with large values
    let x = Matrix::from_vec(3, 1, vec![1000.0, 2000.0, 3000.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2001.0, 4001.0, 6001.0]); // y = 2x + 1

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-2);
    assert!((model.intercept() - 1.0).abs() < 10.0); // Larger tolerance for large values
}

#[test]
fn test_small_values() {
    // Test with small values
    let x = Matrix::from_vec(3, 1, vec![0.001, 0.002, 0.003])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[0.003, 0.005, 0.007]); // y = 2x + 0.001

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-2);
}

#[test]
fn test_zero_intercept_data() {
    // Data that should produce zero intercept
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]); // y = 2x

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 1e-4);
    assert!(model.intercept().abs() < 1e-4);
}

#[test]
fn test_constant_target() {
    // All y values are the same
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[5.0, 5.0, 5.0]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Coefficient should be ~0, intercept should be ~5
    let coef = model.coefficients();
    assert!(coef[0].abs() < 1e-4);
    assert!((model.intercept() - 5.0).abs() < 1e-4);
}

#[test]
fn test_r2_score_bounds() {
    // R² should be in reasonable range for good fit
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.1, 3.9, 6.1, 7.9, 10.1]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let r2 = model.score(&x, &y);
    assert!(r2 > 0.0);
    assert!(r2 <= 1.0);
}

#[test]
fn test_extrapolation() {
    // Test prediction outside training range
    let x_train =
        Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y_train = Vector::from_slice(&[2.0, 4.0, 6.0]); // y = 2x

    let mut model = LinearRegression::new();
    model
        .fit(&x_train, &y_train)
        .expect("Fit should succeed with valid test data");

    // Predict at x = 10 (extrapolation)
    let x_test = Matrix::from_vec(1, 1, vec![10.0]).expect("Valid matrix dimensions for test");
    let predictions = model.predict(&x_test);

    assert!((predictions[0] - 20.0).abs() < 1e-4);
}

#[test]
fn test_underdetermined_system_with_intercept() {
    // n_samples < n_features + 1 (underdetermined with intercept)
    // 3 samples, 5 features, fit_intercept=true means we need 6 parameters
    let x = Matrix::from_vec(
        3,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![10.0, 20.0, 30.0]);

    let mut model = LinearRegression::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let error_msg = result.expect_err("Should fail when underdetermined system with intercept");
    let error_str = error_msg.to_string();
    // Should mention samples, features, and suggest solutions
    assert!(
        error_str.contains("samples") || error_str.contains("features"),
        "Error message should mention samples or features: {error_str}"
    );
}

#[test]
fn test_underdetermined_system_without_intercept() {
    // n_samples < n_features (underdetermined without intercept)
    // 3 samples, 5 features, fit_intercept=false
    let x = Matrix::from_vec(
        3,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![10.0, 20.0, 30.0]);

    let mut model = LinearRegression::new().with_intercept(false);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let error_msg = result.expect_err("Should fail when underdetermined system without intercept");
    let error_str = error_msg.to_string();
    assert!(
        error_str.contains("samples") || error_str.contains("features"),
        "Error message should be helpful: {error_str}"
    );
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_part_04.rs"]
mod tests_part_04;
#[path = "tests_part_05.rs"]
mod tests_part_05;
