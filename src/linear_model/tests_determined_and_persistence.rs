use super::*;

#[test]
fn test_exactly_determined_system() {
    // n_samples == n_features + 1 (exactly determined with intercept)
    // 4 samples, 3 features, fit_intercept=true means 4 parameters
    let x = Matrix::from_vec(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0, 6.0]);

    let mut model = LinearRegression::new();
    let result = model.fit(&x, &y);

    // This should succeed (exactly determined)
    assert!(result.is_ok(), "Exactly determined system should work");
}

#[test]
fn test_save_load_binary() {
    use std::fs;
    use std::path::Path;

    // Train a model
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]); // y = 2x + 1

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Save to file
    let path = Path::new("/tmp/test_linear_regression.bin");
    model.save(path).expect("Failed to save model");

    // Load from file
    let loaded_model = LinearRegression::load(path).expect("Failed to load model");

    // Verify loaded model matches original
    let original_pred = model.predict(&x);
    let loaded_pred = loaded_model.predict(&x);

    for i in 0..original_pred.len() {
        assert!(
            (original_pred[i] - loaded_pred[i]).abs() < 1e-6,
            "Loaded model predictions don't match original"
        );
    }

    // Verify coefficients and intercept match
    assert_eq!(
        model.coefficients().len(),
        loaded_model.coefficients().len()
    );
    for i in 0..model.coefficients().len() {
        assert!((model.coefficients()[i] - loaded_model.coefficients()[i]).abs() < 1e-6);
    }
    assert!((model.intercept() - loaded_model.intercept()).abs() < 1e-6);

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_with_intercept_returns_self() {
    // Test that with_intercept returns the modified self, not a default
    // This catches the mutation: with_intercept -> Default::default()
    let model = LinearRegression::new().with_intercept(false);

    // If mutation returns Default::default(), fit_intercept would be true
    // Since new() sets fit_intercept = true by default

    // We need to verify the model actually has fit_intercept = false
    // by checking the fitted behavior
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]); // y = 2x

    let mut model = model;
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Without intercept, the model should pass through origin
    // Predicting x=0 should give y=0 (no intercept term)
    let x_zero = Matrix::from_vec(1, 1, vec![0.0]).expect("Valid matrix dimensions for test");
    let pred = model.predict(&x_zero);

    assert!(
        pred[0].abs() < 1e-6,
        "Model without intercept should predict 0 at x=0, got {}",
        pred[0]
    );
}

#[test]
fn test_with_intercept_builder_chain() {
    // Test that builder pattern works correctly
    // with_intercept(false) followed by fitting should not have intercept
    let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0]); // y = 2x + 1

    // Model with intercept
    let mut with_int = LinearRegression::new().with_intercept(true);
    with_int
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Model without intercept
    let mut without_int = LinearRegression::new().with_intercept(false);
    without_int
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // The intercept should be different
    // With intercept: should have non-zero intercept for this data
    // Without intercept: intercept is always 0
    assert!(
        with_int.intercept().abs() > 0.1,
        "Model with intercept should have non-zero intercept"
    );
    assert!(
        without_int.intercept().abs() < 1e-6,
        "Model without intercept should have zero intercept, got {}",
        without_int.intercept()
    );
}

// Ridge regression tests
#[test]
fn test_ridge_new() {
    let model = Ridge::new(1.0);
    assert!(!model.is_fitted());
    assert!((model.alpha() - 1.0).abs() < 1e-6);
}

#[test]
fn test_ridge_simple_regression() {
    // y = 2x + 1
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Ridge::new(0.0); // No regularization = OLS
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());

    // Check predictions are close (might not be perfect due to regularization)
    let r2 = model.score(&x, &y);
    assert!(r2 > 0.99);
}

#[test]
fn test_ridge_regularization_shrinks_coefficients() {
    // Test that higher alpha shrinks coefficients
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[4.0, 8.0, 12.0, 16.0, 20.0]);

    // Low regularization
    let mut low_reg = Ridge::new(0.01);
    low_reg
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // High regularization
    let mut high_reg = Ridge::new(100.0);
    high_reg
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Higher regularization should produce smaller coefficient magnitudes
    let low_coef = low_reg.coefficients();
    let high_coef = high_reg.coefficients();
    let low_norm: f32 = (0..low_coef.len()).map(|i| low_coef[i] * low_coef[i]).sum();
    let high_norm: f32 = (0..high_coef.len())
        .map(|i| high_coef[i] * high_coef[i])
        .sum();

    assert!(
        high_norm < low_norm,
        "High regularization should shrink coefficients: {high_norm} < {low_norm}"
    );
}

#[test]
fn test_ridge_multivariate() {
    // y = 1 + 2*x1 + 3*x2
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0, 16.0]);

    let mut model = Ridge::new(0.1);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let r2 = model.score(&x, &y);
    assert!(r2 > 0.95);
}

#[test]
fn test_ridge_no_intercept() {
    // y = 2x
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = Ridge::new(0.1).with_intercept(false);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!((model.intercept() - 0.0).abs() < 1e-6);
}

#[test]
fn test_ridge_dimension_mismatch_error() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = Ridge::new(1.0);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_ridge_empty_data_error() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = Ridge::new(1.0);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_ridge_underdetermined_system() {
    // Ridge can handle underdetermined systems due to regularization
    // 3 samples, 5 features
    let x = Matrix::from_vec(
        3,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![10.0, 20.0, 30.0]);

    // With sufficient regularization, this should work
    let mut model = Ridge::new(10.0);
    let result = model.fit(&x, &y);
    assert!(
        result.is_ok(),
        "Ridge should handle underdetermined systems"
    );
}

#[test]
fn test_ridge_clone() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = Ridge::new(0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
    assert!((cloned.alpha() - model.alpha()).abs() < 1e-6);
    assert!((cloned.intercept() - model.intercept()).abs() < 1e-6);
}

#[test]
fn test_ridge_alpha_zero_equals_ols() {
    // Ridge with alpha=0 should give same results as OLS
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut ridge = Ridge::new(0.0);
    ridge
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let mut ols = LinearRegression::new();
    ols.fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Coefficients should be nearly identical
    assert!(
        (ridge.coefficients()[0] - ols.coefficients()[0]).abs() < 1e-4,
        "Ridge with alpha=0 should equal OLS"
    );
    assert!((ridge.intercept() - ols.intercept()).abs() < 1e-4);
}

#[test]
fn test_ridge_save_load() {
    use std::fs;
    use std::path::Path;

    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Ridge::new(0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let path = Path::new("/tmp/test_ridge.bin");
    model.save(path).expect("Failed to save model");

    let loaded = Ridge::load(path).expect("Failed to load model");

    // Verify loaded model matches original
    assert!((loaded.alpha() - model.alpha()).abs() < 1e-6);
    let original_pred = model.predict(&x);
    let loaded_pred = loaded.predict(&x);

    for i in 0..original_pred.len() {
        assert!((original_pred[i] - loaded_pred[i]).abs() < 1e-6);
    }

    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_with_intercept_builder() {
    let model = Ridge::new(1.0).with_intercept(false);

    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = model;
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Without intercept, predicting at x=0 should give 0
    let x_zero = Matrix::from_vec(1, 1, vec![0.0]).expect("Valid matrix dimensions for test");
    let pred = model.predict(&x_zero);

    assert!(
        pred[0].abs() < 1e-6,
        "Ridge without intercept should predict 0 at x=0"
    );
}

#[test]
fn test_ridge_coefficients_length() {
    let x = Matrix::from_vec(5, 3, vec![1.0; 15]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let mut model = Ridge::new(1.0);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert_eq!(model.coefficients().len(), 3);
}

// Lasso regression tests
#[test]
fn test_lasso_new() {
    let model = Lasso::new(1.0);
    assert!(!model.is_fitted());
    assert!((model.alpha() - 1.0).abs() < 1e-6);
}

#[test]
fn test_lasso_simple_regression() {
    // y = 2x + 1
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0]);

    let mut model = Lasso::new(0.01); // Small regularization
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());

    let r2 = model.score(&x, &y);
    assert!(r2 > 0.98, "R² should be > 0.98, got {r2}");
}

#[test]
fn test_lasso_produces_sparsity() {
    // Test that Lasso with high alpha produces sparse coefficients
    // Create data where only first feature matters: y = x1
    let x = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 0.1, 0.2, 2.0, 0.2, 0.1, 3.0, 0.1, 0.3, 4.0, 0.3, 0.1, 5.0, 0.2, 0.2, 6.0, 0.1,
            0.1,
        ],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut model = Lasso::new(1.0); // High regularization
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Count non-zero coefficients
    let coef = model.coefficients();
    let mut non_zero = 0;
    for i in 0..coef.len() {
        if coef[i].abs() > 1e-4 {
            non_zero += 1;
        }
    }

    // With high alpha, some coefficients should be zeroed out
    assert!(
        non_zero < coef.len(),
        "Lasso should produce sparse solution, got {} non-zero out of {}",
        non_zero,
        coef.len()
    );
}
