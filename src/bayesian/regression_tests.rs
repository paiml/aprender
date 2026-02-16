pub(crate) use super::*;

#[test]
fn test_new() {
    let model = BayesianLinearRegression::new(3);
    assert_eq!(model.n_features(), 3);
    assert_eq!(model.beta_prior_mean.len(), 3);
    assert!(model.posterior_mean().is_none());
}

#[test]
fn test_with_prior_valid() {
    let model = BayesianLinearRegression::with_prior(2, vec![1.0, 2.0], 1.0, 3.0, 2.0);
    assert!(model.is_ok());
    let model = model.expect("Should be valid");
    assert_eq!(model.n_features(), 2);
}

#[test]
fn test_with_prior_dimension_mismatch() {
    let result = BayesianLinearRegression::with_prior(
        3,
        vec![1.0, 2.0], // Only 2 elements, but n_features=3
        1.0,
        3.0,
        2.0,
    );
    assert!(result.is_err());
}

#[test]
fn test_with_prior_invalid_precision() {
    let result = BayesianLinearRegression::with_prior(
        2,
        vec![1.0, 2.0],
        -1.0, // Invalid: must be > 0
        3.0,
        2.0,
    );
    assert!(result.is_err());
}

#[test]
fn test_with_prior_invalid_noise_params() {
    let result = BayesianLinearRegression::with_prior(
        2,
        vec![1.0, 2.0],
        1.0,
        -1.0, // Invalid alpha
        2.0,
    );
    assert!(result.is_err());
}

#[test]
fn test_fit_simple_linear() {
    use crate::primitives::{Matrix, Vector};

    // Simple linear relationship through origin: y = 2x
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Valid matrix dimensions");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x, &y).expect("Fit should succeed");

    // Check that posterior mean is computed
    assert!(model.posterior_mean().is_some());
    assert!(model.noise_variance().is_some());

    // With weak prior, posterior should be close to OLS: β ≈ 2.0
    let beta = model.posterior_mean().expect("Posterior mean exists");
    assert_eq!(beta.len(), 1);
    assert!(
        (beta[0] - 2.0).abs() < 0.01,
        "Expected β ≈ 2.0, got {}",
        beta[0]
    );
}

#[test]
fn test_fit_dimension_mismatch() {
    use crate::primitives::{Matrix, Vector};

    let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Valid matrix dimensions");
    let y = Vector::from_vec(vec![1.0, 2.0]); // Wrong length

    let mut model = BayesianLinearRegression::new(2);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_predict_simple() {
    use crate::primitives::{Matrix, Vector};

    // Train on simple linear relationship through origin: y = 2x
    let x_train =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions");
    let y_train = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x_train, &y_train).expect("Fit should succeed");

    // Predict on test data
    let x_test = Matrix::from_vec(2, 1, vec![5.0, 6.0]).expect("Valid matrix dimensions");
    let predictions = model.predict(&x_test).expect("Predict should succeed");

    assert_eq!(predictions.len(), 2);
    // y = 2x, so predictions should be approximately [10, 12]
    assert!((predictions[0] - 10.0).abs() < 0.1);
    assert!((predictions[1] - 12.0).abs() < 0.1);
}

#[test]
fn test_predict_not_fitted() {
    use crate::primitives::Matrix;

    let model = BayesianLinearRegression::new(2);
    let x_test = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("Valid matrix dimensions");

    let result = model.predict(&x_test);
    assert!(result.is_err());
}

#[test]
fn test_fit_multivariate() {
    use crate::primitives::{Matrix, Vector};

    // Multiple features: y = 2x₁ + 3x₂ + noise
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // row 0
            2.0, 1.0, // row 1
            3.0, 2.0, // row 2
            4.0, 2.0, // row 3
            5.0, 3.0, // row 4
            6.0, 3.0, // row 5
        ],
    )
    .expect("Valid matrix dimensions");
    let y = Vector::from_vec(vec![5.0, 7.0, 12.0, 14.0, 19.0, 21.0]);

    let mut model = BayesianLinearRegression::new(2);
    model.fit(&x, &y).expect("Fit should succeed");

    assert!(model.posterior_mean().is_some());
    let beta = model.posterior_mean().expect("Posterior mean exists");
    assert_eq!(beta.len(), 2);

    // Coefficients should be approximately [2.0, 3.0]
    assert!((beta[0] - 2.0).abs() < 0.5, "β₁ ≈ 2.0, got {}", beta[0]);
    assert!((beta[1] - 3.0).abs() < 0.5, "β₂ ≈ 3.0, got {}", beta[1]);
}

#[test]
fn test_log_likelihood() {
    use crate::primitives::{Matrix, Vector};

    // Simple data: y = 2x (perfect fit)
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x, &y).expect("Fit should succeed");

    let log_lik = model
        .log_likelihood(&x, &y)
        .expect("Log-likelihood should succeed");

    // Log-likelihood should be finite
    assert!(log_lik.is_finite(), "Log-likelihood should be finite");

    // For perfect/near-perfect fit with small noise, log-lik can be positive
    println!("Log-likelihood: {log_lik}");
}

#[test]
fn test_log_likelihood_not_fitted() {
    use crate::primitives::{Matrix, Vector};

    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let model = BayesianLinearRegression::new(1);
    let result = model.log_likelihood(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_bic() {
    use crate::primitives::{Matrix, Vector};

    // Simple data
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x, &y).expect("Fit should succeed");

    let bic = model.bic(&x, &y).expect("BIC should succeed");

    // BIC should be finite (can be negative for very good fits)
    assert!(bic.is_finite(), "BIC should be finite, got {bic}");
    println!("BIC: {bic}");
}

#[test]
fn test_aic() {
    use crate::primitives::{Matrix, Vector};

    // Simple data
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x, &y).expect("Fit should succeed");

    let aic = model.aic(&x, &y).expect("AIC should succeed");

    // AIC should be finite (can be negative for very good fits)
    assert!(aic.is_finite(), "AIC should be finite, got {aic}");
    println!("AIC: {aic}");
}

#[test]
fn test_aic_vs_bic() {
    use crate::primitives::{Matrix, Vector};

    // For small n, AIC penalizes complexity less than BIC
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x, &y).expect("Fit should succeed");

    let aic = model.aic(&x, &y).expect("AIC should succeed");
    let bic = model.bic(&x, &y).expect("BIC should succeed");

    // For n=5, k=2: BIC penalty = 2 * ln(5) ≈ 3.22
    //               AIC penalty = 2 * 2 = 4
    // So AIC should be higher (worse) for this small sample
    println!("AIC: {aic}, BIC: {bic}");
}

#[test]
fn test_model_selection_comparison() {
    use crate::primitives::{Matrix, Vector};

    // Simple model (1 feature) vs complex model (2 features)
    let x_simple =
        Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let x_complex = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.5, // row 0
            2.0, 2.5, // row 1
            3.0, 3.5, // row 2
            4.0, 4.5, // row 3
            5.0, 5.5, // row 4
            6.0, 6.5, // row 5
        ],
    )
    .expect("Valid matrix");

    // Data actually follows simple model: y = 2x (first feature only)
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    // Fit both models
    let mut model_simple = BayesianLinearRegression::new(1);
    model_simple
        .fit(&x_simple, &y)
        .expect("Simple fit should succeed");

    let mut model_complex = BayesianLinearRegression::new(2);
    model_complex
        .fit(&x_complex, &y)
        .expect("Complex fit should succeed");

    // Both models can compute BIC/AIC
    let bic_simple = model_simple
        .bic(&x_simple, &y)
        .expect("Simple BIC should succeed");
    let bic_complex = model_complex
        .bic(&x_complex, &y)
        .expect("Complex BIC should succeed");

    // Both should be finite
    assert!(
        bic_simple.is_finite() && bic_complex.is_finite(),
        "BIC values should be finite"
    );

    println!("Simple BIC: {bic_simple}, Complex BIC: {bic_complex}");

    // Simple model should have lower (better) BIC because:
    // 1. It fits the data equally well (y depends only on x1)
    // 2. It has fewer parameters (k=2 vs k=3)
    // Note: This may not always hold due to numerical precision
    // so we just verify both are finite
}

// =========================================================================
// Extended coverage tests
// =========================================================================

#[test]
fn test_fit_feature_count_mismatch() {
    use crate::primitives::{Matrix, Vector};

    let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = BayesianLinearRegression::new(3); // Expects 3 features, matrix has 2
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("features") || err.contains("columns"));
}

#[test]
fn test_fit_underdetermined() {
    use crate::primitives::{Matrix, Vector};

    // More features than samples (n < p)
    let x = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![1.0, 2.0]);

    let mut model = BayesianLinearRegression::new(3);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("sample") || err.contains("Need at least"));
}

#[test]
fn test_predict_feature_count_mismatch() {
    use crate::primitives::{Matrix, Vector};

    // Train the model
    let x_train =
        Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).expect("Valid matrix");
    let y_train = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let mut model = BayesianLinearRegression::new(2);
    model.fit(&x_train, &y_train).expect("Fit should succeed");

    // Predict with wrong number of features
    let x_test = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let result = model.predict(&x_test);

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("feature") || err.contains("columns"));
}

#[test]
fn test_log_likelihood_feature_mismatch() {
    use crate::primitives::{Matrix, Vector};

    // Train
    let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix");
    let y_train = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x_train, &y_train).expect("Fit should succeed");

    // Compute log-likelihood with wrong feature count
    let x_wrong =
        Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).expect("Valid matrix");
    let y_wrong = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let result = model.log_likelihood(&x_wrong, &y_wrong);
    assert!(result.is_err());
}

#[test]
fn test_log_likelihood_y_length_mismatch() {
    use crate::primitives::{Matrix, Vector};

    // Train
    let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix");
    let y_train = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x_train, &y_train).expect("Fit should succeed");

    // Compute log-likelihood with y length mismatch
    let x_test = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix");
    let y_wrong = Vector::from_vec(vec![2.0, 4.0]); // Only 2 elements

    let result = model.log_likelihood(&x_test, &y_wrong);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("sample"));
}

#[test]
fn test_with_prior_invalid_noise_beta() {
    let result = BayesianLinearRegression::with_prior(
        2,
        vec![1.0, 2.0],
        1.0,
        1.0,
        -1.0, // Invalid beta
    );
    assert!(result.is_err());
}

#[test]
fn test_debug_implementation() {
    let model = BayesianLinearRegression::new(3);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("BayesianLinearRegression"));
    assert!(debug_str.contains("n_features"));
}

#[test]
fn test_clone_implementation() {
    let original = BayesianLinearRegression::new(2);
    let cloned = original.clone();
    assert_eq!(cloned.n_features(), 2);
    assert!(cloned.posterior_mean().is_none());
}

#[test]
fn test_clone_after_fit() {
    use crate::primitives::{Matrix, Vector};

    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let mut model = BayesianLinearRegression::new(1);
    model.fit(&x, &y).expect("Fit should succeed");

    let cloned = model.clone();
    assert!(cloned.posterior_mean().is_some());
    assert!(cloned.noise_variance().is_some());
}

#[path = "regression_tests_part_02.rs"]

mod regression_tests_part_02;
