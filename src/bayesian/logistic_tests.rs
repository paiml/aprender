use super::*;

/// Test: Constructor creates model with correct defaults
#[test]
fn test_new() {
    let model = BayesianLogisticRegression::new(1.0);
    assert!(model.coefficients_map.is_none());
    assert!(model.posterior_covariance.is_none());
}

/// Test: Builder pattern methods
#[test]
fn test_builder_pattern() {
    let model = BayesianLogisticRegression::new(1.0)
        .with_learning_rate(0.1)
        .with_max_iter(500)
        .with_tolerance(1e-3);

    // Model should be created successfully
    assert!(model.coefficients_map.is_none());
}

/// Test: Fit with simple linearly separable data
#[test]
fn test_fit_simple() {
    // Linearly separable data: y = 1 if x > 0, else 0
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Fit should succeed");
    assert!(model.coefficients_map.is_some());
    assert!(model.posterior_covariance.is_some());

    // Coefficient should be positive (positive correlation)
    let beta = model
        .coefficients_map
        .as_ref()
        .expect("MAP estimate exists");
    assert!(
        beta[0] > 0.0,
        "Coefficient should be positive, got {}",
        beta[0]
    );
}

/// Test: Predict probabilities
#[test]
fn test_predict_proba() {
    // Train on simple data
    let x = Matrix::from_vec(4, 1, vec![-1.0, -0.5, 0.5, 1.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    // Predict on new data
    let x_test = Matrix::from_vec(3, 1, vec![-2.0, 0.0, 2.0]).expect("Valid test matrix");
    let probas = model.predict_proba(&x_test).expect("Prediction succeeds");

    assert_eq!(probas.len(), 3);

    // Probabilities should be in [0, 1]
    for &p in probas.as_slice() {
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability should be in [0,1], got {p}"
        );
    }

    // Probabilities should be monotonically increasing
    assert!(probas[0] < probas[1], "P(y=1 | x=-2) < P(y=1 | x=0)");
    assert!(probas[1] < probas[2], "P(y=1 | x=0) < P(y=1 | x=2)");
}

/// Test: Predict binary labels
#[test]
fn test_predict() {
    let x = Matrix::from_vec(4, 1, vec![-1.0, -0.5, 0.5, 1.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    let x_test = Matrix::from_vec(2, 1, vec![-2.0, 2.0]).expect("Valid test matrix");
    let labels = model.predict(&x_test).expect("Prediction succeeds");

    assert_eq!(labels.len(), 2);

    // Labels should be 0.0 or 1.0
    for &label in labels.as_slice() {
        assert!(
            label == 0.0 || label == 1.0,
            "Label should be 0 or 1, got {label}"
        );
    }
}

/// Test: Dimension mismatch in fit
#[test]
fn test_fit_dimension_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 1.0]); // Wrong size!

    let mut model = BayesianLogisticRegression::new(1.0);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
}

/// Test: Invalid labels (not 0 or 1)
#[test]
fn test_fit_invalid_labels() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.5, 1.0]); // 0.5 is invalid!

    let mut model = BayesianLogisticRegression::new(1.0);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Predict before fit should error
#[test]
fn test_predict_not_fitted() {
    let model = BayesianLogisticRegression::new(1.0);
    let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix");

    let result = model.predict_proba(&x_test);
    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Predict with dimension mismatch
#[test]
fn test_predict_dimension_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(1.0);
    model.fit(&x, &y).expect("Fit succeeds");

    // Try to predict with wrong number of features
    let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid test matrix");
    let result = model.predict_proba(&x_test);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
}

/// Test: MAP estimate converges
#[test]
fn test_map_convergence() {
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1)
        .with_max_iter(2000)
        .with_tolerance(1e-5);

    let result = model.fit(&x, &y);
    assert!(result.is_ok(), "MAP estimation should converge");
}

/// Test: Non-convergence with low max_iter
#[test]
fn test_map_non_convergence() {
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1)
        .with_max_iter(5) // Too few iterations
        .with_tolerance(1e-10); // Very strict tolerance

    let result = model.fit(&x, &y);
    assert!(result.is_err(), "Should fail to converge");
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Predict probabilities with credible intervals
#[test]
fn test_predict_proba_interval() {
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    // Predict with 95% credible intervals
    let x_test = Matrix::from_vec(3, 1, vec![-2.0, 0.0, 2.0]).expect("Valid test matrix");
    let (lower, upper) = model
        .predict_proba_interval(&x_test, 0.95)
        .expect("Interval prediction succeeds");

    assert_eq!(lower.len(), 3);
    assert_eq!(upper.len(), 3);

    // Get point predictions
    let probas = model.predict_proba(&x_test).expect("Prediction succeeds");

    // Bounds should contain the point predictions
    for i in 0..3 {
        assert!(
            lower[i] <= probas[i],
            "Lower bound should be <= point estimate: {i}: {} <= {}",
            lower[i],
            probas[i]
        );
        assert!(
            probas[i] <= upper[i],
            "Upper bound should be >= point estimate: {i}: {} >= {}",
            probas[i],
            upper[i]
        );
        assert!(
            lower[i] >= 0.0 && lower[i] <= 1.0,
            "Lower bound should be in [0,1], got {}",
            lower[i]
        );
        assert!(
            upper[i] >= 0.0 && upper[i] <= 1.0,
            "Upper bound should be in [0,1], got {}",
            upper[i]
        );
    }

    // Intervals should have non-negative width (may be small for certain x values)
    for i in 0..3 {
        assert!(
            upper[i] >= lower[i],
            "Upper bound should be >= lower bound at {i}: {} >= {}",
            upper[i],
            lower[i]
        );
    }

    // At least some intervals should have meaningful width
    let max_width = (0..3).map(|i| upper[i] - lower[i]).fold(0.0_f32, f32::max);
    assert!(
        max_width > 0.01,
        "At least one interval should have width > 0.01, max was {max_width}"
    );
}

/// Test: Interval prediction before fit should error
#[test]
fn test_predict_interval_not_fitted() {
    let model = BayesianLogisticRegression::new(1.0);
    let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix");

    let result = model.predict_proba_interval(&x_test, 0.95);
    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

// ========================================================================
// Additional Coverage Tests for bayesian/logistic.rs
// ========================================================================

#[test]
fn test_sigmoid_extreme_values() {
    // Extreme negative value
    let sig_neg = BayesianLogisticRegression::sigmoid(-100.0);
    assert!(sig_neg < 1e-10);

    // Extreme positive value
    let sig_pos = BayesianLogisticRegression::sigmoid(100.0);
    assert!(sig_pos > 0.9999999);

    // Zero
    let sig_zero = BayesianLogisticRegression::sigmoid(0.0);
    assert!((sig_zero - 0.5).abs() < 1e-6);
}

#[test]
fn test_prior_precision_effects() {
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    // Medium prior precision = moderate regularization
    let mut model_low = BayesianLogisticRegression::new(0.1);
    model_low.fit(&x, &y).expect("Fit succeeds");

    // High prior precision = strong regularization
    let mut model_high = BayesianLogisticRegression::new(10.0);
    model_high.fit(&x, &y).expect("Fit succeeds");

    // Higher regularization should shrink coefficients toward zero
    let beta_low = model_low
        .coefficients_map
        .as_ref()
        .expect("has coefficients");
    let beta_high = model_high
        .coefficients_map
        .as_ref()
        .expect("has coefficients");

    assert!(
        beta_low[0].abs() >= beta_high[0].abs(),
        "Higher prior precision should shrink coefficients"
    );
}

#[test]
fn test_multiple_features() {
    // Create data with 2 features
    let x = Matrix::from_vec(4, 2, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        .expect("Valid matrix");
    let y = Vector::from_vec(vec![1.0, 1.0, 0.0, 0.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    // Should have 2 coefficients (plus intercept if applicable)
    let beta = model.coefficients_map.as_ref().unwrap();
    assert!(beta.len() >= 2);
}

#[test]
fn test_predict_interval_dimension_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(1.0);
    model.fit(&x, &y).expect("Fit succeeds");

    // Wrong number of features
    let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid test matrix");
    let result = model.predict_proba_interval(&x_test, 0.95);

    assert!(result.is_err());
}

#[test]
fn test_predict_returns_labels() {
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    let x_test = Matrix::from_vec(4, 1, vec![-3.0, -0.1, 0.1, 3.0]).expect("Valid test matrix");
    let labels = model.predict(&x_test).expect("Prediction succeeds");

    // All labels should be 0.0 or 1.0
    for &label in labels.as_slice() {
        assert!(label == 0.0 || label == 1.0);
    }

    // Very negative x should predict 0, very positive should predict 1
    assert_eq!(labels[0], 0.0);
    assert_eq!(labels[3], 1.0);
}

#[test]
fn test_wide_credible_interval() {
    let x =
        Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    // 99% credible interval should be wider than 90%
    let x_test = Matrix::from_vec(1, 1, vec![0.0]).expect("Valid test matrix");

    let (lower_90, upper_90) = model
        .predict_proba_interval(&x_test, 0.90)
        .expect("Interval succeeds");
    let (lower_99, upper_99) = model
        .predict_proba_interval(&x_test, 0.99)
        .expect("Interval succeeds");

    let width_90 = upper_90[0] - lower_90[0];
    let width_99 = upper_99[0] - lower_99[0];

    assert!(
        width_99 >= width_90,
        "99% CI should be wider than 90% CI: {} >= {}",
        width_99,
        width_90
    );
}
