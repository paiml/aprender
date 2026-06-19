pub(crate) use super::*;

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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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
    let x =
        Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("Valid matrix");
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
    let x =
        Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("Valid matrix");
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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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
    let x =
        Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("Valid matrix");
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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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
    let x = Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
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

// ========================================================================
// PMAT-864: MAP gradient/Hessian must target the SAME posterior (precision λ)
//
// Falsifier for the gradient/Hessian inconsistency: the log-posterior
// gradient must be the UN-AVERAGED ∇ = Xᵀ(y − p) − λβ so that its stationary
// point coincides with the mode used by the un-averaged Hessian
// H = XᵀWX + λI. If the data term is divided by n (the bug) while the prior
// term is not, the fit converges to the MAP of a model with precision n·λ —
// the coefficients are shrunk ~n× toward zero — and the Laplace covariance is
// then evaluated at the WRONG mode.
//
// Reference: Bishop PRML §4.5 (Laplace approximation); sklearn MAP for
// logistic regression: ∇ = Xᵀ(y − p) − λβ, H = XᵀWX + λI at the same mode.
// ========================================================================

/// Reference Newton solver for the logistic-regression MAP at a given prior
/// precision `lambda`. Solves the stationary equation Xᵀ(y − p) = λβ exactly
/// (full Newton on the un-averaged log-posterior), independent of the model's
/// gradient-ascent code path under test. Single feature column, no intercept.
fn reference_map_1d(x_col: &[f32], y: &[f32], lambda: f32) -> f32 {
    let mut beta = 0.0_f64;
    let lambda = f64::from(lambda);
    for _ in 0..200 {
        // p_i = σ(x_i β); gradient g = Σ x_i (y_i − p_i) − λβ
        // Hessian h = Σ x_i² p_i(1 − p_i) + λ
        let mut g = -lambda * beta;
        let mut h = lambda;
        for (&xi, &yi) in x_col.iter().zip(y.iter()) {
            let xi = f64::from(xi);
            let z = xi * beta;
            let p = 1.0 / (1.0 + (-z).exp());
            g += xi * (f64::from(yi) - p);
            h += xi * xi * p * (1.0 - p);
        }
        let step = g / h;
        beta += step;
        if step.abs() < 1e-12 {
            break;
        }
    }
    beta as f32
}

/// PMAT-864 falsifier: the fitted posterior mean must equal the MAP at the
/// DECLARED precision λ, NOT the over-shrunk MAP at precision n·λ.
///
/// RED (bug present, data term divided by n): `beta_fitted ≈ map(n·λ)`, badly
/// failing the `≈ map(λ)` assertion (and the `is clearly distinct from
/// map(n·λ)` assertion).
/// GREEN (fix, un-averaged gradient): `beta_fitted ≈ map(λ)`.
#[test]
fn test_pmat864_map_targets_declared_precision_not_n_lambda() {
    // n = 8 samples, single feature. Monotone-ordered labels give a strong
    // signal so the (finite, regularized) λ-MAP is large; the n·λ-MAP is
    // shrunk to ~½ of it — a clean, unambiguous separation between the two
    // candidate modes.
    let x_col = vec![-3.0_f32, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0];
    let y_vec = vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let n = x_col.len();
    let lambda = 0.2_f32;

    let x = Matrix::from_vec(n, 1, x_col.clone()).expect("Valid matrix");
    let y = Vector::from_vec(y_vec.clone());

    let mut model = BayesianLogisticRegression::new(lambda)
        .with_learning_rate(0.3)
        .with_max_iter(200_000)
        .with_tolerance(1e-5);
    model.fit(&x, &y).expect("Fit should converge");
    let beta_fitted = model
        .coefficients_map
        .as_ref()
        .expect("MAP estimate exists")[0];

    // Reference modes computed by an independent Newton solver.
    let beta_lambda = reference_map_1d(&x_col, &y_vec, lambda);
    let beta_n_lambda = reference_map_1d(&x_col, &y_vec, lambda * n as f32);

    // Sanity: the two modes are materially different (n·λ shrinks ~toward 0).
    assert!(
        beta_lambda.abs() > 2.0 * beta_n_lambda.abs(),
        "test design: λ-mode ({beta_lambda}) should be far larger in magnitude \
         than the n·λ-mode ({beta_n_lambda})"
    );

    // PRIMARY: fitted posterior mean must match the DECLARED-precision MAP.
    assert!(
        (beta_fitted - beta_lambda).abs() < 1e-2,
        "PMAT-864: fitted β ({beta_fitted}) must converge to the λ-MAP \
         ({beta_lambda}), not the over-shrunk n·λ-MAP ({beta_n_lambda}). \
         A match to the n·λ-MAP indicates the data term is divided by n while \
         the prior term is not (gradient/Hessian inconsistency)."
    );

    // GUARD: fitted mean must be clearly distinct from the buggy n·λ-mode.
    assert!(
        (beta_fitted - beta_n_lambda).abs() > 0.5 * (beta_lambda - beta_n_lambda).abs(),
        "PMAT-864: fitted β ({beta_fitted}) is too close to the over-shrunk \
         n·λ-MAP ({beta_n_lambda}) — the 1/n data-term scaling has crept back in."
    );
}

/// PMAT-864 falsifier: replicating every sample (doubling n with the SAME
/// per-sample distribution) must NOT shrink the MAP toward 0. The likelihood
/// term Xᵀ(y − p) doubles, so the MAP at fixed precision λ grows (more data,
/// stronger evidence) — it MUST NOT shrink. Under the bug, the data term was
/// averaged by n, so doubling n halved the effective evidence-to-prior ratio
/// and shrank β toward 0 (the precision-n·λ signature).
#[test]
fn test_pmat864_replicating_samples_does_not_shrink_map() {
    let base_x = [-3.0_f32, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0];
    let base_y = [0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let lambda = 0.2_f32;

    let fit_beta = |reps: usize| -> f32 {
        let mut xv = Vec::new();
        let mut yv = Vec::new();
        for _ in 0..reps {
            xv.extend_from_slice(&base_x);
            yv.extend_from_slice(&base_y);
        }
        let n = xv.len();
        let x = Matrix::from_vec(n, 1, xv).expect("Valid matrix");
        let y = Vector::from_vec(yv);
        let mut model = BayesianLogisticRegression::new(lambda)
            .with_learning_rate(0.3)
            .with_max_iter(200_000)
            .with_tolerance(1e-5);
        model.fit(&x, &y).expect("Fit should converge");
        model
            .coefficients_map
            .as_ref()
            .expect("MAP estimate exists")[0]
    };

    let beta_1x = fit_beta(1);
    let beta_2x = fit_beta(2);

    // More (replicated) data at the same prior precision => MORE evidence =>
    // |β| must grow, never shrink toward 0.
    assert!(
        beta_2x.abs() >= beta_1x.abs() - 1e-3,
        "PMAT-864: doubling the sample count (same distribution) shrank the MAP \
         from {beta_1x} to {beta_2x} — the data term is being averaged by n \
         (precision-n·λ signature)."
    );
    assert!(
        beta_2x.abs() > beta_1x.abs(),
        "PMAT-864: doubling evidence at fixed λ must strictly grow |β| \
         ({beta_1x} -> {beta_2x})."
    );
}
