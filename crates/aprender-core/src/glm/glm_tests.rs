pub(crate) use super::*;

/// Test: Family canonical links
#[test]
fn test_family_canonical_links() {
    assert_eq!(Family::Poisson.canonical_link(), Link::Log);
    assert_eq!(Family::Gamma.canonical_link(), Link::Inverse);
    assert_eq!(Family::Binomial.canonical_link(), Link::Logit);
}

/// Test: Link functions and inverses
#[test]
fn test_link_functions() {
    // Log link
    let link = Link::Log;
    let mu = 5.0;
    let eta = link.link(mu);
    assert!((eta - mu.ln()).abs() < 1e-6);
    assert!((link.inverse_link(eta) - mu).abs() < 1e-6);

    // Inverse link
    let link = Link::Inverse;
    let mu = 2.0;
    let eta = link.link(mu);
    assert!((eta - 1.0 / mu).abs() < 1e-6);
    assert!((link.inverse_link(eta) - mu).abs() < 1e-6);

    // Logit link
    let link = Link::Logit;
    let mu = 0.7;
    let eta = link.link(mu);
    assert!((link.inverse_link(eta) - mu).abs() < 1e-6);

    // Identity link
    let link = Link::Identity;
    let mu = 3.0;
    assert_eq!(link.link(mu), mu);
    assert_eq!(link.inverse_link(mu), mu);
}

/// Test: Negative Binomial regression on overdispersed count data
///
/// Negative Binomial handles count data where variance >> mean (overdispersion).
/// This is the proper solution for overdispersed counts, as documented in notes-poisson.md.
/// V(μ) = μ + α*μ² where α is the dispersion parameter.
#[test]
fn test_negative_binomial_regression() {
    // Gentle linear count data with slight overdispersion
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let mut model = GLM::new(Family::NegativeBinomial)
        .with_dispersion(0.1)
        .with_max_iter(5000); // More iterations for damped convergence
    let result = model.fit(&x, &y);

    assert!(
        result.is_ok(),
        "Negative Binomial GLM should fit, error: {:?}",
        result.err()
    );
    assert!(model.coefficients().is_some());
    assert!(model.intercept().is_some());

    // Verify predictions work
    let predictions = model.predict(&x).expect("Predictions should succeed");
    assert_eq!(predictions.len(), y.len());
}

/// Test: Negative Binomial with low dispersion (approaches Poisson)
#[test]
fn test_negative_binomial_low_dispersion() {
    // Low overdispersion data
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut model = GLM::new(Family::NegativeBinomial)
        .with_dispersion(0.01) // Very low dispersion, close to Poisson
        .with_max_iter(5000);
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Low dispersion NB should converge");
    assert!(model.coefficients().is_some());
}

/// Test: Negative Binomial validation rejects negative counts
#[test]
fn test_negative_binomial_validation() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![3.0, -1.0, 5.0]); // Negative count

    let mut model = GLM::new(Family::NegativeBinomial);
    let result = model.fit(&x, &y);

    assert!(
        result.is_err(),
        "Negative Binomial should reject negative counts"
    );
    assert!(result
        .expect_err("Should return error for negative counts")
        .to_string()
        .contains("Negative Binomial requires non-negative counts"));
}

/// Test: Gamma regression on positive continuous data
#[test]
fn test_gamma_regression() {
    // Simulated positive continuous data
    let x =
        Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7]);

    let mut model = GLM::new(Family::Gamma); // Use default max_iter (1000)
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Gamma GLM should fit");
    assert!(model.coefficients().is_some());

    // Predictions should be positive
    let predictions = model.predict(&x).expect("Predictions should succeed");
    for &pred in predictions.as_slice() {
        assert!(pred > 0.0, "Gamma predictions should be positive");
    }
}

/// Test: Binomial regression on proportions
#[test]
fn test_binomial_regression() {
    // Simulated binary/proportion data
    let x = Matrix::from_vec(8, 1, vec![-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        .expect("Valid matrix");
    let y = Vector::from_vec(vec![0.1, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.9]);

    let mut model = GLM::new(Family::Binomial); // Use default max_iter (1000)
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Binomial GLM should fit");
    assert!(model.coefficients().is_some());

    // Predictions should be in [0, 1]
    let predictions = model.predict(&x).expect("Predictions should succeed");
    for &pred in predictions.as_slice() {
        assert!(
            (0.0..=1.0).contains(&pred),
            "Binomial predictions should be in [0,1], got {pred}"
        );
    }
}

/// Test: Invalid response for Poisson (negative values)
#[test]
fn test_poisson_invalid_response() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![1.0, -2.0, 3.0]); // Negative value!

    let mut model = GLM::new(Family::Poisson);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Invalid response for Gamma (non-positive values)
#[test]
fn test_gamma_invalid_response() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![1.0, 0.0, 3.0]); // Zero value!

    let mut model = GLM::new(Family::Gamma);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Invalid response for Binomial (out of [0,1])
#[test]
fn test_binomial_invalid_response() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.5, 1.2, 0.3]); // 1.2 out of range!

    let mut model = GLM::new(Family::Binomial);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Dimension mismatch in fit
#[test]
fn test_fit_dimension_mismatch() {
    let x =
        Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![1.0, 2.0]); // Wrong size!

    let mut model = GLM::new(Family::Poisson);
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
}

/// Test: Predict before fit
#[test]
fn test_predict_not_fitted() {
    let model = GLM::new(Family::Poisson);
    let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix");

    let result = model.predict(&x_test);
    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::Other(_)));
}

/// Test: Predict with dimension mismatch
#[test]
fn test_predict_dimension_mismatch() {
    // Simpler data for 2-feature model
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0],
    )
    .expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 3.0, 3.0, 4.0, 5.0, 6.0]);

    let mut model = GLM::new(Family::Poisson);
    model.fit(&x, &y).expect("Fit succeeds");

    // Try to predict with wrong number of features
    let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid test matrix");
    let result = model.predict(&x_test);

    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
}

/// Test: Custom link function
#[test]
fn test_custom_link() {
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 1.8, 1.6, 1.4, 1.2, 1.0]);

    // Gamma with log link (instead of canonical inverse)
    let mut model = GLM::new(Family::Gamma)
        .with_link(Link::Log)
        .with_max_iter(5000); // More iterations for non-canonical link

    let result = model.fit(&x, &y);
    assert!(
        result.is_ok(),
        "Custom link should work, error: {:?}",
        result.err()
    );
}

/// Test: Builder pattern
#[test]
fn test_builder_pattern() {
    let model = GLM::new(Family::Poisson)
        .with_max_iter(500)
        .with_tolerance(1e-8)
        .with_link(Link::Log);

    assert!(model.coefficients().is_none());
    assert!(model.intercept().is_none());
}
