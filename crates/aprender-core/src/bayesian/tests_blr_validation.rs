use super::*;

#[test]
fn test_with_prior_zero_precision() {
    let result = BayesianLinearRegression::with_prior(
        2,
        vec![1.0, 2.0],
        0.0, // Invalid: must be > 0
        1.0,
        1.0,
    );
    assert!(result.is_err());
}

#[test]
fn test_with_prior_zero_alpha() {
    let result = BayesianLinearRegression::with_prior(
        2,
        vec![1.0, 2.0],
        1.0,
        0.0, // Invalid alpha
        1.0,
    );
    assert!(result.is_err());
}

#[test]
fn test_with_prior_zero_beta() {
    let result = BayesianLinearRegression::with_prior(
        2,
        vec![1.0, 2.0],
        1.0,
        1.0,
        0.0, // Invalid beta
    );
    assert!(result.is_err());
}

#[test]
fn test_bic_not_fitted() {
    use crate::primitives::{Matrix, Vector};

    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let model = BayesianLinearRegression::new(1);
    let result = model.bic(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_aic_not_fitted() {
    use crate::primitives::{Matrix, Vector};

    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let model = BayesianLinearRegression::new(1);
    let result = model.aic(&x, &y);

    assert!(result.is_err());
}
