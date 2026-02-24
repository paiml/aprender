// =========================================================================
// FALSIFY-BLR: bayesian-v1.yaml contract (aprender BayesianLinearRegression)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-BLR-* tests
//   Why 2: Bayesian tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from bayesian-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: BLR was "obviously correct" (conjugate normal-inverse-gamma)
//
// References:
//   - provable-contracts/contracts/bayesian-v1.yaml
//   - Bishop (2006) "Pattern Recognition and Machine Learning"
// =========================================================================

use super::*;
use crate::primitives::{Matrix, Vector};

/// FALSIFY-BLR-001: Predictions are finite
#[test]
fn falsify_blr_001_finite_predictions() {
    let x = Matrix::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0,
    ]).expect("valid");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 2.5, 4.5]);

    let mut blr = BayesianLinearRegression::new(2);
    blr.fit(&x, &y).expect("fit");

    let preds = blr.predict(&x).expect("predict");
    for i in 0..preds.len() {
        assert!(
            preds[i].is_finite(),
            "FALSIFIED BLR-001: prediction[{i}] = {} is not finite", preds[i]
        );
    }
}

/// FALSIFY-BLR-002: Prediction count matches input count
#[test]
fn falsify_blr_002_prediction_count() {
    let x = Matrix::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0,
    ]).expect("valid");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 2.5, 4.5]);

    let mut blr = BayesianLinearRegression::new(2);
    blr.fit(&x, &y).expect("fit");

    let preds = blr.predict(&x).expect("predict");
    assert_eq!(
        preds.len(), 5,
        "FALSIFIED BLR-002: {} predictions for 5 inputs", preds.len()
    );
}

/// FALSIFY-BLR-003: Deterministic predictions
#[test]
fn falsify_blr_003_deterministic() {
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
    let y = Vector::from_slice(&[2.5, 4.8, 7.1, 9.5]);

    let mut blr = BayesianLinearRegression::new(1);
    blr.fit(&x, &y).expect("fit");

    let p1 = blr.predict(&x).expect("predict 1");
    let p2 = blr.predict(&x).expect("predict 2");
    for i in 0..p1.len() {
        assert_eq!(
            p1[i], p2[i],
            "FALSIFIED BLR-003: prediction differs at index {i}"
        );
    }
}

/// FALSIFY-BLR-004: Posterior mean exists after fit
#[test]
fn falsify_blr_004_posterior_exists() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut blr = BayesianLinearRegression::new(1);
    blr.fit(&x, &y).expect("fit");

    let posterior = blr.posterior_mean();
    assert!(
        posterior.is_some(),
        "FALSIFIED BLR-004: posterior_mean is None after fit"
    );
    let mean = posterior.expect("checked above");
    for (i, &v) in mean.iter().enumerate() {
        assert!(
            v.is_finite(),
            "FALSIFIED BLR-004: posterior_mean[{i}] = {v} is not finite"
        );
    }
}
