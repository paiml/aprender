// =========================================================================
// FALSIFY-LR: linear-models-v1.yaml contract (aprender LinearRegression)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LR-* tests for LinearRegression
//   Why 2: linear model tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from linear-models-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: OLS was "obviously correct" (normal equations)
//
// References:
//   - provable-contracts/contracts/linear-models-v1.yaml
//   - Hastie, Tibshirani, Friedman (2009) "Elements of Statistical Learning"
// =========================================================================

use super::*;
use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;

/// FALSIFY-LR-001: Perfect fit on exact linear data (y = 2x + 1)
#[test]
fn falsify_lr_001_perfect_linear_fit() {
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    for i in 0..4 {
        assert!(
            (preds[i] - y[i]).abs() < 1e-4,
            "FALSIFIED LR-001: pred[{i}]={} != y[{i}]={}", preds[i], y[i]
        );
    }
}

/// FALSIFY-LR-002: R² = 1.0 on perfect linear data
#[test]
fn falsify_lr_002_r2_perfect() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).expect("fit");

    let r2 = lr.score(&x, &y);
    assert!(
        (r2 - 1.0).abs() < 1e-4,
        "FALSIFIED LR-002: R²={r2}, expected ≈ 1.0 for perfect linear data"
    );
}

/// FALSIFY-LR-003: Prediction count matches input count
#[test]
fn falsify_lr_003_prediction_count() {
    let x = Matrix::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0,
    ]).expect("valid");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 2.5, 4.5]);

    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).expect("fit");

    let x_test = Matrix::from_vec(3, 2, vec![0.5, 0.5, 1.5, 1.5, 2.0, 2.0]).expect("valid");
    let preds = lr.predict(&x_test);
    assert_eq!(
        preds.len(), 3,
        "FALSIFIED LR-003: {} predictions for 3 inputs", preds.len()
    );
}

/// FALSIFY-LR-004: Deterministic predictions
#[test]
fn falsify_lr_004_deterministic() {
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
    let y = Vector::from_slice(&[2.5, 4.8, 7.1, 9.5]);

    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).expect("fit");

    let p1 = lr.predict(&x);
    let p2 = lr.predict(&x);
    for i in 0..4 {
        assert_eq!(
            p1[i], p2[i],
            "FALSIFIED LR-004: prediction differs on same input at index {i}"
        );
    }
}
