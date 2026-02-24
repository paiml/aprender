// =========================================================================
// FALSIFY-LOGREG: linear-models-v1.yaml contract (aprender LogisticRegression)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LOGREG-* tests
//   Why 2: logistic regression tests lack contract-mapped naming
//   Why 3: no mapping from linear-models-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: LogReg was "obviously correct" (sigmoid + gradient descent)
//
// References:
//   - provable-contracts/contracts/linear-models-v1.yaml
//   - Bishop (2006) "Pattern Recognition and Machine Learning" ch. 4.3
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-LOGREG-001: Predictions in {0, 1}
#[test]
fn falsify_logreg_001_binary_predictions() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.5, 0.5, 1.0, 0.0,
        5.0, 5.0, 5.5, 5.5, 6.0, 5.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED LOGREG-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-LOGREG-002: Prediction count matches input count
#[test]
fn falsify_logreg_002_prediction_count() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    assert_eq!(preds.len(), 4, "FALSIFIED LOGREG-002: {} predictions for 4 inputs", preds.len());
}

/// FALSIFY-LOGREG-003: Probabilities in [0, 1]
#[test]
fn falsify_logreg_003_probabilities_bounded() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let probas = lr.predict_proba(&x);
    for i in 0..probas.len() {
        assert!(
            (0.0..=1.0).contains(&probas[i]),
            "FALSIFIED LOGREG-003: proba[{i}] = {} not in [0, 1]", probas[i]
        );
    }
}

/// FALSIFY-LOGREG-004: Deterministic predictions
#[test]
fn falsify_logreg_004_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let p1 = lr.predict(&x);
    let p2 = lr.predict(&x);
    assert_eq!(p1, p2, "FALSIFIED LOGREG-004: predictions differ on same input");
}
