// =========================================================================
// FALSIFY-NB: naive-bayes-v1.yaml contract (aprender GaussianNB)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest NB tests but zero inline FALSIFY-NB-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from naive-bayes-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: NB was "obviously correct" (Bayes theorem + independence)
//
// References:
//   - provable-contracts/contracts/naive-bayes-v1.yaml
//   - Murphy (2012) "Machine Learning: A Probabilistic Perspective"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-NB-001: Predictions in training label set
#[test]
fn falsify_nb_001_predictions_in_label_range() {
    let x = Matrix::from_vec(6, 2, vec![
        1.0, 2.0, 1.5, 2.5, 2.0, 3.0,
        5.0, 6.0, 5.5, 6.5, 6.0, 7.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut nb = GaussianNB::new();
    nb.fit(&x, &y).expect("fit");

    let preds = nb.predict(&x).expect("predict");
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED NB-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-NB-002: Deterministic predictions
#[test]
fn falsify_nb_002_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut nb = GaussianNB::new();
    nb.fit(&x, &y).expect("fit");

    let p1 = nb.predict(&x).expect("predict 1");
    let p2 = nb.predict(&x).expect("predict 2");
    assert_eq!(p1, p2, "FALSIFIED NB-002: predictions differ on same input");
}

/// FALSIFY-NB-003: Prediction count matches input count
#[test]
fn falsify_nb_003_prediction_count() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut nb = GaussianNB::new();
    nb.fit(&x, &y).expect("fit");

    let x_test = Matrix::from_vec(3, 2, vec![0.5, 0.5, 3.0, 3.0, 5.5, 5.5]).expect("valid");
    let preds = nb.predict(&x_test).expect("predict");
    assert_eq!(preds.len(), 3, "FALSIFIED NB-003: {} predictions for 3 inputs", preds.len());
}

/// FALSIFY-NB-004: Well-separated clusters classified correctly
#[test]
fn falsify_nb_004_separable_data() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.1, 0.1, 0.2, 0.2,
        100.0, 100.0, 100.1, 100.1, 100.2, 100.2,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut nb = GaussianNB::new();
    nb.fit(&x, &y).expect("fit");

    let preds = nb.predict(&x).expect("predict");
    assert_eq!(
        preds, y,
        "FALSIFIED NB-004: NB cannot classify well-separated clusters"
    );
}
