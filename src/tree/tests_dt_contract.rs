// =========================================================================
// FALSIFY-DT: decision-tree-v1.yaml contract (aprender DecisionTreeClassifier)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest DT tests but zero inline FALSIFY-DT-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from decision-tree-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: CART was "obviously correct" (textbook algorithm)
//
// References:
//   - provable-contracts/contracts/decision-tree-v1.yaml
//   - Breiman et al. (1984) "Classification and Regression Trees"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-DT-001: Predictions in label range — predict(x) ∈ training labels
#[test]
fn falsify_dt_001_predictions_in_label_range() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
    ]).expect("valid matrix");
    let y = vec![0_usize, 0, 1, 1, 2, 2];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x, &y).expect("fit succeeds");

    let preds = dt.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 2,
            "FALSIFIED DT-001: prediction[{i}] = {p}, not in [0, 2]"
        );
    }
}

/// FALSIFY-DT-002: Deterministic — same input produces same output
#[test]
fn falsify_dt_002_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
    ]).expect("valid matrix");
    let y = vec![0_usize, 0, 1, 1];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x, &y).expect("fit");

    let p1 = dt.predict(&x);
    let p2 = dt.predict(&x);
    assert_eq!(
        p1, p2,
        "FALSIFIED DT-002: predictions differ on same input"
    );
}

/// FALSIFY-DT-003: Perfect fit on separable data
///
/// A decision tree should perfectly classify linearly separable data.
#[test]
fn falsify_dt_003_perfect_separable() {
    let x = Matrix::from_vec(4, 1, vec![0.0, 1.0, 10.0, 11.0]).expect("valid matrix");
    let y = vec![0_usize, 0, 1, 1];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x, &y).expect("fit");

    let preds = dt.predict(&x);
    assert_eq!(
        preds, y,
        "FALSIFIED DT-003: tree cannot perfectly fit separable data"
    );
}

/// FALSIFY-DT-004: Prediction count matches input count
#[test]
fn falsify_dt_004_prediction_count() {
    let x_train = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
    ]).expect("valid");
    let y_train = vec![0_usize, 0, 1, 1];

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&x_train, &y_train).expect("fit");

    let x_test = Matrix::from_vec(3, 2, vec![0.5, 0.5, 1.5, 1.5, 2.5, 2.5]).expect("valid");
    let preds = dt.predict(&x_test);
    assert_eq!(
        preds.len(), 3,
        "FALSIFIED DT-004: {} predictions for 3 inputs", preds.len()
    );
}
