// =========================================================================
// FALSIFY-KNN: knn tests (part of linear-models-v1.yaml / classification suite)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-KNN-* tests
//   Why 2: KNN tests only in tests/contracts/, not near implementation
//   Why 3: no mapping to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: KNN was "obviously correct" (nearest neighbor lookup)
//
// References:
//   - Cover & Hart (1967) "Nearest Neighbor Pattern Classification"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-KNN-001: Predictions in training label set
#[test]
fn falsify_knn_001_predictions_in_label_range() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.5, 0.5, 1.0, 0.0,
        5.0, 5.0, 5.5, 5.5, 6.0, 5.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let preds = knn.predict(&x).expect("predict");
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED KNN-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-KNN-002: Prediction count matches input count
#[test]
fn falsify_knn_002_prediction_count() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.5, 0.5, 1.0, 0.0,
        5.0, 5.0, 5.5, 5.5, 6.0, 5.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let x_test = Matrix::from_vec(3, 2, vec![0.2, 0.2, 3.0, 3.0, 5.8, 5.8]).expect("valid");
    let preds = knn.predict(&x_test).expect("predict");
    assert_eq!(preds.len(), 3, "FALSIFIED KNN-002: {} predictions for 3 inputs", preds.len());
}

/// FALSIFY-KNN-003: Well-separated clusters classified correctly
#[test]
fn falsify_knn_003_separable_data() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.1, 0.1, 0.2, 0.2,
        100.0, 100.0, 100.1, 100.1, 100.2, 100.2,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y).expect("fit");

    let preds = knn.predict(&x).expect("predict");
    assert_eq!(
        preds, y,
        "FALSIFIED KNN-003: KNN cannot classify well-separated clusters"
    );
}

/// FALSIFY-KNN-004: Deterministic predictions
#[test]
fn falsify_knn_004_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut knn = KNearestNeighbors::new(1);
    knn.fit(&x, &y).expect("fit");

    let p1 = knn.predict(&x).expect("predict 1");
    let p2 = knn.predict(&x).expect("predict 2");
    assert_eq!(p1, p2, "FALSIFIED KNN-004: predictions differ on same input");
}
