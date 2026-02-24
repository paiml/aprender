// =========================================================================
// FALSIFY-SVM: svm-v1.yaml contract (aprender LinearSVM)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest SVM tests but zero inline FALSIFY-SVM-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from svm-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: SVM was "obviously correct" (hinge loss + SGD)
//
// References:
//   - provable-contracts/contracts/svm-v1.yaml
//   - Cortes & Vapnik (1995) "Support-Vector Networks"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-SVM-001: Binary prediction — predict(x) ∈ {0, 1}
#[test]
fn falsify_svm_001_binary_prediction() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.5, 0.5, 1.0, 1.0,
        5.0, 5.0, 5.5, 5.5, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut svm = LinearSVM::new();
    svm.fit(&x, &y).expect("fit");

    let preds = svm.predict(&x).expect("predict");
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED SVM-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-SVM-002: Deterministic (SVM uses deterministic subgradient descent)
#[test]
fn falsify_svm_002_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut svm1 = LinearSVM::new();
    svm1.fit(&x, &y).expect("fit 1");
    let p1 = svm1.predict(&x).expect("predict 1");

    let mut svm2 = LinearSVM::new();
    svm2.fit(&x, &y).expect("fit 2");
    let p2 = svm2.predict(&x).expect("predict 2");

    assert_eq!(p1, p2, "FALSIFIED SVM-002: same params produce different predictions");
}

/// FALSIFY-SVM-003: Prediction count matches input
#[test]
fn falsify_svm_003_prediction_count() {
    let x = Matrix::from_vec(4, 2, vec![
        0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut svm = LinearSVM::new();
    svm.fit(&x, &y).expect("fit");

    let x_test = Matrix::from_vec(5, 2, vec![
        0.5, 0.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.5, 5.5,
    ]).expect("valid");
    let preds = svm.predict(&x_test).expect("predict");
    assert_eq!(preds.len(), 5, "FALSIFIED SVM-003: {} predictions for 5 inputs", preds.len());
}

/// FALSIFY-SVM-004: Well-separated data classified correctly
#[test]
fn falsify_svm_004_separable_data() {
    let x = Matrix::from_vec(6, 2, vec![
        0.0, 0.0, 0.1, 0.1, 0.2, 0.2,
        100.0, 100.0, 100.1, 100.1, 100.2, 100.2,
    ]).expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut svm = LinearSVM::new().with_max_iter(1000);
    svm.fit(&x, &y).expect("fit");

    let preds = svm.predict(&x).expect("predict");
    assert_eq!(
        preds, y,
        "FALSIFIED SVM-004: SVM cannot classify well-separated data"
    );
}
