// =========================================================================
// FALSIFY-SVC-RBF: svc-rbf-v1.yaml contract (aprender SVCRbf)
//
// Net-new RBF-kernel SVC (PMAT-930). The headline obligation is
// OBLIG-RBF-SVC-SKLEARN-PARITY: on a fixed non-linearly-separable dataset,
// apr's RBF-SVC predictions agree with the PINNED, offline-computed
// scikit-learn `SVC(kernel='rbf')` predictions on a held-out grid within
// tolerance (>= 90% label agreement).
//
// References:
//   - contracts/svc-rbf-v1.yaml
//   - Platt (1998) "Sequential Minimal Optimization"
//   - Cortes & Vapnik (1995) "Support-Vector Networks"
//   - libsvm / scikit-learn SVC dual formulation
// =========================================================================

use super::SVCRbf;
use crate::classification::svc_rbf_sklearn_fixture as fix;
use crate::primitives::Matrix;

/// Builds the pinned XOR-structured training matrix and labels.
fn train_data() -> (Matrix<f32>, Vec<usize>) {
    let x = Matrix::from_vec(fix::N_TRAIN, 2, fix::TRAIN_X.to_vec()).expect("train matrix");
    let y = fix::TRAIN_Y.to_vec();
    (x, y)
}

/// Builds the pinned held-out grid matrix.
fn grid_data() -> Matrix<f32> {
    Matrix::from_vec(fix::N_GRID, 2, fix::GRID_X.to_vec()).expect("grid matrix")
}

/// OBLIG-RBF-SVC-SKLEARN-PARITY (FALSIFY-SVC-RBF-001):
/// On the held-out grid, apr RBF-SVC predictions match the pinned
/// sklearn `SVC(kernel='rbf', gamma=0.5, C=1.0)` predictions in >= 90% of
/// grid points. RED on a linear/degenerate solver (it cannot recover the XOR
/// decision boundary), GREEN on the SMO dual RBF solver.
#[test]
fn falsify_svc_rbf_001_sklearn_parity_grid() {
    let (x, y) = train_data();
    let grid = grid_data();

    let mut svc = SVCRbf::new()
        .with_gamma(fix::GAMMA)
        .with_c(fix::C)
        .with_max_iter(1000);
    svc.fit(&x, &y).expect("fit");

    let preds = svc.predict(&grid).expect("predict grid");
    assert_eq!(
        preds.len(),
        fix::N_GRID,
        "prediction count must match grid size"
    );

    let agree = preds
        .iter()
        .zip(fix::SKLEARN_GRID_PRED.iter())
        .filter(|(a, b)| a == b)
        .count();
    let frac = agree as f32 / fix::N_GRID as f32;

    assert!(
        frac >= 0.90,
        "FALSIFIED SVC-RBF-001: only {agree}/{} grid points ({frac:.3}) agree with \
         pinned sklearn SVC(rbf) — RBF decision boundary diverges from reference",
        fix::N_GRID
    );
}

/// FALSIFY-SVC-RBF-002: the RBF kernel recovers the XOR boundary on the
/// training set (train accuracy = 1.0), which a LINEAR SVM provably cannot.
#[test]
fn falsify_svc_rbf_002_xor_train_accuracy() {
    let (x, y) = train_data();
    let mut svc = SVCRbf::new().with_gamma(fix::GAMMA).with_c(fix::C);
    svc.fit(&x, &y).expect("fit");
    let acc = svc.score(&x, &y).expect("score");
    assert!(
        acc >= 0.95,
        "FALSIFIED SVC-RBF-002: XOR train accuracy {acc:.3} < 0.95 — RBF kernel not \
         separating non-linear classes"
    );
}

/// FALSIFY-SVC-RBF-003: tiny canonical XOR (4 points) is classified exactly.
/// This is the minimal non-linearly-separable problem; a correct RBF SVC must
/// nail all 4.
#[test]
fn falsify_svc_rbf_003_canonical_xor() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).expect("xor");
    let y = vec![0_usize, 1, 1, 0];
    let mut svc = SVCRbf::new().with_gamma(1.0).with_c(10.0);
    svc.fit(&x, &y).expect("fit");
    let preds = svc.predict(&x).expect("predict");
    assert_eq!(
        preds, y,
        "FALSIFIED SVC-RBF-003: canonical XOR not classified exactly"
    );
}

/// FALSIFY-SVC-RBF-004: predictions are deterministic (SMO is deterministic).
#[test]
fn falsify_svc_rbf_004_deterministic() {
    let (x, y) = train_data();
    let grid = grid_data();

    let mut a = SVCRbf::new().with_gamma(fix::GAMMA).with_c(fix::C);
    a.fit(&x, &y).expect("fit a");
    let pa = a.predict(&grid).expect("predict a");

    let mut b = SVCRbf::new().with_gamma(fix::GAMMA).with_c(fix::C);
    b.fit(&x, &y).expect("fit b");
    let pb = b.predict(&grid).expect("predict b");

    assert_eq!(
        pa, pb,
        "FALSIFIED SVC-RBF-004: same params produced different predictions"
    );
}

/// FALSIFY-SVC-RBF-005: predicted labels are a subset of the training labels
/// (binary {0,1} here), and the decision function sign matches predict().
#[test]
fn falsify_svc_rbf_005_label_subset_and_sign() {
    let (x, y) = train_data();
    let grid = grid_data();
    let mut svc = SVCRbf::new().with_gamma(fix::GAMMA).with_c(fix::C);
    svc.fit(&x, &y).expect("fit");

    let preds = svc.predict(&grid).expect("predict");
    for &p in &preds {
        assert!(
            p == 0 || p == 1,
            "FALSIFIED SVC-RBF-005: predicted label {p} not in training label set"
        );
    }

    let df = svc.decision_function(&grid).expect("decision_function");
    for (i, &p) in preds.iter().enumerate() {
        let expected = usize::from(df[i] > 0.0);
        assert_eq!(
            p, expected,
            "FALSIFIED SVC-RBF-005: predict()/decision_function() sign disagree at {i}"
        );
    }
}

/// FALSIFY-SVC-RBF-006: error handling — wrong class count and shape mismatch
/// are rejected, and predict-before-fit errors rather than panicking.
#[test]
fn falsify_svc_rbf_006_error_handling() {
    // Single-class data is rejected (needs exactly 2 classes).
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).expect("x");
    let mut svc = SVCRbf::new();
    assert!(
        svc.fit(&x, &[0, 0, 0]).is_err(),
        "FALSIFIED SVC-RBF-006: single-class fit should error"
    );

    // X/y length mismatch is rejected.
    assert!(
        svc.fit(&x, &[0, 1]).is_err(),
        "FALSIFIED SVC-RBF-006: X/y length mismatch should error"
    );

    // Predict before fit errors (does not panic).
    let unfitted = SVCRbf::new();
    assert!(
        unfitted.predict(&x).is_err(),
        "FALSIFIED SVC-RBF-006: predict before fit should error"
    );
}

/// FALSIFY-SVC-RBF-007: Estimator-trait round-trip (labels via f32) gives the
/// same parity on the grid, so generic cross_validate / grid_search work.
#[test]
fn falsify_svc_rbf_007_estimator_trait_parity() {
    use crate::primitives::Vector;
    use crate::traits::Estimator;

    let (x, _) = train_data();
    let y_f = Vector::from_vec(fix::TRAIN_Y.iter().map(|&l| l as f32).collect());
    let grid = grid_data();

    let mut svc = SVCRbf::new().with_gamma(fix::GAMMA).with_c(fix::C);
    Estimator::fit(&mut svc, &x, &y_f).expect("fit via trait");
    let preds = Estimator::predict(&svc, &grid);

    let agree = preds
        .as_slice()
        .iter()
        .zip(fix::SKLEARN_GRID_PRED.iter())
        .filter(|(&a, &b)| a.round() as usize == b)
        .count();
    let frac = agree as f32 / fix::N_GRID as f32;
    assert!(
        frac >= 0.90,
        "FALSIFIED SVC-RBF-007: Estimator-trait grid parity {frac:.3} < 0.90"
    );
}
