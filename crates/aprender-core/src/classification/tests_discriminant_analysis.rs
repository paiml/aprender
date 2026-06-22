//! Parity falsifiers for LDA + QDA against pinned scikit-learn reference values.
//!
//! Reference values generated (RED gate, before implementation) with:
//! `uv run --with scikit-learn --with numpy python ...` on the fixed 3-class,
//! 2-feature, 15-row fixture below.
//!
//! - QDA  : `QuadraticDiscriminantAnalysis(reg_param=0.0)` — unbiased per-class cov.
//! - LDA  : `LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)` — biased pooled cov.
//!
//! Contract: `contracts/discriminant-analysis-v1.yaml`
//! (`F-QDA-PARITY-001`, `F-QDA-FIT-PSD-002`, `F-LDA-PARITY-004`).

use super::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};
use crate::primitives::Matrix;

/// Fixed 3-class fixture: 5 samples/class, 2 features (15 rows).
fn fixture() -> (Matrix<f32>, Vec<usize>, Matrix<f32>) {
    #[rustfmt::skip]
    let x_train = Matrix::from_vec(15, 2, vec![
        0.0, 0.0,  1.0, 0.5,  0.5, 1.0,  -0.5, 0.2,  0.3, -0.4,   // class 0 ~ (0,0)
        3.0, 3.0,  3.5, 2.6,  2.6, 3.4,   3.2, 2.8,  2.8,  3.2,   // class 1 ~ (3,3)
        0.0, 3.5,  0.5, 3.0, -0.4, 3.8,   0.3, 3.3,  0.1,  4.0,   // class 2 ~ (0,3.5)
    ])
    .expect("15x2 fixture");
    let y_train = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
    #[rustfmt::skip]
    let x_test = Matrix::from_vec(5, 2, vec![
        0.2, 0.2,   // near class 0
        3.0, 3.0,   // near class 1
        0.0, 3.5,   // near class 2
        1.5, 1.8,   // ambiguous
        1.2, 2.5,   // ambiguous (between 0 and 2)
    ])
    .expect("5x2 test");
    (x_train, y_train, x_test)
}

// ---------------------------------------------------------------------------
// QDA parity (F-QDA-PARITY-001)
// ---------------------------------------------------------------------------

/// F-QDA-PARITY-001: QDA `predict` must equal pinned sklearn labels EXACTLY,
/// and `predict_proba` on the ambiguous rows within 1e-4 of pinned sklearn proba.
#[test]
fn falsify_qda_parity_001_predict_and_proba_match_sklearn() {
    let (x_train, y_train, x_test) = fixture();
    let mut qda = QuadraticDiscriminantAnalysis::new();
    qda.fit(&x_train, &y_train).expect("QDA fit succeeds");

    // Pinned sklearn QuadraticDiscriminantAnalysis(reg_param=0.0).predict
    let expected_pred = vec![0usize, 1, 2, 0, 2];
    let pred = qda.predict(&x_test).expect("QDA predict");
    assert_eq!(
        pred, expected_pred,
        "QDA predict must match sklearn exactly"
    );

    // Pinned sklearn predict_proba on the two ambiguous rows (rows 3 and 4).
    let proba = qda.predict_proba(&x_test).expect("QDA predict_proba");
    // row 3: [0.9872614743, 0.0, 0.0127385257]
    assert!(
        (proba[3][0] - 0.9872614743).abs() < 1e-4,
        "qda proba[3][0]={}",
        proba[3][0]
    );
    assert!(
        (proba[3][1] - 0.0).abs() < 1e-4,
        "qda proba[3][1]={}",
        proba[3][1]
    );
    assert!(
        (proba[3][2] - 0.0127385257).abs() < 1e-4,
        "qda proba[3][2]={}",
        proba[3][2]
    );
    // row 4: [0.0081517123, 0.0, 0.9918482877]
    assert!(
        (proba[4][0] - 0.0081517123).abs() < 1e-4,
        "qda proba[4][0]={}",
        proba[4][0]
    );
    assert!(
        (proba[4][1] - 0.0).abs() < 1e-4,
        "qda proba[4][1]={}",
        proba[4][1]
    );
    assert!(
        (proba[4][2] - 0.9918482877).abs() < 1e-4,
        "qda proba[4][2]={}",
        proba[4][2]
    );

    // proba rows must be valid distributions.
    for row in &proba {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "qda proba row sums to {s}");
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
}

/// F-QDA-FIT-PSD-002: each fitted class covariance is positive-definite
/// (its Cholesky factor exists), so log-likelihood is finite — verified by
/// finite, valid log-posteriors / proba on the fixture.
#[test]
fn falsify_qda_fit_psd_002_per_class_cholesky_finite_loglik() {
    let (x_train, y_train, x_test) = fixture();
    let mut qda = QuadraticDiscriminantAnalysis::new();
    qda.fit(&x_train, &y_train)
        .expect("QDA fit must succeed (each class cov PSD)");
    assert_eq!(qda.classes(), Some(&[0usize, 1, 2][..]));

    let proba = qda.predict_proba(&x_test).expect("QDA predict_proba");
    for row in &proba {
        assert!(
            row.iter().all(|&p| p.is_finite()),
            "non-finite proba implies non-PSD covariance / bad log-det"
        );
    }
}

// ---------------------------------------------------------------------------
// LDA parity (F-LDA-PARITY-004)
// ---------------------------------------------------------------------------

/// F-LDA-PARITY-004: LDA(lsqr) `predict` matches pinned sklearn labels EXACTLY,
/// `coef_`/`intercept_` match sklearn, and `predict_proba` on the ambiguous row
/// is within 1e-4 of pinned sklearn proba.
#[test]
fn falsify_lda_parity_004_predict_coef_proba_match_sklearn() {
    let (x_train, y_train, x_test) = fixture();
    let mut lda = LinearDiscriminantAnalysis::new();
    lda.fit(&x_train, &y_train).expect("LDA fit succeeds");

    // Pinned sklearn LinearDiscriminantAnalysis(solver='lsqr').predict
    let expected_pred = vec![0usize, 1, 2, 2, 2];
    let pred = lda.predict(&x_test).expect("LDA predict");
    assert_eq!(
        pred, expected_pred,
        "LDA predict must match sklearn exactly"
    );

    // Pinned sklearn coef_ (per-class) and intercept_.
    let coef = lda.coef().expect("fitted coef");
    // class 0 coef: [2.1633121636, 2.2146565979]
    assert!(
        (coef[0][0] - 2.1633121636).abs() < 1e-3,
        "lda coef[0][0]={}",
        coef[0][0]
    );
    assert!(
        (coef[0][1] - 2.2146565979).abs() < 1e-3,
        "lda coef[0][1]={}",
        coef[0][1]
    );
    // class 1 coef: [25.1021622589, 25.5792705404]
    assert!(
        (coef[1][0] - 25.1021622589).abs() < 1e-2,
        "lda coef[1][0]={}",
        coef[1][0]
    );
    assert!(
        (coef[1][1] - 25.5792705404).abs() < 1e-2,
        "lda coef[1][1]={}",
        coef[1][1]
    );
    // class 2 coef: [5.1994797097, 25.6156066016]
    assert!(
        (coef[2][0] - 5.1994797097).abs() < 1e-2,
        "lda coef[2][0]={}",
        coef[2][0]
    );
    assert!(
        (coef[2][1] - 25.6156066016).abs() < 1e-2,
        "lda coef[2][1]={}",
        coef[2][1]
    );

    let intercept = lda.intercept().expect("fitted intercept");
    // intercept_: [-1.6677482277, -77.3717831103, -46.4420538929]
    assert!(
        (intercept[0] - (-1.6677482277)).abs() < 1e-2,
        "lda intercept[0]={}",
        intercept[0]
    );
    assert!(
        (intercept[1] - (-77.3717831103)).abs() < 5e-2,
        "lda intercept[1]={}",
        intercept[1]
    );
    assert!(
        (intercept[2] - (-46.4420538929)).abs() < 5e-2,
        "lda intercept[2]={}",
        intercept[2]
    );

    // Pinned sklearn predict_proba on the ambiguous row 3:
    // [0.1016630464, 0.2175022585, 0.6808346951]
    let proba = lda.predict_proba(&x_test).expect("LDA predict_proba");
    assert!(
        (proba[3][0] - 0.1016630464).abs() < 1e-4,
        "lda proba[3][0]={}",
        proba[3][0]
    );
    assert!(
        (proba[3][1] - 0.2175022585).abs() < 1e-4,
        "lda proba[3][1]={}",
        proba[3][1]
    );
    assert!(
        (proba[3][2] - 0.6808346951).abs() < 1e-4,
        "lda proba[3][2]={}",
        proba[3][2]
    );

    for row in &proba {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "lda proba row sums to {s}");
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
}

/// LDA `decision_function` matches pinned sklearn decision values on the
/// well-separated rows (sanity that the linear discriminants are correct).
#[test]
fn falsify_lda_decision_function_matches_sklearn() {
    let (x_train, y_train, x_test) = fixture();
    let mut lda = LinearDiscriminantAnalysis::new();
    lda.fit(&x_train, &y_train).expect("LDA fit");
    let dec = lda.decision_function(&x_test).expect("decision_function");
    // Pinned sklearn decision_function row 0: [-0.79215448, -67.23549655, -40.27903663]
    assert!(
        (dec[0][0] - (-0.79215448)).abs() < 1e-2,
        "dec[0][0]={}",
        dec[0][0]
    );
    assert!(
        (dec[0][1] - (-67.23549655)).abs() < 1e-1,
        "dec[0][1]={}",
        dec[0][1]
    );
    assert!(
        (dec[0][2] - (-40.27903663)).abs() < 1e-1,
        "dec[0][2]={}",
        dec[0][2]
    );
    // argmax of decision == predict
    let pred = lda.predict(&x_test).expect("predict");
    for (row, &p) in dec.iter().zip(pred.iter()) {
        let am = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(am, p, "decision argmax must equal predicted class index");
    }
}

/// Both estimators perfectly classify their own well-separated training clusters.
#[test]
fn falsify_da_separable_clusters_perfect_recall() {
    let (x_train, y_train, _x_test) = fixture();
    // Well-separated cluster centers (one per class).
    let centers = Matrix::from_vec(3, 2, vec![0.0, 0.0, 3.0, 3.0, 0.0, 3.5]).expect("3x2");
    let expected = vec![0usize, 1, 2];

    let mut qda = QuadraticDiscriminantAnalysis::new();
    qda.fit(&x_train, &y_train).expect("QDA fit");
    assert_eq!(qda.predict(&centers).expect("qda predict"), expected);

    let mut lda = LinearDiscriminantAnalysis::new();
    lda.fit(&x_train, &y_train).expect("LDA fit");
    assert_eq!(lda.predict(&centers).expect("lda predict"), expected);
}

/// Fit-time guards: empty data, mismatched lengths, and < 2 classes error.
#[test]
fn falsify_da_fit_guards_reject_bad_input() {
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).expect("3x2");

    // Single class -> error for both.
    let mut qda = QuadraticDiscriminantAnalysis::new();
    assert!(
        qda.fit(&x, &[0, 0, 0]).is_err(),
        "QDA must reject < 2 classes"
    );
    let mut lda = LinearDiscriminantAnalysis::new();
    assert!(
        lda.fit(&x, &[0, 0, 0]).is_err(),
        "LDA must reject < 2 classes"
    );

    // Length mismatch.
    let mut qda2 = QuadraticDiscriminantAnalysis::new();
    assert!(
        qda2.fit(&x, &[0, 1]).is_err(),
        "QDA must reject len mismatch"
    );
}
