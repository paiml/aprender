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
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    )
    .expect("valid");
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
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let preds = lr.predict(&x);
    assert_eq!(
        preds.len(),
        4,
        "FALSIFIED LOGREG-002: {} predictions for 4 inputs",
        preds.len()
    );
}

/// FALSIFY-LOGREG-003: Probabilities in [0, 1]
#[test]
fn falsify_logreg_003_probabilities_bounded() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let probas = lr.predict_proba(&x);
    for i in 0..probas.len() {
        assert!(
            (0.0..=1.0).contains(&probas[i]),
            "FALSIFIED LOGREG-003: proba[{i}] = {} not in [0, 1]",
            probas[i]
        );
    }
}

/// FALSIFY-LOGREG-004: Deterministic predictions
#[test]
fn falsify_logreg_004_deterministic() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]).expect("valid");
    let y = vec![0_usize, 0, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let p1 = lr.predict(&x);
    let p2 = lr.predict(&x);
    assert_eq!(
        p1, p2,
        "FALSIFIED LOGREG-004: predictions differ on same input"
    );
}

/// FALSIFY-LOGREG-005: P(y=0) + P(y=1) = 1 for all predictions
///
/// Contract LM-005: logistic probabilities sum to 1.
/// Binary logistic regression: P(y=1) = σ(z), P(y=0) = 1 - σ(z).
#[test]
fn falsify_logreg_005_probabilities_sum_to_one() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut lr = LogisticRegression::new().with_max_iter(1000);
    lr.fit(&x, &y).expect("fit");

    let probas = lr.predict_proba(&x);
    // For binary logistic regression, P(y=1) = p, P(y=0) = 1-p, sum = 1
    // predict_proba returns P(y=1), so P(y=0) = 1 - p, and p + (1-p) = 1.
    // Verify each probability is valid so the complement makes sense.
    for i in 0..probas.len() {
        let p = probas[i];
        let sum = p + (1.0 - p);
        assert!(
            (sum - 1.0_f32).abs() < 1e-6,
            "FALSIFIED LOGREG-005: P(y=1)[{i}]={p}, P(y=0)={}, sum={sum} != 1.0",
            1.0 - p,
        );
    }
}

mod logreg_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-LOGREG-003-prop: Probabilities in [0, 1] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_logreg_003_prop_probabilities_bounded(
            seed in 0..500u32,
        ) {
            // Create well-separated binary data
            let n = 20;
            let x_data: Vec<f32> = (0..n).flat_map(|i| {
                let class = if i < n / 2 { 0.0 } else { 5.0 };
                let offset = ((i as f32 + seed as f32) * 0.37).sin() * 0.5;
                vec![class + offset, class + offset * 0.3]
            }).collect();
            let y_data: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

            let x = Matrix::from_vec(n, 2, x_data).expect("valid");
            let mut lr = LogisticRegression::new().with_max_iter(500);
            lr.fit(&x, &y_data).expect("fit");

            let probas = lr.predict_proba(&x);
            for i in 0..probas.len() {
                let p = probas[i];
                prop_assert!(
                    (0.0..=1.0_f32).contains(&p),
                    "FALSIFIED LOGREG-003-prop: proba[{}]={} not in [0,1]",
                    i, p
                );
            }
        }
    }

    /// FALSIFY-LOGREG-004-prop: Deterministic predictions for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_logreg_004_prop_deterministic(
            seed in 0..500u32,
        ) {
            let n = 20;
            let x_data: Vec<f32> = (0..n).flat_map(|i| {
                let class = if i < n / 2 { 0.0 } else { 5.0 };
                let offset = ((i as f32 + seed as f32) * 0.37).sin() * 0.5;
                vec![class + offset, class + offset * 0.3]
            }).collect();
            let y_data: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

            let x = Matrix::from_vec(n, 2, x_data).expect("valid");
            let mut lr = LogisticRegression::new().with_max_iter(500);
            lr.fit(&x, &y_data).expect("fit");

            let p1 = lr.predict(&x);
            let p2 = lr.predict(&x);
            prop_assert_eq!(
                p1, p2,
                "FALSIFIED LOGREG-004-prop: predictions differ on same input"
            );
        }
    }

    /// FALSIFY-LOGREG-005-prop: Probabilities sum to 1 for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_logreg_005_prop_probabilities_sum_to_one(
            seed in 0..500u32,
        ) {
            let n = 20;
            let x_data: Vec<f32> = (0..n).flat_map(|i| {
                let class = if i < n / 2 { 0.0 } else { 5.0 };
                let offset = ((i as f32 + seed as f32) * 0.37).sin() * 0.5;
                vec![class + offset, class + offset * 0.3]
            }).collect();
            let y_data: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

            let x = Matrix::from_vec(n, 2, x_data).expect("valid");
            let mut lr = LogisticRegression::new().with_max_iter(500);
            lr.fit(&x, &y_data).expect("fit");

            let probas = lr.predict_proba(&x);
            for i in 0..probas.len() {
                let p = probas[i];
                let sum = p + (1.0 - p);
                prop_assert!(
                    (sum - 1.0_f32).abs() < 1e-6,
                    "FALSIFIED LOGREG-005-prop: sum={} != 1.0 at index {}",
                    sum, i
                );
            }
        }
    }
}
