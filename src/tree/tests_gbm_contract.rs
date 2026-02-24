// =========================================================================
// FALSIFY-GBM: gbm-v1.yaml contract (aprender GradientBoostingClassifier)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-GBM-* tests
//   Why 2: GBM tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from gbm-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: GBM was "obviously correct" (staged additive logistic regression)
//
// References:
//   - provable-contracts/contracts/gbm-v1.yaml
//   - Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-GBM-001: Predictions in training label set
#[test]
fn falsify_gbm_001_predictions_in_label_range() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.5, 0.5, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0, 6.5, 5.5,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 0, 1, 1, 1, 1];

    let mut gbm = GradientBoostingClassifier::new();
    gbm.fit(&x, &y).expect("fit");

    let preds = gbm.predict(&x).expect("predict");
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p <= 1,
            "FALSIFIED GBM-001: prediction[{i}] = {p}, not in {{0, 1}}"
        );
    }
}

/// FALSIFY-GBM-002: Prediction count matches input count
#[test]
fn falsify_gbm_002_prediction_count() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 1, 1, 1];

    let mut gbm = GradientBoostingClassifier::new();
    gbm.fit(&x, &y).expect("fit");

    let preds = gbm.predict(&x).expect("predict");
    assert_eq!(
        preds.len(),
        6,
        "FALSIFIED GBM-002: {} predictions for 6 inputs",
        preds.len()
    );
}

/// FALSIFY-GBM-003: Well-separated data classified correctly
#[test]
fn falsify_gbm_003_separable_data() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2, 10.3, 10.3,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 0, 1, 1, 1, 1];

    let mut gbm = GradientBoostingClassifier::new();
    gbm.fit(&x, &y).expect("fit");

    let preds = gbm.predict(&x).expect("predict");
    assert_eq!(
        preds, y,
        "FALSIFIED GBM-003: GBM cannot classify well-separated data"
    );
}

/// FALSIFY-GBM-004: Ensemble not worse than random on training data
#[test]
fn falsify_gbm_004_better_than_random() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2, 10.3, 10.3,
        ],
    )
    .expect("valid");
    let y = vec![0_usize, 0, 0, 0, 1, 1, 1, 1];

    let mut gbm = GradientBoostingClassifier::new();
    gbm.fit(&x, &y).expect("fit");

    let preds = gbm.predict(&x).expect("predict");
    let correct: usize = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
    let accuracy = correct as f32 / y.len() as f32;

    assert!(
        accuracy > 0.5,
        "FALSIFIED GBM-004: accuracy={accuracy} <= 0.5 (worse than random)"
    );
}

mod gbm_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-GBM-002-prop: Prediction count matches input count
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_gbm_002_prop_prediction_count(
            n in 8..=16usize,
            seed in 0..200u32,
        ) {
            let half = n / 2;
            let mut x_data = Vec::with_capacity(n * 2);
            let mut y_data = Vec::with_capacity(n);
            for i in 0..half {
                let offset = (seed as f32 + i as f32) * 0.01;
                x_data.push(0.0 + offset);
                x_data.push(0.0 + offset);
                y_data.push(0_usize);
            }
            for i in 0..(n - half) {
                let offset = (seed as f32 + i as f32) * 0.01;
                x_data.push(10.0 + offset);
                x_data.push(10.0 + offset);
                y_data.push(1_usize);
            }
            let x = Matrix::from_vec(n, 2, x_data).expect("valid");

            let mut gbm = GradientBoostingClassifier::new();
            gbm.fit(&x, &y_data).expect("fit");

            let preds = gbm.predict(&x).expect("predict");
            prop_assert_eq!(
                preds.len(),
                n,
                "FALSIFIED GBM-002-prop: {} predictions for {} inputs",
                preds.len(), n
            );
        }
    }

    /// FALSIFY-GBM-001-prop: Predictions in valid label range
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_gbm_001_prop_predictions_valid(
            seed in 0..200u32,
        ) {
            let n = 8;
            let mut x_data = Vec::with_capacity(n * 2);
            let mut y_data = Vec::with_capacity(n);
            for i in 0..4 {
                let offset = (seed as f32 + i as f32) * 0.01;
                x_data.push(0.0 + offset);
                x_data.push(0.0 + offset);
                y_data.push(0_usize);
            }
            for i in 0..4 {
                let offset = (seed as f32 + i as f32) * 0.01;
                x_data.push(10.0 + offset);
                x_data.push(10.0 + offset);
                y_data.push(1_usize);
            }
            let x = Matrix::from_vec(n, 2, x_data).expect("valid");

            let mut gbm = GradientBoostingClassifier::new();
            gbm.fit(&x, &y_data).expect("fit");

            let preds = gbm.predict(&x).expect("predict");
            for (i, &p) in preds.iter().enumerate() {
                prop_assert!(
                    p <= 1,
                    "FALSIFIED GBM-001-prop: prediction[{}]={} not in {{0, 1}}",
                    i, p
                );
            }
        }
    }
}
