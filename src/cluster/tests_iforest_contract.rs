// =========================================================================
// FALSIFY-IF: Isolation Forest contract (aprender cluster)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-IF-* tests for IsolationForest
//   Why 2: isolation forest tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for isolation forest yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Isolation Forest was "obviously correct" (path length anomaly)
//
// References:
//   - Liu, Ting, Zhou (2008) "Isolation Forest"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-IF-001: Anomaly scores are in [-1, 0] (negated convention)
#[test]
fn falsify_if_001_scores_bounded() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0, 1.0,
        ],
    )
    .expect("valid matrix");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(42);
    iforest.fit(&data).expect("fit succeeds");

    let scores = iforest.score_samples(&data);
    for (i, &score) in scores.iter().enumerate() {
        assert!(
            (-1.0..=0.0).contains(&score),
            "FALSIFIED IF-001: score[{i}]={score}, expected in [-1,0]"
        );
    }
}

/// FALSIFY-IF-002: Predictions are either 1 (normal) or -1 (anomaly)
#[test]
fn falsify_if_002_predictions_binary() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0, 1.0,
        ],
    )
    .expect("valid matrix");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(42)
        .with_contamination(0.1);
    iforest.fit(&data).expect("fit succeeds");

    let preds = iforest.predict(&data);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p == 1 || p == -1,
            "FALSIFIED IF-002: prediction[{i}]={p}, expected 1 or -1"
        );
    }
}

/// FALSIFY-IF-003: Predictions length matches sample count
#[test]
fn falsify_if_003_predictions_length() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 0.9, 1.2, 1.0, 1.1, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0,
            1.0, 0.8, 1.2,
        ],
    )
    .expect("valid matrix");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(42);
    iforest.fit(&data).expect("fit succeeds");

    let preds = iforest.predict(&data);
    assert_eq!(
        preds.len(),
        10,
        "FALSIFIED IF-003: predictions len={}, expected 10",
        preds.len()
    );
}

mod iforest_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-IF-001-prop: Anomaly scores in [-1, 0] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_if_001_prop_scores_bounded(
            n in 8..=20usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut iforest = IsolationForest::new()
                .with_n_estimators(50)
                .with_random_state(seed as u64);
            iforest.fit(&matrix).expect("fit");

            let scores = iforest.score_samples(&matrix);
            for (i, &score) in scores.iter().enumerate() {
                prop_assert!(
                    (-1.0..=0.0).contains(&score),
                    "FALSIFIED IF-001-prop: score[{}]={} not in [-1,0]",
                    i, score
                );
            }
        }
    }

    /// FALSIFY-IF-003-prop: Predictions length matches sample count
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_if_003_prop_predictions_length(
            n in 8..=20usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut iforest = IsolationForest::new()
                .with_n_estimators(50)
                .with_random_state(seed as u64);
            iforest.fit(&matrix).expect("fit");

            let preds = iforest.predict(&matrix);
            prop_assert_eq!(
                preds.len(),
                n,
                "FALSIFIED IF-003-prop: predictions len {} != {}",
                preds.len(), n
            );
        }
    }
}
