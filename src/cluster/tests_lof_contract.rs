// =========================================================================
// FALSIFY-LF: Local Outlier Factor contract (aprender cluster)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LF-* tests for LOF
//   Why 2: LOF tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for LOF yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: LOF was "obviously correct" (density ratio)
//
// References:
//   - Breunig et al. (2000) "LOF: Identifying Density-Based Local Outliers"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-LF-001: LOF scores are positive
#[test]
fn falsify_lf_001_scores_positive() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0, 1.0,
        ],
    )
    .expect("valid matrix");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.1);
    lof.fit(&data).expect("fit succeeds");

    let scores = lof.score_samples(&data);
    for (i, &score) in scores.iter().enumerate() {
        assert!(
            score > 0.0,
            "FALSIFIED LF-001: score[{i}]={score}, expected > 0.0"
        );
    }
}

/// FALSIFY-LF-002: Predictions are either 1 (normal) or -1 (anomaly)
#[test]
fn falsify_lf_002_predictions_binary() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0, 1.0,
        ],
    )
    .expect("valid matrix");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.1);
    lof.fit(&data).expect("fit succeeds");

    let preds = lof.predict(&data);
    for (i, &p) in preds.iter().enumerate() {
        assert!(
            p == 1 || p == -1,
            "FALSIFIED LF-002: prediction[{i}]={p}, expected 1 or -1"
        );
    }
}

/// FALSIFY-LF-003: Scores length matches sample count
#[test]
fn falsify_lf_003_scores_length() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.1, 1.0, 0.9, 0.9, 1.1, 1.0, 1.0,
        ],
    )
    .expect("valid matrix");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.1);
    lof.fit(&data).expect("fit succeeds");

    let scores = lof.score_samples(&data);
    assert_eq!(
        scores.len(),
        8,
        "FALSIFIED LF-003: scores len={}, expected 8",
        scores.len()
    );
}

mod lof_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-LF-001-prop: LOF scores positive for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_lf_001_prop_scores_positive(
            n in 8..=15usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut lof = LocalOutlierFactor::new()
                .with_n_neighbors(3)
                .with_contamination(0.1);
            lof.fit(&matrix).expect("fit");

            let scores = lof.score_samples(&matrix);
            for (i, &score) in scores.iter().enumerate() {
                prop_assert!(
                    score > 0.0,
                    "FALSIFIED LF-001-prop: score[{}]={} not > 0.0",
                    i, score
                );
            }
        }
    }

    /// FALSIFY-LF-003-prop: Scores length matches sample count
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_lf_003_prop_scores_length(
            n in 8..=15usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut lof = LocalOutlierFactor::new()
                .with_n_neighbors(3)
                .with_contamination(0.1);
            lof.fit(&matrix).expect("fit");

            let scores = lof.score_samples(&matrix);
            prop_assert_eq!(
                scores.len(),
                n,
                "FALSIFIED LF-003-prop: scores len {} != {}",
                scores.len(), n
            );
        }
    }
}
