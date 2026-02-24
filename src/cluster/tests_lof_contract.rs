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
