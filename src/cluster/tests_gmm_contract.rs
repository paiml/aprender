// =========================================================================
// FALSIFY-GM: Gaussian Mixture Model contract (aprender cluster)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-GM-* tests for GMM
//   Why 2: GMM tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for GMM yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: EM convergence was "obviously correct" (textbook statistics)
//
// References:
//   - Dempster, Laird, Rubin (1977) "Maximum Likelihood from Incomplete Data via the EM Algorithm"
// =========================================================================

use super::*;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

/// FALSIFY-GM-001: Mixing weights sum to 1.0
#[test]
fn falsify_gm_001_weights_sum_to_one() {
    let data = Matrix::from_vec(6, 2, vec![
        1.0, 1.0,
        1.1, 1.0,
        1.0, 1.1,
        5.0, 5.0,
        5.1, 5.0,
        5.0, 5.1,
    ]).expect("valid matrix");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Spherical)
        .with_random_state(42)
        .with_max_iter(50);
    gmm.fit(&data).expect("fit succeeds");

    let weights = gmm.weights();
    let sum: f32 = weights.as_slice().iter().sum();

    assert!(
        (sum - 1.0).abs() < 1e-4,
        "FALSIFIED GM-001: weights sum={sum}, expected 1.0"
    );
}

/// FALSIFY-GM-002: Labels length matches sample count
#[test]
fn falsify_gm_002_labels_length() {
    let data = Matrix::from_vec(8, 2, vec![
        0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,
        5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
    ]).expect("valid matrix");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Spherical)
        .with_random_state(42)
        .with_max_iter(50);
    gmm.fit(&data).expect("fit succeeds");

    let labels = gmm.labels();
    assert_eq!(
        labels.len(), 8,
        "FALSIFIED GM-002: labels len={}, expected 8",
        labels.len()
    );
}

/// FALSIFY-GM-003: predict_proba rows sum to ~1.0 (responsibilities)
#[test]
fn falsify_gm_003_predict_proba_rows_sum_to_one() {
    let data = Matrix::from_vec(6, 2, vec![
        0.0, 0.0,
        0.1, 0.1,
        0.0, 0.1,
        10.0, 10.0,
        10.1, 10.1,
        10.0, 10.1,
    ]).expect("valid matrix");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Spherical)
        .with_random_state(42)
        .with_max_iter(50);
    gmm.fit(&data).expect("fit succeeds");

    let proba = gmm.predict_proba(&data);
    assert_eq!(proba.shape(), (6, 2), "FALSIFIED GM-003: wrong proba shape");

    for i in 0..6 {
        let row_sum: f32 = (0..2).map(|j| proba.get(i, j)).sum();
        assert!(
            (row_sum - 1.0).abs() < 0.1,
            "FALSIFIED GM-003: row {i} sum={row_sum}, expected ~1.0"
        );
    }
}

/// FALSIFY-GM-004: Number of weights equals n_components
#[test]
fn falsify_gm_004_n_weights_equals_n_components() {
    let data = Matrix::from_vec(6, 2, vec![
        1.0, 1.0, 1.1, 1.0, 1.0, 1.1,
        5.0, 5.0, 5.1, 5.0, 5.0, 5.1,
    ]).expect("valid matrix");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Spherical)
        .with_random_state(42)
        .with_max_iter(50);
    gmm.fit(&data).expect("fit succeeds");

    assert_eq!(
        gmm.weights().len(), 2,
        "FALSIFIED GM-004: weights len={}, expected 2",
        gmm.weights().len()
    );
}
