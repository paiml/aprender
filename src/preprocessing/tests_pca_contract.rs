// =========================================================================
// FALSIFY-PCA: pca-v1.yaml contract (aprender PCA)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest PCA tests but zero inline FALSIFY-PCA-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from pca-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: PCA was "obviously correct" (standard eigendecomposition)
//
// References:
//   - provable-contracts/contracts/pca-v1.yaml
//   - Hotelling (1933) "Analysis of a complex of statistical variables"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-PCA-001: Dimensionality reduction — output has n_components columns
#[test]
fn falsify_pca_001_dimensionality_reduction() {
    let data = Matrix::from_vec(
        5,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0,
            5.0, 6.0, 7.0, 8.0,
        ],
    )
    .expect("valid matrix");

    for &n_components in &[1, 2, 3] {
        let mut pca = PCA::new(n_components);
        pca.fit(&data).expect("fit succeeds");
        let transformed = pca.transform(&data).expect("transform succeeds");

        let (n_samples, n_cols) = transformed.shape();
        assert_eq!(
            n_samples, 5,
            "FALSIFIED PCA-001: n_samples changed"
        );
        assert_eq!(
            n_cols, n_components,
            "FALSIFIED PCA-001: output has {n_cols} cols, expected {n_components}"
        );
    }
}

/// FALSIFY-PCA-002: Explained variance ratio bounded [0, 1] and sums to ≤ 1
#[test]
fn falsify_pca_002_explained_variance_bounded() {
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5,
            0.5, 1.5,
        ],
    )
    .expect("valid matrix");

    let mut pca = PCA::new(3);
    pca.fit(&data).expect("fit succeeds");

    let ratios = pca.explained_variance_ratio().expect("has ratios");
    let sum: f32 = ratios.iter().sum();

    for (i, &r) in ratios.iter().enumerate() {
        assert!(
            r >= -1e-6,
            "FALSIFIED PCA-002: ratio[{i}] = {r} < 0"
        );
        assert!(
            r <= 1.0 + 1e-6,
            "FALSIFIED PCA-002: ratio[{i}] = {r} > 1"
        );
    }

    assert!(
        sum <= 1.0 + 1e-4,
        "FALSIFIED PCA-002: sum(ratios) = {sum} > 1"
    );
}

/// FALSIFY-PCA-003: Variance ordering — components sorted by explained variance
#[test]
fn falsify_pca_003_variance_ordering() {
    let data = Matrix::from_vec(
        5,
        3,
        vec![
            1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.3, 0.03, 4.0, 0.4, 0.04, 5.0, 0.5, 0.05,
        ],
    )
    .expect("valid matrix");

    let mut pca = PCA::new(3);
    pca.fit(&data).expect("fit succeeds");

    let variances = pca.explained_variance().expect("has variances");
    for i in 1..variances.len() {
        assert!(
            variances[i] <= variances[i - 1] + 1e-6,
            "FALSIFIED PCA-003: variance[{i}]={} > variance[{}]={}",
            variances[i],
            i - 1,
            variances[i - 1]
        );
    }
}

/// FALSIFY-PCA-004: Deterministic — same data produces same transform
#[test]
fn falsify_pca_004_deterministic() {
    let data = Matrix::from_vec(
        4,
        2,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .expect("valid matrix");

    let mut pca1 = PCA::new(2);
    pca1.fit(&data).expect("fit 1");
    let t1 = pca1.transform(&data).expect("transform 1");

    let mut pca2 = PCA::new(2);
    pca2.fit(&data).expect("fit 2");
    let t2 = pca2.transform(&data).expect("transform 2");

    let (rows, cols) = t1.shape();
    for i in 0..rows {
        for j in 0..cols {
            assert!(
                (t1.get(i, j) - t2.get(i, j)).abs() < 1e-5,
                "FALSIFIED PCA-004: [{i},{j}] first={} != second={}",
                t1.get(i, j),
                t2.get(i, j)
            );
        }
    }
}
