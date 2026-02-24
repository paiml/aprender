// =========================================================================
// FALSIFY-SC: Spectral Clustering contract (aprender cluster)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-SC-* tests for Spectral Clustering
//   Why 2: spectral tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for spectral clustering yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Spectral clustering was "obviously correct" (eigendecomposition)
//
// References:
//   - Ng, Jordan, Weiss (2001) "On Spectral Clustering: Analysis and an algorithm"
// =========================================================================

use super::*;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

/// FALSIFY-SC-001: Labels length matches sample count
#[test]
fn falsify_sc_001_labels_length() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
        ],
    )
    .expect("valid matrix");

    let mut sc = SpectralClustering::new(2).with_gamma(1.0);
    sc.fit(&data).expect("fit succeeds");

    let labels = sc.predict(&data);
    assert_eq!(
        labels.len(),
        6,
        "FALSIFIED SC-001: labels len={}, expected 6",
        labels.len()
    );
}

/// FALSIFY-SC-002: Label values are in [0, n_clusters)
#[test]
fn falsify_sc_002_label_range() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
        ],
    )
    .expect("valid matrix");

    let mut sc = SpectralClustering::new(2).with_gamma(1.0);
    sc.fit(&data).expect("fit succeeds");

    let labels = sc.predict(&data);
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label < 2,
            "FALSIFIED SC-002: label[{i}]={label}, expected < 2"
        );
    }
}

/// FALSIFY-SC-003: Two well-separated clusters get distinct labels
#[test]
fn falsify_sc_003_distinct_clusters() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 100.0, 100.0, 100.1, 100.0, 100.0, 100.1,
        ],
    )
    .expect("valid matrix");

    let mut sc = SpectralClustering::new(2).with_gamma(0.01);
    sc.fit(&data).expect("fit succeeds");

    let labels = sc.predict(&data);
    // Points in cluster A should share a label
    assert_eq!(
        labels[0], labels[1],
        "FALSIFIED SC-003: cluster A inconsistent"
    );
    assert_eq!(
        labels[1], labels[2],
        "FALSIFIED SC-003: cluster A inconsistent"
    );
    // Points in cluster B should share a label
    assert_eq!(
        labels[3], labels[4],
        "FALSIFIED SC-003: cluster B inconsistent"
    );
    assert_eq!(
        labels[4], labels[5],
        "FALSIFIED SC-003: cluster B inconsistent"
    );
    // Different clusters
    assert_ne!(
        labels[0], labels[3],
        "FALSIFIED SC-003: clusters A and B have same label={}",
        labels[0]
    );
}

mod sc_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-SC-001-prop: Labels length matches sample count for random sizes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_sc_001_prop_labels_length(
            n in 4..=12usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut sc = SpectralClustering::new(2).with_gamma(1.0);
            sc.fit(&matrix).expect("fit");

            let labels = sc.predict(&matrix);
            prop_assert_eq!(
                labels.len(),
                n,
                "FALSIFIED SC-001-prop: labels len {} != {}",
                labels.len(), n
            );
        }
    }

    /// FALSIFY-SC-002-prop: Label values in [0, n_clusters) for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_sc_002_prop_label_range(
            n in 4..=12usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut sc = SpectralClustering::new(2).with_gamma(1.0);
            sc.fit(&matrix).expect("fit");

            let labels = sc.predict(&matrix);
            for (i, &label) in labels.iter().enumerate() {
                prop_assert!(
                    label < 2,
                    "FALSIFIED SC-002-prop: label[{}]={} >= 2",
                    i, label
                );
            }
        }
    }
}
