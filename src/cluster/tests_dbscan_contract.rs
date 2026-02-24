// =========================================================================
// FALSIFY-DB: DBSCAN clustering contract (aprender cluster)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-DB-* tests for DBSCAN
//   Why 2: DBSCAN tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for DBSCAN yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: DBSCAN was "obviously correct" (textbook algorithm)
//
// References:
//   - Ester et al. (1996) "A Density-Based Algorithm for Discovering Clusters"
// =========================================================================

use super::*;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

/// FALSIFY-DB-001: Noise points are labeled -1
#[test]
fn falsify_db_001_noise_labeled_negative_one() {
    // Two tight clusters + one far-away outlier
    let data = Matrix::from_vec(
        7,
        2,
        vec![
            1.0, 1.0, 1.1, 1.1, 1.2, 1.0, 5.0, 5.0, 5.1, 5.1, 5.0, 5.2, 99.0, 99.0, // outlier
        ],
    )
    .expect("valid matrix");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("fit succeeds");

    let labels = dbscan.labels();
    assert_eq!(
        labels[6], -1,
        "FALSIFIED DB-001: outlier label={}, expected -1",
        labels[6]
    );
}

/// FALSIFY-DB-002: Two well-separated clusters get distinct labels
#[test]
fn falsify_db_002_distinct_clusters() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
        ],
    )
    .expect("valid matrix");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("fit succeeds");

    let labels = dbscan.labels();
    // Cluster A points should share a label
    assert_eq!(
        labels[0], labels[1],
        "FALSIFIED DB-002: cluster A inconsistent"
    );
    assert_eq!(
        labels[1], labels[2],
        "FALSIFIED DB-002: cluster A inconsistent"
    );
    // Cluster B points should share a label
    assert_eq!(
        labels[3], labels[4],
        "FALSIFIED DB-002: cluster B inconsistent"
    );
    assert_eq!(
        labels[4], labels[5],
        "FALSIFIED DB-002: cluster B inconsistent"
    );
    // Two clusters should have different labels
    assert_ne!(
        labels[0], labels[3],
        "FALSIFIED DB-002: clusters A and B have same label={}",
        labels[0]
    );
}

/// FALSIFY-DB-003: Labels length matches input sample count
#[test]
fn falsify_db_003_labels_length_matches_samples() {
    let data = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
        .expect("valid matrix");

    let mut dbscan = DBSCAN::new(1.0, 2);
    dbscan.fit(&data).expect("fit succeeds");

    let labels = dbscan.labels();
    assert_eq!(
        labels.len(),
        5,
        "FALSIFIED DB-003: labels len={}, expected 5",
        labels.len()
    );
}

/// FALSIFY-DB-004: Cluster labels are non-negative (except noise = -1)
#[test]
fn falsify_db_004_cluster_labels_nonneg_or_noise() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
        ],
    )
    .expect("valid matrix");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("fit succeeds");

    for (i, &label) in dbscan.labels().iter().enumerate() {
        assert!(
            label >= -1,
            "FALSIFIED DB-004: label[{i}]={label}, expected >= -1"
        );
    }
}

mod dbscan_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-DB-003-prop: Labels length matches sample count for random sizes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_db_003_prop_labels_length(
            n in 3..=20usize,
            seed in 0..500u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut dbscan = DBSCAN::new(1.0, 2);
            dbscan.fit(&matrix).expect("fit");

            prop_assert_eq!(
                dbscan.labels().len(),
                n,
                "FALSIFIED DB-003-prop: labels len {} != {}",
                dbscan.labels().len(), n
            );
        }
    }

    /// FALSIFY-DB-004-prop: Labels >= -1 for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_db_004_prop_valid_labels(
            n in 3..=15usize,
            seed in 0..500u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let matrix = Matrix::from_vec(n, 2, data).expect("valid");
            let mut dbscan = DBSCAN::new(1.0, 2);
            dbscan.fit(&matrix).expect("fit");

            for (i, &label) in dbscan.labels().iter().enumerate() {
                prop_assert!(
                    label >= -1,
                    "FALSIFIED DB-004-prop: label[{}]={} < -1",
                    i, label
                );
            }
        }
    }
}
