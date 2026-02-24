// =========================================================================
// FALSIFY-HC: Agglomerative (hierarchical) clustering contract (aprender cluster)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-HC-* tests for AgglomerativeClustering
//   Why 2: agglomerative tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for hierarchical clustering yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Hierarchical clustering was "obviously correct" (textbook algorithm)
//
// References:
//   - Ward (1963) "Hierarchical grouping to optimize an objective function"
// =========================================================================

use super::*;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

/// FALSIFY-HC-001: Labels length matches sample count
#[test]
fn falsify_hc_001_labels_length() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
        ],
    )
    .expect("valid matrix");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data).expect("fit succeeds");

    let labels = hc.labels();
    assert_eq!(
        labels.len(),
        6,
        "FALSIFIED HC-001: labels len={}, expected 6",
        labels.len()
    );
}

/// FALSIFY-HC-002: Number of distinct labels equals n_clusters
#[test]
fn falsify_hc_002_n_clusters_correct() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
        ],
    )
    .expect("valid matrix");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Ward);
    hc.fit(&data).expect("fit succeeds");

    let labels = hc.labels();
    let mut unique: Vec<usize> = labels.clone();
    unique.sort_unstable();
    unique.dedup();

    assert_eq!(
        unique.len(),
        2,
        "FALSIFIED HC-002: unique labels={}, expected 2",
        unique.len()
    );
}

/// FALSIFY-HC-003: Two well-separated clusters get distinct labels
#[test]
fn falsify_hc_003_distinct_clusters() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 100.0, 100.0, 100.1, 100.0, 100.0, 100.1,
        ],
    )
    .expect("valid matrix");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Single);
    hc.fit(&data).expect("fit succeeds");

    let labels = hc.labels();
    // Cluster A points should share a label
    assert_eq!(
        labels[0], labels[1],
        "FALSIFIED HC-003: cluster A inconsistent"
    );
    assert_eq!(
        labels[1], labels[2],
        "FALSIFIED HC-003: cluster A inconsistent"
    );
    // Different clusters
    assert_ne!(
        labels[0], labels[3],
        "FALSIFIED HC-003: clusters have same label={}",
        labels[0]
    );
}
