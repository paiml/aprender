//! Advanced clustering tests: DBSCAN, Agglomerative, GMM, Isolation Forest, LOF, Spectral.

use crate::cluster::*;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

// ============================================================================
// DBSCAN Tests
// ============================================================================

#[test]
fn test_dbscan_new() {
    let dbscan = DBSCAN::new(0.5, 3);
    assert_eq!(dbscan.eps(), 0.5);
    assert_eq!(dbscan.min_samples(), 3);
    assert!(!dbscan.is_fitted());
}

#[test]
fn test_dbscan_fit_basic() {
    // Two well-separated clusters
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // Cluster 0
            1.2, 1.1, // Cluster 0
            1.1, 1.2, // Cluster 0
            5.0, 5.0, // Cluster 1
            5.1, 5.2, // Cluster 1
            5.2, 5.1, // Cluster 1
        ],
    )
    .expect("Matrix creation should succeed");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("DBSCAN fit should succeed");

    assert!(dbscan.is_fitted());
    let labels = dbscan.labels();
    assert_eq!(labels.len(), 6);
}

#[test]
fn test_dbscan_predicts_clusters() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, 1.2, 1.1, 1.1, 1.2, // Cluster 0
            5.0, 5.0, 5.1, 5.2, 5.2, 5.1, // Cluster 1
        ],
    )
    .expect("Matrix creation should succeed");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("DBSCAN fit should succeed");

    let labels = dbscan.predict(&data);
    assert_eq!(labels.len(), 6);

    // First 3 samples should be in same cluster
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);

    // Last 3 samples should be in same cluster
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);

    // Two clusters should be different
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_dbscan_noise_detection() {
    // Two clusters with one outlier
    let data = Matrix::from_vec(
        7,
        2,
        vec![
            1.0, 1.0, // Cluster 0
            1.2, 1.1, // Cluster 0
            1.1, 1.2, // Cluster 0
            5.0, 5.0, // Cluster 1
            5.1, 5.2, // Cluster 1
            5.2, 5.1, // Cluster 1
            10.0, 10.0, // Noise (far from both clusters)
        ],
    )
    .expect("Matrix creation should succeed");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("DBSCAN fit should succeed");

    let labels = dbscan.labels();

    // Last sample should be noise (-1)
    assert_eq!(labels[6], -1);
}

#[test]
fn test_dbscan_single_cluster() {
    // All points form one dense cluster
    let data = Matrix::from_vec(5, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.0])
        .expect("Matrix creation should succeed");

    let mut dbscan = DBSCAN::new(0.3, 2);
    dbscan.fit(&data).expect("DBSCAN fit should succeed");

    let labels = dbscan.labels();

    // All samples should be in the same cluster (not noise)
    let first_label = labels[0];
    assert_ne!(first_label, -1);
    for &label in labels {
        assert_eq!(label, first_label);
    }
}

#[test]
fn test_dbscan_all_noise() {
    // All points far apart
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
        .expect("Matrix creation should succeed");

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).expect("DBSCAN fit should succeed");

    let labels = dbscan.labels();

    // All samples should be noise
    for &label in labels {
        assert_eq!(label, -1);
    }
}

#[test]
fn test_dbscan_min_samples_effect() {
    // Same data, different min_samples
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1])
        .expect("Matrix creation should succeed");

    // With min_samples=2, should form cluster
    let mut dbscan1 = DBSCAN::new(0.3, 2);
    dbscan1.fit(&data).expect("DBSCAN fit should succeed");
    let labels1 = dbscan1.labels();
    assert!(labels1.iter().any(|&l| l != -1));

    // With min_samples=5, should be all noise
    let mut dbscan2 = DBSCAN::new(0.3, 5);
    dbscan2.fit(&data).expect("DBSCAN fit should succeed");
    let labels2 = dbscan2.labels();
    assert!(labels2.iter().all(|&l| l == -1));
}

#[test]
fn test_dbscan_eps_effect() {
    // Same data, different eps
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5])
        .expect("Matrix creation should succeed");

    // With large eps, should form one cluster
    let mut dbscan1 = DBSCAN::new(2.0, 2);
    dbscan1.fit(&data).expect("DBSCAN fit should succeed");
    let labels1 = dbscan1.labels();
    let unique_clusters: std::collections::HashSet<_> =
        labels1.iter().filter(|&&l| l != -1).collect();
    assert_eq!(unique_clusters.len(), 1);

    // With small eps, more fragmentation
    let mut dbscan2 = DBSCAN::new(0.3, 2);
    dbscan2.fit(&data).expect("DBSCAN fit should succeed");
    let labels2 = dbscan2.labels();
    // Should have noise or multiple small clusters
    assert!(labels2.contains(&-1));
}

#[test]
fn test_dbscan_reproducible() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.2, 1.1, 1.1, 1.2, 5.0, 5.0, 5.1, 5.2, 5.2, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut dbscan1 = DBSCAN::new(0.5, 2);
    dbscan1.fit(&data).expect("DBSCAN fit should succeed");
    let labels1 = dbscan1.labels().clone();

    let mut dbscan2 = DBSCAN::new(0.5, 2);
    dbscan2.fit(&data).expect("DBSCAN fit should succeed");
    let labels2 = dbscan2.labels();

    // Results should be identical
    assert_eq!(labels1, *labels2);
}

#[test]
fn test_dbscan_fit_predict() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1])
        .expect("Matrix creation should succeed");

    let mut dbscan = DBSCAN::new(0.3, 2);
    dbscan.fit(&data).expect("DBSCAN fit should succeed");

    let labels_stored = dbscan.labels().clone();
    let labels_predicted = dbscan.predict(&data);

    // predict() should return same labels as stored from fit()
    assert_eq!(labels_stored, labels_predicted);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_dbscan_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let dbscan = DBSCAN::new(0.5, 2);
    let _ = dbscan.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_dbscan_labels_before_fit() {
    let dbscan = DBSCAN::new(0.5, 2);
    let _ = dbscan.labels(); // Should panic
}

// ==================== AgglomerativeClustering Tests ====================

#[test]
fn test_agglomerative_new() {
    let hc = AgglomerativeClustering::new(3, Linkage::Average);
    assert_eq!(hc.n_clusters(), 3);
    assert_eq!(hc.linkage(), Linkage::Average);
    assert!(!hc.is_fitted());
}

#[test]
fn test_agglomerative_fit_basic() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");
    assert!(hc.is_fitted());
}

#[test]
fn test_agglomerative_predict() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    assert_eq!(labels.len(), 6);

    // All labels should be valid cluster indices
    for &label in &labels {
        assert!(label < 2);
    }

    // Check that two distinct clusters were found
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique_labels.len(), 2);
}

#[test]
fn test_agglomerative_linkage_single() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Single);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    assert_eq!(labels.len(), 4);

    // Check that two clusters were formed
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique_labels.len(), 2);
}

#[test]
fn test_agglomerative_linkage_complete() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Complete);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    assert_eq!(labels.len(), 4);

    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique_labels.len(), 2);
}

#[test]
fn test_agglomerative_linkage_average() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    assert_eq!(labels.len(), 4);

    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique_labels.len(), 2);
}

#[test]
fn test_agglomerative_linkage_ward() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Ward);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    assert_eq!(labels.len(), 4);

    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique_labels.len(), 2);
}

#[test]
fn test_agglomerative_n_clusters_1() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(1, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    // All points should be in same cluster
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn test_agglomerative_n_clusters_equals_samples() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(4, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels = hc.predict(&data);
    // Each point should be its own cluster
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert_eq!(unique_labels.len(), 4);
}

#[test]
fn test_agglomerative_dendrogram() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let dendrogram = hc.dendrogram();
    // Should have n_samples - n_clusters merges
    assert_eq!(dendrogram.len(), 2); // 4 samples - 2 clusters = 2 merges
}

#[test]
fn test_agglomerative_reproducible() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut hc1 = AgglomerativeClustering::new(2, Linkage::Average);
    hc1.fit(&data)
        .expect("Hierarchical clustering fit should succeed");
    let labels1 = hc1.predict(&data);

    let mut hc2 = AgglomerativeClustering::new(2, Linkage::Average);
    hc2.fit(&data)
        .expect("Hierarchical clustering fit should succeed");
    let labels2 = hc2.predict(&data);

    // Results should be deterministic
    assert_eq!(labels1, labels2);
}

#[test]
fn test_agglomerative_fit_predict_consistency() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Complete);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");

    let labels_stored = hc.labels().clone();
    let labels_predicted = hc.predict(&data);

    assert_eq!(labels_stored, labels_predicted);
}

include!("advanced_part_02.rs");
include!("advanced_part_03.rs");
include!("advanced_part_04.rs");
include!("advanced_part_05.rs");
include!("advanced_part_06.rs");
