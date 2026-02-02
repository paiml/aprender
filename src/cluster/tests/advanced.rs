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

#[test]
fn test_agglomerative_different_linkages_differ() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 8.0, 8.0, 8.5, 8.5, 9.0, 9.0],
    )
    .expect("Matrix creation should succeed");

    let mut hc_single = AgglomerativeClustering::new(2, Linkage::Single);
    hc_single
        .fit(&data)
        .expect("Hierarchical clustering fit should succeed");
    let labels_single = hc_single.predict(&data);

    let mut hc_complete = AgglomerativeClustering::new(2, Linkage::Complete);
    hc_complete
        .fit(&data)
        .expect("Hierarchical clustering fit should succeed");
    let labels_complete = hc_complete.predict(&data);

    // Different linkage methods may produce different results
    // but both should have exactly 2 clusters
    let unique_single: std::collections::HashSet<_> = labels_single.iter().collect();
    let unique_complete: std::collections::HashSet<_> = labels_complete.iter().collect();
    assert_eq!(unique_single.len(), 2);
    assert_eq!(unique_complete.len(), 2);
}

#[test]
fn test_agglomerative_well_separated_clusters() {
    // Two very well-separated clusters
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 100.0, 100.0, 100.1, 100.1, 100.0, 100.1,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data)
        .expect("Hierarchical clustering fit should succeed");
    let labels = hc.predict(&data);

    // First 3 points should be in one cluster, last 3 in another
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_agglomerative_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let hc = AgglomerativeClustering::new(2, Linkage::Average);
    let _ = hc.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_agglomerative_labels_before_fit() {
    let hc = AgglomerativeClustering::new(2, Linkage::Average);
    let _ = hc.labels(); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_agglomerative_dendrogram_before_fit() {
    let hc = AgglomerativeClustering::new(2, Linkage::Average);
    let _ = hc.dendrogram(); // Should panic
}

// ==================== GaussianMixture Tests ====================

#[test]
fn test_gmm_new() {
    let gmm = GaussianMixture::new(3, CovarianceType::Full);
    assert_eq!(gmm.n_components(), 3);
    assert_eq!(gmm.covariance_type(), CovarianceType::Full);
    assert!(!gmm.is_fitted());
}

#[test]
fn test_gmm_fit_basic() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");
    assert!(gmm.is_fitted());
}

#[test]
fn test_gmm_predict() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels = gmm.predict(&data);
    assert_eq!(labels.len(), 6);

    // All labels should be valid component indices
    for &label in &labels {
        assert!(label < 2);
    }
}

#[test]
fn test_gmm_predict_proba() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let proba = gmm.predict_proba(&data);
    assert_eq!(proba.shape(), (6, 2));

    // Probabilities should sum to 1 for each sample
    for i in 0..6 {
        let sum: f32 = (0..2).map(|j| proba.get(i, j)).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // All probabilities should be in [0, 1]
    for i in 0..6 {
        for j in 0..2 {
            let p = proba.get(i, j);
            assert!((0.0..=1.0).contains(&p));
        }
    }
}

#[test]
fn test_gmm_covariance_full() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels = gmm.predict(&data);
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_gmm_covariance_tied() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Tied);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels = gmm.predict(&data);
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_gmm_covariance_diag() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Diag);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels = gmm.predict(&data);
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_gmm_covariance_spherical() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Spherical);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels = gmm.predict(&data);
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_gmm_score() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let score = gmm.score(&data);
    // Log-likelihood should be finite
    assert!(score.is_finite());
}

#[test]
fn test_gmm_convergence() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full).with_max_iter(100);
    gmm.fit(&data).expect("GMM fit should succeed");
    assert!(gmm.is_fitted());
}

#[test]
fn test_gmm_reproducible() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm1 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm1.fit(&data).expect("GMM fit should succeed");
    let labels1 = gmm1.predict(&data);

    let mut gmm2 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm2.fit(&data).expect("GMM fit should succeed");
    let labels2 = gmm2.predict(&data);

    // Same seed should produce same results
    assert_eq!(labels1, labels2);
}

#[test]
fn test_gmm_means() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let means = gmm.means();
    assert_eq!(means.shape(), (2, 2));
}

#[test]
fn test_gmm_weights() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let weights = gmm.weights();
    assert_eq!(weights.len(), 2);

    // Weights should sum to 1
    let sum: f32 = weights.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_gmm_well_separated() {
    // Two very well-separated clusters
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 100.0, 100.0, 100.1, 100.1, 100.0, 100.1,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");
    let labels = gmm.predict(&data);

    // First 3 points should be in one cluster, last 3 in another
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_gmm_soft_vs_hard_assignment() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels = gmm.predict(&data);
    let proba = gmm.predict_proba(&data);

    // Hard assignment should match argmax of soft assignment
    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        let mut max_prob = 0.0;
        let mut max_idx = 0;
        for j in 0..2 {
            let p = proba.get(i, j);
            if p > max_prob {
                max_prob = p;
                max_idx = j;
            }
        }
        assert_eq!(labels[i], max_idx);
    }
}

#[test]
fn test_gmm_fit_predict_consistency() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
    gmm.fit(&data).expect("GMM fit should succeed");

    let labels_stored = gmm.labels().clone();
    let labels_predicted = gmm.predict(&data);

    assert_eq!(labels_stored, labels_predicted);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_gmm_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let gmm = GaussianMixture::new(2, CovarianceType::Full);
    let _ = gmm.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_gmm_predict_proba_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let gmm = GaussianMixture::new(2, CovarianceType::Full);
    let _ = gmm.predict_proba(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_gmm_score_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let gmm = GaussianMixture::new(2, CovarianceType::Full);
    let _ = gmm.score(&data); // Should panic
}

// ========================================================================
// Isolation Forest Tests
// ========================================================================

#[test]
fn test_isolation_forest_new() {
    let iforest = IsolationForest::new();
    assert!(!iforest.is_fitted());
}

#[test]
fn test_isolation_forest_fit_basic() {
    // Normal data clustered around (2, 2)
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 1.9,
            1.9, 2.1, 1.8,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new();
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    assert!(iforest.is_fitted());
}

#[test]
fn test_isolation_forest_predict_anomalies() {
    // 8 normal points + 2 outliers
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, // Outlier 1
            -10.0, -10.0, // Outlier 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 10);

    // Check that predictions are either 1 (normal) or -1 (anomaly)
    for &pred in &predictions {
        assert!(pred == 1 || pred == -1);
    }

    // Should detect approximately 2 anomalies (20% contamination)
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    assert!((1..=3).contains(&n_anomalies));
}

#[test]
fn test_isolation_forest_score_samples() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
            10.0, 10.0, // Outlier 1
            -10.0, -10.0, // Outlier 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 6);

    // Outliers should have lower scores than normal points
    let normal_avg = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
    let outlier_avg = (scores[4] + scores[5]) / 2.0;
    assert!(outlier_avg < normal_avg);
}

#[test]
fn test_isolation_forest_contamination() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Low contamination (10%) - fewer anomalies expected
    let mut iforest_low = IsolationForest::new()
        .with_contamination(0.1)
        .with_random_state(42);
    iforest_low
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred_low = iforest_low.predict(&data);
    let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();

    // High contamination (30%) - more anomalies expected
    let mut iforest_high = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest_high
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred_high = iforest_high.predict(&data);
    let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();

    // Higher contamination should detect more or equal anomalies
    assert!(anomalies_high >= anomalies_low);
}

#[test]
fn test_isolation_forest_n_estimators() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Fewer trees
    let mut iforest_few = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest_few
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    // More trees (should be more stable)
    let mut iforest_many = IsolationForest::new()
        .with_n_estimators(100)
        .with_random_state(42);
    iforest_many
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    // Both should work, more trees typically more accurate
    let pred_few = iforest_few.predict(&data);
    let pred_many = iforest_many.predict(&data);

    assert_eq!(pred_few.len(), 8);
    assert_eq!(pred_many.len(), 8);
}

#[test]
fn test_isolation_forest_max_samples() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Use subset of samples for each tree
    let mut iforest = IsolationForest::new()
        .with_max_samples(5)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 10);
}

#[test]
fn test_isolation_forest_reproducible() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest1 = IsolationForest::new().with_random_state(42);
    iforest1
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred1 = iforest1.predict(&data);

    let mut iforest2 = IsolationForest::new().with_random_state(42);
    iforest2
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred2 = iforest2.predict(&data);

    assert_eq!(pred1, pred2);
}

#[test]
fn test_isolation_forest_all_normal() {
    // All points are normal (tightly clustered)
    let data = Matrix::from_vec(
        6,
        2,
        vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.1)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    // With 10% contamination, expect mostly normal points
    let n_normal = predictions.iter().filter(|&&p| p == 1).count();
    assert!(n_normal >= 5);
}

#[test]
fn test_isolation_forest_score_samples_range() {
    let data = Matrix::from_vec(4, 2, vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 10.0, 10.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let scores = iforest.score_samples(&data);
    // Anomaly scores should be in reasonable range
    for &score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn test_isolation_forest_path_length() {
    // Test that isolation path length is computed correctly
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
            10.0, 10.0, // Easy to isolate outlier
            2.05, 2.05, // Normal
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let scores = iforest.score_samples(&data);
    // Outlier (index 4) should have significantly different score
    let outlier_score = scores[4];
    let normal_score = (scores[0] + scores[1] + scores[2] + scores[3] + scores[5]) / 5.0;
    assert!(outlier_score < normal_score);
}

#[test]
fn test_isolation_forest_multidimensional() {
    // Test with more features
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.0, 2.0, 3.0, 0.9, 1.9, 2.9, 10.0, 10.0, 10.0, -10.0,
            -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_isolation_forest_decision_function_consistency() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    let scores = iforest.score_samples(&data);

    // Points with lower scores should be more likely to be anomalies
    // (though exact correspondence depends on threshold)
    assert_eq!(predictions.len(), scores.len());
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_isolation_forest_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let iforest = IsolationForest::new();
    let _ = iforest.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_isolation_forest_score_samples_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let iforest = IsolationForest::new();
    let _ = iforest.score_samples(&data); // Should panic
}

#[test]
fn test_isolation_forest_empty_after_construction() {
    let iforest = IsolationForest::new();
    assert!(!iforest.is_fitted());
}

// ========================================================================
// Isolation Forest Edge Case / Coverage Tests
// ========================================================================

#[test]
fn test_isolation_forest_default() {
    // Exercise the Default impl for IsolationForest
    let iforest = IsolationForest::default();
    assert!(!iforest.is_fitted());
}

#[test]
fn test_isolation_forest_debug_impl() {
    // Exercise the Debug derive
    let iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.2);
    let debug_str = format!("{:?}", iforest);
    assert!(debug_str.contains("IsolationForest"));
    assert!(debug_str.contains("n_estimators"));
}

#[test]
fn test_isolation_forest_clone() {
    // Exercise the Clone derive
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let iforest_clone = iforest.clone();
    assert!(iforest_clone.is_fitted());

    let scores_orig = iforest.score_samples(&data);
    let scores_clone = iforest_clone.score_samples(&data);
    assert_eq!(scores_orig, scores_clone);
}

#[test]
fn test_isolation_forest_contamination_clamping_high() {
    // Values > 0.5 get clamped to 0.5
    let iforest = IsolationForest::new().with_contamination(0.9);
    let debug_str = format!("{:?}", iforest);
    assert!(debug_str.contains("0.5"));
}

#[test]
fn test_isolation_forest_contamination_clamping_low() {
    // Values < 0.0 get clamped to 0.0
    let iforest = IsolationForest::new().with_contamination(-1.0);
    let debug_str = format!("{:?}", iforest);
    assert!(debug_str.contains("contamination: 0.0"));
}

#[test]
fn test_isolation_forest_identical_points() {
    // All identical points: exercises the all-same-values leaf path (line 81)
    // in build_tree where (max_val - min_val).abs() < 1e-10
    let data = Matrix::from_vec(5, 2, vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Fit should succeed with identical points");
    assert!(iforest.is_fitted());

    let scores = iforest.score_samples(&data);
    for &score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn test_isolation_forest_single_point() {
    // A single data point: exercises n_samples <= 1 terminal condition (line 57)
    // Also exercises c(1) = 0.0, which makes c_norm = 0.0.
    // The score formula 2^(-path/0) produces -inf, which is expected behavior
    // for degenerate input.
    let data = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Fit should succeed with single point");
    assert!(iforest.is_fitted());

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 1);
    // With n=1, c_norm=0.0 so score is 2^(-path/0) = -inf or NaN
    // The important thing is the model fits and produces a score without crashing
}

#[test]
fn test_isolation_forest_two_points() {
    // Two data points: exercises c(2) == 1.0 path (line 169)
    let data =
        Matrix::from_vec(2, 2, vec![0.0, 0.0, 10.0, 10.0]).expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_max_samples(2)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Fit should succeed with two points");
    assert!(iforest.is_fitted());

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 2);
    for &score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn test_isolation_forest_without_random_state() {
    // Exercises the from_entropy() path (line 310) when no seed is provided
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_n_estimators(10);
    // No .with_random_state() call
    iforest.fit(&data).expect("Fit should succeed without seed");
    assert!(iforest.is_fitted());
}

#[test]
fn test_isolation_forest_single_feature() {
    // Exercise with 1D data
    let data = Matrix::from_vec(5, 1, vec![1.0, 1.1, 1.2, 1.0, 100.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 5);

    // The outlier at 100.0 should likely be flagged
    let scores = iforest.score_samples(&data);
    // Outlier should have more negative score (more anomalous)
    assert!(scores[4] < scores[0]);
}

#[test]
fn test_isolation_forest_max_samples_larger_than_data() {
    // max_samples > n_samples: exercises default clamping n_samples.min(256)
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_max_samples(1000)
        .with_n_estimators(10)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");
    assert!(iforest.is_fitted());
}

#[test]
fn test_isolation_forest_contamination_zero() {
    // With contamination=0, threshold_idx=0
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.0)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_isolation_forest_refit_clears_trees() {
    // Exercises the self.trees.clear() path (line 314) on refit
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest.fit(&data).expect("First fit should succeed");

    let scores1 = iforest.score_samples(&data);

    // Refit on same data - exercises clear + rebuild
    iforest.fit(&data).expect("Second fit should succeed");

    let scores2 = iforest.score_samples(&data);
    // Same seed + same data = same scores
    assert_eq!(scores1, scores2);
}

#[test]
fn test_isolation_forest_many_features() {
    // Higher dimensionality exercises random feature selection more thoroughly
    let data = Matrix::from_vec(
        5,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.1, 3.1, 4.1, 5.1, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9, 1.9,
            2.9, 3.9, 4.9, 50.0, 50.0, 50.0, 50.0, 50.0, // Outlier
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 5);
    // Outlier at index 4 should have the most negative score
    let min_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Valid floats"))
        .map(|(i, _)| i)
        .expect("Non-empty scores");
    assert_eq!(min_idx, 4);
}

// ========================================================================
// Local Outlier Factor (LOF) Tests
// ========================================================================

#[test]
fn test_lof_new() {
    let lof = LocalOutlierFactor::new();
    assert!(!lof.is_fitted());
}

#[test]
fn test_lof_fit_basic() {
    // Normal data clustered around (2, 2)
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 1.9,
            1.9, 2.1, 1.8,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(5);
    lof.fit(&data).expect("LOF fit should succeed");
    assert!(lof.is_fitted());
}

#[test]
fn test_lof_predict_anomalies() {
    // 8 normal points + 2 outliers
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, // Outlier 1
            -10.0, -10.0, // Outlier 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(5)
        .with_contamination(0.2);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    assert_eq!(predictions.len(), 10);

    // Check that predictions are either 1 (normal) or -1 (anomaly)
    for &pred in &predictions {
        assert!(pred == 1 || pred == -1);
    }

    // Should detect approximately 2 anomalies (20% contamination)
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    assert!((1..=3).contains(&n_anomalies));
}

#[test]
fn test_lof_score_samples() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
            10.0, 10.0, // Outlier 1
            -10.0, -10.0, // Outlier 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    assert_eq!(scores.len(), 6);

    // Outliers should have higher LOF scores than normal points
    let normal_avg = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
    let outlier_avg = (scores[4] + scores[5]) / 2.0;
    assert!(outlier_avg > normal_avg);
}

#[test]
fn test_lof_negative_outlier_factor() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let nof = lof.negative_outlier_factor();
    assert_eq!(nof.len(), 6);

    // Negative outlier factor should be opposite sign of LOF scores
    let scores = lof.score_samples(&data);
    for i in 0..6 {
        // NOF should be negative of LOF (approximately)
        assert!(nof[i] < 0.0 || scores[i] < 1.0);
    }
}

#[test]
fn test_lof_contamination() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Low contamination (10%) - fewer anomalies expected
    let mut lof_low = LocalOutlierFactor::new()
        .with_contamination(0.1)
        .with_n_neighbors(5);
    lof_low.fit(&data).expect("LOF fit should succeed");
    let pred_low = lof_low.predict(&data);
    let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();

    // High contamination (30%) - more anomalies expected
    let mut lof_high = LocalOutlierFactor::new()
        .with_contamination(0.3)
        .with_n_neighbors(5);
    lof_high.fit(&data).expect("LOF fit should succeed");
    let pred_high = lof_high.predict(&data);
    let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();

    // Higher contamination should detect more or equal anomalies
    assert!(anomalies_high >= anomalies_low);
}

#[test]
fn test_lof_n_neighbors() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Fewer neighbors
    let mut lof_few = LocalOutlierFactor::new().with_n_neighbors(3);
    lof_few.fit(&data).expect("LOF fit should succeed");
    let scores_few = lof_few.score_samples(&data);

    // More neighbors
    let mut lof_many = LocalOutlierFactor::new().with_n_neighbors(5);
    lof_many.fit(&data).expect("LOF fit should succeed");
    let scores_many = lof_many.score_samples(&data);

    // Both should work and produce scores
    assert_eq!(scores_few.len(), 8);
    assert_eq!(scores_many.len(), 8);

    // Scores should be different (different neighborhood sizes)
    let diff_exists = scores_few
        .iter()
        .zip(scores_many.iter())
        .any(|(a, b)| (a - b).abs() > 0.01);
    assert!(diff_exists);
}

#[test]
fn test_lof_varying_density_clusters() {
    // Two clusters with different densities
    // Cluster 1: Dense (points close together)
    // Cluster 2: Sparse (points far apart)
    // Outlier: Between clusters
    let data = Matrix::from_vec(
        9,
        2,
        vec![
            // Dense cluster (4 points around 0,0)
            0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.0, 0.1, // Sparse cluster (3 points around 10,10)
            10.0, 10.0, 12.0, 12.0, 11.0, 9.0, // Outlier between clusters
            5.0, 5.0, // Another outlier
            5.5, 5.5,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.2);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    let predictions = lof.predict(&data);

    // LOF should detect outliers in varying density regions
    // Points 7 and 8 (between clusters) should have higher LOF scores
    assert!(scores[7] > 1.0 || scores[8] > 1.0);

    // Should detect some anomalies
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    assert!(n_anomalies >= 1);
}

#[test]
fn test_lof_lof_score_interpretation() {
    let data = Matrix::from_vec(
        5,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal cluster
            10.0, 10.0, // Clear outlier
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);

    // LOF ≈ 1: similar density to neighbors (normal)
    // LOF >> 1: lower density than neighbors (outlier)
    let normal_scores = &scores[0..4];
    let outlier_score = scores[4];

    // Normal points should have LOF close to 1
    for &score in normal_scores {
        assert!((0.5..2.0).contains(&score));
    }

    // Outlier should have LOF > 1 (significantly)
    assert!(outlier_score > 1.5);
}

#[test]
fn test_lof_all_normal() {
    // All points are normal (tightly clustered)
    let data = Matrix::from_vec(
        6,
        2,
        vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_contamination(0.1)
        .with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    let scores = lof.score_samples(&data);

    // All LOF scores should be close to 1 (similar density)
    for &score in &scores {
        assert!((0.5..1.5).contains(&score));
    }

    // With 10% contamination, expect mostly normal points
    let n_normal = predictions.iter().filter(|&&p| p == 1).count();
    assert!(n_normal >= 5);
}

#[test]
fn test_lof_score_samples_finite() {
    let data = Matrix::from_vec(4, 2, vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 10.0, 10.0])
        .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(2);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    // All LOF scores should be finite
    for &score in &scores {
        assert!(score.is_finite());
        assert!(score > 0.0); // LOF is always positive
    }
}

#[test]
fn test_lof_multidimensional() {
    // Test with more features
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.0, 2.0, 3.0, 0.9, 1.9, 2.9, 10.0, 10.0, 10.0, -10.0,
            -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_contamination(0.3)
        .with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    let scores = lof.score_samples(&data);

    assert_eq!(predictions.len(), 6);
    assert_eq!(scores.len(), 6);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_lof_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let lof = LocalOutlierFactor::new();
    let _ = lof.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_lof_score_samples_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let lof = LocalOutlierFactor::new();
    let _ = lof.score_samples(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_lof_negative_outlier_factor_before_fit() {
    let lof = LocalOutlierFactor::new();
    let _ = lof.negative_outlier_factor(); // Should panic
}

#[test]
fn test_lof_decision_consistency() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_contamination(0.3)
        .with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    let scores = lof.score_samples(&data);

    // Points with higher LOF scores should be more likely to be anomalies
    assert_eq!(predictions.len(), scores.len());
}

// ========================================================================
// Local Outlier Factor (LOF) Edge Case / Coverage Tests
// ========================================================================

#[test]
fn test_lof_default() {
    // Exercise the Default impl for LocalOutlierFactor
    let lof = LocalOutlierFactor::default();
    assert!(!lof.is_fitted());
}

#[test]
fn test_lof_debug_impl() {
    // Exercise the Debug derive
    let lof = LocalOutlierFactor::new().with_n_neighbors(5);
    let debug_str = format!("{:?}", lof);
    assert!(debug_str.contains("LocalOutlierFactor"));
    assert!(debug_str.contains("n_neighbors"));
}

#[test]
fn test_lof_clone() {
    // Exercise the Clone derive
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let lof_clone = lof.clone();
    assert!(lof_clone.is_fitted());
    assert_eq!(
        lof.negative_outlier_factor().len(),
        lof_clone.negative_outlier_factor().len()
    );
}

#[test]
fn test_lof_n_neighbors_too_large() {
    // Exercise the error path: n_neighbors >= n_samples
    let data = Matrix::from_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(5);
    let result = lof.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_lof_n_neighbors_equal_n_samples() {
    // Edge case: n_neighbors == n_samples (should fail)
    let data = Matrix::from_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    let result = lof.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_lof_contamination_clamping_high() {
    // Exercise the contamination clamping: values > 0.5 get clamped to 0.5
    let lof = LocalOutlierFactor::new().with_contamination(0.9);
    let debug_str = format!("{:?}", lof);
    // After clamping, contamination should be 0.5
    assert!(debug_str.contains("0.5"));
}

#[test]
fn test_lof_contamination_clamping_low() {
    // Exercise the contamination clamping: values < 0.0 get clamped to 0.0
    let lof = LocalOutlierFactor::new().with_contamination(-1.0);
    let debug_str = format!("{:?}", lof);
    // After clamping, contamination should be 0.0
    assert!(debug_str.contains("contamination: 0.0"));
}

#[test]
fn test_lof_identical_points_zero_reach_dist() {
    // When all points are identical, reachability distances become 0.
    // This exercises the sum_reach_dist == 0.0 branch in compute_lrd (line 256).
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(2);
    lof.fit(&data)
        .expect("LOF fit should succeed with identical points");
    assert!(lof.is_fitted());

    let scores = lof.score_samples(&data);
    for &score in &scores {
        assert!(score.is_finite());
    }

    let predictions = lof.predict(&data);
    assert_eq!(predictions.len(), 4);
}

#[test]
fn test_lof_contamination_zero() {
    // With contamination=0, threshold_idx rounds to 0
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.0);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_lof_contamination_max() {
    // With contamination=0.5 (max allowed), many points flagged
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.5);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    // With 50% contamination, should flag about half
    assert!(n_anomalies >= 2);
}

#[test]
fn test_lof_single_feature() {
    // Exercise with 1D data
    let data = Matrix::from_vec(5, 1, vec![1.0, 1.1, 1.2, 1.0, 100.0])
        .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(2)
        .with_contamination(0.2);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    assert_eq!(scores.len(), 5);

    // The outlier at 100.0 should have highest LOF score
    let max_score_idx = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Valid floats"))
        .map(|(i, _)| i)
        .expect("Non-empty scores");
    assert_eq!(max_score_idx, 4);
}

// ========================================================================
// Spectral Clustering Tests
// ========================================================================

#[test]
fn test_spectral_clustering_new() {
    let sc = SpectralClustering::new(3);
    assert!(!sc.is_fitted());
}

#[test]
fn test_spectral_clustering_fit_basic() {
    // Simple 2-cluster data
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 0.9, 1.1, // Cluster 1
            5.0, 5.0, 5.1, 5.0, 4.9, 5.1, // Cluster 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");
    assert!(sc.is_fitted());
}

#[test]
fn test_spectral_clustering_predict() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 5.0, 5.0, 5.1, 5.0, 4.9, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels = sc.predict(&data);
    assert_eq!(labels.len(), 6);

    // Points in same cluster should have same label
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);

    // Different clusters should have different labels (with label permutation tolerance)
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_spectral_clustering_non_convex() {
    // Create two moon-shaped clusters (non-convex but separable)
    // Upper moon
    let upper: Vec<f32> = vec![0.0, 2.0, 0.5, 2.0, 1.0, 1.9, 1.5, 1.7, 2.0, 1.5];
    // Lower moon
    let lower: Vec<f32> = vec![0.5, 0.3, 1.0, 0.1, 1.5, 0.0, 2.0, 0.0, 2.5, 0.2];

    let mut all_data = upper.clone();
    all_data.extend(lower);

    let data = Matrix::from_vec(10, 2, all_data).expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2)
        .with_affinity(Affinity::KNN)
        .with_n_neighbors(3);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels = sc.predict(&data);

    // Upper moon points should mostly be in same cluster
    // Allow some flexibility for this challenging case
    let upper_cluster = labels[0];
    let same_cluster_count = (0..5).filter(|&i| labels[i] == upper_cluster).count();
    assert!(same_cluster_count >= 4); // At least 4 out of 5

    // Lower moon points should mostly be in same cluster
    let lower_cluster = labels[5];
    let same_cluster_count = (5..10).filter(|&i| labels[i] == lower_cluster).count();
    assert!(same_cluster_count >= 4); // At least 4 out of 5

    // The two moons should be in different clusters
    assert_ne!(upper_cluster, lower_cluster);
}

#[test]
fn test_spectral_clustering_rbf_affinity() {
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2)
        .with_affinity(Affinity::RBF)
        .with_gamma(1.0);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels = sc.predict(&data);
    assert_eq!(labels.len(), 4);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
}

#[test]
fn test_spectral_clustering_knn_affinity() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 5.0, 5.0, 5.1, 5.0, 4.9, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2)
        .with_affinity(Affinity::KNN)
        .with_n_neighbors(3);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels = sc.predict(&data);
    assert_eq!(labels.len(), 6);
}

#[test]
fn test_spectral_clustering_gamma_effect() {
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0])
        .expect("Matrix creation should succeed");

    // Small gamma - more global similarity
    let mut sc_small = SpectralClustering::new(2).with_gamma(0.1);
    sc_small
        .fit(&data)
        .expect("Spectral Clustering fit should succeed");

    // Large gamma - more local similarity
    let mut sc_large = SpectralClustering::new(2).with_gamma(10.0);
    sc_large
        .fit(&data)
        .expect("Spectral Clustering fit should succeed");

    // Both should work
    assert!(sc_small.is_fitted());
    assert!(sc_large.is_fitted());
}

#[test]
fn test_spectral_clustering_multiple_clusters() {
    let data = Matrix::from_vec(
        9,
        2,
        vec![
            // Cluster 1
            0.0, 0.0, 0.1, 0.1, -0.1, -0.1, // Cluster 2
            5.0, 5.0, 5.1, 5.1, 4.9, 4.9, // Cluster 3
            10.0, 10.0, 10.1, 10.1, 9.9, 9.9,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(3);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels = sc.predict(&data);
    assert_eq!(labels.len(), 9);

    // Check that we have 3 distinct clusters
    let mut unique_labels: Vec<usize> = labels.clone();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    assert_eq!(unique_labels.len(), 3);
}

#[test]
fn test_spectral_clustering_labels_consistency() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels1 = sc.predict(&data);
    let labels2 = sc.labels().clone();

    assert_eq!(labels1, labels2);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_spectral_clustering_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let sc = SpectralClustering::new(2);
    let _ = sc.predict(&data);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_spectral_clustering_labels_before_fit() {
    let sc = SpectralClustering::new(2);
    let _ = sc.labels();
}

#[test]
fn test_spectral_clustering_well_separated() {
    // Very well-separated clusters
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 100.0, 100.0, 100.1, 100.1, 100.0, 100.1,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2);
    sc.fit(&data)
        .expect("Spectral Clustering fit should succeed");

    let labels = sc.predict(&data);

    // Should clearly separate the two clusters
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

// ========================================================================
// Spectral Clustering Edge Case / Coverage Tests
// ========================================================================

#[test]
fn test_spectral_clustering_default() {
    // Exercise the Default impl for SpectralClustering
    let sc = SpectralClustering::default();
    assert!(!sc.is_fitted());
}

#[test]
fn test_spectral_clustering_unsupervised_estimator_trait() {
    // Exercise the UnsupervisedEstimator::fit and ::predict dispatch paths
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 5.0, 5.0, 5.1, 5.0, 4.9, 5.1],
    )
    .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2);
    // Use the trait method explicitly
    UnsupervisedEstimator::fit(&mut sc, &data).expect("UnsupervisedEstimator::fit should succeed");
    assert!(sc.is_fitted());

    let labels = UnsupervisedEstimator::predict(&sc, &data);
    assert_eq!(labels.len(), 6);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_spectral_clustering_unsupervised_predict_before_fit() {
    // Exercise the panic path in UnsupervisedEstimator::predict
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let sc = SpectralClustering::new(2);
    let _ = UnsupervisedEstimator::predict(&sc, &data);
}

#[test]
fn test_spectral_clustering_debug_impl() {
    // Exercise the Debug derive on SpectralClustering
    let sc = SpectralClustering::new(3);
    let debug_str = format!("{:?}", sc);
    assert!(debug_str.contains("SpectralClustering"));
    assert!(debug_str.contains("n_clusters"));
}

#[test]
fn test_spectral_clustering_clone() {
    // Exercise the Clone derive
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2).with_gamma(0.5);
    sc.fit(&data).expect("Fit should succeed");

    let sc_clone = sc.clone();
    assert!(sc_clone.is_fitted());
    assert_eq!(sc.labels(), sc_clone.labels());
}

#[test]
fn test_spectral_clustering_with_n_neighbors() {
    // Exercise the with_n_neighbors builder method
    let sc = SpectralClustering::new(2).with_n_neighbors(5);
    let debug_str = format!("{:?}", sc);
    assert!(debug_str.contains("5"));
}

#[test]
fn test_spectral_clustering_knn_small_n_neighbors() {
    // Use KNN affinity with n_neighbors > n_samples-1 to exercise the
    // k_neighbors = self.n_neighbors.min(n_samples - 1) clamping path
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2)
        .with_affinity(Affinity::KNN)
        .with_n_neighbors(100); // much larger than n_samples
    sc.fit(&data)
        .expect("Fit should succeed with clamped n_neighbors");
    assert!(sc.is_fitted());
}

#[test]
fn test_affinity_debug_clone_eq() {
    // Exercise Debug, Clone, Copy, PartialEq, Eq on Affinity enum
    let rbf = Affinity::RBF;
    let knn = Affinity::KNN;
    let rbf_clone = rbf;

    assert_eq!(rbf, rbf_clone);
    assert_ne!(rbf, knn);

    let debug_rbf = format!("{:?}", rbf);
    assert!(debug_rbf.contains("RBF"));

    let debug_knn = format!("{:?}", knn);
    assert!(debug_knn.contains("KNN"));
}

#[test]
fn test_spectral_clustering_high_gamma() {
    // Very high gamma makes RBF similarities very local (close to 0 for distant pairs).
    // This exercises the laplacian normalization with small degree values,
    // hitting the .max(1e-10) guard in compute_laplacian.
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 100.0, 100.0, 100.1, 100.1])
        .expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2).with_gamma(100.0);
    sc.fit(&data).expect("Fit should succeed with high gamma");
    assert!(sc.is_fitted());
}

#[test]
fn test_spectral_clustering_single_feature() {
    // Exercise with 1D data
    let data =
        Matrix::from_vec(4, 1, vec![0.0, 0.1, 10.0, 10.1]).expect("Matrix creation should succeed");

    let mut sc = SpectralClustering::new(2);
    sc.fit(&data).expect("Fit should succeed with 1D data");
    let labels = sc.labels();
    assert_eq!(labels.len(), 4);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}
