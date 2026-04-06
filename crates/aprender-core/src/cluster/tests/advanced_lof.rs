
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
