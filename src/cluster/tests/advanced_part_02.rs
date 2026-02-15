
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
