
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
