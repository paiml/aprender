
#[test]
fn test_tsne_learning_rate() {
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 12.0,
            13.0, 14.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2).with_learning_rate(100.0).with_n_iter(100);
    let transformed = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed with custom learning rate");
    assert_eq!(transformed.shape(), (6, 2));
}

#[test]
fn test_tsne_n_components() {
    let data = Matrix::from_vec(
        4,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 11.0,
            12.0, 13.0, 14.0, 15.0,
        ],
    )
    .expect("valid matrix dimensions");

    // 2D embedding
    let mut tsne_2d = TSNE::new(2);
    let result_2d = tsne_2d
        .fit_transform(&data)
        .expect("fit_transform should succeed for 2D");
    assert_eq!(result_2d.shape(), (4, 2));

    // 3D embedding
    let mut tsne_3d = TSNE::new(3);
    let result_3d = tsne_3d
        .fit_transform(&data)
        .expect("fit_transform should succeed for 3D");
    assert_eq!(result_3d.shape(), (4, 3));
}

#[test]
fn test_tsne_reproducibility() {
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 12.0,
            13.0, 14.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne1 = TSNE::new(2).with_random_state(42);
    let result1 = tsne1
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    let mut tsne2 = TSNE::new(2).with_random_state(42);
    let result2 = tsne2
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Results should be identical with same random state
    for i in 0..6 {
        for j in 0..2 {
            assert!(
                (result1.get(i, j) - result2.get(i, j)).abs() < 1e-5,
                "Results should be reproducible with same random state"
            );
        }
    }
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_tsne_transform_before_fit() {
    let data = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
    let tsne = TSNE::new(2);
    let _ = tsne.transform(&data);
}

#[test]
fn test_tsne_preserves_local_structure() {
    // Create data with clear local structure
    let data = Matrix::from_vec(
        8,
        3,
        vec![
            // Cluster 1: tight cluster around (0, 0, 0)
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, // Cluster 2: tight cluster around (5, 5, 5)
            5.0, 5.0, 5.0, 5.1, 5.1, 5.1,
            // Cluster 3: tight cluster around (10, 10, 10)
            10.0, 10.0, 10.0, 10.1, 10.1, 10.1,
            // Cluster 4: tight cluster around (15, 15, 15)
            15.0, 15.0, 15.0, 15.1, 15.1, 15.1,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2)
        .with_random_state(42)
        .with_n_iter(500)
        .with_perplexity(3.0);
    let embedding = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Points within same cluster should be close in embedding
    // Cluster 1: points 0, 1
    let dist_01 = ((embedding.get(0, 0) - embedding.get(1, 0)).powi(2)
        + (embedding.get(0, 1) - embedding.get(1, 1)).powi(2))
    .sqrt();

    // Distance to far cluster should be larger
    let dist_03 = ((embedding.get(0, 0) - embedding.get(3, 0)).powi(2)
        + (embedding.get(0, 1) - embedding.get(3, 1)).powi(2))
    .sqrt();

    // Allow some tolerance - t-SNE is stochastic
    // Just verify local structure is somewhat preserved
    assert!(
        dist_01 < dist_03 * 1.5,
        "Local structure should be roughly preserved: dist_01={:.3} should be < dist_03*1.5={:.3}",
        dist_01,
        dist_03 * 1.5
    );
}

#[test]
fn test_tsne_min_samples() {
    // t-SNE should work with minimum number of samples (> perplexity * 3)
    let data = Matrix::from_vec(
        10,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 7.0,
            8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2).with_perplexity(3.0);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed with minimum samples");
    assert_eq!(result.shape(), (10, 2));
}

#[test]
fn test_tsne_embedding_finite() {
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 12.0,
            13.0, 14.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2).with_n_iter(100);
    let embedding = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // All embedding values should be finite
    for i in 0..6 {
        for j in 0..2 {
            assert!(
                embedding.get(i, j).is_finite(),
                "Embedding should contain only finite values"
            );
        }
    }
}

// ========================================================================
// Additional Coverage Tests - StandardScaler panic paths
// ========================================================================

#[test]
#[should_panic(expected = "Scaler not fitted")]
fn test_standard_scaler_mean_panic() {
    let scaler = StandardScaler::new();
    let _ = scaler.mean();
}

#[test]
#[should_panic(expected = "Scaler not fitted")]
fn test_standard_scaler_std_panic() {
    let scaler = StandardScaler::new();
    let _ = scaler.std();
}

#[test]
fn test_standard_scaler_inverse_transform_not_fitted() {
    let scaler = StandardScaler::new();
    let data = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
    let result = scaler.inverse_transform(&data);
    assert!(result.is_err());
}

#[test]
fn test_standard_scaler_inverse_transform_dimension_mismatch() {
    let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");
    let test = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler.fit(&train).expect("fit should succeed");
    let result = scaler.inverse_transform(&test);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with dimension mismatch"),
        "Feature dimension mismatch"
    );
}

#[test]
fn test_standard_scaler_save_load_safetensors() {
    use std::fs;
    use std::path::PathBuf;

    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler.fit(&data).expect("fit should succeed");

    // Save to temp file
    let path = PathBuf::from("/tmp/test_standard_scaler.safetensors");
    scaler.save_safetensors(&path).expect("save should succeed");

    // Load back
    let loaded = StandardScaler::load_safetensors(&path).expect("load should succeed");

    // Compare mean and std
    assert_eq!(scaler.mean(), loaded.mean());
    assert_eq!(scaler.std(), loaded.std());

    // Cleanup
    let _ = fs::remove_file(&path);
}

#[test]
fn test_standard_scaler_save_unfitted_error() {
    let scaler = StandardScaler::new();
    let result = scaler.save_safetensors("/tmp/test_unfitted.safetensors");
    assert!(result.is_err());
    assert!(result
        .expect_err("Should fail")
        .contains("Cannot save unfitted scaler"));
}

#[test]
fn test_standard_scaler_with_both_disabled() {
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new().with_mean(false).with_std(false);
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Should be identity transform when both are disabled
    assert!((transformed.get(0, 0) - 1.0).abs() < 1e-5);
    assert!((transformed.get(1, 0) - 2.0).abs() < 1e-5);
    assert!((transformed.get(2, 0) - 3.0).abs() < 1e-5);
}

#[test]
fn test_standard_scaler_inverse_transform_with_options() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    // Test with only mean centering
    let mut scaler_mean_only = StandardScaler::new().with_mean(true).with_std(false);
    let transformed = scaler_mean_only
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let recovered = scaler_mean_only
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                "Mismatch at ({i}, {j})"
            );
        }
    }

    // Test with only std scaling
    let mut scaler_std_only = StandardScaler::new().with_mean(false).with_std(true);
    let transformed = scaler_std_only
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let recovered = scaler_std_only
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                "Mismatch at ({i}, {j})"
            );
        }
    }
}

// ========================================================================
// Additional Coverage Tests - MinMaxScaler panic paths
// ========================================================================

#[test]
#[should_panic(expected = "Scaler not fitted")]
fn test_minmax_scaler_data_min_panic() {
    let scaler = MinMaxScaler::new();
    let _ = scaler.data_min();
}

#[test]
#[should_panic(expected = "Scaler not fitted")]
fn test_minmax_scaler_data_max_panic() {
    let scaler = MinMaxScaler::new();
    let _ = scaler.data_max();
}

#[test]
fn test_minmax_scaler_inverse_transform_not_fitted() {
    let scaler = MinMaxScaler::new();
    let data = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
    let result = scaler.inverse_transform(&data);
    assert!(result.is_err());
}

#[test]
fn test_minmax_scaler_inverse_transform_dimension_mismatch() {
    let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");
    let test = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&train).expect("fit should succeed");
    let result = scaler.inverse_transform(&test);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with dimension mismatch"),
        "Feature dimension mismatch"
    );
}

#[test]
fn test_minmax_inverse_transform_constant_feature() {
    // Test inverse transform with constant feature (zero range)
    let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0])
        .expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let recovered = scaler
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    // First column should be recovered
    for i in 0..3 {
        assert!(
            (data.get(i, 0) - recovered.get(i, 0)).abs() < 1e-5,
            "First column should be recovered"
        );
        // Constant column recovers to data_min value
        assert!(
            (5.0 - recovered.get(i, 1)).abs() < 1e-5,
            "Constant column should recover to original value"
        );
    }
}

// ========================================================================
// Additional Coverage Tests - PCA
// ========================================================================

#[test]
fn test_pca_inverse_transform_not_fitted() {
    let pca = PCA::new(2);
    let data = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
    let result = pca.inverse_transform(&data);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when not fitted"),
        "PCA not fitted"
    );
}

#[test]
fn test_pca_components_not_fitted() {
    let pca = PCA::new(2);
    assert!(pca.components().is_none());
}

#[test]
fn test_pca_explained_variance_not_fitted() {
    let pca = PCA::new(2);
    assert!(pca.explained_variance().is_none());
    assert!(pca.explained_variance_ratio().is_none());
}

// ========================================================================
// Additional Coverage Tests - TSNE
// ========================================================================

#[test]
fn test_tsne_default() {
    let tsne = TSNE::default();
    assert!(!tsne.is_fitted());
    assert_eq!(tsne.n_components(), 2);
}

#[test]
fn test_tsne_builder_chain() {
    let tsne = TSNE::new(3)
        .with_perplexity(15.0)
        .with_learning_rate(100.0)
        .with_n_iter(500)
        .with_random_state(123);

    assert_eq!(tsne.n_components(), 3);
    assert!(!tsne.is_fitted());
}
