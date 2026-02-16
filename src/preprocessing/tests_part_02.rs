use super::*;

#[test]
fn test_minmax_single_sample() {
    let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    scaler
        .fit(&data)
        .expect("fit should succeed with single sample");

    // With single sample, min = max = value
    let data_min = scaler.data_min();
    let data_max = scaler.data_max();
    assert!((data_min[0] - 5.0).abs() < 1e-6);
    assert!((data_max[0] - 5.0).abs() < 1e-6);

    // Transform should give feature_min (0) since range is 0
    let transformed = scaler.transform(&data).expect("transform should succeed");
    assert!((transformed.get(0, 0)).abs() < 1e-5);
    assert!((transformed.get(0, 1)).abs() < 1e-5);
}

#[test]
fn test_minmax_inverse_with_custom_range() {
    let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let recovered = scaler
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    for i in 0..3 {
        assert!(
            (data.get(i, 0) - recovered.get(i, 0)).abs() < 1e-5,
            "Mismatch at row {i}"
        );
    }
}

// PCA tests
#[test]
fn test_pca_basic_fit_transform() {
    // Simple 2D data that should reduce to 1D along diagonal
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        .expect("valid matrix dimensions");

    let mut pca = PCA::new(1);
    let transformed = pca
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Should reduce to (n_samples, n_components)
    assert_eq!(transformed.shape(), (4, 1));

    // Mean should be centered (approximately)
    let mut sum = 0.0;
    for i in 0..4 {
        sum += transformed.get(i, 0);
    }
    let mean = sum / 4.0;
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {mean}");
}

#[test]
fn test_pca_explained_variance() {
    // Data with known variance structure
    let data = Matrix::from_vec(5, 2, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
        .expect("valid matrix dimensions");

    let mut pca = PCA::new(2);
    pca.fit(&data).expect("fit should succeed with valid data");

    let explained_var = pca
        .explained_variance()
        .expect("explained variance should exist after fit");
    let explained_ratio = pca
        .explained_variance_ratio()
        .expect("explained variance ratio should exist after fit");

    // First component should capture all variance (second column is constant)
    assert_eq!(explained_var.len(), 2);
    assert_eq!(explained_ratio.len(), 2);

    // Ratios should sum to approximately 1.0
    let total_ratio: f32 = explained_ratio.iter().sum();
    assert!(
        (total_ratio - 1.0).abs() < 1e-5,
        "Variance ratios should sum to 1.0, got {total_ratio}"
    );

    // First component should explain most variance
    assert!(
        explained_ratio[0] > 0.99,
        "First component should explain >99% variance"
    );
}

#[test]
fn test_pca_inverse_transform() {
    let data = Matrix::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut pca = PCA::new(2);
    let transformed = pca
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let reconstructed = pca
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    // Reconstruction should be close to original (with some loss since n_components < n_features)
    assert_eq!(reconstructed.shape(), data.shape());

    // Check reconstruction error is reasonable
    let mut total_error = 0.0;
    for i in 0..4 {
        for j in 0..3 {
            let error = (data.get(i, j) - reconstructed.get(i, j)).abs();
            total_error += error * error;
        }
    }
    let mse = total_error / 12.0;
    // With dimensionality reduction, some error is expected
    assert!(mse < 10.0, "Reconstruction MSE too large: {mse}");
}

#[test]
fn test_pca_perfect_reconstruction() {
    // When n_components == n_features, reconstruction should be perfect
    let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");

    let mut pca = PCA::new(2);
    let transformed = pca
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let reconstructed = pca
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    // Perfect reconstruction
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (data.get(i, j) - reconstructed.get(i, j)).abs() < 1e-4,
                "Perfect reconstruction failed at ({}, {}): {} vs {}",
                i,
                j,
                data.get(i, j),
                reconstructed.get(i, j)
            );
        }
    }
}

#[test]
fn test_pca_n_components_exceeds_features() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");

    let mut pca = PCA::new(3); // More components than features
    let result = pca.fit(&data);

    assert!(
        result.is_err(),
        "Should fail when n_components > n_features"
    );
    assert_eq!(
        result.expect_err("Should fail when n_components exceeds features"),
        "n_components cannot exceed number of features"
    );
}

#[test]
fn test_pca_not_fitted_error() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");

    let pca = PCA::new(1);
    let result = pca.transform(&data);

    assert!(result.is_err(), "Should fail when transforming before fit");
    assert_eq!(
        result.expect_err("Should fail when PCA not fitted"),
        "PCA not fitted"
    );
}

#[test]
fn test_pca_dimension_mismatch() {
    let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");
    let test = Matrix::from_vec(3, 3, vec![1.0; 9]).expect("valid matrix dimensions");

    let mut pca = PCA::new(1);
    pca.fit(&train).expect("fit should succeed");

    let result = pca.transform(&test);
    assert!(result.is_err(), "Should fail on dimension mismatch");
    assert_eq!(
        result.expect_err("Should fail with dimension mismatch"),
        "Input has wrong number of features"
    );
}

#[test]
fn test_pca_inverse_dimension_mismatch() {
    let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");
    let wrong_transformed = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("valid matrix dimensions");

    let mut pca = PCA::new(1);
    pca.fit(&train).expect("fit should succeed");

    let result = pca.inverse_transform(&wrong_transformed);
    assert!(
        result.is_err(),
        "Should fail on inverse transform dimension mismatch"
    );
    assert_eq!(
        result.expect_err("Should fail with wrong component count"),
        "Input has wrong number of components"
    );
}

#[test]
fn test_pca_components_shape() {
    let data = Matrix::from_vec(5, 4, vec![1.0; 20]).expect("valid matrix dimensions");

    let mut pca = PCA::new(2);
    pca.fit(&data).expect("fit should succeed with valid data");

    let components = pca.components().expect("components should exist after fit");
    // Components should be (n_components, n_features)
    assert_eq!(components.shape(), (2, 4));
}

#[test]
fn test_pca_variance_preservation() {
    // Property test: total variance should be preserved
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0,
            12.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut pca = PCA::new(3);
    pca.fit(&data).expect("fit should succeed with valid data");

    let explained_var = pca
        .explained_variance()
        .expect("explained variance should exist after fit");

    // Sum of explained variance should be close to total variance
    let total_explained: f32 = explained_var.iter().sum();

    // Calculate actual variance of centered data
    let (n_samples, n_features) = data.shape();
    let mut means = vec![0.0; n_features];
    for (j, mean) in means.iter_mut().enumerate() {
        for i in 0..n_samples {
            *mean += data.get(i, j);
        }
        *mean /= n_samples as f32;
    }

    let mut total_var = 0.0;
    for (j, &mean_j) in means.iter().enumerate() {
        for i in 0..n_samples {
            let diff = data.get(i, j) - mean_j;
            total_var += diff * diff;
        }
    }
    total_var /= (n_samples - 1) as f32;

    // Explained variance should match total variance (with full components)
    assert!(
        (total_explained - total_var).abs() < 1e-3,
        "Total explained variance {total_explained} should match total variance {total_var}"
    );
}

#[test]
fn test_pca_component_orthogonality() {
    // Property test: principal components should be orthogonal
    let data = Matrix::from_vec(
        10,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0, 5.0,
            10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0,
            9.0, 18.0, 27.0, 36.0, 10.0, 20.0, 30.0, 40.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut pca = PCA::new(3);
    pca.fit(&data).expect("fit should succeed with valid data");

    let components = pca.components().expect("components should exist after fit");
    let (_n_components, n_features) = components.shape();

    // Check that all pairs of components are orthogonal (dot product ≈ 0)
    for i in 0..3 {
        for j in (i + 1)..3 {
            let mut dot_product = 0.0;
            for k in 0..n_features {
                dot_product += components.get(i, k) * components.get(j, k);
            }
            assert!(
                dot_product.abs() < 1e-4,
                "Components {i} and {j} should be orthogonal, got dot product {dot_product}"
            );
        }
    }

    // Check that each component is normalized (length ≈ 1)
    for i in 0..3 {
        let mut norm_sq = 0.0;
        for k in 0..n_features {
            let val = components.get(i, k);
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Component {i} should be unit length, got {norm}"
        );
    }
}

// ========================================================================
// t-SNE Tests
// ========================================================================

#[test]
fn test_tsne_new() {
    let tsne = TSNE::new(2);
    assert!(!tsne.is_fitted());
    assert_eq!(tsne.n_components(), 2);
}

#[test]
fn test_tsne_fit_basic() {
    // Simple 2D data, reduce to 2D (should work)
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 10.0, 11.0, 12.0, 10.1,
            11.1, 12.1,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2);
    tsne.fit(&data).expect("fit should succeed with valid data");
    assert!(tsne.is_fitted());
}

#[test]
fn test_tsne_transform() {
    let data = Matrix::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2);
    tsne.fit(&data).expect("fit should succeed with valid data");

    let transformed = tsne
        .transform(&data)
        .expect("transform should succeed after fit");
    assert_eq!(transformed.shape(), (4, 2));
}

#[test]
fn test_tsne_fit_transform() {
    let data = Matrix::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
        ],
    )
    .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2);
    let transformed = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(transformed.shape(), (4, 2));
    assert!(tsne.is_fitted());
}

#[test]
fn test_tsne_perplexity() {
    let data = Matrix::from_vec(
        10,
        3,
        vec![
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 5.2, 6.2,
            7.2, 10.0, 11.0, 12.0, 10.1, 11.1, 12.1, 10.2, 11.2, 12.2, 10.3, 11.3, 12.3,
        ],
    )
    .expect("valid matrix dimensions");

    // Low perplexity (more local)
    let mut tsne_low = TSNE::new(2).with_perplexity(2.0);
    let result_low = tsne_low
        .fit_transform(&data)
        .expect("fit_transform should succeed with low perplexity");
    assert_eq!(result_low.shape(), (10, 2));

    // High perplexity (more global)
    let mut tsne_high = TSNE::new(2).with_perplexity(5.0);
    let result_high = tsne_high
        .fit_transform(&data)
        .expect("fit_transform should succeed with high perplexity");
    assert_eq!(result_high.shape(), (10, 2));
}
