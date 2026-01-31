//! Tests for preprocessing module.

use super::*;

#[test]
fn test_new() {
    let scaler = StandardScaler::new();
    assert!(!scaler.is_fitted());
}

#[test]
fn test_default() {
    let scaler = StandardScaler::default();
    assert!(!scaler.is_fitted());
}

#[test]
fn test_fit_basic() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler
        .fit(&data)
        .expect("fit should succeed with valid data");

    assert!(scaler.is_fitted());

    // Mean should be [2.0, 20.0]
    let mean = scaler.mean();
    assert!((mean[0] - 2.0).abs() < 1e-6);
    assert!((mean[1] - 20.0).abs() < 1e-6);

    // Std should be sqrt(2/3) ≈ 0.8165
    let std = scaler.std();
    let expected_std = (2.0_f32 / 3.0).sqrt();
    assert!((std[0] - expected_std).abs() < 1e-4);
    assert!((std[1] - expected_std * 10.0).abs() < 1e-3);
}

#[test]
fn test_transform_basic() {
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler
        .fit(&data)
        .expect("fit should succeed with valid data");

    let transformed = scaler
        .transform(&data)
        .expect("transform should succeed after fit");

    // Mean should be 0
    let mean: f32 = (0..3).map(|i| transformed.get(i, 0)).sum::<f32>() / 3.0;
    assert!(mean.abs() < 1e-6, "Mean should be ~0, got {mean}");

    // Std should be 1
    let variance: f32 = (0..3)
        .map(|i| {
            let v = transformed.get(i, 0);
            v * v
        })
        .sum::<f32>()
        / 3.0;
    assert!(
        (variance.sqrt() - 1.0).abs() < 1e-6,
        "Std should be ~1, got {}",
        variance.sqrt()
    );
}

#[test]
fn test_fit_transform() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed with valid data");

    // Check each column has mean ≈ 0
    for j in 0..2 {
        let mean: f32 = (0..4).map(|i| transformed.get(i, j)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "Column {j} mean should be ~0");
    }
}

#[test]
fn test_inverse_transform() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let recovered = scaler
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    // Should recover original data
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                "Mismatch at ({i}, {j})"
            );
        }
    }
}

#[test]
fn test_transform_new_data() {
    let train = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");
    let test = Matrix::from_vec(2, 1, vec![4.0, 5.0]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler
        .fit(&train)
        .expect("fit should succeed with valid data");

    let transformed = scaler
        .transform(&test)
        .expect("transform should succeed with new data");

    // Test data should be transformed using train stats
    // mean=2, std=sqrt(2/3)
    let mean = 2.0;
    let std = (2.0_f32 / 3.0).sqrt();

    let expected_0 = (4.0 - mean) / std;
    let expected_1 = (5.0 - mean) / std;

    assert!((transformed.get(0, 0) - expected_0).abs() < 1e-5);
    assert!((transformed.get(1, 0) - expected_1).abs() < 1e-5);
}

#[test]
fn test_without_mean() {
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new().with_mean(false);
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Should only scale, not center
    // Original values divided by std
    let std = (2.0_f32 / 3.0).sqrt();
    assert!((transformed.get(0, 0) - 1.0 / std).abs() < 1e-5);
    assert!((transformed.get(1, 0) - 2.0 / std).abs() < 1e-5);
    assert!((transformed.get(2, 0) - 3.0 / std).abs() < 1e-5);
}

#[test]
fn test_without_std() {
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new().with_std(false);
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Should only center, not scale
    // mean = 2.0
    assert!((transformed.get(0, 0) - (-1.0)).abs() < 1e-5);
    assert!((transformed.get(1, 0) - 0.0).abs() < 1e-5);
    assert!((transformed.get(2, 0) - 1.0).abs() < 1e-5);
}

#[test]
fn test_constant_feature() {
    // Feature with zero variance
    let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0])
        .expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Second column has zero std, should remain centered but not scaled
    assert!((transformed.get(0, 1) - 0.0).abs() < 1e-5);
    assert!((transformed.get(1, 1) - 0.0).abs() < 1e-5);
    assert!((transformed.get(2, 1) - 0.0).abs() < 1e-5);
}

#[test]
fn test_empty_data_error() {
    let data = Matrix::from_vec(0, 2, vec![]).expect("empty matrix should be valid");
    let mut scaler = StandardScaler::new();
    assert!(scaler.fit(&data).is_err());
}

#[test]
fn test_transform_not_fitted_error() {
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");
    let scaler = StandardScaler::new();
    assert!(scaler.transform(&data).is_err());
}

#[test]
fn test_dimension_mismatch_error() {
    let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");
    let test = Matrix::from_vec(3, 3, vec![1.0; 9]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler.fit(&train).expect("fit should succeed");

    assert!(scaler.transform(&test).is_err());
}

#[test]
fn test_single_sample() {
    let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).expect("valid matrix dimensions");

    let mut scaler = StandardScaler::new();
    scaler
        .fit(&data)
        .expect("fit should succeed with single sample");

    // With single sample, std is 0
    let std = scaler.std();
    assert!((std[0]).abs() < 1e-6);
    assert!((std[1]).abs() < 1e-6);

    // Transform should center only (std is 0, no scaling)
    let transformed = scaler.transform(&data).expect("transform should succeed");
    assert!((transformed.get(0, 0)).abs() < 1e-5);
    assert!((transformed.get(0, 1)).abs() < 1e-5);
}

#[test]
fn test_builder_chain() {
    let scaler = StandardScaler::new().with_mean(false).with_std(true);

    let data = Matrix::from_vec(2, 1, vec![2.0, 4.0]).expect("valid matrix dimensions");
    let mut scaler = scaler;
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Only scaling, no centering
    // Values: 2, 4; mean=3; std=1
    // Without centering: 2/1=2, 4/1=4
    assert!(transformed.get(0, 0) > 0.0, "Should not be centered");
}

// MinMaxScaler tests
#[test]
fn test_minmax_new() {
    let scaler = MinMaxScaler::new();
    assert!(!scaler.is_fitted());
}

#[test]
fn test_minmax_default() {
    let scaler = MinMaxScaler::default();
    assert!(!scaler.is_fitted());
}

#[test]
fn test_minmax_fit_basic() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    scaler
        .fit(&data)
        .expect("fit should succeed with valid data");

    assert!(scaler.is_fitted());

    // Min should be [1.0, 10.0], max should be [3.0, 30.0]
    let data_min = scaler.data_min();
    let data_max = scaler.data_max();
    assert!((data_min[0] - 1.0).abs() < 1e-6);
    assert!((data_min[1] - 10.0).abs() < 1e-6);
    assert!((data_max[0] - 3.0).abs() < 1e-6);
    assert!((data_max[1] - 30.0).abs() < 1e-6);
}

#[test]
fn test_minmax_transform_basic() {
    let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    scaler
        .fit(&data)
        .expect("fit should succeed with valid data");

    let transformed = scaler
        .transform(&data)
        .expect("transform should succeed after fit");

    // Should scale to [0, 1]
    assert!((transformed.get(0, 0) - 0.0).abs() < 1e-6);
    assert!((transformed.get(1, 0) - 0.5).abs() < 1e-6);
    assert!((transformed.get(2, 0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_minmax_fit_transform() {
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0, 30.0, 300.0])
        .expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed with valid data");

    // Check min is 0 and max is 1 for each column
    for j in 0..2 {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..4 {
            let val = transformed.get(i, j);
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
        assert!(min_val.abs() < 1e-5, "Column {j} min should be ~0");
        assert!((max_val - 1.0).abs() < 1e-5, "Column {j} max should be ~1");
    }
}

#[test]
fn test_minmax_inverse_transform() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    let recovered = scaler
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");

    // Should recover original data
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                "Mismatch at ({i}, {j})"
            );
        }
    }
}

#[test]
fn test_minmax_transform_new_data() {
    let train = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");
    let test = Matrix::from_vec(2, 1, vec![15.0, -5.0]).expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    scaler
        .fit(&train)
        .expect("fit should succeed with valid data");

    let transformed = scaler
        .transform(&test)
        .expect("transform should succeed with new data");

    // 15 should map to 1.5 (beyond training range)
    // -5 should map to -0.5 (below training range)
    assert!((transformed.get(0, 0) - 1.5).abs() < 1e-5);
    assert!((transformed.get(1, 0) - (-0.5)).abs() < 1e-5);
}

#[test]
fn test_minmax_custom_range() {
    let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Should scale to [-1, 1]
    assert!((transformed.get(0, 0) - (-1.0)).abs() < 1e-6);
    assert!((transformed.get(1, 0) - 0.0).abs() < 1e-6);
    assert!((transformed.get(2, 0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_minmax_constant_feature() {
    // Feature with same min and max
    let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0])
        .expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    let transformed = scaler
        .fit_transform(&data)
        .expect("fit_transform should succeed");

    // Second column is constant, should become feature_min (0)
    assert!((transformed.get(0, 1) - 0.0).abs() < 1e-5);
    assert!((transformed.get(1, 1) - 0.0).abs() < 1e-5);
    assert!((transformed.get(2, 1) - 0.0).abs() < 1e-5);
}

#[test]
fn test_minmax_empty_data_error() {
    let data = Matrix::from_vec(0, 2, vec![]).expect("empty matrix should be valid");
    let mut scaler = MinMaxScaler::new();
    assert!(scaler.fit(&data).is_err());
}

#[test]
fn test_minmax_transform_not_fitted_error() {
    let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");
    let scaler = MinMaxScaler::new();
    assert!(scaler.transform(&data).is_err());
}

#[test]
fn test_minmax_dimension_mismatch_error() {
    let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid matrix dimensions");
    let test = Matrix::from_vec(3, 3, vec![1.0; 9]).expect("valid matrix dimensions");

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&train).expect("fit should succeed");

    assert!(scaler.transform(&test).is_err());
}

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

#[test]
fn test_tsne_very_short_iterations() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 11.0, 12.0])
        .expect("valid matrix dimensions");

    // Test with very few iterations (to hit early iteration paths)
    let mut tsne = TSNE::new(2).with_n_iter(10).with_random_state(42);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (4, 2));
}

#[test]
fn test_tsne_high_iterations() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 11.0, 12.0])
        .expect("valid matrix dimensions");

    // Test with iterations past momentum switch (250+)
    let mut tsne = TSNE::new(2).with_n_iter(300).with_random_state(42);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (4, 2));
}

#[test]
fn test_tsne_single_component() {
    let data = Matrix::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
        ],
    )
    .expect("valid matrix dimensions");

    // Reduce to 1D
    let mut tsne = TSNE::new(1).with_n_iter(100).with_random_state(42);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (4, 1));
}

#[test]
fn test_tsne_very_low_perplexity() {
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 5.0, 6.0, 7.0, 5.5, 6.5, 7.5, 10.0, 11.0, 12.0, 10.5,
            11.5, 12.5,
        ],
    )
    .expect("valid matrix dimensions");

    // Very low perplexity to test binary search extremes
    let mut tsne = TSNE::new(2)
        .with_perplexity(1.5)
        .with_n_iter(50)
        .with_random_state(42);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (6, 2));
}

#[test]
fn test_tsne_identical_points() {
    // Test with nearly identical points (tests numerical stability)
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2).with_n_iter(50).with_random_state(42);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (4, 2));

    // Results should be finite
    for i in 0..4 {
        for j in 0..2 {
            assert!(result.get(i, j).is_finite());
        }
    }
}

#[test]
fn test_tsne_large_values() {
    // Test with large values (checks numerical stability)
    let data = Matrix::from_vec(4, 2, vec![1e6, 2e6, 1.1e6, 2.1e6, 5e6, 6e6, 5.1e6, 6.1e6])
        .expect("valid matrix dimensions");

    let mut tsne = TSNE::new(2).with_n_iter(50).with_random_state(42);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (4, 2));

    // Results should be finite
    for i in 0..4 {
        for j in 0..2 {
            assert!(result.get(i, j).is_finite());
        }
    }
}

#[test]
fn test_tsne_without_random_state() {
    let data = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 11.0, 12.0])
        .expect("valid matrix dimensions");

    // Without random state, should use time-based seed
    let mut tsne = TSNE::new(2).with_n_iter(10);
    let result = tsne
        .fit_transform(&data)
        .expect("fit_transform should succeed");
    assert_eq!(result.shape(), (4, 2));
}

// ========================================================================
// StandardScaler save/load edge cases
// ========================================================================

#[test]
fn test_standard_scaler_save_with_options() {
    use std::fs;
    use std::path::PathBuf;

    let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("valid matrix dimensions");

    // Test with mean=false, std=true
    let mut scaler = StandardScaler::new().with_mean(false).with_std(true);
    scaler.fit(&data).expect("fit should succeed");

    let path = PathBuf::from("/tmp/test_standard_scaler_options.safetensors");
    scaler.save_safetensors(&path).expect("save should succeed");

    let loaded = StandardScaler::load_safetensors(&path).expect("load should succeed");

    // Verify transform behavior matches
    let test_data = Matrix::from_vec(1, 2, vec![2.0, 20.0]).expect("valid matrix dimensions");
    let orig_transformed = scaler
        .transform(&test_data)
        .expect("transform should succeed");
    let loaded_transformed = loaded
        .transform(&test_data)
        .expect("transform should succeed");

    assert!((orig_transformed.get(0, 0) - loaded_transformed.get(0, 0)).abs() < 1e-5);
    assert!((orig_transformed.get(0, 1) - loaded_transformed.get(0, 1)).abs() < 1e-5);

    let _ = fs::remove_file(&path);
}
