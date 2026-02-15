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

include!("tests_part_02.rs");
include!("tests_part_03.rs");
include!("tests_part_04.rs");
