use super::*;

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
