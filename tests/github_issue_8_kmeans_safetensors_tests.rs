// GitHub Issue #8: Complete SafeTensors Model Serialization
// Tests for KMeans SafeTensors export functionality
//
// Acceptance Criteria:
// - KMeans::save_safetensors() exports valid SafeTensors format
// - SafeTensors includes centroids matrix, n_clusters, max_iter, tol, random_state, inertia, n_iter
// - Roundtrip: save → load produces identical predictions
// - Centroid positions preserved exactly

use aprender::cluster::KMeans;
use aprender::primitives::Matrix;
use aprender::traits::UnsupervisedEstimator;
use std::fs;
use std::path::Path;

#[test]
fn test_kmeans_save_safetensors_creates_file() {
    // Train a simple KMeans model
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 9.0, 11.0, 9.5, 10.0],
    )
    .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2);
    kmeans.fit(&x).expect("Test data should be valid");

    // Save to SafeTensors format
    let path = "test_kmeans_model.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");

    // Verify file was created
    assert!(
        Path::new(path).exists(),
        "SafeTensors file should be created"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_save_load_roundtrip() {
    // Train KMeans model
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 9.0, 11.0, 9.5, 10.0],
    )
    .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2).with_max_iter(300).with_tol(1e-4);
    kmeans.fit(&x).expect("Test data should be valid");

    // Get original predictions
    let pred_original = kmeans.predict(&x);

    // Save and load
    let path = "test_kmeans_roundtrip.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_kmeans = KMeans::load_safetensors(path).expect("Test data should be valid");

    // Get loaded model predictions
    let pred_loaded = loaded_kmeans.predict(&x);

    // Verify predictions match exactly
    assert_eq!(
        pred_original, pred_loaded,
        "Predictions should be identical after roundtrip"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_safetensors_metadata_includes_all_tensors() {
    // Create and fit KMeans model
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 1.5, 1.8, 8.0, 8.0, 9.0, 11.0])
        .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2)
        .with_max_iter(500)
        .with_tol(1e-5)
        .with_random_state(42);
    kmeans.fit(&x).expect("Test data should be valid");

    let path = "test_kmeans_metadata.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");

    let bytes = fs::read(path).expect("Test data should be valid");

    // Extract metadata
    let header_bytes: [u8; 8] = bytes[0..8].try_into().expect("Test data should be valid");
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json).expect("Test data should be valid");

    // Parse JSON
    let metadata: serde_json::Value =
        serde_json::from_str(metadata_str).expect("Test data should be valid");

    // Verify all required tensors exist
    assert!(
        metadata.get("centroids").is_some(),
        "Metadata must include 'centroids' tensor"
    );
    assert!(
        metadata.get("n_clusters").is_some(),
        "Metadata must include 'n_clusters' tensor"
    );
    assert!(
        metadata.get("max_iter").is_some(),
        "Metadata must include 'max_iter' tensor"
    );
    assert!(
        metadata.get("tol").is_some(),
        "Metadata must include 'tol' tensor"
    );
    assert!(
        metadata.get("random_state").is_some(),
        "Metadata must include 'random_state' tensor"
    );
    assert!(
        metadata.get("inertia").is_some(),
        "Metadata must include 'inertia' tensor"
    );
    assert!(
        metadata.get("n_iter").is_some(),
        "Metadata must include 'n_iter' tensor"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_save_unfitted_model_fails() {
    // Unfitted model should not be saveable
    let kmeans = KMeans::new(2);

    let path = "test_unfitted_kmeans.safetensors";
    let result = kmeans.save_safetensors(path);

    assert!(
        result.is_err(),
        "Saving unfitted model should return an error"
    );
    let error_msg = result.expect_err("Expected error in test");
    assert!(
        error_msg.contains("unfitted") || error_msg.contains("fit"),
        "Error message should mention model is unfitted. Got: {error_msg}"
    );

    // Ensure no file was created
    assert!(
        !Path::new(path).exists(),
        "No file should be created for unfitted model"
    );
}

#[test]
fn test_kmeans_load_nonexistent_file_fails() {
    let result = KMeans::load_safetensors("nonexistent_kmeans_file.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_kmeans_load_corrupted_metadata_fails() {
    // Create a file with invalid SafeTensors format
    let path = "test_corrupted_kmeans.safetensors";
    fs::write(path, b"invalid safetensors data").expect("Test data should be valid");

    let result = KMeans::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return an error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_centroids_preserved() {
    // Verify centroids are exactly preserved through save/load
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 9.0, 11.0, 9.5, 10.0],
    )
    .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2);
    kmeans.fit(&x).expect("Test data should be valid");

    let original_centroids = kmeans.centroids();
    let (n_clusters, n_features) = original_centroids.shape();

    // Save and load
    let path = "test_kmeans_centroids.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_kmeans = KMeans::load_safetensors(path).expect("Test data should be valid");

    let loaded_centroids = loaded_kmeans.centroids();

    // Verify shapes match
    assert_eq!(loaded_centroids.shape(), (n_clusters, n_features));

    // Verify all centroid values match
    for i in 0..n_clusters {
        for j in 0..n_features {
            let orig = original_centroids.get(i, j);
            let loaded = loaded_centroids.get(i, j);
            assert!(
                (orig - loaded).abs() < 1e-6,
                "Centroid[{i},{j}] mismatch: original={orig}, loaded={loaded}"
            );
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_hyperparameters_preserved() {
    // Verify hyperparameters are preserved
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 1.5, 1.8, 8.0, 8.0, 9.0, 11.0])
        .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2)
        .with_max_iter(500)
        .with_tol(1e-5)
        .with_random_state(42);
    kmeans.fit(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_kmeans_hyperparams.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_kmeans = KMeans::load_safetensors(path).expect("Test data should be valid");

    // Verify hyperparameters match
    // Note: We can't directly access private fields, so we verify via behavior
    let pred_orig = kmeans.predict(&x);
    let pred_loaded = loaded_kmeans.predict(&x);
    assert_eq!(pred_orig, pred_loaded, "Predictions should match");

    // Verify inertia preserved
    let orig_inertia = kmeans.inertia();
    let loaded_inertia = loaded_kmeans.inertia();
    assert!(
        (orig_inertia - loaded_inertia).abs() < 1e-6,
        "Inertia mismatch"
    );

    // Verify n_iter preserved
    assert_eq!(kmeans.n_iter(), loaded_kmeans.n_iter(), "n_iter mismatch");

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_multiple_save_load_cycles() {
    // Property test: Multiple save/load cycles should be idempotent
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 9.0, 11.0, 9.5, 10.0],
    )
    .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&x).expect("Test data should be valid");

    let path1 = "test_kmeans_cycle1.safetensors";
    let path2 = "test_kmeans_cycle2.safetensors";
    let path3 = "test_kmeans_cycle3.safetensors";

    // Cycle 1: original → save → load
    kmeans
        .save_safetensors(path1)
        .expect("Test data should be valid");
    let kmeans1 = KMeans::load_safetensors(path1).expect("Test data should be valid");

    // Cycle 2: loaded → save → load
    kmeans1
        .save_safetensors(path2)
        .expect("Test data should be valid");
    let kmeans2 = KMeans::load_safetensors(path2).expect("Test data should be valid");

    // Cycle 3: loaded again → save → load
    kmeans2
        .save_safetensors(path3)
        .expect("Test data should be valid");
    let kmeans3 = KMeans::load_safetensors(path3).expect("Test data should be valid");

    // All models should produce identical predictions
    let pred_orig = kmeans.predict(&x);
    let pred1 = kmeans1.predict(&x);
    let pred2 = kmeans2.predict(&x);
    let pred3 = kmeans3.predict(&x);

    assert_eq!(pred_orig, pred1, "Cycle 1 predictions mismatch");
    assert_eq!(pred_orig, pred2, "Cycle 2 predictions mismatch");
    assert_eq!(pred_orig, pred3, "Cycle 3 predictions mismatch");

    // Cleanup
    fs::remove_file(path1).ok();
    fs::remove_file(path2).ok();
    fs::remove_file(path3).ok();
}

#[test]
fn test_kmeans_different_n_clusters() {
    // Test models with different numbers of clusters
    let x = Matrix::from_vec(
        9,
        2,
        vec![
            1.0, 2.0, 1.5, 1.8, 2.0, 2.5, 5.0, 8.0, 6.0, 9.0, 5.5, 8.5, 9.0, 11.0, 10.0, 12.0, 9.5,
            10.5,
        ],
    )
    .expect("Test data should be valid");

    // Model with 2 clusters
    let mut kmeans2 = KMeans::new(2).with_random_state(42);
    kmeans2.fit(&x).expect("Test data should be valid");

    // Model with 3 clusters
    let mut kmeans3 = KMeans::new(3).with_random_state(42);
    kmeans3.fit(&x).expect("Test data should be valid");

    // Save both models
    let path2 = "test_kmeans_2clusters.safetensors";
    let path3 = "test_kmeans_3clusters.safetensors";

    kmeans2
        .save_safetensors(path2)
        .expect("Test data should be valid");
    kmeans3
        .save_safetensors(path3)
        .expect("Test data should be valid");

    // Load and verify predictions match
    let loaded2 = KMeans::load_safetensors(path2).expect("Test data should be valid");
    let loaded3 = KMeans::load_safetensors(path3).expect("Test data should be valid");

    let pred2_orig = kmeans2.predict(&x);
    let pred2_loaded = loaded2.predict(&x);
    assert_eq!(
        pred2_orig, pred2_loaded,
        "2-cluster model predictions mismatch"
    );

    let pred3_orig = kmeans3.predict(&x);
    let pred3_loaded = loaded3.predict(&x);
    assert_eq!(
        pred3_orig, pred3_loaded,
        "3-cluster model predictions mismatch"
    );

    // Cleanup
    fs::remove_file(path2).ok();
    fs::remove_file(path3).ok();
}

#[test]
fn test_kmeans_file_size_reasonable() {
    // Verify SafeTensors file size is reasonable
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 1.5, 1.8, 8.0, 8.0, 9.0, 11.0])
        .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2);
    kmeans.fit(&x).expect("Test data should be valid");

    let path = "test_kmeans_size.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");

    let file_size = fs::metadata(path).expect("Test data should be valid").len();

    // File should be reasonable:
    // - Metadata: < 1KB
    // - Centroids: 2 clusters × 2 features × 4 bytes = 16 bytes
    // - Hyperparameters: 6 × 4 bytes = 24 bytes
    // - Total: < 2KB for this small model
    assert!(
        file_size < 2048,
        "SafeTensors file should be compact. Got {file_size} bytes"
    );

    assert!(
        file_size > 40,
        "SafeTensors file should contain data. Got {file_size} bytes"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_high_dimensional_data() {
    // Test with higher-dimensional data
    let x = Matrix::from_vec(
        6,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0,
            10.0, 11.0, 12.0, 8.5, 9.5, 10.5, 11.5, 12.5, 9.0, 10.0, 11.0, 12.0, 13.0,
        ],
    )
    .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2).with_random_state(123);
    kmeans.fit(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_kmeans_highdim.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_kmeans = KMeans::load_safetensors(path).expect("Test data should be valid");

    // Verify predictions match
    let pred_original = kmeans.predict(&x);
    let pred_loaded = loaded_kmeans.predict(&x);

    assert_eq!(
        pred_original, pred_loaded,
        "High-dimensional predictions mismatch"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_kmeans_inertia_preserved() {
    // Property test: Inertia should be preserved
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 9.0, 11.0, 9.5, 10.0],
    )
    .expect("Test data should be valid");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&x).expect("Test data should be valid");

    let original_inertia = kmeans.inertia();

    // Save and load
    let path = "test_kmeans_inertia.safetensors";
    kmeans
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_kmeans = KMeans::load_safetensors(path).expect("Test data should be valid");

    let loaded_inertia = loaded_kmeans.inertia();

    // Inertia should be identical
    assert!(
        (original_inertia - loaded_inertia).abs() < 1e-6,
        "Inertia should be preserved: original={original_inertia}, loaded={loaded_inertia}"
    );

    // Cleanup
    fs::remove_file(path).ok();
}
