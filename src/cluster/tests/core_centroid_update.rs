
#[test]
fn test_update_centroids_division() {
    // Test to catch division-related mutations in update_centroids
    // Ensures centroids are computed as mean, not sum
    let data = Matrix::from_vec(
        4,
        1,
        vec![
            0.0, 2.0, // Cluster 0: mean = 1.0
            10.0, 12.0, // Cluster 1: mean = 11.0
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let centroids = kmeans.centroids();

    // Centroids should be at means (approximately 1.0 and 11.0)
    let c0 = centroids.get(0, 0);
    let c1 = centroids.get(1, 0);

    // One centroid should be near 1.0, other near 11.0
    let has_low = (c0 - 1.0).abs() < 1.0 || (c1 - 1.0).abs() < 1.0;
    let has_high = (c0 - 11.0).abs() < 1.0 || (c1 - 11.0).abs() < 1.0;

    assert!(has_low, "Should have centroid near 1.0, got {c0} and {c1}");
    assert!(
        has_high,
        "Should have centroid near 11.0, got {c0} and {c1}"
    );
}

// EXTREME TDD: Additional mutation-killing tests for centroids_converged

#[test]
fn test_centroids_converged_squaring_not_division() {
    // MUTATION TARGET: "replace * with / in diff * diff"
    // Tests that we compute diff² correctly, not diff/diff
    let kmeans = KMeans::new(1).with_tol(1.0);

    // With diff = 0.5, if using * we get 0.25, if using / we get 1.0
    let old = Matrix::from_vec(1, 1, vec![0.0_f32]).expect("Matrix creation should succeed");
    let new = Matrix::from_vec(1, 1, vec![0.5_f32]).expect("Matrix creation should succeed");

    // dist_sq = 0.5² = 0.25 < 1.0², should converge
    assert!(
        kmeans.centroids_converged(&old, &new),
        "With diff=0.5, diff²=0.25 < tol²=1.0, should converge"
    );

    // If mutation uses diff/diff, we get dist_sq=1.0 = tol², still converges
    // Need another case: diff = 2.0
    let new2 = Matrix::from_vec(1, 1, vec![2.0_f32]).expect("Matrix creation should succeed");
    // dist_sq = 2.0² = 4.0 > 1.0², should NOT converge
    // But if mutation uses 2.0/2.0 = 1.0 = 1.0², would converge (WRONG!)
    assert!(
            !kmeans.centroids_converged(&old, &new2),
            "With diff=2.0, diff²=4.0 > tol²=1.0, must NOT converge. If using diff/diff=1.0, test fails."
        );
}

#[test]
fn test_centroids_converged_sum_not_multiply() {
    // MUTATION TARGET: "replace += with *= in dist_sq += diff * diff"
    // Tests that we sum squared differences, not multiply them
    let kmeans = KMeans::new(1).with_tol(0.6);

    // 2D case: diffs = [0.3, 0.4]
    // Correct: dist_sq = 0.09 + 0.16 = 0.25
    // Mutation *= : dist_sq = 0 * 0.09 = 0, then 0 * 0.16 = 0 (always 0!)
    let old = Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).expect("Matrix creation should succeed");
    let new = Matrix::from_vec(1, 2, vec![0.3_f32, 0.4]).expect("Matrix creation should succeed");

    // dist = √(0.25) = 0.5 < 0.6, should converge
    assert!(
        kmeans.centroids_converged(&old, &new),
        "dist²=0.25 < tol²=0.36, should converge"
    );

    // If mutation uses *=, dist_sq stays 0.0, would always converge
    // Test case that should NOT converge:
    let new2 = Matrix::from_vec(1, 2, vec![0.5_f32, 0.5]).expect("Matrix creation should succeed");
    // dist_sq = 0.25 + 0.25 = 0.5 > 0.36, should NOT converge
    // But if *= mutation, dist_sq = 0, would converge (WRONG!)
    assert!(
        !kmeans.centroids_converged(&old, &new2),
        "dist²=0.5 > tol²=0.36, must NOT converge"
    );
}

#[test]
fn test_centroids_converged_addition_not_squaring() {
    // MUTATION TARGET: "replace * with + in diff * diff"
    // Tests that we compute diff², not diff+diff
    let kmeans = KMeans::new(1).with_tol(1.0);

    // With diff = 0.6:
    // Correct: diff² = 0.36 < 1.0, should converge
    // Mutation: diff+diff = 1.2 > 1.0, would NOT converge (WRONG!)
    let old = Matrix::from_vec(1, 1, vec![0.0_f32]).expect("Matrix creation should succeed");
    let new = Matrix::from_vec(1, 1, vec![0.6_f32]).expect("Matrix creation should succeed");

    assert!(
        kmeans.centroids_converged(&old, &new),
        "diff²=0.36 < tol²=1.0, must converge. If using diff+diff=1.2, test fails."
    );
}

#[test]
fn test_centroids_converged_greater_not_equal() {
    // MUTATION TARGET: "replace > with == in dist_sq > tol²"
    let kmeans = KMeans::new(1).with_tol(1.0);

    // Case: dist_sq = 1.5 > tol² = 1.0, should NOT converge
    // If mutation uses ==, would converge (WRONG!)
    let old = Matrix::from_vec(1, 1, vec![0.0_f32]).expect("Matrix creation should succeed");
    let new = Matrix::from_vec(1, 1, vec![1.3_f32]).expect("Matrix creation should succeed"); // dist_sq ≈ 1.69

    assert!(
        !kmeans.centroids_converged(&old, &new),
        "dist²=1.69 > tol²=1.0, must NOT converge"
    );
}

#[test]
fn test_centroids_converged_greater_not_less() {
    // MUTATION TARGET: "replace > with < in dist_sq > tol²"
    let kmeans = KMeans::new(1).with_tol(1.0);

    // Case: dist_sq = 0.5 < tol² = 1.0, should converge
    // If mutation uses <, logic inverts: would NOT converge (WRONG!)
    let old = Matrix::from_vec(1, 1, vec![0.0_f32]).expect("Matrix creation should succeed");
    let new = Matrix::from_vec(1, 1, vec![0.7_f32]).expect("Matrix creation should succeed"); // dist_sq = 0.49

    assert!(
        kmeans.centroids_converged(&old, &new),
        "dist²=0.49 < tol²=1.0, must converge. If using <, test fails."
    );
}

#[test]
fn test_kmeans_save_safetensors_roundtrip() {
    use std::fs;
    use tempfile::tempdir;

    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42).with_max_iter(100);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let dir = tempdir().expect("tempdir creation should succeed");
    let path = dir.path().join("kmeans_model.safetensors");

    kmeans
        .save_safetensors(&path)
        .expect("SafeTensors save should succeed");
    assert!(path.exists(), "SafeTensors file should exist");

    let loaded = KMeans::load_safetensors(&path).expect("SafeTensors load should succeed");

    assert_eq!(loaded.n_clusters(), kmeans.n_clusters());
    assert_eq!(loaded.max_iter(), kmeans.max_iter());
    assert!((loaded.tol() - kmeans.tol()).abs() < 1e-6);
    assert_eq!(loaded.random_state(), kmeans.random_state());
    assert!((loaded.inertia() - kmeans.inertia()).abs() < 1e-3);
    assert_eq!(loaded.n_iter(), kmeans.n_iter());

    // Verify centroids match
    let orig_centroids = kmeans.centroids();
    let loaded_centroids = loaded.centroids();
    assert_eq!(orig_centroids.shape(), loaded_centroids.shape());

    for i in 0..orig_centroids.n_rows() {
        for j in 0..orig_centroids.n_cols() {
            assert!(
                (orig_centroids.get(i, j) - loaded_centroids.get(i, j)).abs() < 1e-6,
                "Centroid mismatch at ({}, {})",
                i,
                j
            );
        }
    }

    // Verify predictions match
    let orig_labels = kmeans.predict(&data);
    let loaded_labels = loaded.predict(&data);
    assert_eq!(orig_labels, loaded_labels);

    fs::remove_file(&path).ok();
}

#[test]
fn test_kmeans_save_safetensors_without_random_state() {
    use std::fs;
    use tempfile::tempdir;

    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
    )
    .expect("Matrix creation should succeed");

    // Create model without explicit random_state (uses default)
    let mut kmeans = KMeans::new(2);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let dir = tempdir().expect("tempdir creation should succeed");
    let path = dir.path().join("kmeans_no_random.safetensors");

    kmeans
        .save_safetensors(&path)
        .expect("SafeTensors save should succeed");

    let loaded = KMeans::load_safetensors(&path).expect("SafeTensors load should succeed");

    // random_state should be None in loaded model
    assert!(loaded.random_state().is_none());

    fs::remove_file(&path).ok();
}

#[test]
fn test_kmeans_save_safetensors_unfitted_error() {
    use tempfile::tempdir;

    let kmeans = KMeans::new(2);
    let dir = tempdir().expect("tempdir creation should succeed");
    let path = dir.path().join("kmeans_unfitted.safetensors");

    let result = kmeans.save_safetensors(&path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Cannot save unfitted model"));
}

#[test]
fn test_kmeans_load_safetensors_missing_file() {
    let result = KMeans::load_safetensors("/nonexistent/path/model.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_kmeans_load_safetensors_invalid_format() {
    use std::fs;
    use tempfile::tempdir;

    let dir = tempdir().expect("tempdir creation should succeed");
    let path = dir.path().join("invalid.safetensors");

    // Write invalid data
    fs::write(&path, b"not a valid safetensors file").expect("write should succeed");

    let result = KMeans::load_safetensors(&path);
    assert!(result.is_err());

    fs::remove_file(&path).ok();
}

#[test]
fn test_kmeans_load_safetensors_invalid_centroids_shape() {
    use crate::serialization::safetensors;
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::tempdir;

    let dir = tempdir().expect("tempdir creation should succeed");
    let path = dir.path().join("invalid_shape.safetensors");

    // Create a safetensors file with 1D centroids tensor (invalid shape)
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "centroids".to_string(),
        (vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]), // 1D shape, should be 2D
    );
    tensors.insert("n_clusters".to_string(), (vec![2.0_f32], vec![1]));
    tensors.insert("max_iter".to_string(), (vec![100.0_f32], vec![1]));
    tensors.insert("tol".to_string(), (vec![1e-4_f32], vec![1]));
    tensors.insert("random_state".to_string(), (vec![-1.0_f32], vec![1]));
    tensors.insert("inertia".to_string(), (vec![0.0_f32], vec![1]));
    tensors.insert("n_iter".to_string(), (vec![10.0_f32], vec![1]));

    safetensors::save_safetensors(&path, &tensors).expect("save should succeed");

    let result = KMeans::load_safetensors(&path);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("Invalid centroids tensor shape"),
        "Error message should indicate invalid shape"
    );

    fs::remove_file(&path).ok();
}
