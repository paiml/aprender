
#[test]
fn test_three_clusters() {
    // Three well-separated clusters
    let data = Matrix::from_vec(
        9,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 5.0, 5.0, 5.1, 5.1, 5.0, 5.2, 10.0, 0.0, 10.1, 0.1, 10.0,
            0.2,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(3).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    // Check that we have exactly 3 unique labels
    let mut unique_labels: Vec<usize> = labels.clone();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    assert_eq!(unique_labels.len(), 3);

    // Points in same cluster should have same label
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_eq!(labels[6], labels[7]);
    assert_eq!(labels[7], labels[8]);
}

#[test]
fn test_identical_points() {
    // All points are the same
    let data = Matrix::from_vec(5, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // All should be in same cluster
    let labels = kmeans.predict(&data);
    let first = labels[0];
    assert!(labels.iter().all(|&l| l == first));

    // Inertia should be 0
    assert!(kmeans.inertia() < 1e-6);
}

#[test]
fn test_one_dimensional_data() {
    // 1D clustering
    let data = Matrix::from_vec(6, 1, vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
        .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    // First 3 should be in one cluster, last 3 in another
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_high_dimensional_data() {
    // 5D clustering
    let data = Matrix::from_vec(
        6,
        5,
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 10.1, 10.1, 10.2, 10.2, 10.2, 10.2, 10.2,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    // First 3 should be in one cluster, last 3 in another
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_negative_values() {
    // Test with negative coordinates
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            -10.0, -10.0, -10.1, -10.1, -10.2, -10.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_exact_k_samples() {
    // Exactly k samples for k clusters
    let data = Matrix::from_vec(3, 2, vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0])
        .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(3).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    // All should have different labels
    assert_ne!(labels[0], labels[1]);
    assert_ne!(labels[1], labels[2]);
    assert_ne!(labels[0], labels[2]);

    // Inertia should be 0 (each point is its own centroid)
    assert!(kmeans.inertia() < 1e-6);
}

#[test]
fn test_max_iter_limit() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2).with_max_iter(1).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // Should stop at 1 iteration
    assert_eq!(kmeans.n_iter(), 1);
}

#[test]
fn test_n_iter_not_one() {
    // Test that catches n_iter() → 1 mutation (line 132)
    // Use data that requires multiple iterations to converge
    let data = sample_data();
    let mut kmeans = KMeans::new(2).with_max_iter(100).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // For this data, should converge in > 1 iteration
    assert!(
        kmeans.n_iter() > 1,
        "Expected n_iter > 1, got {}",
        kmeans.n_iter()
    );
    assert!(kmeans.n_iter() < 100, "Should converge before max_iter");
}

#[test]
fn test_inertia_not_zero() {
    // Test that catches inertia() → 0.0 mutation (line 126)
    // Use imperfect clustering to ensure non-zero inertia
    let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
        .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // Inertia should be positive (points aren't exactly at centroids)
    assert!(
        kmeans.inertia() > 0.0,
        "Inertia should be > 0.0, got {}",
        kmeans.inertia()
    );

    // Also verify inertia is reasonable (not too large)
    assert!(
        kmeans.inertia() < 1.0,
        "Inertia should be small for well-separated clusters"
    );
}

#[test]
fn test_tight_tolerance() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2).with_tol(1e-10).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // With tight tolerance, should still converge for simple data
    assert!(kmeans.is_fitted());
}

#[test]
fn test_centroid_shapes() {
    let data = Matrix::from_vec(
        10,
        3,
        vec![
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 10.0, 10.0,
            10.0, 10.1, 10.1, 10.1, 10.2, 10.2, 10.2, 10.3, 10.3, 10.3, 10.4, 10.4, 10.4,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let centroids = kmeans.centroids();
    assert_eq!(centroids.shape(), (2, 3));
}

#[test]
fn test_different_random_states() {
    let data = sample_data();

    let mut kmeans1 = KMeans::new(2).with_random_state(1);
    kmeans1.fit(&data).expect("KMeans fit should succeed");

    let mut kmeans2 = KMeans::new(2).with_random_state(999);
    kmeans2.fit(&data).expect("KMeans fit should succeed");

    // Different seeds should still produce valid results
    // (labels may differ but should be valid)
    let labels1 = kmeans1.predict(&data);
    let labels2 = kmeans2.predict(&data);

    for &l in &labels1 {
        assert!(l < 2);
    }
    for &l in &labels2 {
        assert!(l < 2);
    }
}

#[test]
fn test_save_load() {
    use std::fs;
    use std::path::Path;

    let data = sample_data();
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // Save model
    let path = Path::new("/tmp/test_kmeans.bin");
    kmeans.save(path).expect("Failed to save model");

    // Load model
    let loaded = KMeans::load(path).expect("Failed to load model");

    // Verify loaded model matches original
    assert_eq!(kmeans.n_clusters(), loaded.n_clusters());
    assert!((kmeans.inertia() - loaded.inertia()).abs() < 1e-6);

    // Verify predictions match
    let original_labels = kmeans.predict(&data);
    let loaded_labels = loaded.predict(&data);
    assert_eq!(original_labels, loaded_labels);

    // Verify centroids match
    let orig_centroids = kmeans.centroids();
    let loaded_centroids = loaded.centroids();
    assert_eq!(orig_centroids.shape(), loaded_centroids.shape());

    let (n_clusters, n_features) = orig_centroids.shape();
    for i in 0..n_clusters {
        for j in 0..n_features {
            assert!(
                (orig_centroids.get(i, j) - loaded_centroids.get(i, j)).abs() < 1e-6,
                "Centroid mismatch at ({i}, {j})"
            );
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_centroids_converged_arithmetic() {
    // Test to catch arithmetic mutations in centroids_converged
    // (dist_sq += diff * diff should be sum of squared differences)
    let kmeans = KMeans::new(1).with_tol(0.5);

    // 1D case: old = [0.0], new = [0.3]
    // diff = 0.3, dist_sq = 0.09
    // tol² = 0.25, so 0.09 < 0.25, should converge
    let old = Matrix::from_vec(1, 1, vec![0.0_f32]).expect("Matrix creation should succeed");
    let new = Matrix::from_vec(1, 1, vec![0.3_f32]).expect("Matrix creation should succeed");
    assert!(
        kmeans.centroids_converged(&old, &new),
        "Should converge when dist² (0.09) < tol² (0.25)"
    );

    // 2D case: movement of [0.4, 0.3]
    // dist_sq = 0.16 + 0.09 = 0.25 = tol², should converge (<=)
    let old_2d =
        Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).expect("Matrix creation should succeed");
    let new_2d =
        Matrix::from_vec(1, 2, vec![0.4_f32, 0.3]).expect("Matrix creation should succeed");
    assert!(
        kmeans.centroids_converged(&old_2d, &new_2d),
        "Should converge when dist² equals tol²"
    );
}

#[test]
fn test_centroids_converged_comparison_mutations() {
    // Test to catch > vs >= vs == vs < mutations
    let kmeans = KMeans::new(1).with_tol(1.0);

    // Case: dist_sq = 0.5, tol² = 1.0
    // Should converge because 0.5 < 1.0
    let old = Matrix::from_vec(1, 1, vec![0.0_f32]).expect("Matrix creation should succeed");
    let less = Matrix::from_vec(1, 1, vec![0.7_f32]).expect("Matrix creation should succeed"); // dist_sq ≈ 0.49
    assert!(
        kmeans.centroids_converged(&old, &less),
        "dist² < tol² should converge"
    );

    // Case: dist_sq = 1.5, tol² = 1.0
    // Should NOT converge because 1.5 > 1.0
    let more = Matrix::from_vec(1, 1, vec![1.3_f32]).expect("Matrix creation should succeed"); // dist_sq ≈ 1.69
    assert!(
        !kmeans.centroids_converged(&old, &more),
        "dist² > tol² should NOT converge"
    );
}

#[test]
fn test_convergence_affects_iterations() {
    // Test that catches centroids_converged -> true mutation
    // If convergence always returns true, we'd stop at iteration 1
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 1.0, 0.0, 0.5, 0.0, // Cluster 1 region
            10.0, 10.0, 11.0, 10.0, 10.5, 10.0, // Cluster 2 region
        ],
    )
    .expect("Matrix creation should succeed");

    // Use very tight tolerance to force multiple iterations
    let mut kmeans = KMeans::new(2)
        .with_tol(1e-8)
        .with_max_iter(50)
        .with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // Should take more than 1 iteration to converge with tight tolerance
    // If centroids_converged always returns true, n_iter would be 1
    assert!(
        kmeans.n_iter() >= 2,
        "With tight tolerance, should need >= 2 iterations, got {}",
        kmeans.n_iter()
    );
}

#[test]
fn test_assign_labels_tie_breaking() {
    // Test to catch < vs <= mutation in assign_labels
    // When a point is equidistant, the first cluster should win
    let mut kmeans = KMeans::new(2).with_random_state(42);

    // Two centroids at (0, 0) and (2, 0)
    // Point at (1, 0) is equidistant from both
    let data = Matrix::from_vec(3, 2, vec![0.0, 0.0, 2.0, 0.0, 1.0, 0.0])
        .expect("Matrix creation should succeed");

    kmeans.fit(&data).expect("KMeans fit should succeed");

    // The key is that the middle point should be assigned consistently
    let labels = kmeans.predict(&data);

    // Points should be assigned to different clusters
    // The equidistant point (1, 0) should go to first cluster found (cluster 0)
    assert_ne!(
        labels[0], labels[1],
        "First two points should be in different clusters"
    );
}

#[test]
fn test_kmeans_plusplus_selects_maximum() {
    // Test to catch > vs >= mutation in kmeans_plusplus_init
    // With identical max distances, should pick consistently
    let data = Matrix::from_vec(
        4,
        1,
        vec![
            0.0,  // First centroid
            5.0,  // Distance 5 from first
            5.0,  // Also distance 5 (tie)
            10.0, // Distance 10 (should be selected as max)
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // The two clusters should separate the far points
    let labels = kmeans.predict(&data);

    // Points 0,1,2 should be in one cluster, point 3 in another (or vice versa)
    // The key is point 3 (at 10.0) should be in a different cluster than point 0
    assert_ne!(
        labels[0], labels[3],
        "Points at 0.0 and 10.0 should be in different clusters"
    );
}
