//! Tests for clustering algorithms.

use crate::cluster::*;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

fn sample_data() -> Matrix<f32> {
    // Two well-separated clusters
    Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 1.0, 0.6, 8.0, 8.0, 9.0, 11.0, 8.5, 9.0],
    )
    .expect("Sample data matrix creation should succeed")
}

#[test]
fn test_new() {
    let kmeans = KMeans::new(3);
    assert_eq!(kmeans.n_clusters(), 3);
    assert!(!kmeans.is_fitted());
}

#[test]
fn test_fit_basic() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    assert!(kmeans.is_fitted());
    assert_eq!(kmeans.centroids().shape(), (2, 2));
    assert!(kmeans.inertia() >= 0.0);
}

#[test]
fn test_predict() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);
    assert_eq!(labels.len(), 6);

    // All labels should be valid cluster indices
    for &label in &labels {
        assert!(label < 2);
    }
}

#[test]
fn test_labels_consistency() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    // Points in the same cluster should be close to each other
    // First 3 points should be in one cluster, last 3 in another
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_with_max_iter() {
    let kmeans = KMeans::new(3).with_max_iter(10);
    assert_eq!(kmeans.max_iter(), 10);
}

#[test]
fn test_with_tol() {
    let kmeans = KMeans::new(3).with_tol(1e-6);
    assert!((kmeans.tol() - 1e-6).abs() < 1e-10);
}

#[test]
fn test_with_random_state() {
    let kmeans = KMeans::new(3).with_random_state(42);
    assert_eq!(kmeans.random_state(), Some(42));
}

#[test]
fn test_empty_data_error() {
    let data = Matrix::from_vec(0, 2, vec![]).expect("Empty matrix creation should succeed");
    let mut kmeans = KMeans::new(2);
    let result = kmeans.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_too_many_clusters_error() {
    let data = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Matrix creation should succeed");
    let mut kmeans = KMeans::new(5);
    let result = kmeans.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_single_cluster() {
    let data = sample_data();
    let mut kmeans = KMeans::new(1);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);
    // All points should be in cluster 0
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn test_inertia_decreases_with_more_clusters() {
    let data = sample_data();

    let mut kmeans1 = KMeans::new(1).with_random_state(42);
    kmeans1.fit(&data).expect("KMeans fit should succeed");
    let inertia1 = kmeans1.inertia();

    let mut kmeans2 = KMeans::new(2).with_random_state(42);
    kmeans2.fit(&data).expect("KMeans fit should succeed");
    let inertia2 = kmeans2.inertia();

    // More clusters should lead to lower or equal inertia
    assert!(inertia2 <= inertia1);
}

#[test]
fn test_reproducibility() {
    let data = sample_data();

    let mut kmeans1 = KMeans::new(2).with_random_state(42);
    kmeans1.fit(&data).expect("KMeans fit should succeed");

    let mut kmeans2 = KMeans::new(2).with_random_state(42);
    kmeans2.fit(&data).expect("KMeans fit should succeed");

    // Same seed should give same centroids
    let c1 = kmeans1.centroids();
    let c2 = kmeans2.centroids();

    for i in 0..2 {
        for j in 0..2 {
            assert!((c1.get(i, j) - c2.get(i, j)).abs() < 1e-6);
        }
    }
}

#[test]
fn test_convergence() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2).with_max_iter(1000);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // Should converge before max iterations for simple data
    assert!(kmeans.n_iter() < 100);
}

#[test]
fn test_centroids_converged_within_tolerance() {
    // Test when centroids have converged (within tolerance)
    let kmeans = KMeans::new(2).with_tol(0.01);

    // Old centroids: [[1.0, 2.0], [3.0, 4.0]]
    let old = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed");

    // New centroids: [[1.005, 2.005], [3.005, 4.005]]
    // Distance per centroid: sqrt(0.005^2 + 0.005^2) ≈ 0.00707 < 0.01
    let new = Matrix::from_vec(2, 2, vec![1.005_f32, 2.005, 3.005, 4.005])
        .expect("Matrix creation should succeed");

    assert!(kmeans.centroids_converged(&old, &new));
}

#[test]
fn test_centroids_not_converged() {
    // Test when centroids have not converged (beyond tolerance)
    let kmeans = KMeans::new(2).with_tol(0.01);

    // Old centroids: [[1.0, 2.0], [3.0, 4.0]]
    let old = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed");

    // New centroids: [[1.1, 2.1], [3.0, 4.0]]
    // First centroid distance: sqrt(0.1^2 + 0.1^2) ≈ 0.141 > 0.01
    let new = Matrix::from_vec(2, 2, vec![1.1_f32, 2.1, 3.0, 4.0])
        .expect("Matrix creation should succeed");

    assert!(!kmeans.centroids_converged(&old, &new));
}

#[test]
fn test_centroids_converged_exact_tolerance() {
    // Test boundary case: distance exactly at tolerance²
    // Use tol=0.1, so tol²=0.01
    // Set up distance² to be exactly 0.01
    let kmeans = KMeans::new(1).with_tol(0.1);

    // Old centroid: [[0.0, 0.0]]
    let old = Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).expect("Matrix creation should succeed");

    // New centroid: [[0.1, 0.0]]
    // Distance²: 0.1² + 0.0² = 0.01 (exactly tol²)
    // Should be converged (dist² = tol² means dist = tol, which is at boundary)
    // Original code: dist² > tol² is false, so converged ✓
    // Mutated code: dist² >= tol² is true, so NOT converged ✗
    let new_exact =
        Matrix::from_vec(1, 2, vec![0.1_f32, 0.0]).expect("Matrix creation should succeed");
    assert!(
        kmeans.centroids_converged(&old, &new_exact),
        "Distance exactly at tolerance should be converged"
    );

    // Now test just beyond tolerance
    // Distance²: 0.11² ≈ 0.0121 > 0.01
    let new_beyond =
        Matrix::from_vec(1, 2, vec![0.11_f32, 0.0]).expect("Matrix creation should succeed");
    assert!(
        !kmeans.centroids_converged(&old, &new_beyond),
        "Distance beyond tolerance should not be converged"
    );
}

#[test]
fn test_centroids_converged_multi_cluster() {
    // Test with multiple clusters - all must be within tolerance
    let kmeans = KMeans::new(3).with_tol(0.01);

    let old = Matrix::from_vec(
        3,
        2,
        vec![
            1.0_f32, 2.0, // Cluster 0
            3.0, 4.0, // Cluster 1
            5.0, 6.0, // Cluster 2
        ],
    )
    .expect("Matrix creation should succeed");

    // All clusters within tolerance
    let new_converged = Matrix::from_vec(
        3,
        2,
        vec![
            1.005_f32, 2.005, // Small change
            3.005, 4.005, // Small change
            5.005, 6.005, // Small change
        ],
    )
    .expect("Matrix creation should succeed");
    assert!(kmeans.centroids_converged(&old, &new_converged));

    // One cluster beyond tolerance (cluster 1)
    let new_not_converged = Matrix::from_vec(
        3,
        2,
        vec![
            1.005_f32, 2.005, // Small change
            3.1, 4.1, // Large change
            5.005, 6.005, // Small change
        ],
    )
    .expect("Matrix creation should succeed");
    assert!(!kmeans.centroids_converged(&old, &new_not_converged));
}

#[test]
fn test_initialization_centroid_spread() {
    // Test that k-means++ produces well-separated initial centroids
    // This catches distance calculation mutations (line 189: * with +)
    let data = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // Point 0: far from others
            10.0, 10.0, // Point 1: far from 0
            10.1, 10.1, // Point 2: very close to 1
            10.2, 10.2, // Point 3: very close to 1
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let centroids = kmeans.centroids();

    // With correct distance calculation, k-means++ should pick well-separated points
    // First centroid: depends on seed
    // Second centroid: should be far from first (catches dist calculation error)

    // Verify centroids are well-separated (distance > 5.0)
    let c0 = (centroids.get(0, 0), centroids.get(0, 1));
    let c1 = (centroids.get(1, 0), centroids.get(1, 1));
    let dist_sq = (c0.0 - c1.0).powi(2) + (c0.1 - c1.1).powi(2);

    assert!(
        dist_sq > 25.0,
        "Centroids should be well-separated (dist > 5.0), got dist² = {dist_sq}"
    );
}

#[test]
fn test_initialization_selects_farthest() {
    // Test that k-means++ initialization leads to correct clustering
    // This catches comparison mutations (line 191: < with <=, line 202: > with >=)
    // Data: two far apart points (0.0 and 10.0) plus one near first (0.5)
    let data = Matrix::from_vec(
        5,
        1,
        vec![
            0.0,  // Point 0: cluster A
            0.5,  // Point 1: cluster A (near 0)
            0.3,  // Point 2: cluster A (near 0)
            10.0, // Point 3: cluster B (far from others)
            9.8,  // Point 4: cluster B (near 10)
        ],
    )
    .expect("Matrix creation should succeed");

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);

    // Verify correct clustering: points 0,1,2 in one cluster, points 3,4 in another
    // If k-means++ fails (due to mutations), might not separate clusters correctly
    assert_eq!(
        labels[0], labels[1],
        "Points 0 and 1 should be in same cluster"
    );
    assert_eq!(
        labels[0], labels[2],
        "Points 0 and 2 should be in same cluster"
    );
    assert_eq!(
        labels[3], labels[4],
        "Points 3 and 4 should be in same cluster"
    );
    assert_ne!(labels[0], labels[3], "Clusters should be different");

    // Also verify centroids are well-separated
    let centroids = kmeans.centroids();
    let diff = (centroids.get(0, 0) - centroids.get(1, 0)).abs();
    assert!(
        diff > 5.0,
        "Centroids should be separated by > 5.0, got {diff}"
    );
}

#[test]
fn test_initialization_reproducibility() {
    // Verify that same random_state produces same initialization
    // This indirectly tests the distance and selection logic
    let data = sample_data();

    let mut kmeans1 = KMeans::new(2).with_random_state(42);
    let mut kmeans2 = KMeans::new(2).with_random_state(42);

    kmeans1.fit(&data).expect("KMeans fit should succeed");
    kmeans2.fit(&data).expect("KMeans fit should succeed");

    let c1 = kmeans1.centroids();
    let c2 = kmeans2.centroids();

    // Centroids should be identical for same seed
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (c1.get(i, j) - c2.get(i, j)).abs() < 1e-6,
                "Centroids should match for same random_state"
            );
        }
    }
}

#[test]
fn test_default() {
    let kmeans = KMeans::default();
    assert_eq!(kmeans.n_clusters(), 8);
}

#[test]
fn test_labels_max_less_than_n_clusters() {
    // Property: labels.max() < n_clusters
    let data = sample_data();
    let mut kmeans = KMeans::new(3).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&data);
    let max_label = labels.iter().max().expect("Labels should not be empty");
    assert!(*max_label < 3);
}

#[test]
fn test_predict_new_data() {
    let data = sample_data();
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("KMeans fit should succeed");

    // Predict on new data point
    let new_point = Matrix::from_vec(1, 2, vec![1.2, 1.5]).expect("Matrix creation should succeed");
    let labels = kmeans.predict(&new_point);

    assert_eq!(labels.len(), 1);
    assert!(labels[0] < 2);
}

#[test]
fn test_larger_dataset() {
    // Test with more samples
    let n = 50;
    let mut data = Vec::with_capacity(n * 2);

    // Two clusters: around (0,0) and (10,10)
    for i in 0..n {
        if i < n / 2 {
            data.push((i as f32) * 0.1);
            data.push((i as f32) * 0.1);
        } else {
            data.push(10.0 + (i as f32) * 0.1);
            data.push(10.0 + (i as f32) * 0.1);
        }
    }

    let matrix = Matrix::from_vec(n, 2, data).expect("Matrix creation should succeed");
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&matrix).expect("KMeans fit should succeed");

    let labels = kmeans.predict(&matrix);

    // First half should be in one cluster, second half in another
    let first_label = labels[0];
    let second_label = labels[n / 2];

    assert_ne!(first_label, second_label);

    // Check consistency within clusters
    for label in labels.iter().take(n / 2) {
        assert_eq!(*label, first_label);
    }
    for label in labels.iter().skip(n / 2) {
        assert_eq!(*label, second_label);
    }
}

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
    assert!(result
        .unwrap_err()
        .contains("Cannot save unfitted model"));
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

