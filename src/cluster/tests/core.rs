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

include!("core_part_02.rs");
include!("core_part_03.rs");
