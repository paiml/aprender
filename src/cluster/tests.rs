//! Tests for clustering algorithms.

use super::*;


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
        assert_eq!(kmeans.n_clusters, 3);
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
        assert_eq!(kmeans.max_iter, 10);
    }

    #[test]
    fn test_with_tol() {
        let kmeans = KMeans::new(3).with_tol(1e-6);
        assert!((kmeans.tol - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_with_random_state() {
        let kmeans = KMeans::new(3).with_random_state(42);
        assert_eq!(kmeans.random_state, Some(42));
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
        let old =
            Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).expect("Matrix creation should succeed");

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
        assert_eq!(kmeans.n_clusters, 8);
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
        let new_point =
            Matrix::from_vec(1, 2, vec![1.2, 1.5]).expect("Matrix creation should succeed");
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
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 5.0, 5.0, 5.1, 5.1, 5.0, 5.2, 10.0, 0.0, 10.1, 0.1,
                10.0, 0.2,
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
                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 10.0,
                10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 10.1, 10.1, 10.2, 10.2, 10.2, 10.2, 10.2,
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
                0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 10.0,
                10.0, 10.0, 10.1, 10.1, 10.1, 10.2, 10.2, 10.2, 10.3, 10.3, 10.3, 10.4, 10.4, 10.4,
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
        assert_eq!(kmeans.n_clusters, loaded.n_clusters);
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
        let old =
            Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).expect("Matrix creation should succeed");
        let new =
            Matrix::from_vec(1, 2, vec![0.3_f32, 0.4]).expect("Matrix creation should succeed");

        // dist = √(0.25) = 0.5 < 0.6, should converge
        assert!(
            kmeans.centroids_converged(&old, &new),
            "dist²=0.25 < tol²=0.36, should converge"
        );

        // If mutation uses *=, dist_sq stays 0.0, would always converge
        // Test case that should NOT converge:
        let new2 =
            Matrix::from_vec(1, 2, vec![0.5_f32, 0.5]).expect("Matrix creation should succeed");
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

    // ============================================================================
    // DBSCAN Tests
    // ============================================================================

    #[test]
    fn test_dbscan_new() {
        let dbscan = DBSCAN::new(0.5, 3);
        assert_eq!(dbscan.eps(), 0.5);
        assert_eq!(dbscan.min_samples(), 3);
        assert!(!dbscan.is_fitted());
    }

    #[test]
    fn test_dbscan_fit_basic() {
        // Two well-separated clusters
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, // Cluster 0
                1.2, 1.1, // Cluster 0
                1.1, 1.2, // Cluster 0
                5.0, 5.0, // Cluster 1
                5.1, 5.2, // Cluster 1
                5.2, 5.1, // Cluster 1
            ],
        )
        .expect("Matrix creation should succeed");

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).expect("DBSCAN fit should succeed");

        assert!(dbscan.is_fitted());
        let labels = dbscan.labels();
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_dbscan_predicts_clusters() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, 1.2, 1.1, 1.1, 1.2, // Cluster 0
                5.0, 5.0, 5.1, 5.2, 5.2, 5.1, // Cluster 1
            ],
        )
        .expect("Matrix creation should succeed");

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).expect("DBSCAN fit should succeed");

        let labels = dbscan.predict(&data);
        assert_eq!(labels.len(), 6);

        // First 3 samples should be in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);

        // Last 3 samples should be in same cluster
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);

        // Two clusters should be different
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_dbscan_noise_detection() {
        // Two clusters with one outlier
        let data = Matrix::from_vec(
            7,
            2,
            vec![
                1.0, 1.0, // Cluster 0
                1.2, 1.1, // Cluster 0
                1.1, 1.2, // Cluster 0
                5.0, 5.0, // Cluster 1
                5.1, 5.2, // Cluster 1
                5.2, 5.1, // Cluster 1
                10.0, 10.0, // Noise (far from both clusters)
            ],
        )
        .expect("Matrix creation should succeed");

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).expect("DBSCAN fit should succeed");

        let labels = dbscan.labels();

        // Last sample should be noise (-1)
        assert_eq!(labels[6], -1);
    }

    #[test]
    fn test_dbscan_single_cluster() {
        // All points form one dense cluster
        let data = Matrix::from_vec(5, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.0])
            .expect("Matrix creation should succeed");

        let mut dbscan = DBSCAN::new(0.3, 2);
        dbscan.fit(&data).expect("DBSCAN fit should succeed");

        let labels = dbscan.labels();

        // All samples should be in the same cluster (not noise)
        let first_label = labels[0];
        assert_ne!(first_label, -1);
        for &label in labels {
            assert_eq!(label, first_label);
        }
    }

    #[test]
    fn test_dbscan_all_noise() {
        // All points far apart
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
            .expect("Matrix creation should succeed");

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).expect("DBSCAN fit should succeed");

        let labels = dbscan.labels();

        // All samples should be noise
        for &label in labels {
            assert_eq!(label, -1);
        }
    }

    #[test]
    fn test_dbscan_min_samples_effect() {
        // Same data, different min_samples
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1])
            .expect("Matrix creation should succeed");

        // With min_samples=2, should form cluster
        let mut dbscan1 = DBSCAN::new(0.3, 2);
        dbscan1.fit(&data).expect("DBSCAN fit should succeed");
        let labels1 = dbscan1.labels();
        assert!(labels1.iter().any(|&l| l != -1));

        // With min_samples=5, should be all noise
        let mut dbscan2 = DBSCAN::new(0.3, 5);
        dbscan2.fit(&data).expect("DBSCAN fit should succeed");
        let labels2 = dbscan2.labels();
        assert!(labels2.iter().all(|&l| l == -1));
    }

    #[test]
    fn test_dbscan_eps_effect() {
        // Same data, different eps
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5])
            .expect("Matrix creation should succeed");

        // With large eps, should form one cluster
        let mut dbscan1 = DBSCAN::new(2.0, 2);
        dbscan1.fit(&data).expect("DBSCAN fit should succeed");
        let labels1 = dbscan1.labels();
        let unique_clusters: std::collections::HashSet<_> =
            labels1.iter().filter(|&&l| l != -1).collect();
        assert_eq!(unique_clusters.len(), 1);

        // With small eps, more fragmentation
        let mut dbscan2 = DBSCAN::new(0.3, 2);
        dbscan2.fit(&data).expect("DBSCAN fit should succeed");
        let labels2 = dbscan2.labels();
        // Should have noise or multiple small clusters
        assert!(labels2.contains(&-1));
    }

    #[test]
    fn test_dbscan_reproducible() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.2, 1.1, 1.1, 1.2, 5.0, 5.0, 5.1, 5.2, 5.2, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut dbscan1 = DBSCAN::new(0.5, 2);
        dbscan1.fit(&data).expect("DBSCAN fit should succeed");
        let labels1 = dbscan1.labels().clone();

        let mut dbscan2 = DBSCAN::new(0.5, 2);
        dbscan2.fit(&data).expect("DBSCAN fit should succeed");
        let labels2 = dbscan2.labels();

        // Results should be identical
        assert_eq!(labels1, *labels2);
    }

    #[test]
    fn test_dbscan_fit_predict() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1])
            .expect("Matrix creation should succeed");

        let mut dbscan = DBSCAN::new(0.3, 2);
        dbscan.fit(&data).expect("DBSCAN fit should succeed");

        let labels_stored = dbscan.labels().clone();
        let labels_predicted = dbscan.predict(&data);

        // predict() should return same labels as stored from fit()
        assert_eq!(labels_stored, labels_predicted);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_dbscan_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let dbscan = DBSCAN::new(0.5, 2);
        let _ = dbscan.predict(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_dbscan_labels_before_fit() {
        let dbscan = DBSCAN::new(0.5, 2);
        let _ = dbscan.labels(); // Should panic
    }

    // ==================== AgglomerativeClustering Tests ====================

    #[test]
    fn test_agglomerative_new() {
        let hc = AgglomerativeClustering::new(3, Linkage::Average);
        assert_eq!(hc.n_clusters(), 3);
        assert_eq!(hc.linkage(), Linkage::Average);
        assert!(!hc.is_fitted());
    }

    #[test]
    fn test_agglomerative_fit_basic() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");
        assert!(hc.is_fitted());
    }

    #[test]
    fn test_agglomerative_predict() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        assert_eq!(labels.len(), 6);

        // All labels should be valid cluster indices
        for &label in &labels {
            assert!(label < 2);
        }

        // Check that two distinct clusters were found
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_agglomerative_linkage_single() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Single);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        assert_eq!(labels.len(), 4);

        // Check that two clusters were formed
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_agglomerative_linkage_complete() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Complete);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        assert_eq!(labels.len(), 4);

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_agglomerative_linkage_average() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        assert_eq!(labels.len(), 4);

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_agglomerative_linkage_ward() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Ward);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        assert_eq!(labels.len(), 4);

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_agglomerative_n_clusters_1() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(1, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        // All points should be in same cluster
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_agglomerative_n_clusters_equals_samples() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(4, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels = hc.predict(&data);
        // Each point should be its own cluster
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 4);
    }

    #[test]
    fn test_agglomerative_dendrogram() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let dendrogram = hc.dendrogram();
        // Should have n_samples - n_clusters merges
        assert_eq!(dendrogram.len(), 2); // 4 samples - 2 clusters = 2 merges
    }

    #[test]
    fn test_agglomerative_reproducible() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut hc1 = AgglomerativeClustering::new(2, Linkage::Average);
        hc1.fit(&data)
            .expect("Hierarchical clustering fit should succeed");
        let labels1 = hc1.predict(&data);

        let mut hc2 = AgglomerativeClustering::new(2, Linkage::Average);
        hc2.fit(&data)
            .expect("Hierarchical clustering fit should succeed");
        let labels2 = hc2.predict(&data);

        // Results should be deterministic
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn test_agglomerative_fit_predict_consistency() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Complete);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");

        let labels_stored = hc.labels().clone();
        let labels_predicted = hc.predict(&data);

        assert_eq!(labels_stored, labels_predicted);
    }

    #[test]
    fn test_agglomerative_different_linkages_differ() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 8.0, 8.0, 8.5, 8.5, 9.0, 9.0],
        )
        .expect("Matrix creation should succeed");

        let mut hc_single = AgglomerativeClustering::new(2, Linkage::Single);
        hc_single
            .fit(&data)
            .expect("Hierarchical clustering fit should succeed");
        let labels_single = hc_single.predict(&data);

        let mut hc_complete = AgglomerativeClustering::new(2, Linkage::Complete);
        hc_complete
            .fit(&data)
            .expect("Hierarchical clustering fit should succeed");
        let labels_complete = hc_complete.predict(&data);

        // Different linkage methods may produce different results
        // but both should have exactly 2 clusters
        let unique_single: std::collections::HashSet<_> = labels_single.iter().collect();
        let unique_complete: std::collections::HashSet<_> = labels_complete.iter().collect();
        assert_eq!(unique_single.len(), 2);
        assert_eq!(unique_complete.len(), 2);
    }

    #[test]
    fn test_agglomerative_well_separated_clusters() {
        // Two very well-separated clusters
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 100.0, 100.0, 100.1, 100.1, 100.0, 100.1,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
        hc.fit(&data)
            .expect("Hierarchical clustering fit should succeed");
        let labels = hc.predict(&data);

        // First 3 points should be in one cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_agglomerative_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let hc = AgglomerativeClustering::new(2, Linkage::Average);
        let _ = hc.predict(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_agglomerative_labels_before_fit() {
        let hc = AgglomerativeClustering::new(2, Linkage::Average);
        let _ = hc.labels(); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_agglomerative_dendrogram_before_fit() {
        let hc = AgglomerativeClustering::new(2, Linkage::Average);
        let _ = hc.dendrogram(); // Should panic
    }

    // ==================== GaussianMixture Tests ====================

    #[test]
    fn test_gmm_new() {
        let gmm = GaussianMixture::new(3, CovarianceType::Full);
        assert_eq!(gmm.n_components(), 3);
        assert_eq!(gmm.covariance_type(), CovarianceType::Full);
        assert!(!gmm.is_fitted());
    }

    #[test]
    fn test_gmm_fit_basic() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");
        assert!(gmm.is_fitted());
    }

    #[test]
    fn test_gmm_predict() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 6);

        // All labels should be valid component indices
        for &label in &labels {
            assert!(label < 2);
        }
    }

    #[test]
    fn test_gmm_predict_proba() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let proba = gmm.predict_proba(&data);
        assert_eq!(proba.shape(), (6, 2));

        // Probabilities should sum to 1 for each sample
        for i in 0..6 {
            let sum: f32 = (0..2).map(|j| proba.get(i, j)).sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        // All probabilities should be in [0, 1]
        for i in 0..6 {
            for j in 0..2 {
                let p = proba.get(i, j);
                assert!((0.0..=1.0).contains(&p));
            }
        }
    }

    #[test]
    fn test_gmm_covariance_full() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_gmm_covariance_tied() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Tied);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_gmm_covariance_diag() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Diag);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_gmm_covariance_spherical() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Spherical);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_gmm_score() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let score = gmm.score(&data);
        // Log-likelihood should be finite
        assert!(score.is_finite());
    }

    #[test]
    fn test_gmm_convergence() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full).with_max_iter(100);
        gmm.fit(&data).expect("GMM fit should succeed");
        assert!(gmm.is_fitted());
    }

    #[test]
    fn test_gmm_reproducible() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm1 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
        gmm1.fit(&data).expect("GMM fit should succeed");
        let labels1 = gmm1.predict(&data);

        let mut gmm2 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
        gmm2.fit(&data).expect("GMM fit should succeed");
        let labels2 = gmm2.predict(&data);

        // Same seed should produce same results
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn test_gmm_means() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let means = gmm.means();
        assert_eq!(means.shape(), (2, 2));
    }

    #[test]
    fn test_gmm_weights() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let weights = gmm.weights();
        assert_eq!(weights.len(), 2);

        // Weights should sum to 1
        let sum: f32 = weights.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gmm_well_separated() {
        // Two very well-separated clusters
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 100.0, 100.0, 100.1, 100.1, 100.0, 100.1,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");
        let labels = gmm.predict(&data);

        // First 3 points should be in one cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_gmm_soft_vs_hard_assignment() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels = gmm.predict(&data);
        let proba = gmm.predict_proba(&data);

        // Hard assignment should match argmax of soft assignment
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            let mut max_prob = 0.0;
            let mut max_idx = 0;
            for j in 0..2 {
                let p = proba.get(i, j);
                if p > max_prob {
                    max_prob = p;
                    max_idx = j;
                }
            }
            assert_eq!(labels[i], max_idx);
        }
    }

    #[test]
    fn test_gmm_fit_predict_consistency() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
        gmm.fit(&data).expect("GMM fit should succeed");

        let labels_stored = gmm.labels().clone();
        let labels_predicted = gmm.predict(&data);

        assert_eq!(labels_stored, labels_predicted);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_gmm_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let gmm = GaussianMixture::new(2, CovarianceType::Full);
        let _ = gmm.predict(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_gmm_predict_proba_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let gmm = GaussianMixture::new(2, CovarianceType::Full);
        let _ = gmm.predict_proba(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_gmm_score_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let gmm = GaussianMixture::new(2, CovarianceType::Full);
        let _ = gmm.score(&data); // Should panic
    }

    // ========================================================================
    // Isolation Forest Tests
    // ========================================================================

    #[test]
    fn test_isolation_forest_new() {
        let iforest = IsolationForest::new();
        assert!(!iforest.is_fitted());
    }

    #[test]
    fn test_isolation_forest_fit_basic() {
        // Normal data clustered around (2, 2)
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                1.9, 1.9, 2.1, 1.8,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new();
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");
        assert!(iforest.is_fitted());
    }

    #[test]
    fn test_isolation_forest_predict_anomalies() {
        // 8 normal points + 2 outliers
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                10.0, 10.0, // Outlier 1
                -10.0, -10.0, // Outlier 2
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new()
            .with_n_estimators(100)
            .with_contamination(0.2)
            .with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let predictions = iforest.predict(&data);
        assert_eq!(predictions.len(), 10);

        // Check that predictions are either 1 (normal) or -1 (anomaly)
        for &pred in &predictions {
            assert!(pred == 1 || pred == -1);
        }

        // Should detect approximately 2 anomalies (20% contamination)
        let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
        assert!((1..=3).contains(&n_anomalies));
    }

    #[test]
    fn test_isolation_forest_score_samples() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
                10.0, 10.0, // Outlier 1
                -10.0, -10.0, // Outlier 2
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new().with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let scores = iforest.score_samples(&data);
        assert_eq!(scores.len(), 6);

        // Outliers should have lower scores than normal points
        let normal_avg = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
        let outlier_avg = (scores[4] + scores[5]) / 2.0;
        assert!(outlier_avg < normal_avg);
    }

    #[test]
    fn test_isolation_forest_contamination() {
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                10.0, 10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        // Low contamination (10%) - fewer anomalies expected
        let mut iforest_low = IsolationForest::new()
            .with_contamination(0.1)
            .with_random_state(42);
        iforest_low
            .fit(&data)
            .expect("Isolation Forest fit should succeed");
        let pred_low = iforest_low.predict(&data);
        let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();

        // High contamination (30%) - more anomalies expected
        let mut iforest_high = IsolationForest::new()
            .with_contamination(0.3)
            .with_random_state(42);
        iforest_high
            .fit(&data)
            .expect("Isolation Forest fit should succeed");
        let pred_high = iforest_high.predict(&data);
        let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();

        // Higher contamination should detect more or equal anomalies
        assert!(anomalies_high >= anomalies_low);
    }

    #[test]
    fn test_isolation_forest_n_estimators() {
        let data = Matrix::from_vec(
            8,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0,
                -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        // Fewer trees
        let mut iforest_few = IsolationForest::new()
            .with_n_estimators(10)
            .with_random_state(42);
        iforest_few
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        // More trees (should be more stable)
        let mut iforest_many = IsolationForest::new()
            .with_n_estimators(100)
            .with_random_state(42);
        iforest_many
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        // Both should work, more trees typically more accurate
        let pred_few = iforest_few.predict(&data);
        let pred_many = iforest_many.predict(&data);

        assert_eq!(pred_few.len(), 8);
        assert_eq!(pred_many.len(), 8);
    }

    #[test]
    fn test_isolation_forest_max_samples() {
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                10.0, 10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        // Use subset of samples for each tree
        let mut iforest = IsolationForest::new()
            .with_max_samples(5)
            .with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let predictions = iforest.predict(&data);
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_isolation_forest_reproducible() {
        let data = Matrix::from_vec(
            8,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0,
                -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest1 = IsolationForest::new().with_random_state(42);
        iforest1
            .fit(&data)
            .expect("Isolation Forest fit should succeed");
        let pred1 = iforest1.predict(&data);

        let mut iforest2 = IsolationForest::new().with_random_state(42);
        iforest2
            .fit(&data)
            .expect("Isolation Forest fit should succeed");
        let pred2 = iforest2.predict(&data);

        assert_eq!(pred1, pred2);
    }

    #[test]
    fn test_isolation_forest_all_normal() {
        // All points are normal (tightly clustered)
        let data = Matrix::from_vec(
            6,
            2,
            vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new()
            .with_contamination(0.1)
            .with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let predictions = iforest.predict(&data);
        // With 10% contamination, expect mostly normal points
        let n_normal = predictions.iter().filter(|&&p| p == 1).count();
        assert!(n_normal >= 5);
    }

    #[test]
    fn test_isolation_forest_score_samples_range() {
        let data = Matrix::from_vec(4, 2, vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 10.0, 10.0])
            .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new().with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let scores = iforest.score_samples(&data);
        // Anomaly scores should be in reasonable range
        for &score in &scores {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_isolation_forest_path_length() {
        // Test that isolation path length is computed correctly
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
                10.0, 10.0, // Easy to isolate outlier
                2.05, 2.05, // Normal
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new()
            .with_n_estimators(100)
            .with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let scores = iforest.score_samples(&data);
        // Outlier (index 4) should have significantly different score
        let outlier_score = scores[4];
        let normal_score = (scores[0] + scores[1] + scores[2] + scores[3] + scores[5]) / 5.0;
        assert!(outlier_score < normal_score);
    }

    #[test]
    fn test_isolation_forest_multidimensional() {
        // Test with more features
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.0, 2.0, 3.0, 0.9, 1.9, 2.9, 10.0, 10.0, 10.0,
                -10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new()
            .with_contamination(0.3)
            .with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let predictions = iforest.predict(&data);
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_isolation_forest_decision_function_consistency() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut iforest = IsolationForest::new()
            .with_contamination(0.3)
            .with_random_state(42);
        iforest
            .fit(&data)
            .expect("Isolation Forest fit should succeed");

        let predictions = iforest.predict(&data);
        let scores = iforest.score_samples(&data);

        // Points with lower scores should be more likely to be anomalies
        // (though exact correspondence depends on threshold)
        assert_eq!(predictions.len(), scores.len());
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_isolation_forest_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let iforest = IsolationForest::new();
        let _ = iforest.predict(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_isolation_forest_score_samples_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let iforest = IsolationForest::new();
        let _ = iforest.score_samples(&data); // Should panic
    }

    #[test]
    fn test_isolation_forest_empty_after_construction() {
        let iforest = IsolationForest::new();
        assert!(!iforest.is_fitted());
    }

    // ========================================================================
    // Local Outlier Factor (LOF) Tests
    // ========================================================================

    #[test]
    fn test_lof_new() {
        let lof = LocalOutlierFactor::new();
        assert!(!lof.is_fitted());
    }

    #[test]
    fn test_lof_fit_basic() {
        // Normal data clustered around (2, 2)
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                1.9, 1.9, 2.1, 1.8,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new().with_n_neighbors(5);
        lof.fit(&data).expect("LOF fit should succeed");
        assert!(lof.is_fitted());
    }

    #[test]
    fn test_lof_predict_anomalies() {
        // 8 normal points + 2 outliers
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                10.0, 10.0, // Outlier 1
                -10.0, -10.0, // Outlier 2
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new()
            .with_n_neighbors(5)
            .with_contamination(0.2);
        lof.fit(&data).expect("LOF fit should succeed");

        let predictions = lof.predict(&data);
        assert_eq!(predictions.len(), 10);

        // Check that predictions are either 1 (normal) or -1 (anomaly)
        for &pred in &predictions {
            assert!(pred == 1 || pred == -1);
        }

        // Should detect approximately 2 anomalies (20% contamination)
        let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
        assert!((1..=3).contains(&n_anomalies));
    }

    #[test]
    fn test_lof_score_samples() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
                10.0, 10.0, // Outlier 1
                -10.0, -10.0, // Outlier 2
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
        lof.fit(&data).expect("LOF fit should succeed");

        let scores = lof.score_samples(&data);
        assert_eq!(scores.len(), 6);

        // Outliers should have higher LOF scores than normal points
        let normal_avg = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
        let outlier_avg = (scores[4] + scores[5]) / 2.0;
        assert!(outlier_avg > normal_avg);
    }

    #[test]
    fn test_lof_negative_outlier_factor() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
        lof.fit(&data).expect("LOF fit should succeed");

        let nof = lof.negative_outlier_factor();
        assert_eq!(nof.len(), 6);

        // Negative outlier factor should be opposite sign of LOF scores
        let scores = lof.score_samples(&data);
        for i in 0..6 {
            // NOF should be negative of LOF (approximately)
            assert!(nof[i] < 0.0 || scores[i] < 1.0);
        }
    }

    #[test]
    fn test_lof_contamination() {
        let data = Matrix::from_vec(
            10,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
                10.0, 10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        // Low contamination (10%) - fewer anomalies expected
        let mut lof_low = LocalOutlierFactor::new()
            .with_contamination(0.1)
            .with_n_neighbors(5);
        lof_low.fit(&data).expect("LOF fit should succeed");
        let pred_low = lof_low.predict(&data);
        let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();

        // High contamination (30%) - more anomalies expected
        let mut lof_high = LocalOutlierFactor::new()
            .with_contamination(0.3)
            .with_n_neighbors(5);
        lof_high.fit(&data).expect("LOF fit should succeed");
        let pred_high = lof_high.predict(&data);
        let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();

        // Higher contamination should detect more or equal anomalies
        assert!(anomalies_high >= anomalies_low);
    }

    #[test]
    fn test_lof_n_neighbors() {
        let data = Matrix::from_vec(
            8,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0,
                -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        // Fewer neighbors
        let mut lof_few = LocalOutlierFactor::new().with_n_neighbors(3);
        lof_few.fit(&data).expect("LOF fit should succeed");
        let scores_few = lof_few.score_samples(&data);

        // More neighbors
        let mut lof_many = LocalOutlierFactor::new().with_n_neighbors(5);
        lof_many.fit(&data).expect("LOF fit should succeed");
        let scores_many = lof_many.score_samples(&data);

        // Both should work and produce scores
        assert_eq!(scores_few.len(), 8);
        assert_eq!(scores_many.len(), 8);

        // Scores should be different (different neighborhood sizes)
        let diff_exists = scores_few
            .iter()
            .zip(scores_many.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(diff_exists);
    }

    #[test]
    fn test_lof_varying_density_clusters() {
        // Two clusters with different densities
        // Cluster 1: Dense (points close together)
        // Cluster 2: Sparse (points far apart)
        // Outlier: Between clusters
        let data = Matrix::from_vec(
            9,
            2,
            vec![
                // Dense cluster (4 points around 0,0)
                0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.0,
                0.1, // Sparse cluster (3 points around 10,10)
                10.0, 10.0, 12.0, 12.0, 11.0, 9.0, // Outlier between clusters
                5.0, 5.0, // Another outlier
                5.5, 5.5,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new()
            .with_n_neighbors(3)
            .with_contamination(0.2);
        lof.fit(&data).expect("LOF fit should succeed");

        let scores = lof.score_samples(&data);
        let predictions = lof.predict(&data);

        // LOF should detect outliers in varying density regions
        // Points 7 and 8 (between clusters) should have higher LOF scores
        assert!(scores[7] > 1.0 || scores[8] > 1.0);

        // Should detect some anomalies
        let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
        assert!(n_anomalies >= 1);
    }

    #[test]
    fn test_lof_lof_score_interpretation() {
        let data = Matrix::from_vec(
            5,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal cluster
                10.0, 10.0, // Clear outlier
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
        lof.fit(&data).expect("LOF fit should succeed");

        let scores = lof.score_samples(&data);

        // LOF ≈ 1: similar density to neighbors (normal)
        // LOF >> 1: lower density than neighbors (outlier)
        let normal_scores = &scores[0..4];
        let outlier_score = scores[4];

        // Normal points should have LOF close to 1
        for &score in normal_scores {
            assert!((0.5..2.0).contains(&score));
        }

        // Outlier should have LOF > 1 (significantly)
        assert!(outlier_score > 1.5);
    }

    #[test]
    fn test_lof_all_normal() {
        // All points are normal (tightly clustered)
        let data = Matrix::from_vec(
            6,
            2,
            vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new()
            .with_contamination(0.1)
            .with_n_neighbors(3);
        lof.fit(&data).expect("LOF fit should succeed");

        let predictions = lof.predict(&data);
        let scores = lof.score_samples(&data);

        // All LOF scores should be close to 1 (similar density)
        for &score in &scores {
            assert!((0.5..1.5).contains(&score));
        }

        // With 10% contamination, expect mostly normal points
        let n_normal = predictions.iter().filter(|&&p| p == 1).count();
        assert!(n_normal >= 5);
    }

    #[test]
    fn test_lof_score_samples_finite() {
        let data = Matrix::from_vec(4, 2, vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 10.0, 10.0])
            .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new().with_n_neighbors(2);
        lof.fit(&data).expect("LOF fit should succeed");

        let scores = lof.score_samples(&data);
        // All LOF scores should be finite
        for &score in &scores {
            assert!(score.is_finite());
            assert!(score > 0.0); // LOF is always positive
        }
    }

    #[test]
    fn test_lof_multidimensional() {
        // Test with more features
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.0, 2.0, 3.0, 0.9, 1.9, 2.9, 10.0, 10.0, 10.0,
                -10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new()
            .with_contamination(0.3)
            .with_n_neighbors(3);
        lof.fit(&data).expect("LOF fit should succeed");

        let predictions = lof.predict(&data);
        let scores = lof.score_samples(&data);

        assert_eq!(predictions.len(), 6);
        assert_eq!(scores.len(), 6);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_lof_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let lof = LocalOutlierFactor::new();
        let _ = lof.predict(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_lof_score_samples_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let lof = LocalOutlierFactor::new();
        let _ = lof.score_samples(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_lof_negative_outlier_factor_before_fit() {
        let lof = LocalOutlierFactor::new();
        let _ = lof.negative_outlier_factor(); // Should panic
    }

    #[test]
    fn test_lof_decision_consistency() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut lof = LocalOutlierFactor::new()
            .with_contamination(0.3)
            .with_n_neighbors(3);
        lof.fit(&data).expect("LOF fit should succeed");

        let predictions = lof.predict(&data);
        let scores = lof.score_samples(&data);

        // Points with higher LOF scores should be more likely to be anomalies
        assert_eq!(predictions.len(), scores.len());
    }

    // ========================================================================
    // Spectral Clustering Tests
    // ========================================================================

    #[test]
    fn test_spectral_clustering_new() {
        let sc = SpectralClustering::new(3);
        assert!(!sc.is_fitted());
    }

    #[test]
    fn test_spectral_clustering_fit_basic() {
        // Simple 2-cluster data
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, 1.1, 1.0, 0.9, 1.1, // Cluster 1
                5.0, 5.0, 5.1, 5.0, 4.9, 5.1, // Cluster 2
            ],
        )
        .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");
        assert!(sc.is_fitted());
    }

    #[test]
    fn test_spectral_clustering_predict() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 5.0, 5.0, 5.1, 5.0, 4.9, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels = sc.predict(&data);
        assert_eq!(labels.len(), 6);

        // Points in same cluster should have same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);

        // Different clusters should have different labels (with label permutation tolerance)
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_spectral_clustering_non_convex() {
        // Create two moon-shaped clusters (non-convex but separable)
        // Upper moon
        let upper: Vec<f32> = vec![0.0, 2.0, 0.5, 2.0, 1.0, 1.9, 1.5, 1.7, 2.0, 1.5];
        // Lower moon
        let lower: Vec<f32> = vec![0.5, 0.3, 1.0, 0.1, 1.5, 0.0, 2.0, 0.0, 2.5, 0.2];

        let mut all_data = upper.clone();
        all_data.extend(lower);

        let data = Matrix::from_vec(10, 2, all_data).expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2)
            .with_affinity(Affinity::KNN)
            .with_n_neighbors(3);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels = sc.predict(&data);

        // Upper moon points should mostly be in same cluster
        // Allow some flexibility for this challenging case
        let upper_cluster = labels[0];
        let same_cluster_count = (0..5).filter(|&i| labels[i] == upper_cluster).count();
        assert!(same_cluster_count >= 4); // At least 4 out of 5

        // Lower moon points should mostly be in same cluster
        let lower_cluster = labels[5];
        let same_cluster_count = (5..10).filter(|&i| labels[i] == lower_cluster).count();
        assert!(same_cluster_count >= 4); // At least 4 out of 5

        // The two moons should be in different clusters
        assert_ne!(upper_cluster, lower_cluster);
    }

    #[test]
    fn test_spectral_clustering_rbf_affinity() {
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2)
            .with_affinity(Affinity::RBF)
            .with_gamma(1.0);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels = sc.predict(&data);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
    }

    #[test]
    fn test_spectral_clustering_knn_affinity() {
        let data = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 5.0, 5.0, 5.1, 5.0, 4.9, 5.1],
        )
        .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2)
            .with_affinity(Affinity::KNN)
            .with_n_neighbors(3);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels = sc.predict(&data);
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_spectral_clustering_gamma_effect() {
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0])
            .expect("Matrix creation should succeed");

        // Small gamma - more global similarity
        let mut sc_small = SpectralClustering::new(2).with_gamma(0.1);
        sc_small
            .fit(&data)
            .expect("Spectral Clustering fit should succeed");

        // Large gamma - more local similarity
        let mut sc_large = SpectralClustering::new(2).with_gamma(10.0);
        sc_large
            .fit(&data)
            .expect("Spectral Clustering fit should succeed");

        // Both should work
        assert!(sc_small.is_fitted());
        assert!(sc_large.is_fitted());
    }

    #[test]
    fn test_spectral_clustering_multiple_clusters() {
        let data = Matrix::from_vec(
            9,
            2,
            vec![
                // Cluster 1
                0.0, 0.0, 0.1, 0.1, -0.1, -0.1, // Cluster 2
                5.0, 5.0, 5.1, 5.1, 4.9, 4.9, // Cluster 3
                10.0, 10.0, 10.1, 10.1, 9.9, 9.9,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(3);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels = sc.predict(&data);
        assert_eq!(labels.len(), 9);

        // Check that we have 3 distinct clusters
        let mut unique_labels: Vec<usize> = labels.clone();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        assert_eq!(unique_labels.len(), 3);
    }

    #[test]
    fn test_spectral_clustering_labels_consistency() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1])
            .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels1 = sc.predict(&data);
        let labels2 = sc.labels().clone();

        assert_eq!(labels1, labels2);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_spectral_clustering_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0])
            .expect("Matrix creation should succeed");
        let sc = SpectralClustering::new(2);
        let _ = sc.predict(&data);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_spectral_clustering_labels_before_fit() {
        let sc = SpectralClustering::new(2);
        let _ = sc.labels();
    }

    #[test]
    fn test_spectral_clustering_well_separated() {
        // Very well-separated clusters
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 100.0, 100.0, 100.1, 100.1, 100.0, 100.1,
            ],
        )
        .expect("Matrix creation should succeed");

        let mut sc = SpectralClustering::new(2);
        sc.fit(&data)
            .expect("Spectral Clustering fit should succeed");

        let labels = sc.predict(&data);

        // Should clearly separate the two clusters
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }
