//! Clustering algorithms.
//!
//! Includes K-Means clustering with k-means++ initialization.

use crate::metrics::inertia;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

/// K-Means clustering algorithm.
///
/// Uses Lloyd's algorithm with k-means++ initialization for faster convergence.
///
/// # Algorithm
///
/// 1. Initialize centroids using k-means++
/// 2. Assign each sample to nearest centroid
/// 3. Update centroids as mean of assigned samples
/// 4. Repeat until convergence or max iterations
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(6, 2, vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     5.0, 8.0,
///     8.0, 8.0,
///     1.0, 0.6,
///     9.0, 11.0,
/// ]).unwrap();
///
/// let mut kmeans = KMeans::new(2);
/// kmeans.fit(&data).unwrap();
///
/// let labels = kmeans.predict(&data);
/// assert_eq!(labels.len(), 6);
/// ```
///
/// # Performance
///
/// - Time complexity: O(nkdi) where n=samples, k=clusters, d=features, i=iterations
/// - Space complexity: O(nk)
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Number of clusters.
    n_clusters: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f32,
    /// Random seed for initialization.
    random_state: Option<u64>,
    /// Cluster centroids after fitting.
    centroids: Option<Matrix<f32>>,
    /// Labels for training data.
    labels: Option<Vec<usize>>,
    /// Sum of squared distances (inertia).
    inertia: f32,
    /// Number of iterations run.
    n_iter: usize,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(8)
    }
}

impl KMeans {
    /// Creates a new K-Means with the specified number of clusters.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            centroids: None,
            labels: None,
            inertia: 0.0,
            n_iter: 0,
        }
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Returns the cluster centroids.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn centroids(&self) -> &Matrix<f32> {
        self.centroids
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the inertia (within-cluster sum of squares).
    #[must_use]
    pub fn inertia(&self) -> f32 {
        self.inertia
    }

    /// Returns the number of iterations run.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.centroids.is_some()
    }

    /// Initializes centroids using k-means++ algorithm.
    fn kmeans_plusplus_init(&self, x: &Matrix<f32>) -> Matrix<f32> {
        let (n_samples, n_features) = x.shape();
        let mut centroids_data = Vec::with_capacity(self.n_clusters * n_features);

        // Simple deterministic initialization based on seed
        let seed = self.random_state.unwrap_or(42);
        let first_idx = (seed as usize) % n_samples;

        // First centroid: chosen based on seed
        for j in 0..n_features {
            centroids_data.push(x.get(first_idx, j));
        }

        // Remaining centroids: k-means++ selection
        for _ in 1..self.n_clusters {
            // Compute distances to nearest centroid for each point
            let n_current = centroids_data.len() / n_features;
            let mut min_distances = vec![f32::INFINITY; n_samples];

            for (i, min_dist) in min_distances.iter_mut().enumerate() {
                for c in 0..n_current {
                    let mut dist_sq = 0.0;
                    for j in 0..n_features {
                        let diff = x.get(i, j) - centroids_data[c * n_features + j];
                        dist_sq += diff * diff;
                    }
                    if dist_sq < *min_dist {
                        *min_dist = dist_sq;
                    }
                }
            }

            // Select point with probability proportional to D²
            // For deterministic behavior, pick the point with max distance
            let mut max_dist = 0.0;
            let mut max_idx = 0;
            for (i, &dist) in min_distances.iter().enumerate() {
                if dist > max_dist {
                    max_dist = dist;
                    max_idx = i;
                }
            }

            // Add new centroid
            for j in 0..n_features {
                centroids_data.push(x.get(max_idx, j));
            }
        }

        Matrix::from_vec(self.n_clusters, n_features, centroids_data)
            .expect("Internal error: centroid matrix creation failed")
    }

    /// Assigns each sample to the nearest centroid.
    fn assign_labels(&self, x: &Matrix<f32>, centroids: &Matrix<f32>) -> Vec<usize> {
        let n_samples = x.n_rows();
        let mut labels = vec![0; n_samples];

        for (i, label) in labels.iter_mut().enumerate() {
            let point = x.row(i);
            let mut min_dist = f32::INFINITY;
            let mut min_cluster = 0;

            for k in 0..self.n_clusters {
                let centroid = centroids.row(k);
                let diff = &point - &centroid;
                let dist = diff.norm_squared();

                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = k;
                }
            }

            *label = min_cluster;
        }

        labels
    }

    /// Updates centroids as the mean of assigned samples.
    fn update_centroids(&self, x: &Matrix<f32>, labels: &[usize]) -> Matrix<f32> {
        let (_, n_features) = x.shape();
        let mut new_centroids = vec![0.0; self.n_clusters * n_features];
        let mut counts = vec![0usize; self.n_clusters];

        // Sum points in each cluster
        for (i, &label) in labels.iter().enumerate() {
            counts[label] += 1;
            for j in 0..n_features {
                new_centroids[label * n_features + j] += x.get(i, j);
            }
        }

        // Compute means
        for k in 0..self.n_clusters {
            if counts[k] > 0 {
                for j in 0..n_features {
                    new_centroids[k * n_features + j] /= counts[k] as f32;
                }
            }
        }

        Matrix::from_vec(self.n_clusters, n_features, new_centroids)
            .expect("Internal error: centroid update failed")
    }

    /// Checks if centroids have converged.
    fn centroids_converged(&self, old: &Matrix<f32>, new: &Matrix<f32>) -> bool {
        let (n_clusters, n_features) = old.shape();

        for k in 0..n_clusters {
            let mut dist_sq = 0.0;
            for j in 0..n_features {
                let diff = old.get(k, j) - new.get(k, j);
                dist_sq += diff * diff;
            }
            if dist_sq > self.tol * self.tol {
                return false;
            }
        }

        true
    }
}

impl UnsupervisedEstimator for KMeans {
    type Labels = Vec<usize>;

    /// Fits the K-Means model to data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data has fewer samples than clusters
    /// - Data is empty
    fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str> {
        let n_samples = x.n_rows();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples");
        }

        if n_samples < self.n_clusters {
            return Err("Number of samples must be >= number of clusters");
        }

        // Initialize centroids using k-means++
        let mut centroids = self.kmeans_plusplus_init(x);

        let mut labels = vec![0; n_samples];

        for iter in 0..self.max_iter {
            // Assign samples to nearest centroid
            labels = self.assign_labels(x, &centroids);

            // Update centroids
            let new_centroids = self.update_centroids(x, &labels);

            // Check convergence
            if self.centroids_converged(&centroids, &new_centroids) {
                self.n_iter = iter + 1;
                centroids = new_centroids;
                break;
            }

            centroids = new_centroids;
            self.n_iter = iter + 1;
        }

        // Compute final inertia
        self.inertia = inertia(x, &centroids, &labels);
        self.labels = Some(labels);
        self.centroids = Some(centroids);

        Ok(())
    }

    /// Predicts cluster labels for new data.
    fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let centroids = self
            .centroids
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        self.assign_labels(x, centroids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Matrix<f32> {
        // Two well-separated clusters
        Matrix::from_vec(
            6,
            2,
            vec![1.0, 2.0, 1.5, 1.8, 1.0, 0.6, 8.0, 8.0, 9.0, 11.0, 8.5, 9.0],
        )
        .unwrap()
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
        kmeans.fit(&data).unwrap();

        assert!(kmeans.is_fitted());
        assert_eq!(kmeans.centroids().shape(), (2, 2));
        assert!(kmeans.inertia() >= 0.0);
    }

    #[test]
    fn test_predict() {
        let data = sample_data();
        let mut kmeans = KMeans::new(2);
        kmeans.fit(&data).unwrap();

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
        kmeans.fit(&data).unwrap();

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
        let data = Matrix::from_vec(0, 2, vec![]).unwrap();
        let mut kmeans = KMeans::new(2);
        let result = kmeans.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_many_clusters_error() {
        let data = Matrix::from_vec(3, 2, vec![1.0; 6]).unwrap();
        let mut kmeans = KMeans::new(5);
        let result = kmeans.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_cluster() {
        let data = sample_data();
        let mut kmeans = KMeans::new(1);
        kmeans.fit(&data).unwrap();

        let labels = kmeans.predict(&data);
        // All points should be in cluster 0
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_inertia_decreases_with_more_clusters() {
        let data = sample_data();

        let mut kmeans1 = KMeans::new(1).with_random_state(42);
        kmeans1.fit(&data).unwrap();
        let inertia1 = kmeans1.inertia();

        let mut kmeans2 = KMeans::new(2).with_random_state(42);
        kmeans2.fit(&data).unwrap();
        let inertia2 = kmeans2.inertia();

        // More clusters should lead to lower or equal inertia
        assert!(inertia2 <= inertia1);
    }

    #[test]
    fn test_reproducibility() {
        let data = sample_data();

        let mut kmeans1 = KMeans::new(2).with_random_state(42);
        kmeans1.fit(&data).unwrap();

        let mut kmeans2 = KMeans::new(2).with_random_state(42);
        kmeans2.fit(&data).unwrap();

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
        kmeans.fit(&data).unwrap();

        // Should converge before max iterations for simple data
        assert!(kmeans.n_iter() < 100);
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
        kmeans.fit(&data).unwrap();

        let labels = kmeans.predict(&data);
        let max_label = labels.iter().max().unwrap();
        assert!(*max_label < 3);
    }

    #[test]
    fn test_predict_new_data() {
        let data = sample_data();
        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

        // Predict on new data point
        let new_point = Matrix::from_vec(1, 2, vec![1.2, 1.5]).unwrap();
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

        let matrix = Matrix::from_vec(n, 2, data).unwrap();
        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&matrix).unwrap();

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
        .unwrap();

        let mut kmeans = KMeans::new(3).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        let data =
            Matrix::from_vec(5, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        let data = Matrix::from_vec(6, 1, vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2]).unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        let data = Matrix::from_vec(3, 2, vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();

        let mut kmeans = KMeans::new(3).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        kmeans.fit(&data).unwrap();

        // Should stop at 1 iteration
        assert_eq!(kmeans.n_iter(), 1);
    }

    #[test]
    fn test_tight_tolerance() {
        let data = sample_data();
        let mut kmeans = KMeans::new(2).with_tol(1e-10).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

        let centroids = kmeans.centroids();
        assert_eq!(centroids.shape(), (2, 3));
    }

    #[test]
    fn test_different_random_states() {
        let data = sample_data();

        let mut kmeans1 = KMeans::new(2).with_random_state(1);
        kmeans1.fit(&data).unwrap();

        let mut kmeans2 = KMeans::new(2).with_random_state(999);
        kmeans2.fit(&data).unwrap();

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
}
