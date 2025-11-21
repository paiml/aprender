//! Clustering algorithms.
//!
//! Includes K-Means clustering with k-means++ initialization.

use crate::error::Result;
use crate::metrics::inertia;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {}", e))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {}", e))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {}", e))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {}", e))?;
        Ok(model)
    }

    /// Saves the K-Means model to a SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the SafeTensors file will be saved
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unfitted or if saving fails.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Check if model is fitted
        let centroids = self
            .centroids
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        let mut tensors = BTreeMap::new();

        // Save centroids matrix as flat array
        let (n_clusters, n_features) = centroids.shape();
        let mut centroids_data = Vec::with_capacity(n_clusters * n_features);
        for i in 0..n_clusters {
            for j in 0..n_features {
                centroids_data.push(centroids.get(i, j));
            }
        }
        tensors.insert(
            "centroids".to_string(),
            (centroids_data, vec![n_clusters, n_features]),
        );

        // Save hyperparameters
        tensors.insert(
            "n_clusters".to_string(),
            (vec![self.n_clusters as f32], vec![1]),
        );
        tensors.insert(
            "max_iter".to_string(),
            (vec![self.max_iter as f32], vec![1]),
        );
        tensors.insert("tol".to_string(), (vec![self.tol], vec![1]));

        let random_state_val = if let Some(state) = self.random_state {
            state as f32
        } else {
            -1.0
        };
        tensors.insert(
            "random_state".to_string(),
            (vec![random_state_val], vec![1]),
        );

        // Save metadata
        tensors.insert("inertia".to_string(), (vec![self.inertia], vec![1]));
        tensors.insert("n_iter".to_string(), (vec![self.n_iter as f32], vec![1]));

        safetensors::save_safetensors(path, tensors)?;
        Ok(())
    }

    /// Loads a K-Means model from a SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SafeTensors file
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails or if the file format is invalid.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract centroids tensor
        let centroids_meta = metadata
            .get("centroids")
            .ok_or("Missing 'centroids' tensor in SafeTensors file")?;
        let centroids_data = safetensors::extract_tensor(&raw_data, centroids_meta)?;

        // Get shape from metadata
        let shape = &centroids_meta.shape;
        if shape.len() != 2 {
            return Err("Invalid centroids tensor shape".to_string());
        }
        let n_clusters_from_shape = shape[0];
        let n_features = shape[1];

        // Reconstruct centroids matrix
        let centroids = Matrix::from_vec(n_clusters_from_shape, n_features, centroids_data)
            .map_err(|e| format!("Failed to reconstruct centroids matrix: {}", e))?;

        // Load hyperparameters
        let n_clusters_meta = metadata
            .get("n_clusters")
            .ok_or("Missing 'n_clusters' tensor")?;
        let n_clusters_data = safetensors::extract_tensor(&raw_data, n_clusters_meta)?;
        let n_clusters = n_clusters_data[0] as usize;

        let max_iter_meta = metadata
            .get("max_iter")
            .ok_or("Missing 'max_iter' tensor")?;
        let max_iter_data = safetensors::extract_tensor(&raw_data, max_iter_meta)?;
        let max_iter = max_iter_data[0] as usize;

        let tol_meta = metadata.get("tol").ok_or("Missing 'tol' tensor")?;
        let tol_data = safetensors::extract_tensor(&raw_data, tol_meta)?;
        let tol = tol_data[0];

        let random_state_meta = metadata
            .get("random_state")
            .ok_or("Missing 'random_state' tensor")?;
        let random_state_data = safetensors::extract_tensor(&raw_data, random_state_meta)?;
        let random_state = if random_state_data[0] < 0.0 {
            None
        } else {
            Some(random_state_data[0] as u64)
        };

        // Load metadata
        let inertia_meta = metadata.get("inertia").ok_or("Missing 'inertia' tensor")?;
        let inertia_data = safetensors::extract_tensor(&raw_data, inertia_meta)?;
        let inertia = inertia_data[0];

        let n_iter_meta = metadata.get("n_iter").ok_or("Missing 'n_iter' tensor")?;
        let n_iter_data = safetensors::extract_tensor(&raw_data, n_iter_meta)?;
        let n_iter = n_iter_data[0] as usize;

        Ok(Self {
            n_clusters,
            max_iter,
            tol,
            random_state,
            centroids: Some(centroids),
            labels: None, // Training labels not serialized
            inertia,
            n_iter,
        })
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
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.n_rows();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        if n_samples < self.n_clusters {
            return Err("Number of samples must be >= number of clusters".into());
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

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
///
/// Density-based clustering algorithm that can find arbitrarily-shaped clusters
/// and identify outliers as noise points.
///
/// # Algorithm
///
/// 1. For each unvisited point:
///    - Find all neighbors within eps distance
///    - If neighbors < min_samples: mark as noise
///    - Else: create new cluster and expand recursively
/// 2. Noise points are labeled as -1
/// 3. Clusters are labeled 0, 1, 2, ...
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(7, 2, vec![
///     1.0, 1.0,  // Cluster 0
///     1.2, 1.1,  // Cluster 0
///     1.1, 1.2,  // Cluster 0
///     5.0, 5.0,  // Cluster 1
///     5.1, 5.2,  // Cluster 1
///     5.2, 5.1,  // Cluster 1
///     10.0, 10.0, // Noise
/// ]).unwrap();
///
/// let mut dbscan = DBSCAN::new(0.5, 2);
/// dbscan.fit(&data).unwrap();
///
/// let labels = dbscan.labels().unwrap();
/// assert_eq!(labels[6], -1); // Last point is noise
/// ```
///
/// # Performance
///
/// - Time complexity: O(n²) for distance computations
/// - Space complexity: O(n)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBSCAN {
    /// Maximum distance between two samples to be neighbors.
    eps: f32,
    /// Minimum number of samples in a neighborhood to form a core point.
    min_samples: usize,
    /// Cluster labels after fitting (-1 for noise).
    labels: Option<Vec<i32>>,
}

impl DBSCAN {
    /// Creates a new DBSCAN with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `eps` - Maximum distance between neighbors
    /// * `min_samples` - Minimum points to form a dense region
    #[must_use]
    pub fn new(eps: f32, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels: None,
        }
    }

    /// Returns the eps parameter.
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Returns the min_samples parameter.
    #[must_use]
    pub fn min_samples(&self) -> usize {
        self.min_samples
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }

    /// Returns the cluster labels (-1 for noise).
    ///
    /// # Panics
    ///
    /// Panics if the model has not been fitted.
    #[must_use]
    pub fn labels(&self) -> &Vec<i32> {
        self.labels
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Finds all neighbors within eps distance of point i.
    fn region_query(&self, x: &Matrix<f32>, i: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let n_samples = x.shape().0;

        for j in 0..n_samples {
            let dist = self.euclidean_distance(x, i, j);
            if dist <= self.eps {
                neighbors.push(j);
            }
        }

        neighbors
    }

    /// Computes Euclidean distance between samples i and j.
    fn euclidean_distance(&self, x: &Matrix<f32>, i: usize, j: usize) -> f32 {
        let n_features = x.shape().1;
        let mut sum = 0.0;

        for k in 0..n_features {
            let diff = x.get(i, k) - x.get(j, k);
            sum += diff * diff;
        }

        sum.sqrt()
    }

    /// Expands a cluster from a core point.
    fn expand_cluster(
        &self,
        x: &Matrix<f32>,
        labels: &mut [i32],
        point: usize,
        neighbors: &mut Vec<usize>,
        cluster_id: i32,
    ) {
        labels[point] = cluster_id;

        let mut i = 0;
        while i < neighbors.len() {
            let neighbor = neighbors[i];

            // If unlabeled or noise, assign to cluster
            if labels[neighbor] == -2 {
                labels[neighbor] = cluster_id;

                // If core point, add its neighbors to expansion
                let neighbor_neighbors = self.region_query(x, neighbor);
                if neighbor_neighbors.len() >= self.min_samples {
                    for &nn in &neighbor_neighbors {
                        if !neighbors.contains(&nn) {
                            neighbors.push(nn);
                        }
                    }
                }
            } else if labels[neighbor] == -1 {
                // Border point: noise becomes part of cluster
                labels[neighbor] = cluster_id;
            }

            i += 1;
        }
    }
}

impl UnsupervisedEstimator for DBSCAN {
    type Labels = Vec<i32>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.shape().0;
        let mut labels = vec![-2; n_samples]; // -2 = unlabeled
        let mut cluster_id = 0;

        for i in 0..n_samples {
            // Skip if already processed
            if labels[i] != -2 {
                continue;
            }

            // Find neighbors
            let mut neighbors = self.region_query(x, i);

            // Not a core point -> mark as noise (for now)
            if neighbors.len() < self.min_samples {
                labels[i] = -1;
                continue;
            }

            // Core point -> expand cluster
            self.expand_cluster(x, &mut labels, i, &mut neighbors, cluster_id);
            cluster_id += 1;
        }

        self.labels = Some(labels);
        Ok(())
    }

    fn predict(&self, _x: &Matrix<f32>) -> Self::Labels {
        // For DBSCAN, predict returns the fitted labels
        // (new points would require a different strategy)
        self.labels().clone()
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
    fn test_centroids_converged_within_tolerance() {
        // Test when centroids have converged (within tolerance)
        let kmeans = KMeans::new(2).with_tol(0.01);

        // Old centroids: [[1.0, 2.0], [3.0, 4.0]]
        let old = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

        // New centroids: [[1.005, 2.005], [3.005, 4.005]]
        // Distance per centroid: sqrt(0.005^2 + 0.005^2) ≈ 0.00707 < 0.01
        let new = Matrix::from_vec(2, 2, vec![1.005_f32, 2.005, 3.005, 4.005]).unwrap();

        assert!(kmeans.centroids_converged(&old, &new));
    }

    #[test]
    fn test_centroids_not_converged() {
        // Test when centroids have not converged (beyond tolerance)
        let kmeans = KMeans::new(2).with_tol(0.01);

        // Old centroids: [[1.0, 2.0], [3.0, 4.0]]
        let old = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

        // New centroids: [[1.1, 2.1], [3.0, 4.0]]
        // First centroid distance: sqrt(0.1^2 + 0.1^2) ≈ 0.141 > 0.01
        let new = Matrix::from_vec(2, 2, vec![1.1_f32, 2.1, 3.0, 4.0]).unwrap();

        assert!(!kmeans.centroids_converged(&old, &new));
    }

    #[test]
    fn test_centroids_converged_exact_tolerance() {
        // Test boundary case: distance exactly at tolerance²
        // Use tol=0.1, so tol²=0.01
        // Set up distance² to be exactly 0.01
        let kmeans = KMeans::new(1).with_tol(0.1);

        // Old centroid: [[0.0, 0.0]]
        let old = Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).unwrap();

        // New centroid: [[0.1, 0.0]]
        // Distance²: 0.1² + 0.0² = 0.01 (exactly tol²)
        // Should be converged (dist² = tol² means dist = tol, which is at boundary)
        // Original code: dist² > tol² is false, so converged ✓
        // Mutated code: dist² >= tol² is true, so NOT converged ✗
        let new_exact = Matrix::from_vec(1, 2, vec![0.1_f32, 0.0]).unwrap();
        assert!(
            kmeans.centroids_converged(&old, &new_exact),
            "Distance exactly at tolerance should be converged"
        );

        // Now test just beyond tolerance
        // Distance²: 0.11² ≈ 0.0121 > 0.01
        let new_beyond = Matrix::from_vec(1, 2, vec![0.11_f32, 0.0]).unwrap();
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
        .unwrap();

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
        .unwrap();
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
        .unwrap();
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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
            "Centroids should be well-separated (dist > 5.0), got dist² = {}",
            dist_sq
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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
            "Centroids should be separated by > 5.0, got {}",
            diff
        );
    }

    #[test]
    fn test_initialization_reproducibility() {
        // Verify that same random_state produces same initialization
        // This indirectly tests the distance and selection logic
        let data = sample_data();

        let mut kmeans1 = KMeans::new(2).with_random_state(42);
        let mut kmeans2 = KMeans::new(2).with_random_state(42);

        kmeans1.fit(&data).unwrap();
        kmeans2.fit(&data).unwrap();

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
    fn test_n_iter_not_one() {
        // Test that catches n_iter() → 1 mutation (line 132)
        // Use data that requires multiple iterations to converge
        let data = sample_data();
        let mut kmeans = KMeans::new(2).with_max_iter(100).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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

    #[test]
    fn test_save_load() {
        use std::fs;
        use std::path::Path;

        let data = sample_data();
        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
                    "Centroid mismatch at ({}, {})",
                    i,
                    j
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
        let old = Matrix::from_vec(1, 1, vec![0.0_f32]).unwrap();
        let new = Matrix::from_vec(1, 1, vec![0.3_f32]).unwrap();
        assert!(
            kmeans.centroids_converged(&old, &new),
            "Should converge when dist² (0.09) < tol² (0.25)"
        );

        // 2D case: movement of [0.4, 0.3]
        // dist_sq = 0.16 + 0.09 = 0.25 = tol², should converge (<=)
        let old_2d = Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).unwrap();
        let new_2d = Matrix::from_vec(1, 2, vec![0.4_f32, 0.3]).unwrap();
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
        let old = Matrix::from_vec(1, 1, vec![0.0_f32]).unwrap();
        let less = Matrix::from_vec(1, 1, vec![0.7_f32]).unwrap(); // dist_sq ≈ 0.49
        assert!(
            kmeans.centroids_converged(&old, &less),
            "dist² < tol² should converge"
        );

        // Case: dist_sq = 1.5, tol² = 1.0
        // Should NOT converge because 1.5 > 1.0
        let more = Matrix::from_vec(1, 1, vec![1.3_f32]).unwrap(); // dist_sq ≈ 1.69
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
        .unwrap();

        // Use very tight tolerance to force multiple iterations
        let mut kmeans = KMeans::new(2)
            .with_tol(1e-8)
            .with_max_iter(50)
            .with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        let data = Matrix::from_vec(3, 2, vec![0.0, 0.0, 2.0, 0.0, 1.0, 0.0]).unwrap();

        kmeans.fit(&data).unwrap();

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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

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
        .unwrap();

        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&data).unwrap();

        let centroids = kmeans.centroids();

        // Centroids should be at means (approximately 1.0 and 11.0)
        let c0 = centroids.get(0, 0);
        let c1 = centroids.get(1, 0);

        // One centroid should be near 1.0, other near 11.0
        let has_low = (c0 - 1.0).abs() < 1.0 || (c1 - 1.0).abs() < 1.0;
        let has_high = (c0 - 11.0).abs() < 1.0 || (c1 - 11.0).abs() < 1.0;

        assert!(
            has_low,
            "Should have centroid near 1.0, got {} and {}",
            c0, c1
        );
        assert!(
            has_high,
            "Should have centroid near 11.0, got {} and {}",
            c0, c1
        );
    }

    // EXTREME TDD: Additional mutation-killing tests for centroids_converged

    #[test]
    fn test_centroids_converged_squaring_not_division() {
        // MUTATION TARGET: "replace * with / in diff * diff"
        // Tests that we compute diff² correctly, not diff/diff
        let kmeans = KMeans::new(1).with_tol(1.0);

        // With diff = 0.5, if using * we get 0.25, if using / we get 1.0
        let old = Matrix::from_vec(1, 1, vec![0.0_f32]).unwrap();
        let new = Matrix::from_vec(1, 1, vec![0.5_f32]).unwrap();

        // dist_sq = 0.5² = 0.25 < 1.0², should converge
        assert!(
            kmeans.centroids_converged(&old, &new),
            "With diff=0.5, diff²=0.25 < tol²=1.0, should converge"
        );

        // If mutation uses diff/diff, we get dist_sq=1.0 = tol², still converges
        // Need another case: diff = 2.0
        let new2 = Matrix::from_vec(1, 1, vec![2.0_f32]).unwrap();
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
        let old = Matrix::from_vec(1, 2, vec![0.0_f32, 0.0]).unwrap();
        let new = Matrix::from_vec(1, 2, vec![0.3_f32, 0.4]).unwrap();

        // dist = √(0.25) = 0.5 < 0.6, should converge
        assert!(
            kmeans.centroids_converged(&old, &new),
            "dist²=0.25 < tol²=0.36, should converge"
        );

        // If mutation uses *=, dist_sq stays 0.0, would always converge
        // Test case that should NOT converge:
        let new2 = Matrix::from_vec(1, 2, vec![0.5_f32, 0.5]).unwrap();
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
        let old = Matrix::from_vec(1, 1, vec![0.0_f32]).unwrap();
        let new = Matrix::from_vec(1, 1, vec![0.6_f32]).unwrap();

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
        let old = Matrix::from_vec(1, 1, vec![0.0_f32]).unwrap();
        let new = Matrix::from_vec(1, 1, vec![1.3_f32]).unwrap(); // dist_sq ≈ 1.69

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
        let old = Matrix::from_vec(1, 1, vec![0.0_f32]).unwrap();
        let new = Matrix::from_vec(1, 1, vec![0.7_f32]).unwrap(); // dist_sq = 0.49

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
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).unwrap();

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
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).unwrap();

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
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).unwrap();

        let labels = dbscan.labels();

        // Last sample should be noise (-1)
        assert_eq!(labels[6], -1);
    }

    #[test]
    fn test_dbscan_single_cluster() {
        // All points form one dense cluster
        let data =
            Matrix::from_vec(5, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.0]).unwrap();

        let mut dbscan = DBSCAN::new(0.3, 2);
        dbscan.fit(&data).unwrap();

        let labels = dbscan.labels();

        // All samples should be in the same cluster (not noise)
        let first_label = labels[0];
        assert_ne!(first_label, -1);
        for &label in labels.iter() {
            assert_eq!(label, first_label);
        }
    }

    #[test]
    fn test_dbscan_all_noise() {
        // All points far apart
        let data =
            Matrix::from_vec(4, 2, vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0]).unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&data).unwrap();

        let labels = dbscan.labels();

        // All samples should be noise
        for &label in labels.iter() {
            assert_eq!(label, -1);
        }
    }

    #[test]
    fn test_dbscan_min_samples_effect() {
        // Same data, different min_samples
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1]).unwrap();

        // With min_samples=2, should form cluster
        let mut dbscan1 = DBSCAN::new(0.3, 2);
        dbscan1.fit(&data).unwrap();
        let labels1 = dbscan1.labels();
        assert!(labels1.iter().any(|&l| l != -1));

        // With min_samples=5, should be all noise
        let mut dbscan2 = DBSCAN::new(0.3, 5);
        dbscan2.fit(&data).unwrap();
        let labels2 = dbscan2.labels();
        assert!(labels2.iter().all(|&l| l == -1));
    }

    #[test]
    fn test_dbscan_eps_effect() {
        // Same data, different eps
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5]).unwrap();

        // With large eps, should form one cluster
        let mut dbscan1 = DBSCAN::new(2.0, 2);
        dbscan1.fit(&data).unwrap();
        let labels1 = dbscan1.labels();
        let unique_clusters: std::collections::HashSet<_> =
            labels1.iter().filter(|&&l| l != -1).collect();
        assert_eq!(unique_clusters.len(), 1);

        // With small eps, more fragmentation
        let mut dbscan2 = DBSCAN::new(0.3, 2);
        dbscan2.fit(&data).unwrap();
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
        .unwrap();

        let mut dbscan1 = DBSCAN::new(0.5, 2);
        dbscan1.fit(&data).unwrap();
        let labels1 = dbscan1.labels().clone();

        let mut dbscan2 = DBSCAN::new(0.5, 2);
        dbscan2.fit(&data).unwrap();
        let labels2 = dbscan2.labels();

        // Results should be identical
        assert_eq!(labels1, *labels2);
    }

    #[test]
    fn test_dbscan_fit_predict() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1]).unwrap();

        let mut dbscan = DBSCAN::new(0.3, 2);
        dbscan.fit(&data).unwrap();

        let labels_stored = dbscan.labels().clone();
        let labels_predicted = dbscan.predict(&data);

        // predict() should return same labels as stored from fit()
        assert_eq!(labels_stored, labels_predicted);
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_dbscan_predict_before_fit() {
        let data = Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let dbscan = DBSCAN::new(0.5, 2);
        let _ = dbscan.predict(&data); // Should panic
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_dbscan_labels_before_fit() {
        let dbscan = DBSCAN::new(0.5, 2);
        let _ = dbscan.labels(); // Should panic
    }
}
