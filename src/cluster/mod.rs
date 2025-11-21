//! Clustering algorithms.
//!
//! Includes K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models, and Isolation Forest.

use crate::error::Result;
use crate::metrics::inertia;
use crate::primitives::{Matrix, Vector};
use crate::traits::UnsupervisedEstimator;
use rand::seq::SliceRandom;
use rand::SeedableRng;
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
/// ]).expect("Valid matrix dimensions and data length");
///
/// let mut kmeans = KMeans::new(2);
/// kmeans.fit(&data).expect("Fit succeeds with valid data");
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
            .expect("Centroid matrix dimensions match allocated data length")
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
            .expect("Updated centroid matrix dimensions match preallocated vector length")
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
/// ]).expect("Valid matrix dimensions and data length");
///
/// let mut dbscan = DBSCAN::new(0.5, 2);
/// dbscan.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = dbscan.labels().expect("Labels available after fit");
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

/// Linkage methods for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Linkage {
    /// Minimum distance between clusters (single linkage).
    Single,
    /// Maximum distance between clusters (complete linkage).
    Complete,
    /// Average distance between all pairs (average linkage).
    Average,
    /// Minimize within-cluster variance (Ward's method).
    Ward,
}

/// Dendrogram merge record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Merge {
    /// Clusters being merged.
    pub clusters: (usize, usize),
    /// Distance at which merge occurs.
    pub distance: f32,
    /// New cluster size after merge.
    pub size: usize,
}

/// Agglomerative (bottom-up) hierarchical clustering.
///
/// Builds a tree of clusters (dendrogram) by iteratively merging
/// closest clusters using various linkage methods.
///
/// # Algorithm
///
/// 1. Start with each point as its own cluster
/// 2. Repeat until reaching n_clusters:
///    - Find two closest clusters using linkage method
///    - Merge them
///    - Update distance matrix
/// 3. Return cluster labels
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(6, 2, vec![
///     1.0, 1.0, 1.1, 1.0, 1.0, 1.1,
///     5.0, 5.0, 5.1, 5.0, 5.0, 5.1,
/// ]).expect("Valid matrix dimensions and data length");
///
/// let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
/// hc.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = hc.predict(&data);
/// assert_eq!(labels.len(), 6);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n³) for naive implementation
/// - Space complexity: O(n²) for distance matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgglomerativeClustering {
    /// Target number of clusters.
    n_clusters: usize,
    /// Linkage method for distance calculation.
    linkage: Linkage,
    /// Cluster labels after fitting.
    labels: Option<Vec<usize>>,
    /// Dendrogram merge history.
    dendrogram: Option<Vec<Merge>>,
}

impl AgglomerativeClustering {
    /// Create new AgglomerativeClustering with target number of clusters and linkage method.
    pub fn new(n_clusters: usize, linkage: Linkage) -> Self {
        Self {
            n_clusters,
            linkage,
            labels: None,
            dendrogram: None,
        }
    }

    /// Get target number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Get linkage method.
    pub fn linkage(&self) -> Linkage {
        self.linkage
    }

    /// Check if model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }

    /// Get cluster labels (panic if not fitted).
    pub fn labels(&self) -> &Vec<usize> {
        self.labels
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Get dendrogram merge history (panic if not fitted).
    pub fn dendrogram(&self) -> &Vec<Merge> {
        self.dendrogram
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Calculate Euclidean distance between two points.
    fn euclidean_distance(&self, x: &Matrix<f32>, i: usize, j: usize) -> f32 {
        let n_features = x.shape().1;
        let mut sum = 0.0;
        for k in 0..n_features {
            let diff = x.get(i, k) - x.get(j, k);
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Calculate pairwise distance matrix.
    #[allow(clippy::needless_range_loop)]
    fn pairwise_distances(&self, x: &Matrix<f32>) -> Vec<Vec<f32>> {
        let n_samples = x.shape().0;
        let mut distances = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = self.euclidean_distance(x, i, j);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        distances
    }

    /// Find minimum distance between two active clusters.
    fn find_closest_clusters(
        &self,
        distances: &[Vec<f32>],
        active: &[bool],
    ) -> (usize, usize, f32) {
        let n = distances.len();
        let mut min_dist = f32::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if distances[i][j] < min_dist {
                    min_dist = distances[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        (min_i, min_j, min_dist)
    }

    /// Update distances for a newly merged cluster using specified linkage.
    fn update_distances(
        &self,
        x: &Matrix<f32>,
        distances: &mut [Vec<f32>],
        clusters: &[Vec<usize>],
        merged_idx: usize,
        other_idx: usize,
    ) {
        let merged_cluster = &clusters[merged_idx];
        let other_cluster = &clusters[other_idx];

        let dist = match self.linkage {
            Linkage::Single => {
                // Minimum distance
                let mut min_dist = f32::INFINITY;
                for &i in merged_cluster {
                    for &j in other_cluster {
                        let d = self.euclidean_distance(x, i, j);
                        if d < min_dist {
                            min_dist = d;
                        }
                    }
                }
                min_dist
            }
            Linkage::Complete => {
                // Maximum distance
                let mut max_dist = 0.0;
                for &i in merged_cluster {
                    for &j in other_cluster {
                        let d = self.euclidean_distance(x, i, j);
                        if d > max_dist {
                            max_dist = d;
                        }
                    }
                }
                max_dist
            }
            Linkage::Average => {
                // Average distance
                let mut sum = 0.0;
                let mut count = 0;
                for &i in merged_cluster {
                    for &j in other_cluster {
                        sum += self.euclidean_distance(x, i, j);
                        count += 1;
                    }
                }
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            Linkage::Ward => {
                // Ward's method: minimize within-cluster variance
                // Simplified: use centroid distance weighted by cluster sizes
                let merged_centroid = self.compute_centroid(x, merged_cluster);
                let other_centroid = self.compute_centroid(x, other_cluster);

                let mut sum = 0.0;
                for k in 0..x.shape().1 {
                    let diff = merged_centroid[k] - other_centroid[k];
                    sum += diff * diff;
                }

                let n1 = merged_cluster.len() as f32;
                let n2 = other_cluster.len() as f32;
                ((n1 * n2) / (n1 + n2)) * sum.sqrt()
            }
        };

        distances[merged_idx][other_idx] = dist;
        distances[other_idx][merged_idx] = dist;
    }

    /// Compute centroid of a cluster.
    #[allow(clippy::needless_range_loop)]
    fn compute_centroid(&self, x: &Matrix<f32>, cluster: &[usize]) -> Vec<f32> {
        let n_features = x.shape().1;
        let mut centroid = vec![0.0; n_features];

        for &idx in cluster {
            for k in 0..n_features {
                centroid[k] += x.get(idx, k);
            }
        }

        let size = cluster.len() as f32;
        for val in &mut centroid {
            *val /= size;
        }

        centroid
    }
}

impl UnsupervisedEstimator for AgglomerativeClustering {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.shape().0;

        // Initialize: each point is its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let mut active = vec![true; n_samples];
        let mut cluster_labels = vec![0; n_samples];
        let mut dendrogram = Vec::new();

        // Calculate initial pairwise distances
        let mut distances = self.pairwise_distances(x);

        // Merge until reaching target number of clusters
        while clusters.iter().filter(|c| !c.is_empty()).count() > self.n_clusters {
            // Find closest pair
            let (i, j, dist) = self.find_closest_clusters(&distances, &active);

            // Merge cluster j into cluster i
            let merged_cluster = clusters[j].clone();
            clusters[i].extend(&merged_cluster);
            clusters[j].clear();
            active[j] = false;

            // Record merge in dendrogram
            dendrogram.push(Merge {
                clusters: (i, j),
                distance: dist,
                size: clusters[i].len(),
            });

            // Update distances for merged cluster
            #[allow(clippy::needless_range_loop)]
            for k in 0..n_samples {
                if k == i || !active[k] {
                    continue;
                }
                self.update_distances(x, &mut distances, &clusters, i, k);
            }
        }

        // Assign labels
        let mut cluster_id = 0;
        for cluster in &clusters {
            if !cluster.is_empty() {
                for &point_idx in cluster {
                    cluster_labels[point_idx] = cluster_id;
                }
                cluster_id += 1;
            }
        }

        self.labels = Some(cluster_labels);
        self.dendrogram = Some(dendrogram);
        Ok(())
    }

    fn predict(&self, _x: &Matrix<f32>) -> Self::Labels {
        // For hierarchical clustering, predict returns fitted labels
        // (new points would require a different strategy)
        self.labels().clone()
    }
}

/// Covariance matrix types for Gaussian Mixture Models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovarianceType {
    /// Full covariance matrix (most flexible, most parameters).
    Full,
    /// Tied covariance (all components share same covariance).
    Tied,
    /// Diagonal covariance (assumes feature independence).
    Diag,
    /// Spherical covariance (isotropic, like K-Means).
    Spherical,
}

/// Gaussian Mixture Model (GMM) for probabilistic clustering.
///
/// Uses Expectation-Maximization (EM) algorithm to fit a mixture of
/// Gaussian distributions to the data, providing soft cluster assignments.
///
/// # Algorithm
///
/// 1. **E-step**: Compute responsibilities (probability each point belongs to each cluster)
/// 2. **M-step**: Update means, covariances, and mixing weights
/// 3. Repeat until convergence
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(6, 2, vec![
///     1.0, 1.0, 1.1, 1.0, 1.0, 1.1,
///     5.0, 5.0, 5.1, 5.0, 5.0, 5.1,
/// ]).expect("Valid matrix dimensions and data length");
///
/// let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
/// gmm.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = gmm.predict(&data);
/// assert_eq!(labels.len(), 6);
///
/// let proba = gmm.predict_proba(&data);
/// assert_eq!(proba.shape(), (6, 2));
/// ```
///
/// # Performance
///
/// - Time complexity: O(nkd²i) where n=samples, k=components, d=features, i=iterations
/// - Space complexity: O(nk + kd²)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixture {
    /// Number of mixture components.
    n_components: usize,
    /// Type of covariance matrix.
    covariance_type: CovarianceType,
    /// Maximum number of EM iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f32,
    /// Random seed for initialization.
    random_state: Option<u64>,
    /// Component means after fitting (k × d).
    means: Option<Matrix<f32>>,
    /// Component covariances after fitting.
    covariances: Option<Vec<Matrix<f32>>>,
    /// Mixing weights after fitting (sums to 1).
    weights: Option<Vector<f32>>,
    /// Cluster labels after fitting.
    labels: Option<Vec<usize>>,
}

impl GaussianMixture {
    /// Create new GaussianMixture with specified number of components and covariance type.
    pub fn new(n_components: usize, covariance_type: CovarianceType) -> Self {
        Self {
            n_components,
            covariance_type,
            max_iter: 100,
            tol: 1e-3,
            random_state: None,
            means: None,
            covariances: None,
            weights: None,
            labels: None,
        }
    }

    /// Set maximum number of EM iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get number of components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get covariance type.
    pub fn covariance_type(&self) -> CovarianceType {
        self.covariance_type
    }

    /// Check if model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.means.is_some()
    }

    /// Get component means (panic if not fitted).
    pub fn means(&self) -> &Matrix<f32> {
        self.means
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Get mixing weights (panic if not fitted).
    pub fn weights(&self) -> &Vector<f32> {
        self.weights
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Get cluster labels (panic if not fitted).
    pub fn labels(&self) -> &Vec<usize> {
        self.labels
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Compute log probability of data under the model.
    pub fn score(&self, x: &Matrix<f32>) -> f32 {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }
        let responsibilities = self.compute_responsibilities(x);
        let n_samples = x.shape().0;

        // Compute log-likelihood
        let mut log_likelihood = 0.0;
        for i in 0..n_samples {
            let mut prob_sum = 0.0;
            for k in 0..self.n_components {
                prob_sum += responsibilities.get(i, k);
            }
            if prob_sum > 0.0 {
                log_likelihood += prob_sum.ln();
            }
        }
        log_likelihood / n_samples as f32
    }

    /// Predict cluster probabilities for each sample (soft assignment).
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Matrix<f32> {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }
        self.compute_responsibilities(x)
    }

    /// Initialize parameters using K-Means.
    fn initialize_parameters(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (_n_samples, n_features) = x.shape();

        // Use K-Means for initialization
        let mut kmeans = KMeans::new(self.n_components);
        if let Some(seed) = self.random_state {
            kmeans = kmeans.with_random_state(seed);
        }
        kmeans.fit(x)?;

        // Set initial means from K-Means centroids
        self.means = Some(kmeans.centroids().clone());

        // Initialize weights uniformly
        let weight = 1.0 / self.n_components as f32;
        self.weights = Some(Vector::from_vec(vec![weight; self.n_components]));

        // Initialize covariances
        let mut covariances = Vec::new();
        for _k in 0..self.n_components {
            let cov = match self.covariance_type {
                CovarianceType::Full
                | CovarianceType::Tied
                | CovarianceType::Diag
                | CovarianceType::Spherical => {
                    // Start with identity matrix
                    let mut cov_matrix = vec![0.0; n_features * n_features];
                    for i in 0..n_features {
                        cov_matrix[i * n_features + i] = 1.0;
                    }
                    Matrix::from_vec(n_features, n_features, cov_matrix)?
                }
            };
            covariances.push(cov);
        }
        self.covariances = Some(covariances);

        Ok(())
    }

    /// Compute Gaussian probability density.
    #[allow(clippy::needless_range_loop)]
    fn gaussian_pdf(&self, x: &[f32], mean: &[f32], cov: &Matrix<f32>) -> f32 {
        let n_features = mean.len();

        // Compute (x - mean)
        let mut diff = vec![0.0; n_features];
        for i in 0..n_features {
            diff[i] = x[i] - mean[i];
        }

        // For numerical stability, use simplified calculation
        // This is a simplified version - production code would use proper matrix inverse
        let mut mahalanobis = 0.0;
        for i in 0..n_features {
            let cov_ii = cov.get(i, i).max(1e-6); // Regularization
            mahalanobis += diff[i] * diff[i] / cov_ii;
        }

        // Compute determinant (simplified for diagonal-like structure)
        let mut det = 1.0;
        for i in 0..n_features {
            det *= cov.get(i, i).max(1e-6);
        }

        let norm_const = ((2.0 * std::f32::consts::PI).powi(n_features as i32) * det).sqrt();
        (-0.5 * mahalanobis).exp() / norm_const.max(1e-10)
    }

    /// E-step: Compute responsibilities (posterior probabilities).
    #[allow(clippy::needless_range_loop)]
    fn compute_responsibilities(&self, x: &Matrix<f32>) -> Matrix<f32> {
        let (n_samples, n_features) = x.shape();
        let means = self
            .means
            .as_ref()
            .expect("Means must be initialized before computing responsibilities");
        let weights = self
            .weights
            .as_ref()
            .expect("Weights must be initialized before computing responsibilities");
        let covariances = self
            .covariances
            .as_ref()
            .expect("Covariances must be initialized before computing responsibilities");

        let mut responsibilities = vec![0.0; n_samples * self.n_components];

        for i in 0..n_samples {
            let mut sample = vec![0.0; n_features];
            for j in 0..n_features {
                sample[j] = x.get(i, j);
            }

            let mut total_prob = 0.0;
            for k in 0..self.n_components {
                let mut mean_k = vec![0.0; n_features];
                for j in 0..n_features {
                    mean_k[j] = means.get(k, j);
                }

                let pdf = self.gaussian_pdf(&sample, &mean_k, &covariances[k]);
                let weighted_pdf = weights[k] * pdf;
                responsibilities[i * self.n_components + k] = weighted_pdf;
                total_prob += weighted_pdf;
            }

            // Normalize responsibilities
            if total_prob > 1e-10 {
                for k in 0..self.n_components {
                    responsibilities[i * self.n_components + k] /= total_prob;
                }
            } else {
                // Uniform if total prob is too small
                for k in 0..self.n_components {
                    responsibilities[i * self.n_components + k] = 1.0 / self.n_components as f32;
                }
            }
        }

        Matrix::from_vec(n_samples, self.n_components, responsibilities)
            .expect("Responsibility matrix dimensions match preallocated vector length")
    }

    /// M-step: Update parameters based on responsibilities.
    #[allow(clippy::needless_range_loop)]
    fn update_parameters(&mut self, x: &Matrix<f32>, responsibilities: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        // Compute effective number of points per component
        let mut n_k = vec![0.0; self.n_components];
        for k in 0..self.n_components {
            for i in 0..n_samples {
                n_k[k] += responsibilities.get(i, k);
            }
            n_k[k] = n_k[k].max(1e-6); // Regularization
        }

        // Update weights
        let mut new_weights = vec![0.0; self.n_components];
        for k in 0..self.n_components {
            new_weights[k] = n_k[k] / n_samples as f32;
        }
        self.weights = Some(Vector::from_vec(new_weights));

        // Update means
        let mut new_means = vec![0.0; self.n_components * n_features];
        for k in 0..self.n_components {
            for j in 0..n_features {
                let mut weighted_sum = 0.0;
                for i in 0..n_samples {
                    weighted_sum += responsibilities.get(i, k) * x.get(i, j);
                }
                new_means[k * n_features + j] = weighted_sum / n_k[k];
            }
        }
        self.means = Some(Matrix::from_vec(self.n_components, n_features, new_means)?);

        // Update covariances (simplified diagonal)
        let means = self
            .means
            .as_ref()
            .expect("Means must exist after update step");
        let mut new_covariances = Vec::new();

        for k in 0..self.n_components {
            let mut cov_data = vec![0.0; n_features * n_features];

            for j in 0..n_features {
                let mut variance = 0.0;
                for i in 0..n_samples {
                    let diff = x.get(i, j) - means.get(k, j);
                    variance += responsibilities.get(i, k) * diff * diff;
                }
                variance = (variance / n_k[k]).max(1e-6); // Regularization
                cov_data[j * n_features + j] = variance;
            }

            new_covariances.push(Matrix::from_vec(n_features, n_features, cov_data)?);
        }
        self.covariances = Some(new_covariances);

        Ok(())
    }
}

impl UnsupervisedEstimator for GaussianMixture {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        // Initialize parameters
        self.initialize_parameters(x)?;

        // EM algorithm
        let mut prev_log_likelihood = f32::NEG_INFINITY;

        for _iter in 0..self.max_iter {
            // E-step: Compute responsibilities
            let responsibilities = self.compute_responsibilities(x);

            // M-step: Update parameters
            self.update_parameters(x, &responsibilities)?;

            // Check convergence
            let log_likelihood = self.score(x);
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }

        // Assign labels based on final responsibilities
        let responsibilities = self.compute_responsibilities(x);
        let n_samples = x.shape().0;
        let mut labels = vec![0; n_samples];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            let mut max_prob = 0.0;
            let mut max_k = 0;
            for k in 0..self.n_components {
                let prob = responsibilities.get(i, k);
                if prob > max_prob {
                    max_prob = prob;
                    max_k = k;
                }
            }
            labels[i] = max_k;
        }

        self.labels = Some(labels);
        Ok(())
    }

    fn predict(&self, x: &Matrix<f32>) -> Self::Labels {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }

        let responsibilities = self.compute_responsibilities(x);
        let n_samples = x.shape().0;
        let mut labels = vec![0; n_samples];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            let mut max_prob = 0.0;
            let mut max_k = 0;
            for k in 0..self.n_components {
                let prob = responsibilities.get(i, k);
                if prob > max_prob {
                    max_prob = prob;
                    max_k = k;
                }
            }
            labels[i] = max_k;
        }

        labels
    }
}

// ============================================================================
// Isolation Forest for Anomaly Detection
// ============================================================================

/// Internal node structure for Isolation Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationNode {
    /// Feature index to split on (None for leaf)
    split_feature: Option<usize>,
    /// Split value (None for leaf)
    split_value: Option<f32>,
    /// Left child (samples with feature < split_value)
    left: Option<Box<IsolationNode>>,
    /// Right child (samples with feature >= split_value)
    right: Option<Box<IsolationNode>>,
    /// Size of node (for path length calculation)
    size: usize,
}

/// Single Isolation Tree for anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationTree {
    root: Option<IsolationNode>,
    max_depth: usize,
}

impl IsolationTree {
    fn new(max_depth: usize) -> Self {
        Self {
            root: None,
            max_depth,
        }
    }

    fn fit(&mut self, x: &Matrix<f32>, rng: &mut impl rand::Rng) {
        let indices: Vec<usize> = (0..x.shape().0).collect();
        self.root = Some(self.build_tree(x, &indices, 0, rng));
    }

    fn build_tree(
        &self,
        x: &Matrix<f32>,
        indices: &[usize],
        depth: usize,
        rng: &mut impl rand::Rng,
    ) -> IsolationNode {
        let n_samples = indices.len();

        // Terminal conditions
        if depth >= self.max_depth || n_samples <= 1 {
            return IsolationNode {
                split_feature: None,
                split_value: None,
                left: None,
                right: None,
                size: n_samples,
            };
        }

        // Random feature selection
        let n_features = x.shape().1;
        let feature_idx = rng.gen_range(0..n_features);

        // Find min/max for this feature in current samples
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &idx in indices {
            let val = x.get(idx, feature_idx);
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // If all values are the same, make leaf
        if (max_val - min_val).abs() < 1e-10 {
            return IsolationNode {
                split_feature: None,
                split_value: None,
                left: None,
                right: None,
                size: n_samples,
            };
        }

        // Random split value between min and max
        let split_val = rng.gen_range(min_val..max_val);

        // Partition samples
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in indices {
            if x.get(idx, feature_idx) < split_val {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // If split doesn't separate, make leaf
        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationNode {
                split_feature: None,
                split_value: None,
                left: None,
                right: None,
                size: n_samples,
            };
        }

        // Recursively build children
        let left = self.build_tree(x, &left_indices, depth + 1, rng);
        let right = self.build_tree(x, &right_indices, depth + 1, rng);

        IsolationNode {
            split_feature: Some(feature_idx),
            split_value: Some(split_val),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            size: n_samples,
        }
    }

    fn path_length(&self, sample: &[f32]) -> f32 {
        if let Some(ref root) = self.root {
            self.path_length_recursive(sample, root, 0.0)
        } else {
            0.0
        }
    }

    fn path_length_recursive(&self, sample: &[f32], node: &IsolationNode, depth: f32) -> f32 {
        // Leaf node - add average path length for remaining samples
        if node.split_feature.is_none() {
            return depth + Self::c(node.size);
        }

        let feature_idx = node
            .split_feature
            .expect("Split feature must exist for non-leaf nodes");
        let split_val = node
            .split_value
            .expect("Split value must exist for non-leaf nodes");

        if sample[feature_idx] < split_val {
            if let Some(ref left) = node.left {
                self.path_length_recursive(sample, left, depth + 1.0)
            } else {
                depth + Self::c(node.size)
            }
        } else if let Some(ref right) = node.right {
            self.path_length_recursive(sample, right, depth + 1.0)
        } else {
            depth + Self::c(node.size)
        }
    }

    /// Average path length of unsuccessful search in BST (for normalization)
    fn c(n: usize) -> f32 {
        if n <= 1 {
            0.0
        } else if n == 2 {
            1.0
        } else {
            let n_f32 = n as f32;
            2.0 * ((n_f32 - 1.0).ln() + 0.577_215_7) - 2.0 * (n_f32 - 1.0) / n_f32
        }
    }
}

/// Isolation Forest for anomaly detection.
///
/// Uses an ensemble of isolation trees to detect outliers based on path length.
/// Anomalies are easier to isolate (shorter paths) than normal points.
///
/// # Algorithm
///
/// 1. Build N isolation trees on random subsamples
/// 2. Each tree recursively splits data by random feature + random threshold
/// 3. Compute average path length across all trees
/// 4. Convert to anomaly score (shorter path = more anomalous)
/// 5. Use contamination parameter to set classification threshold
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(
///     6,
///     2,
///     vec![
///         2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9,  // Normal
///         10.0, 10.0, -10.0, -10.0,                 // Outliers
///     ],
/// )
/// .expect("Valid matrix dimensions and data length");
///
/// let mut iforest = IsolationForest::new()
///     .with_contamination(0.3)
///     .with_random_state(42);
/// iforest.fit(&data).expect("Fit succeeds with valid data");
///
/// // Predict returns 1 for normal, -1 for anomaly
/// let predictions = iforest.predict(&data);
///
/// // score_samples returns anomaly scores (lower = more anomalous)
/// let scores = iforest.score_samples(&data);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n log m) where n=samples, m=max_samples
/// - Space complexity: O(t * m) where t=n_estimators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    /// Number of trees in the ensemble
    n_estimators: usize,
    /// Number of samples to draw for each tree
    max_samples: Option<usize>,
    /// Expected proportion of anomalies in the dataset
    contamination: f32,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Maximum tree depth
    max_depth: usize,
    /// Ensemble of isolation trees
    trees: Vec<IsolationTree>,
    /// Threshold for anomaly classification (computed from contamination)
    threshold: Option<f32>,
    /// Average path length normalization constant
    c_norm: f32,
}

impl IsolationForest {
    /// Create a new Isolation Forest with default parameters.
    ///
    /// Default: 100 trees, auto max_samples, 0.1 contamination
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: None,
            contamination: 0.1,
            random_state: None,
            max_depth: 10,
            trees: Vec::new(),
            threshold: None,
            c_norm: 1.0,
        }
    }

    /// Set the number of trees in the ensemble.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the number of samples to draw for each tree.
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = Some(max_samples);
        self
    }

    /// Set the expected proportion of anomalies (0 to 0.5).
    pub fn with_contamination(mut self, contamination: f32) -> Self {
        self.contamination = contamination.clamp(0.0, 0.5);
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Check if model has been fitted.
    pub fn is_fitted(&self) -> bool {
        !self.trees.is_empty()
    }

    /// Fit the Isolation Forest on training data.
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _n_features) = x.shape();

        // Determine max_samples (default: min(256, n_samples))
        let max_samples = self.max_samples.unwrap_or_else(|| n_samples.min(256));

        // Compute normalization constant
        self.c_norm = IsolationTree::c(max_samples);

        // Compute max tree depth
        self.max_depth = (max_samples as f32).log2().ceil() as usize;

        // Initialize RNG
        let mut rng: Box<dyn rand::RngCore> = if let Some(seed) = self.random_state {
            Box::new(rand::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(rand::rngs::StdRng::from_entropy())
        };

        // Build ensemble of trees
        self.trees.clear();
        for _ in 0..self.n_estimators {
            // Sample random subset
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(max_samples);

            // Extract subsample
            let subsample = self.extract_subsample(x, &indices);

            // Build tree
            let mut tree = IsolationTree::new(self.max_depth);
            tree.fit(&subsample, &mut rng);
            self.trees.push(tree);
        }

        // Compute anomaly scores on training data to determine threshold
        let scores = self.score_samples(x);
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("Anomaly scores must be valid floats for comparison")
        });

        // Threshold at contamination quantile
        let threshold_idx = (self.contamination * n_samples as f32) as usize;
        self.threshold = Some(sorted_scores[threshold_idx.min(n_samples - 1)]);

        Ok(())
    }

    /// Extract subsample from data.
    fn extract_subsample(&self, x: &Matrix<f32>, indices: &[usize]) -> Matrix<f32> {
        let (_, n_features) = x.shape();
        let n_samples = indices.len();
        let mut data = vec![0.0; n_samples * n_features];

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..n_features {
                data[i * n_features + j] = x.get(idx, j);
            }
        }

        Matrix::from_vec(n_samples, n_features, data)
            .expect("Subsampled matrix dimensions match collected data length")
    }

    /// Compute anomaly scores for samples.
    ///
    /// Returns a vector of scores where lower scores indicate higher anomaly likelihood.
    #[allow(clippy::needless_range_loop)]
    pub fn score_samples(&self, x: &Matrix<f32>) -> Vec<f32> {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }

        let (n_samples, n_features) = x.shape();
        let mut scores = vec![0.0; n_samples];

        for i in 0..n_samples {
            // Extract sample
            let sample: Vec<f32> = (0..n_features).map(|j| x.get(i, j)).collect();

            // Average path length across all trees
            let avg_path_length: f32 = self
                .trees
                .iter()
                .map(|tree| tree.path_length(&sample))
                .sum::<f32>()
                / self.n_estimators as f32;

            // Anomaly score: 2^(-avg_path / c_norm)
            // Scores close to 1 = anomaly, close to 0 = normal
            let score = 2f32.powf(-avg_path_length / self.c_norm);

            // Invert so lower = more anomalous (for consistency with decision_function)
            scores[i] = -score;
        }

        scores
    }

    /// Predict anomaly labels for samples.
    ///
    /// Returns 1 for normal points and -1 for anomalies.
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<i32> {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }

        let threshold = self
            .threshold
            .expect("Threshold must be set during fit phase");
        let scores = self.score_samples(x);

        scores
            .iter()
            .map(|&score| if score < threshold { -1 } else { 1 })
            .collect()
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Local Outlier Factor (LOF) for Anomaly Detection
// ============================================================================

/// Local Outlier Factor (LOF) for density-based anomaly detection.
///
/// LOF detects anomalies based on local density deviation. Unlike global methods,
/// it finds outliers in regions with varying density by comparing each point's
/// density to its neighbors' densities.
///
/// # Algorithm
///
/// 1. For each point, find k-nearest neighbors
/// 2. Compute reachability distance for each neighbor
/// 3. Compute local reachability density (LRD) for each point
/// 4. Compute LOF score: ratio of neighbors' LRD to point's LRD
///
/// # LOF Score Interpretation
///
/// - LOF ≈ 1: Similar density to neighbors (normal point)
/// - LOF >> 1: Lower density than neighbors (outlier)
/// - LOF < 1: Higher density than neighbors (core point)
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(
///     6,
///     2,
///     vec![
///         2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9,  // Normal cluster
///         10.0, 10.0, -10.0, -10.0,                 // Outliers
///     ],
/// )
/// .expect("Valid matrix dimensions and data length");
///
/// let mut lof = LocalOutlierFactor::new()
///     .with_n_neighbors(3)
///     .with_contamination(0.3);
/// lof.fit(&data).expect("Fit succeeds with valid data");
///
/// // Predict returns 1 for normal, -1 for anomaly
/// let predictions = lof.predict(&data);
///
/// // score_samples returns LOF scores (higher = more anomalous)
/// let scores = lof.score_samples(&data);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n² log k) for k-NN search
/// - Space complexity: O(n²) for distance matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalOutlierFactor {
    /// Number of neighbors to use
    n_neighbors: usize,
    /// Expected proportion of anomalies
    contamination: f32,
    /// LOF scores for training data
    lof_scores: Option<Vec<f32>>,
    /// Negative outlier factor (opposite of LOF for sklearn compatibility)
    negative_outlier_factor: Option<Vec<f32>>,
    /// Training data (needed for prediction)
    training_data: Option<Matrix<f32>>,
    /// k-NN distances for training data
    knn_distances: Option<Vec<Vec<f32>>>,
    /// k-NN indices for training data
    knn_indices: Option<Vec<Vec<usize>>>,
    /// Local reachability density for training data
    lrd: Option<Vec<f32>>,
    /// Threshold for classification
    threshold: Option<f32>,
}

impl LocalOutlierFactor {
    /// Create a new Local Outlier Factor with default parameters.
    ///
    /// Default: 20 neighbors, 0.1 contamination
    pub fn new() -> Self {
        Self {
            n_neighbors: 20,
            contamination: 0.1,
            lof_scores: None,
            negative_outlier_factor: None,
            training_data: None,
            knn_distances: None,
            knn_indices: None,
            lrd: None,
            threshold: None,
        }
    }

    /// Set the number of neighbors.
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the expected proportion of anomalies (0 to 0.5).
    pub fn with_contamination(mut self, contamination: f32) -> Self {
        self.contamination = contamination.clamp(0.0, 0.5);
        self
    }

    /// Check if model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.lof_scores.is_some()
    }

    /// Fit the Local Outlier Factor on training data.
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _) = x.shape();

        // Validate n_neighbors
        if self.n_neighbors >= n_samples {
            return Err("n_neighbors must be less than number of samples".into());
        }

        // Store training data for prediction
        self.training_data = Some(x.clone());

        // Compute k-NN for all points
        let (knn_distances, knn_indices) = self.compute_knn(x, x);
        self.knn_distances = Some(knn_distances.clone());
        self.knn_indices = Some(knn_indices.clone());

        // Compute local reachability density for all points
        let lrd = self.compute_lrd(&knn_distances, &knn_indices);
        self.lrd = Some(lrd.clone());

        // Compute LOF scores
        let lof_scores = self.compute_lof_scores(&lrd, &knn_indices);
        self.lof_scores = Some(lof_scores.clone());

        // Compute negative outlier factor (for sklearn compatibility)
        let nof: Vec<f32> = lof_scores.iter().map(|&score| -score).collect();
        self.negative_outlier_factor = Some(nof.clone());

        // Determine threshold from contamination
        let mut sorted_scores = lof_scores.clone();
        sorted_scores.sort_by(|a, b| {
            b.partial_cmp(a)
                .expect("LOF scores must be valid floats for comparison")
        }); // Descending order

        let threshold_idx = (self.contamination * n_samples as f32) as usize;
        self.threshold = Some(sorted_scores[threshold_idx.min(n_samples - 1)]);

        Ok(())
    }

    /// Compute k-nearest neighbors for query points.
    fn compute_knn(
        &self,
        query: &Matrix<f32>,
        data: &Matrix<f32>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let (n_query, n_features) = query.shape();
        let (n_data, _) = data.shape();

        let mut knn_distances = Vec::with_capacity(n_query);
        let mut knn_indices = Vec::with_capacity(n_query);

        for i in 0..n_query {
            // Compute distances to all points
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(n_data);

            for j in 0..n_data {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = query.get(i, k) - data.get(j, k);
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                distances.push((dist, j));
            }

            // Sort by distance
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distances must be valid floats for comparison")
            });

            // Take k+1 nearest (skip self if query == data)
            let skip_self = i < n_data;
            let k_start = if skip_self { 1 } else { 0 };
            let k_end = k_start + self.n_neighbors;

            let dists: Vec<f32> = distances[k_start..k_end.min(distances.len())]
                .iter()
                .map(|(d, _)| *d)
                .collect();
            let indices: Vec<usize> = distances[k_start..k_end.min(distances.len())]
                .iter()
                .map(|(_, idx)| *idx)
                .collect();

            knn_distances.push(dists);
            knn_indices.push(indices);
        }

        (knn_distances, knn_indices)
    }

    /// Compute reachability distance between two points.
    fn reachability_distance(&self, dist: f32, k_distance: f32) -> f32 {
        dist.max(k_distance)
    }

    /// Compute local reachability density for all points.
    fn compute_lrd(&self, knn_distances: &[Vec<f32>], knn_indices: &[Vec<usize>]) -> Vec<f32> {
        let n_samples = knn_indices.len();
        let mut lrd = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let neighbors = &knn_indices[i];
            let neighbor_dists = &knn_distances[i];

            if neighbors.is_empty() {
                lrd.push(1.0);
                continue;
            }

            // Compute sum of reachability distances
            let mut sum_reach_dist = 0.0;
            for (j, &neighbor_idx) in neighbors.iter().enumerate() {
                let dist_to_neighbor = neighbor_dists[j];

                // k-distance of neighbor (distance to its k-th neighbor)
                let k_distance = if neighbor_idx < knn_distances.len() {
                    knn_distances[neighbor_idx].last().copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let reach_dist = self.reachability_distance(dist_to_neighbor, k_distance);
                sum_reach_dist += reach_dist;
            }

            // LRD = k / sum of reachability distances
            let lrd_value = if sum_reach_dist > 0.0 {
                neighbors.len() as f32 / sum_reach_dist
            } else {
                1.0 // Avoid division by zero
            };

            lrd.push(lrd_value);
        }

        lrd
    }

    /// Compute LOF scores for all points.
    fn compute_lof_scores(&self, lrd: &[f32], knn_indices: &[Vec<usize>]) -> Vec<f32> {
        let n_samples = knn_indices.len();
        let mut lof_scores = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let neighbors = &knn_indices[i];

            if neighbors.is_empty() || lrd[i] == 0.0 {
                lof_scores.push(1.0);
                continue;
            }

            // Average LRD of neighbors
            let sum_neighbor_lrd: f32 = neighbors.iter().map(|&idx| lrd[idx]).sum();
            let avg_neighbor_lrd = sum_neighbor_lrd / neighbors.len() as f32;

            // LOF = avg(neighbor LRD) / LRD(point)
            let lof = avg_neighbor_lrd / lrd[i];

            lof_scores.push(lof);
        }

        lof_scores
    }

    /// Compute LOF scores for samples.
    ///
    /// Returns a vector of LOF scores where higher scores indicate anomalies.
    pub fn score_samples(&self, x: &Matrix<f32>) -> Vec<f32> {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }

        let training_data = self
            .training_data
            .as_ref()
            .expect("Training data must be stored during fit");
        let training_lrd = self
            .lrd
            .as_ref()
            .expect("LRD values must be computed during fit");

        // Compute k-NN for query points against training data
        let (knn_distances, knn_indices) = self.compute_knn(x, training_data);

        // Compute LRD for query points
        let query_lrd = self.compute_lrd(&knn_distances, &knn_indices);

        // Compute LOF scores for query points
        let n_query = x.shape().0;
        let mut lof_scores = Vec::with_capacity(n_query);

        for i in 0..n_query {
            let neighbors = &knn_indices[i];

            if neighbors.is_empty() || query_lrd[i] == 0.0 {
                lof_scores.push(1.0);
                continue;
            }

            // Average LRD of neighbors (from training data)
            let sum_neighbor_lrd: f32 = neighbors
                .iter()
                .filter_map(|&idx| training_lrd.get(idx).copied())
                .sum();
            let avg_neighbor_lrd = sum_neighbor_lrd / neighbors.len() as f32;

            // LOF = avg(neighbor LRD) / LRD(point)
            let lof = avg_neighbor_lrd / query_lrd[i];

            lof_scores.push(lof);
        }

        lof_scores
    }

    /// Predict anomaly labels for samples.
    ///
    /// Returns 1 for normal points and -1 for anomalies.
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<i32> {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }

        let threshold = self
            .threshold
            .expect("Threshold must be set during fit phase");
        let scores = self.score_samples(x);

        scores
            .iter()
            .map(|&score| if score > threshold { -1 } else { 1 })
            .collect()
    }

    /// Get the negative outlier factor for training samples.
    ///
    /// Returns negative of LOF scores (sklearn compatibility).
    pub fn negative_outlier_factor(&self) -> &[f32] {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }

        self.negative_outlier_factor
            .as_ref()
            .expect("Negative outlier factor must be computed during fit")
    }
}

impl Default for LocalOutlierFactor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Spectral Clustering for Graph-Based Clustering
// ============================================================================

/// Affinity types for constructing similarity graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Affinity {
    /// Radial Basis Function (Gaussian) kernel
    RBF,
    /// k-Nearest Neighbors graph
    KNN,
}

/// Spectral Clustering using graph Laplacian and eigendecomposition.
///
/// Uses graph theory to find clusters by analyzing the spectrum (eigenvalues)
/// of the graph Laplacian. Effective for non-convex cluster shapes.
///
/// # Algorithm
///
/// 1. Construct affinity matrix W (RBF or k-NN)
/// 2. Compute graph Laplacian: L = D - W (D = degree matrix)
/// 3. Find k smallest eigenvectors of L
/// 4. Cluster rows of eigenvector matrix using K-Means
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// // Non-convex clusters (concentric circles)
/// let data = Matrix::from_vec(
///     8,
///     2,
///     vec![
///         0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0,  // Inner circle
///         0.0, 3.0, 3.0, 0.0, 0.0, -3.0, -3.0, 0.0,  // Outer circle
///     ],
/// )
/// .expect("Valid matrix dimensions and data length");
///
/// let mut sc = SpectralClustering::new(2).with_gamma(0.5);
/// sc.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = sc.predict(&data);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n² + n³) for eigendecomposition
/// - Space complexity: O(n²) for affinity matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Affinity type (RBF or KNN)
    affinity: Affinity,
    /// RBF kernel coefficient (higher = more local)
    gamma: f32,
    /// Number of neighbors for KNN affinity
    n_neighbors: usize,
    /// Cluster labels
    labels: Option<Vec<usize>>,
}

impl SpectralClustering {
    /// Create a new Spectral Clustering with default parameters.
    ///
    /// Default: RBF affinity, gamma=1.0, n_neighbors=10
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            affinity: Affinity::RBF,
            gamma: 1.0,
            n_neighbors: 10,
            labels: None,
        }
    }

    /// Set the affinity type.
    pub fn with_affinity(mut self, affinity: Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    /// Set gamma for RBF kernel (higher = more local similarity).
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set number of neighbors for KNN affinity.
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Check if model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }

    /// Get cluster labels (panics if not fitted).
    pub fn labels(&self) -> &Vec<usize> {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }
        self.labels
            .as_ref()
            .expect("Labels must exist after successful fit")
    }

    /// Fit the Spectral Clustering model.
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _) = x.shape();

        // 1. Construct affinity matrix
        let affinity_matrix = match self.affinity {
            Affinity::RBF => self.compute_rbf_affinity(x),
            Affinity::KNN => self.compute_knn_affinity(x),
        };

        // 2. Compute graph Laplacian
        let laplacian = self.compute_laplacian(&affinity_matrix, n_samples);

        // 3. Eigendecomposition - find k smallest eigenvectors
        let mut embedding = self.compute_embedding(&laplacian, n_samples)?;

        // 4. Normalize rows of embedding (critical for normalized spectral clustering)
        for i in 0..n_samples {
            let mut row_norm = 0.0;
            for j in 0..self.n_clusters {
                let val = embedding.get(i, j);
                row_norm += val * val;
            }
            row_norm = row_norm.sqrt().max(1e-10); // Avoid division by zero
            for j in 0..self.n_clusters {
                let val = embedding.get(i, j);
                embedding.set(i, j, val / row_norm);
            }
        }

        // 5. Cluster in eigenspace using K-Means
        let mut kmeans = KMeans::new(self.n_clusters).with_random_state(42);
        kmeans.fit(&embedding)?;
        let labels = kmeans.predict(&embedding);

        self.labels = Some(labels);
        Ok(())
    }

    /// Compute RBF (Gaussian) affinity matrix.
    fn compute_rbf_affinity(&self, x: &Matrix<f32>) -> Vec<f32> {
        let (n_samples, n_features) = x.shape();
        let mut affinity = vec![0.0; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    affinity[i * n_samples + j] = 0.0;
                    continue;
                }

                // Compute squared Euclidean distance
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x.get(i, k) - x.get(j, k);
                    dist_sq += diff * diff;
                }

                // RBF kernel: exp(-gamma * ||x_i - x_j||^2)
                let similarity = (-self.gamma * dist_sq).exp();
                affinity[i * n_samples + j] = similarity;
            }
        }

        affinity
    }

    /// Compute k-NN affinity matrix.
    fn compute_knn_affinity(&self, x: &Matrix<f32>) -> Vec<f32> {
        let (n_samples, n_features) = x.shape();
        let mut affinity = vec![0.0; n_samples * n_samples];

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
            // Compute distances to all points
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(n_samples);
            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x.get(i, k) - x.get(j, k);
                    dist_sq += diff * diff;
                }
                distances.push((dist_sq.sqrt(), j));
            }

            // Sort and take k nearest
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distances must be valid floats for comparison")
            });
            let k_neighbors = self.n_neighbors.min(n_samples - 1);

            // Set affinity to 1 for k-nearest neighbors
            for (_, neighbor_idx) in distances.iter().take(k_neighbors) {
                affinity[i * n_samples + neighbor_idx] = 1.0;
            }
        }

        // Make symmetric
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let val = f32::max(affinity[i * n_samples + j], affinity[j * n_samples + i]);
                affinity[i * n_samples + j] = val;
                affinity[j * n_samples + i] = val;
            }
        }

        affinity
    }

    /// Compute normalized graph Laplacian.
    fn compute_laplacian(&self, affinity: &[f32], n_samples: usize) -> Vec<f32> {
        // Compute degree matrix D
        let mut degrees = vec![0.0; n_samples];
        for i in 0..n_samples {
            let mut degree = 0.0;
            for j in 0..n_samples {
                degree += affinity[i * n_samples + j];
            }
            degrees[i] = degree;
        }

        // Compute normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
        let mut laplacian = vec![0.0; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    laplacian[i * n_samples + j] = 1.0;
                } else {
                    let d_i = degrees[i].max(1e-10); // Avoid division by zero
                    let d_j = degrees[j].max(1e-10);
                    let w_ij = affinity[i * n_samples + j];
                    laplacian[i * n_samples + j] = -w_ij / (d_i * d_j).sqrt();
                }
            }
        }

        laplacian
    }

    /// Compute embedding via eigendecomposition.
    fn compute_embedding(&self, laplacian: &[f32], n_samples: usize) -> Result<Matrix<f32>> {
        use nalgebra::{DMatrix, SymmetricEigen};

        // Convert to nalgebra matrix
        let laplacian_matrix = DMatrix::from_row_slice(n_samples, n_samples, laplacian);

        // Eigendecomposition
        let eigen = SymmetricEigen::new(laplacian_matrix);

        // Sort eigenvalues and get indices of k smallest
        let mut eigen_pairs: Vec<(usize, f64)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val as f64))
            .collect();
        eigen_pairs.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .expect("Eigenvalues must be valid floats for comparison")
        });

        // Get k smallest eigenvectors
        let k = self.n_clusters;
        let smallest_indices: Vec<usize> = eigen_pairs.iter().take(k).map(|(i, _)| *i).collect();

        let mut embedding_data = Vec::with_capacity(n_samples * k);

        // Build embedding matrix in row-major order
        for row_idx in 0..n_samples {
            for &col_idx in &smallest_indices {
                let eigenvector = eigen.eigenvectors.column(col_idx);
                embedding_data.push(eigenvector[row_idx]);
            }
        }

        Ok(Matrix::from_vec(n_samples, k, embedding_data)?)
    }
}

impl UnsupervisedEstimator for SpectralClustering {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        SpectralClustering::fit(self, x)
    }

    fn predict(&self, _x: &Matrix<f32>) -> Self::Labels {
        if !self.is_fitted() {
            panic!("Model not fitted. Call fit() first.");
        }
        self.labels
            .as_ref()
            .expect("Labels must exist after successful fit")
            .clone()
    }
}

impl Default for SpectralClustering {
    fn default() -> Self {
        Self::new(2)
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
        for &label in labels.iter() {
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
        for &label in labels.iter() {
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
}
