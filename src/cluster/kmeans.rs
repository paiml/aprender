//! K-Means clustering algorithm.
//!
//! Uses Lloyd's algorithm with k-means++ initialization for faster convergence.

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

    /// Returns the number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Returns the maximum number of iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f32 {
        self.tol
    }

    /// Returns the random state.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the K-Means model to a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the `SafeTensors` file will be saved
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

        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a K-Means model from a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `SafeTensors` file
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
            .map_err(|e| format!("Failed to reconstruct centroids matrix: {e}"))?;

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
    pub(crate) fn centroids_converged(&self, old: &Matrix<f32>, new: &Matrix<f32>) -> bool {
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
