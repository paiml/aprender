//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
//!
//! Density-based clustering algorithm that can find arbitrarily-shaped clusters
//! and identify outliers as noise points.

use crate::error::Result;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;
use serde::{Deserialize, Serialize};

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
///
/// Density-based clustering algorithm that can find arbitrarily-shaped clusters
/// and identify outliers as noise points.
///
/// # Algorithm
///
/// 1. For each unvisited point:
///    - Find all neighbors within eps distance
///    - If neighbors < `min_samples`: mark as noise
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
/// let labels = dbscan.labels();
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

    /// Returns the `min_samples` parameter.
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

    /// ONE PATH: Core computation delegates to `nn::functional::euclidean_distance` (UCBD §4).
    #[allow(clippy::unused_self)]
    fn euclidean_distance(&self, x: &Matrix<f32>, i: usize, j: usize) -> f32 {
        let n_features = x.shape().1;
        let row_i: Vec<f32> = (0..n_features).map(|k| x.get(i, k)).collect();
        let row_j: Vec<f32> = (0..n_features).map(|k| x.get(j, k)).collect();
        crate::nn::functional::euclidean_distance(&row_i, &row_j)
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
