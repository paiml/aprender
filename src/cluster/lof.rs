//! Local Outlier Factor (LOF) for density-based anomaly detection.
//!
//! LOF detects anomalies based on local density deviation. Unlike global methods,
//! it finds outliers in regions with varying density by comparing each point's
//! density to its neighbors' densities.

use crate::error::Result;
use crate::primitives::Matrix;
use serde::{Deserialize, Serialize};

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
    #[must_use]
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
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the expected proportion of anomalies (0 to 0.5).
    #[must_use]
    pub fn with_contamination(mut self, contamination: f32) -> Self {
        self.contamination = contamination.clamp(0.0, 0.5);
        self
    }

    /// Check if model has been fitted.
    #[must_use]
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
            let k_start = usize::from(skip_self);
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
    #[allow(clippy::unused_self)]
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
    #[allow(clippy::unused_self)]
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
    #[must_use]
    pub fn score_samples(&self, x: &Matrix<f32>) -> Vec<f32> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");

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
    #[must_use]
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<i32> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");

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
    #[must_use]
    pub fn negative_outlier_factor(&self) -> &[f32] {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");

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

#[cfg(test)]
#[path = "tests_lof_contract.rs"]
mod tests_lof_contract;
