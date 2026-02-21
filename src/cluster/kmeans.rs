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

include!("kmeans_impl.rs");
