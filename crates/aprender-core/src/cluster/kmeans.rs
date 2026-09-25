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
/// 5. Repeat steps 1-4 `n_init` times from different seeded starts and keep
///    the run with the lowest inertia (sklearn's `n_init`; default 1, as
///    sklearn >= 1.4 `n_init="auto"` picks for D²-family seeding)
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
    /// Number of seeded restarts; the lowest-inertia run is kept.
    ///
    /// LAST on purpose: bincode is positional, so a file saved before this
    /// field existed ends exactly where it would start. `KMeans::load` reads
    /// such a file through [`LegacyKMeans`] with `n_init = 1`, the single
    /// start it was fitted with.
    #[serde(default = "default_n_init")]
    n_init: usize,
}

/// The on-disk bincode layout of `KMeans` before `n_init` existed (#3146).
/// Field order must never change: it IS the old file format.
#[derive(Deserialize)]
struct LegacyKMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f32,
    random_state: Option<u64>,
    centroids: Option<Matrix<f32>>,
    labels: Option<Vec<usize>>,
    inertia: f32,
    n_iter: usize,
}

/// sklearn >= 1.4 `n_init="auto"`: one run for k-means++-family seeding
/// (10 only for random init, which this type does not offer). One run is
/// also the pre-`n_init` behaviour, so a default fit is unchanged; ask for
/// restarts with [`KMeans::with_n_init`].
pub(crate) const DEFAULT_N_INIT: usize = 1;

fn default_n_init() -> usize {
    DEFAULT_N_INIT
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(8)
    }
}

include!("kmeans_impl.rs");

#[cfg(test)]
#[path = "tests_kmeans_contract.rs"]
mod tests_kmeans_contract;
