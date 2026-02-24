//! Evaluation metrics for ML models.
//!
//! Includes regression metrics (R², MSE, MAE), clustering metrics
//! (inertia, silhouette score), classification metrics
//! (accuracy, precision, recall, F1-score, confusion matrix),
//! ranking metrics (Hit@K, MRR, NDCG), model evaluation framework,
//! and drift detection.

pub mod classification;
pub mod drift;
pub mod evaluator;
pub mod ranking;

use crate::primitives::{Matrix, Vector};

/// Computes the coefficient of determination (R²).
///
/// R² = 1 - (`SS_res` / `SS_tot`)
///
/// where `SS_res` is the residual sum of squares and `SS_tot` is the total
/// sum of squares.
///
/// # Examples
///
/// ```
/// use aprender::metrics::r_squared;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
/// let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
/// let r2 = r_squared(&y_pred, &y_true);
/// assert!(r2 > 0.9);
/// ```
///
/// # Panics
///
/// Panics if vectors have different lengths.
#[must_use]
#[provable_contracts_macros::contract("metrics-regression-v1", equation = "r_squared")]
pub fn r_squared(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");

    let y_mean = y_true.mean();

    let ss_res: f32 = y_true
        .as_slice()
        .iter()
        .zip(y_pred.as_slice().iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    let ss_tot: f32 = y_true.as_slice().iter().map(|t| (t - y_mean).powi(2)).sum();

    if ss_tot == 0.0 {
        return 0.0;
    }

    1.0 - (ss_res / ss_tot)
}

/// Computes the Mean Squared Error (MSE).
///
/// MSE = (1/n) * `Σ(y_true` - `y_pred)²`
///
/// # Examples
///
/// ```
/// use aprender::metrics::mse;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
/// let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
/// let error = mse(&y_pred, &y_true);
/// assert!(error < 1.0);
/// ```
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
#[must_use]
#[provable_contracts_macros::contract("metrics-regression-v1", equation = "mse")]
pub fn mse(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n = y_true.len() as f32;

    let sum_sq_error: f32 = y_true
        .as_slice()
        .iter()
        .zip(y_pred.as_slice().iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    sum_sq_error / n
}

/// Computes the Mean Absolute Error (MAE).
///
/// MAE = (1/n) * `Σ|y_true` - `y_pred`|
///
/// # Examples
///
/// ```
/// use aprender::metrics::mae;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
/// let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
/// let error = mae(&y_pred, &y_true);
/// assert!(error < 1.0);
/// ```
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
#[must_use]
#[provable_contracts_macros::contract("metrics-regression-v1", equation = "mae")]
pub fn mae(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n = y_true.len() as f32;

    let sum_abs_error: f32 = y_true
        .as_slice()
        .iter()
        .zip(y_pred.as_slice().iter())
        .map(|(t, p)| (t - p).abs())
        .sum();

    sum_abs_error / n
}

/// Computes the Root Mean Squared Error (RMSE).
///
/// RMSE = sqrt(MSE)
///
/// # Examples
///
/// ```
/// use aprender::metrics::rmse;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
/// let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
/// let error = rmse(&y_pred, &y_true);
/// assert!(error < 1.0);
/// ```
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
#[must_use]
#[provable_contracts_macros::contract("metrics-regression-v1", equation = "rmse")]
pub fn rmse(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    mse(y_pred, y_true).sqrt()
}

/// Computes the inertia (within-cluster sum of squares).
///
/// Inertia = Σ ||x - centroid||²
///
/// # Examples
///
/// ```
/// use aprender::metrics::inertia;
/// use aprender::primitives::Matrix;
///
/// let data = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
/// ]).expect("Matrix dimensions and data length are valid");
/// let centroids = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("Matrix dimensions and data length are valid");
/// let labels = vec![0, 0, 0, 0];
/// let score = inertia(&data, &centroids, &labels);
/// assert!(score > 0.0);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-clustering-v1", equation = "inertia")]
pub fn inertia(data: &Matrix<f32>, centroids: &Matrix<f32>, labels: &[usize]) -> f32 {
    let mut total = 0.0;

    for (i, &label) in labels.iter().enumerate() {
        let point = data.row(i);
        let centroid = centroids.row(label);
        let diff = &point - &centroid;
        total += diff.norm_squared();
    }

    total
}

/// Computes the mean distance from a point to other points in the same cluster.
fn mean_intra_cluster_distance(
    data: &Matrix<f32>,
    point_idx: usize,
    cluster: usize,
    labels: &[usize],
) -> f32 {
    let point = data.row(point_idx);
    let distances: Vec<f32> = labels
        .iter()
        .enumerate()
        .filter(|&(j, &label)| j != point_idx && label == cluster)
        .map(|(j, _)| {
            let other = data.row(j);
            (&point - &other).norm()
        })
        .collect();

    if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    }
}

/// Computes the minimum mean distance from a point to points in other clusters.
fn min_inter_cluster_distance(
    data: &Matrix<f32>,
    point_idx: usize,
    cluster: usize,
    labels: &[usize],
    n_clusters: usize,
) -> f32 {
    let point = data.row(point_idx);
    let mut min_mean = f32::INFINITY;

    for other_cluster in 0..n_clusters {
        if other_cluster == cluster {
            continue;
        }

        let distances: Vec<f32> = labels
            .iter()
            .enumerate()
            .filter(|&(_, &label)| label == other_cluster)
            .map(|(j, _)| {
                let other = data.row(j);
                (&point - &other).norm()
            })
            .collect();

        if !distances.is_empty() {
            let mean_dist = distances.iter().sum::<f32>() / distances.len() as f32;
            min_mean = min_mean.min(mean_dist);
        }
    }

    if min_mean == f32::INFINITY {
        0.0
    } else {
        min_mean
    }
}

/// Computes the silhouette coefficient for a single point.
fn silhouette_coefficient(a_i: f32, b_i: f32) -> f32 {
    let max_ab = a_i.max(b_i);
    if max_ab == 0.0 {
        0.0
    } else {
        (b_i - a_i) / max_ab
    }
}

/// Computes the silhouette score for clustering quality.
///
/// The silhouette score measures how similar a point is to its own cluster
/// compared to other clusters. Values range from -1 to 1, where higher is better.
///
/// s(i) = (b(i) - a(i)) / max(a(i), b(i))
///
/// where:
/// - a(i) = mean distance to other points in same cluster
/// - b(i) = mean distance to points in nearest other cluster
///
/// # Examples
///
/// ```
/// use aprender::metrics::silhouette_score;
/// use aprender::primitives::Matrix;
///
/// let data = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).expect("Matrix dimensions and data length are valid");
/// let labels = vec![0, 0, 1, 1];
/// let score = silhouette_score(&data, &labels);
/// assert!(score > 0.5);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-clustering-v1", equation = "silhouette_score")]
pub fn silhouette_score(data: &Matrix<f32>, labels: &[usize]) -> f32 {
    let n_samples = data.n_rows();

    if n_samples < 2 {
        return 0.0;
    }

    let n_clusters = labels.iter().max().map_or(0, |&m| m + 1);

    if n_clusters < 2 {
        return 0.0;
    }

    let silhouettes: Vec<f32> = (0..n_samples)
        .map(|i| {
            let cluster = labels[i];
            let a_i = mean_intra_cluster_distance(data, i, cluster, labels);
            let b_i = min_inter_cluster_distance(data, i, cluster, labels, n_clusters);
            silhouette_coefficient(a_i, b_i)
        })
        .collect();

    silhouettes.iter().sum::<f32>() / silhouettes.len() as f32
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_regression_contract.rs"]
mod tests_regression_contract;

#[cfg(test)]
#[path = "tests_clustering_contract.rs"]
mod tests_clustering_contract;

#[cfg(test)]
#[path = "tests_ranking_contract.rs"]
mod tests_ranking_contract;
