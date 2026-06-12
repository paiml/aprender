//! Evaluation metrics for ML models.
//!
//! Includes regression metrics (R², MSE, MAE), clustering metrics
//! (inertia, silhouette score), classification metrics
//! (accuracy, precision, recall, F1-score, confusion matrix),
//! ranking metrics (Hit@K, MRR, NDCG), model evaluation framework,
//! and drift detection.

pub mod agreement;
pub mod classification;
pub mod regression;
pub use agreement::{balanced_accuracy_score, cohen_kappa_score, hamming_loss, matthews_corrcoef};
pub use regression::{
    max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
    mean_squared_log_error, median_absolute_error, r2_score,
};
pub mod probabilistic;
pub use probabilistic::{average_precision_score, log_loss, roc_auc_score};
pub mod drift;
pub mod evaluator;
pub mod grad_norm;
pub mod percentile;
pub mod perplexity;
pub mod ranking;
pub mod ship_005;

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
    contract_pre_r_squared!(y_pred.as_slice());
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
    contract_pre_mse!(y_pred.as_slice());
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
    contract_pre_mae!(y_pred.as_slice());
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
    contract_pre_rmse!(y_pred.as_slice());
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
    contract_pre_inertia!();
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
    contract_pre_silhouette_coefficient!();
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
    contract_pre_silhouette_score!();
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
pub use classification::{fbeta_score, jaccard_score};

/// Davies–Bouldin score (lower is better), matching `sklearn.metrics.davies_bouldin_score`.
/// Mean over clusters of the worst-case ratio `(S_i + S_j) / d(c_i, c_j)`, where
/// `S` is mean intra-cluster distance to centroid and `d` is centroid distance.
#[must_use]
pub fn davies_bouldin_score(data: &Matrix<f32>, labels: &[usize]) -> f32 {
    let (n, nf) = data.shape();
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    if k < 2 {
        return 0.0;
    }
    let mut centroids = vec![vec![0.0f64; nf]; k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..nf {
            centroids[c][j] += f64::from(data.get(i, j));
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..nf {
                centroids[c][j] /= counts[c] as f64;
            }
        }
    }
    let mut scatter = vec![0.0f64; k];
    for i in 0..n {
        let c = labels[i];
        let mut d = 0.0f64;
        for j in 0..nf {
            let diff = f64::from(data.get(i, j)) - centroids[c][j];
            d += diff * diff;
        }
        scatter[c] += d.sqrt();
    }
    for c in 0..k {
        if counts[c] > 0 {
            scatter[c] /= counts[c] as f64;
        }
    }
    let mut db = 0.0f64;
    for c in 0..k {
        let mut max_r = 0.0f64;
        for cp in 0..k {
            if cp == c {
                continue;
            }
            let mut dc = 0.0f64;
            for j in 0..nf {
                let diff = centroids[c][j] - centroids[cp][j];
                dc += diff * diff;
            }
            let dc = dc.sqrt();
            if dc > 0.0 {
                let r = (scatter[c] + scatter[cp]) / dc;
                if r > max_r {
                    max_r = r;
                }
            }
        }
        db += max_r;
    }
    (db / k as f64) as f32
}

/// Calinski–Harabasz score (variance ratio; higher is better), matching
/// `sklearn.metrics.calinski_harabasz_score`: `(B/(k-1)) / (W/(n-k))`.
#[must_use]
pub fn calinski_harabasz_score(data: &Matrix<f32>, labels: &[usize]) -> f32 {
    let (n, nf) = data.shape();
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    if k < 2 || n <= k {
        return 0.0;
    }
    let mut overall = vec![0.0f64; nf];
    let mut centroids = vec![vec![0.0f64; nf]; k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..nf {
            let v = f64::from(data.get(i, j));
            centroids[c][j] += v;
            overall[j] += v;
        }
    }
    for j in 0..nf {
        overall[j] /= n as f64;
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..nf {
                centroids[c][j] /= counts[c] as f64;
            }
        }
    }
    let mut w = 0.0f64;
    for i in 0..n {
        let c = labels[i];
        for j in 0..nf {
            let diff = f64::from(data.get(i, j)) - centroids[c][j];
            w += diff * diff;
        }
    }
    let mut b = 0.0f64;
    for c in 0..k {
        let mut d = 0.0f64;
        for j in 0..nf {
            let diff = centroids[c][j] - overall[j];
            d += diff * diff;
        }
        b += counts[c] as f64 * d;
    }
    if w == 0.0 {
        return 0.0;
    }
    ((b / (k - 1) as f64) / (w / (n - k) as f64)) as f32
}

#[cfg(test)]
mod tests_clustering_extra {
    use super::*;

    /// FT-METRIC-DBI / CHI: match sklearn clustering metrics within 1e-3.
    #[test]
    fn davies_bouldin_and_calinski_match_sklearn() {
        let data = Matrix::from_vec(
            7,
            2,
            vec![
                1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 3.5, 5.0, 4.5, 5.0, 3.5, 4.5,
            ],
        )
        .expect("valid");
        let labels = [0usize, 0, 1, 1, 1, 1, 1];
        assert!((davies_bouldin_score(&data, &labels) - 0.364_795).abs() < 1e-3);
        assert!((calinski_harabasz_score(&data, &labels) - 16.742_773).abs() < 1e-2);
    }
}
