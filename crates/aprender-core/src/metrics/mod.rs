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
    explained_variance_score, max_error, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score,
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
///
/// Returns `None` when the point's cluster has size 1 (no other members), i.e. a
/// singleton cluster. sklearn computes `intra_clust_dist = sum / (size - 1)`, which
/// is `0 / 0 = NaN` for a singleton; it then `np.nan_to_num`s that to 0, and the
/// docstring states "clusters of size 1 ... are assigned a value of 0". Returning
/// `None` lets the caller assign that sample a silhouette of exactly 0, rather than
/// the wrong `+1.0` produced by treating the intra-distance as 0.0 (PMAT-845).
fn mean_intra_cluster_distance(
    data: &Matrix<f32>,
    point_idx: usize,
    cluster: usize,
    labels: &[usize],
) -> Option<f32> {
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
        // Singleton cluster: sklearn assigns silhouette 0 to this sample.
        None
    } else {
        Some(distances.iter().sum::<f32>() / distances.len() as f32)
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
            let b_i = min_inter_cluster_distance(data, i, cluster, labels, n_clusters);
            // PMAT-845: a singleton cluster (no intra-cluster neighbors) is assigned
            // silhouette 0 per sklearn, not (b_i - 0)/max(0, b_i) = +1.0.
            match mean_intra_cluster_distance(data, i, cluster, labels) {
                None => 0.0,
                Some(a_i) => silhouette_coefficient(a_i, b_i),
            }
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

/// Remaps cluster labels to a dense `0..k` range based on the set of DISTINCT
/// labels present, mirroring sklearn's `LabelEncoder`. Returns the dense labels
/// and `k = n_distinct`. This makes clustering metrics invariant under relabeling
/// and avoids phantom empty clusters when labels are non-contiguous (e.g. a cluster
/// was dropped, leaving a gap, or DBSCAN-style sparse output).
fn dense_relabel(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut distinct: Vec<usize> = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let dense: Vec<usize> = labels
        .iter()
        .map(|&l| {
            distinct
                .binary_search(&l)
                .expect("label present in distinct set")
        })
        .collect();
    (dense, distinct.len())
}

/// Davies–Bouldin score (lower is better), matching `sklearn.metrics.davies_bouldin_score`.
/// Mean over clusters of the worst-case ratio `(S_i + S_j) / d(c_i, c_j)`, where
/// `S` is mean intra-cluster distance to centroid and `d` is centroid distance.
///
/// The score depends only on the partition and is invariant under relabeling:
/// cluster count is `k = |distinct labels|`, so non-contiguous labels never create
/// phantom empty clusters (sklearn `LabelEncoder` semantics).
#[must_use]
pub fn davies_bouldin_score(data: &Matrix<f32>, labels: &[usize]) -> f32 {
    let (n, nf) = data.shape();
    let (labels, k) = dense_relabel(labels);
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
///
/// The score depends only on the partition and is invariant under relabeling:
/// cluster count is `k = |distinct labels|`, so non-contiguous labels never create
/// phantom empty clusters that would corrupt the `(k-1)` and `(n-k)` divisors
/// (sklearn `LabelEncoder` semantics).
#[must_use]
pub fn calinski_harabasz_score(data: &Matrix<f32>, labels: &[usize]) -> f32 {
    let (n, nf) = data.shape();
    let (labels, k) = dense_relabel(labels);
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

    /// PMAT-871 falsifier: clustering metrics depend ONLY on the partition and must
    /// be invariant under relabeling. Non-contiguous labels (a gap left by a dropped
    /// cluster, or DBSCAN-style sparse output) must NOT create phantom empty clusters.
    ///
    /// data = [[1,1],[1.5,2],[3,4],[5,7],[3.5,5]]; the partition `{0,1}|{2,3,4}` is the
    /// same whether encoded `[0,0,1,1,1]` (contiguous) or `[0,0,2,2,2]` (gap at index 1).
    /// sklearn (LabelEncoder → dense labels) gives CH=10.3140, DB=0.4150 for BOTH.
    ///
    /// RED (pre-fix, `k = max+1`): gapped CH=3.4380 (3x error), gapped DB=0.3721.
    /// GREEN (post-fix, `k = |distinct|`): both encodings give CH≈10.3140, DB≈0.4150.
    #[test]
    fn clustering_metrics_relabel_invariant_pmat_871() {
        let data = Matrix::from_vec(5, 2, vec![1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 3.5, 5.0])
            .expect("valid");
        let contiguous = [0usize, 0, 1, 1, 1];
        let gapped = [0usize, 0, 2, 2, 2];

        let ch_contig = calinski_harabasz_score(&data, &contiguous);
        let ch_gapped = calinski_harabasz_score(&data, &gapped);
        let db_contig = davies_bouldin_score(&data, &contiguous);
        let db_gapped = davies_bouldin_score(&data, &gapped);

        // Invariance under relabeling.
        assert!(
            (ch_contig - ch_gapped).abs() < 1e-2,
            "CH must be relabeling-invariant: contiguous={ch_contig} gapped={ch_gapped}"
        );
        assert!(
            (db_contig - db_gapped).abs() < 1e-3,
            "DB must be relabeling-invariant: contiguous={db_contig} gapped={db_gapped}"
        );

        // Absolute match to sklearn reference.
        assert!(
            (ch_gapped - 10.3140).abs() < 1e-2,
            "CH must match sklearn 10.3140, got {ch_gapped}"
        );
        assert!(
            (db_gapped - 0.4150).abs() < 1e-3,
            "DB must match sklearn 0.4150, got {db_gapped}"
        );
    }
}

/// Adjusted Rand Index — similarity between two clusterings corrected for chance,
/// matching `sklearn.metrics.adjusted_rand_score`. Range ~[-0.5, 1] (1 = identical).
#[must_use]
pub fn adjusted_rand_score(labels_true: &[usize], labels_pred: &[usize]) -> f32 {
    assert_eq!(
        labels_true.len(),
        labels_pred.len(),
        "adjusted_rand_score: length mismatch"
    );
    let n = labels_true.len();
    if n == 0 {
        return 1.0;
    }
    let kt = labels_true.iter().max().map_or(0, |&m| m + 1);
    let kp = labels_pred.iter().max().map_or(0, |&m| m + 1);
    let mut cont = vec![vec![0u64; kp]; kt];
    for i in 0..n {
        cont[labels_true[i]][labels_pred[i]] += 1;
    }
    let comb2 = |x: u64| -> f64 { (x as f64 * (x as f64 - 1.0)) / 2.0 };
    let index: f64 = cont.iter().flat_map(|r| r.iter()).map(|&x| comb2(x)).sum();
    let a: f64 = (0..kt).map(|i| comb2(cont[i].iter().sum::<u64>())).sum();
    let b: f64 = (0..kp)
        .map(|j| comb2((0..kt).map(|i| cont[i][j]).sum::<u64>()))
        .sum();
    let expected = a * b / comb2(n as u64);
    let max_index = 0.5 * (a + b);
    if (max_index - expected).abs() < 1e-12 {
        return 1.0;
    }
    ((index - expected) / (max_index - expected)) as f32
}

#[cfg(test)]
mod tests_ari {
    use super::*;
    /// FT-METRIC-ARI: matches sklearn.metrics.adjusted_rand_score within 1e-4.
    #[test]
    fn adjusted_rand_matches_sklearn() {
        assert!(
            (adjusted_rand_score(&[0, 0, 1, 1, 2, 2], &[0, 0, 1, 2, 2, 2]) - 0.444_444).abs()
                < 1e-4
        );
        assert!((adjusted_rand_score(&[0, 0, 1, 1], &[0, 0, 1, 1]) - 1.0).abs() < 1e-6);
    }
}

/// Builds the `kt x kp` contingency table and the natural-log entropy of each
/// label assignment from its marginal totals. Shared kernel for the
/// mutual-information clustering metrics.
fn contingency_and_entropies(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> (Vec<Vec<u64>>, f64, f64, usize) {
    let n = labels_true.len();
    let kt = labels_true.iter().max().map_or(0, |&m| m + 1);
    let kp = labels_pred.iter().max().map_or(0, |&m| m + 1);
    let mut cont = vec![vec![0u64; kp]; kt];
    for i in 0..n {
        cont[labels_true[i]][labels_pred[i]] += 1;
    }
    // Entropy (natural log) of a marginal: H = -sum_i (n_i/n) ln(n_i/n).
    let nf = n as f64;
    let entropy = |totals: &[u64]| -> f64 {
        let mut h = 0.0f64;
        for &t in totals {
            if t > 0 {
                let p = t as f64 / nf;
                h -= p * p.ln();
            }
        }
        h
    };
    let row_totals: Vec<u64> = cont.iter().map(|r| r.iter().sum()).collect();
    let col_totals: Vec<u64> = (0..kp).map(|j| (0..kt).map(|i| cont[i][j]).sum()).collect();
    let h_true = entropy(&row_totals);
    let h_pred = entropy(&col_totals);
    (cont, h_true, h_pred, n)
}

/// Mutual information (natural log, nats) between two clusterings, matching
/// `sklearn.metrics.mutual_info_score`. MI = sum_ij (n_ij/n) ln(n*n_ij /
/// (a_i * b_j)).
#[must_use]
pub fn mutual_info_score(labels_true: &[usize], labels_pred: &[usize]) -> f32 {
    assert_eq!(
        labels_true.len(),
        labels_pred.len(),
        "mutual_info_score: length mismatch"
    );
    if labels_true.is_empty() {
        return 0.0;
    }
    let (cont, _, _, n) = contingency_and_entropies(labels_true, labels_pred);
    let nf = n as f64;
    let row_totals: Vec<u64> = cont.iter().map(|r| r.iter().sum()).collect();
    let kp = cont.first().map_or(0, Vec::len);
    let col_totals: Vec<u64> = (0..kp).map(|j| cont.iter().map(|r| r[j]).sum()).collect();
    let mut mi = 0.0f64;
    for (i, row) in cont.iter().enumerate() {
        for (j, &nij) in row.iter().enumerate() {
            if nij > 0 {
                let pij = nij as f64 / nf;
                // ln(n * n_ij / (a_i * b_j)) computed stably.
                let term = (nij as f64 * nf) / (row_totals[i] as f64 * col_totals[j] as f64);
                mi += pij * term.ln();
            }
        }
    }
    // Clamp tiny negative round-off to zero (sklearn does likewise).
    if mi < 0.0 {
        mi = 0.0;
    }
    mi as f32
}

/// Normalized mutual information between two clusterings with the arithmetic
/// normalizer (sklearn default `average_method="arithmetic"`), matching
/// `sklearn.metrics.normalized_mutual_info_score`. NMI = MI / ((H_true +
/// H_pred) / 2). Range [0, 1] (1 = clusterings identical up to relabeling).
///
/// Follows sklearn's degenerate-case convention: if exactly one clustering has a
/// single cluster (zero entropy), the metric is 0.0; if *both* are trivially a
/// single cluster, it returns 1.0.
#[must_use]
pub fn normalized_mutual_info_score(labels_true: &[usize], labels_pred: &[usize]) -> f32 {
    assert_eq!(
        labels_true.len(),
        labels_pred.len(),
        "normalized_mutual_info_score: length mismatch"
    );
    if labels_true.is_empty() {
        return 1.0;
    }
    let (_, h_true, h_pred, _) = contingency_and_entropies(labels_true, labels_pred);
    // A clustering is "trivial" (single cluster) iff every label is identical.
    let single_true = labels_true.iter().all(|&x| x == labels_true[0]);
    let single_pred = labels_pred.iter().all(|&x| x == labels_pred[0]);
    if single_true || single_pred {
        return if single_true && single_pred { 1.0 } else { 0.0 };
    }
    let mi = f64::from(mutual_info_score(labels_true, labels_pred));
    let normalizer = (h_true + h_pred) / 2.0;
    if normalizer == 0.0 {
        return 0.0;
    }
    (mi / normalizer).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests_nmi {
    use super::*;
    // ─────────────────────────────────────────────────────────────────────────
    // sklearn 1.9.0 oracle (PINNED OFFLINE — no Python at CI time).
    // Generation recipe (re-run to regenerate the constants below):
    //   uv run --with scikit-learn --with numpy python3 -c "
    //   from sklearn.metrics import normalized_mutual_info_score as nmi, mutual_info_score
    //   print(nmi([0,0,1,1,2,2],[0,0,1,2,2,2],average_method='arithmetic'))  # 0.7396673768007592
    //   print(mutual_info_score([0,0,1,1,2,2],[0,0,1,2,2,2]))                # 0.7803552045207032
    //   print(nmi([0,0,1,1,2,2],[2,2,0,0,1,1],average_method='arithmetic'))  # 1.0  (relabel-invariant)
    //   print(nmi([0,0,1,1],[0,0,0,0],average_method='arithmetic'))          # 0.0  (one trivial)
    //   print(nmi([0,0,0,0],[0,0,0,0],average_method='arithmetic'))          # 1.0  (both trivial)
    //   print(nmi([0,0,0,1,1,1,2,2],[0,0,1,1,1,2,2,2],average_method='arithmetic')) # 0.5588730382170324
    //   print(mutual_info_score([0,0,0,1,1,1,2,2],[0,0,1,1,1,2,2,2]))        # 0.6048099038176575
    //   "
    // sklearn 1.9.0, numpy float64. Tolerance 1e-4 (f32 round-off of an f64 ratio).
    const SK_NMI_PARTIAL: f32 = 0.739_667_4;
    const SK_MI_PARTIAL: f32 = 0.780_355_2;
    const SK_NMI_E: f32 = 0.558_873_04;
    const SK_MI_E: f32 = 0.604_809_9;

    /// FT-METRIC-NMI-001: NMI matches sklearn (arithmetic) on a partial-agreement
    /// fixture, is relabel-invariant (=1.0), and honours both degenerate cases.
    #[test]
    fn nmi_matches_sklearn() {
        let t = [0, 0, 1, 1, 2, 2];
        let p = [0, 0, 1, 2, 2, 2];
        assert!(
            (normalized_mutual_info_score(&t, &p) - SK_NMI_PARTIAL).abs() < 1e-4,
            "NMI partial: got {}",
            normalized_mutual_info_score(&t, &p)
        );
        // Relabel invariance: a permutation of cluster ids is a perfect match.
        assert!(
            (normalized_mutual_info_score(&[0, 0, 1, 1, 2, 2], &[2, 2, 0, 0, 1, 1]) - 1.0).abs()
                < 1e-6
        );
        // One trivial clustering (single cluster) => 0.0.
        assert!((normalized_mutual_info_score(&[0, 0, 1, 1], &[0, 0, 0, 0])).abs() < 1e-6);
        // Both trivial => 1.0 (sklearn convention).
        assert!((normalized_mutual_info_score(&[0, 0, 0, 0], &[0, 0, 0, 0]) - 1.0).abs() < 1e-6);
        // Asymmetric overlap fixture E.
        let te = [0, 0, 0, 1, 1, 1, 2, 2];
        let pe = [0, 0, 1, 1, 1, 2, 2, 2];
        assert!((normalized_mutual_info_score(&te, &pe) - SK_NMI_E).abs() < 1e-4);
    }

    /// FT-METRIC-NMI-002: raw mutual_info_score (nats) matches sklearn.
    #[test]
    fn mutual_info_matches_sklearn() {
        let t = [0, 0, 1, 1, 2, 2];
        let p = [0, 0, 1, 2, 2, 2];
        assert!(
            (mutual_info_score(&t, &p) - SK_MI_PARTIAL).abs() < 1e-4,
            "MI partial: got {}",
            mutual_info_score(&t, &p)
        );
        let te = [0, 0, 0, 1, 1, 1, 2, 2];
        let pe = [0, 0, 1, 1, 1, 2, 2, 2];
        assert!((mutual_info_score(&te, &pe) - SK_MI_E).abs() < 1e-4);
        // Independent labelling (pred trivial) has zero mutual information.
        assert!((mutual_info_score(&[0, 0, 1, 1], &[0, 0, 0, 0])).abs() < 1e-6);
    }

    /// FT-METRIC-NMI-003 (mutation guard): NMI must lie in [0,1] and be bounded
    /// strictly below 1 for a genuinely imperfect clustering, and below the
    /// geometric-mean normalizer (catches a normalizer that collapses to MI or
    /// to max/min averaging).
    #[test]
    fn nmi_bounded_and_imperfect_below_one() {
        let t = [0, 0, 1, 1, 2, 2];
        let p = [0, 0, 1, 2, 2, 2];
        let v = normalized_mutual_info_score(&t, &p);
        assert!((0.0..=1.0).contains(&v), "NMI out of [0,1]: {v}");
        assert!(v < 0.999, "imperfect clustering must score < 1: {v}");
        // Arithmetic mean >= geometric mean, so arithmetic NMI (0.7397) is
        // strictly below geometric NMI (0.7403) on this fixture.
        assert!(v < 0.740_3, "arithmetic NMI should be < geometric NMI: {v}");
    }
}
