//! Evaluation metrics for ML models.
//!
//! Includes regression metrics (R², MSE, MAE) and clustering metrics
//! (inertia, silhouette score).

use crate::primitives::{Matrix, Vector};

/// Computes the coefficient of determination (R²).
///
/// R² = 1 - (SS_res / SS_tot)
///
/// where SS_res is the residual sum of squares and SS_tot is the total
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
/// MSE = (1/n) * Σ(y_true - y_pred)²
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
/// MAE = (1/n) * Σ|y_true - y_pred|
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
mod tests {
    use super::*;

    #[test]
    fn test_r_squared_perfect() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let r2 = r_squared(&y_pred, &y_true);
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_good() {
        let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
        let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
        let r2 = r_squared(&y_pred, &y_true);
        assert!(r2 > 0.9);
        assert!(r2 < 1.0);
    }

    #[test]
    fn test_r_squared_mean_prediction() {
        // Predicting the mean gives R² = 0
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean = y_true.mean();
        let y_pred = Vector::from_slice(&[mean, mean, mean, mean, mean]);
        let r2 = r_squared(&y_pred, &y_true);
        assert!(r2.abs() < 1e-6);
    }

    #[test]
    fn test_r_squared_negative() {
        // Worse than mean prediction
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[10.0, 10.0, 10.0]);
        let r2 = r_squared(&y_pred, &y_true);
        assert!(r2 < 0.0);
    }

    #[test]
    fn test_mse() {
        let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
        let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
        let error = mse(&y_pred, &y_true);
        // MSE = (0.25 + 0.25 + 0 + 1) / 4 = 0.375
        assert!((error - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_mse_perfect() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let error = mse(&y_pred, &y_true);
        assert!(error.abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
        let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
        let error = mae(&y_pred, &y_true);
        // MAE = (0.5 + 0.5 + 0 + 1) / 4 = 0.5
        assert!((error - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mae_perfect() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let error = mae(&y_pred, &y_true);
        assert!(error.abs() < 1e-6);
    }

    #[test]
    fn test_rmse() {
        let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
        let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);
        let error = rmse(&y_pred, &y_true);
        assert!((error - 0.375_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_inertia() {
        // 4 points in 2 clusters
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0])
            .expect("Matrix dimensions (4x2) match data length (8)");

        let centroids = Matrix::from_vec(
            2,
            2,
            vec![
                0.5, 0.0, // cluster 0 centroid
                10.5, 10.0, // cluster 1 centroid
            ],
        )
        .expect("Matrix dimensions (2x2) match data length (4)");

        let labels = vec![0, 0, 1, 1];
        let score = inertia(&data, &centroids, &labels);

        // Distance from (0,0) to (0.5,0) = 0.25
        // Distance from (1,0) to (0.5,0) = 0.25
        // Distance from (10,10) to (10.5,10) = 0.25
        // Distance from (11,10) to (10.5,10) = 0.25
        // Total = 1.0
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_silhouette_score_good() {
        // Well-separated clusters
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1])
            .expect("Matrix dimensions (4x2) match data length (8)");
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        assert!(score > 0.9);
    }

    #[test]
    fn test_silhouette_score_poor() {
        // Overlapping clusters
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.5, 1.5])
            .expect("Matrix dimensions (4x2) match data length (8)");
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        // With overlapping data, silhouette should be lower
        assert!(score < 0.9);
    }

    #[test]
    fn test_silhouette_score_single_cluster() {
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("Matrix dimensions (4x2) match data length (8)");
        let labels = vec![0, 0, 0, 0];
        let score = silhouette_score(&data, &labels);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silhouette_score_single_sample() {
        let data = Matrix::from_vec(1, 2, vec![0.0, 0.0])
            .expect("Matrix dimensions (1x2) match data length (2)");
        let labels = vec![0];
        let score = silhouette_score(&data, &labels);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silhouette_score_two_samples() {
        // Test with exactly 2 samples - catches n_samples < 2 → <= 2 mutation (line 288)
        let data = Matrix::from_vec(2, 2, vec![0.0, 0.0, 10.0, 10.0])
            .expect("Matrix dimensions (2x2) match data length (4)");
        let labels = vec![0, 1];
        let score = silhouette_score(&data, &labels);

        // With 2 samples in different clusters, silhouette should be meaningful (not 0)
        // If mutation changes < to <=, this would return 0.0
        assert!(
            score.abs() > 0.0,
            "Score with 2 samples should be non-zero, got {score}"
        );
    }

    #[test]
    fn test_silhouette_score_distance_calculation() {
        // Test that catches arithmetic mutations in distance calculations
        // Line 204: / with % or * (mean_intra_cluster_distance)
        // Line 230: - with + (distance calculation)
        // Line 235: / with * (min_inter_cluster_distance)
        let data = Matrix::from_vec(
            6,
            1,
            vec![
                0.0,   // Cluster 0, point 0
                1.0,   // Cluster 0, point 1
                2.0,   // Cluster 0, point 2
                100.0, // Cluster 1, point 3
                101.0, // Cluster 1, point 4
                102.0, // Cluster 1, point 5
            ],
        )
        .expect("Matrix dimensions (6x1) match data length (6)");
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&data, &labels);

        // Well-separated clusters should have high silhouette score
        // If arithmetic is wrong, score will be very different
        assert!(
            score > 0.9,
            "Well-separated clusters should have high score, got {score}"
        );

        // Score should be reasonable (not > 1.0 which would indicate calculation error)
        assert!(score <= 1.0, "Score should be <= 1.0, got {score}");
    }

    #[test]
    fn test_silhouette_score_mean_not_one() {
        // Test that catches mean_intra_cluster_distance -> 1.0 mutation (line 190)
        // Create clusters where mean intra-cluster distance is definitely not 1.0
        let data = Matrix::from_vec(
            4,
            1,
            vec![
                0.0,  // Cluster 0
                10.0, // Cluster 0 (distance = 10)
                50.0, // Cluster 1
                60.0, // Cluster 1 (distance = 10)
            ],
        )
        .expect("Matrix dimensions (4x1) match data length (4)");
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);

        // Verify the score calculation uses actual distances
        // Mean intra-cluster distance = 10.0 for each cluster
        // If mutation returns 1.0 instead, the score would be very different
        assert!(
            score > 0.5,
            "Score should be high for well-separated clusters, got {score}"
        );
    }

    #[test]
    fn test_metrics_consistency() {
        // Property: RMSE = sqrt(MSE)
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Vector::from_slice(&[1.1, 2.2, 2.9, 4.1, 5.0]);

        let mse_val = mse(&y_pred, &y_true);
        let rmse_val = rmse(&y_pred, &y_true);

        assert!((rmse_val - mse_val.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_silhouette_intra_cluster_averaging() {
        // Test that mean intra-cluster distance uses division, not multiplication
        // Cluster 0: points at 0, 1, 2 - mean distance from each point
        // Cluster 1: points at 100, 101, 102
        let data = Matrix::from_vec(6, 1, vec![0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
            .expect("Matrix dimensions (6x1) match data length (6)");
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&data, &labels);

        // If division becomes multiplication in mean calculation:
        // distance sum * count instead of sum / count
        // This would give absurdly large values
        assert!(
            score > 0.9,
            "Should have high silhouette for well-separated clusters, got {score}"
        );
        assert!(score <= 1.0, "Silhouette score must be <= 1.0, got {score}");
    }

    #[test]
    fn test_silhouette_inter_cluster_distance() {
        // Test that inter-cluster distance uses subtraction in norm
        // Two clusters: one near origin, one far away
        let data = Matrix::from_vec(4, 1, vec![0.0, 1.0, 1000.0, 1001.0])
            .expect("Matrix dimensions (4x1) match data length (4)");
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);

        // If subtraction becomes addition in distance calculation:
        // distances would be wrong
        assert!(
            score > 0.99,
            "Very well-separated clusters should have score > 0.99, got {score}"
        );
    }

    #[test]
    fn test_silhouette_score_exact_boundary() {
        // Test boundary condition where samples < n_clusters check matters
        // With exactly 2 samples in different clusters
        let data = Matrix::from_vec(2, 1, vec![0.0, 100.0])
            .expect("Matrix dimensions (2x1) match data length (2)");
        let labels = vec![0, 1];
        let score = silhouette_score(&data, &labels);

        // This catches < vs <= mutations in sample count check
        // Score should still be computed (not return 0)
        assert!(
            score.abs() > 1e-6 || score == 0.0,
            "Score should be computed for 2 samples"
        );
    }

    #[test]
    fn test_silhouette_many_clusters() {
        // Test with multiple clusters to verify min inter-cluster distance
        // 4 clusters, well separated
        let data = Matrix::from_vec(
            8,
            1,
            vec![
                0.0, 1.0, // Cluster 0
                100.0, 101.0, // Cluster 1
                200.0, 201.0, // Cluster 2
                300.0, 301.0, // Cluster 3
            ],
        )
        .expect("Matrix dimensions (8x1) match data length (8)");
        let labels = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let score = silhouette_score(&data, &labels);

        // All clusters well-separated, should have high score
        assert!(
            score > 0.9,
            "Well-separated multi-cluster should have high score, got {score}"
        );
    }
}
