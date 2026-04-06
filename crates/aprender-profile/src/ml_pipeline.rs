//! ML Pipeline for Enhanced Anomaly Detection (Sprint 48)
//!
//! Integrates aprender v0.10.0 algorithms for comprehensive syscall analysis:
//! - `StandardScaler` for feature normalization
//! - Silhouette Score for cluster quality assessment
//! - DBSCAN for density-based clustering
//! - LOF for local outlier factor analysis
//! - PCA for dimensionality reduction
//!
//! # Toyota Way Principles
//!
//! - *Kaizen* (改善): Continuous improvement through standardized preprocessing
//! - *Muda* (無駄): Eliminate waste by using proven aprender algorithms
//! - *Poka-yoke* (ポカヨケ): Error-proofing via type-safe APIs

use aprender::cluster::{LocalOutlierFactor, DBSCAN};
use aprender::metrics::silhouette_score;
use aprender::preprocessing::{StandardScaler, PCA};
use aprender::primitives::Matrix;
use aprender::traits::{Transformer, UnsupervisedEstimator};
use std::collections::HashMap;
use thiserror::Error;

/// Errors for ML pipeline operations
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Feature extraction failed: {0}")]
    FeatureExtractionError(String),

    #[error("Preprocessing failed: {0}")]
    PreprocessingError(String),

    #[error("Clustering failed: {0}")]
    ClusteringError(String),

    #[error("Model not fitted")]
    NotFitted,
}

pub type Result<T> = std::result::Result<T, PipelineError>;

/// Normalized syscall features
#[derive(Debug, Clone)]
pub struct NormalizedFeatures {
    /// Feature matrix (`n_samples` x `n_features`)
    pub data: Matrix<f32>,
    /// Syscall names corresponding to each row
    pub syscall_names: Vec<String>,
    /// Feature names for each column
    pub feature_names: Vec<String>,
    /// Mean values used for normalization
    pub means: Vec<f32>,
    /// Standard deviations used for normalization
    pub stds: Vec<f32>,
}

/// Result of DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DBSCANResult {
    /// Cluster labels (-1 = noise/anomaly)
    pub labels: Vec<i32>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
    /// Syscall names
    pub syscall_names: Vec<String>,
    /// Cluster silhouette score (if more than 1 cluster)
    pub silhouette: Option<f32>,
}

/// Result of LOF analysis
#[derive(Debug, Clone)]
pub struct LOFResult {
    /// Outlier labels (1 = normal, -1 = outlier)
    pub labels: Vec<i32>,
    /// LOF scores (higher = more anomalous)
    pub scores: Vec<f32>,
    /// Syscall names
    pub syscall_names: Vec<String>,
    /// Outlier syscalls
    pub outliers: Vec<OutlierInfo>,
}

/// Information about a detected outlier
#[derive(Debug, Clone)]
pub struct OutlierInfo {
    pub syscall: String,
    pub lof_score: f32,
    pub avg_time_us: f64,
    pub call_count: u64,
}

/// Result of PCA dimensionality reduction
#[derive(Debug, Clone)]
pub struct PCAResult {
    /// Reduced feature matrix
    pub reduced_data: Matrix<f32>,
    /// Explained variance ratio per component
    pub explained_variance_ratio: Vec<f32>,
    /// Total variance explained
    pub total_variance_explained: f32,
    /// Syscall names
    pub syscall_names: Vec<String>,
}

/// Extract features from syscall data
///
/// Features extracted per syscall:
/// 1. Average duration (microseconds)
/// 2. Call count (log scale)
/// 3. Total duration (log scale)
pub fn extract_features(
    syscall_data: &HashMap<String, (u64, u64)>,
) -> Result<(Vec<String>, Matrix<f32>)> {
    let mut syscall_names = Vec::new();
    let mut features_data = Vec::new();

    for (name, (count, total_time_ns)) in syscall_data {
        if *count == 0 {
            continue;
        }

        let total_time_us = *total_time_ns as f64 / 1000.0;
        let avg_time_us = total_time_us / *count as f64;

        syscall_names.push(name.clone());

        // Feature vector: [avg_duration, log(count), log(total_duration)]
        features_data.push(avg_time_us as f32);
        features_data.push((*count as f32).ln().max(0.0));
        features_data.push((total_time_us as f32).ln().max(0.0));
    }

    let n_samples = syscall_names.len();
    if n_samples < 2 {
        return Err(PipelineError::InsufficientData { required: 2, actual: n_samples });
    }

    let n_features = 3;
    let matrix = Matrix::from_vec(n_samples, n_features, features_data)
        .map_err(|e| PipelineError::FeatureExtractionError(e.to_string()))?;

    Ok((syscall_names, matrix))
}

/// Normalize features using `StandardScaler`
pub fn normalize_features(
    syscall_names: Vec<String>,
    features: Matrix<f32>,
) -> Result<NormalizedFeatures> {
    let mut scaler = StandardScaler::new().with_mean(true).with_std(true);

    scaler.fit(&features).map_err(|e| PipelineError::PreprocessingError(e.to_string()))?;

    let normalized = scaler
        .transform(&features)
        .map_err(|e| PipelineError::PreprocessingError(e.to_string()))?;

    Ok(NormalizedFeatures {
        data: normalized,
        syscall_names,
        feature_names: vec![
            "avg_duration".to_string(),
            "log_count".to_string(),
            "log_total_duration".to_string(),
        ],
        means: scaler.mean().to_vec(),
        stds: scaler.std().to_vec(),
    })
}

/// Calculate silhouette score for clustering quality
pub fn calculate_silhouette(features: &Matrix<f32>, labels: &[i32]) -> Option<f32> {
    // Convert i32 labels to usize, filtering out noise points (-1)
    let valid_indices: Vec<usize> =
        labels.iter().enumerate().filter(|(_, &l)| l >= 0).map(|(i, _)| i).collect();

    if valid_indices.len() < 2 {
        return None;
    }

    // Check if we have at least 2 clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&l| l >= 0).collect();
    if unique_labels.len() < 2 {
        return None;
    }

    // Extract valid rows and labels
    let (_n_rows, n_cols) = features.shape();
    let mut valid_data = Vec::with_capacity(valid_indices.len() * n_cols);
    let mut valid_labels = Vec::with_capacity(valid_indices.len());

    for &i in &valid_indices {
        for j in 0..n_cols {
            valid_data.push(features.get(i, j));
        }
        valid_labels.push(labels[i] as usize);
    }

    let valid_matrix = Matrix::from_vec(valid_indices.len(), n_cols, valid_data).ok()?;
    Some(silhouette_score(&valid_matrix, &valid_labels))
}

/// Run DBSCAN clustering on syscall features
pub fn run_dbscan(
    features: &NormalizedFeatures,
    eps: f32,
    min_samples: usize,
) -> Result<DBSCANResult> {
    let mut dbscan = DBSCAN::new(eps, min_samples);

    dbscan.fit(&features.data).map_err(|e| PipelineError::ClusteringError(e.to_string()))?;

    let labels = dbscan.labels().clone();

    // Count clusters and noise
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    let n_clusters =
        labels.iter().filter(|&&l| l >= 0).collect::<std::collections::HashSet<_>>().len();

    // Calculate silhouette score if we have enough clusters
    let silhouette = calculate_silhouette(&features.data, &labels);

    Ok(DBSCANResult {
        labels,
        n_clusters,
        n_noise,
        syscall_names: features.syscall_names.clone(),
        silhouette,
    })
}

/// Run Local Outlier Factor analysis
pub fn run_lof(
    features: &NormalizedFeatures,
    syscall_data: &HashMap<String, (u64, u64)>,
    n_neighbors: usize,
    contamination: f32,
) -> Result<LOFResult> {
    let mut lof =
        LocalOutlierFactor::new().with_n_neighbors(n_neighbors).with_contamination(contamination);

    lof.fit(&features.data).map_err(|e| PipelineError::ClusteringError(e.to_string()))?;

    let labels = lof.predict(&features.data);
    let scores = lof.score_samples(&features.data);

    // Build outlier info for anomalies
    let mut outliers = Vec::new();
    for (i, (&label, &score)) in labels.iter().zip(scores.iter()).enumerate() {
        if label == -1 {
            // This is an outlier
            let syscall = &features.syscall_names[i];
            let (count, total_ns) = syscall_data.get(syscall).copied().unwrap_or((0, 0));

            let avg_time_us =
                if count > 0 { (total_ns as f64 / 1000.0) / count as f64 } else { 0.0 };

            outliers.push(OutlierInfo {
                syscall: syscall.clone(),
                lof_score: score,
                avg_time_us,
                call_count: count,
            });
        }
    }

    // Sort by LOF score (highest first)
    outliers
        .sort_by(|a, b| b.lof_score.partial_cmp(&a.lof_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(LOFResult { labels, scores, syscall_names: features.syscall_names.clone(), outliers })
}

/// Run PCA dimensionality reduction
pub fn run_pca(features: &NormalizedFeatures, n_components: usize) -> Result<PCAResult> {
    let (n_samples, n_features) = features.data.shape();
    let actual_components = n_components.min(n_samples).min(n_features);

    let mut pca = PCA::new(actual_components);

    let reduced = pca
        .fit_transform(&features.data)
        .map_err(|e| PipelineError::PreprocessingError(e.to_string()))?;

    let explained_variance_ratio =
        pca.explained_variance_ratio().map(<[f32]>::to_vec).unwrap_or_default();

    let total_variance_explained: f32 = explained_variance_ratio.iter().sum();

    Ok(PCAResult {
        reduced_data: reduced,
        explained_variance_ratio,
        total_variance_explained,
        syscall_names: features.syscall_names.clone(),
    })
}

/// Find optimal k for `KMeans` using silhouette score
pub fn find_optimal_k(features: &NormalizedFeatures, k_range: std::ops::Range<usize>) -> usize {
    use aprender::cluster::KMeans;
    use aprender::traits::UnsupervisedEstimator;

    let mut best_k = k_range.start.max(2);
    let mut best_score = f32::MIN;

    for k in k_range {
        if k < 2 || k >= features.syscall_names.len() {
            continue;
        }

        let mut kmeans = KMeans::new(k);
        if kmeans.fit(&features.data).is_err() {
            continue;
        }

        let labels = kmeans.predict(&features.data);
        let labels_i32: Vec<i32> = labels.iter().map(|&l| l as i32).collect();

        if let Some(score) = calculate_silhouette(&features.data, &labels_i32) {
            if score > best_score {
                best_score = score;
                best_k = k;
            }
        }
    }

    best_k
}

/// Format DBSCAN results for display
impl DBSCANResult {
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("\n=== DBSCAN Clustering Results ===\n");
        output.push_str(&format!("Clusters found: {}\n", self.n_clusters));
        output.push_str(&format!("Noise points (potential anomalies): {}\n", self.n_noise));

        if let Some(sil) = self.silhouette {
            output.push_str(&format!("Silhouette score: {sil:.3}\n"));
        }

        // List noise points
        if self.n_noise > 0 {
            output.push_str("\nNoise/anomaly syscalls:\n");
            for (i, label) in self.labels.iter().enumerate() {
                if *label == -1 {
                    output.push_str(&format!("  - {}\n", self.syscall_names[i]));
                }
            }
        }

        output
    }
}

/// Format LOF results for display
impl LOFResult {
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("\n=== Local Outlier Factor Analysis ===\n");
        output.push_str(&format!("Total samples: {}\n", self.labels.len()));
        output.push_str(&format!("Outliers detected: {}\n", self.outliers.len()));

        if !self.outliers.is_empty() {
            output.push_str("\nOutlier syscalls (by LOF score):\n");
            for outlier in &self.outliers {
                output.push_str(&format!(
                    "  - {} (LOF: {:.2}, avg: {:.2}µs, calls: {})\n",
                    outlier.syscall, outlier.lof_score, outlier.avg_time_us, outlier.call_count
                ));
            }
        }

        output
    }
}

/// Format PCA results for display
impl PCAResult {
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("\n=== PCA Dimensionality Reduction ===\n");
        let (n_samples, n_components) = self.reduced_data.shape();
        output.push_str(&format!("Reduced: {n_samples} samples x {n_components} components\n"));
        output.push_str(&format!(
            "Total variance explained: {:.1}%\n",
            self.total_variance_explained * 100.0
        ));

        output.push_str("\nVariance per component:\n");
        for (i, &var) in self.explained_variance_ratio.iter().enumerate() {
            output.push_str(&format!("  PC{}: {:.1}%\n", i + 1, var * 100.0));
        }

        output
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(PipelineError: Send, Sync);
static_assertions::assert_impl_all!(NormalizedFeatures: Send, Sync);
static_assertions::assert_impl_all!(DBSCANResult: Send, Sync);
static_assertions::assert_impl_all!(LOFResult: Send, Sync);
static_assertions::assert_impl_all!(OutlierInfo: Send, Sync);
static_assertions::assert_impl_all!(PCAResult: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== FEATURE EXTRACTION TESTS ====================

    #[test]
    fn test_extract_features_basic() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000)); // 10µs avg
        data.insert("read".to_string(), (50, 500_000)); // 10µs avg

        let (names, features) = extract_features(&data).expect("test");

        assert_eq!(names.len(), 2);
        let (rows, cols) = features.shape();
        assert_eq!(rows, 2);
        assert_eq!(cols, 3); // 3 features per syscall
    }

    #[test]
    fn test_extract_features_insufficient_data() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000));

        let result = extract_features(&data);
        assert!(matches!(result, Err(PipelineError::InsufficientData { .. })));
    }

    #[test]
    fn test_extract_features_empty() {
        let data = HashMap::new();
        let result = extract_features(&data);
        assert!(matches!(result, Err(PipelineError::InsufficientData { .. })));
    }

    #[test]
    fn test_extract_features_skips_zero_count() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000));
        data.insert("read".to_string(), (50, 500_000));
        data.insert("empty".to_string(), (0, 0)); // Should be skipped

        let (names, _) = extract_features(&data).expect("test");
        assert_eq!(names.len(), 2);
        assert!(!names.contains(&"empty".to_string()));
    }

    // ==================== NORMALIZATION TESTS ====================

    #[test]
    fn test_normalize_features_basic() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000));
        data.insert("read".to_string(), (50, 500_000));
        data.insert("openat".to_string(), (20, 200_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names.clone(), features).expect("test");

        assert_eq!(normalized.syscall_names.len(), 3);
        assert_eq!(normalized.feature_names.len(), 3);
        assert_eq!(normalized.means.len(), 3);
        assert_eq!(normalized.stds.len(), 3);
    }

    #[test]
    fn test_normalize_features_zero_mean() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000));
        data.insert("read".to_string(), (50, 500_000));
        data.insert("openat".to_string(), (20, 200_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        // After normalization, mean of each column should be ~0
        let (n_rows, n_cols) = normalized.data.shape();
        for j in 0..n_cols {
            let sum: f32 = (0..n_rows).map(|i| normalized.data.get(i, j)).sum();
            let mean = sum / n_rows as f32;
            assert!(mean.abs() < 0.01, "Column {} mean should be ~0, got {}", j, mean);
        }
    }

    // ==================== SILHOUETTE SCORE TESTS ====================

    #[test]
    fn test_silhouette_score_well_separated() {
        // Two well-separated clusters
        let data = vec![
            1.0, 1.0, // Cluster 0
            1.1, 1.1, // Cluster 0
            10.0, 10.0, // Cluster 1
            10.1, 10.1, // Cluster 1
        ];
        let matrix = Matrix::from_vec(4, 2, data).expect("test");
        let labels = vec![0, 0, 1, 1];

        let score = calculate_silhouette(&matrix, &labels);
        assert!(score.is_some());
        let s = score.expect("test");
        assert!(s > 0.8, "Well-separated clusters should have high silhouette, got {}", s);
    }

    #[test]
    fn test_silhouette_score_single_cluster() {
        let data = vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2];
        let matrix = Matrix::from_vec(3, 2, data).expect("test");
        let labels = vec![0, 0, 0]; // Single cluster

        let score = calculate_silhouette(&matrix, &labels);
        assert!(score.is_none()); // Need at least 2 clusters
    }

    #[test]
    fn test_silhouette_score_with_noise() {
        let data = vec![
            1.0, 1.0, // Cluster 0
            1.1, 1.1, // Cluster 0
            10.0, 10.0, // Cluster 1
            5.0, 5.0, // Noise
        ];
        let matrix = Matrix::from_vec(4, 2, data).expect("test");
        let labels = vec![0, 0, 1, -1]; // -1 is noise

        let score = calculate_silhouette(&matrix, &labels);
        assert!(score.is_some()); // Should work, ignoring noise
    }

    // ==================== DBSCAN TESTS ====================

    #[test]
    fn test_dbscan_finds_clusters() {
        let mut data = HashMap::new();
        // Group 1: fast syscalls
        data.insert("write".to_string(), (1000, 10_000_000)); // 10us avg
        data.insert("read".to_string(), (1000, 10_000_000)); // 10us avg

        // Group 2: slow syscalls
        data.insert("mmap".to_string(), (100, 100_000_000)); // 1000us avg
        data.insert("munmap".to_string(), (100, 100_000_000)); // 1000µs avg

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        let result = run_dbscan(&normalized, 1.0, 2).expect("test");

        // Should find clusters (exact number depends on parameters)
        assert!(result.n_clusters >= 1);
        assert_eq!(result.syscall_names.len(), 4);
    }

    #[test]
    fn test_dbscan_identifies_noise() {
        let mut data = HashMap::new();
        // Normal syscalls
        data.insert("write".to_string(), (1000, 10_000_000));
        data.insert("read".to_string(), (1000, 10_000_000));
        data.insert("close".to_string(), (1000, 10_000_000));
        // Outlier
        data.insert("slow_syscall".to_string(), (10, 1_000_000_000)); // Very slow

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        // Use strict parameters to identify outliers
        let result = run_dbscan(&normalized, 0.5, 2).expect("test");

        // Should have some noise points
        assert!(result.n_noise > 0 || result.n_clusters > 1);
    }

    // ==================== LOF TESTS ====================

    #[test]
    fn test_lof_detects_outliers() {
        let mut data = HashMap::new();
        // Normal syscalls
        data.insert("write".to_string(), (1000, 10_000_000));
        data.insert("read".to_string(), (1000, 10_000_000));
        data.insert("close".to_string(), (1000, 10_000_000));
        // Outlier
        data.insert("slow_syscall".to_string(), (10, 1_000_000_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        let result = run_lof(&normalized, &data, 2, 0.25).expect("test");

        // Should detect outliers
        assert!(!result.outliers.is_empty() || result.labels.contains(&-1));
        assert_eq!(result.scores.len(), 4);
    }

    #[test]
    fn test_lof_scores_are_positive() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (1000, 10_000_000));
        data.insert("read".to_string(), (1000, 10_000_000));
        data.insert("close".to_string(), (1000, 10_000_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        let result = run_lof(&normalized, &data, 2, 0.1).expect("test");

        // LOF scores should be positive
        for score in &result.scores {
            assert!(*score > 0.0, "LOF scores should be positive");
        }
    }

    // ==================== PCA TESTS ====================

    #[test]
    fn test_pca_reduces_dimensions() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (1000, 10_000_000));
        data.insert("read".to_string(), (1000, 10_000_000));
        data.insert("close".to_string(), (1000, 10_000_000));
        data.insert("openat".to_string(), (500, 5_000_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        let result = run_pca(&normalized, 2).expect("test");

        let (n_samples, n_components) = result.reduced_data.shape();
        assert_eq!(n_samples, 4);
        assert_eq!(n_components, 2); // Reduced from 3 to 2
    }

    #[test]
    fn test_pca_variance_explained() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (1000, 10_000_000));
        data.insert("read".to_string(), (1000, 10_000_000));
        data.insert("close".to_string(), (1000, 10_000_000));
        data.insert("openat".to_string(), (500, 5_000_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        let result = run_pca(&normalized, 3).expect("test");

        // Total variance should be <= 1.0
        assert!(
            result.total_variance_explained <= 1.01, // Allow small float error
            "Total variance {} should be <= 1.0",
            result.total_variance_explained
        );
    }

    // ==================== FIND OPTIMAL K TESTS ====================

    #[test]
    fn test_find_optimal_k() {
        let mut data = HashMap::new();
        // Two distinct groups
        data.insert("write".to_string(), (1000, 10_000_000));
        data.insert("read".to_string(), (1000, 10_000_000));
        data.insert("mmap".to_string(), (100, 100_000_000));
        data.insert("munmap".to_string(), (100, 100_000_000));

        let (names, features) = extract_features(&data).expect("test");
        let normalized = normalize_features(names, features).expect("test");

        let optimal_k = find_optimal_k(&normalized, 2..5);

        // Should be 2 for two distinct groups
        assert!((2..=4).contains(&optimal_k));
    }

    // ==================== FORMATTING TESTS ====================

    #[test]
    fn test_dbscan_result_format() {
        let result = DBSCANResult {
            labels: vec![0, 0, -1, 1],
            n_clusters: 2,
            n_noise: 1,
            syscall_names: vec![
                "write".to_string(),
                "read".to_string(),
                "anomaly".to_string(),
                "close".to_string(),
            ],
            silhouette: Some(0.75),
        };

        let formatted = result.format();
        assert!(formatted.contains("DBSCAN"));
        assert!(formatted.contains("Clusters found: 2"));
        assert!(formatted.contains("Noise points"));
        assert!(formatted.contains("Silhouette"));
        assert!(formatted.contains("anomaly"));
    }

    #[test]
    fn test_lof_result_format() {
        let result = LOFResult {
            labels: vec![1, 1, -1],
            scores: vec![1.0, 1.1, 3.5],
            syscall_names: vec!["write".to_string(), "read".to_string(), "slow".to_string()],
            outliers: vec![OutlierInfo {
                syscall: "slow".to_string(),
                lof_score: 3.5,
                avg_time_us: 1000.0,
                call_count: 10,
            }],
        };

        let formatted = result.format();
        assert!(formatted.contains("Local Outlier Factor"));
        assert!(formatted.contains("Outliers detected: 1"));
        assert!(formatted.contains("slow"));
        assert!(formatted.contains("LOF: 3.50"));
    }

    #[test]
    fn test_pca_result_format() {
        let result = PCAResult {
            reduced_data: Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test"),
            explained_variance_ratio: vec![0.8, 0.15],
            total_variance_explained: 0.95,
            syscall_names: vec!["write".to_string(), "read".to_string(), "close".to_string()],
        };

        let formatted = result.format();
        assert!(formatted.contains("PCA"));
        assert!(formatted.contains("3 samples x 2 components"));
        assert!(formatted.contains("95.0%"));
    }

    // ==================== PROPERTY-BASED TESTS ====================

    #[test]
    fn test_normalization_preserves_sample_count() {
        use proptest::prelude::*;

        proptest::proptest!(|(n_syscalls in 3usize..10)| {
            let mut data = HashMap::new();
            for i in 0..n_syscalls {
                data.insert(format!("syscall_{}", i), ((i + 1) as u64 * 100, (i + 1) as u64 * 1_000_000));
            }

            let (names, features) = extract_features(&data).expect("test");
            let normalized = normalize_features(names.clone(), features).expect("test");

            prop_assert_eq!(normalized.syscall_names.len(), names.len());
        });
    }

    #[test]
    fn test_silhouette_bounds() {
        // Silhouette score should always be in [-1, 1]
        let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let matrix = Matrix::from_vec(4, 2, data).expect("test");
        let labels = vec![0, 0, 1, 1];

        if let Some(score) = calculate_silhouette(&matrix, &labels) {
            assert!((-1.0..=1.0).contains(&score));
        }
    }
}
