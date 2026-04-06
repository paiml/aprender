//! Tiny Model Representation (spec §4.3)
//!
//! Specialized representations for small models (< 1 MB) that minimize
//! overhead while preserving full functionality. Designed for:
//! - Educational examples
//! - Edge deployment
//! - WASM playgrounds
//! - Embedded systems
//!
//! # Model Types
//! - **Linear**: Coefficients + intercept (< 1 KB typical)
//! - **Stump**: Single decision split (< 100 bytes)
//! - **`NaiveBayes`**: Means + variances per class (< 10 KB typical)
//! - **`KMeans`**: Cluster centroids (< 100 KB typical)
//! - **Compressed**: Larger models with compression

use super::DataCompression;

/// Compact representation for tiny models (educational/edge deployment)
///
/// Provides specialized storage for common small model architectures,
/// avoiding the overhead of generic serialization formats.
///
/// # Example
/// ```
/// use aprender::embed::TinyModelRepr;
///
/// // Linear model with 10 features
/// let linear = TinyModelRepr::linear(
///     vec![0.5, -0.3, 0.8, 0.2, -0.1, 0.4, -0.6, 0.9, 0.1, -0.4],
///     1.5,
/// );
/// assert_eq!(linear.size_bytes(), 44); // 10 * 4 + 4
///
/// // Decision stump
/// let stump = TinyModelRepr::stump(3, 0.5, -1.0, 1.0);
/// assert_eq!(stump.size_bytes(), 14); // 2 + 4 + 4 + 4
///
/// // K-Means with 3 clusters, 2 features each
/// let kmeans = TinyModelRepr::kmeans(vec![
///     vec![1.0, 2.0],
///     vec![4.0, 5.0],
///     vec![7.0, 8.0],
/// ]);
/// assert_eq!(kmeans.size_bytes(), 24); // 3 * 2 * 4
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum TinyModelRepr {
    /// Linear model: coefficients + intercept (< 1 KB typical)
    Linear {
        /// Model coefficients (one per feature)
        coefficients: Vec<f32>,
        /// Intercept (bias) term
        intercept: f32,
    },

    /// Decision stump: single split (< 100 bytes)
    Stump {
        /// Feature index to split on
        feature_idx: u16,
        /// Threshold value for the split
        threshold: f32,
        /// Prediction for left branch (< threshold)
        left_value: f32,
        /// Prediction for right branch (>= threshold)
        right_value: f32,
    },

    /// Naive Bayes: means + variances per class (< 10 KB typical)
    NaiveBayes {
        /// Prior probabilities for each class
        class_priors: Vec<f32>,
        /// Mean values per class per feature `[n_classes][n_features]`
        means: Vec<Vec<f32>>,
        /// Variance values per class per feature `[n_classes][n_features]`
        variances: Vec<Vec<f32>>,
    },

    /// K-Means: cluster centroids (< 100 KB typical)
    KMeans {
        /// Centroid coordinates `[n_clusters][n_features]`
        centroids: Vec<Vec<f32>>,
    },

    /// Logistic regression: coefficients per class
    LogisticRegression {
        /// Coefficients `[n_classes][n_features]` or `[n_features]` for binary
        coefficients: Vec<Vec<f32>>,
        /// Intercepts per class
        intercepts: Vec<f32>,
    },

    /// k-Nearest Neighbors: stored reference points
    KNN {
        /// Reference points `[n_samples][n_features]`
        reference_points: Vec<Vec<f32>>,
        /// Labels for reference points
        labels: Vec<u32>,
        /// Number of neighbors
        k: u32,
    },

    /// Compressed representation for larger tiny models
    Compressed {
        /// Compression strategy used
        compression: DataCompression,
        /// Compressed data bytes
        data: Vec<u8>,
        /// Original (uncompressed) size in bytes
        original_size: usize,
    },
}

include!("arguments.rs");
include!("tiny_model_error.rs");
include!("tests_tiny_models.rs");
