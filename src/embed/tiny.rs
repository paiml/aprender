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
//! - **NaiveBayes**: Means + variances per class (< 10 KB typical)
//! - **KMeans**: Cluster centroids (< 100 KB typical)
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
        /// Mean values per class per feature [n_classes][n_features]
        means: Vec<Vec<f32>>,
        /// Variance values per class per feature [n_classes][n_features]
        variances: Vec<Vec<f32>>,
    },

    /// K-Means: cluster centroids (< 100 KB typical)
    KMeans {
        /// Centroid coordinates [n_clusters][n_features]
        centroids: Vec<Vec<f32>>,
    },

    /// Logistic regression: coefficients per class
    LogisticRegression {
        /// Coefficients [n_classes][n_features] or [n_features] for binary
        coefficients: Vec<Vec<f32>>,
        /// Intercepts per class
        intercepts: Vec<f32>,
    },

    /// k-Nearest Neighbors: stored reference points
    KNN {
        /// Reference points [n_samples][n_features]
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

impl TinyModelRepr {
    /// Create a linear model representation
    ///
    /// # Arguments
    /// * `coefficients` - Feature coefficients
    /// * `intercept` - Bias term
    #[must_use]
    pub fn linear(coefficients: Vec<f32>, intercept: f32) -> Self {
        Self::Linear {
            coefficients,
            intercept,
        }
    }

    /// Create a decision stump representation
    ///
    /// # Arguments
    /// * `feature_idx` - Index of feature to split on
    /// * `threshold` - Split threshold
    /// * `left_value` - Prediction for values < threshold
    /// * `right_value` - Prediction for values >= threshold
    #[must_use]
    pub const fn stump(
        feature_idx: u16,
        threshold: f32,
        left_value: f32,
        right_value: f32,
    ) -> Self {
        Self::Stump {
            feature_idx,
            threshold,
            left_value,
            right_value,
        }
    }

    /// Create a Naive Bayes model representation
    ///
    /// # Arguments
    /// * `class_priors` - Prior probability for each class
    /// * `means` - Mean values [n_classes][n_features]
    /// * `variances` - Variance values [n_classes][n_features]
    #[must_use]
    pub fn naive_bayes(
        class_priors: Vec<f32>,
        means: Vec<Vec<f32>>,
        variances: Vec<Vec<f32>>,
    ) -> Self {
        Self::NaiveBayes {
            class_priors,
            means,
            variances,
        }
    }

    /// Create a K-Means model representation
    ///
    /// # Arguments
    /// * `centroids` - Cluster centroids [n_clusters][n_features]
    #[must_use]
    pub fn kmeans(centroids: Vec<Vec<f32>>) -> Self {
        Self::KMeans { centroids }
    }

    /// Create a logistic regression representation
    ///
    /// # Arguments
    /// * `coefficients` - Coefficients [n_classes][n_features]
    /// * `intercepts` - Intercept per class
    #[must_use]
    pub fn logistic_regression(coefficients: Vec<Vec<f32>>, intercepts: Vec<f32>) -> Self {
        Self::LogisticRegression {
            coefficients,
            intercepts,
        }
    }

    /// Create a KNN model representation
    ///
    /// # Arguments
    /// * `reference_points` - Stored reference samples
    /// * `labels` - Labels for each reference point
    /// * `k` - Number of neighbors
    #[must_use]
    pub fn knn(reference_points: Vec<Vec<f32>>, labels: Vec<u32>, k: u32) -> Self {
        Self::KNN {
            reference_points,
            labels,
            k,
        }
    }

    /// Create a compressed model representation
    ///
    /// # Arguments
    /// * `compression` - Compression strategy
    /// * `data` - Compressed bytes
    /// * `original_size` - Size before compression
    #[must_use]
    pub fn compressed(compression: DataCompression, data: Vec<u8>, original_size: usize) -> Self {
        Self::Compressed {
            compression,
            data,
            original_size,
        }
    }

    /// Total size in bytes (for bundling decisions)
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Linear { coefficients, .. } => coefficients.len() * 4 + 4,
            Self::Stump { .. } => 2 + 4 + 4 + 4, // u16 + 3*f32
            Self::NaiveBayes {
                class_priors,
                means,
                ..
            } => {
                let n_classes = class_priors.len();
                let n_features = means.first().map_or(0, Vec::len);
                n_classes * 4 + n_classes * n_features * 8 // priors + means + variances
            }
            Self::KMeans { centroids } => centroids.iter().map(|c| c.len() * 4).sum(),
            Self::LogisticRegression {
                coefficients,
                intercepts,
            } => {
                let coef_size: usize = coefficients.iter().map(|c| c.len() * 4).sum();
                coef_size + intercepts.len() * 4
            }
            Self::KNN {
                reference_points,
                labels,
                ..
            } => {
                let points_size: usize = reference_points.iter().map(|p| p.len() * 4).sum();
                points_size + labels.len() * 4 + 4 // points + labels + k
            }
            Self::Compressed { data, .. } => data.len(),
        }
    }

    /// Model type name
    #[must_use]
    pub const fn model_type(&self) -> &'static str {
        match self {
            Self::Linear { .. } => "linear",
            Self::Stump { .. } => "stump",
            Self::NaiveBayes { .. } => "naive_bayes",
            Self::KMeans { .. } => "kmeans",
            Self::LogisticRegression { .. } => "logistic_regression",
            Self::KNN { .. } => "knn",
            Self::Compressed { .. } => "compressed",
        }
    }

    /// Number of parameters in the model
    #[must_use]
    pub fn n_parameters(&self) -> usize {
        match self {
            Self::Linear { coefficients, .. } => coefficients.len() + 1,
            Self::Stump { .. } => 4, // feature_idx, threshold, left, right
            Self::NaiveBayes {
                class_priors,
                means,
                variances,
            } => {
                class_priors.len()
                    + means.iter().map(Vec::len).sum::<usize>()
                    + variances.iter().map(Vec::len).sum::<usize>()
            }
            Self::KMeans { centroids } => centroids.iter().map(Vec::len).sum(),
            Self::LogisticRegression {
                coefficients,
                intercepts,
            } => coefficients.iter().map(Vec::len).sum::<usize>() + intercepts.len(),
            Self::KNN {
                reference_points, ..
            } => reference_points.iter().map(Vec::len).sum(),
            Self::Compressed { original_size, .. } => original_size / 4, // approximate
        }
    }

    /// Number of features expected in input
    #[must_use]
    pub fn n_features(&self) -> Option<usize> {
        match self {
            Self::Linear { coefficients, .. } => Some(coefficients.len()),
            Self::NaiveBayes { means, .. } => means.first().map(Vec::len),
            Self::KMeans { centroids } => centroids.first().map(Vec::len),
            Self::LogisticRegression { coefficients, .. } => coefficients.first().map(Vec::len),
            Self::KNN {
                reference_points, ..
            } => reference_points.first().map(Vec::len),
            // Stump depends on feature_idx, Compressed unknown
            Self::Stump { .. } | Self::Compressed { .. } => None,
        }
    }

    /// Predict for a single sample (linear models only)
    ///
    /// # Arguments
    /// * `features` - Input feature vector
    ///
    /// # Returns
    /// Prediction or None if model type doesn't support direct prediction
    pub fn predict_linear(&self, features: &[f32]) -> Option<f32> {
        match self {
            Self::Linear {
                coefficients,
                intercept,
            } => {
                if features.len() != coefficients.len() {
                    return None;
                }
                let dot: f32 = coefficients
                    .iter()
                    .zip(features.iter())
                    .map(|(c, x)| c * x)
                    .sum();
                Some(dot + intercept)
            }
            _ => None,
        }
    }

    /// Predict for a single sample (stump only)
    ///
    /// # Arguments
    /// * `features` - Input feature vector
    ///
    /// # Returns
    /// Prediction or None if model type is not stump
    pub fn predict_stump(&self, features: &[f32]) -> Option<f32> {
        match self {
            Self::Stump {
                feature_idx,
                threshold,
                left_value,
                right_value,
            } => {
                let idx = *feature_idx as usize;
                features.get(idx).map(|&x| {
                    if x < *threshold {
                        *left_value
                    } else {
                        *right_value
                    }
                })
            }
            _ => None,
        }
    }

    /// Find nearest centroid (K-Means only)
    ///
    /// # Arguments
    /// * `features` - Input feature vector
    ///
    /// # Returns
    /// Index of nearest centroid or None if not K-Means
    pub fn predict_kmeans(&self, features: &[f32]) -> Option<usize> {
        match self {
            Self::KMeans { centroids } => {
                if centroids.is_empty() {
                    return None;
                }
                let expected_len = centroids[0].len();
                if features.len() != expected_len {
                    return None;
                }

                centroids
                    .iter()
                    .enumerate()
                    .map(|(idx, centroid)| {
                        let dist: f32 = centroid
                            .iter()
                            .zip(features.iter())
                            .map(|(c, x)| (c - x).powi(2))
                            .sum();
                        (idx, dist)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
            }
            _ => None,
        }
    }

    /// Validate model integrity
    pub fn validate(&self) -> Result<(), TinyModelError> {
        match self {
            Self::Linear { coefficients, .. } => {
                if coefficients.is_empty() {
                    return Err(TinyModelError::EmptyModel);
                }
                for (i, &c) in coefficients.iter().enumerate() {
                    if !c.is_finite() {
                        return Err(TinyModelError::InvalidCoefficient { index: i, value: c });
                    }
                }
                Ok(())
            }
            Self::NaiveBayes {
                class_priors,
                means,
                variances,
            } => {
                if class_priors.is_empty() {
                    return Err(TinyModelError::EmptyModel);
                }
                if means.len() != class_priors.len() || variances.len() != class_priors.len() {
                    return Err(TinyModelError::ShapeMismatch {
                        message: "Means/variances length must match class_priors".into(),
                    });
                }
                // Check all variances are positive
                for (class_idx, var) in variances.iter().enumerate() {
                    for (feat_idx, &v) in var.iter().enumerate() {
                        if v <= 0.0 {
                            return Err(TinyModelError::InvalidVariance {
                                class: class_idx,
                                feature: feat_idx,
                                value: v,
                            });
                        }
                    }
                }
                Ok(())
            }
            Self::KMeans { centroids } => {
                if centroids.is_empty() {
                    return Err(TinyModelError::EmptyModel);
                }
                let first_len = centroids[0].len();
                for (i, c) in centroids.iter().enumerate() {
                    if c.len() != first_len {
                        return Err(TinyModelError::ShapeMismatch {
                            message: format!(
                                "Centroid {i} has length {}, expected {first_len}",
                                c.len()
                            ),
                        });
                    }
                }
                Ok(())
            }
            Self::KNN {
                reference_points,
                labels,
                k,
            } => {
                if reference_points.is_empty() {
                    return Err(TinyModelError::EmptyModel);
                }
                if reference_points.len() != labels.len() {
                    return Err(TinyModelError::ShapeMismatch {
                        message: "Reference points and labels must have same length".into(),
                    });
                }
                if *k == 0 || *k as usize > reference_points.len() {
                    return Err(TinyModelError::InvalidK {
                        k: *k,
                        n_samples: reference_points.len(),
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Check if this representation fits within a size budget
    #[must_use]
    pub fn fits_within(&self, max_bytes: usize) -> bool {
        self.size_bytes() <= max_bytes
    }

    /// Human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::Linear { coefficients, .. } => {
                format!(
                    "Linear({} features, {} bytes)",
                    coefficients.len(),
                    self.size_bytes()
                )
            }
            Self::Stump {
                feature_idx,
                threshold,
                ..
            } => {
                format!("Stump(feature[{feature_idx}] < {threshold:.3})")
            }
            Self::NaiveBayes {
                class_priors,
                means,
                ..
            } => {
                let n_features = means.first().map_or(0, Vec::len);
                format!(
                    "NaiveBayes({} classes, {} features, {} bytes)",
                    class_priors.len(),
                    n_features,
                    self.size_bytes()
                )
            }
            Self::KMeans { centroids } => {
                let n_features = centroids.first().map_or(0, Vec::len);
                format!(
                    "KMeans({} clusters, {} features, {} bytes)",
                    centroids.len(),
                    n_features,
                    self.size_bytes()
                )
            }
            Self::LogisticRegression { coefficients, .. } => {
                let n_features = coefficients.first().map_or(0, Vec::len);
                format!(
                    "LogisticRegression({} classes, {} features, {} bytes)",
                    coefficients.len(),
                    n_features,
                    self.size_bytes()
                )
            }
            Self::KNN {
                reference_points,
                k,
                ..
            } => {
                let n_features = reference_points.first().map_or(0, Vec::len);
                format!(
                    "KNN(k={k}, {} samples, {} features, {} bytes)",
                    reference_points.len(),
                    n_features,
                    self.size_bytes()
                )
            }
            Self::Compressed {
                compression,
                data,
                original_size,
            } => {
                let ratio = if data.is_empty() {
                    1.0
                } else {
                    *original_size as f32 / data.len() as f32
                };
                format!(
                    "Compressed({}, {} bytes, {:.1}x ratio)",
                    compression.name(),
                    data.len(),
                    ratio
                )
            }
        }
    }
}

/// Errors specific to tiny model operations
#[derive(Debug, Clone)]
pub enum TinyModelError {
    /// Model has no parameters
    EmptyModel,
    /// Invalid coefficient value
    InvalidCoefficient { index: usize, value: f32 },
    /// Invalid variance (must be positive)
    InvalidVariance {
        class: usize,
        feature: usize,
        value: f32,
    },
    /// Shape mismatch in model components
    ShapeMismatch { message: String },
    /// Invalid k for KNN
    InvalidK { k: u32, n_samples: usize },
    /// Feature dimension mismatch
    FeatureMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for TinyModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModel => write!(f, "Model has no parameters"),
            Self::InvalidCoefficient { index, value } => {
                write!(f, "Invalid coefficient at index {index}: {value}")
            }
            Self::InvalidVariance {
                class,
                feature,
                value,
            } => {
                write!(
                    f,
                    "Invalid variance for class {class}, feature {feature}: {value}"
                )
            }
            Self::ShapeMismatch { message } => write!(f, "Shape mismatch: {message}"),
            Self::InvalidK { k, n_samples } => {
                write!(f, "Invalid k={k} for {n_samples} samples")
            }
            Self::FeatureMismatch { expected, got } => {
                write!(f, "Expected {expected} features, got {got}")
            }
        }
    }
}

impl std::error::Error for TinyModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model() {
        let model = TinyModelRepr::linear(vec![0.5, -0.3, 0.8], 1.0);

        assert_eq!(model.model_type(), "linear");
        assert_eq!(model.size_bytes(), 16); // 3*4 + 4
        assert_eq!(model.n_parameters(), 4); // 3 coefs + 1 intercept
        assert_eq!(model.n_features(), Some(3));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_linear_predict() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0, 3.0], 1.0);

        // 1*1 + 2*2 + 3*3 + 1 = 1 + 4 + 9 + 1 = 15
        let pred = model.predict_linear(&[1.0, 2.0, 3.0]);
        assert!((pred.unwrap() - 15.0).abs() < f32::EPSILON);

        // Wrong number of features
        assert!(model.predict_linear(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_stump_model() {
        let model = TinyModelRepr::stump(2, 0.5, -1.0, 1.0);

        assert_eq!(model.model_type(), "stump");
        assert_eq!(model.size_bytes(), 14);
        assert_eq!(model.n_parameters(), 4);
    }

    #[test]
    fn test_stump_predict() {
        let model = TinyModelRepr::stump(1, 0.5, -1.0, 1.0);

        // Feature 1 < 0.5 -> left value (-1.0)
        assert_eq!(model.predict_stump(&[0.0, 0.3, 0.0]), Some(-1.0));

        // Feature 1 >= 0.5 -> right value (1.0)
        assert_eq!(model.predict_stump(&[0.0, 0.7, 0.0]), Some(1.0));

        // Feature 1 == 0.5 -> right value (>= threshold)
        assert_eq!(model.predict_stump(&[0.0, 0.5, 0.0]), Some(1.0));

        // Out of bounds feature index
        assert!(model.predict_stump(&[0.0]).is_none());
    }

    #[test]
    fn test_naive_bayes_model() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );

        assert_eq!(model.model_type(), "naive_bayes");
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_naive_bayes_invalid_variance() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0], vec![2.0]],
            vec![vec![0.1], vec![-0.1]], // negative variance
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidVariance { .. })
        ));
    }

    #[test]
    fn test_kmeans_model() {
        let model = TinyModelRepr::kmeans(vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]]);

        assert_eq!(model.model_type(), "kmeans");
        assert_eq!(model.size_bytes(), 24); // 3 * 2 * 4
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_kmeans_predict() {
        let model = TinyModelRepr::kmeans(vec![vec![0.0, 0.0], vec![10.0, 10.0]]);

        // Closer to cluster 0
        assert_eq!(model.predict_kmeans(&[1.0, 1.0]), Some(0));

        // Closer to cluster 1
        assert_eq!(model.predict_kmeans(&[9.0, 9.0]), Some(1));

        // Wrong feature count
        assert!(model.predict_kmeans(&[1.0]).is_none());
    }

    #[test]
    fn test_kmeans_shape_mismatch() {
        let model = TinyModelRepr::kmeans(vec![
            vec![1.0, 2.0],
            vec![3.0], // different length
        ]);

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_logistic_regression_model() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![0.5, 0.6],
        );

        assert_eq!(model.model_type(), "logistic_regression");
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_knn_model() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        assert_eq!(model.model_type(), "knn");
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_knn_invalid_k() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0, 1],
            5, // k > n_samples
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidK { .. })
        ));
    }

    #[test]
    fn test_compressed_model() {
        let model = TinyModelRepr::compressed(DataCompression::zstd(), vec![1, 2, 3, 4, 5], 100);

        assert_eq!(model.model_type(), "compressed");
        assert_eq!(model.size_bytes(), 5);
    }

    #[test]
    fn test_fits_within() {
        let small = TinyModelRepr::linear(vec![1.0, 2.0], 0.0);
        let large = TinyModelRepr::kmeans(vec![vec![0.0; 1000]; 100]);

        assert!(small.fits_within(100));
        assert!(!small.fits_within(5));
        assert!(large.fits_within(1_000_000));
        assert!(!large.fits_within(100));
    }

    #[test]
    fn test_summary() {
        let linear = TinyModelRepr::linear(vec![1.0, 2.0, 3.0], 0.0);
        let summary = linear.summary();
        assert!(summary.contains("Linear"));
        assert!(summary.contains("3 features"));

        let stump = TinyModelRepr::stump(5, 0.5, -1.0, 1.0);
        let summary = stump.summary();
        assert!(summary.contains("Stump"));
        assert!(summary.contains("feature[5]"));

        let kmeans = TinyModelRepr::kmeans(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let summary = kmeans.summary();
        assert!(summary.contains("KMeans"));
        assert!(summary.contains("2 clusters"));
    }

    #[test]
    fn test_empty_model_validation() {
        let empty_linear = TinyModelRepr::linear(vec![], 0.0);
        assert!(matches!(
            empty_linear.validate(),
            Err(TinyModelError::EmptyModel)
        ));

        let empty_kmeans = TinyModelRepr::kmeans(vec![]);
        assert!(matches!(
            empty_kmeans.validate(),
            Err(TinyModelError::EmptyModel)
        ));
    }

    #[test]
    fn test_invalid_coefficient() {
        let model = TinyModelRepr::linear(vec![1.0, f32::NAN, 3.0], 0.0);
        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidCoefficient { index: 1, .. })
        ));

        let model = TinyModelRepr::linear(vec![f32::INFINITY, 2.0], 0.0);
        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidCoefficient { index: 0, .. })
        ));
    }

    #[test]
    fn test_tiny_model_error_display() {
        let err = TinyModelError::EmptyModel;
        assert_eq!(format!("{err}"), "Model has no parameters");

        let err = TinyModelError::InvalidK {
            k: 10,
            n_samples: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_naive_bayes_shape_mismatch() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0]], // only 1 class but 2 priors
            vec![vec![0.1], vec![0.2]],
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_knn_labels_mismatch() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0], // only 1 label for 2 points
            1,
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::ShapeMismatch { .. })
        ));
    }

    // ============================================================================
    // Additional Coverage Tests
    // ============================================================================

    #[test]
    fn test_kmeans_predict_empty_centroids() {
        let model = TinyModelRepr::kmeans(vec![]);
        assert!(model.predict_kmeans(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_logistic_regression_size_bytes() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            vec![0.7, 0.8],
        );

        // 2 classes * 3 features * 4 bytes + 2 intercepts * 4 bytes = 24 + 8 = 32
        assert_eq!(model.size_bytes(), 32);
    }

    #[test]
    fn test_knn_size_bytes() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        // 3 points * 2 features * 4 bytes + 3 labels * 4 bytes + 4 (k)
        // = 24 + 12 + 4 = 40
        assert_eq!(model.size_bytes(), 40);
    }

    #[test]
    fn test_compressed_summary() {
        let model = TinyModelRepr::compressed(DataCompression::None, vec![1, 2, 3, 4, 5], 100);

        let summary = model.summary();
        assert!(summary.contains("Compressed"));
        assert!(summary.contains("5 bytes"));
        assert!(summary.contains("ratio"));
    }

    #[test]
    fn test_compressed_summary_empty() {
        let model = TinyModelRepr::compressed(DataCompression::zstd(), vec![], 0);

        let summary = model.summary();
        assert!(summary.contains("Compressed"));
        // Should handle empty data without panicking
    }

    #[test]
    fn test_logistic_regression_summary() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
            vec![0.1, 0.2, 0.3],
        );

        let summary = model.summary();
        assert!(summary.contains("LogisticRegression"));
        assert!(summary.contains("3 classes"));
        assert!(summary.contains("2 features"));
    }

    #[test]
    fn test_knn_summary() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        let summary = model.summary();
        assert!(summary.contains("KNN"));
        assert!(summary.contains("k=2"));
        assert!(summary.contains("3 samples"));
        assert!(summary.contains("2 features"));
    }

    #[test]
    fn test_naive_bayes_summary() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.3, 0.7],
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        );

        let summary = model.summary();
        assert!(summary.contains("NaiveBayes"));
        assert!(summary.contains("2 classes"));
        assert!(summary.contains("3 features"));
    }

    #[test]
    fn test_feature_mismatch_error_display() {
        let err = TinyModelError::FeatureMismatch {
            expected: 10,
            got: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_invalid_coefficient_display() {
        let err = TinyModelError::InvalidCoefficient {
            index: 3,
            value: f32::NAN,
        };
        let msg = format!("{err}");
        assert!(msg.contains("3"));
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_invalid_variance_display() {
        let err = TinyModelError::InvalidVariance {
            class: 1,
            feature: 2,
            value: -0.5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("class 1"));
        assert!(msg.contains("feature 2"));
        assert!(msg.contains("-0.5"));
    }

    #[test]
    fn test_shape_mismatch_display() {
        let err = TinyModelError::ShapeMismatch {
            message: "test error".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("test error"));
    }

    #[test]
    fn test_n_features_stump() {
        let model = TinyModelRepr::stump(5, 0.5, -1.0, 1.0);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_n_features_compressed() {
        let model = TinyModelRepr::compressed(DataCompression::None, vec![1, 2, 3], 100);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_n_parameters_compressed() {
        let model = TinyModelRepr::compressed(DataCompression::None, vec![0; 40], 160);
        // original_size / 4 = 160 / 4 = 40
        assert_eq!(model.n_parameters(), 40);
    }

    #[test]
    fn test_naive_bayes_size_bytes() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );

        // 2 priors * 4 + 2 classes * 2 features * 8 (means + variances) = 8 + 32 = 40
        assert_eq!(model.size_bytes(), 40);
    }

    #[test]
    fn test_naive_bayes_empty_means() {
        let model = TinyModelRepr::naive_bayes(vec![1.0], vec![], vec![]);

        // With empty means, n_features should be 0
        assert_eq!(model.n_features(), None);
    }

    #[test]
    fn test_knn_k_zero() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0, 1],
            0, // k = 0
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidK { k: 0, .. })
        ));
    }

    #[test]
    fn test_knn_empty_reference_points() {
        let model = TinyModelRepr::knn(vec![], vec![], 1);

        assert!(matches!(model.validate(), Err(TinyModelError::EmptyModel)));
    }

    #[test]
    fn test_naive_bayes_empty_priors() {
        let model = TinyModelRepr::naive_bayes(vec![], vec![], vec![]);

        assert!(matches!(model.validate(), Err(TinyModelError::EmptyModel)));
    }

    #[test]
    fn test_linear_predict_on_non_linear() {
        let model = TinyModelRepr::stump(0, 0.5, -1.0, 1.0);
        assert!(model.predict_linear(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_stump_predict_on_non_stump() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.0);
        assert!(model.predict_stump(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_kmeans_predict_on_non_kmeans() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.0);
        assert!(model.predict_kmeans(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_stump_validate_always_ok() {
        let model = TinyModelRepr::stump(0, 0.5, -1.0, 1.0);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_logistic_regression_validate_always_ok() {
        let model = TinyModelRepr::logistic_regression(vec![vec![0.1, 0.2]], vec![0.3]);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_compressed_validate_always_ok() {
        let model = TinyModelRepr::compressed(DataCompression::zstd(), vec![1, 2, 3], 10);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_partial_eq() {
        let model1 = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let model2 = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let model3 = TinyModelRepr::linear(vec![1.0, 2.0], 0.6);

        assert_eq!(model1, model2);
        assert_ne!(model1, model3);
    }

    #[test]
    fn test_logistic_regression_n_parameters() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            vec![0.7, 0.8],
        );

        // 2 classes * 3 features + 2 intercepts = 6 + 2 = 8
        assert_eq!(model.n_parameters(), 8);
    }

    #[test]
    fn test_knn_n_parameters() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        // 3 points * 2 features = 6
        assert_eq!(model.n_parameters(), 6);
    }

    #[test]
    fn test_error_source() {
        let err = TinyModelError::EmptyModel;
        // Test that std::error::Error is implemented
        let _source = std::error::Error::source(&err);
    }

    #[test]
    fn test_tiny_model_clone() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let cloned = model.clone();
        assert_eq!(model, cloned);
    }

    #[test]
    fn test_tiny_model_debug() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("Linear"));
        assert!(debug_str.contains("coefficients"));
    }

    #[test]
    fn test_naive_bayes_n_parameters() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.3, 0.7],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );

        // 2 priors + 2*2 means + 2*2 variances = 2 + 4 + 4 = 10
        assert_eq!(model.n_parameters(), 10);
    }

    #[test]
    fn test_kmeans_n_features_empty() {
        let model = TinyModelRepr::kmeans(vec![]);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_logistic_regression_n_features_empty() {
        let model = TinyModelRepr::logistic_regression(vec![], vec![]);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_knn_n_features_empty() {
        let model = TinyModelRepr::knn(vec![], vec![], 1);
        assert!(model.n_features().is_none());
    }
}
