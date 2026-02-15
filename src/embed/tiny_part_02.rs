
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
    /// * `means` - Mean values `[n_classes][n_features]`
    /// * `variances` - Variance values `[n_classes][n_features]`
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
    /// * `centroids` - Cluster centroids `[n_clusters][n_features]`
    #[must_use]
    pub fn kmeans(centroids: Vec<Vec<f32>>) -> Self {
        Self::KMeans { centroids }
    }

    /// Create a logistic regression representation
    ///
    /// # Arguments
    /// * `coefficients` - Coefficients `[n_classes][n_features]`
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
