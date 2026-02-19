
impl MinMaxScaler {
    /// Creates a new `MinMaxScaler` with default range [0, 1].
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_min: None,
            data_max: None,
            feature_min: 0.0,
            feature_max: 1.0,
        }
    }

    /// Sets the target range for scaling.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::preprocessing::MinMaxScaler;
    ///
    /// let scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
    /// ```
    #[must_use]
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.feature_min = min;
        self.feature_max = max;
        self
    }

    /// Returns the minimum value of each feature.
    ///
    /// # Panics
    ///
    /// Panics if the scaler is not fitted.
    #[must_use]
    pub fn data_min(&self) -> &[f32] {
        self.data_min
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns the maximum value of each feature.
    ///
    /// # Panics
    ///
    /// Panics if the scaler is not fitted.
    #[must_use]
    pub fn data_max(&self) -> &[f32] {
        self.data_max
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns true if the scaler has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.data_min.is_some()
    }

    /// Transforms data back to original scale.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaler is not fitted or dimensions mismatch.
    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let data_min = self
            .data_min
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let data_max = self
            .data_max
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != data_min.len() {
            return Err("Feature dimension mismatch".into());
        }

        let feature_range = self.feature_max - self.feature_min;
        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x.get(i, j);
                let data_range = data_max[j] - data_min[j];

                let original = if data_range.abs() > 1e-10 {
                    (val - self.feature_min) / feature_range * data_range + data_min[j]
                } else {
                    data_min[j]
                };

                result[i * n_features + j] = original;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

// Contract: preprocessing-normalization-v1, equation = "minmax_scaler"
impl Transformer for MinMaxScaler {
    /// Computes the min and max of each feature.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        let mut data_min = vec![f32::INFINITY; n_features];
        let mut data_max = vec![f32::NEG_INFINITY; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x.get(i, j);
                if val < data_min[j] {
                    data_min[j] = val;
                }
                if val > data_max[j] {
                    data_max[j] = val;
                }
            }
        }

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);

        Ok(())
    }

    /// Scales the data to the target range.
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let data_min = self
            .data_min
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let data_max = self
            .data_max
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != data_min.len() {
            return Err("Feature dimension mismatch".into());
        }

        let feature_range = self.feature_max - self.feature_min;
        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x.get(i, j);
                let data_range = data_max[j] - data_min[j];

                let scaled = if data_range.abs() > 1e-10 {
                    (val - data_min[j]) / data_range * feature_range + self.feature_min
                } else {
                    self.feature_min
                };

                result[i * n_features + j] = scaled;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

/// Principal Component Analysis (PCA) for dimensionality reduction.
///
/// PCA reduces dimensionality by projecting data onto principal components
/// (directions of maximum variance).
///
/// # Example
///
/// ```
/// use aprender::preprocessing::PCA;
/// use aprender::traits::Transformer;
/// use aprender::primitives::Matrix;
///
/// let data = Matrix::from_vec(4, 3, vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
///     10.0, 11.0, 12.0,
/// ]).expect("valid matrix dimensions");
///
/// let mut pca = PCA::new(2); // Reduce to 2 components
/// let transformed = pca.fit_transform(&data).expect("fit_transform should succeed");
/// assert_eq!(transformed.shape(), (4, 2));
/// ```
#[derive(Debug, Clone)]
pub struct PCA {
    /// Number of components to keep.
    n_components: usize,
    /// Mean of each feature (computed during fit).
    mean: Option<Vec<f32>>,
    /// Principal components (eigenvectors).
    components: Option<Matrix<f32>>,
    /// Variance explained by each component.
    explained_variance: Option<Vec<f32>>,
    /// Ratio of variance explained by each component.
    explained_variance_ratio: Option<Vec<f32>>,
}

impl PCA {
    /// Creates a new PCA transformer.
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of principal components to keep
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            mean: None,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
        }
    }

    /// Returns the variance explained by each component.
    #[must_use]
    pub fn explained_variance(&self) -> Option<&[f32]> {
        self.explained_variance.as_deref()
    }

    /// Returns the ratio of variance explained by each component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> Option<&[f32]> {
        self.explained_variance_ratio.as_deref()
    }

    /// Returns the principal components.
    #[must_use]
    pub fn components(&self) -> Option<&Matrix<f32>> {
        self.components.as_ref()
    }

    /// Reconstructs data from principal component space.
    ///
    /// # Errors
    ///
    /// Returns error if PCA is not fitted.
    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| AprenderError::from("PCA not fitted"))?;
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| AprenderError::from("PCA not fitted"))?;

        let (n_samples, n_components) = x.shape();
        let n_features = mean.len();

        if n_components != self.n_components {
            return Err("Input has wrong number of components".into());
        }

        // X_reconstructed = X_pca @ components^T + mean
        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut value = mean[j];
                for k in 0..n_components {
                    value += x.get(i, k) * components.get(k, j);
                }
                result[i * n_features + j] = value;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

impl Transformer for PCA {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        use trueno::SymmetricEigen;

        let (n_samples, n_features) = x.shape();

        if self.n_components > n_features {
            return Err("n_components cannot exceed number of features".into());
        }

        // Compute mean
        let mut mean = vec![0.0; n_features];
        #[allow(clippy::needless_range_loop)]
        for j in 0..n_features {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += x.get(i, j);
            }
            mean[j] = sum / n_samples as f32;
        }

        // Center the data
        let mut centered = vec![0.0; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                centered[i * n_features + j] = x.get(i, j) - mean[j];
            }
        }

        // Compute covariance matrix: Σ = (X^T X) / (n-1)
        let mut cov = vec![0.0; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += centered[k * n_features + i] * centered[k * n_features + j];
                }
                cov[i * n_features + j] = sum / (n_samples - 1) as f32;
            }
        }

        // Convert to trueno Matrix for eigendecomposition
        let cov_matrix = trueno::Matrix::from_vec(n_features, n_features, cov)
            .map_err(|e| format!("Failed to create covariance matrix: {e}"))?;
        let eigen = SymmetricEigen::new(&cov_matrix)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        // trueno returns eigenvalues in descending order (largest first) - perfect for PCA
        let eigenvalues = eigen.eigenvalues();
        let eigenvectors = eigen.eigenvectors();

        // Select top n_components (already sorted descending)
        let mut components_data = vec![0.0; self.n_components * n_features];
        let mut explained_variance = vec![0.0; self.n_components];

        for i in 0..self.n_components {
            explained_variance[i] = eigenvalues[i];
            for j in 0..n_features {
                // trueno eigenvectors: columns are eigenvectors, access with get(row, col)
                components_data[i * n_features + j] = *eigenvectors
                    .get(j, i)
                    .ok_or_else(|| format!("Invalid eigenvector index ({j}, {i})"))?;
            }
        }

        // Compute explained variance ratio
        let total_variance: f32 = eigenvalues.iter().copied().sum();
        let explained_variance_ratio: Vec<f32> = explained_variance
            .iter()
            .map(|&v| v / total_variance)
            .collect();

        self.mean = Some(mean);
        self.components = Some(Matrix::from_vec(
            self.n_components,
            n_features,
            components_data,
        )?);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| AprenderError::from("PCA not fitted"))?;
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| AprenderError::from("PCA not fitted"))?;

        let (n_samples, n_features) = x.shape();

        if n_features != mean.len() {
            return Err("Input has wrong number of features".into());
        }

        // Project onto principal components: X_pca = (X - mean) @ components^T
        let mut result = vec![0.0; n_samples * self.n_components];

        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut value = 0.0;
                #[allow(clippy::needless_range_loop)]
                for k in 0..n_features {
                    value += (x.get(i, k) - mean[k]) * components.get(j, k);
                }
                result[i * self.n_components + j] = value;
            }
        }

        Matrix::from_vec(n_samples, self.n_components, result).map_err(Into::into)
    }
}
