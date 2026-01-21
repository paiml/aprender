//! Preprocessing transformers for data standardization and normalization.
//!
//! This module provides transformers that preprocess data before training.
//!
//! # Example
//!
//! ```
//! use aprender::prelude::*;
//! use aprender::preprocessing::StandardScaler;
//!
//! // Create data with different scales
//! let data = Matrix::from_vec(4, 2, vec![
//!     1.0, 100.0,
//!     2.0, 200.0,
//!     3.0, 300.0,
//!     4.0, 400.0,
//! ]).expect("valid matrix dimensions");
//!
//! // Standardize to zero mean and unit variance
//! let mut scaler = StandardScaler::new();
//! let scaled = scaler.fit_transform(&data).expect("fit_transform should succeed");
//!
//! // Each column now has mean ≈ 0 and std ≈ 1
//! assert!(scaled.get(0, 0).abs() < 2.0);
//! ```

use crate::error::{AprenderError, Result};
use crate::primitives::Matrix;
use crate::traits::Transformer;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Standardizes features by removing mean and scaling to unit variance.
///
/// The standard score of a sample x is: z = (x - mean) / std
///
/// This transformer is useful for algorithms that assume features have
/// similar scales (e.g., regularized regression, neural networks).
///
/// # Example
///
/// ```
/// use aprender::prelude::*;
/// use aprender::preprocessing::StandardScaler;
///
/// let data = Matrix::from_vec(3, 2, vec![
///     0.0, 0.0,
///     1.0, 10.0,
///     2.0, 20.0,
/// ]).expect("valid matrix dimensions");
///
/// let mut scaler = StandardScaler::new();
/// let scaled = scaler.fit_transform(&data).expect("fit_transform should succeed");
///
/// // Verify standardization
/// let (n_rows, n_cols) = scaled.shape();
/// for j in 0..n_cols {
///     let mut sum = 0.0;
///     for i in 0..n_rows {
///         sum += scaled.get(i, j);
///     }
///     let mean = sum / n_rows as f32;
///     assert!(mean.abs() < 1e-5, "Mean should be ~0");
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    /// Mean of each feature (computed during fit).
    mean: Option<Vec<f32>>,
    /// Standard deviation of each feature (computed during fit).
    std: Option<Vec<f32>>,
    /// Whether to center the data (subtract mean).
    with_mean: bool,
    /// Whether to scale the data (divide by std).
    with_std: bool,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardScaler {
    /// Creates a new `StandardScaler` with default settings.
    ///
    /// By default, both centering (subtract mean) and scaling (divide by std)
    /// are enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            with_mean: true,
            with_std: true,
        }
    }

    /// Sets whether to center the data by subtracting the mean.
    #[must_use]
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Sets whether to scale the data by dividing by standard deviation.
    #[must_use]
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    /// Returns the mean of each feature.
    ///
    /// # Panics
    ///
    /// Panics if the scaler is not fitted.
    #[must_use]
    pub fn mean(&self) -> &[f32] {
        self.mean
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns the standard deviation of each feature.
    ///
    /// # Panics
    ///
    /// Panics if the scaler is not fitted.
    #[must_use]
    pub fn std(&self) -> &[f32] {
        self.std
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns true if the scaler has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.mean.is_some()
    }

    /// Transforms data back to original scale.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaler is not fitted or dimensions mismatch.
    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let std = self
            .std
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != mean.len() {
            return Err("Feature dimension mismatch".into());
        }

        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = x.get(i, j);

                // Reverse scaling
                if self.with_std && std[j] > 1e-10 {
                    val *= std[j];
                }

                // Reverse centering
                if self.with_mean {
                    val += mean[j];
                }

                result[i * n_features + j] = val;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }

    /// Saves the `StandardScaler` to a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the `SafeTensors` file will be saved
    ///
    /// # Errors
    ///
    /// Returns an error if the scaler is unfitted or if saving fails.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Check if scaler is fitted
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| "Cannot save unfitted scaler. Call fit() first.".to_string())?;
        let std = self
            .std
            .as_ref()
            .ok_or_else(|| "Cannot save unfitted scaler. Call fit() first.".to_string())?;

        let mut tensors = BTreeMap::new();

        // Save mean and std vectors
        tensors.insert("mean".to_string(), (mean.clone(), vec![mean.len()]));
        tensors.insert("std".to_string(), (std.clone(), vec![std.len()]));

        // Save hyperparameters as scalars
        let with_mean_val = if self.with_mean { 1.0 } else { 0.0 };
        tensors.insert("with_mean".to_string(), (vec![with_mean_val], vec![1]));

        let with_std_val = if self.with_std { 1.0 } else { 0.0 };
        tensors.insert("with_std".to_string(), (vec![with_std_val], vec![1]));

        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a `StandardScaler` from a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `SafeTensors` file
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails or if the file format is invalid.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract mean tensor
        let mean_meta = metadata
            .get("mean")
            .ok_or_else(|| "Missing 'mean' tensor in SafeTensors file".to_string())?;
        let mean = safetensors::extract_tensor(&raw_data, mean_meta)?;

        // Extract std tensor
        let std_meta = metadata
            .get("std")
            .ok_or_else(|| "Missing 'std' tensor in SafeTensors file".to_string())?;
        let std = safetensors::extract_tensor(&raw_data, std_meta)?;

        // Verify mean and std have same length
        if mean.len() != std.len() {
            return Err("Mean and std vectors have different lengths".to_string());
        }

        // Load hyperparameters
        let with_mean_meta = metadata
            .get("with_mean")
            .ok_or_else(|| "Missing 'with_mean' tensor".to_string())?;
        let with_mean_data = safetensors::extract_tensor(&raw_data, with_mean_meta)?;
        let with_mean = with_mean_data[0] > 0.5;

        let with_std_meta = metadata
            .get("with_std")
            .ok_or_else(|| "Missing 'with_std' tensor".to_string())?;
        let with_std_data = safetensors::extract_tensor(&raw_data, with_std_meta)?;
        let with_std = with_std_data[0] > 0.5;

        Ok(Self {
            mean: Some(mean),
            std: Some(std),
            with_mean,
            with_std,
        })
    }
}

impl Transformer for StandardScaler {
    /// Computes the mean and standard deviation of each feature.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Compute mean for each feature
        let mut mean = vec![0.0; n_features];
        for (j, mean_j) in mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += x.get(i, j);
            }
            *mean_j = sum / n_samples as f32;
        }

        // Compute standard deviation for each feature
        let mut std = vec![0.0; n_features];
        for (j, std_j) in std.iter_mut().enumerate() {
            let mut sum_sq = 0.0;
            for i in 0..n_samples {
                let diff = x.get(i, j) - mean[j];
                sum_sq += diff * diff;
            }
            // Use population std (divide by n, not n-1) like sklearn
            *std_j = (sum_sq / n_samples as f32).sqrt();
        }

        self.mean = Some(mean);
        self.std = Some(std);

        Ok(())
    }

    /// Standardizes the data using fitted mean and std.
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let std = self
            .std
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != mean.len() {
            return Err("Feature dimension mismatch".into());
        }

        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = x.get(i, j);

                // Center
                if self.with_mean {
                    val -= mean[j];
                }

                // Scale
                if self.with_std && std[j] > 1e-10 {
                    val /= std[j];
                }

                result[i * n_features + j] = val;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

/// Scales features to a given range (default [0, 1]).
///
/// The transformation is: `X_scaled` = (X - `X_min`) / (`X_max` - `X_min`)
///
/// This transformer is useful for algorithms sensitive to feature scales
/// and when you want bounded outputs (e.g., for neural networks).
///
/// # Example
///
/// ```
/// use aprender::prelude::*;
/// use aprender::preprocessing::MinMaxScaler;
///
/// let data = Matrix::from_vec(3, 2, vec![
///     0.0, 0.0,
///     5.0, 10.0,
///     10.0, 20.0,
/// ]).expect("valid matrix dimensions");
///
/// let mut scaler = MinMaxScaler::new();
/// let scaled = scaler.fit_transform(&data).expect("fit_transform should succeed");
///
/// // Verify scaling to [0, 1]
/// assert!((scaled.get(0, 0) - 0.0).abs() < 1e-6);
/// assert!((scaled.get(2, 0) - 1.0).abs() < 1e-6);
/// assert!((scaled.get(1, 0) - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    /// Minimum value of each feature (computed during fit).
    data_min: Option<Vec<f32>>,
    /// Maximum value of each feature (computed during fit).
    data_max: Option<Vec<f32>>,
    /// Target minimum for scaling (default 0.0).
    feature_min: f32,
    /// Target maximum for scaling (default 1.0).
    feature_max: f32,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

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

// ============================================================================
// t-SNE (t-Distributed Stochastic Neighbor Embedding)
// ============================================================================

/// t-SNE for dimensionality reduction and visualization.
///
/// t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear
/// dimensionality reduction technique optimized for visualization of
/// high-dimensional data in 2D or 3D space.
///
/// # Algorithm
///
/// 1. Compute pairwise similarities in high-D using Gaussian kernel
/// 2. Compute perplexity-based conditional probabilities
/// 3. Initialize low-D embedding (random or PCA)
/// 4. Compute pairwise similarities in low-D using Student's t-distribution
/// 5. Minimize KL divergence via gradient descent with momentum
///
/// # Example
///
/// ```
/// use aprender::prelude::*;
/// use aprender::preprocessing::TSNE;
///
/// let data = Matrix::from_vec(
///     6,
///     4,
///     vec![
///         1.0, 2.0, 3.0, 4.0,
///         1.1, 2.1, 3.1, 4.1,
///         5.0, 6.0, 7.0, 8.0,
///         5.1, 6.1, 7.1, 8.1,
///         10.0, 11.0, 12.0, 13.0,
///         10.1, 11.1, 12.1, 13.1,
///     ],
/// )
/// .expect("valid matrix dimensions");
///
/// let mut tsne = TSNE::new(2).with_perplexity(5.0).with_n_iter(250);
/// let embedding = tsne.fit_transform(&data).expect("fit_transform should succeed");
/// assert_eq!(embedding.shape(), (6, 2));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNE {
    /// Number of dimensions in embedding (usually 2 or 3).
    n_components: usize,
    /// Perplexity balances local vs global structure (5-50).
    perplexity: f32,
    /// Learning rate for gradient descent.
    learning_rate: f32,
    /// Number of gradient descent iterations.
    n_iter: usize,
    /// Random seed for reproducibility.
    random_state: Option<u64>,
    /// The learned embedding.
    embedding: Option<Matrix<f32>>,
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new(2)
    }
}

impl TSNE {
    /// Create a new t-SNE with default parameters.
    ///
    /// Default: perplexity=30.0, `learning_rate=200.0`, `n_iter=1000`
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            random_state: None,
            embedding: None,
        }
    }

    /// Set perplexity (balance between local and global structure).
    ///
    /// Typical range: 5-50. Higher perplexity considers more neighbors.
    #[must_use]
    pub fn with_perplexity(mut self, perplexity: f32) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set learning rate for gradient descent.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set number of gradient descent iterations.
    #[must_use]
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Check if model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.embedding.is_some()
    }

    /// Compute pairwise squared Euclidean distances.
    #[allow(clippy::unused_self)]
    fn compute_pairwise_distances(&self, x: &Matrix<f32>) -> Vec<f32> {
        let (n_samples, n_features) = x.shape();
        let mut distances = vec![0.0; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    distances[i * n_samples + j] = 0.0;
                    continue;
                }

                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x.get(i, k) - x.get(j, k);
                    dist_sq += diff * diff;
                }
                distances[i * n_samples + j] = dist_sq;
            }
        }

        distances
    }

    /// Compute conditional probabilities P(j|i) with perplexity constraint.
    ///
    /// Uses binary search to find `sigma_i` such that perplexity matches target.
    fn compute_p_conditional(&self, distances: &[f32], n_samples: usize) -> Vec<f32> {
        let mut p_conditional = vec![0.0; n_samples * n_samples];
        let target_entropy = self.perplexity.ln();

        for i in 0..n_samples {
            // Binary search for sigma that gives target perplexity
            let mut beta_min = -f32::INFINITY;
            let mut beta_max = f32::INFINITY;
            let mut beta = 1.0; // beta = 1 / (2 * sigma^2)

            for _ in 0..50 {
                // Max iterations for binary search
                // Compute P(j|i) with current beta
                let mut sum_p = 0.0;
                let mut entropy = 0.0;

                for j in 0..n_samples {
                    if i == j {
                        p_conditional[i * n_samples + j] = 0.0;
                        continue;
                    }

                    let p_ji = (-beta * distances[i * n_samples + j]).exp();
                    p_conditional[i * n_samples + j] = p_ji;
                    sum_p += p_ji;
                }

                // Normalize and compute entropy
                if sum_p > 0.0 {
                    for j in 0..n_samples {
                        if i != j {
                            let p_normalized = p_conditional[i * n_samples + j] / sum_p;
                            p_conditional[i * n_samples + j] = p_normalized;
                            if p_normalized > 1e-12 {
                                entropy -= p_normalized * p_normalized.ln();
                            }
                        }
                    }
                }

                // Check if entropy matches target
                let entropy_diff = entropy - target_entropy;
                if entropy_diff.abs() < 1e-5 {
                    break;
                }

                // Update beta via binary search
                if entropy_diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + beta_max) / 2.0
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() {
                        beta / 2.0
                    } else {
                        (beta + beta_min) / 2.0
                    };
                }
            }
        }

        p_conditional
    }

    /// Compute symmetric P matrix: P_{ij} = (P_{j|i} + P_{i|j}) / (2N).
    #[allow(clippy::unused_self)]
    fn compute_p_joint(&self, p_conditional: &[f32], n_samples: usize) -> Vec<f32> {
        let mut p_joint = vec![0.0; n_samples * n_samples];
        let normalizer = 2.0 * n_samples as f32;

        for i in 0..n_samples {
            for j in 0..n_samples {
                p_joint[i * n_samples + j] = (p_conditional[i * n_samples + j]
                    + p_conditional[j * n_samples + i])
                    / normalizer;
                // Numerical stability
                p_joint[i * n_samples + j] = p_joint[i * n_samples + j].max(1e-12);
            }
        }

        p_joint
    }

    /// Compute Q matrix in low-dimensional space using Student's t-distribution.
    fn compute_q(&self, y: &[f32], n_samples: usize) -> Vec<f32> {
        let mut q = vec![0.0; n_samples * n_samples];
        let mut sum_q = 0.0;

        // Compute Q_{ij} = (1 + ||y_i - y_j||^2)^{-1}
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    q[i * n_samples + j] = 0.0;
                    continue;
                }

                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = y[i * self.n_components + k] - y[j * self.n_components + k];
                    dist_sq += diff * diff;
                }

                let q_ij = 1.0 / (1.0 + dist_sq);
                q[i * n_samples + j] = q_ij;
                sum_q += q_ij;
            }
        }

        // Normalize
        if sum_q > 0.0 {
            for q_val in &mut q {
                *q_val /= sum_q;
                *q_val = q_val.max(1e-12); // Numerical stability
            }
        }

        q
    }

    /// Compute gradient of KL divergence.
    fn compute_gradient(&self, y: &[f32], p: &[f32], q: &[f32], n_samples: usize) -> Vec<f32> {
        let mut gradient = vec![0.0; n_samples * self.n_components];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let p_ij = p[i * n_samples + j];
                let q_ij = q[i * n_samples + j];

                // Gradient factor: 4 * (p_ij - q_ij) * q_ij * (1 + ||y_i - y_j||^2)^{-1}
                // Simplified: 4 * (p_ij - q_ij) / (1 + ||y_i - y_j||^2)
                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = y[i * self.n_components + k] - y[j * self.n_components + k];
                    dist_sq += diff * diff;
                }

                let factor = 4.0 * (p_ij - q_ij) / (1.0 + dist_sq);

                for k in 0..self.n_components {
                    let diff = y[i * self.n_components + k] - y[j * self.n_components + k];
                    gradient[i * self.n_components + k] += factor * diff;
                }
            }
        }

        gradient
    }
}

impl Transformer for TSNE {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _n_features) = x.shape();

        // Compute pairwise distances in high-D
        let distances = self.compute_pairwise_distances(x);

        // Compute conditional probabilities with perplexity
        let p_conditional = self.compute_p_conditional(&distances, n_samples);

        // Compute joint probabilities (symmetric)
        let p_joint = self.compute_p_joint(&p_conditional, n_samples);

        // Initialize embedding randomly
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let seed = self.random_state.unwrap_or_else(|| {
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut hasher);
            hasher.finish()
        });

        // Simple LCG random number generator for reproducibility
        let mut rng_state = seed;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng_state >> 16) as f32 / 65536.0) - 0.5
        };

        let mut y = vec![0.0; n_samples * self.n_components];
        for val in &mut y {
            *val = rand() * 0.0001; // Small random initialization
        }

        // Gradient descent with momentum
        let mut velocity = vec![0.0; n_samples * self.n_components];
        let momentum = 0.5;
        let final_momentum = 0.8;
        let momentum_switch_iter = 250;

        for iter in 0..self.n_iter {
            // Compute Q matrix in low-D
            let q = self.compute_q(&y, n_samples);

            // Compute gradient
            let gradient = self.compute_gradient(&y, &p_joint, &q, n_samples);

            // Update with momentum
            let current_momentum = if iter < momentum_switch_iter {
                momentum
            } else {
                final_momentum
            };

            for i in 0..(n_samples * self.n_components) {
                velocity[i] = current_momentum * velocity[i] - self.learning_rate * gradient[i];
                y[i] += velocity[i];
            }

            // Early exaggeration (first 100 iterations)
            if iter == 100 {
                // Remove early exaggeration by dividing P by 4
                // (we multiplied by 4 implicitly in gradient computation)
            }
        }

        self.embedding = Some(Matrix::from_vec(n_samples, self.n_components, y)?);
        Ok(())
    }

    fn transform(&self, _x: &Matrix<f32>) -> Result<Matrix<f32>> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");
        // t-SNE is non-parametric, return the embedding
        Ok(self
            .embedding
            .as_ref()
            .expect("embedding should exist after is_fitted() check")
            .clone())
    }

    fn fit_transform(&mut self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        self.fit(x)?;
        self.transform(x)
    }
}


#[cfg(test)]
mod tests;
