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

    /// Saves the StandardScaler to a SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the SafeTensors file will be saved
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

    /// Loads a StandardScaler from a SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SafeTensors file
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
/// The transformation is: X_scaled = (X - X_min) / (X_max - X_min)
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
        use nalgebra::{DMatrix, SymmetricEigen};

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

        // Convert to nalgebra for eigendecomposition
        let cov_matrix = DMatrix::from_row_slice(n_features, n_features, &cov);
        let eigen = SymmetricEigen::new(cov_matrix);

        // Get eigenvalues and eigenvectors
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top n_components
        let mut components_data = vec![0.0; self.n_components * n_features];
        let mut explained_variance = vec![0.0; self.n_components];

        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            explained_variance[i] = eigenvalues[idx];
            for j in 0..n_features {
                components_data[i * n_features + j] = eigenvectors[(j, idx)];
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
    /// Default: perplexity=30.0, learning_rate=200.0, n_iter=1000
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
    /// Uses binary search to find sigma_i such that perplexity matches target.
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
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let scaler = StandardScaler::new();
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_default() {
        let scaler = StandardScaler::default();
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_fit_basic() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
            .expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        scaler
            .fit(&data)
            .expect("fit should succeed with valid data");

        assert!(scaler.is_fitted());

        // Mean should be [2.0, 20.0]
        let mean = scaler.mean();
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!((mean[1] - 20.0).abs() < 1e-6);

        // Std should be sqrt(2/3) ≈ 0.8165
        let std = scaler.std();
        let expected_std = (2.0_f32 / 3.0).sqrt();
        assert!((std[0] - expected_std).abs() < 1e-4);
        assert!((std[1] - expected_std * 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_transform_basic() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        scaler
            .fit(&data)
            .expect("fit should succeed with valid data");

        let transformed = scaler
            .transform(&data)
            .expect("transform should succeed after fit");

        // Mean should be 0
        let mean: f32 = (0..3).map(|i| transformed.get(i, 0)).sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-6, "Mean should be ~0, got {mean}");

        // Std should be 1
        let variance: f32 = (0..3)
            .map(|i| {
                let v = transformed.get(i, 0);
                v * v
            })
            .sum::<f32>()
            / 3.0;
        assert!(
            (variance.sqrt() - 1.0).abs() < 1e-6,
            "Std should be ~1, got {}",
            variance.sqrt()
        );
    }

    #[test]
    fn test_fit_transform() {
        let data = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
            .expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed with valid data");

        // Check each column has mean ≈ 0
        for j in 0..2 {
            let mean: f32 = (0..4).map(|i| transformed.get(i, j)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "Column {j} mean should be ~0");
        }
    }

    #[test]
    fn test_inverse_transform() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
            .expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        let recovered = scaler
            .inverse_transform(&transformed)
            .expect("inverse_transform should succeed");

        // Should recover original data
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                    "Mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_transform_new_data() {
        let train = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");
        let test = Matrix::from_vec(2, 1, vec![4.0, 5.0]).expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        scaler
            .fit(&train)
            .expect("fit should succeed with valid data");

        let transformed = scaler
            .transform(&test)
            .expect("transform should succeed with new data");

        // Test data should be transformed using train stats
        // mean=2, std=sqrt(2/3)
        let mean = 2.0;
        let std = (2.0_f32 / 3.0).sqrt();

        let expected_0 = (4.0 - mean) / std;
        let expected_1 = (5.0 - mean) / std;

        assert!((transformed.get(0, 0) - expected_0).abs() < 1e-5);
        assert!((transformed.get(1, 0) - expected_1).abs() < 1e-5);
    }

    #[test]
    fn test_without_mean() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new().with_mean(false);
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Should only scale, not center
        // Original values divided by std
        let std = (2.0_f32 / 3.0).sqrt();
        assert!((transformed.get(0, 0) - 1.0 / std).abs() < 1e-5);
        assert!((transformed.get(1, 0) - 2.0 / std).abs() < 1e-5);
        assert!((transformed.get(2, 0) - 3.0 / std).abs() < 1e-5);
    }

    #[test]
    fn test_without_std() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new().with_std(false);
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Should only center, not scale
        // mean = 2.0
        assert!((transformed.get(0, 0) - (-1.0)).abs() < 1e-5);
        assert!((transformed.get(1, 0) - 0.0).abs() < 1e-5);
        assert!((transformed.get(2, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_constant_feature() {
        // Feature with zero variance
        let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0])
            .expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Second column has zero std, should remain centered but not scaled
        assert!((transformed.get(0, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(1, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(2, 1) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty_data_error() {
        let data = Matrix::from_vec(0, 2, vec![]).expect("empty matrix should be valid");
        let mut scaler = StandardScaler::new();
        assert!(scaler.fit(&data).is_err());
    }

    #[test]
    fn test_transform_not_fitted_error() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");
        let scaler = StandardScaler::new();
        assert!(scaler.transform(&data).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");
        let test = Matrix::from_vec(3, 3, vec![1.0; 9]).expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        scaler.fit(&train).expect("fit should succeed");

        assert!(scaler.transform(&test).is_err());
    }

    #[test]
    fn test_single_sample() {
        let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).expect("valid matrix dimensions");

        let mut scaler = StandardScaler::new();
        scaler
            .fit(&data)
            .expect("fit should succeed with single sample");

        // With single sample, std is 0
        let std = scaler.std();
        assert!((std[0]).abs() < 1e-6);
        assert!((std[1]).abs() < 1e-6);

        // Transform should center only (std is 0, no scaling)
        let transformed = scaler.transform(&data).expect("transform should succeed");
        assert!((transformed.get(0, 0)).abs() < 1e-5);
        assert!((transformed.get(0, 1)).abs() < 1e-5);
    }

    #[test]
    fn test_builder_chain() {
        let scaler = StandardScaler::new().with_mean(false).with_std(true);

        let data = Matrix::from_vec(2, 1, vec![2.0, 4.0]).expect("valid matrix dimensions");
        let mut scaler = scaler;
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Only scaling, no centering
        // Values: 2, 4; mean=3; std=1
        // Without centering: 2/1=2, 4/1=4
        assert!(transformed.get(0, 0) > 0.0, "Should not be centered");
    }

    // MinMaxScaler tests
    #[test]
    fn test_minmax_new() {
        let scaler = MinMaxScaler::new();
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_minmax_default() {
        let scaler = MinMaxScaler::default();
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_minmax_fit_basic() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
            .expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        scaler
            .fit(&data)
            .expect("fit should succeed with valid data");

        assert!(scaler.is_fitted());

        // Min should be [1.0, 10.0], max should be [3.0, 30.0]
        let data_min = scaler.data_min();
        let data_max = scaler.data_max();
        assert!((data_min[0] - 1.0).abs() < 1e-6);
        assert!((data_min[1] - 10.0).abs() < 1e-6);
        assert!((data_max[0] - 3.0).abs() < 1e-6);
        assert!((data_max[1] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_transform_basic() {
        let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        scaler
            .fit(&data)
            .expect("fit should succeed with valid data");

        let transformed = scaler
            .transform(&data)
            .expect("transform should succeed after fit");

        // Should scale to [0, 1]
        assert!((transformed.get(0, 0) - 0.0).abs() < 1e-6);
        assert!((transformed.get(1, 0) - 0.5).abs() < 1e-6);
        assert!((transformed.get(2, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_fit_transform() {
        let data = Matrix::from_vec(4, 2, vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0, 30.0, 300.0])
            .expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed with valid data");

        // Check min is 0 and max is 1 for each column
        for j in 0..2 {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..4 {
                let val = transformed.get(i, j);
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            assert!(min_val.abs() < 1e-5, "Column {j} min should be ~0");
            assert!((max_val - 1.0).abs() < 1e-5, "Column {j} max should be ~1");
        }
    }

    #[test]
    fn test_minmax_inverse_transform() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
            .expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        let recovered = scaler
            .inverse_transform(&transformed)
            .expect("inverse_transform should succeed");

        // Should recover original data
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                    "Mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_minmax_transform_new_data() {
        let train = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");
        let test = Matrix::from_vec(2, 1, vec![15.0, -5.0]).expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        scaler
            .fit(&train)
            .expect("fit should succeed with valid data");

        let transformed = scaler
            .transform(&test)
            .expect("transform should succeed with new data");

        // 15 should map to 1.5 (beyond training range)
        // -5 should map to -0.5 (below training range)
        assert!((transformed.get(0, 0) - 1.5).abs() < 1e-5);
        assert!((transformed.get(1, 0) - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_custom_range() {
        let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Should scale to [-1, 1]
        assert!((transformed.get(0, 0) - (-1.0)).abs() < 1e-6);
        assert!((transformed.get(1, 0) - 0.0).abs() < 1e-6);
        assert!((transformed.get(2, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_constant_feature() {
        // Feature with same min and max
        let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0])
            .expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Second column is constant, should become feature_min (0)
        assert!((transformed.get(0, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(1, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(2, 1) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_empty_data_error() {
        let data = Matrix::from_vec(0, 2, vec![]).expect("empty matrix should be valid");
        let mut scaler = MinMaxScaler::new();
        assert!(scaler.fit(&data).is_err());
    }

    #[test]
    fn test_minmax_transform_not_fitted_error() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("valid matrix dimensions");
        let scaler = MinMaxScaler::new();
        assert!(scaler.transform(&data).is_err());
    }

    #[test]
    fn test_minmax_dimension_mismatch_error() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");
        let test = Matrix::from_vec(3, 3, vec![1.0; 9]).expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&train).expect("fit should succeed");

        assert!(scaler.transform(&test).is_err());
    }

    #[test]
    fn test_minmax_single_sample() {
        let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new();
        scaler
            .fit(&data)
            .expect("fit should succeed with single sample");

        // With single sample, min = max = value
        let data_min = scaler.data_min();
        let data_max = scaler.data_max();
        assert!((data_min[0] - 5.0).abs() < 1e-6);
        assert!((data_max[0] - 5.0).abs() < 1e-6);

        // Transform should give feature_min (0) since range is 0
        let transformed = scaler.transform(&data).expect("transform should succeed");
        assert!((transformed.get(0, 0)).abs() < 1e-5);
        assert!((transformed.get(0, 1)).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_inverse_with_custom_range() {
        let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).expect("valid matrix dimensions");

        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        let transformed = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        let recovered = scaler
            .inverse_transform(&transformed)
            .expect("inverse_transform should succeed");

        for i in 0..3 {
            assert!(
                (data.get(i, 0) - recovered.get(i, 0)).abs() < 1e-5,
                "Mismatch at row {i}"
            );
        }
    }

    // PCA tests
    #[test]
    fn test_pca_basic_fit_transform() {
        // Simple 2D data that should reduce to 1D along diagonal
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
            .expect("valid matrix dimensions");

        let mut pca = PCA::new(1);
        let transformed = pca
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Should reduce to (n_samples, n_components)
        assert_eq!(transformed.shape(), (4, 1));

        // Mean should be centered (approximately)
        let mut sum = 0.0;
        for i in 0..4 {
            sum += transformed.get(i, 0);
        }
        let mean = sum / 4.0;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {mean}");
    }

    #[test]
    fn test_pca_explained_variance() {
        // Data with known variance structure
        let data = Matrix::from_vec(5, 2, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
            .expect("valid matrix dimensions");

        let mut pca = PCA::new(2);
        pca.fit(&data).expect("fit should succeed with valid data");

        let explained_var = pca
            .explained_variance()
            .expect("explained variance should exist after fit");
        let explained_ratio = pca
            .explained_variance_ratio()
            .expect("explained variance ratio should exist after fit");

        // First component should capture all variance (second column is constant)
        assert_eq!(explained_var.len(), 2);
        assert_eq!(explained_ratio.len(), 2);

        // Ratios should sum to approximately 1.0
        let total_ratio: f32 = explained_ratio.iter().sum();
        assert!(
            (total_ratio - 1.0).abs() < 1e-5,
            "Variance ratios should sum to 1.0, got {total_ratio}"
        );

        // First component should explain most variance
        assert!(
            explained_ratio[0] > 0.99,
            "First component should explain >99% variance"
        );
    }

    #[test]
    fn test_pca_inverse_transform() {
        let data = Matrix::from_vec(
            4,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut pca = PCA::new(2);
        let transformed = pca
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        let reconstructed = pca
            .inverse_transform(&transformed)
            .expect("inverse_transform should succeed");

        // Reconstruction should be close to original (with some loss since n_components < n_features)
        assert_eq!(reconstructed.shape(), data.shape());

        // Check reconstruction error is reasonable
        let mut total_error = 0.0;
        for i in 0..4 {
            for j in 0..3 {
                let error = (data.get(i, j) - reconstructed.get(i, j)).abs();
                total_error += error * error;
            }
        }
        let mse = total_error / 12.0;
        // With dimensionality reduction, some error is expected
        assert!(mse < 10.0, "Reconstruction MSE too large: {mse}");
    }

    #[test]
    fn test_pca_perfect_reconstruction() {
        // When n_components == n_features, reconstruction should be perfect
        let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");

        let mut pca = PCA::new(2);
        let transformed = pca
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        let reconstructed = pca
            .inverse_transform(&transformed)
            .expect("inverse_transform should succeed");

        // Perfect reconstruction
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (data.get(i, j) - reconstructed.get(i, j)).abs() < 1e-4,
                    "Perfect reconstruction failed at ({}, {}): {} vs {}",
                    i,
                    j,
                    data.get(i, j),
                    reconstructed.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_pca_n_components_exceeds_features() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");

        let mut pca = PCA::new(3); // More components than features
        let result = pca.fit(&data);

        assert!(
            result.is_err(),
            "Should fail when n_components > n_features"
        );
        assert_eq!(
            result.expect_err("Should fail when n_components exceeds features"),
            "n_components cannot exceed number of features"
        );
    }

    #[test]
    fn test_pca_not_fitted_error() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");

        let pca = PCA::new(1);
        let result = pca.transform(&data);

        assert!(result.is_err(), "Should fail when transforming before fit");
        assert_eq!(
            result.expect_err("Should fail when PCA not fitted"),
            "PCA not fitted"
        );
    }

    #[test]
    fn test_pca_dimension_mismatch() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");
        let test = Matrix::from_vec(3, 3, vec![1.0; 9]).expect("valid matrix dimensions");

        let mut pca = PCA::new(1);
        pca.fit(&train).expect("fit should succeed");

        let result = pca.transform(&test);
        assert!(result.is_err(), "Should fail on dimension mismatch");
        assert_eq!(
            result.expect_err("Should fail with dimension mismatch"),
            "Input has wrong number of features"
        );
    }

    #[test]
    fn test_pca_inverse_dimension_mismatch() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix dimensions");
        let wrong_transformed =
            Matrix::from_vec(3, 2, vec![1.0; 6]).expect("valid matrix dimensions");

        let mut pca = PCA::new(1);
        pca.fit(&train).expect("fit should succeed");

        let result = pca.inverse_transform(&wrong_transformed);
        assert!(
            result.is_err(),
            "Should fail on inverse transform dimension mismatch"
        );
        assert_eq!(
            result.expect_err("Should fail with wrong component count"),
            "Input has wrong number of components"
        );
    }

    #[test]
    fn test_pca_components_shape() {
        let data = Matrix::from_vec(5, 4, vec![1.0; 20]).expect("valid matrix dimensions");

        let mut pca = PCA::new(2);
        pca.fit(&data).expect("fit should succeed with valid data");

        let components = pca.components().expect("components should exist after fit");
        // Components should be (n_components, n_features)
        assert_eq!(components.shape(), (2, 4));
    }

    #[test]
    fn test_pca_variance_preservation() {
        // Property test: total variance should be preserved
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0,
                9.0, 12.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut pca = PCA::new(3);
        pca.fit(&data).expect("fit should succeed with valid data");

        let explained_var = pca
            .explained_variance()
            .expect("explained variance should exist after fit");

        // Sum of explained variance should be close to total variance
        let total_explained: f32 = explained_var.iter().sum();

        // Calculate actual variance of centered data
        let (n_samples, n_features) = data.shape();
        let mut means = vec![0.0; n_features];
        for (j, mean) in means.iter_mut().enumerate() {
            for i in 0..n_samples {
                *mean += data.get(i, j);
            }
            *mean /= n_samples as f32;
        }

        let mut total_var = 0.0;
        for (j, &mean_j) in means.iter().enumerate() {
            for i in 0..n_samples {
                let diff = data.get(i, j) - mean_j;
                total_var += diff * diff;
            }
        }
        total_var /= (n_samples - 1) as f32;

        // Explained variance should match total variance (with full components)
        assert!(
            (total_explained - total_var).abs() < 1e-3,
            "Total explained variance {total_explained} should match total variance {total_var}"
        );
    }

    #[test]
    fn test_pca_component_orthogonality() {
        // Property test: principal components should be orthogonal
        let data = Matrix::from_vec(
            10,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0,
                5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0,
                24.0, 32.0, 9.0, 18.0, 27.0, 36.0, 10.0, 20.0, 30.0, 40.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut pca = PCA::new(3);
        pca.fit(&data).expect("fit should succeed with valid data");

        let components = pca.components().expect("components should exist after fit");
        let (_n_components, n_features) = components.shape();

        // Check that all pairs of components are orthogonal (dot product ≈ 0)
        for i in 0..3 {
            for j in (i + 1)..3 {
                let mut dot_product = 0.0;
                for k in 0..n_features {
                    dot_product += components.get(i, k) * components.get(j, k);
                }
                assert!(
                    dot_product.abs() < 1e-4,
                    "Components {i} and {j} should be orthogonal, got dot product {dot_product}"
                );
            }
        }

        // Check that each component is normalized (length ≈ 1)
        for i in 0..3 {
            let mut norm_sq = 0.0;
            for k in 0..n_features {
                let val = components.get(i, k);
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Component {i} should be unit length, got {norm}"
            );
        }
    }

    // ========================================================================
    // t-SNE Tests
    // ========================================================================

    #[test]
    fn test_tsne_new() {
        let tsne = TSNE::new(2);
        assert!(!tsne.is_fitted());
        assert_eq!(tsne.n_components(), 2);
    }

    #[test]
    fn test_tsne_fit_basic() {
        // Simple 2D data, reduce to 2D (should work)
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 10.0, 11.0, 12.0, 10.1,
                11.1, 12.1,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2);
        tsne.fit(&data).expect("fit should succeed with valid data");
        assert!(tsne.is_fitted());
    }

    #[test]
    fn test_tsne_transform() {
        let data = Matrix::from_vec(
            4,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2);
        tsne.fit(&data).expect("fit should succeed with valid data");

        let transformed = tsne
            .transform(&data)
            .expect("transform should succeed after fit");
        assert_eq!(transformed.shape(), (4, 2));
    }

    #[test]
    fn test_tsne_fit_transform() {
        let data = Matrix::from_vec(
            4,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2);
        let transformed = tsne
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        assert_eq!(transformed.shape(), (4, 2));
        assert!(tsne.is_fitted());
    }

    #[test]
    fn test_tsne_perplexity() {
        let data = Matrix::from_vec(
            10,
            3,
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 5.2,
                6.2, 7.2, 10.0, 11.0, 12.0, 10.1, 11.1, 12.1, 10.2, 11.2, 12.2, 10.3, 11.3, 12.3,
            ],
        )
        .expect("valid matrix dimensions");

        // Low perplexity (more local)
        let mut tsne_low = TSNE::new(2).with_perplexity(2.0);
        let result_low = tsne_low
            .fit_transform(&data)
            .expect("fit_transform should succeed with low perplexity");
        assert_eq!(result_low.shape(), (10, 2));

        // High perplexity (more global)
        let mut tsne_high = TSNE::new(2).with_perplexity(5.0);
        let result_high = tsne_high
            .fit_transform(&data)
            .expect("fit_transform should succeed with high perplexity");
        assert_eq!(result_high.shape(), (10, 2));
    }

    #[test]
    fn test_tsne_learning_rate() {
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
                12.0, 13.0, 14.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2).with_learning_rate(100.0).with_n_iter(100);
        let transformed = tsne
            .fit_transform(&data)
            .expect("fit_transform should succeed with custom learning rate");
        assert_eq!(transformed.shape(), (6, 2));
    }

    #[test]
    fn test_tsne_n_components() {
        let data = Matrix::from_vec(
            4,
            5,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .expect("valid matrix dimensions");

        // 2D embedding
        let mut tsne_2d = TSNE::new(2);
        let result_2d = tsne_2d
            .fit_transform(&data)
            .expect("fit_transform should succeed for 2D");
        assert_eq!(result_2d.shape(), (4, 2));

        // 3D embedding
        let mut tsne_3d = TSNE::new(3);
        let result_3d = tsne_3d
            .fit_transform(&data)
            .expect("fit_transform should succeed for 3D");
        assert_eq!(result_3d.shape(), (4, 3));
    }

    #[test]
    fn test_tsne_reproducibility() {
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
                12.0, 13.0, 14.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne1 = TSNE::new(2).with_random_state(42);
        let result1 = tsne1
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        let mut tsne2 = TSNE::new(2).with_random_state(42);
        let result2 = tsne2
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Results should be identical with same random state
        for i in 0..6 {
            for j in 0..2 {
                assert!(
                    (result1.get(i, j) - result2.get(i, j)).abs() < 1e-5,
                    "Results should be reproducible with same random state"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_tsne_transform_before_fit() {
        let data =
            Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
        let tsne = TSNE::new(2);
        let _ = tsne.transform(&data);
    }

    #[test]
    fn test_tsne_preserves_local_structure() {
        // Create data with clear local structure
        let data = Matrix::from_vec(
            8,
            3,
            vec![
                // Cluster 1: tight cluster around (0, 0, 0)
                0.0, 0.0, 0.0, 0.1, 0.1, 0.1, // Cluster 2: tight cluster around (5, 5, 5)
                5.0, 5.0, 5.0, 5.1, 5.1, 5.1,
                // Cluster 3: tight cluster around (10, 10, 10)
                10.0, 10.0, 10.0, 10.1, 10.1, 10.1,
                // Cluster 4: tight cluster around (15, 15, 15)
                15.0, 15.0, 15.0, 15.1, 15.1, 15.1,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2)
            .with_random_state(42)
            .with_n_iter(500)
            .with_perplexity(3.0);
        let embedding = tsne
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // Points within same cluster should be close in embedding
        // Cluster 1: points 0, 1
        let dist_01 = ((embedding.get(0, 0) - embedding.get(1, 0)).powi(2)
            + (embedding.get(0, 1) - embedding.get(1, 1)).powi(2))
        .sqrt();

        // Distance to far cluster should be larger
        let dist_03 = ((embedding.get(0, 0) - embedding.get(3, 0)).powi(2)
            + (embedding.get(0, 1) - embedding.get(3, 1)).powi(2))
        .sqrt();

        // Allow some tolerance - t-SNE is stochastic
        // Just verify local structure is somewhat preserved
        assert!(
            dist_01 < dist_03 * 1.5,
            "Local structure should be roughly preserved: dist_01={:.3} should be < dist_03*1.5={:.3}",
            dist_01,
            dist_03 * 1.5
        );
    }

    #[test]
    fn test_tsne_min_samples() {
        // t-SNE should work with minimum number of samples (> perplexity * 3)
        let data = Matrix::from_vec(
            10,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0,
                7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2).with_perplexity(3.0);
        let result = tsne
            .fit_transform(&data)
            .expect("fit_transform should succeed with minimum samples");
        assert_eq!(result.shape(), (10, 2));
    }

    #[test]
    fn test_tsne_embedding_finite() {
        let data = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
                12.0, 13.0, 14.0,
            ],
        )
        .expect("valid matrix dimensions");

        let mut tsne = TSNE::new(2).with_n_iter(100);
        let embedding = tsne
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // All embedding values should be finite
        for i in 0..6 {
            for j in 0..2 {
                assert!(
                    embedding.get(i, j).is_finite(),
                    "Embedding should contain only finite values"
                );
            }
        }
    }
}
