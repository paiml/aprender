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
//! ]).unwrap();
//!
//! // Standardize to zero mean and unit variance
//! let mut scaler = StandardScaler::new();
//! let scaled = scaler.fit_transform(&data).unwrap();
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
/// ]).unwrap();
///
/// let mut scaler = StandardScaler::new();
/// let scaled = scaler.fit_transform(&data).unwrap();
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

        Matrix::from_vec(n_samples, n_features, result).map_err(|e| e.into())
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

        safetensors::save_safetensors(path, tensors)?;
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

        Matrix::from_vec(n_samples, n_features, result).map_err(|e| e.into())
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
/// ]).unwrap();
///
/// let mut scaler = MinMaxScaler::new();
/// let scaled = scaler.fit_transform(&data).unwrap();
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

        Matrix::from_vec(n_samples, n_features, result).map_err(|e| e.into())
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

        Matrix::from_vec(n_samples, n_features, result).map_err(|e| e.into())
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
/// ]).unwrap();
///
/// let mut pca = PCA::new(2); // Reduce to 2 components
/// let transformed = pca.fit_transform(&data).unwrap();
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

        Matrix::from_vec(n_samples, n_features, result).map_err(|e| e.into())
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
            explained_variance[i] = eigenvalues[idx] as f32;
            for j in 0..n_features {
                components_data[i * n_features + j] = eigenvectors[(j, idx)] as f32;
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

        Matrix::from_vec(n_samples, self.n_components, result).map_err(|e| e.into())
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
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data).unwrap();

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
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data).unwrap();

        let transformed = scaler.transform(&data).unwrap();

        // Mean should be 0
        let mean: f32 = (0..3).map(|i| transformed.get(i, 0)).sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-6, "Mean should be ~0, got {}", mean);

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
        let data =
            Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data).unwrap();

        // Check each column has mean ≈ 0
        for j in 0..2 {
            let mean: f32 = (0..4).map(|i| transformed.get(i, j)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "Column {} mean should be ~0", j);
        }
    }

    #[test]
    fn test_inverse_transform() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data).unwrap();
        let recovered = scaler.inverse_transform(&transformed).unwrap();

        // Should recover original data
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_transform_new_data() {
        let train = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let test = Matrix::from_vec(2, 1, vec![4.0, 5.0]).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&train).unwrap();

        let transformed = scaler.transform(&test).unwrap();

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
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();

        let mut scaler = StandardScaler::new().with_mean(false);
        let transformed = scaler.fit_transform(&data).unwrap();

        // Should only scale, not center
        // Original values divided by std
        let std = (2.0_f32 / 3.0).sqrt();
        assert!((transformed.get(0, 0) - 1.0 / std).abs() < 1e-5);
        assert!((transformed.get(1, 0) - 2.0 / std).abs() < 1e-5);
        assert!((transformed.get(2, 0) - 3.0 / std).abs() < 1e-5);
    }

    #[test]
    fn test_without_std() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();

        let mut scaler = StandardScaler::new().with_std(false);
        let transformed = scaler.fit_transform(&data).unwrap();

        // Should only center, not scale
        // mean = 2.0
        assert!((transformed.get(0, 0) - (-1.0)).abs() < 1e-5);
        assert!((transformed.get(1, 0) - 0.0).abs() < 1e-5);
        assert!((transformed.get(2, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_constant_feature() {
        // Feature with zero variance
        let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data).unwrap();

        // Second column has zero std, should remain centered but not scaled
        assert!((transformed.get(0, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(1, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(2, 1) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty_data_error() {
        let data = Matrix::from_vec(0, 2, vec![]).unwrap();
        let mut scaler = StandardScaler::new();
        assert!(scaler.fit(&data).is_err());
    }

    #[test]
    fn test_transform_not_fitted_error() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let scaler = StandardScaler::new();
        assert!(scaler.transform(&data).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let test = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&train).unwrap();

        assert!(scaler.transform(&test).is_err());
    }

    #[test]
    fn test_single_sample() {
        let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data).unwrap();

        // With single sample, std is 0
        let std = scaler.std();
        assert!((std[0]).abs() < 1e-6);
        assert!((std[1]).abs() < 1e-6);

        // Transform should center only (std is 0, no scaling)
        let transformed = scaler.transform(&data).unwrap();
        assert!((transformed.get(0, 0)).abs() < 1e-5);
        assert!((transformed.get(0, 1)).abs() < 1e-5);
    }

    #[test]
    fn test_builder_chain() {
        let scaler = StandardScaler::new().with_mean(false).with_std(true);

        let data = Matrix::from_vec(2, 1, vec![2.0, 4.0]).unwrap();
        let mut scaler = scaler;
        let transformed = scaler.fit_transform(&data).unwrap();

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
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&data).unwrap();

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
        let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&data).unwrap();

        let transformed = scaler.transform(&data).unwrap();

        // Should scale to [0, 1]
        assert!((transformed.get(0, 0) - 0.0).abs() < 1e-6);
        assert!((transformed.get(1, 0) - 0.5).abs() < 1e-6);
        assert!((transformed.get(2, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_fit_transform() {
        let data =
            Matrix::from_vec(4, 2, vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0, 30.0, 300.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&data).unwrap();

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
            assert!(min_val.abs() < 1e-5, "Column {} min should be ~0", j);
            assert!(
                (max_val - 1.0).abs() < 1e-5,
                "Column {} max should be ~1",
                j
            );
        }
    }

    #[test]
    fn test_minmax_inverse_transform() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&data).unwrap();
        let recovered = scaler.inverse_transform(&transformed).unwrap();

        // Should recover original data
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (data.get(i, j) - recovered.get(i, j)).abs() < 1e-5,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_minmax_transform_new_data() {
        let train = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).unwrap();
        let test = Matrix::from_vec(2, 1, vec![15.0, -5.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&train).unwrap();

        let transformed = scaler.transform(&test).unwrap();

        // 15 should map to 1.5 (beyond training range)
        // -5 should map to -0.5 (below training range)
        assert!((transformed.get(0, 0) - 1.5).abs() < 1e-5);
        assert!((transformed.get(1, 0) - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_custom_range() {
        let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).unwrap();

        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        let transformed = scaler.fit_transform(&data).unwrap();

        // Should scale to [-1, 1]
        assert!((transformed.get(0, 0) - (-1.0)).abs() < 1e-6);
        assert!((transformed.get(1, 0) - 0.0).abs() < 1e-6);
        assert!((transformed.get(2, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_constant_feature() {
        // Feature with same min and max
        let data = Matrix::from_vec(3, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&data).unwrap();

        // Second column is constant, should become feature_min (0)
        assert!((transformed.get(0, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(1, 1) - 0.0).abs() < 1e-5);
        assert!((transformed.get(2, 1) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_empty_data_error() {
        let data = Matrix::from_vec(0, 2, vec![]).unwrap();
        let mut scaler = MinMaxScaler::new();
        assert!(scaler.fit(&data).is_err());
    }

    #[test]
    fn test_minmax_transform_not_fitted_error() {
        let data = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let scaler = MinMaxScaler::new();
        assert!(scaler.transform(&data).is_err());
    }

    #[test]
    fn test_minmax_dimension_mismatch_error() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let test = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&train).unwrap();

        assert!(scaler.transform(&test).is_err());
    }

    #[test]
    fn test_minmax_single_sample() {
        let data = Matrix::from_vec(1, 2, vec![5.0, 10.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&data).unwrap();

        // With single sample, min = max = value
        let data_min = scaler.data_min();
        let data_max = scaler.data_max();
        assert!((data_min[0] - 5.0).abs() < 1e-6);
        assert!((data_max[0] - 5.0).abs() < 1e-6);

        // Transform should give feature_min (0) since range is 0
        let transformed = scaler.transform(&data).unwrap();
        assert!((transformed.get(0, 0)).abs() < 1e-5);
        assert!((transformed.get(0, 1)).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_inverse_with_custom_range() {
        let data = Matrix::from_vec(3, 1, vec![0.0, 5.0, 10.0]).unwrap();

        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        let transformed = scaler.fit_transform(&data).unwrap();
        let recovered = scaler.inverse_transform(&transformed).unwrap();

        for i in 0..3 {
            assert!(
                (data.get(i, 0) - recovered.get(i, 0)).abs() < 1e-5,
                "Mismatch at row {}",
                i
            );
        }
    }

    // PCA tests
    #[test]
    fn test_pca_basic_fit_transform() {
        // Simple 2D data that should reduce to 1D along diagonal
        let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let mut pca = PCA::new(1);
        let transformed = pca.fit_transform(&data).unwrap();

        // Should reduce to (n_samples, n_components)
        assert_eq!(transformed.shape(), (4, 1));

        // Mean should be centered (approximately)
        let mut sum = 0.0;
        for i in 0..4 {
            sum += transformed.get(i, 0);
        }
        let mean = sum / 4.0;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_pca_explained_variance() {
        // Data with known variance structure
        let data =
            Matrix::from_vec(5, 2, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]).unwrap();

        let mut pca = PCA::new(2);
        pca.fit(&data).unwrap();

        let explained_var = pca
            .explained_variance()
            .expect("Should have explained variance");
        let explained_ratio = pca
            .explained_variance_ratio()
            .expect("Should have explained variance ratio");

        // First component should capture all variance (second column is constant)
        assert_eq!(explained_var.len(), 2);
        assert_eq!(explained_ratio.len(), 2);

        // Ratios should sum to approximately 1.0
        let total_ratio: f32 = explained_ratio.iter().sum();
        assert!(
            (total_ratio - 1.0).abs() < 1e-5,
            "Variance ratios should sum to 1.0, got {}",
            total_ratio
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
        .unwrap();

        let mut pca = PCA::new(2);
        let transformed = pca.fit_transform(&data).unwrap();
        let reconstructed = pca.inverse_transform(&transformed).unwrap();

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
        assert!(mse < 10.0, "Reconstruction MSE too large: {}", mse);
    }

    #[test]
    fn test_pca_perfect_reconstruction() {
        // When n_components == n_features, reconstruction should be perfect
        let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut pca = PCA::new(2);
        let transformed = pca.fit_transform(&data).unwrap();
        let reconstructed = pca.inverse_transform(&transformed).unwrap();

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
        let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut pca = PCA::new(3); // More components than features
        let result = pca.fit(&data);

        assert!(
            result.is_err(),
            "Should fail when n_components > n_features"
        );
        assert_eq!(
            result.unwrap_err(),
            "n_components cannot exceed number of features"
        );
    }

    #[test]
    fn test_pca_not_fitted_error() {
        let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let pca = PCA::new(1);
        let result = pca.transform(&data);

        assert!(result.is_err(), "Should fail when transforming before fit");
        assert_eq!(result.unwrap_err(), "PCA not fitted");
    }

    #[test]
    fn test_pca_dimension_mismatch() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let test = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();

        let mut pca = PCA::new(1);
        pca.fit(&train).unwrap();

        let result = pca.transform(&test);
        assert!(result.is_err(), "Should fail on dimension mismatch");
        assert_eq!(result.unwrap_err(), "Input has wrong number of features");
    }

    #[test]
    fn test_pca_inverse_dimension_mismatch() {
        let train = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let wrong_transformed = Matrix::from_vec(3, 2, vec![1.0; 6]).unwrap(); // Wrong n_components

        let mut pca = PCA::new(1);
        pca.fit(&train).unwrap();

        let result = pca.inverse_transform(&wrong_transformed);
        assert!(
            result.is_err(),
            "Should fail on inverse transform dimension mismatch"
        );
        assert_eq!(result.unwrap_err(), "Input has wrong number of components");
    }

    #[test]
    fn test_pca_components_shape() {
        let data = Matrix::from_vec(5, 4, vec![1.0; 20]).unwrap();

        let mut pca = PCA::new(2);
        pca.fit(&data).unwrap();

        let components = pca.components().expect("Should have components");
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
        .unwrap();

        let mut pca = PCA::new(3);
        pca.fit(&data).unwrap();

        let explained_var = pca.explained_variance().unwrap();

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
            "Total explained variance {} should match total variance {}",
            total_explained,
            total_var
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
        .unwrap();

        let mut pca = PCA::new(3);
        pca.fit(&data).unwrap();

        let components = pca.components().unwrap();
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
                    "Components {} and {} should be orthogonal, got dot product {}",
                    i,
                    j,
                    dot_product
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
                "Component {} should be unit length, got {}",
                i,
                norm
            );
        }
    }
}
