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

// Contract: preprocessing-normalization-v1, equation = "standard_scaler"
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

/// Scales features using statistics robust to outliers.
///
/// Uses the median and interquartile range (IQR = Q3 - Q1) instead of
/// mean and standard deviation, making it robust to outliers.
///
/// The transformation is: `X_scaled` = (X - median) / IQR
///
/// # Examples
///
/// ```
/// use aprender::preprocessing::RobustScaler;
/// use aprender::traits::Transformer;
/// use aprender::primitives::Matrix;
///
/// let data = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 100.0])
///     .expect("valid matrix dimensions");
///
/// let mut scaler = RobustScaler::new();
/// let scaled = scaler.fit_transform(&data).expect("fit_transform should succeed");
/// // Median=3.0, IQR=Q3-Q1. Outlier (100) does not distort scaling.
/// ```
// Contract: preprocessing-normalization-v1, equation = "robust_scaler"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustScaler {
    /// Median of each feature (computed during fit).
    median: Option<Vec<f32>>,
    /// Interquartile range of each feature (computed during fit).
    iqr: Option<Vec<f32>>,
    /// Whether to center the data (subtract median).
    with_centering: bool,
    /// Whether to scale the data (divide by IQR).
    with_scaling: bool,
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}


include!("mod_include_01.rs");
