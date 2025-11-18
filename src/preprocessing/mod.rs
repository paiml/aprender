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

use crate::primitives::Matrix;
use crate::traits::Transformer;
use serde::{Deserialize, Serialize};

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
    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str> {
        let mean = self.mean.as_ref().ok_or("Scaler not fitted")?;
        let std = self.std.as_ref().ok_or("Scaler not fitted")?;

        let (n_samples, n_features) = x.shape();
        if n_features != mean.len() {
            return Err("Feature dimension mismatch");
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

        Matrix::from_vec(n_samples, n_features, result)
    }
}

impl Transformer for StandardScaler {
    /// Computes the mean and standard deviation of each feature.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples");
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
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str> {
        let mean = self.mean.as_ref().ok_or("Scaler not fitted")?;
        let std = self.std.as_ref().ok_or("Scaler not fitted")?;

        let (n_samples, n_features) = x.shape();
        if n_features != mean.len() {
            return Err("Feature dimension mismatch");
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

        Matrix::from_vec(n_samples, n_features, result)
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
    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str> {
        let data_min = self.data_min.as_ref().ok_or("Scaler not fitted")?;
        let data_max = self.data_max.as_ref().ok_or("Scaler not fitted")?;

        let (n_samples, n_features) = x.shape();
        if n_features != data_min.len() {
            return Err("Feature dimension mismatch");
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

        Matrix::from_vec(n_samples, n_features, result)
    }
}

impl Transformer for MinMaxScaler {
    /// Computes the min and max of each feature.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples");
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
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str> {
        let data_min = self.data_min.as_ref().ok_or("Scaler not fitted")?;
        let data_max = self.data_max.as_ref().ok_or("Scaler not fitted")?;

        let (n_samples, n_features) = x.shape();
        if n_features != data_min.len() {
            return Err("Feature dimension mismatch");
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

        Matrix::from_vec(n_samples, n_features, result)
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
}
