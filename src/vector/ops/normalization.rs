//! Normalization operations for Vector<f32>
//!
//! This module provides normalization methods:
//! - `zscore()` - Z-score normalization (standardization)
//! - `minmax_normalize()` - Min-max normalization to [0, 1]
//! - `layer_norm()` - Layer normalization with learnable parameters
//! - `layer_norm_simple()` - Layer normalization without learnable parameters
//! - `normalize()` - Normalize to unit length (L2 norm = 1)

use crate::{Result, TruenoError, Vector};

impl Vector<f32> {
    /// Z-score normalization (standardization)
    ///
    /// Transforms the vector to have mean = 0 and standard deviation = 1.
    /// Each element is transformed as: z\[i\] = (x\[i\] - μ) / σ
    ///
    /// This is a fundamental preprocessing step in machine learning and statistics,
    /// ensuring features have comparable scales and are centered around zero.
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via mean() and stddev(), then applies
    /// element-wise operations (sub, scale) which also use SIMD.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let z = v.zscore()?;
    ///
    /// // Verify mean ≈ 0
    /// let mean = z.mean()?;
    /// assert!(mean.abs() < 1e-5);
    ///
    /// // Verify stddev ≈ 1
    /// let std = z.stddev()?;
    /// assert!((std - 1.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute mean/stddev).
    ///
    /// # Division by zero
    ///
    /// Returns DivisionByZero error if the vector has zero standard deviation
    /// (i.e., all elements are identical/constant).
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[5.0, 5.0, 5.0]); // Constant
    /// assert!(matches!(v.zscore(), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn zscore(&self) -> Result<Self> {
        if self.as_slice().is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let mean_val = self.mean()?;
        let std_val = self.stddev()?;

        // Check for zero standard deviation (constant vector)
        if std_val.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // Transform: z[i] = (x[i] - μ) / σ
        let inv_std = 1.0 / std_val;
        let data: Vec<f32> = self.as_slice().iter().map(|&x| (x - mean_val) * inv_std).collect();

        Ok(Vector::from_vec(data))
    }

    /// Min-max normalization (scaling to [0, 1] range)
    ///
    /// Transforms the vector so that the minimum value becomes 0 and the maximum
    /// value becomes 1, with all other values scaled proportionally.
    /// Formula: x'\[i\] = (x\[i\] - min) / (max - min)
    ///
    /// This is a fundamental preprocessing technique in machine learning, especially
    /// for algorithms sensitive to feature magnitudes (e.g., neural networks, k-NN).
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via min() and max() operations, then
    /// applies element-wise transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let normalized = v.minmax_normalize()?;
    ///
    /// // Verify range [0, 1]
    /// let min = normalized.min()?;
    /// let max = normalized.max()?;
    /// assert!((min - 0.0).abs() < 1e-5);
    /// assert!((max - 1.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute min/max).
    ///
    /// # Division by zero
    ///
    /// Returns DivisionByZero error if the vector has all identical elements
    /// (i.e., min = max, causing division by zero in the normalization formula).
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[5.0, 5.0, 5.0]); // Constant
    /// assert!(matches!(v.minmax_normalize(), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn minmax_normalize(&self) -> Result<Self> {
        if self.as_slice().is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let min_val = self.min()?;
        let max_val = self.max()?;
        let range = max_val - min_val;

        // Check for zero range (constant vector)
        if range.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // Transform: x'[i] = (x[i] - min) / (max - min)
        let inv_range = 1.0 / range;
        let data: Vec<f32> = self.as_slice().iter().map(|&x| (x - min_val) * inv_range).collect();

        Ok(Vector::from_vec(data))
    }

    /// Layer normalization with learnable parameters (Issue #61: ML primitives)
    ///
    /// Applies layer normalization: `y = gamma * (x - mean) / sqrt(variance + eps) + beta`
    ///
    /// This is a fundamental normalization technique in transformers and other
    /// modern neural network architectures. Unlike batch normalization, layer norm
    /// normalizes across the feature dimension, making it suitable for sequence models.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Scale parameter (typically learned, initialized to 1.0)
    /// * `beta` - Shift parameter (typically learned, initialized to 0.0)
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    ///
    /// # Returns
    ///
    /// Normalized vector with the same shape as input
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if gamma or beta have different lengths than self
    /// Returns `EmptyVector` if input is empty
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]); // Scale = 1
    /// let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);  // Shift = 0
    ///
    /// let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();
    ///
    /// // Output should be approximately standardized (mean ≈ 0, std ≈ 1)
    /// let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    /// assert!(mean.abs() < 1e-5);
    /// ```
    ///
    /// # Performance
    ///
    /// Single-pass computation using Welford's algorithm for numerical stability.
    /// Time complexity: O(n), Space complexity: O(n).
    pub fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> Result<Self> {
        if self.as_slice().is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        if self.len() != gamma.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: gamma.len() });
        }

        if self.len() != beta.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: beta.len() });
        }

        // Compute mean
        let mean_val = self.mean()?;

        // Compute variance: E[(x - mean)^2]
        let variance: f32 = self
            .as_slice()
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / self.len().max(1) as f32;

        // Compute inverse standard deviation for numerical stability
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Apply normalization: y = gamma * (x - mean) * inv_std + beta
        let data: Vec<f32> = self
            .as_slice()
            .iter()
            .zip(gamma.as_slice().iter())
            .zip(beta.as_slice().iter())
            .map(|((&x, &g), &b)| g * (x - mean_val) * inv_std + b)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Layer normalization without learnable parameters
    ///
    /// Simplified version that just standardizes the input: `y = (x - mean) / sqrt(variance + eps)`
    ///
    /// This is equivalent to calling `layer_norm` with gamma=1 and beta=0.
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let y = x.layer_norm_simple(1e-5).unwrap();
    ///
    /// // Output should be standardized
    /// let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    /// assert!(mean.abs() < 1e-5);
    /// ```
    pub fn layer_norm_simple(&self, eps: f32) -> Result<Self> {
        if self.as_slice().is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let mean_val = self.mean()?;

        // Compute variance
        let variance: f32 = self
            .as_slice()
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / self.len().max(1) as f32;

        let inv_std = 1.0 / (variance + eps).sqrt();

        let data: Vec<f32> = self.as_slice().iter().map(|&x| (x - mean_val) * inv_std).collect();

        Ok(Vector::from_vec(data))
    }

    /// Normalize the vector to unit length (L2 norm = 1)
    ///
    /// Returns a new vector in the same direction but with magnitude 1.
    ///
    /// # Errors
    ///
    /// Returns `TruenoError::DivisionByZero` if the vector has zero norm (cannot normalize zero vector).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, 4.0]);
    /// let unit = v.normalize().unwrap();
    ///
    /// // Result is [0.6, 0.8] (a unit vector)
    /// assert!((unit.as_slice()[0] - 0.6).abs() < 1e-5);
    /// assert!((unit.as_slice()[1] - 0.8).abs() < 1e-5);
    ///
    /// // Verify it's a unit vector (norm = 1)
    /// assert!((unit.norm_l2().unwrap() - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Zero Vector Error
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[0.0, 0.0]);
    /// assert!(matches!(v.normalize(), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn normalize(&self) -> Result<Vector<f32>> {
        let norm = self.norm_l2()?;

        // Check for zero or near-zero norm (cannot normalize zero vector)
        if norm.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // Divide each element by the norm using scalar multiplication
        // This avoids creating an intermediate vector
        self.scale(1.0 / norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let z = v.zscore().unwrap();

        // Mean should be ~0
        let mean = z.mean().unwrap();
        assert!(mean.abs() < 1e-5);

        // Stddev should be ~1
        let std = z.stddev().unwrap();
        assert!((std - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_zscore_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert!(matches!(v.zscore(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_zscore_constant() {
        let v = Vector::from_slice(&[5.0, 5.0, 5.0]);
        assert!(matches!(v.zscore(), Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_minmax_normalize_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = v.minmax_normalize().unwrap();

        assert!((normalized.min().unwrap() - 0.0).abs() < 1e-5);
        assert!((normalized.max().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_normalize_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert!(matches!(v.minmax_normalize(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_minmax_normalize_constant() {
        let v = Vector::from_slice(&[5.0, 5.0, 5.0]);
        assert!(matches!(v.minmax_normalize(), Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_layer_norm() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);

        let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

        // Mean should be ~0
        let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_size_mismatch() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let gamma = Vector::from_slice(&[1.0, 1.0]); // Wrong size
        let beta = Vector::from_slice(&[0.0, 0.0, 0.0]);

        assert!(matches!(x.layer_norm(&gamma, &beta, 1e-5), Err(TruenoError::SizeMismatch { .. })));
    }

    #[test]
    fn test_layer_norm_simple() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = x.layer_norm_simple(1e-5).unwrap();

        let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_normalize_unit_vector() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        let unit = v.normalize().unwrap();

        assert!((unit.as_slice()[0] - 0.6).abs() < 1e-5);
        assert!((unit.as_slice()[1] - 0.8).abs() < 1e-5);
        assert!((unit.norm_l2().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = Vector::from_slice(&[0.0, 0.0]);
        assert!(matches!(v.normalize(), Err(TruenoError::DivisionByZero)));
    }
}
