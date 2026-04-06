//! Vector norm operations
//!
//! This module provides vector norm calculations:
//! - `norm_l1()` - L1 norm (Manhattan norm)
//! - `norm_l2()` - L2 norm (Euclidean norm)
//! - `norm_linf()` - L∞ norm (infinity norm / max norm)
//!
//! All three norms share the same `unsafe fn(&[f32]) -> f32` backend
//! signature, so dispatch is handled by the crate-level
//! [`dispatch_reduction!`] macro -- no per-module dispatch struct needed.

use crate::backends::VectorBackend;
use crate::{dispatch_reduction, Result, Vector};

impl Vector<f32> {
    /// L2 norm (Euclidean norm)
    ///
    /// Computes the Euclidean length of the vector: sqrt(sum(a\[i\]^2)).
    /// This is mathematically equivalent to sqrt(dot(self, self)).
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via the dot product operation.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, 4.0]);
    /// let norm = v.norm_l2()?;
    /// assert!((norm - 5.0).abs() < 1e-5); // sqrt(3^2 + 4^2) = 5
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns 0.0 for empty vectors (consistent with the mathematical definition).
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_l2()?, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm_l2(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }
        Ok(dispatch_reduction!(self.backend, norm_l2, &self.data))
    }

    /// Compute the L1 norm (Manhattan norm) of the vector
    ///
    /// Returns the sum of absolute values: ||v||₁ = sum(|v\[i\]|)
    ///
    /// The L1 norm is used in:
    /// - Machine learning (L1 regularization, Lasso regression)
    /// - Distance metrics (Manhattan distance)
    /// - Sparse modeling and feature selection
    /// - Signal processing
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, -4.0, 5.0]);
    /// let norm = v.norm_l1().unwrap();
    ///
    /// // |3| + |-4| + |5| = 12
    /// assert!((norm - 12.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty Vector
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_l1().unwrap(), 0.0);
    /// ```
    pub fn norm_l1(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }
        Ok(dispatch_reduction!(self.backend, norm_l1, &self.data))
    }

    /// Compute the L∞ norm (infinity norm / max norm) of the vector
    ///
    /// Returns the maximum absolute value: ||v||∞ = max(|v\[i\]|)
    ///
    /// The L∞ norm is used in:
    /// - Numerical analysis (error bounds, stability analysis)
    /// - Optimization (Chebyshev approximation)
    /// - Signal processing (peak detection)
    /// - Distance metrics (Chebyshev distance)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, -7.0, 5.0, -2.0]);
    /// let norm = v.norm_linf().unwrap();
    ///
    /// // max(|3|, |-7|, |5|, |-2|) = 7
    /// assert!((norm - 7.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty Vector
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_linf().unwrap(), 0.0);
    /// ```
    pub fn norm_linf(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }
        Ok(dispatch_reduction!(self.backend, norm_linf, &self.data))
    }
}

#[cfg(test)]
mod tests;
