//! Rounding and sign functions for Vector<f32>
//!
//! This module provides rounding, truncation, and sign-related operations:
//! - Rounding: `floor`, `ceil`, `round`, `trunc`
//! - Parts: `fract` (fractional part)
//! - Sign: `signum`, `copysign`, `neg`

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(unused_imports)]
use crate::backends::neon::NeonBackend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{dispatch_unary_op, Result, TruenoError};

impl Vector<f32> {
    /// Computes the floor (round down to nearest integer) of each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.7, -2.3, 5.0]);
    /// let result = v.floor()?;
    /// assert_eq!(result.as_slice(), &[3.0, -3.0, 5.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn floor(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, floor, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Computes the ceiling (round up to nearest integer) of each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.2, -2.7, 5.0]);
    /// let result = v.ceil()?;
    /// assert_eq!(result.as_slice(), &[4.0, -2.0, 5.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn ceil(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, ceil, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Rounds each element to the nearest integer.
    ///
    /// Uses "round half away from zero" strategy:
    /// - 0.5 rounds to 1.0, 1.5 rounds to 2.0, -1.5 rounds to -2.0, etc.
    /// - Positive halfway cases round up, negative halfway cases round down.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8]);
    /// let result = v.round()?;
    /// assert_eq!(result.as_slice(), &[3.0, 4.0, -2.0, -3.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn round(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, round, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Truncates each element toward zero (removes fractional part).
    ///
    /// Truncation always moves toward zero:
    /// - Positive values: equivalent to floor() (e.g., 3.7 → 3.0)
    /// - Negative values: equivalent to ceil() (e.g., -3.7 → -3.0)
    /// - This differs from floor() which always rounds down
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.7, -2.7, 5.0]);
    /// let result = v.trunc()?;
    /// assert_eq!(result.as_slice(), &[3.0, -2.0, 5.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn trunc(&self) -> Result<Vector<f32>> {
        let trunc_data: Vec<f32> = self.data.iter().map(|x| x.trunc()).collect();
        Ok(Vector { data: trunc_data, backend: self.backend })
    }

    /// Returns the fractional part of each element.
    ///
    /// The fractional part has the same sign as the original value:
    /// - Positive: fract(3.7) = 0.7
    /// - Negative: fract(-3.7) = -0.7
    /// - Decomposition property: x = trunc(x) + fract(x)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.7, -2.3, 5.0]);
    /// let result = v.fract()?;
    /// // Fractional parts: 0.7, -0.3, 0.0
    /// assert!((result.as_slice()[0] - 0.7).abs() < 1e-5);
    /// assert!((result.as_slice()[1] - (-0.3)).abs() < 1e-5);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn fract(&self) -> Result<Vector<f32>> {
        let fract_data: Vec<f32> = self.data.iter().map(|x| x.fract()).collect();
        Ok(Vector { data: fract_data, backend: self.backend })
    }

    /// Returns the sign of each element.
    ///
    /// Returns:
    /// - `1.0` if the value is positive (including +0.0 and +∞)
    /// - `-1.0` if the value is negative (including -0.0 and -∞)
    /// - `NaN` if the value is NaN
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[5.0, -3.0, 0.0, -0.0]);
    /// let result = v.signum()?;
    /// assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn signum(&self) -> Result<Vector<f32>> {
        let signum_data: Vec<f32> = self.data.iter().map(|x| x.signum()).collect();
        Ok(Vector { data: signum_data, backend: self.backend })
    }

    /// Returns a vector with the magnitude of `self` and the sign of `sign`.
    ///
    /// For each element pair, takes the magnitude from `self` and the sign from `sign`.
    /// Equivalent to `abs(self\[i\])` with the sign of `sign\[i\]`.
    ///
    /// # Arguments
    ///
    /// * `sign` - Vector providing the sign for each element
    ///
    /// # Errors
    ///
    /// Returns `TruenoError::SizeMismatch` if vectors have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let magnitude = Vector::from_slice(&[5.0, 3.0, 2.0]);
    /// let sign = Vector::from_slice(&[-1.0, 1.0, -1.0]);
    /// let result = magnitude.copysign(&sign)?;
    /// assert_eq!(result.as_slice(), &[-5.0, 3.0, -2.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn copysign(&self, sign: &Self) -> Result<Vector<f32>> {
        if self.len() != sign.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: sign.len() });
        }

        let copysign_data: Vec<f32> =
            self.data.iter().zip(sign.data.iter()).map(|(mag, sgn)| mag.copysign(*sgn)).collect();

        Ok(Vector { data: copysign_data, backend: self.backend })
    }

    /// Element-wise minimum of two vectors.
    ///
    /// Returns a new vector where each element is the minimum of the corresponding
    /// elements from self and other.
    ///
    /// NaN handling: Prefers non-NaN values (NAN.min(x) = x).
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// let a = Vector::from_slice(&[1.0, 5.0, 3.0]);
    /// let b = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let result = a.minimum(&b)?;
    /// assert_eq!(result.as_slice(), &[1.0, 3.0, 3.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn minimum(&self, other: &Self) -> Result<Vector<f32>> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        let minimum_data: Vec<f32> =
            self.data.iter().zip(other.data.iter()).map(|(a, b)| a.min(*b)).collect();

        Ok(Vector { data: minimum_data, backend: self.backend })
    }

    /// Element-wise maximum of two vectors.
    ///
    /// Returns a new vector where each element is the maximum of the corresponding
    /// elements from self and other.
    ///
    /// NaN handling: Prefers non-NaN values (NAN.max(x) = x).
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// let a = Vector::from_slice(&[1.0, 5.0, 3.0]);
    /// let b = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let result = a.maximum(&b)?;
    /// assert_eq!(result.as_slice(), &[2.0, 5.0, 4.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn maximum(&self, other: &Self) -> Result<Vector<f32>> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        let maximum_data: Vec<f32> =
            self.data.iter().zip(other.data.iter()).map(|(a, b)| a.max(*b)).collect();

        Ok(Vector { data: maximum_data, backend: self.backend })
    }

    /// Element-wise negation (unary minus).
    ///
    /// Returns a new vector where each element is the negation of the corresponding
    /// element from self.
    ///
    /// Properties: Double negation is identity: -(-x) = x
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// let a = Vector::from_slice(&[1.0, -2.0, 3.0]);
    /// let result = a.neg()?;
    /// assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn neg(&self) -> Result<Vector<f32>> {
        let neg_data: Vec<f32> = self.data.iter().map(|x| -x).collect();
        Ok(Vector { data: neg_data, backend: self.backend })
    }
}
