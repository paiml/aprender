//! Vector transformation operations
//!
//! This module provides element-wise transformation methods:
//! - `abs()` - Element-wise absolute value
//! - `clamp()` / `clip()` - Clamp values to a range
//! - `lerp()` - Linear interpolation between two vectors
//! - `sqrt()` - Element-wise square root (in `math` submodule)
//! - `recip()` - Element-wise reciprocal (1/x) (in `math` submodule)
//! - `pow()` - Element-wise power (in `math` submodule)

mod math;

#[cfg(target_arch = "x86_64")]
use crate::backends::avx2::Avx2Backend;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::backends::neon::NeonBackend;
use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
use crate::backends::VectorBackend;
use crate::{Backend, Result, TruenoError, Vector};

impl Vector<f32> {
    /// Compute element-wise absolute value
    ///
    /// Returns a new vector where each element is the absolute value of the corresponding input element.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
    /// let result = v.abs()?;
    ///
    /// assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Empty Vector
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// let result = v.abs()?;
    /// assert_eq!(result.len(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn abs(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.as_slice().is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend() {
                    Backend::Scalar => ScalarBackend::abs(self.as_slice(), &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::abs(self.as_slice(), &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::abs(self.as_slice(), &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::abs(self.as_slice(), &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::abs(self.as_slice(), &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                        ScalarBackend::abs(self.as_slice(), &mut result_data)
                    }
                    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                    Backend::NEON => ScalarBackend::abs(self.as_slice(), &mut result_data),
                    #[cfg(not(target_arch = "wasm32"))]
                    Backend::WasmSIMD => ScalarBackend::abs(self.as_slice(), &mut result_data),
                }
            }
        }

        // Construct directly (no copy) — from_slice_with_backend would copy 4MB!
        Ok(Vector { data: result_data, backend: self.backend() })
    }

    /// Clip values to a specified range [min_val, max_val]
    ///
    /// Constrains each element to be within the specified range:
    /// - Values below min_val become min_val
    /// - Values above max_val become max_val
    /// - Values within range stay unchanged
    ///
    /// This is useful for outlier handling, gradient clipping in neural networks,
    /// and ensuring values stay within valid bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    /// let clipped = v.clip(0.0, 10.0)?;
    ///
    /// // Values: [-5, 0, 5, 10, 15] → [0, 0, 5, 10, 10]
    /// assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Invalid range
    ///
    /// Returns InvalidInput error if min_val > max_val.
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.clip(10.0, 5.0); // min > max
    /// assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    /// ```
    pub fn clip(&self, min_val: f32, max_val: f32) -> Result<Self> {
        if min_val > max_val {
            return Err(TruenoError::InvalidInput(format!(
                "min_val ({}) must be <= max_val ({})",
                min_val, max_val
            )));
        }

        // Scalar fallback: Element-wise clamp
        let data: Vec<f32> = self.as_slice().iter().map(|&x| x.max(min_val).min(max_val)).collect();

        Ok(Vector::from_vec(data))
    }

    /// Clamp elements to range [min_val, max_val]
    ///
    /// Returns a new vector where each element is constrained to the specified range.
    /// Elements below min_val become min_val, elements above max_val become max_val.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    /// let result = v.clamp(0.0, 10.0)?;
    ///
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Negative Range
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
    /// let result = v.clamp(-8.0, -2.0)?;
    /// assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if min_val > max_val.
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Result<Vector<f32>> {
        // Validate range
        if min_val > max_val {
            return Err(TruenoError::InvalidInput(format!(
                "Invalid clamp range: min ({}) > max ({})",
                min_val, max_val
            )));
        }

        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.as_slice().is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend() {
                    Backend::Scalar => {
                        ScalarBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                        ScalarBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                    Backend::NEON => {
                        ScalarBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    Backend::WasmSIMD => {
                        ScalarBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                }
            }
        }

        Ok(Vector { data: result_data, backend: self.backend() })
    }

    /// Linear interpolation between two vectors
    ///
    /// Computes element-wise linear interpolation: `result\[i\] = a\[i\] + t * (b\[i\] - a\[i\])`
    ///
    /// - When `t = 0.0`, returns `self`
    /// - When `t = 1.0`, returns `other`
    /// - Values outside `[0, 1]` perform extrapolation
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
    /// let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
    /// let result = a.lerp(&b, 0.5)?;
    ///
    /// assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Extrapolation
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[0.0, 10.0]);
    /// let b = Vector::from_slice(&[10.0, 20.0]);
    ///
    /// // t > 1.0 extrapolates beyond b
    /// let result = a.lerp(&b, 2.0)?;
    /// assert_eq!(result.as_slice(), &[20.0, 30.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if vectors have different lengths.
    pub fn lerp(&self, other: &Vector<f32>, t: f32) -> Result<Vector<f32>> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.as_slice().is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend() {
                    Backend::Scalar => {
                        ScalarBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                        ScalarBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                    Backend::NEON => {
                        ScalarBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    Backend::WasmSIMD => {
                        ScalarBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                }
            }
        }

        Ok(Vector { data: result_data, backend: self.backend() })
    }
}

#[cfg(test)]
mod tests;
