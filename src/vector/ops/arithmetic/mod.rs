//! Arithmetic operations for Vector<f32>
//!
//! This module provides element-wise arithmetic operations:
//! - Basic: `add`, `sub`, `mul`, `div`
//! - Scalar: `scale`
//! - Fused: `fma` (fused multiply-add)

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
use crate::vector::Vector;
use crate::{dispatch_binary_op, Backend, Result, TruenoError};

impl Vector<f32> {
    /// Element-wise addition
    ///
    /// # Performance
    ///
    /// Auto-selects the best available backend:
    /// - **AVX2**: ~4x faster than scalar for 1K+ elements
    /// - **GPU**: ~50x faster than scalar for 10M+ elements
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    /// let result = a.add(&b)?;
    ///
    /// assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::SizeMismatch`] if vectors have different lengths.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        // Uninit allocation: avoids the zero-fill cost (70µs+ at 1M elements)
        // since every element will be overwritten by dispatch_binary_op below.
        // SAFETY: dispatch_binary_op!(..., add, a, b, out) writes to EVERY element
        // of `out` (it's an element-wise add). No reads before writes.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        unsafe {
            result.set_len(n);
        }

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000; // Threshold for element-wise ops
            const CHUNK_SIZE: usize = 65536; // 64K elements = 256KB, cache-friendly

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, add, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self { data: result, backend: self.backend });
            }
        }

        dispatch_binary_op!(self.backend, add, &self.data, &other.data, &mut result);

        Ok(Self { data: result, backend: self.backend })
    }

    /// Element-wise subtraction
    ///
    /// # Performance
    ///
    /// Auto-selects the best available backend:
    /// - **AVX2**: ~4x faster than scalar for 1K+ elements
    /// - **GPU**: ~50x faster than scalar for 10M+ elements
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
    /// let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = a.sub(&b)?;
    ///
    /// assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::SizeMismatch`] if vectors have different lengths.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        // Uninit allocation: skip zero-fill since dispatch_binary_op writes all elements.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Every element is written before any read (by element-wise op below).
        unsafe {
            result.set_len(n);
        }

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000;
            const CHUNK_SIZE: usize = 65536;

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, sub, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self { data: result, backend: self.backend });
            }
        }

        dispatch_binary_op!(self.backend, sub, &self.data, &other.data, &mut result);

        Ok(Self { data: result, backend: self.backend })
    }

    /// Element-wise multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    /// let result = a.mul(&b)?;
    ///
    /// assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        // Uninit allocation: skip zero-fill since dispatch_binary_op writes all elements.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Every element is written before any read (by element-wise op below).
        unsafe {
            result.set_len(n);
        }

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000;
            const CHUNK_SIZE: usize = 65536;

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, mul, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self { data: result, backend: self.backend });
            }
        }

        dispatch_binary_op!(self.backend, mul, &self.data, &other.data, &mut result);

        Ok(Self { data: result, backend: self.backend })
    }

    /// Element-wise division
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[10.0, 20.0, 30.0]);
    /// let b = Vector::from_slice(&[2.0, 4.0, 5.0]);
    /// let result = a.div(&b)?;
    ///
    /// assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: other.len() });
        }

        // Uninit allocation: skip zero-fill since dispatch_binary_op writes all elements.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Every element is written before any read (by element-wise op below).
        unsafe {
            result.set_len(n);
        }

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000;
            const CHUNK_SIZE: usize = 65536;

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, div, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self { data: result, backend: self.backend });
            }
        }

        dispatch_binary_op!(self.backend, div, &self.data, &other.data, &mut result);

        Ok(Self { data: result, backend: self.backend })
    }

    /// Scalar multiplication (scale all elements by a scalar value)
    ///
    /// Returns a new vector where each element is multiplied by the scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = v.scale(2.0)?;
    ///
    /// assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Scaling by Zero
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.scale(0.0)?;
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Negative Scaling
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, -2.0, 3.0]);
    /// let result = v.scale(-2.0)?;
    /// assert_eq!(result.as_slice(), &[-2.0, 4.0, -6.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn scale(&self, scalar: f32) -> Result<Vector<f32>> {
        // Uninit allocation: backend writes all elements.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: backend scale() writes every element before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => ScalarBackend::scale(&self.data, scalar, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::scale(&self.data, scalar, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::scale(&self.data, scalar, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::scale(&self.data, scalar, &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::scale(&self.data, scalar, &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                        ScalarBackend::scale(&self.data, scalar, &mut result_data)
                    }
                    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                    Backend::NEON => ScalarBackend::scale(&self.data, scalar, &mut result_data),
                    #[cfg(not(target_arch = "wasm32"))]
                    Backend::WasmSIMD => ScalarBackend::scale(&self.data, scalar, &mut result_data),
                }
            }
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Fused multiply-add: result\[i\] = self\[i\] * b\[i\] + c\[i\]
    ///
    /// Computes element-wise fused multiply-add operation. On hardware with FMA support
    /// (AVX2, NEON), this is a single instruction with better performance and numerical
    /// accuracy (no intermediate rounding). On platforms without FMA (SSE2, WASM), uses
    /// separate multiply and add operations.
    ///
    /// # Arguments
    ///
    /// * `b` - The second vector to multiply with
    /// * `c` - The vector to add to the product
    ///
    /// # Returns
    ///
    /// A new vector where each element is `self\[i\] * b\[i\] + c\[i\]`
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if vector lengths don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    /// let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = a.fma(&b, &c)?;
    /// assert_eq!(result.as_slice(), &[11.0, 20.0, 31.0]);  // [2*5+1, 3*6+2, 4*7+3]
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Use Cases
    ///
    /// - Neural networks: matrix multiplication, backpropagation
    /// - Scientific computing: polynomial evaluation, numerical integration
    /// - Graphics: transformation matrices, shader computations
    /// - Physics simulations: force calculations, particle systems
    pub fn fma(&self, b: &Vector<f32>, c: &Vector<f32>) -> Result<Vector<f32>> {
        if self.len() != b.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: b.len() });
        }
        if self.len() != c.len() {
            return Err(TruenoError::SizeMismatch { expected: self.len(), actual: c.len() });
        }

        // Uninit allocation: backend fma writes all elements.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: backend fma() writes every element before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => {
                        ScalarBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                        ScalarBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                    Backend::NEON => {
                        ScalarBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    Backend::WasmSIMD => {
                        ScalarBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                }
            }
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }
}

#[cfg(test)]
mod tests;
