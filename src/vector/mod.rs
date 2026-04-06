//! Vector type with multi-backend support
//!
//! This module provides the core `Vector<T>` type with SIMD-optimized operations
//! across multiple backends (Scalar, SSE2, AVX2, AVX-512, NEON, WASM SIMD).
//!
//! GPU thresholds intentionally set to usize::MAX to disable GPU for element-wise ops.
//! See docs/performance-analysis.md - GPU is 2-65,000x SLOWER than scalar for these ops.

#![allow(clippy::absurd_extreme_comparisons)]

// Submodules
pub mod dispatch;
mod ops;

// Tests (~10K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;

use crate::{Backend, Result, TruenoError};

/// High-performance vector with multi-backend support
///
/// # Examples
///
/// ```
/// use trueno::Vector;
///
/// let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
/// let result = a.add(&b).unwrap();
///
/// assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T> {
    data: Vec<T>,
    backend: Backend,
}

impl<T> Vector<T>
where
    T: Clone,
{
    /// Create vector from slice using auto-selected optimal backend
    ///
    /// # Performance
    ///
    /// Auto-selects the best available backend at creation time based on:
    /// - CPU feature detection (AVX-512 > AVX2 > AVX > SSE2)
    /// - Vector size (GPU for large workloads)
    /// - Platform availability (NEON on ARM, WASM SIMD in browser)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(v.len(), 4);
    /// ```
    pub fn from_slice(data: &[T]) -> Self {
        Self { data: data.to_vec(), backend: crate::select_best_available_backend() }
    }

    /// Create vector from an existing Vec (takes ownership, no copy)
    ///
    /// This is more efficient than `from_slice` when you already have a Vec
    /// and don't need to keep it, as it avoids an extra allocation and copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let v = Vector::from_vec(data);
    /// assert_eq!(v.len(), 3);
    /// ```
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data, backend: crate::select_best_available_backend() }
    }

    /// Create vector with specific backend (for benchmarking or testing)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::{Vector, Backend};
    ///
    /// let v = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
    /// assert_eq!(v.len(), 2);
    /// ```
    pub fn from_slice_with_backend(data: &[T], backend: Backend) -> Self {
        let resolved_backend = match backend {
            Backend::Auto => crate::select_best_available_backend(),
            other => other,
        };

        Self { data: data.to_vec(), backend: resolved_backend }
    }
}

impl Vector<f32> {
    /// Create vector with specified alignment for optimal SIMD performance
    ///
    /// This method attempts to create a vector with memory aligned to the specified byte boundary.
    /// Note: Rust's Vec allocator may already provide sufficient alignment for most use cases.
    /// This method validates the alignment requirement but uses standard Vec allocation.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements to allocate
    /// * `backend` - Backend to use for operations
    /// * `alignment` - Requested alignment in bytes (must be power of 2: 16, 32, 64)
    ///
    /// # Recommended Alignments
    ///
    /// - SSE2: 16 bytes (128-bit)
    /// - AVX2: 32 bytes (256-bit)
    /// - AVX-512: 64 bytes (512-bit)
    ///
    /// # Note on Implementation
    ///
    /// Currently uses Rust's default Vec allocator, which typically provides 16-byte alignment
    /// on modern systems. Custom allocators for specific alignments will be added in future versions.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::{Vector, Backend};
    ///
    /// // Create vector with requested 16-byte alignment
    /// let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();
    /// assert_eq!(v.len(), 100);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `TruenoError::InvalidInput` if alignment is not a power of 2.
    pub fn with_alignment(size: usize, backend: Backend, alignment: usize) -> Result<Self> {
        // Validate alignment is power of 2
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(TruenoError::InvalidInput(format!(
                "Alignment must be power of 2, got {}",
                alignment
            )));
        }

        // Resolve backend
        let resolved_backend = match backend {
            Backend::Auto => crate::select_best_available_backend(),
            other => other,
        };

        // For now, use standard Vec allocation which typically provides good alignment
        // Future enhancement: use custom allocator for guaranteed alignment > 16 bytes
        let data = vec![0.0f32; size];

        // Verify actual alignment (for informational purposes)
        let ptr = data.as_ptr() as usize;
        let actual_alignment = ptr & !(ptr - 1); // Find lowest set bit

        // Log warning if alignment requirement not met (for future enhancement)
        if alignment > actual_alignment {
            // Note: This is not an error, just informational
            // The unaligned loads in SSE2 (_mm_loadu_ps) will still work correctly
            eprintln!(
                "Note: Requested {}-byte alignment, got {}-byte alignment. Using unaligned loads.",
                alignment, actual_alignment
            );
        }

        Ok(Self { data, backend: resolved_backend })
    }
}

impl<T> Vector<T>
where
    T: Clone,
{
    /// Get underlying data as slice
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get vector length
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(v.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if vector is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v1: Vector<f32> = Vector::from_slice(&[]);
    /// assert!(v1.is_empty());
    ///
    /// let v2 = Vector::from_slice(&[1.0]);
    /// assert!(!v2.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the backend being used
    pub fn backend(&self) -> Backend {
        self.backend
    }
}

// Note: Vector<f32> operations have been moved to submodules in ops/:
// - ops/normalization.rs: zscore, minmax_normalize, layer_norm, layer_norm_simple, normalize
// - ops/norms.rs: norm_l1, norm_l2, norm_linf
// - ops/transforms.rs: abs, clamp, clip, lerp, sqrt, recip, pow
// - ops/arithmetic.rs: add, sub, mul, div, scale, fma
// - ops/reductions.rs: dot, sum, max, min, argmax, argmin, mean, variance, stddev, covariance, correlation
// - ops/activations.rs: relu, sigmoid, gelu, etc.
// - ops/transcendental.rs: exp, log, sin, cos, etc.
// - ops/rounding.rs: floor, ceil, round, trunc, etc.
