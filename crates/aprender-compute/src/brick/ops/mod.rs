#![allow(missing_docs)]
//! Built-in Compute Operations
//!
//! Pre-defined operations that implement the ComputeOp trait:
//! - DotOp: Vector dot product
//! - AddOp: Element-wise vector addition
//! - MatmulOp: Matrix multiplication (SIMD-optimized)
//! - SoftmaxOp: Softmax with SIMD exp approximation (SIMD-EXP)

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

// ============================================================================
// DotOp: Dot Product
// ============================================================================

/// Dot product operation.
#[derive(Debug, Clone)]
pub struct DotOp {
    /// Expected vector length
    pub len: usize,
}

impl DotOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for DotOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        if a.len() != b.len() {
            return Err(TruenoError::SizeMismatch { expected: a.len(), actual: b.len() });
        }
        // Simple scalar implementation for now
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(sum)
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        // Each element pair is roughly 1 "token" of work
        input.0.len()
    }
}

// ============================================================================
// AddOp: Element-wise Addition
// ============================================================================

/// Element-wise add operation.
#[derive(Debug, Clone)]
pub struct AddOp {
    /// Expected vector length
    pub len: usize,
}

impl AddOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for AddOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "add"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        if a.len() != b.len() {
            return Err(TruenoError::SizeMismatch { expected: a.len(), actual: b.len() });
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        input.0.len()
    }
}

// ============================================================================
// MatmulOp: Matrix Multiplication
// ============================================================================

/// Matrix multiplication operation.
#[derive(Debug, Clone)]
pub struct MatmulOp {
    /// M dimension (rows of A)
    pub m: usize,
    /// K dimension (cols of A = rows of B)
    pub k: usize,
    /// N dimension (cols of B)
    pub n: usize,
}

impl MatmulOp {
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self { m, k, n }
    }
}

impl ComputeOp for MatmulOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "matmul"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        let expected_a = self.m * self.k;
        let expected_b = self.k * self.n;

        if a.len() != expected_a {
            return Err(TruenoError::SizeMismatch { expected: expected_a, actual: a.len() });
        }
        if b.len() != expected_b {
            return Err(TruenoError::SizeMismatch { expected: expected_b, actual: b.len() });
        }

        // SIMD-optimized matrix multiplication via Matrix type
        // Uses AVX2/AVX-512 with cache blocking for ~10-50x speedup
        let simd_backend = crate::Backend::select_best();
        let mat_a = crate::Matrix::from_vec_with_backend(self.m, self.k, a, simd_backend);
        let mat_b = crate::Matrix::from_vec_with_backend(self.k, self.n, b, simd_backend);

        let result = mat_a.matmul(&mat_b)?;
        // Take ownership of the data Vec directly — avoids redundant copy.
        Ok(result.data)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // For matmul, "tokens" = number of output elements
        // Each output requires K multiply-adds
        self.m * self.n
    }
}

// ============================================================================
// SoftmaxOp: Softmax with SIMD Exp (SIMD-EXP)
// ============================================================================

/// Softmax operation.
#[derive(Debug, Clone)]
pub struct SoftmaxOp {
    /// Expected vector length
    pub len: usize,
}

impl SoftmaxOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for SoftmaxOp {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "softmax"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        // CGP-DBUF: delegate to blis::softmax which has AVX2 fused exp+sum
        // (3-pass: max, fused exp+sum, normalize). This eliminates 3 intermediate
        // allocations (shifted, exp_vals, result) and uses polynomial fast_exp.
        Ok(crate::blis::softmax::softmax_1d_alloc(&input))
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        input.len()
    }
}

impl SoftmaxOp {
    /// Check if backend supports SIMD acceleration
    #[inline]
    pub fn is_simd_backend(backend: Backend) -> bool {
        matches!(
            backend,
            Backend::Avx2 | Backend::Avx512 | Backend::Sse2 | Backend::Neon | Backend::Auto
        )
    }
    // CGP-DBUF: SIMD helper methods (simd_max, simd_exp, simd_sum, simd_scale,
    // avx2_max, avx2_exp, avx2_sum, avx2_scale) removed — execute() now delegates
    // to blis::softmax::softmax_1d_alloc which has fused AVX2 fast_exp path.
}

#[cfg(test)]
mod tests;
