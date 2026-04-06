//! Sparse matrix operations (SpMV, SpMM).
//!
//! # Contract: sparse-spmv-v1.yaml
//!
//! SpMV equation: `y_i = α · Σ_j A_{ij} · x_j + β · y_i`
//!
//! ## Proof obligations
//! - Output dimension: `len(y) == A.rows()`
//! - Backward error: `|Ax - y_exact| ≤ nnz_per_row · u · |A| · |x|`
//! - SIMD-scalar equivalence: within 8 ULP
//!
//! ## Kernel phases
//! 1. format_validation (at construction)
//! 2. row-split accumulation (scalar/SIMD)
//! 3. output scaling (α, β)

use crate::csr::CsrMatrix;
use crate::error::SparseError;

/// Backend dispatch trait for pluggable SIMD SpMV kernels.
///
/// Implementations provide SpMV for a specific hardware target.
/// The default dispatch in `SparseOps::spmv` selects the best
/// available backend at runtime.
pub trait SparseBackend {
    /// Perform SpMV: `y = alpha * A * x + beta * y` using this backend.
    fn spmv_kernel(a: &CsrMatrix<f32>, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]);

    /// Perform SpMM: `C = alpha * A * B + beta * C` using this backend.
    ///
    /// B is row-major with `b_cols` columns.
    fn spmm_kernel(
        a: &CsrMatrix<f32>,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    );
}

/// Scalar (portable) SpMV backend.
pub struct ScalarBackend;

impl SparseBackend for ScalarBackend {
    fn spmv_kernel(a: &CsrMatrix<f32>, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
        spmv_csr_scalar(a, alpha, x, beta, y);
    }

    fn spmm_kernel(
        a: &CsrMatrix<f32>,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) {
        spmm_csr_scalar(a, alpha, b, b_cols, beta, c);
    }
}

/// AVX2 SpMV backend (x86_64 with AVX2+FMA).
#[cfg(target_arch = "x86_64")]
pub struct Avx2Backend;

#[cfg(target_arch = "x86_64")]
impl SparseBackend for Avx2Backend {
    fn spmv_kernel(a: &CsrMatrix<f32>, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
        // SAFETY: caller must ensure AVX2+FMA is available
        unsafe { spmv_csr_avx2(a, alpha, x, beta, y) }
    }

    fn spmm_kernel(
        a: &CsrMatrix<f32>,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) {
        // AVX2 SpMM not yet specialized — fall back to scalar
        spmm_csr_scalar(a, alpha, b, b_cols, beta, c);
    }
}

/// NEON SpMV backend stub (aarch64).
#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SparseBackend for NeonBackend {
    fn spmv_kernel(a: &CsrMatrix<f32>, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
        // NEON dispatch not yet implemented — falls back to scalar
        spmv_csr_scalar(a, alpha, x, beta, y);
    }

    fn spmm_kernel(
        a: &CsrMatrix<f32>,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) {
        // NEON SpMM not yet specialized — fall back to scalar
        spmm_csr_scalar(a, alpha, b, b_cols, beta, c);
    }
}

/// Sparse matrix operations trait.
///
/// Provides SpMV and SpMM with provable error bounds.
pub trait SparseOps {
    /// Sparse matrix-vector multiply: `y = α * A * x + β * y`
    ///
    /// # Contract: sparse-spmv-v1.yaml / spmv
    ///
    /// **Preconditions**: `x.len() == self.cols()`, `y.len() == self.rows()`
    /// **Postcondition**: backward error ≤ `nnz_per_row * f32::EPSILON * ||A||_inf * ||x||_inf`
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    fn spmv(&self, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) -> Result<(), SparseError>;

    /// Sparse matrix-dense matrix multiply: `C = α * A * B + β * C`
    ///
    /// B is row-major with `b_cols` columns. C is row-major with `b_cols` columns.
    ///
    /// # Errors
    ///
    /// Returns error on dimension mismatch.
    fn spmm(
        &self,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), SparseError>;
}

impl SparseOps for CsrMatrix<f32> {
    fn spmv(&self, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) -> Result<(), SparseError> {
        // Dimension checks (contract enforcement)
        if x.len() != self.cols() {
            return Err(SparseError::SpMVDimensionMismatch {
                matrix_cols: self.cols(),
                x_len: x.len(),
            });
        }
        if y.len() != self.rows() {
            return Err(SparseError::SpMVOutputDimensionMismatch {
                matrix_rows: self.rows(),
                y_len: y.len(),
            });
        }

        // Dispatch to best available backend
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA detected at runtime
                unsafe {
                    spmv_csr_avx2(self, alpha, x, beta, y);
                    return Ok(());
                }
            }
        }

        // Scalar fallback
        spmv_csr_scalar(self, alpha, x, beta, y);
        Ok(())
    }

    fn spmm(
        &self,
        alpha: f32,
        b: &[f32],
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), SparseError> {
        if b.len() != self.cols() * b_cols {
            return Err(SparseError::SpMVDimensionMismatch {
                matrix_cols: self.cols(),
                x_len: b.len(),
            });
        }
        if c.len() != self.rows() * b_cols {
            return Err(SparseError::SpMVOutputDimensionMismatch {
                matrix_rows: self.rows(),
                y_len: c.len(),
            });
        }

        spmm_csr_scalar(self, alpha, b, b_cols, beta, c);
        Ok(())
    }
}

/// Scalar SpMV reference implementation.
///
/// Contract: this is the ground truth for SIMD/GPU parity testing.
fn spmv_csr_scalar(a: &CsrMatrix<f32>, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
    let offsets = a.offsets();
    let col_indices = a.col_indices();
    let values = a.values();

    for i in 0..a.rows() {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        // Kahan summation for improved accuracy (LAProof-aligned)
        let mut sum = 0.0_f64;
        let mut comp = 0.0_f64;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            let product = f64::from(values[idx]) * f64::from(x[j]);
            let t = sum + product;
            if sum.abs() >= product.abs() {
                comp += (sum - t) + product;
            } else {
                comp += (product - t) + sum;
            }
            sum = t;
        }
        sum += comp;

        y[i] = (f64::from(alpha) * sum + f64::from(beta) * f64::from(y[i])) as f32;
    }
}

/// AVX2 SpMV with gather instructions.
///
/// Uses `_mm256_i32gather_ps` for indirect x[col_indices[j]] access
/// and FMA for accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn spmv_csr_avx2(a: &CsrMatrix<f32>, alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
    use std::arch::x86_64::*;

    let offsets = a.offsets();
    let col_indices = a.col_indices();
    let values = a.values();

    for i in 0..a.rows() {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let row_nnz = end - start;

        // SAFETY: AVX2 feature gate checked by caller via is_x86_feature_detected
        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = row_nnz / 8;
        for c in 0..chunks {
            let base = start + c * 8;
            unsafe {
                let idx = _mm256_loadu_si256(col_indices[base..].as_ptr().cast());
                let v = _mm256_loadu_ps(values[base..].as_ptr());
                let x_gathered = _mm256_i32gather_ps::<4>(x.as_ptr(), idx);
                acc = _mm256_fmadd_ps(v, x_gathered, acc);
            }
        }

        // Horizontal sum of acc
        let hi = _mm256_extractf128_ps::<1>(acc);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        let mut row_sum = _mm_cvtss_f32(sums2);

        // Scalar tail for remaining elements
        for idx in (start + chunks * 8)..end {
            unsafe {
                let j = *col_indices.get_unchecked(idx) as usize;
                row_sum += *values.get_unchecked(idx) * *x.get_unchecked(j);
            }
        }

        unsafe {
            *y.get_unchecked_mut(i) = alpha * row_sum + beta * *y.get_unchecked(i);
        }
    }
}

/// Scalar SpMM reference implementation.
fn spmm_csr_scalar(
    a: &CsrMatrix<f32>,
    alpha: f32,
    b: &[f32],
    b_cols: usize,
    beta: f32,
    c: &mut [f32],
) {
    let offsets = a.offsets();
    let col_indices = a.col_indices();
    let values = a.values();

    for i in 0..a.rows() {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        // Scale existing C values by beta
        for k in 0..b_cols {
            c[i * b_cols + k] *= beta;
        }

        // Accumulate A[i,:] * B
        for idx in start..end {
            let j = col_indices[idx] as usize;
            let a_val = alpha * values[idx];
            for k in 0..b_cols {
                c[i * b_cols + k] += a_val * b[j * b_cols + k];
            }
        }
    }
}
