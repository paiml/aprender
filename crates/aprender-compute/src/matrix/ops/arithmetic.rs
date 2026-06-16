//! Matrix arithmetic operations
//!
//! This module provides matrix multiplication and related operations:
//! - `matmul()` - Standard matrix multiplication with SIMD optimization
//! - `batched_matmul()` - Batched 3D tensor multiplication
//! - `batched_matmul_4d()` - 4D tensor multiplication for attention
//!
//! ## Domain Separation (PMAT-018)
//!
//! Arithmetic operations (multiplication, addition) are separate from storage
//! operations (allocation, indexing). This allows optimizing compute kernels
//! independently of memory layout decisions.
//!
//! ## Performance Hierarchy
//!
//! 1. GPU for large matrices (≥500×500) - 2-10x speedup
//! 2. BLIS/SIMD for medium-large matrices (>64×64) - 2-8x speedup
//! 3. Naive for small matrices - lowest overhead

use crate::TruenoError;

#[cfg(feature = "tracing")]
use tracing::instrument;

use super::super::Matrix;

impl Matrix<f32> {
    /// Matrix multiplication (matmul)
    ///
    /// Computes `C = A × B` where A is `m×n`, B is `n×p`, and C is `m×p`.
    ///
    /// # Arguments
    ///
    /// * `other` - The matrix to multiply with (right operand)
    ///
    /// # Returns
    ///
    /// A new matrix containing the result of matrix multiplication
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if matrix dimensions are incompatible
    /// (i.e., `self.cols != other.rows`)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    /// let c = a.matmul(&b).unwrap();
    ///
    /// // [[1, 2],   [[5, 6],   [[19, 22],
    /// //  [3, 4]] ×  [7, 8]] =  [43, 50]]
    /// assert_eq!(c.get(0, 0), Some(&19.0));
    /// assert_eq!(c.get(0, 1), Some(&22.0));
    /// assert_eq!(c.get(1, 0), Some(&43.0));
    /// assert_eq!(c.get(1, 1), Some(&50.0));
    /// ```
    // =========================================================================
    // HOT PATH - PERFORMANCE CRITICAL
    // =========================================================================
    // Core matrix operation used in neural network forward passes.
    // Changes to inner loops REQUIRE benchmark verification: make bench-check
    // =========================================================================
    #[cfg_attr(feature = "tracing", instrument(skip(self, other), fields(dims = %format!("{}x{} @ {}x{}", self.rows, self.cols, other.rows, other.cols))))]
    pub fn matmul(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        if self.cols != other.rows {
            return Err(TruenoError::InvalidInput(format!(
                "Matrix dimension mismatch for multiplication: {}×{} × {}×{} (inner dimensions {} and {} must match)",
                self.rows, self.cols, other.rows, other.cols, self.cols, other.rows
            )));
        }

        // Fast path for vector-matrix multiply (rows=1)
        if self.rows == 1 {
            return self.matmul_vector_matrix(other);
        }

        // NOTE: zeros required — BLIS GEMM accumulates (c += A*B) via load_c_tile.
        let mut result = Matrix::zeros_with_backend(self.rows, other.cols, self.backend);

        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = 500;
        const SIMD_THRESHOLD: usize = 64;

        // Try GPU first for very large matrices
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.rows >= GPU_THRESHOLD
                && self.cols >= GPU_THRESHOLD
                && other.cols >= GPU_THRESHOLD
            {
                if let Ok(gpu_result) = self.matmul_gpu(other) {
                    return Ok(gpu_result);
                }
            }
        }

        // Use SIMD for medium-large matrices
        if self.rows >= SIMD_THRESHOLD
            || self.cols >= SIMD_THRESHOLD
            || other.cols >= SIMD_THRESHOLD
        {
            #[cfg(target_arch = "wasm32")]
            {
                self.matmul_wasm_tiled(other, &mut result)?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                crate::blis::parallel::gemm_blis_parallel(
                    self.rows,
                    other.cols,
                    self.cols,
                    &self.data,
                    &other.data,
                    &mut result.data,
                )?;
            }
        } else {
            self.matmul_naive(other, &mut result)?;
        }

        Ok(result)
    }

    /// Batched matrix multiplication for 3D tensors.
    ///
    /// Computes `[batch, m, k] @ [batch, k, n] -> [batch, m, n]` using SIMD for each batch.
    #[cfg_attr(feature = "tracing", instrument(skip(a_data, b_data), fields(batch, m, k, n)))]
    pub fn batched_matmul(
        a_data: &[f32],
        b_data: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, TruenoError> {
        let a_stride = m * k;
        let b_stride = k * n;
        let out_stride = m * n;

        if a_data.len() != batch * a_stride {
            return Err(TruenoError::InvalidInput(format!(
                "A data size mismatch: expected {} ({}×{}×{}), got {}",
                batch * a_stride,
                batch,
                m,
                k,
                a_data.len()
            )));
        }
        if b_data.len() != batch * b_stride {
            return Err(TruenoError::InvalidInput(format!(
                "B data size mismatch: expected {} ({}×{}×{}), got {}",
                batch * b_stride,
                batch,
                k,
                n,
                b_data.len()
            )));
        }

        // NOTE: zeros required — gemm_blis accumulates (c += A*B) via load_c_tile.
        let mut output = vec![0.0f32; batch * out_stride];

        // KAIZEN-039: Call gemm_blis directly on sub-slices instead of
        // Matrix::from_slice (which copies data). Eliminates 2 × batch Vec
        // allocations per call (e.g., 64 copies for 32-head attention).
        for ba in 0..batch {
            let a_offset = ba * a_stride;
            let b_offset = ba * b_stride;
            let out_offset = ba * out_stride;

            let a_slice = &a_data[a_offset..a_offset + a_stride];
            let b_slice = &b_data[b_offset..b_offset + b_stride];
            let c_slice = &mut output[out_offset..out_offset + out_stride];

            #[cfg(not(target_arch = "wasm32"))]
            {
                crate::blis::gemm_blis(m, n, k, a_slice, b_slice, c_slice, None)?;
            }
            #[cfg(target_arch = "wasm32")]
            {
                let a_mat = Matrix::from_slice(m, k, a_slice)?;
                let b_mat = Matrix::from_slice(k, n, b_slice)?;
                let result = a_mat.matmul(&b_mat)?;
                c_slice.copy_from_slice(result.as_slice());
            }
        }

        Ok(output)
    }

    /// Batched matrix multiplication for 4D tensors (attention pattern).
    ///
    /// Computes `[batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]`
    #[cfg_attr(
        feature = "tracing",
        instrument(skip(a_data, b_data), fields(batch, heads, m, k, n))
    )]
    pub fn batched_matmul_4d(
        a_data: &[f32],
        b_data: &[f32],
        batch: usize,
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, TruenoError> {
        let a_head_stride = m * k;
        let b_head_stride = k * n;
        let out_head_stride = m * n;
        let total_heads = batch * heads;

        let expected_a = total_heads * a_head_stride;
        let expected_b = total_heads * b_head_stride;
        if a_data.len() != expected_a {
            return Err(TruenoError::InvalidInput(format!(
                "A data size mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected_a,
                batch,
                heads,
                m,
                k,
                a_data.len()
            )));
        }
        if b_data.len() != expected_b {
            return Err(TruenoError::InvalidInput(format!(
                "B data size mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected_b,
                batch,
                heads,
                k,
                n,
                b_data.len()
            )));
        }

        // NOTE: zeros required — gemm_blis accumulates (c += A*B) via load_c_tile.
        let mut output = vec![0.0f32; total_heads * out_head_stride];

        // KAIZEN-039: Call gemm_blis directly — eliminates 2 × total_heads
        // Vec copies per call (e.g., 64 copies for batch=1, heads=32).
        for bh in 0..total_heads {
            let a_offset = bh * a_head_stride;
            let b_offset = bh * b_head_stride;
            let out_offset = bh * out_head_stride;

            let a_slice = &a_data[a_offset..a_offset + a_head_stride];
            let b_slice = &b_data[b_offset..b_offset + b_head_stride];
            let c_slice = &mut output[out_offset..out_offset + out_head_stride];

            #[cfg(not(target_arch = "wasm32"))]
            {
                crate::blis::gemm_blis(m, n, k, a_slice, b_slice, c_slice, None)?;
            }
            #[cfg(target_arch = "wasm32")]
            {
                let a_mat = Matrix::from_slice(m, k, a_slice)?;
                let b_mat = Matrix::from_slice(k, n, b_slice)?;
                let result = a_mat.matmul(&b_mat)?;
                c_slice.copy_from_slice(result.as_slice());
            }
        }

        Ok(output)
    }

    /// Fast path for vector-matrix multiplication (1×K @ K×N → 1×N)
    ///
    /// Dispatches to AVX2 SIMD GEMV kernel when available (explicit VFMADD
    /// with 4-way K-unrolling), falls back to scalar 4-way axpy.
    /// Bypasses BLIS packing which dominates for M=1.
    ///
    /// Contract: matvec-kernel-v1, equation "matvec"
    #[cfg_attr(feature = "tracing", instrument(skip(self, other), fields(k = self.cols, n = other.cols)))]
    fn matmul_vector_matrix(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        debug_assert_eq!(self.rows, 1);

        let k = self.cols;
        let n = other.cols;
        // NOTE: zeros required — gemv accumulates (c[j] += a[k]*b[k*n+j]).
        let mut c = vec![0.0f32; n];

        crate::blis::gemv::gemv(k, n, &self.data, &other.data, &mut c);

        Matrix::from_vec(1, n, c)
    }

    /// Small-matrix matmul (baseline for dims < 64, below the SIMD/GPU threshold).
    ///
    /// Cache-friendly `ikj` ordering: the innermost loop is a contiguous AXPY
    /// (`c_row[j] += a_ik * b_row[j]`) over row-major memory, which LLVM
    /// auto-vectorizes. The previous `ijk` form indexed `b[kk * n + j]` with a
    /// column stride in the hot loop, defeating both the cache and SIMD; it
    /// measured ~4.7x slower than this AXPY form on 32x32–63x63 matrices — the
    /// shapes a small-MLP autograd training step hits (the matmul kernel that
    /// powers the Pillar-1 LinearRegression beat already uses this ordering in
    /// `aprender::primitives::Matrix::matmul`). `result.data` is pre-zeroed by
    /// the caller (`Matrix::zeros_with_backend`), so the `+=` accumulation
    /// starts from 0.
    fn matmul_naive(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let a = &self.data;
        let b = &other.data;
        let c = &mut result.data;

        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            let c_row = &mut c[i * n..i * n + n];
            for kk in 0..k {
                let a_ik = a_row[kk];
                let b_row = &b[kk * n..kk * n + n];
                for (cj, &bj) in c_row.iter_mut().zip(b_row.iter()) {
                    *cj += a_ik * bj;
                }
            }
        }
        Ok(())
    }

    /// WASM-optimized tiled matrix multiplication
    #[allow(dead_code)]
    fn matmul_wasm_tiled(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        for i in 0..m {
            let a_row_start = i * k;
            let result_row_start = i * n;

            let simd_width = 8;
            let n_simd = (n / simd_width) * simd_width;

            #[allow(clippy::needless_range_loop)]
            for j0 in (0..n_simd).step_by(simd_width) {
                let mut acc = [0.0f32; 8];

                for kk in 0..k {
                    let a_val = self.data[a_row_start + kk];
                    let b_row_start = kk * n + j0;

                    for jj in 0..simd_width {
                        acc[jj] += a_val * other.data[b_row_start + jj];
                    }
                }

                for jj in 0..simd_width {
                    result.data[result_row_start + j0 + jj] = acc[jj];
                }
            }

            for j in n_simd..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += self.data[a_row_start + kk] * other.data[kk * n + j];
                }
                result.data[result_row_start + j] = sum;
            }
        }

        Ok(())
    }

    /// GPU-accelerated matrix multiplication
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn matmul_gpu(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        // Track 1 (CGP-DBUF): Try cuBLAS first when cuda feature is enabled.
        // cuBLAS at 105-150 TFLOP/s vs wgpu shader at ~5 TFLOP/s — 20-30× faster.
        #[cfg(feature = "cuda")]
        {
            if let Ok(result) = self.matmul_cublas(other) {
                return Ok(result);
            }
            // cuBLAS unavailable (no GPU, driver error) — fall through to wgpu
        }

        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            return Err(TruenoError::InvalidInput("GPU not available".to_string()));
        }

        let mut gpu = GpuBackend::new();
        let result_data = gpu
            .matmul(&self.data, &other.data, self.rows, self.cols, other.cols)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU matmul failed: {}", e)))?;

        let mut result = Matrix::zeros(self.rows, other.cols);
        result.data = result_data;

        Ok(result)
    }

    /// cuBLAS FP32 GEMM via trueno-gpu's own FFI bindings (OWN THE STACK).
    /// Uses cublasGemmEx with CUBLAS_COMPUTE_32F for numerical safety.
    #[cfg(feature = "cuda")]
    fn matmul_cublas(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        use trueno_gpu::driver::{CublasHandle, CudaContext, CudaStream, GemmOp, GpuBuffer};

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let ctx = CudaContext::new(0)
            .map_err(|e| TruenoError::InvalidInput(format!("CUDA init: {e}")))?;
        let stream = CudaStream::new(&ctx)
            .map_err(|e| TruenoError::InvalidInput(format!("CUDA stream: {e}")))?;
        let handle = CublasHandle::new(&ctx)
            .map_err(|e| TruenoError::InvalidInput(format!("cuBLAS init: {e}")))?;
        handle
            .set_stream(&stream)
            .map_err(|e| TruenoError::InvalidInput(format!("cuBLAS stream: {e}")))?;

        let a_buf = GpuBuffer::from_host(&ctx, &self.data)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU alloc A: {e}")))?;
        let b_buf = GpuBuffer::from_host(&ctx, &other.data)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU alloc B: {e}")))?;
        let c_data = vec![0.0f32; m * n];
        let c_buf = GpuBuffer::from_host(&ctx, &c_data)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU alloc C: {e}")))?;

        handle
            .gemm_f32_row_major(
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_buf.as_ptr(),
                b_buf.as_ptr(),
                0.0,
                c_buf.as_ptr(),
            )
            .map_err(|e| TruenoError::InvalidInput(format!("cuBLAS GEMM: {e}")))?;

        stream.synchronize().map_err(|e| TruenoError::InvalidInput(format!("CUDA sync: {e}")))?;

        let mut result_data = vec![0.0f32; m * n];
        c_buf
            .copy_to_host(&mut result_data)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU readback: {e}")))?;

        Ok(Matrix { rows: m, cols: n, data: result_data, backend: self.backend })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.get(0, 0), Some(&19.0));
        assert_eq!(c.get(0, 1), Some(&22.0));
        assert_eq!(c.get(1, 0), Some(&43.0));
        assert_eq!(c.get(1, 1), Some(&50.0));
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Matrix::from_vec(2, 3, vec![1.0; 6]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matmul_identity() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let i = Matrix::identity(2);
        let result = a.matmul(&i).unwrap();
        assert_eq!(result.as_slice(), a.as_slice());
    }

    #[test]
    fn test_batched_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 batches of 2×2
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 2 identity matrices
        let result = Matrix::batched_matmul(&a, &b, 2, 2, 2, 2).unwrap();
        assert_eq!(result, a); // A × I = A
    }

    #[test]
    fn test_batched_matmul_a_size_mismatch() {
        let a = vec![1.0, 2.0, 3.0]; // Wrong size: should be 2*2*2=8
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let result = Matrix::batched_matmul(&a, &b, 2, 2, 2, 2);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_batched_matmul_b_size_mismatch() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.0]; // Wrong size: should be 2*2*2=8
        let result = Matrix::batched_matmul(&a, &b, 2, 2, 2, 2);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_batched_matmul_single_batch() {
        // Single batch 3x2 @ 2x4 = 3x4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let result = Matrix::batched_matmul(&a, &b, 1, 3, 2, 4).unwrap();
        assert_eq!(result.len(), 12); // 3x4
    }

    #[test]
    fn test_batched_matmul_4d_basic() {
        // batch=1, heads=1, m=2, k=2, n=2
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let result = Matrix::batched_matmul_4d(&a, &b, 1, 1, 2, 2, 2).unwrap();
        assert_eq!(result, a);
    }

    #[test]
    fn test_batched_matmul_4d_a_size_mismatch() {
        let a = vec![1.0]; // Wrong: should be 2*2*3*4=48
        let b: Vec<f32> = (0..80).map(|x| x as f32 * 0.1).collect();
        let result = Matrix::batched_matmul_4d(&a, &b, 2, 2, 3, 4, 5);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_batched_matmul_4d_b_size_mismatch() {
        let a: Vec<f32> = (0..48).map(|x| x as f32 * 0.1).collect();
        let b = vec![1.0]; // Wrong: should be 2*2*4*5=80
        let result = Matrix::batched_matmul_4d(&a, &b, 2, 2, 3, 4, 5);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_batched_matmul_4d_multi_head() {
        // batch=1, heads=4, m=2, k=2, n=2 (like attention heads)
        let total = 4 * 2 * 2; // 16 elements for A
        let a: Vec<f32> = (0..total).map(|_| 1.0).collect();
        let b: Vec<f32> = (0..total).map(|_| 1.0).collect();
        let result = Matrix::batched_matmul_4d(&a, &b, 1, 4, 2, 2, 2).unwrap();
        assert_eq!(result.len(), total);
        // Each element should be 2.0 (dot product of two 1.0 vectors of length 2)
        for val in &result {
            assert!((*val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_matmul_vector_matrix_path() {
        // 1×K @ K×N triggers the vector-matrix fast path
        let a = Matrix::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(
            4,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.rows(), 1);
        assert_eq!(result.cols(), 3);
        // [1*1+2*0+3*0+4*1, 1*0+2*1+3*0+4*1, 1*0+2*0+3*1+4*1] = [5, 6, 7]
        assert!((result.get(0, 0).unwrap() - 5.0).abs() < 1e-5);
        assert!((result.get(0, 1).unwrap() - 6.0).abs() < 1e-5);
        assert!((result.get(0, 2).unwrap() - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_vector_matrix_with_zeros() {
        // Test that zero elements in the vector skip computation
        let a = Matrix::from_vec(1, 3, vec![0.0, 2.0, 0.0]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![100.0, 200.0, 3.0, 4.0, 500.0, 600.0]).unwrap();
        let result = a.matmul(&b).unwrap();
        // Only the second row of B contributes: [2*3, 2*4] = [6, 8]
        assert!((result.get(0, 0).unwrap() - 6.0).abs() < 1e-5);
        assert!((result.get(0, 1).unwrap() - 8.0).abs() < 1e-5);
    }

    // =========================================================================
    // matmul_wasm_tiled tests
    // =========================================================================
    // These tests call the private WASM-tiled matmul directly (not behind
    // #[cfg(target_arch = "wasm32")]) to achieve coverage on non-WASM hosts.

    #[test]
    fn test_matmul_wasm_tiled_small_no_simd() {
        // n=3 < simd_width(8), so only the remainder path executes.
        // 2x4 @ 4x3 = 2x3
        let a = Matrix::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let b = Matrix::from_vec(
            4,
            3,
            vec![1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        )
        .unwrap();
        let mut result = Matrix::zeros(2, 3);
        a.matmul_wasm_tiled(&b, &mut result).unwrap();

        // Row 0: [1*1+2*0+3*2+4*0, 1*0+2*1+3*0+4*2, 1*2+2*0+3*1+4*0] = [7, 10, 5]
        assert!((result.get(0, 0).unwrap() - 7.0).abs() < 1e-5);
        assert!((result.get(0, 1).unwrap() - 10.0).abs() < 1e-5);
        assert!((result.get(0, 2).unwrap() - 5.0).abs() < 1e-5);

        // Row 1: [5*1+6*0+7*2+8*0, 5*0+6*1+7*0+8*2, 5*2+6*0+7*1+8*0] = [19, 22, 17]
        assert!((result.get(1, 0).unwrap() - 19.0).abs() < 1e-5);
        assert!((result.get(1, 1).unwrap() - 22.0).abs() < 1e-5);
        assert!((result.get(1, 2).unwrap() - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_wasm_tiled_exact_simd_width() {
        // n=8 exactly equals simd_width, so the SIMD path handles all columns
        // and the remainder path has zero iterations.
        // 2x3 @ 3x8 = 2x8
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b_data: Vec<f32> = (1..=24).map(|x| x as f32).collect(); // 3x8
        let b = Matrix::from_vec(3, 8, b_data).unwrap();
        let mut result = Matrix::zeros(2, 8);
        a.matmul_wasm_tiled(&b, &mut result).unwrap();

        // Verify against naive matmul
        let mut expected = Matrix::zeros(2, 8);
        a.matmul_naive(&b, &mut expected).unwrap();
        for i in 0..2 {
            for j in 0..8 {
                assert!(
                    (result.get(i, j).unwrap() - expected.get(i, j).unwrap()).abs() < 1e-4,
                    "Mismatch at ({}, {}): wasm_tiled={}, naive={}",
                    i,
                    j,
                    result.get(i, j).unwrap(),
                    expected.get(i, j).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_matmul_wasm_tiled_simd_plus_remainder() {
        // n=11 => n_simd=8 (SIMD path handles columns 0..8),
        // remainder path handles columns 8..11. Both paths exercise.
        // 3x4 @ 4x11 = 3x11
        let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a = Matrix::from_vec(3, 4, a_data).unwrap();
        let b_data: Vec<f32> = (1..=44).map(|x| x as f32 * 0.1).collect();
        let b = Matrix::from_vec(4, 11, b_data).unwrap();
        let mut result = Matrix::zeros(3, 11);
        a.matmul_wasm_tiled(&b, &mut result).unwrap();

        // Verify against naive
        let mut expected = Matrix::zeros(3, 11);
        a.matmul_naive(&b, &mut expected).unwrap();
        for i in 0..3 {
            for j in 0..11 {
                assert!(
                    (result.get(i, j).unwrap() - expected.get(i, j).unwrap()).abs() < 1e-3,
                    "Mismatch at ({}, {}): wasm_tiled={}, naive={}",
                    i,
                    j,
                    result.get(i, j).unwrap(),
                    expected.get(i, j).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_matmul_wasm_tiled_multiple_simd_blocks() {
        // n=16 => two full SIMD blocks (0..8 and 8..16), no remainder.
        // 2x2 @ 2x16 = 2x16
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b_data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let b = Matrix::from_vec(2, 16, b_data).unwrap();
        let mut result = Matrix::zeros(2, 16);
        a.matmul_wasm_tiled(&b, &mut result).unwrap();

        let mut expected = Matrix::zeros(2, 16);
        a.matmul_naive(&b, &mut expected).unwrap();
        for i in 0..2 {
            for j in 0..16 {
                assert!(
                    (result.get(i, j).unwrap() - expected.get(i, j).unwrap()).abs() < 1e-3,
                    "Mismatch at ({}, {})",
                    i,
                    j,
                );
            }
        }
    }

    #[test]
    fn test_matmul_wasm_tiled_single_row() {
        // m=1, n=10 => SIMD block 0..8 + remainder 8..10
        // 1x5 @ 5x10 = 1x10
        let a = Matrix::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let b_data: Vec<f32> = (1..=50).map(|x| x as f32 * 0.1).collect();
        let b = Matrix::from_vec(5, 10, b_data).unwrap();
        let mut result = Matrix::zeros(1, 10);
        a.matmul_wasm_tiled(&b, &mut result).unwrap();

        let mut expected = Matrix::zeros(1, 10);
        a.matmul_naive(&b, &mut expected).unwrap();
        for j in 0..10 {
            assert!(
                (result.get(0, j).unwrap() - expected.get(0, j).unwrap()).abs() < 1e-3,
                "Mismatch at col {}: wasm_tiled={}, naive={}",
                j,
                result.get(0, j).unwrap(),
                expected.get(0, j).unwrap()
            );
        }
    }

    #[test]
    fn test_matmul_wasm_tiled_identity() {
        // Multiplying by identity should return the original matrix.
        // 4x4 identity, n=4 < 8 so only remainder path.
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let identity = Matrix::identity(4);
        let mut result = Matrix::zeros(4, 4);
        a.matmul_wasm_tiled(&identity, &mut result).unwrap();

        assert_eq!(result.as_slice(), a.as_slice());
    }

    #[test]
    fn test_matmul_wasm_tiled_large_mixed() {
        // Larger test: 5x10 @ 10x19 = 5x19
        // n=19 => n_simd=16, remainder 16..19
        // Exercises multiple SIMD blocks (0..8, 8..16) plus remainder (16..19).
        let a_data: Vec<f32> = (0..50).map(|x| (x as f32) * 0.1).collect();
        let a = Matrix::from_vec(5, 10, a_data).unwrap();
        let b_data: Vec<f32> = (0..190).map(|x| (x as f32) * 0.01).collect();
        let b = Matrix::from_vec(10, 19, b_data).unwrap();
        let mut result = Matrix::zeros(5, 19);
        a.matmul_wasm_tiled(&b, &mut result).unwrap();

        let mut expected = Matrix::zeros(5, 19);
        a.matmul_naive(&b, &mut expected).unwrap();
        for i in 0..5 {
            for j in 0..19 {
                assert!(
                    (result.get(i, j).unwrap() - expected.get(i, j).unwrap()).abs() < 1e-2,
                    "Mismatch at ({}, {}): wasm_tiled={}, naive={}",
                    i,
                    j,
                    result.get(i, j).unwrap(),
                    expected.get(i, j).unwrap()
                );
            }
        }
    }

    // =========================================================================
    // FALSIFY-MM: matmul-kernel-v1.yaml contract (trueno Matrix::matmul)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: trueno had 10+ matmul tests but zero FALSIFY-MM-* tests
    //   Why 2: unit tests verify known products, not mathematical invariants
    //   Why 3: no mapping from matmul-kernel-v1.yaml to trueno test names
    //   Why 4: trueno predates the provable-contracts YAML convention
    //   Why 5: matmul was "obviously correct" (standard GEMM)
    //
    // References:
    //   - provable-contracts/contracts/matmul-kernel-v1.yaml
    // =========================================================================

    /// FALSIFY-MM-001: Shape correctness — matmul(A[m,p], B[p,n]) = [m,n]
    #[test]
    fn falsify_mm_001_shape_correctness() {
        for &(m, p, n) in &[(1, 1, 1), (2, 3, 4), (16, 32, 8), (1, 100, 1), (64, 1, 64)] {
            let a = Matrix::from_vec(m, p, vec![1.0; m * p]).unwrap();
            let b = Matrix::from_vec(p, n, vec![1.0; p * n]).unwrap();
            let c = a.matmul(&b).unwrap();
            assert_eq!(
                (c.rows(), c.cols()),
                (m, n),
                "FALSIFIED MM-001: matmul([{m},{p}], [{p},{n}]) shape = [{},{}], expected [{m},{n}]",
                c.rows(),
                c.cols()
            );
        }
    }

    /// FALSIFY-MM-005: Identity matrix — matmul(A, I) = A
    #[test]
    fn falsify_mm_005_identity_matrix() {
        let a = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let eye =
            Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

        let ai = a.matmul(&eye).unwrap();
        let ia = eye.matmul(&a).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = a.get(i, j).unwrap();
                assert!(
                    (*ai.get(i, j).unwrap() - expected).abs() < 1e-6,
                    "FALSIFIED MM-005: (A*I)[{i},{j}] = {}, expected {expected}",
                    ai.get(i, j).unwrap()
                );
                assert!(
                    (*ia.get(i, j).unwrap() - expected).abs() < 1e-6,
                    "FALSIFIED MM-005: (I*A)[{i},{j}] = {}, expected {expected}",
                    ia.get(i, j).unwrap()
                );
            }
        }
    }

    /// FALSIFY-MM-002: Numerical accuracy — known product verified
    #[test]
    fn falsify_mm_002_numerical_accuracy() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        let expected = [19.0, 22.0, 43.0, 50.0];
        for (i, &exp) in expected.iter().enumerate() {
            let row = i / 2;
            let col = i % 2;
            let val = *c.get(row, col).unwrap();
            assert!(
                (val - exp).abs() < 1e-5,
                "FALSIFIED MM-002: C[{row},{col}] = {val}, expected {exp}"
            );
        }
    }

    /// FALSIFY-MM-002b: matmul(zeros, B) = zeros
    #[test]
    fn falsify_mm_002b_zero_annihilation() {
        let zero = Matrix::from_vec(3, 4, vec![0.0; 12]).unwrap();
        let b = Matrix::from_vec(4, 2, vec![1.0; 8]).unwrap();
        let c = zero.matmul(&b).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                let val = *c.get(i, j).unwrap();
                assert!(
                    val.abs() < 1e-10,
                    "FALSIFIED MM-002b: zeros*B [{i},{j}] = {val}, expected 0"
                );
            }
        }
    }
}

#[cfg(test)]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod gpu_tests {
    use super::*;

    /// Test matmul_gpu via public API with matrices large enough to exceed
    /// GPU_THRESHOLD (all dimensions >= 500).
    /// Uses identity multiplication: A * I = A.
    #[test]
    fn test_matmul_gpu_identity() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_matmul_gpu_identity");
            return;
        }

        let n = 500; // Meets GPU_THRESHOLD for all three dimensions

        // Create a simple test matrix: A[i,j] = (i*n + j) mod 100 * 0.01
        let a_data: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32 * 0.01).collect();

        // Identity matrix
        let mut i_data = vec![0.0f32; n * n];
        for i in 0..n {
            i_data[i * n + i] = 1.0;
        }

        let a = Matrix::from_vec(n, n, a_data.clone()).expect("valid matrix A");
        let identity = Matrix::from_vec(n, n, i_data).expect("valid identity matrix");

        let result = a.matmul(&identity).expect("matmul should succeed");

        assert_eq!(result.rows(), n);
        assert_eq!(result.cols(), n);

        // A * I = A: sample verification (check corners and center)
        let check_indices = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1), (n / 2, n / 2)];
        for &(r, c) in &check_indices {
            let expected = a_data[r * n + c];
            let actual = *result.get(r, c).unwrap();
            assert!(
                (actual - expected).abs() < 1e-2,
                "A*I mismatch at ({},{}): gpu={}, expected={}",
                r,
                c,
                actual,
                expected
            );
        }
    }

    /// Test matmul_gpu with all-ones matrices: result should be all-K.
    #[test]
    fn test_matmul_gpu_ones() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_matmul_gpu_ones");
            return;
        }

        let m = 500;
        let k = 500;
        let n = 500;

        let a = Matrix::from_vec(m, k, vec![1.0f32; m * k]).expect("valid matrix A");
        let b = Matrix::from_vec(k, n, vec![1.0f32; k * n]).expect("valid matrix B");

        let result = a.matmul(&b).expect("matmul should succeed");

        assert_eq!(result.rows(), m);
        assert_eq!(result.cols(), n);

        // Each element of C should be K (dot product of K ones with K ones)
        let expected = k as f32;
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (result.get(i, j).unwrap() - expected).abs() < 1.0,
                    "C[{},{}] = {}, expected {}",
                    i,
                    j,
                    result.get(i, j).unwrap(),
                    expected
                );
            }
        }
    }

    /// Test matmul_gpu directly via the private helper method.
    #[test]
    fn test_matmul_gpu_direct() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test_matmul_gpu_direct");
            return;
        }

        // Small matrix for direct private method test
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid A");
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("valid B");

        let result = a.matmul_gpu(&b).expect("matmul_gpu should succeed");

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);

        // C = A * B
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert!(
            (result.get(0, 0).unwrap() - 58.0).abs() < 1e-2,
            "Expected 58.0, got {}",
            result.get(0, 0).unwrap()
        );
        assert!(
            (result.get(0, 1).unwrap() - 64.0).abs() < 1e-2,
            "Expected 64.0, got {}",
            result.get(0, 1).unwrap()
        );
        assert!(
            (result.get(1, 0).unwrap() - 139.0).abs() < 1e-2,
            "Expected 139.0, got {}",
            result.get(1, 0).unwrap()
        );
        assert!(
            (result.get(1, 1).unwrap() - 154.0).abs() < 1e-2,
            "Expected 154.0, got {}",
            result.get(1, 1).unwrap()
        );
    }

    /// Test matmul_gpu returns error when GPU is unavailable.
    #[test]
    fn test_matmul_gpu_not_available_path() {
        use crate::backends::gpu::GpuBackend;

        // This test verifies the GpuBackend::is_available() check in matmul_gpu
        // If GPU IS available, the function should succeed; test the full path
        if !GpuBackend::is_available() {
            // If GPU is not available, matmul_gpu should return an error
            let a = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
            let b = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
            let result = a.matmul_gpu(&b);
            assert!(result.is_err(), "matmul_gpu should fail without GPU");
        }
    }
}
