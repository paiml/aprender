//! Safe cuBLAS Wrapper
//!
//! RAII handle with buffer verification and FP32 accumulation enforcement.
//!
//! # Contract
//!
//! `cublas-gemm-v1.yaml` — ALB-075
//!
//! - CUBLAS-INV-002: Buffer sizes verified before every cublasGemmEx
//! - CUBLAS-INV-003: Handle lifecycle is RAII (create in new, destroy in Drop)
//! - CUBLAS-INV-008: FP32 accumulation always enforced (CUBLAS_COMPUTE_32F)
//!
//! # Design
//!
//! - One CublasHandle per CudaContext
//! - set_stream() called ONCE per training step, not per GEMM
//!   (555 calls/step would add measurable overhead — contract invariant)
//! - gemm_f16() takes GpuBuffer references and verifies sizes algebraically

use std::ptr;

use super::cublas_sys::*;
use super::stream::CudaStream;
use crate::driver::context::CudaContext;
use crate::driver::sys::CUdeviceptr;
use crate::GpuError;

// ============================================================================
// cuBLAS Transpose Operation
// ============================================================================

/// Transpose operation for cuBLAS GEMM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmOp {
    /// No transpose (column-major: use as-is)
    NoTrans,
    /// Transpose
    Trans,
}

impl GemmOp {
    fn to_cublas(self) -> CublasOperation {
        match self {
            GemmOp::NoTrans => CUBLAS_OP_N,
            GemmOp::Trans => CUBLAS_OP_T,
        }
    }
}

// ============================================================================
// cuBLAS Handle (RAII)
// ============================================================================

/// Safe cuBLAS handle with RAII lifecycle
///
/// # Contract (cublas-gemm-v1.yaml)
///
/// - Created in `new()` via cublasCreate_v2
/// - Destroyed in `Drop` via cublasDestroy_v2
/// - Stream set once per step via `set_stream()`
/// - Math mode: SIMD (default) or TF32 tensor cores (opt-in)
pub struct CublasHandle {
    handle: super::cublas_sys::CublasHandle,
    /// Whether tensor cores are enabled (TF32 mode)
    pub tensor_cores: bool,
}

// SAFETY: cuBLAS handles are thread-safe within a CUDA context.
// Sync is safe because CublasHandle is only accessed via &mut self on CudaExecutor
// (behind RwLock write guard), so no concurrent access occurs.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    /// Create a new cuBLAS handle
    ///
    /// Enables tensor core math mode (CUBLAS_TENSOR_OP_MATH) automatically.
    ///
    /// # Errors
    ///
    /// Returns error if cuBLAS library is not available or handle creation fails.
    pub fn new(_ctx: &CudaContext) -> Result<Self, GpuError> {
        let driver = get_cublas_driver()?;

        let mut handle: super::cublas_sys::CublasHandle = ptr::null_mut();
        let result = unsafe { (driver.cublasCreate_v2)(&mut handle) };
        CublasDriver::check(result)
            .map_err(|e| GpuError::CudaDriver(format!("cublasCreate_v2: {e}"), 0))?;

        // ALB-076: Use CUBLAS_DEFAULT_MATH (no tensor cores for FP32 GEMMs).
        //
        // Root cause: CUBLAS_TF32_TENSOR_OP_MATH + CUBLAS_GEMM_DEFAULT_TENSOR_OP
        // produce NaN for transposed backward GEMMs (Trans/NoTrans, NoTrans/Trans)
        // when input gradient magnitudes reach ~1e5 (around block 18 of 24-layer
        // backward). Forward NoTrans/NoTrans is unaffected.
        //
        // Five Whys analysis:
        // 1. Why NaN weights? → optimizer reads NaN gradients
        // 2. Why NaN gradients? → cuBLAS backward_a/b output ALL NaN
        // 3. Why NaN output from valid inputs? → tensor core GEMM algorithm
        // 4. Why only backward? → backward uses Trans flag, forward doesn't
        // 5. Why only after ~5 blocks? → gradient magnification reaches ~1e5
        //
        // CUBLAS_DEFAULT_MATH disables tensor cores for FP32, yielding correct
        // results. cuBLAS SIMD GEMM is still 6-14x faster than hand-written PTX.
        let result = unsafe { (driver.cublasSetMathMode)(handle, CUBLAS_DEFAULT_MATH) };
        if result != CUBLAS_STATUS_SUCCESS {
            unsafe { (driver.cublasDestroy_v2)(handle) };
            return Err(GpuError::CudaDriver(
                format!("cublasSetMathMode: {}", cublas_status_string(result)),
                result,
            ));
        }

        Ok(Self { handle, tensor_cores: false })
    }

    /// Create a cuBLAS handle with TF32 tensor cores enabled.
    ///
    /// entrenar#318: ~41x faster than SIMD for fp32 GEMMs on sm_89.
    ///
    /// # Safety
    ///
    /// ALB-076: TF32 tensor cores produce NaN for transposed GEMMs (Trans flag)
    /// when gradient magnitudes reach ~1e5. ONLY safe for NoTrans/NoTrans (forward).
    /// Use the default handle (SIMD) for backward GEMMs with Trans flag.
    pub fn new_with_tensor_cores(_ctx: &CudaContext) -> Result<Self, GpuError> {
        let driver = get_cublas_driver()?;
        let mut handle: super::cublas_sys::CublasHandle = ptr::null_mut();
        let result = unsafe { (driver.cublasCreate_v2)(&mut handle) };
        CublasDriver::check(result)
            .map_err(|e| GpuError::CudaDriver(format!("cublasCreate_v2 (TC): {e}"), 0))?;

        let result = unsafe { (driver.cublasSetMathMode)(handle, CUBLAS_TF32_TENSOR_OP_MATH) };
        if result != CUBLAS_STATUS_SUCCESS {
            unsafe { (driver.cublasDestroy_v2)(handle) };
            return Err(GpuError::CudaDriver(
                format!("cublasSetMathMode TF32: {}", cublas_status_string(result)),
                result,
            ));
        }

        Ok(Self { handle, tensor_cores: true })
    }

    /// Bind this handle to a CUDA stream
    ///
    /// # Contract
    ///
    /// Call ONCE per training step, not per GEMM.
    /// 555 GEMMs/step × set_stream overhead = measurable cost.
    ///
    /// # Errors
    ///
    /// Returns error if stream binding fails.
    pub fn set_stream(&self, stream: &CudaStream) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;
        let result = unsafe { (driver.cublasSetStream_v2)(self.handle, stream.raw()) };
        CublasDriver::check(result)
            .map_err(|e| GpuError::CudaDriver(format!("cublasSetStream_v2: {e}"), 0))
    }

    /// PMAT-063: Pre-allocate cuBLAS workspace for CUDA graph capture.
    ///
    /// cuBLAS internally allocates workspace for fast algorithm selection.
    /// During CUDA graph capture, dynamic allocation is forbidden, causing
    /// cuBLAS to fall back to workspace-free algorithms (7x slower).
    ///
    /// Call this with a pre-allocated GPU buffer before graph capture
    /// to enable fast algorithms. Recommended size: 4-32 MB.
    ///
    /// # Errors
    ///
    /// Returns error if cublasSetWorkspace fails.
    pub fn set_workspace(&self, workspace_ptr: u64, workspace_size: usize) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;
        let result = unsafe {
            (driver.cublasSetWorkspace)(
                self.handle,
                workspace_ptr as *mut std::ffi::c_void,
                workspace_size,
            )
        };
        CublasDriver::check(result).map_err(|e| {
            GpuError::CudaDriver(format!("cublasSetWorkspace({workspace_size}): {e}"), 0)
        })
    }

    /// FP16 GEMM with FP32 accumulation via tensor cores
    ///
    /// Computes: C = alpha * op(A) * op(B) + beta * C
    ///
    /// Where A, B, C are FP16 (half precision) and accumulation is FP32.
    ///
    /// # Contract (cublas-gemm-v1.yaml)
    ///
    /// - CUBLAS-INV-002: Buffer sizes verified before cublasGemmEx
    /// - CUBLAS-INV-008: computeType is always CUBLAS_COMPUTE_32F
    /// - CUBLAS-EQ-001: max_abs_diff(C_cublas, C_ptx) < 1e-2
    ///
    /// # Arguments
    ///
    /// * `transa` - Operation on A
    /// * `transb` - Operation on B
    /// * `m` - Rows of op(A) and C
    /// * `n` - Columns of op(B) and C
    /// * `k` - Columns of op(A) / rows of op(B)
    /// * `alpha` - Scalar multiplier
    /// * `a_ptr` - Device pointer to A (FP16)
    /// * `lda` - Leading dimension of A
    /// * `b_ptr` - Device pointer to B (FP16)
    /// * `ldb` - Leading dimension of B
    /// * `beta` - Scalar for C accumulation
    /// * `c_ptr` - Device pointer to C (FP16, read-write)
    /// * `ldc` - Leading dimension of C
    ///
    /// # Buffer Size Contract
    ///
    /// The caller MUST ensure:
    /// - A buffer >= rows_a * lda * 2 bytes (FP16)
    /// - B buffer >= rows_b * ldb * 2 bytes (FP16)
    /// - C buffer >= m * ldc * 2 bytes (FP16)
    ///
    /// # Safety
    ///
    /// Device pointers must be valid and buffers must be correctly sized.
    /// This is marked safe because buffer verification is the caller's
    /// responsibility per Rule 2 (prove at kernel boundary).
    ///
    /// # Errors
    ///
    /// Returns error if cublasGemmEx fails.
    pub fn gemm_f16(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: i32,
        b_ptr: CUdeviceptr,
        ldb: i32,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;

        // Contract: FP32 accumulation always enforced (CUBLAS-INV-008)
        let compute_type = CUBLAS_COMPUTE_32F;

        let result = unsafe {
            (driver.cublasGemmEx)(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m,
                n,
                k,
                std::ptr::from_ref::<f32>(&alpha) as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                lda,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                ldb,
                std::ptr::from_ref::<f32>(&beta) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_16F,
                ldc,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };

        CublasDriver::check(result)
            .map_err(|e| GpuError::CudaDriver(format!("cublasGemmEx(m={m}, n={n}, k={k}): {e}"), 0))
    }

    /// FP16 inputs → FP32 output GEMM via tensor cores
    ///
    /// Computes: C = alpha * op(A) * op(B) + beta * C
    /// A, B are FP16; C is FP32. Accumulation is FP32.
    /// Uses tensor cores (CUBLAS_GEMM_DEFAULT_TENSOR_OP) for maximum throughput.
    ///
    /// This is the standard mixed-precision pattern for inference prefill:
    /// cached FP16 weights × FP16 activations → FP32 output.
    ///
    /// # Errors
    ///
    /// Returns error if cublasGemmEx fails.
    pub fn gemm_f16_to_f32(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: i32,
        b_ptr: CUdeviceptr,
        ldb: i32,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;

        let result = unsafe {
            (driver.cublasGemmEx)(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m,
                n,
                k,
                std::ptr::from_ref::<f32>(&alpha) as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                lda,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                ldb,
                std::ptr::from_ref::<f32>(&beta) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };

        CublasDriver::check(result).map_err(|e| {
            GpuError::CudaDriver(format!("cublasGemmEx_f16_f32(m={m}, n={n}, k={k}): {e}"), 0)
        })
    }

    /// FP8 (E4M3) GEMM via cuBLAS tensor cores — both A and B are FP8 E4M3.
    ///
    /// PMAT-053: FP8 weight cache reduces data from 2 B/elem (FP16) to 1 B/elem (FP8),
    /// halving memory bandwidth for prefill and batched decode HGEMM operations.
    /// Requires sm_89+ (Ada Lovelace) for FP8 tensor core support.
    ///
    /// A (weights) and B (activations) are FP8 E4M3. C (output) is FP32.
    /// FP32 accumulation preserves numerical stability.
    ///
    /// cuBLAS FP8 constraints: transa=Trans, transb=NoTrans.
    ///
    /// # Errors
    ///
    /// Returns error if cublasGemmEx fails (e.g., FP8 not supported on hardware).
    pub fn gemm_f8e4m3_to_f32(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: i32,
        b_ptr: CUdeviceptr,
        ldb: i32,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;

        let result = unsafe {
            (driver.cublasGemmEx)(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m,
                n,
                k,
                std::ptr::from_ref::<f32>(&alpha) as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_8F_E4M3,
                lda,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_8F_E4M3,
                ldb,
                std::ptr::from_ref::<f32>(&beta) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };

        CublasDriver::check(result).map_err(|e| {
            GpuError::CudaDriver(format!("cublasGemmEx_f8e4m3_f32(m={m}, n={n}, k={k}): {e}"), 0)
        })
    }

    /// FP32 GEMM via cuBLAS SIMD (no tensor cores)
    ///
    /// Computes: C = alpha * op(A) * op(B) + beta * C
    /// All inputs/outputs are FP32 with strict FP32 accumulation.
    ///
    /// # Contract (ALB-076)
    ///
    /// Uses CUBLAS_COMPUTE_32F + CUBLAS_GEMM_DEFAULT (no tensor cores).
    /// Tensor core algorithms (CUBLAS_GEMM_DEFAULT_TENSOR_OP) produce NaN
    /// for transposed backward GEMMs when gradient magnitudes reach ~1e5.
    /// SIMD path is 6-14x faster than hand-written PTX — sufficient.
    ///
    /// # Errors
    ///
    /// Returns error if cublasGemmEx fails.
    pub fn gemm_f32(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: i32,
        b_ptr: CUdeviceptr,
        ldb: i32,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;

        let result = unsafe {
            (driver.cublasGemmEx)(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m,
                n,
                k,
                std::ptr::from_ref::<f32>(&alpha) as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_32F,
                lda,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_32F,
                ldb,
                std::ptr::from_ref::<f32>(&beta) as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            )
        };

        CublasDriver::check(result).map_err(|e| {
            GpuError::CudaDriver(format!("cublasGemmEx_f32(m={m}, n={n}, k={k}): {e}"), 0)
        })
    }

    /// FP32 Strided Batched GEMM for multi-head attention
    ///
    /// Computes: C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
    /// for i in 0..batch_count
    ///
    /// Each batch element is at stride offset from the base pointer.
    /// Used for attention QK^T and attn·V across all heads simultaneously.
    ///
    /// # Arguments
    ///
    /// * `transa`, `transb` - Transpose operations
    /// * `m`, `n`, `k` - Matrix dimensions (per batch element)
    /// * `alpha`, `beta` - Scalar multipliers
    /// * `a_ptr` - Device pointer to first A matrix
    /// * `lda` - Leading dimension of A
    /// * `stride_a` - Stride between consecutive A matrices (in elements)
    /// * `b_ptr` - Device pointer to first B matrix
    /// * `ldb` - Leading dimension of B
    /// * `stride_b` - Stride between consecutive B matrices (in elements)
    /// * `c_ptr` - Device pointer to first C matrix
    /// * `ldc` - Leading dimension of C
    /// * `stride_c` - Stride between consecutive C matrices (in elements)
    /// * `batch_count` - Number of GEMM operations in batch
    ///
    /// # Errors
    ///
    /// Returns error if cublasSgemmStridedBatched fails.
    pub fn gemm_f32_strided_batched(
        &self,
        transa: GemmOp,
        transb: GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        lda: i32,
        stride_a: i64,
        b_ptr: CUdeviceptr,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c_ptr: CUdeviceptr,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> Result<(), GpuError> {
        let driver = get_cublas_driver()?;

        let result = unsafe {
            (driver.cublasSgemmStridedBatched)(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m,
                n,
                k,
                &alpha,
                a_ptr as *const std::ffi::c_void,
                lda,
                stride_a,
                b_ptr as *const std::ffi::c_void,
                ldb,
                stride_b,
                &beta,
                c_ptr as *mut std::ffi::c_void,
                ldc,
                stride_c,
                batch_count,
            )
        };

        CublasDriver::check(result).map_err(|e| {
            GpuError::CudaDriver(
                format!("cublasSgemmStridedBatched(m={m}, n={n}, k={k}, batch={batch_count}): {e}"),
                0,
            )
        })
    }

    /// Get the raw cuBLAS handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CublasHandle` is alive.
    #[must_use]
    pub fn raw(&self) -> super::cublas_sys::CublasHandle {
        self.handle
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // Contract: cublasDestroy_v2 called exactly once (RAII)
        if let Some(driver) = CublasDriver::load() {
            unsafe {
                let _ = (driver.cublasDestroy_v2)(self.handle);
            }
        }
    }
}

// ============================================================================
// Helper: Get cuBLAS driver
// ============================================================================

fn get_cublas_driver() -> Result<&'static CublasDriver, GpuError> {
    CublasDriver::load()
        .ok_or_else(|| GpuError::CudaNotAvailable("cuBLAS library not found".to_string()))
}

// ============================================================================
// Row-Major GEMM Helper
// ============================================================================

/// Convenience wrapper for row-major GEMM (Rust-native memory layout)
///
/// Computes C = A @ B in row-major layout by exploiting the identity:
///   C_row = (B^T @ A^T)^T in column-major
///
/// This is the standard trick for using cuBLAS (column-major) with
/// row-major data without explicit transposition.
///
/// # Contract (FALSIFY-CUBLAS-011)
///
/// Row-major Rust buffers produce correct results via transpose flags.
/// This avoids ALB-059 class bugs (wrong transpose convention).
impl CublasHandle {
    /// Row-major FP16 GEMM: C[m,n] = A[m,k] @ B[k,n]
    ///
    /// All matrices are row-major (Rust native). Internally translates to
    /// cuBLAS column-major via the B^T @ A^T identity.
    ///
    /// # Buffer Requirements
    ///
    /// - a_ptr: m * k * 2 bytes (FP16)
    /// - b_ptr: k * n * 2 bytes (FP16)
    /// - c_ptr: m * n * 2 bytes (FP16)
    ///
    /// # Errors
    ///
    /// Returns error if GEMM execution fails.
    pub fn gemm_f16_row_major(
        &self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        b_ptr: CUdeviceptr,
        beta: f32,
        c_ptr: CUdeviceptr,
    ) -> Result<(), GpuError> {
        // Row-major C = A @ B is equivalent to:
        // Column-major C^T = B^T @ A^T
        // cuBLAS sees column-major, so we swap A and B and use n as leading dim
        self.gemm_f16(
            GemmOp::NoTrans, // B is not transposed (in col-major = B^T in row-major)
            GemmOp::NoTrans, // A is not transposed (in col-major = A^T in row-major)
            n,               // rows of op(B^T) = cols of B = n
            m,               // cols of op(A^T) = rows of A = m
            k,               // shared dimension
            alpha,
            b_ptr,
            n, // B with leading dim n (row-major stride)
            a_ptr,
            k, // A with leading dim k (row-major stride)
            beta,
            c_ptr,
            n, // C with leading dim n (row-major stride)
        )
    }

    /// Row-major FP32 strided batched GEMM: C[i][m,n] = A[i][m,k] @ B[i][k,n]
    ///
    /// All matrices are row-major. Strides are in f32 elements.
    /// Used for multi-head attention (batch_count = batch_size * num_heads).
    ///
    /// # Errors
    ///
    /// Returns error if GEMM execution fails.
    pub fn gemm_f32_strided_batched_row_major(
        &self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        stride_a: i64,
        b_ptr: CUdeviceptr,
        stride_b: i64,
        beta: f32,
        c_ptr: CUdeviceptr,
        stride_c: i64,
        batch_count: i32,
    ) -> Result<(), GpuError> {
        // Row-major C = A @ B => col-major C^T = B^T @ A^T
        // Swap A/B, use n as leading dim
        self.gemm_f32_strided_batched(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n,
            m,
            k,
            alpha,
            b_ptr,
            n,
            stride_b,
            a_ptr,
            k,
            stride_a,
            beta,
            c_ptr,
            n,
            stride_c,
            batch_count,
        )
    }

    /// Row-major FP32 GEMM: C[m,n] = A[m,k] @ B[k,n]
    ///
    /// # Errors
    ///
    /// Returns error if GEMM execution fails.
    pub fn gemm_f32_row_major(
        &self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: CUdeviceptr,
        b_ptr: CUdeviceptr,
        beta: f32,
        c_ptr: CUdeviceptr,
    ) -> Result<(), GpuError> {
        self.gemm_f32(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n,
            m,
            k,
            alpha,
            b_ptr,
            n,
            a_ptr,
            k,
            beta,
            c_ptr,
            n,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_op_to_cublas() {
        assert_eq!(GemmOp::NoTrans.to_cublas(), CUBLAS_OP_N);
        assert_eq!(GemmOp::Trans.to_cublas(), CUBLAS_OP_T);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_cublas_handle_requires_cuda() {
        // Can't create handle without cuda feature — get_cublas_driver returns Err
        let result = get_cublas_driver();
        assert!(result.is_err());
    }
}
