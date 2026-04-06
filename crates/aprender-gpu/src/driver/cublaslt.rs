//! cuBLASLt Safe Wrapper — FP8 E4M3 GEMM
//!
//! PMAT-053: cuBLASLt is required for FP8 (cublasGemmEx doesn't support FP8).
//! Provides `CublasLtHandle` with `gemm_fp8_e4m3_to_f16` for FP8×FP8→FP16 matmul.
//!
//! FP8 GEMM requires FP16/BF16 output — FP32 output is NOT supported.
//! Internal accumulation is FP32 (CUBLAS_COMPUTE_32F), but the output tensor
//! must be FP16 or BF16. Caller converts FP16→FP32 if needed.

use super::cublas_sys::{
    CublasOperation, CUBLAS_COMPUTE_32F, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16F, CUDA_R_32F,
    CUDA_R_8F_E4M3,
};
use super::cublaslt_sys::*;
use super::stream::CudaStream;
use crate::GpuError;
use std::ffi::c_void;

/// PMAT-086: Cached cuBLASLt execution plan for a specific (M, N, K) shape.
///
/// Creating matmul descriptors + layouts + running the heuristic algorithm search
/// costs ~20-50μs per GEMM call. With 168 GEMMs per prefill (7 per layer × 28 layers),
/// this adds ~3-8ms of CPU-side overhead. Caching plans for the 4 unique shapes
/// reduces this to a one-time cost of ~0.1ms.
struct CachedFp8Plan {
    matmul_desc: CublasLtMatmulDesc,
    a_layout: CublasLtMatrixLayout,
    b_layout: CublasLtMatrixLayout,
    c_layout: CublasLtMatrixLayout,
    d_layout: CublasLtMatrixLayout,
    algo: CublasLtMatmulAlgo,
}

/// Safe wrapper around cuBLASLt handle
pub struct CublasLtHandle {
    handle: CublasLtHandleRaw,
    /// PMAT-086: Cached FP8→FP16 GEMM plans keyed by (m_padded, n, k)
    fp8_plan_cache: std::collections::HashMap<(i32, i32, i32), CachedFp8Plan>,
}

// SAFETY: cuBLASLt handles are thread-safe when used with proper stream synchronization.
// Same safety guarantee as CublasHandle (cublas.rs:69-70).
unsafe impl Send for CublasLtHandle {}
unsafe impl Sync for CublasLtHandle {}

type CublasLtHandleRaw = super::cublaslt_sys::CublasLtHandle;

impl CublasLtHandle {
    /// Create a new cuBLASLt handle
    pub fn new() -> Result<Self, GpuError> {
        let driver = CublasLtDriver::load()
            .ok_or_else(|| GpuError::CudaNotAvailable("cuBLASLt library not found".to_string()))?;

        let mut handle: CublasLtHandleRaw = std::ptr::null_mut();
        let status = unsafe { (driver.cublasLtCreate)(&mut handle) };
        CublasLtDriver::check(status)?;

        Ok(Self { handle, fp8_plan_cache: std::collections::HashMap::new() })
    }

    /// FP8 E4M3 × FP8 E4M3 → FP16 GEMM via cuBLASLt
    ///
    /// D_f16 = alpha * op(A_fp8) * op(B_fp8) + beta * C_f16
    ///
    /// A, B: FP8 E4M3 (1 byte/elem)
    /// C, D: FP16 (2 bytes/elem) — FP8 GEMM does NOT support FP32 output
    ///
    /// Internal accumulation uses FP32 (CUBLAS_COMPUTE_32F).
    /// Caller must convert FP16→FP32 if FP32 output is needed.
    ///
    /// For weight×activation: transa=Trans, transb=NoTrans (TN format required for FP8)
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_e4m3_to_f16(
        &self,
        transa: super::cublas::GemmOp,
        transb: super::cublas::GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: u64,
        lda: i32,
        b_ptr: u64,
        ldb: i32,
        beta: f32,
        d_ptr: u64, // FP16 output buffer
        ldd: i32,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let driver = CublasLtDriver::load()
            .ok_or_else(|| GpuError::CudaNotAvailable("cuBLASLt not loaded".to_string()))?;

        let op_a: CublasOperation = match transa {
            super::cublas::GemmOp::NoTrans => CUBLAS_OP_N,
            super::cublas::GemmOp::Trans => CUBLAS_OP_T,
        };
        let op_b: CublasOperation = match transb {
            super::cublas::GemmOp::NoTrans => CUBLAS_OP_N,
            super::cublas::GemmOp::Trans => CUBLAS_OP_T,
        };

        unsafe {
            // 1. Create matmul descriptor (compute=FP32, scale=FP32)
            let mut matmul_desc: CublasLtMatmulDesc = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatmulDescCreate)(
                &mut matmul_desc,
                CUBLAS_COMPUTE_32F,
                CUDA_R_32F, // scale type
            ))?;

            // Set transa/transb
            CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                matmul_desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                std::ptr::from_ref(&op_a) as *const c_void,
                std::mem::size_of::<CublasOperation>(),
            ))?;
            CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                matmul_desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                std::ptr::from_ref(&op_b) as *const c_void,
                std::mem::size_of::<CublasOperation>(),
            ))?;

            // 2. Create matrix layouts
            // A: FP8 — physical dims depend on transa
            let (a_rows, a_cols) =
                if op_a == CUBLAS_OP_T { (k as u64, m as u64) } else { (m as u64, k as u64) };
            let mut a_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut a_layout,
                CUDA_R_8F_E4M3,
                a_rows,
                a_cols,
                lda as i64,
            ))?;

            // B: FP8
            let (b_rows, b_cols) =
                if op_b == CUBLAS_OP_T { (n as u64, k as u64) } else { (k as u64, n as u64) };
            let mut b_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut b_layout,
                CUDA_R_8F_E4M3,
                b_rows,
                b_cols,
                ldb as i64,
            ))?;

            // C: FP16 (bias input — same layout as D, beta=0 so unused)
            let mut c_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut c_layout,
                CUDA_R_16F,
                m as u64,
                n as u64,
                ldd as i64,
            ))?;

            // D: FP16 output (FP8 GEMM does not support FP32 output)
            let mut d_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut d_layout,
                CUDA_R_16F,
                m as u64,
                n as u64,
                ldd as i64,
            ))?;

            // 3. Create preference (no workspace needed for typical sizes)
            let mut pref: CublasLtMatmulPreference = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatmulPreferenceCreate)(&mut pref))?;

            let max_workspace: usize = 0; // no workspace
            CublasLtDriver::check((driver.cublasLtMatmulPreferenceSetAttribute)(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                std::ptr::from_ref(&max_workspace) as *const c_void,
                std::mem::size_of::<usize>(),
            ))?;

            // 4. Get best algorithm via heuristic
            let mut heur_result = std::mem::zeroed::<CublasLtMatmulHeuristicResult>();
            let mut returned_count: i32 = 0;

            let heur_status = (driver.cublasLtMatmulAlgoGetHeuristic)(
                self.handle,
                matmul_desc,
                a_layout,
                b_layout,
                c_layout,
                d_layout,
                pref,
                1,
                &mut heur_result,
                &mut returned_count,
            );

            if heur_status != CUBLASLT_STATUS_SUCCESS || returned_count == 0 {
                (driver.cublasLtMatmulPreferenceDestroy)(pref);
                (driver.cublasLtMatrixLayoutDestroy)(d_layout);
                (driver.cublasLtMatrixLayoutDestroy)(c_layout);
                (driver.cublasLtMatrixLayoutDestroy)(b_layout);
                (driver.cublasLtMatrixLayoutDestroy)(a_layout);
                (driver.cublasLtMatmulDescDestroy)(matmul_desc);

                return Err(GpuError::CudaDriver(
                    format!(
                        "cublasLtMatmulAlgoGetHeuristic fp8_f16 failed: status={heur_status}, returned={returned_count}, m={m}, n={n}, k={k}"
                    ),
                    heur_status,
                ));
            }

            // 5. Execute matmul
            let matmul_status = (driver.cublasLtMatmul)(
                self.handle,
                matmul_desc,
                std::ptr::from_ref::<f32>(&alpha) as *const c_void,
                a_ptr as *const c_void,
                a_layout,
                b_ptr as *const c_void,
                b_layout,
                std::ptr::from_ref::<f32>(&beta) as *const c_void,
                d_ptr as *const c_void, // C = D when beta=0
                c_layout,
                d_ptr as *mut c_void,
                d_layout,
                &heur_result.algo,
                std::ptr::null_mut(),
                0,
                stream.raw(),
            );

            // 6. Cleanup
            (driver.cublasLtMatmulPreferenceDestroy)(pref);
            (driver.cublasLtMatrixLayoutDestroy)(d_layout);
            (driver.cublasLtMatrixLayoutDestroy)(c_layout);
            (driver.cublasLtMatrixLayoutDestroy)(b_layout);
            (driver.cublasLtMatrixLayoutDestroy)(a_layout);
            (driver.cublasLtMatmulDescDestroy)(matmul_desc);

            CublasLtDriver::check(matmul_status).map_err(|e| {
                GpuError::CudaDriver(format!("cublasLtMatmul_fp8_f16(m={m}, n={n}, k={k}): {e}"), 0)
            })
        }
    }

    /// FP8 E4M3 × FP8 E4M3 → FP16 GEMM with per-tensor scaling
    ///
    /// D_f16 = alpha * a_scale * b_scale * (A_fp8 × B_fp8)
    ///
    /// a_scale_ptr, b_scale_ptr: device pointers to single FP32 values.
    /// cuBLASLt multiplies the GEMM result by (a_scale * b_scale) internally,
    /// recovering the original dynamic range lost during FP8 quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_e4m3_to_f16_scaled(
        &self,
        transa: super::cublas::GemmOp,
        transb: super::cublas::GemmOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: u64,
        lda: i32,
        a_scale_ptr: u64, // device pointer to FP32 scale
        b_ptr: u64,
        ldb: i32,
        b_scale_ptr: u64, // device pointer to FP32 scale
        beta: f32,
        d_ptr: u64,
        ldd: i32,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let driver = CublasLtDriver::load()
            .ok_or_else(|| GpuError::CudaNotAvailable("cuBLASLt not loaded".to_string()))?;

        let op_a: CublasOperation = match transa {
            super::cublas::GemmOp::NoTrans => CUBLAS_OP_N,
            super::cublas::GemmOp::Trans => CUBLAS_OP_T,
        };
        let op_b: CublasOperation = match transb {
            super::cublas::GemmOp::NoTrans => CUBLAS_OP_N,
            super::cublas::GemmOp::Trans => CUBLAS_OP_T,
        };

        unsafe {
            // 1. Create matmul descriptor
            let mut matmul_desc: CublasLtMatmulDesc = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatmulDescCreate)(
                &mut matmul_desc,
                CUBLAS_COMPUTE_32F,
                CUDA_R_32F,
            ))?;

            // Set transa/transb
            CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                matmul_desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                std::ptr::from_ref(&op_a) as *const c_void,
                std::mem::size_of::<CublasOperation>(),
            ))?;
            CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                matmul_desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                std::ptr::from_ref(&op_b) as *const c_void,
                std::mem::size_of::<CublasOperation>(),
            ))?;

            // Set per-tensor scale pointers (device pointers to FP32 scalars)
            let a_scale_device_ptr = a_scale_ptr as *const c_void;
            CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                matmul_desc,
                CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                std::ptr::from_ref(&a_scale_device_ptr) as *const c_void,
                std::mem::size_of::<*const c_void>(),
            ))?;
            let b_scale_device_ptr = b_scale_ptr as *const c_void;
            CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                matmul_desc,
                CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                std::ptr::from_ref(&b_scale_device_ptr) as *const c_void,
                std::mem::size_of::<*const c_void>(),
            ))?;

            // 2. Create matrix layouts (same as unscaled)
            let (a_rows, a_cols) =
                if op_a == CUBLAS_OP_T { (k as u64, m as u64) } else { (m as u64, k as u64) };
            let mut a_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut a_layout,
                CUDA_R_8F_E4M3,
                a_rows,
                a_cols,
                lda as i64,
            ))?;

            let (b_rows, b_cols) =
                if op_b == CUBLAS_OP_T { (n as u64, k as u64) } else { (k as u64, n as u64) };
            let mut b_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut b_layout,
                CUDA_R_8F_E4M3,
                b_rows,
                b_cols,
                ldb as i64,
            ))?;

            let mut c_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut c_layout,
                CUDA_R_16F,
                m as u64,
                n as u64,
                ldd as i64,
            ))?;

            let mut d_layout: CublasLtMatrixLayout = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                &mut d_layout,
                CUDA_R_16F,
                m as u64,
                n as u64,
                ldd as i64,
            ))?;

            // 3. Get algorithm
            let mut pref: CublasLtMatmulPreference = std::ptr::null_mut();
            CublasLtDriver::check((driver.cublasLtMatmulPreferenceCreate)(&mut pref))?;

            let max_workspace: usize = 0;
            CublasLtDriver::check((driver.cublasLtMatmulPreferenceSetAttribute)(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                std::ptr::from_ref(&max_workspace) as *const c_void,
                std::mem::size_of::<usize>(),
            ))?;

            let mut heur_result = std::mem::zeroed::<CublasLtMatmulHeuristicResult>();
            let mut returned_count: i32 = 0;

            let heur_status = (driver.cublasLtMatmulAlgoGetHeuristic)(
                self.handle,
                matmul_desc,
                a_layout,
                b_layout,
                c_layout,
                d_layout,
                pref,
                1,
                &mut heur_result,
                &mut returned_count,
            );

            if heur_status != CUBLASLT_STATUS_SUCCESS || returned_count == 0 {
                (driver.cublasLtMatmulPreferenceDestroy)(pref);
                (driver.cublasLtMatrixLayoutDestroy)(d_layout);
                (driver.cublasLtMatrixLayoutDestroy)(c_layout);
                (driver.cublasLtMatrixLayoutDestroy)(b_layout);
                (driver.cublasLtMatrixLayoutDestroy)(a_layout);
                (driver.cublasLtMatmulDescDestroy)(matmul_desc);

                return Err(GpuError::CudaDriver(
                    format!(
                        "cublasLtMatmulAlgoGetHeuristic fp8_scaled failed: status={heur_status}, m={m}, n={n}, k={k}"
                    ),
                    heur_status,
                ));
            }

            // 4. Execute scaled matmul
            let matmul_status = (driver.cublasLtMatmul)(
                self.handle,
                matmul_desc,
                std::ptr::from_ref::<f32>(&alpha) as *const c_void,
                a_ptr as *const c_void,
                a_layout,
                b_ptr as *const c_void,
                b_layout,
                std::ptr::from_ref::<f32>(&beta) as *const c_void,
                d_ptr as *const c_void,
                c_layout,
                d_ptr as *mut c_void,
                d_layout,
                &heur_result.algo,
                std::ptr::null_mut(),
                0,
                stream.raw(),
            );

            // Cleanup
            (driver.cublasLtMatmulPreferenceDestroy)(pref);
            (driver.cublasLtMatrixLayoutDestroy)(d_layout);
            (driver.cublasLtMatrixLayoutDestroy)(c_layout);
            (driver.cublasLtMatrixLayoutDestroy)(b_layout);
            (driver.cublasLtMatrixLayoutDestroy)(a_layout);
            (driver.cublasLtMatmulDescDestroy)(matmul_desc);

            CublasLtDriver::check(matmul_status).map_err(|e| {
                GpuError::CudaDriver(
                    format!("cublasLtMatmul_fp8_scaled(m={m}, n={n}, k={k}): {e}"),
                    0,
                )
            })
        }
    }

    /// PMAT-086: Cached FP8 E4M3 × FP8 E4M3 → FP16 GEMM
    ///
    /// Same as `gemm_fp8_e4m3_to_f16` but caches the matmul descriptors, layouts,
    /// and algorithm selection per unique (m, n, k) shape. Eliminates ~20-50μs of
    /// cuBLASLt API overhead per GEMM call (descriptor creation + heuristic search).
    ///
    /// For prefill with 168 GEMMs across 4 unique shapes, this reduces CPU overhead
    /// from ~5ms to ~0.1ms.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_e4m3_to_f16_cached(
        &mut self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: u64,
        lda: i32,
        b_ptr: u64,
        ldb: i32,
        beta: f32,
        d_ptr: u64,
        ldd: i32,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let driver = CublasLtDriver::load()
            .ok_or_else(|| GpuError::CudaNotAvailable("cuBLASLt not loaded".to_string()))?;

        let cache_key = (m, n, k);

        // PMAT-086: Build and cache plan on first use for this shape
        if !self.fp8_plan_cache.contains_key(&cache_key) {
            unsafe {
                let mut matmul_desc: CublasLtMatmulDesc = std::ptr::null_mut();
                CublasLtDriver::check((driver.cublasLtMatmulDescCreate)(
                    &mut matmul_desc,
                    CUBLAS_COMPUTE_32F,
                    CUDA_R_32F,
                ))?;

                let op_a = CUBLAS_OP_T;
                let op_b = CUBLAS_OP_N;
                CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                    matmul_desc,
                    CUBLASLT_MATMUL_DESC_TRANSA,
                    std::ptr::from_ref(&op_a) as *const c_void,
                    std::mem::size_of::<CublasOperation>(),
                ))?;
                CublasLtDriver::check((driver.cublasLtMatmulDescSetAttribute)(
                    matmul_desc,
                    CUBLASLT_MATMUL_DESC_TRANSB,
                    std::ptr::from_ref(&op_b) as *const c_void,
                    std::mem::size_of::<CublasOperation>(),
                ))?;

                // A: FP8 physical [K, M] col-major → transposed to logical [M, K]
                let mut a_layout: CublasLtMatrixLayout = std::ptr::null_mut();
                CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                    &mut a_layout,
                    CUDA_R_8F_E4M3,
                    k as u64,
                    m as u64,
                    lda as i64,
                ))?;

                // B: FP8 physical [K, N] col-major (NoTrans)
                let mut b_layout: CublasLtMatrixLayout = std::ptr::null_mut();
                CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                    &mut b_layout,
                    CUDA_R_8F_E4M3,
                    k as u64,
                    n as u64,
                    ldb as i64,
                ))?;

                // C/D: FP16 [M, N] col-major
                let mut c_layout: CublasLtMatrixLayout = std::ptr::null_mut();
                CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                    &mut c_layout,
                    CUDA_R_16F,
                    m as u64,
                    n as u64,
                    ldd as i64,
                ))?;

                let mut d_layout: CublasLtMatrixLayout = std::ptr::null_mut();
                CublasLtDriver::check((driver.cublasLtMatrixLayoutCreate)(
                    &mut d_layout,
                    CUDA_R_16F,
                    m as u64,
                    n as u64,
                    ldd as i64,
                ))?;

                // Heuristic algorithm search (expensive — ~10-50μs)
                let mut pref: CublasLtMatmulPreference = std::ptr::null_mut();
                CublasLtDriver::check((driver.cublasLtMatmulPreferenceCreate)(&mut pref))?;

                let max_workspace: usize = 0;
                CublasLtDriver::check((driver.cublasLtMatmulPreferenceSetAttribute)(
                    pref,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    std::ptr::from_ref(&max_workspace) as *const c_void,
                    std::mem::size_of::<usize>(),
                ))?;

                let mut heur_result = std::mem::zeroed::<CublasLtMatmulHeuristicResult>();
                let mut returned_count: i32 = 0;

                let heur_status = (driver.cublasLtMatmulAlgoGetHeuristic)(
                    self.handle,
                    matmul_desc,
                    a_layout,
                    b_layout,
                    c_layout,
                    d_layout,
                    pref,
                    1,
                    &mut heur_result,
                    &mut returned_count,
                );

                (driver.cublasLtMatmulPreferenceDestroy)(pref);

                if heur_status != CUBLASLT_STATUS_SUCCESS || returned_count == 0 {
                    (driver.cublasLtMatrixLayoutDestroy)(d_layout);
                    (driver.cublasLtMatrixLayoutDestroy)(c_layout);
                    (driver.cublasLtMatrixLayoutDestroy)(b_layout);
                    (driver.cublasLtMatrixLayoutDestroy)(a_layout);
                    (driver.cublasLtMatmulDescDestroy)(matmul_desc);

                    return Err(GpuError::CudaDriver(
                        format!(
                            "cublasLtMatmulAlgoGetHeuristic fp8_cached failed: status={heur_status}, m={m}, n={n}, k={k}"
                        ),
                        heur_status,
                    ));
                }

                self.fp8_plan_cache.insert(
                    cache_key,
                    CachedFp8Plan {
                        matmul_desc,
                        a_layout,
                        b_layout,
                        c_layout,
                        d_layout,
                        algo: heur_result.algo,
                    },
                );
            }
        }

        // Execute with cached plan — only the cublasLtMatmul call, no descriptor overhead
        let plan = self.fp8_plan_cache.get(&cache_key).expect("just inserted");

        unsafe {
            let matmul_status = (driver.cublasLtMatmul)(
                self.handle,
                plan.matmul_desc,
                std::ptr::from_ref::<f32>(&alpha) as *const c_void,
                a_ptr as *const c_void,
                plan.a_layout,
                b_ptr as *const c_void,
                plan.b_layout,
                std::ptr::from_ref::<f32>(&beta) as *const c_void,
                d_ptr as *const c_void,
                plan.c_layout,
                d_ptr as *mut c_void,
                plan.d_layout,
                &plan.algo,
                std::ptr::null_mut(),
                0,
                stream.raw(),
            );

            CublasLtDriver::check(matmul_status).map_err(|e| {
                GpuError::CudaDriver(
                    format!("cublasLtMatmul_fp8_cached(m={m}, n={n}, k={k}): {e}"),
                    0,
                )
            })
        }
    }
}

impl Drop for CublasLtHandle {
    fn drop(&mut self) {
        if let Some(driver) = CublasLtDriver::load() {
            unsafe {
                // PMAT-086: Clean up cached plans
                for plan in self.fp8_plan_cache.values() {
                    (driver.cublasLtMatrixLayoutDestroy)(plan.d_layout);
                    (driver.cublasLtMatrixLayoutDestroy)(plan.c_layout);
                    (driver.cublasLtMatrixLayoutDestroy)(plan.b_layout);
                    (driver.cublasLtMatrixLayoutDestroy)(plan.a_layout);
                    (driver.cublasLtMatmulDescDestroy)(plan.matmul_desc);
                }
                (driver.cublasLtDestroy)(self.handle);
            }
        }
    }
}
