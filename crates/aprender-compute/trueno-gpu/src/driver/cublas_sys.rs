//! cuBLAS Runtime API FFI Bindings
//!
//! Hand-written FFI for cuBLAS tensor core GEMM operations.
//! Follows the same pattern as driver/sys.rs: dynamic loading via libcublas.so.
//!
//! # Design Philosophy
//!
//! **OWN THE STACK**: We built 400 lines of CUDA driver FFI. We can build
//! 200 lines of cuBLAS FFI. No external cublas-sys or bindgen dependency.
//!
//! # Contract
//!
//! `cublas-gemm-v1.yaml` — ALB-075: cuBLAS tensor core GEMM integration
//!
//! # Safety
//!
//! All functions in this module are unsafe. Safe wrappers in cublas.rs.

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::GpuError;

// ============================================================================
// cuBLAS Type Definitions (from cublas_v2.h)
// ============================================================================

/// cuBLAS handle (opaque pointer)
pub type CublasHandle = *mut c_void;

/// cuBLAS status code
pub type CublasStatus = c_int;

// ============================================================================
// cuBLAS Status Codes
// ============================================================================

/// Success
pub const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
/// Library not initialized
pub const CUBLAS_STATUS_NOT_INITIALIZED: CublasStatus = 1;
/// Resource allocation failed
pub const CUBLAS_STATUS_ALLOC_FAILED: CublasStatus = 3;
/// Invalid value
pub const CUBLAS_STATUS_INVALID_VALUE: CublasStatus = 7;
/// Arch mismatch
pub const CUBLAS_STATUS_ARCH_MISMATCH: CublasStatus = 8;
/// Execution failed
pub const CUBLAS_STATUS_EXECUTION_FAILED: CublasStatus = 13;
/// Internal error
pub const CUBLAS_STATUS_INTERNAL_ERROR: CublasStatus = 14;
/// Feature not supported
pub const CUBLAS_STATUS_NOT_SUPPORTED: CublasStatus = 15;

// ============================================================================
// cuBLAS Operation Types
// ============================================================================

/// cublasOperation_t
pub type CublasOperation = c_int;

/// Non-transposed
pub const CUBLAS_OP_N: CublasOperation = 0;
/// Transposed
pub const CUBLAS_OP_T: CublasOperation = 1;
/// Conjugate transposed
pub const CUBLAS_OP_C: CublasOperation = 2;

// ============================================================================
// CUDA Data Types (from library_types.h, subset for cuBLAS)
// ============================================================================

/// cudaDataType_t
pub type CudaDataType = c_int;

/// 16-bit floating point (half)
pub const CUDA_R_16F: CudaDataType = 2;
/// 32-bit floating point
pub const CUDA_R_32F: CudaDataType = 0;
/// 16-bit brain floating point
pub const CUDA_R_16BF: CudaDataType = 14;
/// 8-bit floating point E4M3 (sign=1, exponent=4, mantissa=3)
/// Range: ±448, precision: ~1/16. Requires sm_89+ (Ada Lovelace).
pub const CUDA_R_8F_E4M3: CudaDataType = 28;

// ============================================================================
// cuBLAS Compute Types
// ============================================================================

/// cublasComputeType_t
pub type CublasComputeType = c_int;

/// FP32 accumulation (required for training stability)
pub const CUBLAS_COMPUTE_32F: CublasComputeType = 68;
/// FP32 inputs with TF32 tensor core acceleration (10-bit mantissa, 2x faster)
/// Standard for NN training (PyTorch default since v1.7)
pub const CUBLAS_COMPUTE_32F_FAST_TF32: CublasComputeType = 74;
/// FP16 accumulation (faster but less precise)
pub const CUBLAS_COMPUTE_16F: CublasComputeType = 64;

// ============================================================================
// cuBLAS Math Mode
// ============================================================================

/// cublasMath_t
pub type CublasMathMode = c_int;

/// Default math mode (no tensor cores for FP32)
pub const CUBLAS_DEFAULT_MATH: CublasMathMode = 0;
/// Use tensor cores when possible (deprecated since CUDA 11)
pub const CUBLAS_TENSOR_OP_MATH: CublasMathMode = 1;
/// Strict FP32, no tensor cores
pub const CUBLAS_PEDANTIC_MATH: CublasMathMode = 2;
/// TF32 tensor cores for FP32 ops, tensor cores for FP16/BF16
pub const CUBLAS_TF32_TENSOR_OP_MATH: CublasMathMode = 3;

// ============================================================================
// cuBLAS Function Pointers (dynamically loaded)
// ============================================================================

/// Dynamically loaded cuBLAS functions
///
/// Loaded from libcublas.so (Linux) at runtime.
/// Follows same pattern as CudaDriver in driver/sys.rs.
#[allow(non_snake_case)]
pub struct CublasDriver {
    /// cublasCreate_v2 — Create cuBLAS handle
    pub cublasCreate_v2: unsafe extern "C" fn(handle: *mut CublasHandle) -> CublasStatus,

    /// cublasDestroy_v2 — Destroy cuBLAS handle
    pub cublasDestroy_v2: unsafe extern "C" fn(handle: CublasHandle) -> CublasStatus,

    /// cublasSetStream_v2 — Bind handle to CUDA stream
    pub cublasSetStream_v2:
        unsafe extern "C" fn(handle: CublasHandle, stream: *mut c_void) -> CublasStatus,

    /// cublasSetMathMode — Set math mode (enable tensor cores)
    pub cublasSetMathMode:
        unsafe extern "C" fn(handle: CublasHandle, mode: CublasMathMode) -> CublasStatus,

    /// cublasSetWorkspace — Pre-allocate workspace for CUDA graph capture
    ///
    /// PMAT-063: cuBLAS requires pre-allocated workspace during graph capture.
    /// Without this, cuBLAS falls back to workspace-free algorithms (7x slower).
    pub cublasSetWorkspace: unsafe extern "C" fn(
        handle: CublasHandle,
        workspace: *mut c_void,
        workspace_size: usize,
    ) -> CublasStatus,

    /// cublasGemmEx — General matrix multiply with mixed precision
    ///
    /// C = alpha * op(A) * op(B) + beta * C
    ///
    /// Contract (cublas-gemm-v1.yaml):
    /// - computeType MUST be CUBLAS_COMPUTE_32F for training (FP32 accumulation)
    /// - Buffer sizes verified BEFORE call (Rule 2: prove at kernel boundary)
    #[allow(clippy::type_complexity)]
    pub cublasGemmEx: unsafe extern "C" fn(
        handle: CublasHandle,
        transa: CublasOperation,
        transb: CublasOperation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: CudaDataType,
        lda: c_int,
        b: *const c_void,
        b_type: CudaDataType,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: CudaDataType,
        ldc: c_int,
        compute_type: CublasComputeType,
        algo: c_int,
    ) -> CublasStatus,

    /// cublasSgemmStridedBatched — Batched FP32 GEMM with strided memory
    ///
    /// C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]  for i in 0..batch_count
    ///
    /// Used for multi-head attention: QK^T and attn·V across all heads.
    #[allow(clippy::type_complexity)]
    pub cublasSgemmStridedBatched: unsafe extern "C" fn(
        handle: CublasHandle,
        transa: CublasOperation,
        transb: CublasOperation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        ldb: c_int,
        stride_b: i64,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> CublasStatus,
}

// ============================================================================
// Dynamic Loading
// ============================================================================

#[cfg(feature = "cuda")]
mod loading {
    use super::*;
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    /// Global cuBLAS driver instance
    static CUBLAS_DRIVER: OnceLock<Option<CublasDriver>> = OnceLock::new();

    /// Library handle (must outlive function pointers)
    static CUBLAS_LIBRARY: OnceLock<Option<Library>> = OnceLock::new();

    impl CublasDriver {
        /// Load cuBLAS driver dynamically
        ///
        /// Returns `None` if cuBLAS is not available.
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            let _ = CUBLAS_LIBRARY.get_or_init(|| {
                let lib_names = ["libcublas.so.12", "libcublas.so"];
                for name in lib_names {
                    if let Ok(lib) = unsafe { Library::new(name) } {
                        return Some(lib);
                    }
                }
                None
            });

            CUBLAS_DRIVER
                .get_or_init(|| {
                    let lib = CUBLAS_LIBRARY.get()?.as_ref()?;
                    Self::load_from_library(lib)
                })
                .as_ref()
        }

        /// Load function pointers from library
        fn load_from_library(lib: &Library) -> Option<Self> {
            unsafe {
                macro_rules! load_sym {
                    ($name:ident, $ty:ty) => {{
                        let sym: Symbol<'_, $ty> = lib.get(stringify!($name).as_bytes()).ok()?;
                        *sym
                    }};
                }

                type FnCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
                type FnDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
                type FnSetStream = unsafe extern "C" fn(CublasHandle, *mut c_void) -> CublasStatus;
                type FnSetMathMode =
                    unsafe extern "C" fn(CublasHandle, CublasMathMode) -> CublasStatus;
                type FnSetWorkspace =
                    unsafe extern "C" fn(CublasHandle, *mut c_void, usize) -> CublasStatus;
                type FnGemmEx = unsafe extern "C" fn(
                    CublasHandle,
                    CublasOperation,
                    CublasOperation,
                    c_int,
                    c_int,
                    c_int,
                    *const c_void,
                    *const c_void,
                    CudaDataType,
                    c_int,
                    *const c_void,
                    CudaDataType,
                    c_int,
                    *const c_void,
                    *mut c_void,
                    CudaDataType,
                    c_int,
                    CublasComputeType,
                    c_int,
                ) -> CublasStatus;
                type FnSgemmStridedBatched = unsafe extern "C" fn(
                    CublasHandle,
                    CublasOperation,
                    CublasOperation,
                    c_int,
                    c_int,
                    c_int,
                    *const f32,
                    *const c_void,
                    c_int,
                    i64,
                    *const c_void,
                    c_int,
                    i64,
                    *const f32,
                    *mut c_void,
                    c_int,
                    i64,
                    c_int,
                ) -> CublasStatus;

                Some(CublasDriver {
                    cublasCreate_v2: load_sym!(cublasCreate_v2, FnCreate),
                    cublasDestroy_v2: load_sym!(cublasDestroy_v2, FnDestroy),
                    cublasSetStream_v2: load_sym!(cublasSetStream_v2, FnSetStream),
                    cublasSetMathMode: load_sym!(cublasSetMathMode, FnSetMathMode),
                    cublasSetWorkspace: {
                        // cuBLAS exports this as cublasSetWorkspace_v2 (CUDA 11.0+)
                        let sym: Symbol<'_, FnSetWorkspace> =
                            lib.get(b"cublasSetWorkspace_v2").ok()?;
                        *sym
                    },
                    cublasGemmEx: load_sym!(cublasGemmEx, FnGemmEx),
                    cublasSgemmStridedBatched: load_sym!(
                        cublasSgemmStridedBatched,
                        FnSgemmStridedBatched
                    ),
                })
            }
        }

        /// Check cuBLAS result and convert to GpuError
        pub fn check(result: CublasStatus) -> Result<(), GpuError> {
            if result == CUBLAS_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(GpuError::CudaDriver(cublas_status_string(result).to_string(), result))
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod loading {
    use super::*;

    impl CublasDriver {
        /// cuBLAS not available without cuda feature
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            None
        }

        /// Check is a no-op without CUDA
        pub fn check(_result: CublasStatus) -> Result<(), GpuError> {
            Err(GpuError::CudaNotAvailable("cuda feature not enabled".to_string()))
        }
    }
}

// ============================================================================
// cuBLAS GemmEx algorithm constants
// ============================================================================

/// Default algorithm (let cuBLAS choose)
pub const CUBLAS_GEMM_DEFAULT: c_int = -1;
/// Default algorithm with tensor ops
pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: c_int = 99;

// ============================================================================
// Error String Conversion
// ============================================================================

/// Convert cuBLAS status code to human-readable string
#[must_use]
pub fn cublas_status_string(status: CublasStatus) -> &'static str {
    match status {
        CUBLAS_STATUS_SUCCESS => "CUBLAS_STATUS_SUCCESS",
        CUBLAS_STATUS_NOT_INITIALIZED => "CUBLAS_STATUS_NOT_INITIALIZED",
        CUBLAS_STATUS_ALLOC_FAILED => "CUBLAS_STATUS_ALLOC_FAILED",
        CUBLAS_STATUS_INVALID_VALUE => "CUBLAS_STATUS_INVALID_VALUE",
        CUBLAS_STATUS_ARCH_MISMATCH => "CUBLAS_STATUS_ARCH_MISMATCH",
        CUBLAS_STATUS_EXECUTION_FAILED => "CUBLAS_STATUS_EXECUTION_FAILED",
        CUBLAS_STATUS_INTERNAL_ERROR => "CUBLAS_STATUS_INTERNAL_ERROR",
        CUBLAS_STATUS_NOT_SUPPORTED => "CUBLAS_STATUS_NOT_SUPPORTED",
        _ => "CUBLAS_STATUS_UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_strings() {
        assert_eq!(cublas_status_string(CUBLAS_STATUS_SUCCESS), "CUBLAS_STATUS_SUCCESS");
        assert_eq!(
            cublas_status_string(CUBLAS_STATUS_INVALID_VALUE),
            "CUBLAS_STATUS_INVALID_VALUE"
        );
        assert_eq!(cublas_status_string(999), "CUBLAS_STATUS_UNKNOWN");
    }

    #[test]
    fn test_operation_constants() {
        assert_eq!(CUBLAS_OP_N, 0);
        assert_eq!(CUBLAS_OP_T, 1);
    }

    #[test]
    fn test_data_type_constants() {
        assert_eq!(CUDA_R_16F, 2);
        assert_eq!(CUDA_R_32F, 0);
    }

    #[test]
    fn test_compute_type_constants() {
        assert_eq!(CUBLAS_COMPUTE_32F, 68);
        assert_eq!(CUBLAS_COMPUTE_32F_FAST_TF32, 74);
        assert_eq!(CUBLAS_COMPUTE_16F, 64);
    }

    #[test]
    fn test_math_mode_constants() {
        assert_eq!(CUBLAS_DEFAULT_MATH, 0);
        assert_eq!(CUBLAS_TENSOR_OP_MATH, 1);
        assert_eq!(CUBLAS_PEDANTIC_MATH, 2);
        assert_eq!(CUBLAS_TF32_TENSOR_OP_MATH, 3);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_cublas_not_available_without_feature() {
        assert!(CublasDriver::load().is_none());
    }
}
