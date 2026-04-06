//! Error types for trueno-gpu operations
//!
//! Provides comprehensive error handling for PTX generation, CUDA driver operations,
//! and memory management.
//!
//! Design: Toyota Principle #7 (Visual Control) - Clear error messages with GPU state context

use thiserror::Error;

/// Result type alias for trueno-gpu operations
pub type Result<T> = std::result::Result<T, GpuError>;

/// Errors that can occur during GPU operations
#[derive(Error, Debug)]
pub enum GpuError {
    /// PTX generation error
    #[error("PTX generation error: {0}")]
    PtxGeneration(String),

    /// I/O error (file operations)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Invalid PTX version
    #[error("Invalid PTX version: {major}.{minor} (requires >= 7.0)")]
    InvalidPtxVersion {
        /// Major version
        major: u32,
        /// Minor version
        minor: u32,
    },

    /// Invalid compute capability target
    #[error("Invalid compute capability: {0} (requires sm_70+)")]
    InvalidTarget(String),

    /// CUDA driver error
    #[error("CUDA driver error: {0} (code: {1})")]
    CudaDriver(String, i32),

    /// Memory allocation error
    #[error("GPU memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// Kernel launch error
    #[error("Kernel launch failed: {0}")]
    KernelLaunch(String),

    /// Invalid kernel configuration
    #[error("Invalid launch config: {0}")]
    InvalidLaunchConfig(String),

    /// Register allocation error
    #[error("Register allocation failed: {0}")]
    RegisterAllocation(String),

    /// Bank conflict detected (debugging)
    #[error("Bank conflict detected in shared memory access")]
    BankConflict,

    // =========================================================================
    // CUDA Runtime Errors (CRT-001 to CRT-006)
    // =========================================================================
    /// CUDA device initialization failed
    #[error("CUDA device initialization failed: {0}")]
    DeviceInit(String),

    /// CUDA device not found
    #[error("CUDA device {0} not found (available: {1})")]
    DeviceNotFound(i32, usize),

    /// CUDA module/PTX loading failed
    #[error("CUDA module loading failed: {0}")]
    ModuleLoad(String),

    /// CUDA function not found in module
    #[error("CUDA function '{0}' not found in module")]
    FunctionNotFound(String),

    /// CUDA stream creation failed
    #[error("CUDA stream creation failed: {0}")]
    StreamCreate(String),

    /// CUDA stream synchronization failed
    #[error("CUDA stream synchronization failed: {0}")]
    StreamSync(String),

    /// CUDA memory transfer (H2D/D2H) failed
    #[error("CUDA memory transfer failed: {0}")]
    Transfer(String),

    /// Out of GPU memory
    #[error("Out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Bytes requested
        requested: usize,
        /// Bytes available
        available: usize,
    },

    /// CUDA not available (no driver or no GPU)
    #[error("CUDA not available: {0}")]
    CudaNotAvailable(String),

    /// Feature not supported on this device/platform
    #[error("Not supported: {0}")]
    NotSupported(String),

    // =========================================================================
    // CUDA Graph Errors (PAR-037)
    // =========================================================================
    /// CUDA graph creation failed
    #[error("CUDA graph creation failed: {0}")]
    GraphCreate(String),

    /// CUDA graph capture failed
    #[error("CUDA graph capture failed: {0}")]
    GraphCapture(String),

    /// CUDA graph instantiation failed
    #[error("CUDA graph instantiation failed: {0}")]
    GraphInstantiate(String),

    /// CUDA graph launch failed
    #[error("CUDA graph launch failed: {0}")]
    GraphLaunch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_generation_error() {
        let err = GpuError::PtxGeneration("invalid instruction".to_string());
        assert!(err.to_string().contains("PTX generation error"));
        assert!(err.to_string().contains("invalid instruction"));
    }

    #[test]
    fn test_invalid_ptx_version() {
        let err = GpuError::InvalidPtxVersion { major: 6, minor: 5 };
        assert!(err.to_string().contains("6.5"));
        assert!(err.to_string().contains("requires >= 7.0"));
    }

    #[test]
    fn test_invalid_target() {
        let err = GpuError::InvalidTarget("sm_50".to_string());
        assert!(err.to_string().contains("sm_50"));
        assert!(err.to_string().contains("requires sm_70+"));
    }

    #[test]
    fn test_cuda_driver_error() {
        let err = GpuError::CudaDriver("out of memory".to_string(), 2);
        assert!(err.to_string().contains("out of memory"));
        assert!(err.to_string().contains("code: 2"));
    }

    #[test]
    fn test_memory_allocation_error() {
        let err = GpuError::MemoryAllocation("insufficient device memory".to_string());
        assert!(err.to_string().contains("allocation failed"));
    }

    #[test]
    fn test_kernel_launch_error() {
        let err = GpuError::KernelLaunch("invalid grid dimensions".to_string());
        assert!(err.to_string().contains("launch failed"));
    }

    #[test]
    fn test_error_debug() {
        let err = GpuError::BankConflict;
        // Just verify Debug is implemented
        let _ = format!("{:?}", err);
    }

    #[test]
    fn test_error_display() {
        let err = GpuError::PtxGeneration("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: GpuError = io_err.into();
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_invalid_parameter() {
        let err = GpuError::InvalidParameter("bad value".to_string());
        assert!(err.to_string().contains("Invalid parameter"));
        assert!(err.to_string().contains("bad value"));
    }

    // =========================================================================
    // CUDA Runtime Error Tests (CRT-001 to CRT-006)
    // =========================================================================

    #[test]
    fn test_device_init_error() {
        let err = GpuError::DeviceInit("no CUDA driver".to_string());
        assert!(err.to_string().contains("initialization failed"));
        assert!(err.to_string().contains("no CUDA driver"));
    }

    #[test]
    fn test_device_not_found_error() {
        let err = GpuError::DeviceNotFound(5, 2);
        assert!(err.to_string().contains("device 5"));
        assert!(err.to_string().contains("available: 2"));
    }

    #[test]
    fn test_module_load_error() {
        let err = GpuError::ModuleLoad("invalid PTX".to_string());
        assert!(err.to_string().contains("module loading failed"));
    }

    #[test]
    fn test_function_not_found_error() {
        let err = GpuError::FunctionNotFound("my_kernel".to_string());
        assert!(err.to_string().contains("my_kernel"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_stream_create_error() {
        let err = GpuError::StreamCreate("resource exhausted".to_string());
        assert!(err.to_string().contains("stream creation"));
    }

    #[test]
    fn test_stream_sync_error() {
        let err = GpuError::StreamSync("timeout".to_string());
        assert!(err.to_string().contains("synchronization"));
    }

    #[test]
    fn test_transfer_error() {
        let err = GpuError::Transfer("DMA error".to_string());
        assert!(err.to_string().contains("transfer failed"));
    }

    #[test]
    fn test_out_of_memory_error() {
        let err = GpuError::OutOfMemory { requested: 1_000_000_000, available: 500_000_000 };
        assert!(err.to_string().contains("1000000000"));
        assert!(err.to_string().contains("500000000"));
    }

    #[test]
    fn test_cuda_not_available_error() {
        let err = GpuError::CudaNotAvailable("no GPU detected".to_string());
        assert!(err.to_string().contains("not available"));
    }

    #[test]
    fn test_not_supported_error() {
        let err = GpuError::NotSupported("CPU temperature monitoring".to_string());
        assert!(err.to_string().contains("Not supported"));
        assert!(err.to_string().contains("CPU temperature"));
    }
}
