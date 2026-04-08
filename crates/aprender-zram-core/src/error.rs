//! Error types for trueno-zram-core.

use thiserror::Error;

/// Errors that can occur during compression operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Input data is invalid or malformed.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Compressed data is corrupted or truncated.
    #[error("corrupted data: {0}")]
    CorruptedData(String),

    /// Output buffer is too small.
    #[error("buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall {
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        available: usize,
    },

    /// Unsupported feature or algorithm.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// SIMD backend not available on this CPU.
    #[error("SIMD backend {0:?} not available on this CPU")]
    SimdNotAvailable(crate::SimdBackend),

    /// GPU not available or unsupported.
    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),

    /// I/O error (filesystem, device access).
    #[error("I/O error: {0}")]
    IoError(String),

    /// Internal error (should not happen).
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type for compression operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_invalid_input() {
        let err = Error::InvalidInput("page size must be 4096".to_string());
        assert!(err.to_string().contains("invalid input"));
        assert!(err.to_string().contains("4096"));
    }

    #[test]
    fn test_error_display_corrupted_data() {
        let err = Error::CorruptedData("invalid magic bytes".to_string());
        assert!(err.to_string().contains("corrupted data"));
    }

    #[test]
    fn test_error_display_buffer_too_small() {
        let err = Error::BufferTooSmall {
            needed: 4096,
            available: 1024,
        };
        let msg = err.to_string();
        assert!(msg.contains("4096"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_error_implements_std_error() {
        fn assert_std_error<T: std::error::Error>() {}
        assert_std_error::<Error>();
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn test_error_display_gpu_not_available() {
        let err = Error::GpuNotAvailable("CUDA feature not enabled".to_string());
        let msg = err.to_string();
        assert!(msg.contains("GPU not available"));
        assert!(msg.contains("CUDA"));
    }
}
