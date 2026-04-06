//! Error types for Trueno operations

use thiserror::Error;

use crate::Backend;

/// Result type for Trueno operations
pub type Result<T> = std::result::Result<T, TruenoError>;

/// Errors that can occur during Trueno operations
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TruenoError {
    /// Backend not supported on this platform
    #[error("Backend not supported on this platform: {0:?}")]
    UnsupportedBackend(Backend),

    /// Size mismatch between operands
    #[error("Size mismatch: expected {expected}, got {actual}")]
    SizeMismatch {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Division by zero (e.g., normalizing zero vector)
    #[error("Division by zero")]
    DivisionByZero,

    /// Empty vector (e.g., computing mean of empty vector)
    #[error("Empty vector")]
    EmptyVector,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_backend_error() {
        let err = TruenoError::UnsupportedBackend(Backend::AVX512);
        assert_eq!(err.to_string(), "Backend not supported on this platform: AVX512");
    }

    #[test]
    fn test_size_mismatch_error() {
        let err = TruenoError::SizeMismatch { expected: 10, actual: 5 };
        assert_eq!(err.to_string(), "Size mismatch: expected 10, got 5");
    }

    #[test]
    fn test_gpu_error() {
        let err = TruenoError::GpuError("Device not found".to_string());
        assert_eq!(err.to_string(), "GPU error: Device not found");
    }

    #[test]
    fn test_invalid_input_error() {
        let err = TruenoError::InvalidInput("Empty vector".to_string());
        assert_eq!(err.to_string(), "Invalid input: Empty vector");
    }

    #[test]
    fn test_error_equality() {
        let err1 = TruenoError::SizeMismatch { expected: 10, actual: 5 };
        let err2 = TruenoError::SizeMismatch { expected: 10, actual: 5 };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_division_by_zero_error() {
        let err = TruenoError::DivisionByZero;
        assert_eq!(err.to_string(), "Division by zero");
    }

    #[test]
    fn test_empty_vector_error() {
        let err = TruenoError::EmptyVector;
        assert_eq!(err.to_string(), "Empty vector");
    }

    // =========================================================================
    // Debug trait coverage
    // =========================================================================

    #[test]
    fn test_error_debug_unsupported_backend() {
        let err = TruenoError::UnsupportedBackend(Backend::AVX512);
        let debug = format!("{:?}", err);
        assert!(debug.contains("UnsupportedBackend"));
        assert!(debug.contains("AVX512"));
    }

    #[test]
    fn test_error_debug_size_mismatch() {
        let err = TruenoError::SizeMismatch { expected: 100, actual: 50 };
        let debug = format!("{:?}", err);
        assert!(debug.contains("SizeMismatch"));
        assert!(debug.contains("100"));
        assert!(debug.contains("50"));
    }

    #[test]
    fn test_error_debug_gpu_error() {
        let err = TruenoError::GpuError("out of memory".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("GpuError"));
        assert!(debug.contains("out of memory"));
    }

    #[test]
    fn test_error_debug_invalid_input() {
        let err = TruenoError::InvalidInput("negative dimension".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidInput"));
        assert!(debug.contains("negative dimension"));
    }

    #[test]
    fn test_error_debug_division_by_zero() {
        let err = TruenoError::DivisionByZero;
        let debug = format!("{:?}", err);
        assert!(debug.contains("DivisionByZero"));
    }

    #[test]
    fn test_error_debug_empty_vector() {
        let err = TruenoError::EmptyVector;
        let debug = format!("{:?}", err);
        assert!(debug.contains("EmptyVector"));
    }

    // =========================================================================
    // Error inequality (PartialEq, Eq)
    // =========================================================================

    #[test]
    fn test_error_inequality_different_variants() {
        let err1 = TruenoError::DivisionByZero;
        let err2 = TruenoError::EmptyVector;
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_error_inequality_different_values() {
        let err1 = TruenoError::SizeMismatch { expected: 10, actual: 5 };
        let err2 = TruenoError::SizeMismatch { expected: 20, actual: 5 };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_error_equality_gpu_errors() {
        let err1 = TruenoError::GpuError("same".to_string());
        let err2 = TruenoError::GpuError("same".to_string());
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_error_inequality_gpu_errors() {
        let err1 = TruenoError::GpuError("err a".to_string());
        let err2 = TruenoError::GpuError("err b".to_string());
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_error_equality_invalid_input() {
        let err1 = TruenoError::InvalidInput("test".to_string());
        let err2 = TruenoError::InvalidInput("test".to_string());
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_error_inequality_invalid_input() {
        let err1 = TruenoError::InvalidInput("a".to_string());
        let err2 = TruenoError::InvalidInput("b".to_string());
        assert_ne!(err1, err2);
    }

    // =========================================================================
    // Display formatting for all variants (with varied content)
    // =========================================================================

    #[test]
    fn test_unsupported_backend_all_backends() {
        let backends = [
            (Backend::Scalar, "Scalar"),
            (Backend::SSE2, "SSE2"),
            (Backend::AVX, "AVX"),
            (Backend::AVX2, "AVX2"),
            (Backend::AVX512, "AVX512"),
            (Backend::NEON, "NEON"),
            (Backend::WasmSIMD, "WasmSIMD"),
            (Backend::GPU, "GPU"),
            (Backend::Auto, "Auto"),
        ];
        for (backend, name) in backends {
            let err = TruenoError::UnsupportedBackend(backend);
            let msg = err.to_string();
            assert!(msg.contains(name), "Expected '{}' in '{}'", name, msg);
        }
    }

    #[test]
    fn test_size_mismatch_various_values() {
        let err = TruenoError::SizeMismatch { expected: 0, actual: 0 };
        assert_eq!(err.to_string(), "Size mismatch: expected 0, got 0");

        let err = TruenoError::SizeMismatch { expected: 1000000, actual: 999999 };
        assert!(err.to_string().contains("1000000"));
        assert!(err.to_string().contains("999999"));
    }

    #[test]
    fn test_gpu_error_empty_message() {
        let err = TruenoError::GpuError(String::new());
        assert_eq!(err.to_string(), "GPU error: ");
    }

    #[test]
    fn test_invalid_input_empty_message() {
        let err = TruenoError::InvalidInput(String::new());
        assert_eq!(err.to_string(), "Invalid input: ");
    }

    // =========================================================================
    // std::error::Error trait coverage
    // =========================================================================

    #[test]
    fn test_error_is_std_error() {
        let err = TruenoError::DivisionByZero;
        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_error_source_is_none() {
        use std::error::Error;
        let err = TruenoError::DivisionByZero;
        assert!(err.source().is_none());

        let err = TruenoError::EmptyVector;
        assert!(err.source().is_none());

        let err = TruenoError::GpuError("test".to_string());
        assert!(err.source().is_none());

        let err = TruenoError::InvalidInput("test".to_string());
        assert!(err.source().is_none());

        let err = TruenoError::SizeMismatch { expected: 1, actual: 2 };
        assert!(err.source().is_none());

        let err = TruenoError::UnsupportedBackend(Backend::Scalar);
        assert!(err.source().is_none());
    }

    // =========================================================================
    // Result type alias coverage
    // =========================================================================

    #[test]
    fn test_result_type_alias_ok() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_result_type_alias_err() {
        let result: Result<i32> = Err(TruenoError::EmptyVector);
        assert!(result.is_err());
    }

    // =========================================================================
    // Clone / copy semantics
    // =========================================================================

    #[test]
    fn test_error_clone_all_variants() {
        let errors: Vec<TruenoError> = vec![
            TruenoError::UnsupportedBackend(Backend::AVX512),
            TruenoError::SizeMismatch { expected: 10, actual: 5 },
            TruenoError::GpuError("test clone".to_string()),
            TruenoError::InvalidInput("test clone".to_string()),
            TruenoError::DivisionByZero,
            TruenoError::EmptyVector,
        ];

        for err in &errors {
            let cloned = err.clone();
            assert_eq!(*err, cloned, "Clone mismatch for {:?}", err);
        }
    }
}
