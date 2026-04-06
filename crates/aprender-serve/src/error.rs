//! Error types for Realizar
//!
//! This module defines all error types used throughout the library.

use thiserror::Error;

/// Result type alias for Realizar operations
pub type Result<T> = std::result::Result<T, RealizarError>;

/// Error type for all Realizar operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RealizarError {
    /// Shape mismatch between tensors
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid shape specification
    #[error("Invalid shape: {reason}")]
    InvalidShape {
        /// Reason for invalidity
        reason: String,
    },

    /// Data size does not match shape
    #[error("Data size {data_size} does not match shape {shape:?} (expected {expected})")]
    DataShapeMismatch {
        /// Actual data size
        data_size: usize,
        /// Specified shape
        shape: Vec<usize>,
        /// Expected size from shape
        expected: usize,
    },

    /// Invalid dimension for operation
    #[error("Invalid dimension {dim} for tensor with {ndim} dimensions")]
    InvalidDimension {
        /// Requested dimension
        dim: usize,
        /// Number of dimensions
        ndim: usize,
    },

    /// Matrix multiplication dimension mismatch
    #[error("Matrix multiplication dimension mismatch: ({m}×{k}) × ({k2}×{n})")]
    MatmulDimensionMismatch {
        /// Rows in first matrix
        m: usize,
        /// Columns in first matrix / rows in second matrix
        k: usize,
        /// Rows in second matrix (should equal k)
        k2: usize,
        /// Columns in second matrix
        n: usize,
    },

    /// Operation not supported for this tensor type
    #[error("Operation '{operation}' not supported: {reason}")]
    UnsupportedOperation {
        /// Operation name
        operation: String,
        /// Reason it's not supported
        reason: String,
    },

    /// Trueno backend error
    #[error("Trueno backend error: {0}")]
    TruenoError(String),

    /// Index out of bounds
    #[error("Index out of bounds: index {index} for dimension of size {size}")]
    IndexOutOfBounds {
        /// Requested index
        index: usize,
        /// Size of dimension
        size: usize,
    },

    /// Model registry error
    #[error("Model registry error: {0}")]
    RegistryError(String),

    /// Model not found in registry
    #[error("Model '{0}' not found")]
    ModelNotFound(String),

    /// Model already registered
    #[error("Model '{0}' already registered")]
    ModelAlreadyExists(String),

    /// MOE routing error
    #[error("MOE routing error: {0}")]
    MoeError(String),

    /// Expert capacity exceeded
    #[error("Expert {expert_id} capacity exceeded: {queue_depth}/{capacity}")]
    ExpertCapacityExceeded {
        /// Expert index
        expert_id: usize,
        /// Current queue depth
        queue_depth: usize,
        /// Maximum capacity
        capacity: usize,
    },

    /// Invalid URI format
    #[error("Invalid URI: {0}")]
    InvalidUri(String),

    /// File format error
    #[error("Format error: {reason}")]
    FormatError {
        /// Reason for format error
        reason: String,
    },

    /// IO error
    #[error("IO error: {message}")]
    IoError {
        /// Error message
        message: String,
    },

    /// Security error (F-SEC-222: path traversal, unauthorized access)
    #[error("Security error: {reason}")]
    SecurityError {
        /// Reason for security error
        reason: String,
    },

    /// Connection error (network/HTTP)
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// GPU compute error
    #[error("GPU error: {reason}")]
    GpuError {
        /// Reason for GPU error
        reason: String,
    },

    /// Invalid configuration error
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Inference execution error
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// GH-167: Context limit exceeded (prompt too long for model)
    #[error("Context limit exceeded: {provided} tokens > max {maximum}. Hint: Use --truncate or reduce prompt length.")]
    ContextLimitExceeded {
        /// Number of tokens provided
        provided: usize,
        /// Maximum context length
        maximum: usize,
    },

    /// GH-280: GPU capability mismatch — model requires ops the GPU backend lacks
    #[error("Capability mismatch for '{architecture}': missing GPU support for [{missing_ops}]. {suggestion}")]
    CapabilityMismatch {
        /// Model architecture name
        architecture: String,
        /// Comma-separated list of missing operations
        missing_ops: String,
        /// Actionable suggestion (e.g., "Use CPU inference")
        suggestion: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RealizarError::ShapeMismatch {
            expected: vec![3, 3],
            actual: vec![2, 2],
        };
        assert!(err.to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = RealizarError::InvalidShape {
            reason: "Empty shape".to_string(),
        };
        let err2 = RealizarError::InvalidShape {
            reason: "Empty shape".to_string(),
        };
        assert_eq!(err1, err2);
    }

    // ========================================================================
    // Coverage Tests: All Error Variants
    // ========================================================================

    #[test]
    fn test_data_shape_mismatch_display() {
        let err = RealizarError::DataShapeMismatch {
            data_size: 10,
            shape: vec![2, 3],
            expected: 6,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("6"));
    }

    #[test]
    fn test_invalid_dimension_display() {
        let err = RealizarError::InvalidDimension { dim: 5, ndim: 3 };
        let msg = err.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_matmul_dimension_mismatch_display() {
        let err = RealizarError::MatmulDimensionMismatch {
            m: 2,
            k: 3,
            k2: 4,
            n: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("2"));
        assert!(msg.contains("3"));
        assert!(msg.contains("4"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_unsupported_operation_display() {
        let err = RealizarError::UnsupportedOperation {
            operation: "transpose".to_string(),
            reason: "not implemented".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("transpose"));
        assert!(msg.contains("not implemented"));
    }

    #[test]
    fn test_trueno_error_display() {
        let err = RealizarError::TruenoError("kernel failed".to_string());
        assert!(err.to_string().contains("kernel failed"));
    }

    #[test]
    fn test_index_out_of_bounds_display() {
        let err = RealizarError::IndexOutOfBounds { index: 10, size: 5 };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_registry_error_display() {
        let err = RealizarError::RegistryError("lock failed".to_string());
        assert!(err.to_string().contains("lock failed"));
    }

    #[test]
    fn test_model_not_found_display() {
        let err = RealizarError::ModelNotFound("llama-7b".to_string());
        assert!(err.to_string().contains("llama-7b"));
    }

    #[test]
    fn test_model_already_exists_display() {
        let err = RealizarError::ModelAlreadyExists("phi-2".to_string());
        assert!(err.to_string().contains("phi-2"));
    }

    #[test]
    fn test_moe_error_display() {
        let err = RealizarError::MoeError("routing failed".to_string());
        assert!(err.to_string().contains("routing failed"));
    }

    #[test]
    fn test_expert_capacity_exceeded_display() {
        let err = RealizarError::ExpertCapacityExceeded {
            expert_id: 3,
            queue_depth: 10,
            capacity: 8,
        };
        let msg = err.to_string();
        assert!(msg.contains("3"));
        assert!(msg.contains("10"));
        assert!(msg.contains("8"));
    }

    #[test]
    fn test_invalid_uri_display() {
        let err = RealizarError::InvalidUri("bad://url".to_string());
        assert!(err.to_string().contains("bad://url"));
    }

    #[test]
    fn test_format_error_display() {
        let err = RealizarError::FormatError {
            reason: "invalid header".to_string(),
        };
        assert!(err.to_string().contains("invalid header"));
    }

    #[test]
    fn test_io_error_display() {
        let err = RealizarError::IoError {
            message: "file not found".to_string(),
        };
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_connection_error_display() {
        let err = RealizarError::ConnectionError("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_gpu_error_display() {
        let err = RealizarError::GpuError {
            reason: "out of memory".to_string(),
        };
        assert!(err.to_string().contains("out of memory"));
    }

    #[test]
    fn test_invalid_configuration_display() {
        let err = RealizarError::InvalidConfiguration("missing field".to_string());
        assert!(err.to_string().contains("missing field"));
    }

    #[test]
    fn test_inference_error_display() {
        let err = RealizarError::InferenceError("model failed".to_string());
        assert!(err.to_string().contains("model failed"));
    }

    #[test]
    fn test_error_debug() {
        let err = RealizarError::ShapeMismatch {
            expected: vec![1, 2],
            actual: vec![3, 4],
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("ShapeMismatch"));
    }

    #[test]
    fn test_error_clone() {
        let err = RealizarError::InvalidShape {
            reason: "test".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    /// GH-280: Test CapabilityMismatch error formatting
    #[test]
    fn test_capability_mismatch_display() {
        let err = RealizarError::CapabilityMismatch {
            architecture: "qwen3".to_string(),
            missing_ops: "QkNorm".to_string(),
            suggestion: "Use CPU inference".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("qwen3"), "Should show architecture");
        assert!(msg.contains("QkNorm"), "Should show missing ops");
        assert!(msg.contains("CPU inference"), "Should show suggestion");
    }

    /// GH-167: Test ContextLimitExceeded error formatting
    #[test]
    fn test_context_limit_exceeded_display() {
        let err = RealizarError::ContextLimitExceeded {
            provided: 8192,
            maximum: 4096,
        };
        let msg = err.to_string();
        assert!(msg.contains("8192"), "Should show provided tokens");
        assert!(msg.contains("4096"), "Should show maximum allowed");
        assert!(
            msg.contains("Context") || msg.contains("context"),
            "Should mention context"
        );
        assert!(msg.contains("Hint"), "Should provide helpful hint");
    }
}
