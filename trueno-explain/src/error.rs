//! Error types for trueno-explain

use thiserror::Error;

/// Errors that can occur during analysis
#[derive(Error, Debug)]
pub enum ExplainError {
    /// PTX parsing failed
    #[error("Failed to parse PTX: {0}")]
    PtxParseError(String),

    /// Requested kernel not found
    #[error("Invalid kernel name: {0}")]
    InvalidKernel(String),

    /// File I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type alias for trueno-explain operations
pub type Result<T> = std::result::Result<T, ExplainError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ExplainError::PtxParseError("unexpected token".to_string());
        assert!(err.to_string().contains("unexpected token"));
    }

    #[test]
    fn test_invalid_kernel_error() {
        let err = ExplainError::InvalidKernel("bad_kernel".to_string());
        assert!(err.to_string().contains("bad_kernel"));
    }
}
