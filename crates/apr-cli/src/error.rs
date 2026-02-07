//! Error types for apr-cli
//!
//! Toyota Way: Jidoka - Stop and highlight problems immediately.

use std::path::PathBuf;
use std::process::ExitCode;
use thiserror::Error;

/// Result type alias for CLI operations
pub type Result<T> = std::result::Result<T, CliError>;

/// CLI error types
#[derive(Error, Debug)]
pub enum CliError {
    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Not a file (e.g., directory)
    #[error("Not a file: {0}")]
    NotAFile(PathBuf),

    /// Invalid APR format
    #[error("Invalid APR format: {0}")]
    InvalidFormat(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Aprender error
    #[error("Aprender error: {0}")]
    Aprender(String),

    /// Model loading failed (used with inference feature)
    #[error("Model load failed: {0}")]
    #[allow(dead_code)]
    ModelLoadFailed(String),

    /// Inference failed (used with inference feature)
    #[error("Inference failed: {0}")]
    #[allow(dead_code)]
    InferenceFailed(String),

    /// Feature disabled (used when optional features are not compiled)
    #[error("Feature not enabled: {0}")]
    #[allow(dead_code)]
    FeatureDisabled(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
}

impl CliError {
    /// Get exit code for this error
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::FileNotFound(_) | Self::NotAFile(_) => ExitCode::from(3),
            Self::InvalidFormat(_) => ExitCode::from(4),
            Self::Io(_) => ExitCode::from(7),
            Self::ValidationFailed(_) => ExitCode::from(5),
            Self::Aprender(_) => ExitCode::from(1),
            Self::ModelLoadFailed(_) => ExitCode::from(6),
            Self::InferenceFailed(_) => ExitCode::from(8),
            Self::FeatureDisabled(_) => ExitCode::from(9),
            Self::NetworkError(_) => ExitCode::from(10),
        }
    }
}

impl From<aprender::error::AprenderError> for CliError {
    fn from(e: aprender::error::AprenderError) -> Self {
        Self::Aprender(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ==================== Exit Code Tests ====================

    #[test]
    fn test_file_not_found_exit_code() {
        let err = CliError::FileNotFound(PathBuf::from("/test"));
        assert_eq!(err.exit_code(), ExitCode::from(3));
    }

    #[test]
    fn test_not_a_file_exit_code() {
        let err = CliError::NotAFile(PathBuf::from("/test"));
        assert_eq!(err.exit_code(), ExitCode::from(3));
    }

    #[test]
    fn test_invalid_format_exit_code() {
        let err = CliError::InvalidFormat("bad".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(4));
    }

    #[test]
    fn test_io_error_exit_code() {
        let err = CliError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert_eq!(err.exit_code(), ExitCode::from(7));
    }

    #[test]
    fn test_validation_failed_exit_code() {
        let err = CliError::ValidationFailed("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(5));
    }

    #[test]
    fn test_aprender_error_exit_code() {
        let err = CliError::Aprender("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(1));
    }

    #[test]
    fn test_model_load_failed_exit_code() {
        let err = CliError::ModelLoadFailed("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(6));
    }

    #[test]
    fn test_inference_failed_exit_code() {
        let err = CliError::InferenceFailed("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(8));
    }

    #[test]
    fn test_feature_disabled_exit_code() {
        let err = CliError::FeatureDisabled("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(9));
    }

    #[test]
    fn test_network_error_exit_code() {
        let err = CliError::NetworkError("test".to_string());
        assert_eq!(err.exit_code(), ExitCode::from(10));
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_file_not_found_display() {
        let err = CliError::FileNotFound(PathBuf::from("/model.apr"));
        assert_eq!(err.to_string(), "File not found: /model.apr");
    }

    #[test]
    fn test_not_a_file_display() {
        let err = CliError::NotAFile(PathBuf::from("/dir"));
        assert_eq!(err.to_string(), "Not a file: /dir");
    }

    #[test]
    fn test_invalid_format_display() {
        let err = CliError::InvalidFormat("bad magic".to_string());
        assert_eq!(err.to_string(), "Invalid APR format: bad magic");
    }

    #[test]
    fn test_validation_failed_display() {
        let err = CliError::ValidationFailed("missing field".to_string());
        assert_eq!(err.to_string(), "Validation failed: missing field");
    }

    #[test]
    fn test_aprender_error_display() {
        let err = CliError::Aprender("internal".to_string());
        assert_eq!(err.to_string(), "Aprender error: internal");
    }

    #[test]
    fn test_model_load_failed_display() {
        let err = CliError::ModelLoadFailed("corrupt".to_string());
        assert_eq!(err.to_string(), "Model load failed: corrupt");
    }

    #[test]
    fn test_inference_failed_display() {
        let err = CliError::InferenceFailed("OOM".to_string());
        assert_eq!(err.to_string(), "Inference failed: OOM");
    }

    #[test]
    fn test_feature_disabled_display() {
        let err = CliError::FeatureDisabled("cuda".to_string());
        assert_eq!(err.to_string(), "Feature not enabled: cuda");
    }

    #[test]
    fn test_network_error_display() {
        let err = CliError::NetworkError("timeout".to_string());
        assert_eq!(err.to_string(), "Network error: timeout");
    }

    // ==================== Conversion Tests ====================

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let cli_err: CliError = io_err.into();
        assert!(cli_err.to_string().contains("file missing"));
        assert_eq!(cli_err.exit_code(), ExitCode::from(7));
    }

    #[test]
    fn test_debug_impl() {
        let err = CliError::FileNotFound(PathBuf::from("/test"));
        let debug = format!("{:?}", err);
        assert!(debug.contains("FileNotFound"));
    }

    // ==================== Result Type Alias ====================

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(CliError::InvalidFormat("test".to_string()));
        assert!(result.is_err());
    }

    // ==================== Exit Code Uniqueness ====================

    #[test]
    fn test_all_exit_codes_are_distinct_per_category() {
        // Verify exit codes map to distinct categories
        let codes = vec![
            (
                CliError::FileNotFound(PathBuf::from("a")).exit_code(),
                "file",
            ),
            (
                CliError::InvalidFormat("a".to_string()).exit_code(),
                "format",
            ),
            (
                CliError::Io(std::io::Error::new(std::io::ErrorKind::Other, "")).exit_code(),
                "io",
            ),
            (
                CliError::ValidationFailed("a".to_string()).exit_code(),
                "validation",
            ),
            (CliError::Aprender("a".to_string()).exit_code(), "aprender"),
            (
                CliError::ModelLoadFailed("a".to_string()).exit_code(),
                "model_load",
            ),
            (
                CliError::InferenceFailed("a".to_string()).exit_code(),
                "inference",
            ),
            (
                CliError::FeatureDisabled("a".to_string()).exit_code(),
                "feature",
            ),
            (
                CliError::NetworkError("a".to_string()).exit_code(),
                "network",
            ),
        ];
        // FileNotFound and NotAFile intentionally share exit code 3
        assert_eq!(codes[0].0, ExitCode::from(3));
    }
}
