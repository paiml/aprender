//! Error types for apr-cli
//!
//! Toyota Way: Jidoka - Stop and highlight problems immediately.

use std::path::PathBuf;
use std::process::ExitCode;
use thiserror::Error;

/// Result type alias for CLI operations
pub(crate) type Result<T> = std::result::Result<T, CliError>;

/// CLI error types
#[derive(Error, Debug)]
pub(crate) enum CliError {
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
    pub(crate) fn exit_code(&self) -> ExitCode {
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
