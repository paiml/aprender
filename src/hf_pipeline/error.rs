//! Error types for HuggingFace pipeline operations
//!
//! Comprehensive error handling with retry hints and recovery options.

use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Result type for HF pipeline operations
pub type Result<T> = std::result::Result<T, FetchError>;

/// Errors that can occur during HuggingFace operations
///
/// Designed for Jidoka (built-in quality) - explicit error types enable
/// proper error handling and recovery strategies.
#[derive(Debug, Error)]
pub enum FetchError {
    /// Network timeout during download
    #[error("Network timeout for {repo} after {elapsed:?}")]
    NetworkTimeout { repo: String, elapsed: Duration },

    /// Rate limited by HuggingFace API
    #[error("Rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Duration },

    /// Model repository not found
    #[error("Repository not found: {repo}")]
    ModelNotFound { repo: String },

    /// File not found in repository
    #[error("File not found in {repo}: {file}")]
    FileNotFound { repo: String, file: String },

    /// Downloaded file is corrupt (checksum mismatch)
    #[error("Corrupt file at {path}: expected SHA256 {expected_hash}, got {actual_hash}")]
    CorruptFile { path: PathBuf, expected_hash: String, actual_hash: String },

    /// Insufficient disk space
    #[error("Insufficient disk space: need {required} bytes, have {available} bytes")]
    InsufficientDisk { required: u64, available: u64 },

    /// Out of memory during model loading
    #[error("Out of memory: model requires {required} bytes, available {available} bytes")]
    OutOfMemory { required: u64, available: u64 },

    /// Authentication failed
    #[error("Authentication failed: {message}")]
    AuthenticationFailed { message: String },

    /// Missing authentication token
    #[error("Missing HF_TOKEN - set environment variable or use with_token()")]
    MissingToken,

    /// Invalid repository ID format
    #[error("Invalid repository ID format (expected 'org/name'): {repo_id}")]
    InvalidRepoId { repo_id: String },

    /// Unsupported model format
    #[error("Unsupported model format: {format}")]
    UnsupportedFormat { format: String },

    /// SECURITY: PyTorch pickle file detected
    #[error("SECURITY: PyTorch .bin files may contain arbitrary code. Enable allow_pytorch_pickle to proceed.")]
    PickleSecurityRisk,

    /// Model config parsing error
    #[error("Failed to parse config.json: {message}")]
    ConfigParseError { message: String },

    /// Tensor shape mismatch
    #[error("Tensor shape mismatch for {tensor}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { tensor: String, expected: Vec<usize>, actual: Vec<usize> },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// SafeTensors parsing error
    #[error("SafeTensors parse error: {message}")]
    SafeTensorsParseError { message: String },

    /// Leaderboard not found
    #[error("Leaderboard not found: {kind}")]
    LeaderboardNotFound { kind: String },

    /// Leaderboard data parsing error
    #[error("Failed to parse leaderboard data: {message}")]
    LeaderboardParseError { message: String },

    /// HTTP request error
    #[error("HTTP error: {message}")]
    HttpError { message: String },

    /// GGUF write/serialization error
    #[error("GGUF write error: {message}")]
    GgufWriteError { message: String },
}

impl FetchError {
    /// Check if error is retryable
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::NetworkTimeout { .. } | Self::RateLimited { .. })
    }

    /// Get retry delay if applicable
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after } => Some(*retry_after),
            Self::NetworkTimeout { .. } => Some(Duration::from_secs(5)),
            _ => None,
        }
    }

    /// Check if error is a security concern
    #[must_use]
    pub fn is_security_risk(&self) -> bool {
        matches!(self, Self::PickleSecurityRisk | Self::CorruptFile { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_timeout_is_retryable() {
        let err = FetchError::NetworkTimeout {
            repo: "test/model".into(),
            elapsed: Duration::from_secs(30),
        };
        assert!(err.is_retryable());
        assert!(err.retry_after().is_some());
    }

    #[test]
    fn test_rate_limited_is_retryable() {
        let err = FetchError::RateLimited { retry_after: Duration::from_secs(60) };
        assert!(err.is_retryable());
        assert_eq!(err.retry_after(), Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_model_not_found_not_retryable() {
        let err = FetchError::ModelNotFound { repo: "test/model".into() };
        assert!(!err.is_retryable());
        assert!(err.retry_after().is_none());
    }

    #[test]
    fn test_pickle_is_security_risk() {
        let err = FetchError::PickleSecurityRisk;
        assert!(err.is_security_risk());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_corrupt_file_is_security_risk() {
        let err = FetchError::CorruptFile {
            path: PathBuf::from("/tmp/model.safetensors"),
            expected_hash: "abc123".into(),
            actual_hash: "def456".into(),
        };
        assert!(err.is_security_risk());
    }

    #[test]
    fn test_missing_token_display() {
        let err = FetchError::MissingToken;
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"));
    }

    #[test]
    fn test_invalid_repo_id_display() {
        let err = FetchError::InvalidRepoId { repo_id: "invalid".into() };
        let msg = err.to_string();
        assert!(msg.contains("org/name"));
    }

    #[test]
    fn test_all_error_variants_display() {
        // Ensure all variants have proper Display
        let errors: Vec<FetchError> = vec![
            FetchError::NetworkTimeout { repo: "r".into(), elapsed: Duration::from_secs(1) },
            FetchError::RateLimited { retry_after: Duration::from_secs(1) },
            FetchError::ModelNotFound { repo: "r".into() },
            FetchError::FileNotFound { repo: "r".into(), file: "f".into() },
            FetchError::CorruptFile {
                path: PathBuf::from("p"),
                expected_hash: "e".into(),
                actual_hash: "a".into(),
            },
            FetchError::InsufficientDisk { required: 100, available: 50 },
            FetchError::OutOfMemory { required: 100, available: 50 },
            FetchError::AuthenticationFailed { message: "m".into() },
            FetchError::MissingToken,
            FetchError::InvalidRepoId { repo_id: "r".into() },
            FetchError::UnsupportedFormat { format: "f".into() },
            FetchError::PickleSecurityRisk,
            FetchError::ConfigParseError { message: "m".into() },
            FetchError::ShapeMismatch {
                tensor: "t".into(),
                expected: vec![1, 2],
                actual: vec![3, 4],
            },
            FetchError::LeaderboardNotFound { kind: "OpenASR".into() },
            FetchError::LeaderboardParseError { message: "missing field".into() },
            FetchError::HttpError { message: "connection refused".into() },
            FetchError::GgufWriteError { message: "alignment error".into() },
        ];

        for err in errors {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "Error display should not be empty: {err:?}");
        }
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let fetch_err: FetchError = io_err.into();
        assert!(matches!(fetch_err, FetchError::Io(_)));
    }
}
