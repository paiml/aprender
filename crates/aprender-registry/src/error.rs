//! Error types for Pacha registry operations.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for Pacha operations.
pub type Result<T> = std::result::Result<T, PachaError>;

/// Errors that can occur during Pacha registry operations.
#[derive(Error, Debug)]
pub enum PachaError {
    /// Database operation failed.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// IO operation failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML serialization/deserialization failed.
    #[error("TOML error: {0}")]
    TomlDeserialize(#[from] toml::de::Error),

    /// TOML serialization failed.
    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    /// `MessagePack` serialization/deserialization failed.
    #[error("MessagePack error: {0}")]
    MessagePack(String),

    /// Artifact not found.
    #[error("artifact not found: {kind} '{name}' version {version}")]
    NotFound {
        /// Kind of artifact (model, dataset, recipe).
        kind: String,
        /// Name of the artifact.
        name: String,
        /// Version requested.
        version: String,
    },

    /// Artifact already exists.
    #[error("artifact already exists: {kind} '{name}' version {version}")]
    AlreadyExists {
        /// Kind of artifact.
        kind: String,
        /// Name of the artifact.
        name: String,
        /// Version that exists.
        version: String,
    },

    /// Invalid version string.
    #[error("invalid version string: {0}")]
    InvalidVersion(String),

    /// Content hash mismatch.
    #[error("content hash mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        /// Expected hash.
        expected: String,
        /// Actual hash computed.
        actual: String,
    },

    /// Storage path error.
    #[error("storage path error: {0}")]
    StoragePath(PathBuf),

    /// Compression error.
    #[error("compression error: {0}")]
    Compression(String),

    /// Invalid artifact state transition.
    #[error("invalid stage transition from {from} to {to}")]
    InvalidStageTransition {
        /// Current stage.
        from: String,
        /// Attempted target stage.
        to: String,
    },

    /// Validation error.
    #[error("validation error: {0}")]
    Validation(String),

    /// Registry not initialized.
    #[error("registry not initialized at {0}")]
    NotInitialized(PathBuf),

    /// Invalid URI format.
    #[error("invalid URI: {0}")]
    InvalidUri(String),

    /// Operation not supported.
    #[error("unsupported operation '{operation}': {reason}")]
    UnsupportedOperation {
        /// Operation that was attempted.
        operation: String,
        /// Reason it's not supported.
        reason: String,
    },

    /// Signature verification failed.
    #[error("signature verification failed")]
    SignatureInvalid,

    /// Remote registry error.
    #[error("remote registry error: {0}")]
    RemoteRegistry(String),

    /// Invalid format (encryption, file structure).
    #[error("invalid format: {0}")]
    InvalidFormat(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_not_found() {
        let err = PachaError::NotFound {
            kind: "model".to_string(),
            name: "fraud-detector".to_string(),
            version: "1.0.0".to_string(),
        };
        assert_eq!(err.to_string(), "artifact not found: model 'fraud-detector' version 1.0.0");
    }

    #[test]
    fn test_error_display_hash_mismatch() {
        let err = PachaError::HashMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert_eq!(err.to_string(), "content hash mismatch: expected abc123, got def456");
    }

    #[test]
    fn test_error_display_invalid_stage() {
        let err = PachaError::InvalidStageTransition {
            from: "development".to_string(),
            to: "archived".to_string(),
        };
        assert_eq!(err.to_string(), "invalid stage transition from development to archived");
    }
}
