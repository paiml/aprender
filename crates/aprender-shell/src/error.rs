//! Error types for aprender-shell
//!
//! Follows Toyota Way principle *Jidoka* (Build quality in):
//! Stop and fix problems immediately; never pass defects downstream.

use std::fmt;
use std::path::PathBuf;

/// Errors that can occur in aprender-shell operations.
///
/// Each variant provides specific context and actionable hints for resolution.
/// This eliminates the Cloudflare-class defects caused by unwrap()/expect().
#[derive(Debug)]
pub enum ShellError {
    /// Model file was not found at the expected path
    ModelNotFound { path: PathBuf, hint: String },

    /// Model file exists but is corrupted (checksum mismatch, invalid format)
    ModelCorrupted { path: PathBuf, hint: String },

    /// Generic model loading failure
    ModelLoadFailed { path: PathBuf, cause: String },

    /// Invalid input provided to a command
    InvalidInput { message: String },

    /// Error parsing history file
    HistoryParseError {
        path: PathBuf,
        line: usize,
        cause: String,
    },

    /// Security violation detected (sensitive command blocked)
    SecurityViolation { command: String, reason: String },
}

impl fmt::Display for ShellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound { path, hint } => {
                write!(
                    f,
                    "Error: Model not found at '{}'\nHint: {}",
                    path.display(),
                    hint
                )
            }
            Self::ModelCorrupted { path, hint } => {
                write!(
                    f,
                    "Error: Model corrupted at '{}'\nHint: {}",
                    path.display(),
                    hint
                )
            }
            Self::ModelLoadFailed { path, cause } => {
                write!(
                    f,
                    "Error: Failed to load model '{}': {}",
                    path.display(),
                    cause
                )
            }
            Self::InvalidInput { message } => {
                write!(f, "Error: {message}")
            }
            Self::HistoryParseError { path, line, cause } => {
                write!(
                    f,
                    "Error: Failed to parse {} at line {}: {}",
                    path.display(),
                    line,
                    cause
                )
            }
            Self::SecurityViolation { command, reason } => {
                write!(f, "Security: Blocked '{}' - {}", command, reason)
            }
        }
    }
}

impl std::error::Error for ShellError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_found_display() {
        let err = ShellError::ModelNotFound {
            path: PathBuf::from("/path/to/model.bin"),
            hint: "Run 'aprender-shell train' to create a model".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("not found"));
        assert!(msg.contains("/path/to/model.bin"));
        assert!(msg.contains("Hint:"));
    }

    #[test]
    fn test_model_corrupted_display() {
        let err = ShellError::ModelCorrupted {
            path: PathBuf::from("/path/to/model.bin"),
            hint: "Run 'aprender-shell train' to rebuild".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("corrupted"));
        assert!(msg.contains("rebuild"));
    }

    #[test]
    fn test_invalid_input_display() {
        let err = ShellError::InvalidInput {
            message: "Empty prefix".into(),
        };
        assert_eq!(err.to_string(), "Error: Empty prefix");
    }

    #[test]
    fn test_security_violation_display() {
        let err = ShellError::SecurityViolation {
            command: "export SECRET=abc".into(),
            reason: "Contains sensitive data".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Security:"));
        assert!(msg.contains("Blocked"));
    }
}
