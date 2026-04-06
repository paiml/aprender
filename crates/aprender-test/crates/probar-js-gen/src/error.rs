//! Error types for `probar-js-gen`.
//!
//! # References
//! - DO-178C (2011) Section 6.3.4: Error handling requirements
//! - Leveson (2012) "Engineering a Safer World" - Fault tolerance patterns

use thiserror::Error;

/// Result type alias for js-gen operations.
pub type Result<T> = std::result::Result<T, JsGenError>;

/// Errors that can occur during JavaScript generation.
///
/// Each variant includes context for debugging while maintaining
/// type safety per DO-178C requirements.
#[derive(Debug, Error)]
pub enum JsGenError {
    /// Invalid identifier name (reserved word, invalid characters, etc.)
    #[error("Invalid identifier '{name}': {reason}")]
    InvalidIdentifier {
        /// The invalid identifier
        name: String,
        /// Why it's invalid
        reason: String,
    },

    /// Invalid string literal (unterminated, invalid escapes, etc.)
    #[error("Invalid string literal: {0}")]
    InvalidString(String),

    /// Code generation failed
    #[error("Code generation failed: {0}")]
    GenerationError(String),

    /// Validation failed
    #[error("Validation failed: {message}")]
    ValidationError {
        /// What validation failed
        message: String,
        /// Location in generated code (if available)
        location: Option<CodeLocation>,
    },

    /// Manifest verification failed (file was manually modified)
    #[error("Manifest verification failed for '{path}': {reason}")]
    ManifestError {
        /// Path to the file
        path: String,
        /// Why verification failed
        reason: String,
    },

    /// Hash mismatch (generated file was modified)
    #[error("Hash mismatch for '{path}': expected {expected}, got {actual}")]
    HashMismatch {
        /// Path to the file
        path: String,
        /// Expected hash
        expected: String,
        /// Actual hash
        actual: String,
    },

    /// IO error during file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Location in generated code for error reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeLocation {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
}

impl std::fmt::Display for CodeLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_invalid_identifier() {
        let err = JsGenError::InvalidIdentifier {
            name: "class".to_string(),
            reason: "reserved word".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid identifier 'class': reserved word");
    }

    #[test]
    fn error_display_hash_mismatch() {
        let err = JsGenError::HashMismatch {
            path: "worker.js".to_string(),
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert!(err.to_string().contains("Hash mismatch"));
        assert!(err.to_string().contains("worker.js"));
    }

    #[test]
    fn code_location_display() {
        let loc = CodeLocation {
            line: 42,
            column: 10,
        };
        assert_eq!(loc.to_string(), "line 42, column 10");
    }
}
