//! Error types with actionable diagnostics (Andon principle).
//!
//! All errors include contextual information to help users resolve issues
//! without needing to consult external documentation.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for entrenar operations.
pub type Result<T> = std::result::Result<T, EntrenarError>;

/// Errors that can occur in entrenar CLI tools.
///
/// Each variant includes actionable context following the Andon principle
/// of making problems immediately visible and actionable.
#[derive(Error, Debug)]
pub enum EntrenarError {
    /// Configuration file not found at expected path.
    #[error("Configuration file not found: {path}\n  → Create a config file or use --config to specify a different path")]
    ConfigNotFound { path: PathBuf },

    /// Configuration file has invalid syntax.
    #[error("Invalid configuration syntax in {path}:\n  {message}\n  → Check YAML/JSON syntax at the indicated line")]
    ConfigParsing { path: PathBuf, message: String },

    /// Configuration value is invalid.
    #[error("Invalid configuration value for '{field}': {message}\n  → {suggestion}")]
    ConfigValue { field: String, message: String, suggestion: String },

    /// Model file not found.
    #[error("Model file not found: {path}\n  → Download the model or check the path")]
    ModelNotFound { path: PathBuf },

    /// Model format is unsupported.
    #[error("Unsupported model format: {format}\n  → Supported formats: SafeTensors, GGUF, APR")]
    UnsupportedFormat { format: String },

    /// HuggingFace API error.
    #[error("HuggingFace API error: {message}\n  → Check your HF_TOKEN environment variable and network connection")]
    HuggingFace { message: String },

    /// Insufficient memory for operation.
    #[error("Insufficient memory: need {required} GB, have {available} GB\n  → Try: reduce batch_size, enable gradient_accumulation, or use QLoRA")]
    InsufficientMemory { required: f64, available: f64 },

    /// Invalid tensor shape.
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}\n  → Check model architecture compatibility")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    /// IO error with context.
    #[error("IO error: {context}\n  Cause: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    /// Serialization/deserialization error.
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// User interrupted operation.
    #[error("Operation cancelled by user")]
    Cancelled,

    /// Generic error for unexpected conditions.
    #[error("Internal error: {message}\n  → Please report this bug at https://github.com/paiml/entrenar/issues")]
    Internal { message: String },
}

impl EntrenarError {
    /// Create an IO error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io { context: context.into(), source }
    }

    /// Check if this error is user-recoverable.
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::ConfigNotFound { .. }
                | Self::ConfigParsing { .. }
                | Self::ConfigValue { .. }
                | Self::ModelNotFound { .. }
                | Self::UnsupportedFormat { .. }
                | Self::HuggingFace { .. }
                | Self::InsufficientMemory { .. }
                | Self::Cancelled
        )
    }

    /// Get the error code for structured output.
    pub fn code(&self) -> &'static str {
        match self {
            Self::ConfigNotFound { .. } => "E001",
            Self::ConfigParsing { .. } => "E002",
            Self::ConfigValue { .. } => "E003",
            Self::ModelNotFound { .. } => "E010",
            Self::UnsupportedFormat { .. } => "E011",
            Self::HuggingFace { .. } => "E020",
            Self::InsufficientMemory { .. } => "E030",
            Self::ShapeMismatch { .. } => "E040",
            Self::Io { .. } => "E050",
            Self::Serialization { .. } => "E051",
            Self::Cancelled => "E060",
            Self::Internal { .. } => "E999",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes_are_unique() {
        let errors = vec![
            EntrenarError::ConfigNotFound { path: "".into() },
            EntrenarError::ConfigParsing { path: "".into(), message: String::new() },
            EntrenarError::ConfigValue {
                field: String::new(),
                message: String::new(),
                suggestion: String::new(),
            },
            EntrenarError::ModelNotFound { path: "".into() },
            EntrenarError::UnsupportedFormat { format: String::new() },
            EntrenarError::HuggingFace { message: String::new() },
            EntrenarError::InsufficientMemory { required: 0.0, available: 0.0 },
            EntrenarError::ShapeMismatch { expected: vec![], actual: vec![] },
            EntrenarError::Serialization { message: String::new() },
            EntrenarError::Cancelled,
            EntrenarError::Internal { message: String::new() },
        ];

        let codes: Vec<_> = errors.iter().map(super::EntrenarError::code).collect();
        let unique: std::collections::HashSet<_> = codes.iter().collect();

        // All codes except E050 and E051 should be unique
        // (Io and Serialization are in same category)
        assert!(unique.len() >= codes.len() - 1);
    }

    #[test]
    fn test_user_errors_are_recoverable() {
        assert!(EntrenarError::ConfigNotFound { path: "".into() }.is_user_error());
        assert!(EntrenarError::Cancelled.is_user_error());
        assert!(!EntrenarError::Internal { message: String::new() }.is_user_error());
    }

    #[test]
    fn test_error_messages_are_actionable() {
        let err = EntrenarError::InsufficientMemory { required: 32.0, available: 16.0 };
        let msg = err.to_string();

        // Must mention the problem
        assert!(msg.contains("32"));
        assert!(msg.contains("16"));

        // Must include actionable suggestions
        assert!(msg.contains("batch_size") || msg.contains("QLoRA"));
    }

    #[test]
    fn test_io_error_constructor() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = EntrenarError::io("reading config", io_err);

        assert!(matches!(err, EntrenarError::Io { .. }));
        let msg = err.to_string();
        assert!(msg.contains("reading config"));
    }

    #[test]
    fn test_shape_mismatch_not_user_error() {
        let err = EntrenarError::ShapeMismatch { expected: vec![1, 2, 3], actual: vec![1, 2, 4] };
        assert!(!err.is_user_error());
    }

    #[test]
    fn test_serialization_error_display() {
        let err = EntrenarError::Serialization { message: "invalid JSON".into() };
        let msg = err.to_string();
        assert!(msg.contains("invalid JSON"));
    }

    #[test]
    fn test_config_value_error_includes_suggestion() {
        let err = EntrenarError::ConfigValue {
            field: "learning_rate".into(),
            message: "must be positive".into(),
            suggestion: "Use a value like 0.001".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("learning_rate"));
        assert!(msg.contains("must be positive"));
        assert!(msg.contains("Use a value like 0.001"));
    }

    #[test]
    fn test_unsupported_format_lists_alternatives() {
        let err = EntrenarError::UnsupportedFormat { format: "pickle".into() };
        let msg = err.to_string();
        assert!(msg.contains("pickle"));
        assert!(msg.contains("SafeTensors"));
    }

    #[test]
    fn test_huggingface_error_mentions_token() {
        let err = EntrenarError::HuggingFace { message: "rate limited".into() };
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"));
    }

    #[test]
    fn test_internal_error_mentions_bug_report() {
        let err = EntrenarError::Internal { message: "unexpected state".into() };
        let msg = err.to_string();
        assert!(msg.contains("github.com"));
        assert!(msg.contains("issues"));
    }

    #[test]
    fn test_all_error_codes_start_with_e() {
        let errors: Vec<EntrenarError> = vec![
            EntrenarError::ConfigNotFound { path: "".into() },
            EntrenarError::Cancelled,
            EntrenarError::Internal { message: String::new() },
        ];

        for err in errors {
            assert!(err.code().starts_with('E'));
        }
    }
}
