//! Content creation errors
//!
//! Error types for content operations.

use thiserror::Error;

/// Errors that can occur during content operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ContentError {
    #[error("Invalid content type: {0}")]
    InvalidContentType(String),

    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    #[error("Template parse error: {0}")]
    TemplateParseFailed(String),

    #[error("Token budget exceeded: {used} tokens exceeds {limit} limit")]
    TokenBudgetExceeded { used: usize, limit: usize },

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Source context error: {0}")]
    SourceContextError(String),

    #[error("RAG context error: {0}")]
    RagContextError(String),
}
