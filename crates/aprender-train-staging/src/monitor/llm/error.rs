//! LLM evaluation error types.

use thiserror::Error;

/// LLM evaluation errors
#[derive(Debug, Error)]
pub enum LLMError {
    #[error("Run not found: {0}")]
    RunNotFound(String),

    #[error("Prompt not found: {0}")]
    PromptNotFound(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Invalid metric: {0}")]
    InvalidMetric(String),

    #[error("LLM error: {0}")]
    Internal(String),
}

/// Result type for LLM operations
pub type Result<T> = std::result::Result<T, LLMError>;
