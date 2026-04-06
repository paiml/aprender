//! Tokenizer error types.

use thiserror::Error;

/// Tokenizer errors
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Vocabulary not trained")]
    NotTrained,

    #[error("Unknown token: {0}")]
    UnknownToken(String),

    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),

    #[error("Training error: {0}")]
    Training(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for tokenizer operations
pub type Result<T> = std::result::Result<T, TokenizerError>;
