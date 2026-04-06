//! Error types for Verificar
//!
//! This module defines the error types used throughout the library.

use thiserror::Error;

/// Result type alias for Verificar operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during Verificar operations
#[derive(Error, Debug)]
pub enum Error {
    /// Grammar parsing error
    #[error("grammar error: {0}")]
    Grammar(String),

    /// Code generation error
    #[error("generation error: {0}")]
    Generation(String),

    /// AST mutation error
    #[error("mutation error: {0}")]
    Mutation(String),

    /// Transpilation error
    #[error("transpile error: {0}")]
    Transpile(String),

    /// Verification oracle error
    #[error("verification error: {0}")]
    Verification(String),

    /// Execution timeout
    #[error("execution timeout after {0}ms")]
    Timeout(u64),

    /// Runtime error during execution
    #[error("runtime error in {phase}: {message}")]
    Runtime {
        /// Phase where error occurred (source or target)
        phase: String,
        /// Error message
        message: String,
    },

    /// I/O error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Configuration error
    #[error("configuration error: {0}")]
    Configuration(String),

    /// Data pipeline error
    #[error("data error: {0}")]
    Data(String),
}
