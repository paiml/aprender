//! Error types for Trueno-DB
//!
//! Toyota Way: Clear error messages with actionable guidance (Respect for People)

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// Trueno-DB error types
#[derive(Error, Debug)]
pub enum Error {
    /// GPU initialization failed
    #[error("GPU initialization failed: {0}\nFalling back to SIMD backend")]
    GpuInitFailed(String),

    /// VRAM exhaustion (Poka-Yoke: should not occur with morsel paging)
    #[error(
        "VRAM exhausted: {0}\nMorsel-based paging failed to prevent OOM. Please report this issue."
    )]
    VramExhausted(String),

    /// Backend equivalence test failed (critical bug)
    #[error("Backend equivalence failed: GPU result != SIMD result\nGPU: {gpu_result}\nSIMD: {simd_result}")]
    BackendMismatch {
        /// GPU computed result
        gpu_result: String,
        /// SIMD computed result
        simd_result: String,
    },

    /// Query parsing error
    #[error("SQL parse error: {0}")]
    ParseError(String),

    /// Storage error (Parquet/Arrow)
    #[error("Storage error: {0}")]
    StorageError(String),

    /// GPU transfer queue closed
    #[error("GPU transfer queue closed (receiver dropped)")]
    QueueClosed,

    /// Invalid input parameter
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Arrow/Parquet error
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<batuta_common::compression::CompressionError> for Error {
    fn from(e: batuta_common::compression::CompressionError) -> Self {
        Self::StorageError(e.to_string())
    }
}
