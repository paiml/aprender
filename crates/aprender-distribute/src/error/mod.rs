//! Error types for repartir.
//!
//! Per Iron Lotus Framework: All errors are explicit, no panics allowed.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for repartir operations.
pub type Result<T> = std::result::Result<T, RepartirError>;

/// Comprehensive error type for all repartir operations.
///
/// Following Iron Lotus principle of explicit error handling,
/// this enum covers all failure modes without panics.
#[derive(Error, Debug)]
pub enum RepartirError {
    /// Task execution failed.
    #[error("Task execution failed: {message}")]
    ExecutionFailed {
        /// Human-readable error message.
        message: String,
        /// Exit code from the binary (if applicable).
        exit_code: Option<i32>,
    },

    /// Binary not found at specified path.
    #[error("Binary not found: {path}")]
    BinaryNotFound {
        /// Path where binary was expected.
        path: PathBuf,
    },

    /// I/O error during task execution.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Task timeout exceeded.
    #[error("Task timeout exceeded: {seconds}s")]
    Timeout {
        /// Timeout duration in seconds.
        seconds: u64,
    },

    /// Worker not found (for distributed execution).
    #[error("Worker not found: {worker_id}")]
    WorkerNotFound {
        /// UUID of the missing worker.
        worker_id: uuid::Uuid,
    },

    /// Scheduler is full and cannot accept more tasks.
    #[error("Scheduler queue full: {capacity} tasks")]
    QueueFull {
        /// Maximum queue capacity.
        capacity: usize,
    },

    /// Invalid task configuration.
    #[error("Invalid task configuration: {reason}")]
    InvalidTask {
        /// Reason why task is invalid.
        reason: String,
    },

    /// Serialization error (for remote execution).
    #[cfg(feature = "remote")]
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Tensor operation error (v2.0).
    #[cfg(feature = "tensor")]
    #[error("Tensor operation failed: {0}")]
    Tensor(String),
    // NOTE: GPU backend errors will be added in v1.1+
    // #[cfg(feature = "gpu")]
    // #[error("GPU error: {0}")]
    // Gpu(String),
}

#[cfg(feature = "tensor")]
impl From<trueno::TruenoError> for RepartirError {
    fn from(err: trueno::TruenoError) -> Self {
        Self::Tensor(err.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RepartirError::ExecutionFailed {
            message: "command failed".to_string(),
            exit_code: Some(1),
        };
        assert!(format!("{err}").contains("Task execution failed"));
    }

    #[test]
    fn test_binary_not_found() {
        let err = RepartirError::BinaryNotFound { path: PathBuf::from("/nonexistent") };
        assert!(format!("{err}").contains("Binary not found"));
    }

    #[test]
    fn test_timeout_error() {
        let err = RepartirError::Timeout { seconds: 30 };
        assert!(format!("{err}").contains("30"));
    }
}
