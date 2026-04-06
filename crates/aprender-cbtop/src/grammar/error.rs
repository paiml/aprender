//! Error types for grammar operations.

use std::time::Duration;

/// Result type for grammar operations
pub type GrammarResult<T> = Result<T, GrammarError>;

/// Error types for grammar operations
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarError {
    /// Missing required workload specification
    MissingWorkload,
    /// Invalid dimensions (zero or negative)
    InvalidDimensions(String),
    /// Invalid scale domain (min >= max)
    InvalidScaleDomain { min: f64, max: f64 },
    /// Device not found
    DeviceNotFound(u32),
    /// Execution timeout
    Timeout(Duration),
    /// Strategy not supported on current hardware
    UnsupportedStrategy(String),
    /// Validation error
    ValidationError(String),
    /// Execution error
    ExecutionError(String),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarError::MissingWorkload => write!(f, "Missing required workload specification"),
            GrammarError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            GrammarError::InvalidScaleDomain { min, max } => {
                write!(f, "Invalid scale domain: min {} >= max {}", min, max)
            }
            GrammarError::DeviceNotFound(id) => write!(f, "Device {} not found", id),
            GrammarError::Timeout(d) => write!(f, "Execution timeout after {:?}", d),
            GrammarError::UnsupportedStrategy(s) => write!(f, "Unsupported strategy: {}", s),
            GrammarError::ValidationError(s) => write!(f, "Validation error: {}", s),
            GrammarError::ExecutionError(s) => write!(f, "Execution error: {}", s),
        }
    }
}

impl std::error::Error for GrammarError {}
