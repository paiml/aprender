//! Error types for counterfactual operations.

/// Error type for counterfactual operations
#[derive(Debug, Clone, PartialEq)]
pub enum CounterfactualError {
    /// Insufficient data
    InsufficientData { expected: usize, actual: usize },
    /// Version mismatch
    VersionMismatch { expected: u8, actual: u8 },
}

impl std::fmt::Display for CounterfactualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CounterfactualError::InsufficientData { expected, actual } => {
                write!(f, "Insufficient data: expected {expected}, got {actual}")
            }
            CounterfactualError::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for CounterfactualError {}
