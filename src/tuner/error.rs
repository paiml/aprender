//! Tuner Error Types
//!
//! Error types for the ML-based tuner.

/// Tuner error type
#[derive(Debug, Clone)]
pub enum TunerError {
    /// Invalid feature value
    InvalidFeature(String),
    /// Insufficient training data
    InsufficientData(usize),
    /// Serialization error
    Serialization(String),
    /// Model not found
    ModelNotFound,
    /// Prediction failed
    PredictionFailed(String),
    /// Training failed (ml-tuner feature)
    TrainingFailed(String),
    /// I/O error (file operations)
    Io(String),
    /// Invalid format (APR magic/version mismatch)
    InvalidFormat(String),
}

impl std::fmt::Display for TunerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TunerError::InvalidFeature(msg) => write!(f, "Invalid feature: {}", msg),
            TunerError::InsufficientData(n) => {
                write!(f, "Insufficient training data: {} samples (need >= 10)", n)
            }
            TunerError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            TunerError::ModelNotFound => write!(f, "Tuner model not found"),
            TunerError::PredictionFailed(msg) => write!(f, "Prediction failed: {}", msg),
            TunerError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            TunerError::Io(msg) => write!(f, "I/O error: {}", msg),
            TunerError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for TunerError {}
