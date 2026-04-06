//! Error types for quantization operations (Jidoka halt conditions)

use std::fmt;

/// Error types for quantization operations (Jidoka halt conditions)
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationError {
    /// Embedding dimension mismatch
    DimensionMismatch { expected: usize, actual: usize },
    /// Non-finite value detected (NaN/Inf)
    NonFiniteValue { index: usize, value: f32 },
    /// Empty embedding
    EmptyEmbedding,
    /// Calibration not initialized
    CalibrationNotInitialized,
    /// Scale factor invalid (zero or negative)
    InvalidScale { scale: f32 },
    /// Overflow during computation
    ComputationOverflow,
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::NonFiniteValue { index, value } => {
                write!(f, "Non-finite value {} at index {}", value, index)
            }
            Self::EmptyEmbedding => write!(f, "Empty embedding"),
            Self::CalibrationNotInitialized => write!(f, "Calibration not initialized"),
            Self::InvalidScale { scale } => write!(f, "Invalid scale factor: {}", scale),
            Self::ComputationOverflow => write!(f, "Computation overflow"),
        }
    }
}

impl std::error::Error for QuantizationError {}

/// Validate an embedding for common error conditions (Poka-Yoke)
///
/// Checks:
/// 1. Non-empty
/// 2. Correct dimensionality
/// 3. All values finite (no NaN/Inf)
pub fn validate_embedding(
    embedding: &[f32],
    expected_dims: usize,
) -> Result<(), QuantizationError> {
    if embedding.len() != expected_dims {
        return Err(QuantizationError::DimensionMismatch {
            expected: expected_dims,
            actual: embedding.len(),
        });
    }
    if embedding.is_empty() {
        return Err(QuantizationError::EmptyEmbedding);
    }
    for (i, &v) in embedding.iter().enumerate() {
        if !v.is_finite() {
            return Err(QuantizationError::NonFiniteValue { index: i, value: v });
        }
    }
    Ok(())
}
