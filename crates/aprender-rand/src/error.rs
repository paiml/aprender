//! RNG error types.

/// Errors from RNG operations.
#[derive(Debug, thiserror::Error)]
pub enum RngError {
    /// Output buffer is empty.
    #[error("output buffer must not be empty")]
    EmptyBuffer,

    /// Invalid standard deviation for normal distribution.
    #[error("standard deviation must be positive, got {0}")]
    InvalidStdDev(f32),
}
