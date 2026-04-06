//! Ruchy bridge error types.

/// Errors that can occur during session bridging.
#[derive(Debug, thiserror::Error)]
pub enum RuchyBridgeError {
    /// Session data is invalid or corrupted
    #[error("Invalid session data: {0}")]
    InvalidSession(String),

    /// Required session field is missing
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Session has no training history
    #[error("Session has no training history")]
    NoTrainingHistory,
}
