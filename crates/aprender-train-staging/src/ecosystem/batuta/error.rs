//! Error types for Batuta integration.

/// Errors that can occur when interacting with Batuta.
#[derive(Debug, thiserror::Error)]
pub enum BatutaError {
    /// Batuta service is unavailable
    #[error("Batuta service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Invalid GPU type requested
    #[error("Unknown GPU type: {0}")]
    UnknownGpuType(String),

    /// Network or connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Response parsing error
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}
