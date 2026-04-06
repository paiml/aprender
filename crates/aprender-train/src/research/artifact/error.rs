//! Validation error types for research artifacts.

/// Validation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid ORCID format: {0}")]
    InvalidOrcid(String),
    #[error("Invalid ROR ID format: {0}")]
    InvalidRorId(String),
}
