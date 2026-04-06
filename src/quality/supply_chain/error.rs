//! Error types for supply chain auditing.

use thiserror::Error;

/// Errors for supply chain auditing
#[derive(Debug, Error)]
pub enum SupplyChainError {
    #[error("Failed to parse cargo-deny output: {0}")]
    ParseError(String),

    #[error("Vulnerable dependency found: {0}")]
    VulnerabilityFound(String),
}

/// Result type for supply chain operations
pub type Result<T> = std::result::Result<T, SupplyChainError>;
