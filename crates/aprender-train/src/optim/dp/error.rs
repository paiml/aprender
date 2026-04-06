//! Error types for the Differential Privacy module.

use thiserror::Error;

/// DP errors
#[derive(Debug, Error)]
pub enum DpError {
    #[error("Privacy budget exhausted: spent {spent:.4} > allowed {budget:.4}")]
    BudgetExhausted { spent: f64, budget: f64 },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Gradient computation failed: {0}")]
    GradientError(String),

    #[error("DP error: {0}")]
    Internal(String),
}

/// Result type for DP operations
pub type Result<T> = std::result::Result<T, DpError>;
