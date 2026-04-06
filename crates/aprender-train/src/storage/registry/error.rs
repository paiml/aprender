//! Registry error types

use thiserror::Error;

use super::stage::ModelStage;

/// Registry errors
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Version not found: {0} v{1}")]
    VersionNotFound(String, u32),

    #[error("Invalid stage transition from {0} to {1}")]
    InvalidTransition(ModelStage, ModelStage),

    #[error("Policy check failed: {0}")]
    PolicyFailed(String),

    #[error("Registry error: {0}")]
    Internal(String),
}

/// Result type for registry operations
pub type Result<T> = std::result::Result<T, RegistryError>;
