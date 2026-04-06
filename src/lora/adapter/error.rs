//! LoRA adapter errors

use thiserror::Error;

/// LoRA adapter save/load errors
#[derive(Error, Debug)]
pub enum AdapterError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Adapter validation error: {0}")]
    Validation(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("PEFT format error: {0}")]
    PeftFormatError(String),
}
