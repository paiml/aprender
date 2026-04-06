//! GGUF export error types.

/// Errors that can occur during GGUF export.
#[derive(Debug, thiserror::Error)]
pub enum GgufExportError {
    /// Invalid quantization configuration
    #[error("Invalid quantization: {0}")]
    InvalidQuantization(String),

    /// Model data validation failed
    #[error("Model validation failed: {0}")]
    ValidationFailed(String),

    /// I/O error during export
    #[error("Export I/O error: {0}")]
    IoError(String),

    /// Unsupported model architecture
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Metadata serialization error
    #[error("Metadata error: {0}")]
    MetadataError(String),
}
