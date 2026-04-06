//! Storage error types

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Storage errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Storage not initialized")]
    NotInitialized,
}
