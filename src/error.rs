//! Error types for Trueno-RAG

use thiserror::Error;

/// Result type for Trueno-RAG operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for RAG pipeline operations
#[derive(Error, Debug)]
pub enum Error {
    /// Document is empty or invalid
    #[error("empty document: {0}")]
    EmptyDocument(String),

    /// Chunk too large for processing
    #[error("chunk exceeds maximum size: {size} > {max}")]
    ChunkTooLarge {
        /// Actual size
        size: usize,
        /// Maximum allowed size
        max: usize,
    },

    /// Embedding dimension mismatch
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Index not found
    #[error("index not found: {0}")]
    IndexNotFound(String),

    /// Vector store error
    #[error("vector store error: {0}")]
    VectorStore(String),

    /// Serialization error (serde_json)
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Serialization error (bincode/compression) - GH-2
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Query processing error
    #[error("query error: {0}")]
    Query(String),

    /// Embedding error (GH-1: production embeddings)
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Invalid input error (PMAT-003: multi-vector)
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl From<batuta_common::compression::CompressionError> for Error {
    fn from(e: batuta_common::compression::CompressionError) -> Self {
        Self::SerializationError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_empty_document() {
        let err = Error::EmptyDocument("test.txt".to_string());
        assert_eq!(err.to_string(), "empty document: test.txt");
    }

    #[test]
    fn test_error_display_chunk_too_large() {
        let err = Error::ChunkTooLarge { size: 1000, max: 512 };
        assert_eq!(err.to_string(), "chunk exceeds maximum size: 1000 > 512");
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = Error::DimensionMismatch { expected: 384, actual: 768 };
        assert_eq!(err.to_string(), "embedding dimension mismatch: expected 384, got 768");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = Error::from(io_err);
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_result_type() {
        fn may_fail(succeed: bool) -> Result<i32> {
            if succeed {
                Ok(42)
            } else {
                Err(Error::InvalidConfig("test".to_string()))
            }
        }

        assert_eq!(may_fail(true).unwrap(), 42);
        assert!(may_fail(false).is_err());
    }
}
