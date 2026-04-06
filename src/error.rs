//! Error types for alimentar.

use std::path::PathBuf;

/// Result type alias for alimentar operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in alimentar operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// I/O error during file operations.
    #[error("I/O error at {path:?}: {source}")]
    Io {
        /// The path where the error occurred, if known.
        path: Option<PathBuf>,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Arrow error during data processing.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// Parquet error during file operations.
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// Index out of bounds when accessing dataset.
    #[error("Index {index} out of bounds for dataset with {len} rows")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The actual length of the dataset.
        len: usize,
    },

    /// Column not found in schema.
    #[error("Column '{name}' not found in schema")]
    ColumnNotFound {
        /// The name of the missing column.
        name: String,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {message}")]
    InvalidConfig {
        /// Description of the configuration error.
        message: String,
    },

    /// Unsupported file format.
    #[error("Unsupported format: {format}")]
    UnsupportedFormat {
        /// The unsupported format name or extension.
        format: String,
    },

    /// Empty dataset error.
    #[error("Dataset is empty")]
    EmptyDataset,

    /// Schema mismatch between datasets or batches.
    #[error("Schema mismatch: {message}")]
    SchemaMismatch {
        /// Description of the schema mismatch.
        message: String,
    },

    /// Backend storage error.
    #[error("Storage backend error: {message}")]
    Storage {
        /// Description of the storage error.
        message: String,
    },

    /// Transform error.
    #[error("Transform error: {message}")]
    Transform {
        /// Description of the transform error.
        message: String,
    },

    /// Parse error.
    #[error("Parse error: {message}")]
    Parse {
        /// Description of the parse error.
        message: String,
    },

    /// Data error.
    #[error("Data error: {message}")]
    Data {
        /// Description of the data error.
        message: String,
    },

    /// Format error (header, checksum, etc.).
    #[error("Format error: {0}")]
    Format(String),

    /// Checksum mismatch.
    #[error("Checksum mismatch: expected {expected:08X}, got {actual:08X}")]
    ChecksumMismatch {
        /// Expected checksum.
        expected: u32,
        /// Actual checksum.
        actual: u32,
    },

    /// License has expired.
    #[error("License expired at {expired_at} (current time: {current_time})")]
    LicenseExpired {
        /// Expiration timestamp (Unix epoch seconds).
        expired_at: u64,
        /// Current timestamp (Unix epoch seconds).
        current_time: u64,
    },

    /// Signature verification failed.
    #[error("Signature verification failed")]
    SignatureInvalid,

    /// Decryption failed.
    #[error("Decryption failed: wrong password or corrupted data")]
    DecryptionFailed,

    /// Resource not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Invalid format for output/conversion.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

impl Error {
    /// Create an I/O error with a path context.
    pub fn io(source: std::io::Error, path: impl Into<PathBuf>) -> Self {
        Self::Io {
            path: Some(path.into()),
            source,
        }
    }

    /// Create an I/O error without path context.
    pub fn io_no_path(source: std::io::Error) -> Self {
        Self::Io { path: None, source }
    }

    /// Create a column not found error.
    pub fn column_not_found(name: impl Into<String>) -> Self {
        Self::ColumnNotFound { name: name.into() }
    }

    /// Create an invalid configuration error.
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    /// Create an unsupported format error.
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
        }
    }

    /// Create a schema mismatch error.
    pub fn schema_mismatch(message: impl Into<String>) -> Self {
        Self::SchemaMismatch {
            message: message.into(),
        }
    }

    /// Create a storage error.
    pub fn storage(message: impl Into<String>) -> Self {
        Self::Storage {
            message: message.into(),
        }
    }

    /// Create a transform error.
    pub fn transform(message: impl Into<String>) -> Self {
        Self::Transform {
            message: message.into(),
        }
    }

    /// Create a parse error.
    pub fn parse(message: impl Into<String>) -> Self {
        Self::Parse {
            message: message.into(),
        }
    }

    /// Create a data error.
    pub fn data(message: impl Into<String>) -> Self {
        Self::Data {
            message: message.into(),
        }
    }

    /// Create an empty dataset error.
    #[must_use]
    pub fn empty_dataset(_name: &str) -> Self {
        Self::EmptyDataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_error_with_path() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = Error::io(io_err, "/path/to/file");
        assert!(err.to_string().contains("/path/to/file"));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_io_error_without_path() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = Error::io_no_path(io_err);
        assert!(err.to_string().contains("None"));
    }

    #[test]
    fn test_index_out_of_bounds() {
        let err = Error::IndexOutOfBounds { index: 10, len: 5 };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains('5'));
    }

    #[test]
    fn test_column_not_found() {
        let err = Error::column_not_found("my_column");
        assert!(err.to_string().contains("my_column"));
    }

    #[test]
    fn test_invalid_config() {
        let err = Error::invalid_config("batch_size must be positive");
        assert!(err.to_string().contains("batch_size must be positive"));
    }

    #[test]
    fn test_unsupported_format() {
        let err = Error::unsupported_format("xlsx");
        assert!(err.to_string().contains("xlsx"));
    }

    #[test]
    fn test_schema_mismatch() {
        let err = Error::schema_mismatch("expected Int64, got Utf8");
        assert!(err.to_string().contains("expected Int64, got Utf8"));
    }

    #[test]
    fn test_storage_error() {
        let err = Error::storage("connection refused");
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn test_transform_error() {
        let err = Error::transform("filter predicate failed");
        assert!(err.to_string().contains("filter predicate failed"));
    }

    #[test]
    fn test_empty_dataset() {
        let err = Error::EmptyDataset;
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_empty_dataset_constructor() {
        let err = Error::empty_dataset("test");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_parse_error() {
        let err = Error::parse("invalid JSON syntax");
        assert!(err.to_string().contains("invalid JSON syntax"));
    }

    #[test]
    fn test_data_error() {
        let err = Error::data("corrupted record");
        assert!(err.to_string().contains("corrupted record"));
    }

    #[test]
    fn test_format_error() {
        let err = Error::Format("invalid magic bytes".to_string());
        assert!(err.to_string().contains("invalid magic bytes"));
    }

    #[test]
    fn test_checksum_mismatch() {
        let err = Error::ChecksumMismatch {
            expected: 0xDEADBEEF,
            actual: 0xCAFEBABE,
        };
        let msg = err.to_string();
        assert!(msg.contains("DEADBEEF"));
        assert!(msg.contains("CAFEBABE"));
    }

    #[test]
    fn test_license_expired() {
        let err = Error::LicenseExpired {
            expired_at: 1700000000,
            current_time: 1700100000,
        };
        let msg = err.to_string();
        assert!(msg.contains("expired"));
        assert!(msg.contains("1700000000"));
        assert!(msg.contains("1700100000"));
    }

    #[test]
    fn test_signature_invalid() {
        let err = Error::SignatureInvalid;
        assert!(err.to_string().contains("Signature verification failed"));
    }

    #[test]
    fn test_decryption_failed() {
        let err = Error::DecryptionFailed;
        assert!(err.to_string().contains("Decryption failed"));
    }
}
