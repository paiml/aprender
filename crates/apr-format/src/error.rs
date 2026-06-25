//! Error type for the sovereign `apr-format` leaf crate.
//!
//! # Wrapper error seam (issue #2231, locked decision 3)
//!
//! `apr-format` owns its own `AprFormatError`. It deliberately does **not**
//! share an error crate with `aprender-core`. Instead, `aprender-core` provides
//! `impl From<AprFormatError> for AprenderError` so the leaf's errors wrap
//! transparently into the framework's error type at the crate boundary. This
//! keeps the leaf sovereign (no dependency back on the framework's error type)
//! while preserving `?`-ergonomics for core consumers.

/// Errors produced while reading or writing the `.apr` container format.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AprFormatError {
    /// CRC32 integrity check failed: the stored trailer does not match the
    /// recomputed checksum of the file body.
    #[error("Checksum mismatch: expected 0x{expected:08X}, got 0x{actual:08X}")]
    ChecksumMismatch {
        /// Checksum stored in the file trailer.
        expected: u32,
        /// Checksum recomputed over the file body.
        actual: u32,
    },

    /// Invalid or corrupt container structure (bad magic, truncated header,
    /// unknown model type, out-of-range sizes, …).
    #[error("Invalid model format: {message}")]
    FormatError {
        /// Human-readable description of the format violation.
        message: String,
    },

    /// Serialization / deserialization failure (bincode, msgpack, JSON).
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Poka-yoke / Jidoka validation failure (e.g. refusing to save a model
    /// whose quality gate scored zero).
    #[error("Validation failed: {message}")]
    ValidationError {
        /// Description of the validation failure.
        message: String,
    },

    /// Underlying I/O failure (file open / read / write).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A computed offset fell outside the file/data boundary.
    #[error("Invalid offset: out of bounds for the supplied buffer")]
    InvalidOffset,

    /// The declared header / metadata section exceeds the permitted maximum
    /// (compression-bomb / overflow protection).
    #[error("Header or metadata section too large")]
    HeaderTooLarge,

    /// The on-disk format version is newer than this reader supports.
    #[error("Unsupported format version: found {}.{}, max supported {}.{}", found.0, found.1, supported.0, supported.1)]
    UnsupportedVersion {
        /// Version found in the file header.
        found: (u8, u8),
        /// Maximum version this reader supports.
        supported: (u8, u8),
    },
}

/// Convenience alias for results in this crate.
pub type Result<T> = std::result::Result<T, AprFormatError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_mismatch_display() {
        let e = AprFormatError::ChecksumMismatch {
            expected: 0xDEAD_BEEF,
            actual: 0xCAFE_BABE,
        };
        let s = e.to_string();
        assert!(s.contains("Checksum mismatch"));
        assert!(s.contains("DEADBEEF"));
    }

    #[test]
    fn test_format_error_display() {
        let e = AprFormatError::FormatError {
            message: "corrupt header".to_string(),
        };
        assert!(e.to_string().contains("corrupt header"));
    }

    #[test]
    fn test_io_from() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let e: AprFormatError = io.into();
        assert!(matches!(e, AprFormatError::Io(_)));
    }

    #[test]
    fn test_unsupported_version_display() {
        let e = AprFormatError::UnsupportedVersion {
            found: (3, 0),
            supported: (1, 0),
        };
        assert!(e.to_string().contains("3.0"));
    }
}
