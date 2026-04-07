//! Shared compression utilities for the Batuta stack.
//!
//! Provides LZ4 (fast, real-time) and ZSTD (better ratio) compression
//! with a common enum interface. Used by trueno-db and trueno-rag.

/// Compression algorithm selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// LZ4 - Fast compression with prepended size, good for real-time
    #[default]
    Lz4,
    /// ZSTD - Better compression ratio, slower
    Zstd,
}

impl Compression {
    /// Get algorithm name as string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
        }
    }

    /// Compress data using this algorithm.
    ///
    /// Returns empty vec for empty input (short-circuit).
    ///
    /// # Errors
    /// Returns [`CompressionError`] if the compression algorithm fails.
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),
            Self::Zstd => zstd::encode_all(data, 3)
                .map_err(|e| CompressionError(format!("ZSTD compression failed: {e}"))),
        }
    }

    /// Decompress data using this algorithm.
    ///
    /// Returns empty vec for empty input (short-circuit).
    ///
    /// # Errors
    /// Returns [`CompressionError`] if the decompression algorithm fails.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Lz4 => lz4_flex::decompress_size_prepended(data)
                .map_err(|e| CompressionError(format!("LZ4 decompression failed: {e}"))),
            Self::Zstd => zstd::decode_all(data)
                .map_err(|e| CompressionError(format!("ZSTD decompression failed: {e}"))),
        }
    }
}

/// Error type for compression operations.
#[derive(Debug, Clone)]
pub struct CompressionError(pub String);

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CompressionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let data = b"hello world hello world hello world";
        let compressed = Compression::Lz4.compress(data).unwrap();
        let decompressed = Compression::Lz4.decompress(&compressed).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_zstd_roundtrip() {
        let data = b"hello world hello world hello world";
        let compressed = Compression::Zstd.compress(data).unwrap();
        let decompressed = Compression::Zstd.decompress(&compressed).unwrap();
        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_as_str() {
        assert_eq!(Compression::Lz4.as_str(), "lz4");
        assert_eq!(Compression::Zstd.as_str(), "zstd");
    }

    #[test]
    fn test_default_is_lz4() {
        assert_eq!(Compression::default(), Compression::Lz4);
    }

    #[test]
    fn test_empty_data_lz4() {
        let empty: &[u8] = &[];
        let compressed = Compression::Lz4.compress(empty).unwrap();
        assert!(compressed.is_empty());
        let decompressed = Compression::Lz4.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_empty_data_zstd() {
        let empty: &[u8] = &[];
        let compressed = Compression::Zstd.compress(empty).unwrap();
        assert!(compressed.is_empty());
        let decompressed = Compression::Zstd.decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_lz4_compresses_repeated_data() {
        let data = vec![0u8; 10000];
        let compressed = Compression::Lz4.compress(&data).unwrap();
        assert!(compressed.len() < data.len() / 10);
    }

    #[test]
    fn test_zstd_compresses_repeated_data() {
        let data = vec![0u8; 10000];
        let compressed = Compression::Zstd.compress(&data).unwrap();
        assert!(compressed.len() < data.len() / 10);
    }

    #[test]
    fn test_compression_error_display() {
        let err = CompressionError("test error".to_string());
        assert_eq!(err.to_string(), "test error");
    }
}
