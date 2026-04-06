//! Content addressing using BLAKE3 hashing.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Read;

/// Compression algorithm used for stored content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    /// No compression.
    #[default]
    None,
    /// Zstandard compression.
    #[cfg(feature = "compression")]
    Zstd,
}

impl fmt::Display for Compression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            #[cfg(feature = "compression")]
            Self::Zstd => write!(f, "zstd"),
        }
    }
}

/// Content-addressed identifier for stored artifacts.
///
/// Uses BLAKE3 hashing for:
/// - Deduplication across versions
/// - Tamper detection
/// - Efficient delta storage
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentAddress {
    /// BLAKE3 hash of content (32 bytes).
    hash: [u8; 32],
    /// Content size in bytes (uncompressed).
    size: u64,
    /// Compression algorithm used.
    compression: Compression,
}

impl ContentAddress {
    /// Create a new content address from raw components.
    #[must_use]
    pub fn new(hash: [u8; 32], size: u64, compression: Compression) -> Self {
        Self { hash, size, compression }
    }

    /// Compute content address from bytes.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self { hash: *hash.as_bytes(), size: data.len() as u64, compression: Compression::None }
    }

    /// Compute content address from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    pub fn from_reader<R: Read>(mut reader: R) -> std::io::Result<Self> {
        let mut hasher = blake3::Hasher::new();
        let mut size = 0u64;
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
            size += bytes_read as u64;
        }

        let hash = hasher.finalize();
        Ok(Self { hash: *hash.as_bytes(), size, compression: Compression::None })
    }

    /// Get the hash as bytes.
    #[must_use]
    pub fn hash_bytes(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Get the hash as a hex string.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        hex::encode(&self.hash)
    }

    /// Get the content size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the compression algorithm.
    #[must_use]
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Set the compression algorithm (returns new instance).
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Get the storage path prefix (first 2 hex chars for sharding).
    #[must_use]
    pub fn storage_prefix(&self) -> String {
        hex::encode(&self.hash[..1])
    }

    /// Get the full storage path (`prefix/full_hash`).
    #[must_use]
    pub fn storage_path(&self) -> String {
        format!("{}/{}", self.storage_prefix(), self.hash_hex())
    }

    /// Verify that data matches this content address.
    #[must_use]
    pub fn verify(&self, data: &[u8]) -> bool {
        let computed = Self::from_bytes(data);
        self.hash == computed.hash
    }
}

impl fmt::Display for ContentAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blake3:{}:{}:{}", self.hash_hex(), self.size, self.compression)
    }
}

// Need hex encoding
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub(super) fn encode(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            result.push(HEX_CHARS[(byte >> 4) as usize] as char);
            result.push(HEX_CHARS[(byte & 0x0f) as usize] as char);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_content_address_from_bytes() {
        let data = b"hello world";
        let addr = ContentAddress::from_bytes(data);

        assert_eq!(addr.size(), 11);
        assert_eq!(addr.compression(), Compression::None);
        // BLAKE3 hash of "hello world"
        assert_eq!(
            addr.hash_hex(),
            "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
        );
    }

    #[test]
    fn test_content_address_from_reader() {
        let data = b"hello world";
        let cursor = std::io::Cursor::new(data);
        let addr = ContentAddress::from_reader(cursor).unwrap();

        assert_eq!(addr.size(), 11);
        assert_eq!(
            addr.hash_hex(),
            "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
        );
    }

    #[test]
    fn test_content_address_verify() {
        let data = b"hello world";
        let addr = ContentAddress::from_bytes(data);

        assert!(addr.verify(data));
        assert!(!addr.verify(b"hello world!"));
        assert!(!addr.verify(b"Hello world"));
    }

    #[test]
    fn test_storage_path() {
        let data = b"hello world";
        let addr = ContentAddress::from_bytes(data);

        // First byte is 0xd7, so prefix is "d7"
        assert_eq!(addr.storage_prefix(), "d7");
        assert!(addr.storage_path().starts_with("d7/"));
        assert!(addr.storage_path().ends_with(&addr.hash_hex()));
    }

    #[test]
    fn test_display() {
        let data = b"test";
        let addr = ContentAddress::from_bytes(data);
        let display = addr.to_string();

        assert!(display.starts_with("blake3:"));
        assert!(display.contains(":4:none"));
    }

    #[test]
    fn test_with_compression() {
        let addr = ContentAddress::from_bytes(b"data");
        assert_eq!(addr.compression(), Compression::None);

        #[cfg(feature = "compression")]
        {
            let compressed = addr.with_compression(Compression::Zstd);
            assert_eq!(compressed.compression(), Compression::Zstd);
        }
    }

    #[test]
    fn test_serialization() {
        let addr = ContentAddress::from_bytes(b"test data");
        let json = serde_json::to_string(&addr).unwrap();
        let deserialized: ContentAddress = serde_json::from_str(&json).unwrap();

        assert_eq!(addr, deserialized);
    }

    // Property-based tests
    proptest! {
        #[test]
        fn prop_content_address_deterministic(data: Vec<u8>) {
            let addr1 = ContentAddress::from_bytes(&data);
            let addr2 = ContentAddress::from_bytes(&data);
            prop_assert_eq!(addr1, addr2);
        }

        #[test]
        fn prop_content_address_size_matches(data: Vec<u8>) {
            let addr = ContentAddress::from_bytes(&data);
            prop_assert_eq!(addr.size(), data.len() as u64);
        }

        #[test]
        fn prop_content_address_verify_self(data: Vec<u8>) {
            let addr = ContentAddress::from_bytes(&data);
            prop_assert!(addr.verify(&data));
        }

        #[test]
        fn prop_different_data_different_hash(data1: Vec<u8>, data2: Vec<u8>) {
            prop_assume!(data1 != data2);
            let addr1 = ContentAddress::from_bytes(&data1);
            let addr2 = ContentAddress::from_bytes(&data2);
            prop_assert_ne!(addr1.hash_bytes(), addr2.hash_bytes());
        }

        #[test]
        fn prop_hash_hex_length(data: Vec<u8>) {
            let addr = ContentAddress::from_bytes(&data);
            prop_assert_eq!(addr.hash_hex().len(), 64); // 32 bytes = 64 hex chars
        }

        #[test]
        fn prop_storage_prefix_length(data: Vec<u8>) {
            let addr = ContentAddress::from_bytes(&data);
            prop_assert_eq!(addr.storage_prefix().len(), 2); // 1 byte = 2 hex chars
        }
    }
}

// ─── Kani Formal Verification ────────────────────────────────────────────

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn verify_content_address_size_invariant() {
        let size: u64 = kani::any();
        let hash = [0u8; 32];
        let addr = ContentAddress::new(hash, size, Compression::None);
        assert!(addr.size() == size);
    }

    #[kani::proof]
    fn verify_hash_bytes_length() {
        let data: [u8; 8] = kani::any();
        let addr = ContentAddress::from_bytes(&data);
        assert!(addr.hash_bytes().len() == 32);
    }

    #[kani::proof]
    fn verify_hex_encode_length() {
        let bytes: [u8; 4] = kani::any();
        let encoded = super::hex::encode(&bytes);
        assert!(encoded.len() == 8); // 4 bytes = 8 hex chars
    }
}
