//! Compressed KV Store wrapper (GH-5)
//!
//! Provides transparent LZ4/ZSTD compression for any `KvStore` backend.
//! Ideal for reducing memory footprint of LLM KV caches.
//!
//! Compression algorithm is shared via `batuta_common::compression`.

use crate::kv::KvStore;
use crate::Result;

pub use batuta_common::compression::Compression;

/// Compressed KV store wrapper
///
/// Wraps any `KvStore` implementation and transparently compresses/decompresses values.
///
/// # Example
///
/// ```rust,no_run
/// use trueno_db::kv::{CompressedKvStore, Compression, MemoryKvStore, KvStore};
///
/// # async fn example() -> trueno_db::Result<()> {
/// let inner = MemoryKvStore::new();
/// let store = CompressedKvStore::new(inner, Compression::Lz4);
///
/// // Values are transparently compressed
/// store.set("key", vec![0u8; 10000]).await?;
/// let value = store.get("key").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CompressedKvStore<S: KvStore> {
    inner: S,
    compression: Compression,
}

impl<S: KvStore> CompressedKvStore<S> {
    /// Create a new compressed KV store wrapping the given store
    #[must_use]
    pub const fn new(inner: S, compression: Compression) -> Self {
        Self { inner, compression }
    }

    /// Get reference to inner store (for inspection/testing)
    #[must_use]
    pub const fn inner(&self) -> &S {
        &self.inner
    }

    /// Get compression algorithm
    #[must_use]
    pub const fn compression(&self) -> Compression {
        self.compression
    }
}

impl<S: KvStore> KvStore for CompressedKvStore<S> {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self.inner.get(key).await? {
            Some(compressed) => {
                let decompressed = self.compression.decompress(&compressed)?;
                Ok(Some(decompressed))
            }
            None => Ok(None),
        }
    }

    async fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let compressed = self.compression.compress(&value)?;
        self.inner.set(key, compressed).await
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key).await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_as_str() {
        assert_eq!(Compression::Lz4.as_str(), "lz4");
        assert_eq!(Compression::Zstd.as_str(), "zstd");
    }

    #[test]
    fn test_compression_default() {
        assert_eq!(Compression::default(), Compression::Lz4);
    }

    #[test]
    fn test_lz4_compress_decompress() {
        let data = b"hello world hello world hello world".to_vec();
        let compressed = Compression::Lz4.compress(&data).unwrap();
        let decompressed = Compression::Lz4.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zstd_compress_decompress() {
        let data = b"hello world hello world hello world".to_vec();
        let compressed = Compression::Zstd.compress(&data).unwrap();
        let decompressed = Compression::Zstd.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_empty_data_compression() {
        let empty: Vec<u8> = vec![];

        let lz4_compressed = Compression::Lz4.compress(&empty).unwrap();
        assert!(lz4_compressed.is_empty());
        let lz4_decompressed = Compression::Lz4.decompress(&lz4_compressed).unwrap();
        assert!(lz4_decompressed.is_empty());

        let zstd_compressed = Compression::Zstd.compress(&empty).unwrap();
        assert!(zstd_compressed.is_empty());
        let zstd_decompressed = Compression::Zstd.decompress(&zstd_compressed).unwrap();
        assert!(zstd_decompressed.is_empty());
    }

    #[test]
    fn test_lz4_compresses_repeated_data() {
        let data = vec![0u8; 10000];
        let compressed = Compression::Lz4.compress(&data).unwrap();
        // LZ4 should achieve >10x compression on zeros
        assert!(compressed.len() < data.len() / 10);
    }

    #[test]
    fn test_zstd_compresses_repeated_data() {
        let data = vec![0u8; 10000];
        let compressed = Compression::Zstd.compress(&data).unwrap();
        // ZSTD should achieve >10x compression on zeros
        assert!(compressed.len() < data.len() / 10);
    }
}
