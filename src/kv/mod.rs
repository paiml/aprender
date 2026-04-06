//! Key-Value Store Module for PAIML Stack Integration (Phase 6)
//!
//! Provides a simple, high-performance key-value store with:
//! - SIMD-optimized key hashing via `trueno::hash`
//! - In-memory and persistent backends
//! - Async-first API compatible with pforge `StateManager`
//!
//! # Example
//!
//! ```rust,no_run
//! use trueno_db::kv::{KvStore, MemoryKvStore};
//!
//! # async fn example() -> trueno_db::Result<()> {
//! let store = MemoryKvStore::new();
//!
//! // Basic operations
//! store.set("key", b"value".to_vec()).await?;
//! let value = store.get("key").await?;
//! assert_eq!(value, Some(b"value".to_vec()));
//!
//! store.delete("key").await?;
//! assert!(!store.exists("key").await?);
//! # Ok(())
//! # }
//! ```

mod memory;

#[cfg(feature = "compression")]
mod compressed;

pub use memory::MemoryKvStore;

#[cfg(feature = "compression")]
pub use compressed::{CompressedKvStore, Compression};

// Re-export trueno hash functions for KV consumers
pub use trueno::{hash_bytes, hash_key, hash_keys_batch};

use crate::Result;
use std::future::Future;

/// Key-value store trait for pforge state management integration.
///
/// This trait is designed to match pforge's `StateManager` interface
/// for seamless integration while leveraging trueno's SIMD capabilities.
pub trait KvStore: Send + Sync {
    /// Get a value by key.
    ///
    /// Returns `None` if the key doesn't exist.
    fn get(&self, key: &str) -> impl Future<Output = Result<Option<Vec<u8>>>> + Send;

    /// Set a value for a key.
    ///
    /// Overwrites any existing value.
    fn set(&self, key: &str, value: Vec<u8>) -> impl Future<Output = Result<()>> + Send;

    /// Delete a key.
    ///
    /// No-op if the key doesn't exist.
    fn delete(&self, key: &str) -> impl Future<Output = Result<()>> + Send;

    /// Check if a key exists.
    fn exists(&self, key: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Get multiple keys in a batch (SIMD-optimized).
    ///
    /// Returns values in the same order as keys. Missing keys return `None`.
    fn batch_get(
        &self,
        keys: &[&str],
    ) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>>> + Send {
        async move {
            let mut results = Vec::with_capacity(keys.len());
            for key in keys {
                results.push(self.get(key).await?);
            }
            Ok(results)
        }
    }

    /// Set multiple key-value pairs in a batch (SIMD-optimized).
    fn batch_set(&self, pairs: Vec<(&str, Vec<u8>)>) -> impl Future<Output = Result<()>> + Send {
        async move {
            for (key, value) in pairs {
                self.set(key, value).await?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // RED PHASE: These tests define the expected behavior
    // ============================================================

    #[tokio::test]
    async fn test_memory_kv_set_get() {
        let store = MemoryKvStore::new();

        store.set("key1", b"value1".to_vec()).await.unwrap();
        let value = store.get("key1").await.unwrap();

        assert_eq!(value, Some(b"value1".to_vec()));
    }

    #[tokio::test]
    async fn test_memory_kv_get_nonexistent() {
        let store = MemoryKvStore::new();

        let value = store.get("nonexistent").await.unwrap();

        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_memory_kv_overwrite() {
        let store = MemoryKvStore::new();

        store.set("key", b"value1".to_vec()).await.unwrap();
        store.set("key", b"value2".to_vec()).await.unwrap();
        let value = store.get("key").await.unwrap();

        assert_eq!(value, Some(b"value2".to_vec()));
    }

    #[tokio::test]
    async fn test_memory_kv_delete() {
        let store = MemoryKvStore::new();

        store.set("key", b"value".to_vec()).await.unwrap();
        store.delete("key").await.unwrap();
        let value = store.get("key").await.unwrap();

        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_memory_kv_delete_nonexistent() {
        let store = MemoryKvStore::new();

        // Should not error
        store.delete("nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn test_memory_kv_exists() {
        let store = MemoryKvStore::new();

        assert!(!store.exists("key").await.unwrap());

        store.set("key", b"value".to_vec()).await.unwrap();
        assert!(store.exists("key").await.unwrap());

        store.delete("key").await.unwrap();
        assert!(!store.exists("key").await.unwrap());
    }

    #[tokio::test]
    async fn test_memory_kv_batch_get() {
        let store = MemoryKvStore::new();

        store.set("a", b"1".to_vec()).await.unwrap();
        store.set("b", b"2".to_vec()).await.unwrap();
        // "c" intentionally not set

        let results = store.batch_get(&["a", "b", "c"]).await.unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(b"1".to_vec()));
        assert_eq!(results[1], Some(b"2".to_vec()));
        assert_eq!(results[2], None);
    }

    #[tokio::test]
    async fn test_memory_kv_batch_set() {
        let store = MemoryKvStore::new();

        store
            .batch_set(vec![("a", b"1".to_vec()), ("b", b"2".to_vec()), ("c", b"3".to_vec())])
            .await
            .unwrap();

        assert_eq!(store.get("a").await.unwrap(), Some(b"1".to_vec()));
        assert_eq!(store.get("b").await.unwrap(), Some(b"2".to_vec()));
        assert_eq!(store.get("c").await.unwrap(), Some(b"3".to_vec()));
    }

    #[tokio::test]
    async fn test_memory_kv_concurrent_access() {
        use std::sync::Arc;

        let store = Arc::new(MemoryKvStore::new());
        let mut handles = vec![];

        // Spawn 100 concurrent writers
        for i in 0..100 {
            let store = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                let key = format!("key{i}");
                let value = format!("value{i}").into_bytes();
                store.set(&key, value).await.unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all writes succeeded
        for i in 0..100 {
            let key = format!("key{i}");
            let expected = format!("value{i}").into_bytes();
            assert_eq!(store.get(&key).await.unwrap(), Some(expected));
        }
    }

    #[tokio::test]
    async fn test_memory_kv_empty_key() {
        let store = MemoryKvStore::new();

        store.set("", b"empty_key_value".to_vec()).await.unwrap();
        assert_eq!(store.get("").await.unwrap(), Some(b"empty_key_value".to_vec()));
    }

    #[tokio::test]
    async fn test_memory_kv_empty_value() {
        let store = MemoryKvStore::new();

        store.set("key", vec![]).await.unwrap();
        assert_eq!(store.get("key").await.unwrap(), Some(vec![]));
    }

    #[tokio::test]
    async fn test_memory_kv_large_value() {
        let store = MemoryKvStore::new();

        let large_value = vec![0u8; 1024 * 1024]; // 1MB
        store.set("large", large_value.clone()).await.unwrap();

        assert_eq!(store.get("large").await.unwrap(), Some(large_value));
    }

    #[tokio::test]
    async fn test_memory_kv_with_capacity() {
        let store = MemoryKvStore::with_capacity(100);
        store.set("key", b"value".to_vec()).await.unwrap();
        assert_eq!(store.get("key").await.unwrap(), Some(b"value".to_vec()));
    }

    #[tokio::test]
    async fn test_memory_kv_len_and_is_empty() {
        let store = MemoryKvStore::new();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.set("key1", b"value1".to_vec()).await.unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);

        store.set("key2", b"value2".to_vec()).await.unwrap();
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_kv_clear() {
        let store = MemoryKvStore::new();

        store.set("key1", b"value1".to_vec()).await.unwrap();
        store.set("key2", b"value2".to_vec()).await.unwrap();
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.get("key1").await.unwrap(), None);
    }

    #[test]
    fn test_memory_kv_default() {
        let store: MemoryKvStore = MemoryKvStore::default();
        assert!(store.is_empty());
    }

    // ============================================================
    // Compression Tests (GH-5) - RED PHASE
    // ============================================================

    #[cfg(feature = "compression")]
    mod compression_tests {
        use super::*;
        use crate::kv::{CompressedKvStore, Compression};

        #[tokio::test]
        async fn test_compressed_kv_lz4_roundtrip() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Lz4);

            let data = b"hello world compressed with lz4".to_vec();
            store.set("key", data.clone()).await.unwrap();
            let retrieved = store.get("key").await.unwrap();

            assert_eq!(retrieved, Some(data));
        }

        #[tokio::test]
        async fn test_compressed_kv_zstd_roundtrip() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Zstd);

            let data = b"hello world compressed with zstd".to_vec();
            store.set("key", data.clone()).await.unwrap();
            let retrieved = store.get("key").await.unwrap();

            assert_eq!(retrieved, Some(data));
        }

        #[tokio::test]
        async fn test_compressed_kv_reduces_size() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Lz4);

            // Highly compressible data (repeated zeros)
            let data = vec![0u8; 10000];
            store.set("key", data.clone()).await.unwrap();

            // Get raw compressed size from inner store
            let compressed = store.inner().get("key").await.unwrap().unwrap();

            // LZ4 should compress zeros significantly
            assert!(
                compressed.len() < data.len() / 5,
                "Compressed {} -> {} bytes (expected >5x reduction)",
                data.len(),
                compressed.len()
            );
        }

        #[tokio::test]
        async fn test_compressed_kv_empty_value() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Lz4);

            store.set("empty", vec![]).await.unwrap();
            assert_eq!(store.get("empty").await.unwrap(), Some(vec![]));
        }

        #[tokio::test]
        async fn test_compressed_kv_delete() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Lz4);

            store.set("key", b"value".to_vec()).await.unwrap();
            store.delete("key").await.unwrap();
            assert_eq!(store.get("key").await.unwrap(), None);
        }

        #[tokio::test]
        async fn test_compressed_kv_exists() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Lz4);

            assert!(!store.exists("key").await.unwrap());
            store.set("key", b"value".to_vec()).await.unwrap();
            assert!(store.exists("key").await.unwrap());
        }

        #[tokio::test]
        async fn test_compressed_kv_batch_operations() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Lz4);

            store.batch_set(vec![("a", b"alpha".to_vec()), ("b", b"beta".to_vec())]).await.unwrap();

            let results = store.batch_get(&["a", "b", "c"]).await.unwrap();
            assert_eq!(results[0], Some(b"alpha".to_vec()));
            assert_eq!(results[1], Some(b"beta".to_vec()));
            assert_eq!(results[2], None);
        }

        #[tokio::test]
        async fn test_compression_enum_variants() {
            assert_eq!(Compression::Lz4.as_str(), "lz4");
            assert_eq!(Compression::Zstd.as_str(), "zstd");
        }

        #[tokio::test]
        async fn test_compressed_kv_large_value() {
            let inner = MemoryKvStore::new();
            let store = CompressedKvStore::new(inner, Compression::Zstd);

            // 1MB of random-ish data
            let data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
            store.set("large", data.clone()).await.unwrap();

            let retrieved = store.get("large").await.unwrap();
            assert_eq!(retrieved, Some(data));
        }
    }
}
