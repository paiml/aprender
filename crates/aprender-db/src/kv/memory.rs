//! In-memory KV store implementation using `DashMap`.
//!
//! This is the default backend - data is lost on process restart.
//! For persistence, use `ParquetKvStore` (future).

use super::KvStore;
use crate::Result;
use dashmap::DashMap;

/// In-memory key-value store using lock-free concurrent hashmap.
///
/// Thread-safe and optimized for high-concurrency read/write workloads.
/// Uses `DashMap` internally for O(1) average-case operations.
///
/// # Example
///
/// ```rust
/// use trueno_db::kv::{KvStore, MemoryKvStore};
///
/// # async fn example() -> trueno_db::Result<()> {
/// let store = MemoryKvStore::new();
/// store.set("hello", b"world".to_vec()).await?;
/// assert_eq!(store.get("hello").await?, Some(b"world".to_vec()));
/// # Ok(())
/// # }
/// ```
pub struct MemoryKvStore {
    store: DashMap<String, Vec<u8>>,
}

impl MemoryKvStore {
    /// Create a new in-memory KV store.
    #[must_use]
    pub fn new() -> Self {
        Self { store: DashMap::new() }
    }

    /// Create with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { store: DashMap::with_capacity(capacity) }
    }

    /// Get the number of entries in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.store.clear();
    }
}

impl Default for MemoryKvStore {
    fn default() -> Self {
        Self::new()
    }
}

impl KvStore for MemoryKvStore {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.store.get(key).map(|v| v.value().clone()))
    }

    async fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        self.store.insert(key.to_string(), value);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.store.remove(key);
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.store.contains_key(key))
    }
}
