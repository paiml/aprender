//! In-memory KV store implementation using `DashMap`.
//!
//! This is the default backend - data is lost on process restart.
//! For persistence, use `ParquetKvStore` (future).

use super::KvStore;
use crate::Result;
use dashmap::DashMap;
use std::future::Future;

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
    // NOT `async fn`: every operation here is a synchronous DashMap access, so
    // the futures resolve on their first poll. `std::future::ready` says that
    // in the type instead of leaving an `async` block with nothing to suspend
    // on. `+ Send` is carried over from the trait's own declaration in
    // kv/mod.rs, which clippy's suggestion drops.
    // clippy::unused_async_trait_impl (new in 1.98).
    fn get(&self, key: &str) -> impl Future<Output = Result<Option<Vec<u8>>>> + Send {
        std::future::ready(Ok(self.store.get(key).map(|v| v.value().clone())))
    }

    fn set(&self, key: &str, value: Vec<u8>) -> impl Future<Output = Result<()>> + Send {
        self.store.insert(key.to_string(), value);
        std::future::ready(Ok(()))
    }

    fn delete(&self, key: &str) -> impl Future<Output = Result<()>> + Send {
        self.store.remove(key);
        std::future::ready(Ok(()))
    }

    fn exists(&self, key: &str) -> impl Future<Output = Result<bool>> + Send {
        std::future::ready(Ok(self.store.contains_key(key)))
    }
}
