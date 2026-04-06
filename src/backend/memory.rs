//! In-memory storage backend.

use std::{collections::HashMap, sync::RwLock};

use bytes::Bytes;

use super::StorageBackend;
use crate::error::{Error, Result};

/// An in-memory storage backend.
///
/// Useful for testing and WASM environments where filesystem access
/// is not available. All data is stored in memory and lost when the
/// backend is dropped.
///
/// # Thread Safety
///
/// This backend is thread-safe and can be shared across threads.
///
/// # Example
///
/// ```
/// use alimentar::backend::{MemoryBackend, StorageBackend};
/// use bytes::Bytes;
///
/// let backend = MemoryBackend::new();
/// backend.put("key", Bytes::from("value")).unwrap();
/// let data = backend.get("key").unwrap();
/// assert_eq!(data, Bytes::from("value"));
/// ```
#[derive(Debug, Default)]
pub struct MemoryBackend {
    data: RwLock<HashMap<String, Bytes>>,
}

impl MemoryBackend {
    /// Creates a new empty memory backend.
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a memory backend with initial data.
    pub fn with_data(data: HashMap<String, Bytes>) -> Self {
        Self {
            data: RwLock::new(data),
        }
    }

    /// Returns the number of keys stored.
    pub fn len(&self) -> usize {
        self.data.read().map(|d| d.len()).unwrap_or(0)
    }

    /// Returns true if no data is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all stored data.
    pub fn clear(&self) {
        if let Ok(mut data) = self.data.write() {
            data.clear();
        }
    }
}

impl StorageBackend for MemoryBackend {
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        let keys: Vec<String> = data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        Ok(keys)
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        data.get(key)
            .cloned()
            .ok_or_else(|| Error::storage(format!("Key not found: {}", key)))
    }

    fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let mut store = self
            .data
            .write()
            .map_err(|_| Error::storage("Failed to acquire write lock"))?;

        store.insert(key.to_string(), data);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|_| Error::storage("Failed to acquire write lock"))?;

        data.remove(key);
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        Ok(data.contains_key(key))
    }

    fn size(&self, key: &str) -> Result<u64> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        data.get(key)
            .map(|d| d.len() as u64)
            .ok_or_else(|| Error::storage(format!("Key not found: {}", key)))
    }
}

impl Clone for MemoryBackend {
    fn clone(&self) -> Self {
        let data = self.data.read().map(|d| d.clone()).unwrap_or_default();
        Self::with_data(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let backend = MemoryBackend::new();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_put_and_get() {
        let backend = MemoryBackend::new();

        let data = Bytes::from("hello world");
        backend
            .put("key", data.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let retrieved = backend
            .get("key")
            .ok()
            .unwrap_or_else(|| panic!("Should get"));
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_exists() {
        let backend = MemoryBackend::new();

        assert!(!backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should check")));

        backend
            .put("key", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert!(backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should check")));
    }

    #[test]
    fn test_delete() {
        let backend = MemoryBackend::new();

        backend
            .put("key", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert!(backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should exist")));

        backend
            .delete("key")
            .ok()
            .unwrap_or_else(|| panic!("Should delete"));

        assert!(!backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should not exist")));
    }

    #[test]
    fn test_list() {
        let backend = MemoryBackend::new();

        backend
            .put("foo/bar", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("foo/baz", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("other", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let foo_keys = backend
            .list("foo/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(foo_keys.len(), 2);

        let all_keys = backend
            .list("")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_size() {
        let backend = MemoryBackend::new();

        let data = Bytes::from("1234567890"); // 10 bytes
        backend
            .put("key", data)
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let size = backend
            .size("key")
            .ok()
            .unwrap_or_else(|| panic!("Should get size"));
        assert_eq!(size, 10);
    }

    #[test]
    fn test_get_nonexistent() {
        let backend = MemoryBackend::new();
        let result = backend.get("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_len() {
        let backend = MemoryBackend::new();
        assert_eq!(backend.len(), 0);

        backend
            .put("a", Bytes::from("1"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("b", Bytes::from("2"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert_eq!(backend.len(), 2);
    }

    #[test]
    fn test_clear() {
        let backend = MemoryBackend::new();

        backend
            .put("a", Bytes::from("1"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("b", Bytes::from("2"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert_eq!(backend.len(), 2);

        backend.clear();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_with_data() {
        let mut initial = HashMap::new();
        initial.insert("key".to_string(), Bytes::from("value"));

        let backend = MemoryBackend::with_data(initial);
        assert_eq!(backend.len(), 1);
        assert!(backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should check")));
    }

    #[test]
    fn test_clone() {
        let backend = MemoryBackend::new();
        backend
            .put("key", Bytes::from("value"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let cloned = backend;
        assert_eq!(cloned.len(), 1);
        assert_eq!(
            cloned
                .get("key")
                .ok()
                .unwrap_or_else(|| panic!("Should get")),
            Bytes::from("value")
        );
    }

    #[test]
    fn test_delete_nonexistent_is_ok() {
        let backend = MemoryBackend::new();
        let result = backend.delete("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_size_nonexistent() {
        let backend = MemoryBackend::new();
        let result = backend.size("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_default() {
        let backend = MemoryBackend::default();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_debug() {
        let backend = MemoryBackend::new();
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("MemoryBackend"));
    }

    #[test]
    fn test_list_with_prefix_filter() {
        let backend = MemoryBackend::new();

        backend
            .put("data/train.parquet", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("data/test.parquet", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("metadata/info.json", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let data_keys = backend
            .list("data/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(data_keys.len(), 2);

        let metadata_keys = backend
            .list("metadata/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(metadata_keys.len(), 1);
    }

    // === Additional coverage tests ===

    #[test]
    fn test_put_overwrite() {
        let backend = MemoryBackend::new();

        backend
            .put("key", Bytes::from("original"))
            .ok()
            .unwrap_or_else(|| panic!("put 1"));
        backend
            .put("key", Bytes::from("updated"))
            .ok()
            .unwrap_or_else(|| panic!("put 2"));

        let content = backend.get("key").ok().unwrap_or_else(|| panic!("get"));
        assert_eq!(content, Bytes::from("updated"));
        assert_eq!(backend.len(), 1);
    }

    #[test]
    fn test_clone_independence() {
        let backend = MemoryBackend::new();
        backend
            .put("key", Bytes::from("value"))
            .ok()
            .unwrap_or_else(|| panic!("put"));

        let cloned = backend.clone();

        // Modify original
        backend
            .put("new_key", Bytes::from("new_value"))
            .ok()
            .unwrap_or_else(|| panic!("put new"));

        // Clone should not have the new key (they're independent)
        assert_eq!(backend.len(), 2);
        assert_eq!(cloned.len(), 1);
    }

    #[test]
    fn test_list_empty_prefix() {
        let backend = MemoryBackend::new();
        backend
            .put("a", Bytes::from("1"))
            .ok()
            .unwrap_or_else(|| panic!("put"));
        backend
            .put("b", Bytes::from("2"))
            .ok()
            .unwrap_or_else(|| panic!("put"));
        backend
            .put("c", Bytes::from("3"))
            .ok()
            .unwrap_or_else(|| panic!("put"));

        let all = backend.list("").ok().unwrap_or_else(|| panic!("list"));
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_list_no_matches() {
        let backend = MemoryBackend::new();
        backend
            .put("data/file.txt", Bytes::from("content"))
            .ok()
            .unwrap_or_else(|| panic!("put"));

        let matches = backend
            .list("nonexistent/")
            .ok()
            .unwrap_or_else(|| panic!("list"));
        assert!(matches.is_empty());
    }

    #[test]
    fn test_size_empty_value() {
        let backend = MemoryBackend::new();
        backend
            .put("empty", Bytes::new())
            .ok()
            .unwrap_or_else(|| panic!("put"));

        let size = backend.size("empty").ok().unwrap_or_else(|| panic!("size"));
        assert_eq!(size, 0);
    }

    #[test]
    fn test_is_empty_after_operations() {
        let backend = MemoryBackend::new();
        assert!(backend.is_empty());

        backend
            .put("key", Bytes::from("value"))
            .ok()
            .unwrap_or_else(|| panic!("put"));
        assert!(!backend.is_empty());

        backend
            .delete("key")
            .ok()
            .unwrap_or_else(|| panic!("delete"));
        assert!(backend.is_empty());
    }

    #[test]
    fn test_many_keys() {
        let backend = MemoryBackend::new();

        for i in 0..100 {
            backend
                .put(&format!("key_{}", i), Bytes::from(format!("value_{}", i)))
                .ok()
                .unwrap_or_else(|| panic!("put"));
        }

        assert_eq!(backend.len(), 100);

        // Verify all keys exist
        for i in 0..100 {
            assert!(backend
                .exists(&format!("key_{}", i))
                .ok()
                .unwrap_or_else(|| panic!("exists")));
        }
    }

    #[test]
    fn test_delete_then_reput() {
        let backend = MemoryBackend::new();

        backend
            .put("key", Bytes::from("v1"))
            .ok()
            .unwrap_or_else(|| panic!("put 1"));
        backend
            .delete("key")
            .ok()
            .unwrap_or_else(|| panic!("delete"));
        backend
            .put("key", Bytes::from("v2"))
            .ok()
            .unwrap_or_else(|| panic!("put 2"));

        let content = backend.get("key").ok().unwrap_or_else(|| panic!("get"));
        assert_eq!(content, Bytes::from("v2"));
    }

    #[test]
    fn test_with_data_multiple() {
        let mut initial = HashMap::new();
        initial.insert("key1".to_string(), Bytes::from("value1"));
        initial.insert("key2".to_string(), Bytes::from("value2"));
        initial.insert("key3".to_string(), Bytes::from("value3"));

        let backend = MemoryBackend::with_data(initial);
        assert_eq!(backend.len(), 3);

        for i in 1..=3 {
            let content = backend
                .get(&format!("key{}", i))
                .ok()
                .unwrap_or_else(|| panic!("get"));
            assert_eq!(content, Bytes::from(format!("value{}", i)));
        }
    }

    #[test]
    fn test_large_value() {
        let backend = MemoryBackend::new();

        // 1MB value
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        backend
            .put("large", Bytes::from(data.clone()))
            .ok()
            .unwrap_or_else(|| panic!("put"));

        let retrieved = backend.get("large").ok().unwrap_or_else(|| panic!("get"));
        assert_eq!(retrieved.len(), 1_000_000);
        assert_eq!(&retrieved[..], &data[..]);
    }

    #[test]
    fn test_clear_and_reuse() {
        let backend = MemoryBackend::new();

        backend
            .put("old", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("put"));
        backend.clear();

        assert!(backend.is_empty());
        assert!(!backend
            .exists("old")
            .ok()
            .unwrap_or_else(|| panic!("exists")));

        backend
            .put("new", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("put"));
        assert_eq!(backend.len(), 1);
    }

    // === Additional memory backend tests ===

    #[test]
    fn test_list_with_various_prefixes() {
        let backend = MemoryBackend::new();

        backend
            .put("data/train/file1.parquet", Bytes::from("1"))
            .ok()
            .unwrap();
        backend
            .put("data/train/file2.parquet", Bytes::from("2"))
            .ok()
            .unwrap();
        backend
            .put("data/test/file1.parquet", Bytes::from("3"))
            .ok()
            .unwrap();
        backend
            .put("metadata/schema.json", Bytes::from("4"))
            .ok()
            .unwrap();

        assert_eq!(backend.list("data/train/").ok().unwrap().len(), 2);
        assert_eq!(backend.list("data/test/").ok().unwrap().len(), 1);
        assert_eq!(backend.list("data/").ok().unwrap().len(), 3);
        assert_eq!(backend.list("metadata/").ok().unwrap().len(), 1);
        assert_eq!(backend.list("").ok().unwrap().len(), 4);
    }

    #[test]
    fn test_size_of_empty_and_non_empty() {
        let backend = MemoryBackend::new();

        backend.put("empty", Bytes::new()).ok().unwrap();
        backend.put("small", Bytes::from("abc")).ok().unwrap();
        backend
            .put("medium", Bytes::from("0123456789"))
            .ok()
            .unwrap();

        assert_eq!(backend.size("empty").ok().unwrap(), 0);
        assert_eq!(backend.size("small").ok().unwrap(), 3);
        assert_eq!(backend.size("medium").ok().unwrap(), 10);
    }

    #[test]
    fn test_delete_idempotent() {
        let backend = MemoryBackend::new();

        backend.put("key", Bytes::from("value")).ok().unwrap();

        // First delete succeeds
        assert!(backend.delete("key").is_ok());
        // Second delete also succeeds (idempotent)
        assert!(backend.delete("key").is_ok());
        // Third delete still succeeds
        assert!(backend.delete("key").is_ok());
    }

    #[test]
    fn test_exists_after_operations() {
        let backend = MemoryBackend::new();

        // Initially doesn't exist
        assert!(!backend.exists("key").ok().unwrap());

        // After put, exists
        backend.put("key", Bytes::from("value")).ok().unwrap();
        assert!(backend.exists("key").ok().unwrap());

        // After delete, doesn't exist
        backend.delete("key").ok().unwrap();
        assert!(!backend.exists("key").ok().unwrap());

        // After re-put, exists again
        backend.put("key", Bytes::from("new value")).ok().unwrap();
        assert!(backend.exists("key").ok().unwrap());
    }

    #[test]
    fn test_clone_deep_copy() {
        let backend = MemoryBackend::new();
        backend.put("key1", Bytes::from("value1")).ok().unwrap();
        backend.put("key2", Bytes::from("value2")).ok().unwrap();

        let cloned = backend.clone();

        // Modify original
        backend.put("key3", Bytes::from("value3")).ok().unwrap();
        backend.delete("key1").ok().unwrap();

        // Clone should be unchanged
        assert_eq!(cloned.len(), 2);
        assert!(cloned.exists("key1").ok().unwrap());
        assert!(!cloned.exists("key3").ok().unwrap());
    }

    #[test]
    fn test_list_partial_prefix_match() {
        let backend = MemoryBackend::new();

        backend.put("prefix_a_1", Bytes::from("1")).ok().unwrap();
        backend.put("prefix_a_2", Bytes::from("2")).ok().unwrap();
        backend.put("prefix_b_1", Bytes::from("3")).ok().unwrap();
        backend.put("other", Bytes::from("4")).ok().unwrap();

        // Partial prefix match
        assert_eq!(backend.list("prefix_a").ok().unwrap().len(), 2);
        assert_eq!(backend.list("prefix_b").ok().unwrap().len(), 1);
        assert_eq!(backend.list("prefix_").ok().unwrap().len(), 3);
        assert_eq!(backend.list("pre").ok().unwrap().len(), 3);
    }

    #[test]
    fn test_get_error_message() {
        let backend = MemoryBackend::new();

        let result = backend.get("nonexistent_key");
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("nonexistent_key") || msg.contains("not found"));
        }
    }

    #[test]
    fn test_size_error_message() {
        let backend = MemoryBackend::new();

        let result = backend.size("missing_file");
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("missing_file") || msg.contains("not found"));
        }
    }

    #[test]
    fn test_with_data_preserves_all() {
        let mut initial = HashMap::new();
        initial.insert("a".to_string(), Bytes::from("1"));
        initial.insert("b".to_string(), Bytes::from("2"));
        initial.insert("c".to_string(), Bytes::from("3"));

        let backend = MemoryBackend::with_data(initial);

        assert_eq!(backend.len(), 3);
        assert_eq!(backend.get("a").ok().unwrap(), Bytes::from("1"));
        assert_eq!(backend.get("b").ok().unwrap(), Bytes::from("2"));
        assert_eq!(backend.get("c").ok().unwrap(), Bytes::from("3"));
    }

    #[test]
    fn test_binary_data_roundtrip() {
        let backend = MemoryBackend::new();

        // Binary data including null bytes
        let binary: Vec<u8> = (0..=255).collect();
        backend
            .put("binary", Bytes::from(binary.clone()))
            .ok()
            .unwrap();

        let retrieved = backend.get("binary").ok().unwrap();
        assert_eq!(retrieved.as_ref(), binary.as_slice());
    }

    #[test]
    fn test_concurrent_access_simulation() {
        let backend = MemoryBackend::new();

        // Simulate multiple operations
        for i in 0..100 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i);
            backend.put(&key, Bytes::from(value.clone())).ok().unwrap();
        }

        assert_eq!(backend.len(), 100);

        // Read all back
        for i in 0..100 {
            let key = format!("key_{}", i);
            let expected = format!("value_{}", i);
            assert_eq!(backend.get(&key).ok().unwrap(), Bytes::from(expected));
        }
    }

    #[test]
    fn test_clear_multiple_times() {
        let backend = MemoryBackend::new();

        backend.put("key", Bytes::from("value")).ok().unwrap();
        backend.clear();
        backend.clear(); // Should be safe to call multiple times
        backend.clear();

        assert!(backend.is_empty());
    }
}
