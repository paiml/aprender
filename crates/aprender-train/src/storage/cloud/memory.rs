//! In-memory artifact backend for testing

use crate::storage::cloud::error::{CloudError, Result};
use crate::storage::cloud::metadata::ArtifactMetadata;
use crate::storage::cloud::traits::{compute_hash, ArtifactBackend};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// In-memory artifact backend for testing
#[derive(Debug, Default)]
pub struct InMemoryBackend {
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    metadata: Arc<RwLock<HashMap<String, ArtifactMetadata>>>,
}

impl InMemoryBackend {
    /// Create a new in-memory backend
    pub fn new() -> Self {
        Self::default()
    }
}

impl ArtifactBackend for InMemoryBackend {
    fn put(&self, name: &str, data: &[u8]) -> Result<String> {
        let hash = compute_hash(data);

        self.data
            .write()
            .map_err(|e| CloudError::Backend(e.to_string()))?
            .insert(hash.clone(), data.to_vec());

        let metadata = ArtifactMetadata::new(name, &hash, data.len() as u64);
        self.metadata
            .write()
            .map_err(|e| CloudError::Backend(e.to_string()))?
            .insert(hash.clone(), metadata);

        Ok(hash)
    }

    fn get(&self, hash: &str) -> Result<Vec<u8>> {
        self.data
            .read()
            .map_err(|e| CloudError::Backend(e.to_string()))?
            .get(hash)
            .cloned()
            .ok_or_else(|| CloudError::NotFound(hash.to_string()))
    }

    fn exists(&self, hash: &str) -> Result<bool> {
        Ok(self.data.read().map_err(|e| CloudError::Backend(e.to_string()))?.contains_key(hash))
    }

    fn delete(&self, hash: &str) -> Result<()> {
        let removed =
            self.data.write().map_err(|e| CloudError::Backend(e.to_string()))?.remove(hash);
        if removed.is_none() {
            return Err(CloudError::NotFound(hash.to_string()));
        }
        self.metadata.write().map_err(|e| CloudError::Backend(e.to_string()))?.remove(hash);
        Ok(())
    }

    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata> {
        self.metadata
            .read()
            .map_err(|e| CloudError::Backend(e.to_string()))?
            .get(hash)
            .cloned()
            .ok_or_else(|| CloudError::NotFound(hash.to_string()))
    }

    fn list(&self) -> Result<Vec<ArtifactMetadata>> {
        Ok(self
            .metadata
            .read()
            .map_err(|e| CloudError::Backend(e.to_string()))?
            .values()
            .cloned()
            .collect())
    }

    fn backend_type(&self) -> &'static str {
        "memory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_backend_put_get() {
        let backend = InMemoryBackend::new();
        let data = b"test data";
        let hash = backend.put("test.bin", data).expect("operation should succeed");

        let retrieved = backend.get(&hash).expect("key should exist");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_in_memory_backend_exists() {
        let backend = InMemoryBackend::new();
        let hash = backend.put("test.bin", b"data").expect("operation should succeed");

        assert!(backend.exists(&hash).expect("operation should succeed"));
        assert!(!backend.exists("nonexistent").expect("operation should succeed"));
    }

    #[test]
    fn test_in_memory_backend_delete() {
        let backend = InMemoryBackend::new();
        let hash = backend.put("test.bin", b"data").expect("operation should succeed");

        backend.delete(&hash).expect("operation should succeed");
        assert!(!backend.exists(&hash).expect("operation should succeed"));
    }

    #[test]
    fn test_in_memory_backend_delete_not_found() {
        let backend = InMemoryBackend::new();
        let result = backend.delete("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_in_memory_backend_get_metadata() {
        let backend = InMemoryBackend::new();
        let data = b"test data 123";
        let hash = backend.put("model.bin", data).expect("operation should succeed");

        let meta = backend.get_metadata(&hash).expect("operation should succeed");
        assert_eq!(meta.name, "model.bin");
        assert_eq!(meta.size, data.len() as u64);
    }

    #[test]
    fn test_in_memory_backend_list() {
        let backend = InMemoryBackend::new();
        backend.put("file1.bin", b"data1").expect("operation should succeed");
        backend.put("file2.bin", b"data2").expect("operation should succeed");

        let list = backend.list().expect("operation should succeed");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_in_memory_backend_type() {
        let backend = InMemoryBackend::new();
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_in_memory_backend_get_not_found() {
        let backend = InMemoryBackend::new();
        let result = backend.get("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_in_memory_backend_get_metadata_not_found() {
        let backend = InMemoryBackend::new();
        let result = backend.get_metadata("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_in_memory_backend_default() {
        let backend = InMemoryBackend::default();
        assert_eq!(backend.backend_type(), "memory");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_memory_backend_roundtrip(
            name in "[a-zA-Z0-9_]{1,50}",
            data in prop::collection::vec(any::<u8>(), 1..1000)
        ) {
            let backend = InMemoryBackend::new();
            let hash = backend.put(&name, &data).expect("operation should succeed");
            let retrieved = backend.get(&hash).expect("key should exist");
            prop_assert_eq!(retrieved, data);
        }
    }
}
