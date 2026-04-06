//! Local filesystem artifact backend

use crate::storage::cloud::error::{CloudError, Result};
use crate::storage::cloud::metadata::ArtifactMetadata;
use crate::storage::cloud::traits::{compute_hash, ArtifactBackend};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Local filesystem artifact backend
#[derive(Debug)]
pub struct LocalBackend {
    base_path: PathBuf,
    metadata: Arc<RwLock<HashMap<String, ArtifactMetadata>>>,
}

impl LocalBackend {
    /// Create a new local backend
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path, metadata: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Create a new local backend and ensure directory exists
    pub fn new_and_init(base_path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;
        Ok(Self::new(base_path))
    }

    /// Get the file path for a hash
    fn hash_to_path(&self, hash: &str) -> PathBuf {
        // Use subdirectories based on hash prefix for better filesystem performance
        let prefix = hash.get(..2).unwrap_or(hash);
        self.base_path.join(prefix).join(hash)
    }
}

impl ArtifactBackend for LocalBackend {
    fn put(&self, name: &str, data: &[u8]) -> Result<String> {
        let hash = compute_hash(data);
        let path = self.hash_to_path(&hash);

        // Create parent directory
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write data
        let mut file = std::fs::File::create(&path)?;
        file.write_all(data)?;

        // Store metadata
        let metadata = ArtifactMetadata::new(name, &hash, data.len() as u64);
        self.metadata
            .write()
            .expect("metadata RwLock must not be poisoned")
            .insert(hash.clone(), metadata);

        Ok(hash)
    }

    fn get(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.hash_to_path(hash);

        if !path.exists() {
            return Err(CloudError::NotFound(hash.to_string()));
        }

        let mut file = std::fs::File::open(&path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        // Verify hash
        let computed = compute_hash(&data);
        if computed != hash {
            return Err(CloudError::Backend(format!(
                "Hash mismatch: expected {hash}, got {computed}"
            )));
        }

        Ok(data)
    }

    fn exists(&self, hash: &str) -> Result<bool> {
        let path = self.hash_to_path(hash);
        Ok(path.exists())
    }

    fn delete(&self, hash: &str) -> Result<()> {
        let path = self.hash_to_path(hash);

        if !path.exists() {
            return Err(CloudError::NotFound(hash.to_string()));
        }

        std::fs::remove_file(&path)?;
        self.metadata.write().expect("metadata RwLock must not be poisoned").remove(hash);

        Ok(())
    }

    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata> {
        self.metadata
            .read()
            .expect("metadata RwLock must not be poisoned")
            .get(hash)
            .cloned()
            .ok_or_else(|| CloudError::NotFound(hash.to_string()))
    }

    fn list(&self) -> Result<Vec<ArtifactMetadata>> {
        Ok(self
            .metadata
            .read()
            .expect("metadata RwLock must not be poisoned")
            .values()
            .cloned()
            .collect())
    }

    fn backend_type(&self) -> &'static str {
        "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_local_backend_put_get() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let data = b"local test data";
        let hash = backend.put("test.bin", data).expect("operation should succeed");

        let retrieved = backend.get(&hash).expect("key should exist");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_local_backend_exists() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let hash = backend.put("test.bin", b"data").expect("operation should succeed");
        assert!(backend.exists(&hash).expect("operation should succeed"));
        assert!(!backend.exists("nonexistent").expect("operation should succeed"));
    }

    #[test]
    fn test_local_backend_delete() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let hash = backend.put("test.bin", b"data").expect("operation should succeed");
        backend.delete(&hash).expect("operation should succeed");
        assert!(!backend.exists(&hash).expect("operation should succeed"));
    }

    #[test]
    fn test_local_backend_type() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");
        assert_eq!(backend.backend_type(), "local");
    }

    #[test]
    fn test_local_backend_get_not_found() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let result = backend.get("nonexistent_hash");
        assert!(result.is_err());
        match result {
            Err(CloudError::NotFound(hash)) => assert_eq!(hash, "nonexistent_hash"),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_local_backend_delete_not_found() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let result = backend.delete("nonexistent_hash");
        assert!(result.is_err());
    }

    #[test]
    fn test_local_backend_get_metadata() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let data = b"test data for metadata";
        let hash = backend.put("test_file.bin", data).expect("operation should succeed");

        let meta = backend.get_metadata(&hash).expect("operation should succeed");
        assert_eq!(meta.name, "test_file.bin");
        assert_eq!(meta.size, data.len() as u64);
    }

    #[test]
    fn test_local_backend_get_metadata_not_found() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        let result = backend.get_metadata("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_local_backend_list() {
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let backend =
            LocalBackend::new_and_init(tmp.path().to_path_buf()).expect("operation should succeed");

        backend.put("file1.bin", b"data1").expect("operation should succeed");
        backend.put("file2.bin", b"data2").expect("operation should succeed");
        backend.put("file3.bin", b"data3").expect("operation should succeed");

        let list = backend.list().expect("operation should succeed");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_local_backend_new() {
        let backend = LocalBackend::new(PathBuf::from("/tmp/test"));
        assert_eq!(backend.backend_type(), "local");
    }
}
