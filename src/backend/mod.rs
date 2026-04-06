//! Storage backends for alimentar.
//!
//! Backends provide abstracted storage operations for datasets and registries.
//! The [`StorageBackend`] trait defines the interface, with implementations
//! for local filesystem, S3-compatible storage, and in-memory storage.

#[cfg(feature = "http")]
pub mod http;
#[cfg(feature = "local")]
pub mod local;
pub mod memory;
#[cfg(feature = "s3")]
pub mod s3;

use bytes::Bytes;
#[cfg(feature = "http")]
pub use http::{HttpBackend, RangeHttpBackend};
#[cfg(feature = "local")]
pub use local::LocalBackend;
pub use memory::MemoryBackend;
#[cfg(feature = "s3")]
pub use s3::{CredentialSource, S3Backend};

use crate::error::Result;

/// A storage backend for reading and writing data.
///
/// Backends abstract the underlying storage mechanism, allowing datasets
/// and registries to work with local files, cloud storage, or in-memory
/// buffers using the same interface.
///
/// # Async Design
///
/// All operations are synchronous for now (v0.1). Future versions will
/// add async variants behind the `tokio-runtime` feature flag.
pub trait StorageBackend: Send + Sync {
    /// Lists all keys with the given prefix.
    ///
    /// Returns a vector of key names (relative to the backend root).
    ///
    /// # Errors
    ///
    /// Returns an error if the listing operation fails.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Reads data from the given key.
    ///
    /// # Errors
    ///
    /// Returns an error if the key does not exist or cannot be read.
    fn get(&self, key: &str) -> Result<Bytes>;

    /// Writes data to the given key.
    ///
    /// Creates parent directories/prefixes as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn put(&self, key: &str, data: Bytes) -> Result<()>;

    /// Deletes the given key.
    ///
    /// # Errors
    ///
    /// Returns an error if the key cannot be deleted.
    fn delete(&self, key: &str) -> Result<()>;

    /// Checks if the given key exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the existence check fails.
    fn exists(&self, key: &str) -> Result<bool>;

    /// Returns the size of the data at the given key in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the key does not exist.
    fn size(&self, key: &str) -> Result<u64>;
}

/// Configuration for storage backends.
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// Local filesystem backend.
    Local {
        /// Root directory for storage.
        root: std::path::PathBuf,
    },
    /// In-memory backend (for testing or WASM).
    Memory,
    /// S3-compatible backend (requires `s3` feature).
    #[cfg(feature = "s3")]
    S3 {
        /// Bucket name.
        bucket: String,
        /// AWS region.
        region: String,
        /// Custom endpoint URL (None = AWS, Some = MinIO/Ceph/etc).
        endpoint: Option<String>,
        /// Credential source for authentication.
        credentials: CredentialSource,
    },
}

impl BackendConfig {
    /// Creates a local backend configuration.
    pub fn local(root: impl Into<std::path::PathBuf>) -> Self {
        Self::Local { root: root.into() }
    }

    /// Creates an in-memory backend configuration.
    pub fn memory() -> Self {
        Self::Memory
    }

    /// Creates an S3 backend configuration for AWS.
    #[cfg(feature = "s3")]
    pub fn s3_aws(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self::S3 {
            bucket: bucket.into(),
            region: region.into(),
            endpoint: None,
            credentials: CredentialSource::Environment,
        }
    }

    /// Creates an S3 backend configuration for a custom endpoint (MinIO, etc.).
    #[cfg(feature = "s3")]
    pub fn s3_custom(
        bucket: impl Into<String>,
        region: impl Into<String>,
        endpoint: impl Into<String>,
        credentials: CredentialSource,
    ) -> Self {
        Self::S3 {
            bucket: bucket.into(),
            region: region.into(),
            endpoint: Some(endpoint.into()),
            credentials,
        }
    }
}

/// Creates a storage backend from configuration.
///
/// # Errors
///
/// Returns an error if the backend cannot be created.
pub fn create_backend(config: BackendConfig) -> Result<Box<dyn StorageBackend>> {
    match config {
        #[cfg(feature = "local")]
        BackendConfig::Local { root } => Ok(Box::new(LocalBackend::new(root)?)),
        #[cfg(not(feature = "local"))]
        BackendConfig::Local { .. } => Err(crate::error::Error::invalid_config(
            "Local backend requires 'local' feature",
        )),
        BackendConfig::Memory => Ok(Box::new(MemoryBackend::new())),
        #[cfg(feature = "s3")]
        BackendConfig::S3 {
            bucket,
            region,
            endpoint,
            credentials,
        } => Ok(Box::new(S3Backend::new(
            bucket,
            region,
            endpoint,
            credentials,
        )?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_local() {
        let config = BackendConfig::local("/tmp/test");
        if let BackendConfig::Local { root } = config {
            assert_eq!(root, std::path::PathBuf::from("/tmp/test"));
        } else {
            panic!("Expected Local config");
        }
    }

    #[test]
    fn test_backend_config_memory() {
        let config = BackendConfig::memory();
        assert!(matches!(config, BackendConfig::Memory));
    }

    #[test]
    fn test_create_memory_backend() {
        let backend = create_backend(BackendConfig::Memory);
        assert!(backend.is_ok());
    }

    #[cfg(feature = "local")]
    #[test]
    fn test_create_local_backend() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = create_backend(BackendConfig::local(temp_dir.path()));
        assert!(backend.is_ok());
    }

    #[test]
    fn test_create_memory_backend_operations() {
        let backend = create_backend(BackendConfig::Memory)
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        // Test put and get
        backend
            .put("test_key", bytes::Bytes::from("test_value"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let data = backend
            .get("test_key")
            .ok()
            .unwrap_or_else(|| panic!("Should get"));
        assert_eq!(data, bytes::Bytes::from("test_value"));

        // Test exists
        let exists = backend
            .exists("test_key")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists"));
        assert!(exists);

        // Test size
        let size = backend
            .size("test_key")
            .ok()
            .unwrap_or_else(|| panic!("Should get size"));
        assert_eq!(size, 10);

        // Test list
        let list = backend
            .list("")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(list.len(), 1);

        // Test delete
        backend
            .delete("test_key")
            .ok()
            .unwrap_or_else(|| panic!("Should delete"));

        let exists_after = backend
            .exists("test_key")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists"));
        assert!(!exists_after);
    }

    #[cfg(feature = "local")]
    #[test]
    fn test_create_local_backend_operations() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = create_backend(BackendConfig::local(temp_dir.path()))
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        // Basic operations
        backend
            .put("data.txt", bytes::Bytes::from("content"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let exists = backend
            .exists("data.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists"));
        assert!(exists);
    }

    #[test]
    fn test_backend_config_debug() {
        let config = BackendConfig::local("/tmp/test");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Local"));

        let config2 = BackendConfig::memory();
        let debug_str2 = format!("{:?}", config2);
        assert!(debug_str2.contains("Memory"));
    }

    #[test]
    fn test_backend_config_clone() {
        let config = BackendConfig::local("/tmp/test");
        let cloned = config;

        if let BackendConfig::Local { root } = cloned {
            assert_eq!(root, std::path::PathBuf::from("/tmp/test"));
        } else {
            panic!("Expected Local config");
        }
    }
}
