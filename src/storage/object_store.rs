//! Object store for content-addressed artifact storage.

use crate::error::{PachaError, Result};
use crate::storage::ContentAddress;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Content-addressed object store.
///
/// Stores artifacts using BLAKE3 hash prefixes for sharding:
/// ```text
/// objects/
/// ├── ab/
/// │   └── cdef1234...
/// ├── cd/
/// │   └── ef5678...
/// └── ...
/// ```
#[derive(Debug)]
pub struct ObjectStore {
    /// Base path for object storage.
    base_path: PathBuf,
}

impl ObjectStore {
    /// Create a new object store at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }

    /// Open an existing object store.
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't exist.
    pub fn open<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        if !base_path.exists() {
            return Err(PachaError::NotInitialized(base_path));
        }
        Ok(Self { base_path })
    }

    /// Get the base path.
    #[must_use]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Store bytes and return their content address.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn put(&self, data: &[u8]) -> Result<ContentAddress> {
        let addr = ContentAddress::from_bytes(data);
        self.put_with_address(data, &addr)?;
        Ok(addr)
    }

    /// Store bytes at a specific content address.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails or hash doesn't match.
    pub fn put_with_address(&self, data: &[u8], addr: &ContentAddress) -> Result<()> {
        // Verify hash matches
        if !addr.verify(data) {
            return Err(PachaError::HashMismatch {
                expected: addr.hash_hex(),
                actual: ContentAddress::from_bytes(data).hash_hex(),
            });
        }

        let path = self.object_path(addr);

        // Skip if already exists (content-addressed = idempotent)
        if path.exists() {
            return Ok(());
        }

        // Create parent directory
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let temp_path = path.with_extension("tmp");
        {
            let file = File::create(&temp_path)?;
            let mut writer = BufWriter::new(file);
            writer.write_all(data)?;
            writer.flush()?;
        }

        // Atomic rename
        fs::rename(&temp_path, &path)?;

        Ok(())
    }

    /// Store from a reader and return content address.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or writing fails.
    pub fn put_reader<R: Read>(&self, mut reader: R) -> Result<ContentAddress> {
        // Read all data first to compute hash
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        self.put(&data)
    }

    /// Get data by content address.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't exist or reading fails.
    pub fn get(&self, addr: &ContentAddress) -> Result<Vec<u8>> {
        let path = self.object_path(addr);

        if !path.exists() {
            return Err(PachaError::NotFound {
                kind: "object".to_string(),
                name: addr.hash_hex(),
                version: "n/a".to_string(),
            });
        }

        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let capacity = usize::try_from(addr.size()).unwrap_or(0);
        let mut data = Vec::with_capacity(capacity);
        reader.read_to_end(&mut data)?;

        // Verify integrity
        if !addr.verify(&data) {
            return Err(PachaError::HashMismatch {
                expected: addr.hash_hex(),
                actual: ContentAddress::from_bytes(&data).hash_hex(),
            });
        }

        Ok(data)
    }

    /// Check if an object exists.
    #[must_use]
    pub fn exists(&self, addr: &ContentAddress) -> bool {
        self.object_path(addr).exists()
    }

    /// Delete an object by content address.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete(&self, addr: &ContentAddress) -> Result<bool> {
        let path = self.object_path(addr);

        if !path.exists() {
            return Ok(false);
        }

        fs::remove_file(&path)?;

        // Try to remove empty parent directory
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir(parent); // Ignore if not empty
        }

        Ok(true)
    }

    /// List all content addresses in the store.
    ///
    /// # Errors
    ///
    /// Returns an error if reading the directory fails.
    pub fn list(&self) -> Result<Vec<String>> {
        let mut addresses = Vec::new();

        if !self.base_path.exists() {
            return Ok(addresses);
        }

        for prefix_entry in fs::read_dir(&self.base_path)? {
            let prefix_entry = prefix_entry?;
            if !prefix_entry.file_type()?.is_dir() {
                continue;
            }

            for entry in fs::read_dir(prefix_entry.path())? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    if let Some(name) = entry.file_name().to_str() {
                        // Exclude .tmp working files from listing
                        #[allow(clippy::case_sensitive_file_extension_comparisons)]
                        if !name.ends_with(".tmp") {
                            addresses.push(name.to_string());
                        }
                    }
                }
            }
        }

        Ok(addresses)
    }

    /// Get total size of all stored objects in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    pub fn total_size(&self) -> Result<u64> {
        let mut total = 0u64;

        if !self.base_path.exists() {
            return Ok(0);
        }

        for prefix_entry in fs::read_dir(&self.base_path)? {
            let prefix_entry = prefix_entry?;
            if !prefix_entry.file_type()?.is_dir() {
                continue;
            }

            for entry in fs::read_dir(prefix_entry.path())? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    total += entry.metadata()?.len();
                }
            }
        }

        Ok(total)
    }

    /// Get the file path for a content address.
    fn object_path(&self, addr: &ContentAddress) -> PathBuf {
        self.base_path.join(addr.storage_prefix()).join(addr.hash_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, ObjectStore) {
        let dir = TempDir::new().unwrap();
        let store = ObjectStore::new(dir.path().join("objects")).unwrap();
        (dir, store)
    }

    #[test]
    fn test_put_and_get() {
        let (_dir, store) = setup();
        let data = b"hello world";

        let addr = store.put(data).unwrap();
        assert_eq!(addr.size(), 11);

        let retrieved = store.get(&addr).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_put_idempotent() {
        let (_dir, store) = setup();
        let data = b"test data";

        let addr1 = store.put(data).unwrap();
        let addr2 = store.put(data).unwrap();

        assert_eq!(addr1, addr2);
    }

    #[test]
    fn test_exists() {
        let (_dir, store) = setup();
        let data = b"test";

        let addr = ContentAddress::from_bytes(data);
        assert!(!store.exists(&addr));

        store.put(data).unwrap();
        assert!(store.exists(&addr));
    }

    #[test]
    fn test_delete() {
        let (_dir, store) = setup();
        let data = b"delete me";

        let addr = store.put(data).unwrap();
        assert!(store.exists(&addr));

        let deleted = store.delete(&addr).unwrap();
        assert!(deleted);
        assert!(!store.exists(&addr));

        // Delete non-existent returns false
        let deleted_again = store.delete(&addr).unwrap();
        assert!(!deleted_again);
    }

    #[test]
    fn test_get_not_found() {
        let (_dir, store) = setup();
        let addr = ContentAddress::from_bytes(b"nonexistent");

        let result = store.get(&addr);
        assert!(matches!(result, Err(PachaError::NotFound { .. })));
    }

    #[test]
    fn test_put_with_wrong_address() {
        let (_dir, store) = setup();
        let data = b"actual data";
        let wrong_addr = ContentAddress::from_bytes(b"different data");

        let result = store.put_with_address(data, &wrong_addr);
        assert!(matches!(result, Err(PachaError::HashMismatch { .. })));
    }

    #[test]
    fn test_list() {
        let (_dir, store) = setup();

        store.put(b"one").unwrap();
        store.put(b"two").unwrap();
        store.put(b"three").unwrap();

        let addresses = store.list().unwrap();
        assert_eq!(addresses.len(), 3);
    }

    #[test]
    fn test_total_size() {
        let (_dir, store) = setup();

        store.put(b"12345").unwrap();
        store.put(b"67890").unwrap();

        let size = store.total_size().unwrap();
        assert_eq!(size, 10);
    }

    #[test]
    fn test_open_nonexistent() {
        let dir = TempDir::new().unwrap();
        let result = ObjectStore::open(dir.path().join("nonexistent"));
        assert!(matches!(result, Err(PachaError::NotInitialized(_))));
    }

    // Property-based tests
    proptest! {
        #[test]
        fn prop_roundtrip(data: Vec<u8>) {
            let dir = TempDir::new().unwrap();
            let store = ObjectStore::new(dir.path().join("objects")).unwrap();

            let addr = store.put(&data).unwrap();
            let retrieved = store.get(&addr).unwrap();

            prop_assert_eq!(data, retrieved);
        }

        #[test]
        fn prop_idempotent(data: Vec<u8>) {
            let dir = TempDir::new().unwrap();
            let store = ObjectStore::new(dir.path().join("objects")).unwrap();

            let addr1 = store.put(&data).unwrap();
            let addr2 = store.put(&data).unwrap();

            prop_assert_eq!(addr1, addr2);
        }

        #[test]
        fn prop_deduplication(data: Vec<u8>) {
            let dir = TempDir::new().unwrap();
            let store = ObjectStore::new(dir.path().join("objects")).unwrap();

            // Store same data twice
            store.put(&data).unwrap();
            store.put(&data).unwrap();

            // Should only have one object
            let addresses = store.list().unwrap();
            prop_assert_eq!(addresses.len(), 1);
        }
    }
}
