//! Browser storage bindings for localStorage and sessionStorage.
//!
//! Provides a unified API for persisting data in the browser.
//!
//! # Example
//!
//! ```ignore
//! use presentar::browser::storage::{Storage, StorageType};
//!
//! let storage = Storage::new(StorageType::Local);
//! storage.set("key", "value");
//! let value = storage.get("key");
//! ```

use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;

/// Storage type (local or session).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageType {
    /// localStorage - persists across browser sessions
    #[default]
    Local,
    /// sessionStorage - cleared when browser tab closes
    Session,
}

/// Browser storage interface.
///
/// This provides a typed interface to browser storage APIs.
/// In WASM, this uses actual localStorage/sessionStorage.
/// In tests/non-WASM, this uses an in-memory fallback.
#[derive(Debug)]
pub struct Storage {
    storage_type: StorageType,
    /// In-memory fallback for non-WASM environments
    #[cfg(not(target_arch = "wasm32"))]
    memory: std::sync::Mutex<HashMap<String, String>>,
}

impl Default for Storage {
    fn default() -> Self {
        Self::new(StorageType::Local)
    }
}

impl Storage {
    /// Create a new storage instance.
    #[must_use]
    pub fn new(storage_type: StorageType) -> Self {
        Self {
            storage_type,
            #[cfg(not(target_arch = "wasm32"))]
            memory: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create localStorage instance.
    #[must_use]
    pub fn local() -> Self {
        Self::new(StorageType::Local)
    }

    /// Create sessionStorage instance.
    #[must_use]
    pub fn session() -> Self {
        Self::new(StorageType::Session)
    }

    /// Get the storage type.
    #[must_use]
    pub const fn storage_type(&self) -> StorageType {
        self.storage_type
    }

    /// Get a value from storage.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        #[cfg(target_arch = "wasm32")]
        {
            self.get_wasm(key)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.memory.lock().ok()?.get(key).cloned()
        }
    }

    /// Set a value in storage.
    pub fn set(&self, key: &str, value: &str) -> Result<(), StorageError> {
        #[cfg(target_arch = "wasm32")]
        {
            self.set_wasm(key, value)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.memory
                .lock()
                .map_err(|_| StorageError::AccessDenied)?
                .insert(key.to_string(), value.to_string());
            Ok(())
        }
    }

    /// Remove a value from storage.
    pub fn remove(&self, key: &str) -> Result<(), StorageError> {
        #[cfg(target_arch = "wasm32")]
        {
            self.remove_wasm(key)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.memory
                .lock()
                .map_err(|_| StorageError::AccessDenied)?
                .remove(key);
            Ok(())
        }
    }

    /// Clear all values in storage.
    pub fn clear(&self) -> Result<(), StorageError> {
        #[cfg(target_arch = "wasm32")]
        {
            self.clear_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.memory
                .lock()
                .map_err(|_| StorageError::AccessDenied)?
                .clear();
            Ok(())
        }
    }

    /// Get the number of items in storage.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            self.len_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.memory.lock().map(|m| m.len()).unwrap_or(0)
        }
    }

    /// Check if storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a key at the given index.
    #[must_use]
    pub fn key(&self, index: usize) -> Option<String> {
        #[cfg(target_arch = "wasm32")]
        {
            self.key_wasm(index)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.memory.lock().ok()?.keys().nth(index).cloned()
        }
    }

    /// Get all keys in storage.
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for i in 0..self.len() {
            if let Some(key) = self.key(i) {
                keys.push(key);
            }
        }
        keys
    }

    /// Get a value and deserialize it as JSON.
    pub fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, StorageError> {
        match self.get(key) {
            Some(json) => {
                let value = serde_json::from_str(&json)
                    .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Serialize a value as JSON and store it.
    pub fn set_json<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StorageError> {
        let json = serde_json::to_string(value)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        self.set(key, &json)
    }

    // WASM implementations
    #[cfg(target_arch = "wasm32")]
    fn get_storage(&self) -> Option<web_sys::Storage> {
        let window = web_sys::window()?;
        match self.storage_type {
            StorageType::Local => window.local_storage().ok()?,
            StorageType::Session => window.session_storage().ok()?,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn get_wasm(&self, key: &str) -> Option<String> {
        self.get_storage()?.get_item(key).ok()?
    }

    #[cfg(target_arch = "wasm32")]
    fn set_wasm(&self, key: &str, value: &str) -> Result<(), StorageError> {
        self.get_storage()
            .ok_or(StorageError::NotAvailable)?
            .set_item(key, value)
            .map_err(|_| StorageError::QuotaExceeded)
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_wasm(&self, key: &str) -> Result<(), StorageError> {
        self.get_storage()
            .ok_or(StorageError::NotAvailable)?
            .remove_item(key)
            .map_err(|_| StorageError::AccessDenied)
    }

    #[cfg(target_arch = "wasm32")]
    fn clear_wasm(&self) -> Result<(), StorageError> {
        self.get_storage()
            .ok_or(StorageError::NotAvailable)?
            .clear()
            .map_err(|_| StorageError::AccessDenied)
    }

    #[cfg(target_arch = "wasm32")]
    fn len_wasm(&self) -> usize {
        self.get_storage()
            .and_then(|s| s.length().ok())
            .unwrap_or(0) as usize
    }

    #[cfg(target_arch = "wasm32")]
    fn key_wasm(&self, index: usize) -> Option<String> {
        self.get_storage()?.key(index as u32).ok()?
    }
}

/// Storage error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// Storage is not available (e.g., in incognito mode)
    NotAvailable,
    /// Storage quota exceeded
    QuotaExceeded,
    /// Access denied
    AccessDenied,
    /// Serialization/deserialization error
    SerializationError(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable => write!(f, "storage not available"),
            Self::QuotaExceeded => write!(f, "storage quota exceeded"),
            Self::AccessDenied => write!(f, "storage access denied"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl std::error::Error for StorageError {}

/// Scoped storage with automatic key prefixing.
///
/// Useful for isolating storage between different parts of an application.
#[derive(Debug)]
pub struct ScopedStorage {
    inner: Storage,
    prefix: String,
}

impl ScopedStorage {
    /// Create a new scoped storage with the given prefix.
    #[must_use]
    pub fn new(storage: Storage, prefix: impl Into<String>) -> Self {
        Self {
            inner: storage,
            prefix: prefix.into(),
        }
    }

    /// Create a localStorage instance with the given prefix.
    #[must_use]
    pub fn local(prefix: impl Into<String>) -> Self {
        Self::new(Storage::local(), prefix)
    }

    /// Create a sessionStorage instance with the given prefix.
    #[must_use]
    pub fn session(prefix: impl Into<String>) -> Self {
        Self::new(Storage::session(), prefix)
    }

    /// Get the prefix.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    fn prefixed_key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }

    /// Get a value from storage.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        self.inner.get(&self.prefixed_key(key))
    }

    /// Set a value in storage.
    pub fn set(&self, key: &str, value: &str) -> Result<(), StorageError> {
        self.inner.set(&self.prefixed_key(key), value)
    }

    /// Remove a value from storage.
    pub fn remove(&self, key: &str) -> Result<(), StorageError> {
        self.inner.remove(&self.prefixed_key(key))
    }

    /// Get a value and deserialize it as JSON.
    pub fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, StorageError> {
        self.inner.get_json(&self.prefixed_key(key))
    }

    /// Serialize a value as JSON and store it.
    pub fn set_json<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StorageError> {
        self.inner.set_json(&self.prefixed_key(key), value)
    }

    /// Clear all values with this prefix.
    pub fn clear(&self) -> Result<(), StorageError> {
        let keys: Vec<_> = self
            .inner
            .keys()
            .into_iter()
            .filter(|k| k.starts_with(&format!("{}:", self.prefix)))
            .collect();

        for key in keys {
            // Remove the raw key (already prefixed)
            self.inner.remove(&key)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_type_default() {
        assert_eq!(StorageType::default(), StorageType::Local);
    }

    #[test]
    fn test_storage_new() {
        let storage = Storage::new(StorageType::Local);
        assert_eq!(storage.storage_type(), StorageType::Local);

        let storage = Storage::new(StorageType::Session);
        assert_eq!(storage.storage_type(), StorageType::Session);
    }

    #[test]
    fn test_storage_local() {
        let storage = Storage::local();
        assert_eq!(storage.storage_type(), StorageType::Local);
    }

    #[test]
    fn test_storage_session() {
        let storage = Storage::session();
        assert_eq!(storage.storage_type(), StorageType::Session);
    }

    #[test]
    fn test_storage_set_get() {
        let storage = Storage::local();
        storage.set("test_key", "test_value").unwrap();
        assert_eq!(storage.get("test_key"), Some("test_value".to_string()));
    }

    #[test]
    fn test_storage_get_nonexistent() {
        let storage = Storage::local();
        assert_eq!(storage.get("nonexistent"), None);
    }

    #[test]
    fn test_storage_remove() {
        let storage = Storage::local();
        storage.set("to_remove", "value").unwrap();
        assert!(storage.get("to_remove").is_some());
        storage.remove("to_remove").unwrap();
        assert!(storage.get("to_remove").is_none());
    }

    #[test]
    fn test_storage_clear() {
        let storage = Storage::local();
        storage.set("key1", "value1").unwrap();
        storage.set("key2", "value2").unwrap();
        assert!(!storage.is_empty());
        storage.clear().unwrap();
        assert!(storage.is_empty());
    }

    #[test]
    fn test_storage_len() {
        let storage = Storage::local();
        assert_eq!(storage.len(), 0);
        storage.set("key1", "value1").unwrap();
        assert_eq!(storage.len(), 1);
        storage.set("key2", "value2").unwrap();
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn test_storage_is_empty() {
        let storage = Storage::local();
        assert!(storage.is_empty());
        storage.set("key", "value").unwrap();
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_storage_keys() {
        let storage = Storage::local();
        storage.set("a", "1").unwrap();
        storage.set("b", "2").unwrap();
        let keys = storage.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
    }

    #[test]
    fn test_storage_json() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct TestData {
            name: String,
            value: i32,
        }

        let storage = Storage::local();
        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };

        storage.set_json("json_key", &data).unwrap();
        let loaded: Option<TestData> = storage.get_json("json_key").unwrap();
        assert_eq!(loaded, Some(data));
    }

    #[test]
    fn test_storage_json_nonexistent() {
        let storage = Storage::local();
        let result: Result<Option<String>, _> = storage.get_json("nonexistent");
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_scoped_storage_new() {
        let scoped = ScopedStorage::local("myapp");
        assert_eq!(scoped.prefix(), "myapp");
    }

    #[test]
    fn test_scoped_storage_set_get() {
        let scoped = ScopedStorage::local("test");
        scoped.set("key", "value").unwrap();
        assert_eq!(scoped.get("key"), Some("value".to_string()));

        // Note: In non-WASM mode, each Storage instance has its own memory,
        // so we can only verify the scoped behavior, not cross-instance sharing.
        // WASM mode uses actual browser localStorage which is shared.
    }

    #[test]
    fn test_scoped_storage_isolation() {
        let scope1 = ScopedStorage::local("scope1");
        let scope2 = ScopedStorage::local("scope2");

        scope1.set("key", "value1").unwrap();
        scope2.set("key", "value2").unwrap();

        assert_eq!(scope1.get("key"), Some("value1".to_string()));
        assert_eq!(scope2.get("key"), Some("value2".to_string()));
    }

    #[test]
    fn test_scoped_storage_json() {
        let scoped = ScopedStorage::local("json_test");
        scoped.set_json("data", &vec![1, 2, 3]).unwrap();
        let loaded: Option<Vec<i32>> = scoped.get_json("data").unwrap();
        assert_eq!(loaded, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_storage_error_display() {
        assert_eq!(
            StorageError::NotAvailable.to_string(),
            "storage not available"
        );
        assert_eq!(
            StorageError::QuotaExceeded.to_string(),
            "storage quota exceeded"
        );
        assert_eq!(
            StorageError::AccessDenied.to_string(),
            "storage access denied"
        );
        assert_eq!(
            StorageError::SerializationError("test".to_string()).to_string(),
            "serialization error: test"
        );
    }

    #[test]
    fn test_storage_default() {
        let storage = Storage::default();
        assert_eq!(storage.storage_type(), StorageType::Local);
    }
}
