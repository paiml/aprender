//! Persistent file-based cache for crates.io responses.

use super::types::{CrateResponse, PersistentCacheEntry};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

/// Persistent cache stored on disk
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PersistentCache {
    pub entries: HashMap<String, PersistentCacheEntry>,
}

impl PersistentCache {
    /// Get cache file path
    pub fn cache_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("batuta")
            .join("crates_io_cache.json")
    }

    /// Load cache from disk
    pub fn load() -> Self {
        Self::load_from(&Self::cache_path())
    }

    /// Load cache from a specific path
    pub fn load_from(path: &std::path::Path) -> Self {
        if path.exists() {
            if let Ok(data) = fs::read_to_string(path) {
                if let Ok(cache) = serde_json::from_str(&data) {
                    return cache;
                }
            }
        }
        Self::default()
    }

    /// Save cache to disk
    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::cache_path())
    }

    /// Save cache to a specific path
    pub fn save_to(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Get cached response
    pub fn get(&self, name: &str) -> Option<&CrateResponse> {
        self.entries.get(name).and_then(|entry| {
            if !entry.is_expired() {
                Some(&entry.response)
            } else {
                None
            }
        })
    }

    /// Insert response into cache
    pub fn insert(&mut self, name: String, response: CrateResponse, ttl: Duration) {
        self.entries.insert(name, PersistentCacheEntry::new(response, ttl));
    }

    /// Clear expired entries
    pub fn clear_expired(&mut self) {
        self.entries.retain(|_, entry| !entry.is_expired());
    }
}
