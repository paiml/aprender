//! Cache implementation for publish status.
//!
//! Provides persistent caching with content-addressable keys for O(1) lookups.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::CacheEntry;

// ============================================================================
// PUB-002: Cache Implementation
// ============================================================================

/// Persistent cache for publish status
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PublishStatusCache {
    /// Cache entries by crate name
    pub(crate) entries: HashMap<String, CacheEntry>,
    /// Cache file path
    #[serde(skip)]
    pub(crate) cache_path: Option<PathBuf>,
}

impl PublishStatusCache {
    /// Default cache path
    pub(crate) fn default_cache_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("batuta")
            .join("publish-status.json")
    }

    /// Load cache from disk
    #[must_use]
    pub fn load() -> Self {
        let path = Self::default_cache_path();
        Self::load_from(&path).unwrap_or_default()
    }

    /// Load cache from specific path
    pub fn load_from(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(path)?;
        let mut cache: Self = serde_json::from_str(&data)?;
        cache.cache_path = Some(path.to_path_buf());
        Ok(cache)
    }

    /// Save cache to disk
    pub fn save(&self) -> Result<()> {
        let path = self.cache_path.clone().unwrap_or_else(Self::default_cache_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, data)?;
        Ok(())
    }

    /// Get cached entry if valid
    #[must_use]
    pub fn get(&self, name: &str, cache_key: &str) -> Option<&CacheEntry> {
        self.entries.get(name).filter(|e| e.cache_key == cache_key)
    }

    /// Insert or update entry
    pub fn insert(&mut self, name: String, entry: CacheEntry) {
        self.entries.insert(name, entry);
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ============================================================================
// PUB-003: Cache Key Computation
// ============================================================================

/// Compute cache key for a repo
/// Key = hash(Cargo.toml content || git HEAD SHA || Cargo.toml mtime)
pub fn compute_cache_key(repo_path: &Path) -> Result<String> {
    let cargo_toml = repo_path.join("Cargo.toml");

    if !cargo_toml.exists() {
        return Err(anyhow!("No Cargo.toml found at {:?}", repo_path));
    }

    // Read Cargo.toml content
    let content = std::fs::read(&cargo_toml)?;

    // Get mtime
    let mtime = std::fs::metadata(&cargo_toml)?
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Get git HEAD (if available)
    let head_sha = get_git_head(repo_path).unwrap_or_else(|_| "no-git".to_string());

    // Compute hash using DefaultHasher (fast, good enough for cache keys)
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    head_sha.hash(&mut hasher);
    mtime.hash(&mut hasher);

    Ok(format!("{:016x}", hasher.finish()))
}

/// Get git HEAD commit SHA
pub(crate) fn get_git_head(repo_path: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(anyhow!("git rev-parse failed"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
