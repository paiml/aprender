//! Publish status scanner implementation.
//!
//! Scans workspace for PAIML crates and returns publish status with caching.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::cache::{compute_cache_key, PublishStatusCache};
use super::git::{determine_action, get_git_status, get_local_version};
use super::types::{CacheEntry, CrateStatus, GitStatus, PublishAction, PublishStatusReport};
use crate::stack::PAIML_CRATES;

// ============================================================================
// PUB-006: Scanner Implementation
// ============================================================================

/// Scan workspace for PAIML crates and return publish status
pub struct PublishStatusScanner {
    /// Workspace root (parent of crate directories)
    pub(crate) workspace_root: PathBuf,
    /// Cache
    pub(crate) cache: PublishStatusCache,
    /// crates.io client (for async fetches)
    #[cfg(feature = "native")]
    pub(crate) crates_io: Option<crate::stack::crates_io::CratesIoClient>,
}

impl PublishStatusScanner {
    /// Create scanner for workspace
    #[must_use]
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            cache: PublishStatusCache::load(),
            #[cfg(feature = "native")]
            crates_io: None,
        }
    }

    /// Initialize crates.io client
    #[cfg(feature = "native")]
    pub fn with_crates_io(mut self) -> Self {
        self.crates_io =
            Some(crate::stack::crates_io::CratesIoClient::new().with_persistent_cache());
        self
    }

    /// Find all PAIML crate directories in workspace
    #[must_use]
    pub fn find_crate_dirs(&self) -> Vec<(String, PathBuf)> {
        PAIML_CRATES
            .iter()
            .filter_map(|name| {
                let path = self.workspace_root.join(name);
                if path.join("Cargo.toml").exists() {
                    Some(((*name).to_string(), path))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check single crate status (with cache)
    pub fn check_crate(&mut self, name: &str, path: &Path) -> CrateStatus {
        // Compute cache key
        let cache_key = match compute_cache_key(path) {
            Ok(key) => key,
            Err(e) => {
                return CrateStatus {
                    name: name.to_string(),
                    local_version: None,
                    crates_io_version: None,
                    git_status: GitStatus::default(),
                    action: PublishAction::Error,
                    path: path.to_path_buf(),
                    error: Some(e.to_string()),
                };
            }
        };

        // Check cache
        if let Some(entry) = self.cache.get(name, &cache_key) {
            if !entry.is_crates_io_stale() {
                // Cache hit - O(1)
                return entry.status.clone();
            }
        }

        // Cache miss - need to refresh
        self.refresh_crate(name, path, &cache_key)
    }

    /// Refresh crate status (cache miss path)
    pub(crate) fn refresh_crate(
        &mut self,
        name: &str,
        path: &Path,
        cache_key: &str,
    ) -> CrateStatus {
        let local_version = get_local_version(path).ok();
        let git_status = get_git_status(path).unwrap_or_default();

        // crates.io version fetched separately (async)
        let crates_io_version = None; // Will be filled by async scan

        let action =
            determine_action(local_version.as_deref(), crates_io_version.as_deref(), &git_status);

        let status = CrateStatus {
            name: name.to_string(),
            local_version,
            crates_io_version,
            git_status,
            action,
            path: path.to_path_buf(),
            error: None,
        };

        // Update cache
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        self.cache.insert(
            name.to_string(),
            CacheEntry {
                cache_key: cache_key.to_string(),
                status: status.clone(),
                crates_io_checked_at: now,
                created_at: now,
            },
        );

        status
    }

    /// Scan all crates and return report
    #[cfg(feature = "native")]
    pub async fn scan(&mut self) -> Result<PublishStatusReport> {
        use std::time::Instant;

        let start = Instant::now();
        let crate_dirs = self.find_crate_dirs();
        let mut statuses = Vec::with_capacity(crate_dirs.len());
        let mut cache_hits = 0;

        // First pass: check cache and collect statuses
        for (name, path) in &crate_dirs {
            let cache_key = compute_cache_key(path).unwrap_or_default();

            if let Some(entry) = self.cache.get(name, &cache_key) {
                if !entry.is_crates_io_stale() {
                    cache_hits += 1;
                    statuses.push(entry.status.clone());
                    continue;
                }
            }

            // Need refresh - get local info first
            let mut status = self.refresh_crate(name, path, &cache_key);

            // Fetch crates.io version
            if let Some(ref mut client) = self.crates_io {
                if let Ok(response) = client.get_crate(name).await {
                    status.crates_io_version = Some(response.krate.max_version.clone());
                    status.action = determine_action(
                        status.local_version.as_deref(),
                        status.crates_io_version.as_deref(),
                        &status.git_status,
                    );

                    // Update cache with crates.io version
                    let now =
                        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

                    self.cache.insert(
                        name.clone(),
                        CacheEntry {
                            cache_key: cache_key.clone(),
                            status: status.clone(),
                            crates_io_checked_at: now,
                            created_at: now,
                        },
                    );
                }
            }

            statuses.push(status);
        }

        // Save cache
        let _ = self.cache.save();

        let elapsed_ms = start.elapsed().as_millis() as u64;
        Ok(PublishStatusReport::from_statuses(statuses, cache_hits, elapsed_ms))
    }

    /// Synchronous scan (for non-async contexts)
    #[cfg(feature = "native")]
    pub fn scan_sync(&mut self) -> Result<PublishStatusReport> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.scan())
    }
}
