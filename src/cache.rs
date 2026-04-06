//! Cache Management and Download Progress
//!
//! Manages the local model cache with cleanup, statistics, and download progress.
//!
//! ## Features
//!
//! - Cache size tracking and limits
//! - LRU eviction for space management
//! - Download progress callbacks
//! - Cache statistics and health
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::cache::{CacheManager, CacheConfig};
//!
//! let config = CacheConfig::new()
//!     .with_max_size_gb(50.0)
//!     .with_auto_cleanup(true);
//!
//! let cache = CacheManager::new(config, registry);
//!
//! // Clean up old models
//! let freed = cache.cleanup()?;
//! println!("Freed {} bytes", freed);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

// ============================================================================
// CACHE-001: Configuration
// ============================================================================

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes (0 = unlimited)
    pub max_size_bytes: u64,
    /// Minimum free space to maintain (in bytes)
    pub min_free_space: u64,
    /// Maximum age for unused models (in seconds, 0 = unlimited)
    pub max_age_secs: u64,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Cleanup threshold (trigger when usage exceeds this percentage)
    pub cleanup_threshold: f64,
    /// Target usage after cleanup (percentage)
    pub cleanup_target: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 50 * 1024 * 1024 * 1024, // 50 GB
            min_free_space: 5 * 1024 * 1024 * 1024,  // 5 GB
            max_age_secs: 30 * 24 * 60 * 60,         // 30 days
            auto_cleanup: true,
            cleanup_threshold: 0.90, // 90%
            cleanup_target: 0.70,    // 70%
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum cache size in GB
    #[must_use]
    pub fn with_max_size_gb(mut self, gb: f64) -> Self {
        self.max_size_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
        self
    }

    /// Set maximum cache size in bytes
    #[must_use]
    pub fn with_max_size_bytes(mut self, bytes: u64) -> Self {
        self.max_size_bytes = bytes;
        self
    }

    /// Set minimum free space in GB
    #[must_use]
    pub fn with_min_free_space_gb(mut self, gb: f64) -> Self {
        self.min_free_space = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
        self
    }

    /// Set maximum age for unused models in days
    #[must_use]
    pub fn with_max_age_days(mut self, days: u64) -> Self {
        self.max_age_secs = days * 24 * 60 * 60;
        self
    }

    /// Enable/disable auto cleanup
    #[must_use]
    pub fn with_auto_cleanup(mut self, enabled: bool) -> Self {
        self.auto_cleanup = enabled;
        self
    }

    /// Set cleanup threshold (percentage)
    #[must_use]
    pub fn with_cleanup_threshold(mut self, threshold: f64) -> Self {
        self.cleanup_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set cleanup target (percentage)
    #[must_use]
    pub fn with_cleanup_target(mut self, target: f64) -> Self {
        self.cleanup_target = target.clamp(0.0, 1.0);
        self
    }

    /// Get max size in GB
    #[must_use]
    pub fn max_size_gb(&self) -> f64 {
        self.max_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

// ============================================================================
// CACHE-002: Cache Entry
// ============================================================================

/// A cached model entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Content hash
    pub hash: String,
    /// File path in cache
    pub path: PathBuf,
    /// Whether this entry is pinned (won't be evicted)
    pub pinned: bool,
}

impl CacheEntry {
    /// Create a new cache entry
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        size_bytes: u64,
        hash: impl Into<String>,
        path: PathBuf,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            name: name.into(),
            version: version.into(),
            size_bytes,
            last_accessed: now,
            created_at: now,
            access_count: 0,
            hash: hash.into(),
            path,
            pinned: false,
        }
    }

    /// Mark as accessed
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }

    /// Pin this entry (prevent eviction)
    pub fn pin(&mut self) {
        self.pinned = true;
    }

    /// Unpin this entry
    pub fn unpin(&mut self) {
        self.pinned = false;
    }

    /// Get age since last access
    #[must_use]
    pub fn age(&self) -> Duration {
        SystemTime::now().duration_since(self.last_accessed).unwrap_or(Duration::ZERO)
    }

    /// Get size in GB
    #[must_use]
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get unique key
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}:{}", self.name, self.version)
    }
}

// ============================================================================
// CACHE-003: Cache Statistics
// ============================================================================

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache size in bytes
    pub total_size_bytes: u64,
    /// Number of cached models
    pub model_count: usize,
    /// Maximum configured size
    pub max_size_bytes: u64,
    /// Usage percentage
    pub usage_percent: f64,
    /// Number of pinned models
    pub pinned_count: usize,
    /// Total pinned size
    pub pinned_size_bytes: u64,
    /// Oldest entry age (seconds)
    pub oldest_age_secs: u64,
    /// Most accessed model
    pub most_accessed: Option<String>,
    /// Hit rate (if tracking enabled)
    pub hit_rate: Option<f64>,
}

impl CacheStats {
    /// Get total size in GB
    #[must_use]
    pub fn total_size_gb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get max size in GB
    #[must_use]
    pub fn max_size_gb(&self) -> f64 {
        self.max_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get available space in bytes
    #[must_use]
    pub fn available_bytes(&self) -> u64 {
        self.max_size_bytes.saturating_sub(self.total_size_bytes)
    }

    /// Get available space in GB
    #[must_use]
    pub fn available_gb(&self) -> f64 {
        self.available_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

// ============================================================================
// CACHE-004: Download Progress
// ============================================================================

/// Download progress information
#[derive(Debug, Clone, Copy)]
pub struct DownloadProgress {
    /// Total bytes to download
    pub total_bytes: u64,
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Download speed (bytes per second)
    pub speed_bps: f64,
    /// Estimated time remaining (seconds)
    pub eta_secs: f64,
    /// Whether download is complete
    pub is_complete: bool,
    /// Start time
    pub started_at: Instant,
}

impl DownloadProgress {
    /// Create new progress tracker
    #[must_use]
    pub fn new(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            downloaded_bytes: 0,
            speed_bps: 0.0,
            eta_secs: 0.0,
            is_complete: false,
            started_at: Instant::now(),
        }
    }

    /// Update progress
    pub fn update(&mut self, downloaded_bytes: u64) {
        self.downloaded_bytes = downloaded_bytes;
        let elapsed = self.started_at.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            self.speed_bps = downloaded_bytes as f64 / elapsed;
        }

        if self.speed_bps > 0.0 {
            let remaining = self.total_bytes.saturating_sub(downloaded_bytes);
            self.eta_secs = remaining as f64 / self.speed_bps;
        }

        self.is_complete = downloaded_bytes >= self.total_bytes;
    }

    /// Get completion percentage
    #[must_use]
    pub fn percent(&self) -> f64 {
        if self.total_bytes == 0 {
            100.0
        } else {
            (self.downloaded_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }

    /// Get human-readable speed
    #[must_use]
    pub fn speed_human(&self) -> String {
        format_bytes_per_sec(self.speed_bps)
    }

    /// Get human-readable ETA
    #[must_use]
    pub fn eta_human(&self) -> String {
        format_duration(Duration::from_secs_f64(self.eta_secs))
    }

    /// Get human-readable downloaded size
    #[must_use]
    pub fn downloaded_human(&self) -> String {
        format_bytes(self.downloaded_bytes)
    }

    /// Get human-readable total size
    #[must_use]
    pub fn total_human(&self) -> String {
        format_bytes(self.total_bytes)
    }

    /// Format progress bar (width characters)
    #[must_use]
    pub fn progress_bar(&self, width: usize) -> String {
        let filled = (self.percent() / 100.0 * width as f64) as usize;
        let empty = width.saturating_sub(filled);

        format!("[{}{}] {:5.1}%", "█".repeat(filled), "░".repeat(empty), self.percent())
    }
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(&DownloadProgress) + Send + Sync>;

// ============================================================================
// CACHE-005: Eviction Policy
// ============================================================================

/// Eviction policy for cache cleanup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    #[default]
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Largest First
    LargestFirst,
    /// Oldest First (by creation time)
    OldestFirst,
}

impl EvictionPolicy {
    /// Sort entries by eviction priority (lowest priority first)
    pub fn sort_for_eviction<'a>(&self, entries: &mut [&'a CacheEntry]) {
        match self {
            Self::LRU => entries.sort_by(|a, b| a.last_accessed.cmp(&b.last_accessed)),
            Self::LFU => entries.sort_by(|a, b| a.access_count.cmp(&b.access_count)),
            Self::FIFO => entries.sort_by(|a, b| a.created_at.cmp(&b.created_at)),
            Self::LargestFirst => entries.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes)),
            Self::OldestFirst => entries.sort_by(|a, b| a.created_at.cmp(&b.created_at)),
        }
    }
}

// ============================================================================
// CACHE-006: Cache Manager
// ============================================================================

/// Cache manager for model storage
#[derive(Debug)]
pub struct CacheManager {
    /// Configuration
    config: CacheConfig,
    /// Cached entries
    entries: HashMap<String, CacheEntry>,
    /// Eviction policy
    policy: EvictionPolicy,
    /// Cache hits counter
    cache_hits: u64,
    /// Cache misses counter
    cache_misses: u64,
}

impl CacheManager {
    /// Create a new cache manager
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            policy: EvictionPolicy::LRU,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Set eviction policy
    #[must_use]
    pub fn with_policy(mut self, policy: EvictionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Add an entry to the cache
    pub fn add(&mut self, entry: CacheEntry) {
        // Check if we need to make space
        if self.config.auto_cleanup && self.needs_cleanup() {
            let _ = self.cleanup_to_target();
        }

        self.entries.insert(entry.key(), entry);
    }

    /// Get an entry from the cache
    pub fn get(&mut self, name: &str, version: &str) -> Option<&CacheEntry> {
        let key = format!("{name}:{version}");
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.touch();
            self.cache_hits += 1;
            Some(entry)
        } else {
            self.cache_misses += 1;
            None
        }
    }

    /// Check if an entry exists
    #[must_use]
    pub fn contains(&self, name: &str, version: &str) -> bool {
        let key = format!("{name}:{version}");
        self.entries.contains_key(&key)
    }

    /// Remove an entry
    pub fn remove(&mut self, name: &str, version: &str) -> Option<CacheEntry> {
        let key = format!("{name}:{version}");
        self.entries.remove(&key)
    }

    /// Pin an entry (prevent eviction)
    pub fn pin(&mut self, name: &str, version: &str) -> bool {
        let key = format!("{name}:{version}");
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pin();
            true
        } else {
            false
        }
    }

    /// Unpin an entry
    pub fn unpin(&mut self, name: &str, version: &str) -> bool {
        let key = format!("{name}:{version}");
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.unpin();
            true
        } else {
            false
        }
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let total_size_bytes: u64 = self.entries.values().map(|e| e.size_bytes).sum();
        let pinned_entries: Vec<_> = self.entries.values().filter(|e| e.pinned).collect();
        let pinned_size_bytes: u64 = pinned_entries.iter().map(|e| e.size_bytes).sum();

        let oldest_age = self.entries.values().map(|e| e.age().as_secs()).max().unwrap_or(0);

        let most_accessed = self.entries.values().max_by_key(|e| e.access_count).map(|e| e.key());

        let usage_percent = if self.config.max_size_bytes > 0 {
            total_size_bytes as f64 / self.config.max_size_bytes as f64
        } else {
            0.0
        };

        let total_requests = self.cache_hits + self.cache_misses;
        let hit_rate = if total_requests > 0 {
            Some(self.cache_hits as f64 / total_requests as f64)
        } else {
            None
        };

        CacheStats {
            total_size_bytes,
            model_count: self.entries.len(),
            max_size_bytes: self.config.max_size_bytes,
            usage_percent,
            pinned_count: pinned_entries.len(),
            pinned_size_bytes,
            oldest_age_secs: oldest_age,
            most_accessed,
            hit_rate,
        }
    }

    /// Check if cleanup is needed
    #[must_use]
    pub fn needs_cleanup(&self) -> bool {
        let stats = self.stats();
        stats.usage_percent >= self.config.cleanup_threshold
    }

    /// Cleanup old/unused entries to reach target
    ///
    /// Returns bytes freed
    pub fn cleanup_to_target(&mut self) -> u64 {
        let target_bytes = (self.config.max_size_bytes as f64 * self.config.cleanup_target) as u64;
        self.cleanup_to_size(target_bytes)
    }

    /// Cleanup to reach a specific size
    ///
    /// Returns bytes freed
    pub fn cleanup_to_size(&mut self, target_bytes: u64) -> u64 {
        let mut current_size: u64 = self.entries.values().map(|e| e.size_bytes).sum();

        if current_size <= target_bytes {
            return 0;
        }

        // Get eviction candidates (non-pinned entries)
        let mut candidates: Vec<&CacheEntry> =
            self.entries.values().filter(|e| !e.pinned).collect();

        // Sort by eviction priority
        self.policy.sort_for_eviction(&mut candidates);

        // Collect keys to remove
        let mut to_remove = Vec::new();
        let mut freed = 0u64;

        for entry in candidates {
            if current_size <= target_bytes {
                break;
            }
            to_remove.push(entry.key());
            current_size -= entry.size_bytes;
            freed += entry.size_bytes;
        }

        // Remove entries
        for key in to_remove {
            self.entries.remove(&key);
        }

        freed
    }

    /// Remove entries older than max age
    ///
    /// Returns bytes freed
    pub fn cleanup_old_entries(&mut self) -> u64 {
        if self.config.max_age_secs == 0 {
            return 0;
        }

        let max_age = Duration::from_secs(self.config.max_age_secs);
        let to_remove: Vec<String> = self
            .entries
            .values()
            .filter(|e| !e.pinned && e.age() > max_age)
            .map(|e| e.key())
            .collect();

        let mut freed = 0u64;
        for key in to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                freed += entry.size_bytes;
            }
        }

        freed
    }

    /// List all entries
    #[must_use]
    pub fn list(&self) -> Vec<&CacheEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));
        entries
    }

    /// Clear the entire cache
    ///
    /// Returns bytes freed
    pub fn clear(&mut self) -> u64 {
        let freed: u64 = self.entries.values().map(|e| e.size_bytes).sum();
        self.entries.clear();
        freed
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Format bytes as human-readable string
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format bytes per second as human-readable string
#[must_use]
pub fn format_bytes_per_sec(bps: f64) -> String {
    format!("{}/s", format_bytes(bps as u64))
}

/// Format duration as human-readable string
#[must_use]
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CACHE-001: Config Tests
    // ========================================================================

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_size_bytes, 50 * 1024 * 1024 * 1024);
        assert!(config.auto_cleanup);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new()
            .with_max_size_gb(100.0)
            .with_min_free_space_gb(10.0)
            .with_max_age_days(60)
            .with_auto_cleanup(false)
            .with_cleanup_threshold(0.80)
            .with_cleanup_target(0.50);

        assert!((config.max_size_gb() - 100.0).abs() < 0.1);
        assert!(!config.auto_cleanup);
        assert!((config.cleanup_threshold - 0.80).abs() < f64::EPSILON);
        assert!((config.cleanup_target - 0.50).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_config_clamp() {
        let config = CacheConfig::new().with_cleanup_threshold(1.5).with_cleanup_target(-0.5);

        assert!((config.cleanup_threshold - 1.0).abs() < f64::EPSILON);
        assert!((config.cleanup_target - 0.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // CACHE-002: Entry Tests
    // ========================================================================

    #[test]
    fn test_cache_entry_new() {
        let entry = CacheEntry::new(
            "llama3",
            "8b",
            4_000_000_000,
            "hash123",
            PathBuf::from("/cache/llama3"),
        );

        assert_eq!(entry.name, "llama3");
        assert_eq!(entry.version, "8b");
        assert_eq!(entry.size_bytes, 4_000_000_000);
        assert_eq!(entry.access_count, 0);
        assert!(!entry.pinned);
    }

    #[test]
    fn test_cache_entry_touch() {
        let mut entry = CacheEntry::new("test", "1.0", 1000, "hash", PathBuf::new());
        let initial_access = entry.last_accessed;

        std::thread::sleep(Duration::from_millis(10));
        entry.touch();

        assert!(entry.last_accessed > initial_access);
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_cache_entry_pin_unpin() {
        let mut entry = CacheEntry::new("test", "1.0", 1000, "hash", PathBuf::new());

        assert!(!entry.pinned);
        entry.pin();
        assert!(entry.pinned);
        entry.unpin();
        assert!(!entry.pinned);
    }

    #[test]
    fn test_cache_entry_size_gb() {
        let entry = CacheEntry::new("test", "1.0", 4 * 1024 * 1024 * 1024, "hash", PathBuf::new());
        assert!((entry.size_gb() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_entry_key() {
        let entry = CacheEntry::new("model", "v1.0", 1000, "hash", PathBuf::new());
        assert_eq!(entry.key(), "model:v1.0");
    }

    // ========================================================================
    // CACHE-003: Stats Tests
    // ========================================================================

    #[test]
    fn test_cache_stats_sizes() {
        let stats = CacheStats {
            total_size_bytes: 10 * 1024 * 1024 * 1024,
            model_count: 5,
            max_size_bytes: 50 * 1024 * 1024 * 1024,
            usage_percent: 0.20,
            pinned_count: 1,
            pinned_size_bytes: 2 * 1024 * 1024 * 1024,
            oldest_age_secs: 3600,
            most_accessed: Some("llama3:8b".to_string()),
            hit_rate: Some(0.95),
        };

        assert!((stats.total_size_gb() - 10.0).abs() < 0.1);
        assert!((stats.max_size_gb() - 50.0).abs() < 0.1);
        assert!((stats.available_gb() - 40.0).abs() < 0.1);
    }

    // ========================================================================
    // CACHE-004: Progress Tests
    // ========================================================================

    #[test]
    fn test_download_progress_new() {
        let progress = DownloadProgress::new(1000);
        assert_eq!(progress.total_bytes, 1000);
        assert_eq!(progress.downloaded_bytes, 0);
        assert!(!progress.is_complete);
    }

    #[test]
    fn test_download_progress_update() {
        let mut progress = DownloadProgress::new(1000);
        progress.update(500);

        assert_eq!(progress.downloaded_bytes, 500);
        assert!((progress.percent() - 50.0).abs() < 0.1);
        assert!(!progress.is_complete);
    }

    #[test]
    fn test_download_progress_complete() {
        let mut progress = DownloadProgress::new(1000);
        progress.update(1000);

        assert!(progress.is_complete);
        assert!((progress.percent() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_download_progress_bar() {
        let mut progress = DownloadProgress::new(100);
        progress.update(50);

        let bar = progress.progress_bar(20);
        assert!(bar.contains("50.0%"));
        assert!(bar.contains("█"));
        assert!(bar.contains("░"));
    }

    #[test]
    fn test_download_progress_zero_total() {
        let progress = DownloadProgress::new(0);
        assert!((progress.percent() - 100.0).abs() < 0.1);
    }

    // ========================================================================
    // CACHE-005: Eviction Tests
    // ========================================================================

    #[test]
    fn test_eviction_policy_lru() {
        let now = SystemTime::now();
        let old_time = now - Duration::from_secs(3600);

        let mut entry1 = CacheEntry::new("old", "1.0", 100, "h1", PathBuf::new());
        entry1.last_accessed = old_time;

        let entry2 = CacheEntry::new("new", "1.0", 100, "h2", PathBuf::new());

        let mut entries: Vec<&CacheEntry> = vec![&entry2, &entry1];
        EvictionPolicy::LRU.sort_for_eviction(&mut entries);

        // Oldest should be first (lowest priority)
        assert_eq!(entries[0].name, "old");
    }

    #[test]
    fn test_eviction_policy_lfu() {
        let mut entry1 = CacheEntry::new("popular", "1.0", 100, "h1", PathBuf::new());
        entry1.access_count = 100;

        let entry2 = CacheEntry::new("unpopular", "1.0", 100, "h2", PathBuf::new());

        let mut entries: Vec<&CacheEntry> = vec![&entry1, &entry2];
        EvictionPolicy::LFU.sort_for_eviction(&mut entries);

        // Least accessed should be first
        assert_eq!(entries[0].name, "unpopular");
    }

    #[test]
    fn test_eviction_policy_largest() {
        let entry1 = CacheEntry::new("small", "1.0", 100, "h1", PathBuf::new());
        let entry2 = CacheEntry::new("large", "1.0", 1000, "h2", PathBuf::new());

        let mut entries: Vec<&CacheEntry> = vec![&entry1, &entry2];
        EvictionPolicy::LargestFirst.sort_for_eviction(&mut entries);

        // Largest should be first
        assert_eq!(entries[0].name, "large");
    }

    // ========================================================================
    // CACHE-006: Manager Tests
    // ========================================================================

    #[test]
    fn test_cache_manager_new() {
        let config = CacheConfig::new().with_max_size_gb(10.0);
        let manager = CacheManager::new(config);

        assert_eq!(manager.stats().model_count, 0);
    }

    #[test]
    fn test_cache_manager_add_get() {
        let config = CacheConfig::new().with_auto_cleanup(false);
        let mut manager = CacheManager::new(config);

        let entry = CacheEntry::new("model", "1.0", 1000, "hash", PathBuf::new());
        manager.add(entry);

        assert!(manager.contains("model", "1.0"));
        assert!(manager.get("model", "1.0").is_some());
    }

    #[test]
    fn test_cache_manager_remove() {
        let config = CacheConfig::new().with_auto_cleanup(false);
        let mut manager = CacheManager::new(config);

        let entry = CacheEntry::new("model", "1.0", 1000, "hash", PathBuf::new());
        manager.add(entry);

        let removed = manager.remove("model", "1.0");
        assert!(removed.is_some());
        assert!(!manager.contains("model", "1.0"));
    }

    #[test]
    fn test_cache_manager_pin() {
        let config = CacheConfig::new().with_auto_cleanup(false);
        let mut manager = CacheManager::new(config);

        let entry = CacheEntry::new("model", "1.0", 1000, "hash", PathBuf::new());
        manager.add(entry);

        assert!(manager.pin("model", "1.0"));
        assert!(!manager.pin("nonexistent", "1.0"));

        let stats = manager.stats();
        assert_eq!(stats.pinned_count, 1);
    }

    #[test]
    fn test_cache_manager_cleanup() {
        let config = CacheConfig::new()
            .with_max_size_bytes(1000)
            .with_auto_cleanup(false)
            .with_cleanup_target(0.5);
        let mut manager = CacheManager::new(config);

        // Add entries totaling 800 bytes
        for i in 0..8 {
            let entry =
                CacheEntry::new(format!("model{i}"), "1.0", 100, format!("h{i}"), PathBuf::new());
            manager.add(entry);
        }

        let freed = manager.cleanup_to_target(); // Target is 500 bytes
        assert!(freed >= 300); // Should free at least 300 bytes
    }

    #[test]
    fn test_cache_manager_cleanup_respects_pinned() {
        let config = CacheConfig::new()
            .with_max_size_bytes(200)
            .with_auto_cleanup(false)
            .with_cleanup_target(0.5);
        let mut manager = CacheManager::new(config);

        // Add two entries
        let entry1 = CacheEntry::new("pinned", "1.0", 100, "h1", PathBuf::new());
        let entry2 = CacheEntry::new("unpinned", "1.0", 100, "h2", PathBuf::new());

        manager.add(entry1);
        manager.add(entry2);
        manager.pin("pinned", "1.0");

        manager.cleanup_to_target();

        // Pinned entry should remain
        assert!(manager.contains("pinned", "1.0"));
    }

    #[test]
    fn test_cache_manager_stats() {
        let config = CacheConfig::new().with_max_size_bytes(1000).with_auto_cleanup(false);
        let mut manager = CacheManager::new(config);

        let entry = CacheEntry::new("model", "1.0", 500, "hash", PathBuf::new());
        manager.add(entry);

        let stats = manager.stats();
        assert_eq!(stats.model_count, 1);
        assert_eq!(stats.total_size_bytes, 500);
        assert!((stats.usage_percent - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_manager_hit_rate() {
        let config = CacheConfig::new().with_auto_cleanup(false);
        let mut manager = CacheManager::new(config);

        let entry = CacheEntry::new("model", "1.0", 100, "hash", PathBuf::new());
        manager.add(entry);

        // 2 hits, 1 miss
        manager.get("model", "1.0");
        manager.get("model", "1.0");
        manager.get("nonexistent", "1.0");

        let stats = manager.stats();
        assert!((stats.hit_rate.unwrap() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_manager_clear() {
        let config = CacheConfig::new().with_auto_cleanup(false);
        let mut manager = CacheManager::new(config);

        manager.add(CacheEntry::new("m1", "1.0", 100, "h1", PathBuf::new()));
        manager.add(CacheEntry::new("m2", "1.0", 200, "h2", PathBuf::new()));

        let freed = manager.clear();
        assert_eq!(freed, 300);
        assert!(manager.stats().model_count == 0);
    }

    // ========================================================================
    // Helper Tests
    // ========================================================================

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 1024), "1.00 TB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3700)), "1h 1m");
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_cache_config_serialization() {
        let config = CacheConfig::new().with_max_size_gb(100.0);
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_size_bytes, config.max_size_bytes);
    }

    #[test]
    fn test_cache_entry_serialization() {
        let entry = CacheEntry::new("model", "1.0", 1000, "hash", PathBuf::from("/cache"));
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: CacheEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "model");
    }

    #[test]
    fn test_eviction_policy_serialization() {
        let policy = EvictionPolicy::LRU;
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: EvictionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, EvictionPolicy::LRU);
    }
}
