//! Memory Paging for Model Bundles
//!
//! Implements LRU-based paging for loading models larger than available RAM.

use super::format::BundleReader;
use super::manifest::{BundleManifest, ModelEntry};
use super::mmap::PageTable;
use super::{DEFAULT_MAX_MEMORY, DEFAULT_PAGE_SIZE};
use crate::error::{AprenderError, Result};
use std::collections::{HashMap, VecDeque};
use std::path::Path;

// ============================================================================
// Paging Configuration
// ============================================================================

/// Configuration for paged model loading.
#[derive(Debug, Clone)]
pub struct PagingConfig {
    /// Maximum memory to use for cached model data (bytes).
    pub max_memory: usize,
    /// Page size for loading (bytes).
    pub page_size: usize,
    /// Enable pre-fetching of likely-needed pages.
    pub prefetch: bool,
    /// Number of pages to pre-fetch.
    pub prefetch_count: usize,
    /// Eviction strategy.
    pub eviction: EvictionStrategy,
}

/// Strategy for evicting pages when memory is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum EvictionStrategy {
    /// Least Recently Used - evict oldest accessed page.
    #[default]
    LRU,
    /// Least Frequently Used - evict least accessed page.
    LFU,
}

impl Default for PagingConfig {
    fn default() -> Self {
        Self {
            max_memory: DEFAULT_MAX_MEMORY,
            page_size: DEFAULT_PAGE_SIZE,
            prefetch: true,
            prefetch_count: 2,
            eviction: EvictionStrategy::default(),
        }
    }
}

impl PagingConfig {
    /// Create a new paging configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum memory.
    ///
    /// Note: The minimum is 1024 bytes to ensure meaningful paging behavior.
    #[must_use]
    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.max_memory = max_memory.max(1024);
        self
    }

    /// Set page size.
    #[must_use]
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size.max(512);
        self
    }

    /// Enable or disable pre-fetching.
    #[must_use]
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// Set pre-fetch count.
    #[must_use]
    pub fn with_prefetch_count(mut self, count: usize) -> Self {
        self.prefetch_count = count;
        self
    }

    /// Set eviction strategy.
    #[must_use]
    pub fn with_eviction(mut self, strategy: EvictionStrategy) -> Self {
        self.eviction = strategy;
        self
    }
}

// ============================================================================
// Paging Statistics
// ============================================================================

/// Statistics for paged bundle access.
#[derive(Debug, Clone, Default)]
pub struct PagingStats {
    /// Number of page hits (data already in memory).
    pub hits: usize,
    /// Number of page misses (data loaded from disk).
    pub misses: usize,
    /// Number of page evictions.
    pub evictions: usize,
    /// Total bytes loaded.
    pub bytes_loaded: usize,
    /// Current memory usage.
    pub memory_used: usize,
    /// Number of pre-fetches.
    pub prefetches: usize,
}

impl PagingStats {
    /// Calculate hit rate.
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ============================================================================
// Paged Bundle
// ============================================================================

/// A model bundle with memory paging support.
///
/// Enables loading models larger than available RAM by dynamically
/// loading and evicting model data as needed.
pub struct PagedBundle {
    /// Bundle reader.
    reader: BundleReader,
    /// Bundle manifest.
    manifest: BundleManifest,
    /// Cached model data.
    cache: HashMap<String, Vec<u8>>,
    /// LRU order for eviction.
    lru_order: VecDeque<String>,
    /// Page table for tracking.
    page_table: PageTable,
    /// Paging configuration.
    config: PagingConfig,
    /// Paging statistics.
    stats: PagingStats,
    /// Access history for pre-fetching.
    access_history: VecDeque<String>,
}

impl PagedBundle {
    /// Open a bundle with paging enabled.
    pub fn open(path: impl AsRef<Path>, config: PagingConfig) -> Result<Self> {
        let mut reader = BundleReader::open(path)?;
        let manifest = reader.read_manifest()?;

        Ok(Self {
            reader,
            manifest,
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            page_table: PageTable::new(),
            config,
            stats: PagingStats::default(),
            access_history: VecDeque::with_capacity(10),
        })
    }

    /// Get a model's data, loading from disk if needed.
    pub fn get_model(&mut self, name: &str) -> Result<&[u8]> {
        // Check cache first
        if self.cache.contains_key(name) {
            self.stats.hits += 1;
            self.update_lru(name);
            self.record_access(name);

            // Pre-fetch if enabled
            if self.config.prefetch {
                self.try_prefetch();
            }

            return Ok(self.cache.get(name).expect("Key should exist"));
        }

        // Cache miss - load from disk
        self.stats.misses += 1;
        self.load_model(name)?;

        // Record access
        self.record_access(name);

        // Pre-fetch if enabled
        if self.config.prefetch {
            self.try_prefetch();
        }

        Ok(self.cache.get(name).expect("Just loaded"))
    }

    /// Check if a model is currently in cache.
    #[must_use]
    pub fn is_cached(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }

    /// Get all model names.
    #[must_use]
    pub fn model_names(&self) -> Vec<&str> {
        self.manifest.model_names()
    }

    /// Get model metadata.
    #[must_use]
    pub fn get_metadata(&self, name: &str) -> Option<&ModelEntry> {
        self.manifest.get_model(name)
    }

    /// Get paging statistics.
    #[must_use]
    pub fn stats(&self) -> &PagingStats {
        &self.stats
    }

    /// Get paging configuration.
    #[must_use]
    pub fn config(&self) -> &PagingConfig {
        &self.config
    }

    /// Get current memory usage.
    #[must_use]
    pub fn memory_used(&self) -> usize {
        self.stats.memory_used
    }

    /// Get number of cached models.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Explicitly evict a model from cache.
    pub fn evict(&mut self, name: &str) -> bool {
        if let Some(data) = self.cache.remove(name) {
            self.stats.memory_used = self.stats.memory_used.saturating_sub(data.len());
            self.stats.evictions += 1;
            self.lru_order.retain(|n| n != name);
            true
        } else {
            false
        }
    }

    /// Clear all cached data.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.stats.memory_used = 0;
    }

    /// Hint that a model will be needed soon.
    pub fn prefetch_hint(&mut self, name: &str) -> Result<()> {
        if !self.cache.contains_key(name) && self.manifest.get_model(name).is_some() {
            self.load_model(name)?;
            self.stats.prefetches += 1;
        }
        Ok(())
    }

    /// Load a model into cache.
    fn load_model(&mut self, name: &str) -> Result<()> {
        let entry = self
            .manifest
            .get_model(name)
            .ok_or_else(|| AprenderError::Other(format!("Model '{name}' not found")))?
            .clone();

        // Evict if necessary
        while self.stats.memory_used + entry.size > self.config.max_memory {
            if !self.evict_lru() {
                // Can't evict anything, but try to load anyway
                break;
            }
        }

        // Load the data
        let data = self.reader.read_model(&entry)?;
        let size = data.len();

        // Update stats
        self.stats.bytes_loaded += size;
        self.stats.memory_used += size;

        // Add to cache
        self.cache.insert(name.to_string(), data);
        self.lru_order.push_back(name.to_string());

        // Update page table
        self.page_table.add_page(entry.offset, size);

        Ok(())
    }

    /// Update LRU order for a model.
    fn update_lru(&mut self, name: &str) {
        self.lru_order.retain(|n| n != name);
        self.lru_order.push_back(name.to_string());

        // Update page table timestamp
        if let Some(entry) = self.manifest.get_model(name) {
            self.page_table.touch(entry.offset);
        }
    }

    /// Evict the least recently used model.
    fn evict_lru(&mut self) -> bool {
        let to_evict = match self.config.eviction {
            EvictionStrategy::LRU => self.lru_order.pop_front(),
            EvictionStrategy::LFU => {
                if let Some(offset) = self.page_table.lfu_page() {
                    // Find model with this offset
                    self.manifest
                        .iter()
                        .find(|e| e.offset == offset)
                        .map(|e| e.name.clone())
                } else {
                    self.lru_order.pop_front()
                }
            }
        };

        if let Some(name) = to_evict {
            if let Some(data) = self.cache.remove(&name) {
                self.stats.memory_used = self.stats.memory_used.saturating_sub(data.len());
                self.stats.evictions += 1;

                // Remove from page table
                if let Some(entry) = self.manifest.get_model(&name) {
                    self.page_table.remove(entry.offset);
                }

                return true;
            }
        }

        false
    }

    /// Record an access for prediction.
    fn record_access(&mut self, name: &str) {
        if self.access_history.len() >= 10 {
            self.access_history.pop_front();
        }
        self.access_history.push_back(name.to_string());
    }

    /// Try to pre-fetch likely-needed models.
    fn try_prefetch(&mut self) {
        if self.access_history.len() < 2 {
            return;
        }

        // Simple pattern: if A -> B happened before, pre-fetch B after A
        let last = self.access_history.back().cloned();
        if let Some(last_name) = last {
            // Look for patterns in history
            let patterns: Vec<_> = self
                .access_history
                .iter()
                .zip(self.access_history.iter().skip(1))
                .filter(|(prev, _)| *prev == &last_name)
                .map(|(_, next)| next.clone())
                .take(self.config.prefetch_count)
                .collect();

            for name in patterns {
                if !self.cache.contains_key(&name)
                    && self.stats.memory_used + self.estimate_size(&name) <= self.config.max_memory
                {
                    let _ = self.load_model(&name);
                    self.stats.prefetches += 1;
                }
            }
        }
    }

    /// Estimate size of a model.
    fn estimate_size(&self, name: &str) -> usize {
        self.manifest.get_model(name).map_or(0, |e| e.size)
    }
}

impl std::fmt::Debug for PagedBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedBundle")
            .field("models", &self.manifest.len())
            .field("cached", &self.cache.len())
            .field("memory_used", &self.stats.memory_used)
            .field("max_memory", &self.config.max_memory)
            .field("hit_rate", &self.stats.hit_rate())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{BundleManifest, BundleWriter, ModelEntry};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    fn create_test_bundle(models: &[(&str, Vec<u8>)]) -> NamedTempFile {
        let temp = NamedTempFile::new().expect("Failed to create temp file");

        let mut manifest = BundleManifest::new();
        let mut model_map = HashMap::new();

        for (name, data) in models {
            manifest.add_model(ModelEntry::new(*name, data.len()));
            model_map.insert((*name).to_string(), data.clone());
        }

        let writer = BundleWriter::create(temp.path()).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &model_map)
            .expect("Failed to write bundle");

        temp
    }

    #[test]
    fn test_paging_config_default() {
        let config = PagingConfig::default();
        assert_eq!(config.max_memory, DEFAULT_MAX_MEMORY);
        assert_eq!(config.page_size, DEFAULT_PAGE_SIZE);
        assert!(config.prefetch);
        assert_eq!(config.eviction, EvictionStrategy::LRU);
    }

    #[test]
    fn test_paging_config_builder() {
        let config = PagingConfig::new()
            .with_max_memory(50_000)
            .with_page_size(8192)
            .with_prefetch(false)
            .with_eviction(EvictionStrategy::LFU);

        assert_eq!(config.max_memory, 50_000);
        assert_eq!(config.page_size, 8192);
        assert!(!config.prefetch);
        assert_eq!(config.eviction, EvictionStrategy::LFU);
    }

    #[test]
    fn test_paging_stats() {
        let mut stats = PagingStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.hits = 3;
        stats.misses = 1;
        assert!((stats.hit_rate() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_paged_bundle_open() {
        let temp = create_test_bundle(&[("model1", vec![1, 2, 3]), ("model2", vec![4, 5, 6, 7])]);

        let bundle =
            PagedBundle::open(temp.path(), PagingConfig::default()).expect("Failed to open bundle");

        assert_eq!(bundle.model_names().len(), 2);
        assert_eq!(bundle.cached_count(), 0);
        assert_eq!(bundle.memory_used(), 0);
    }

    #[test]
    fn test_paged_bundle_get_model() {
        let temp = create_test_bundle(&[("weights", vec![10, 20, 30, 40, 50])]);

        let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
            .expect("Failed to open");

        let data = bundle.get_model("weights").expect("Failed to get model");
        assert_eq!(data, &[10, 20, 30, 40, 50]);
        assert_eq!(bundle.cached_count(), 1);
        assert_eq!(bundle.stats().misses, 1);
        assert_eq!(bundle.stats().hits, 0);

        // Second access should be a hit
        let _ = bundle.get_model("weights").expect("Failed to get model");
        assert_eq!(bundle.stats().hits, 1);
    }

    #[test]
    fn test_paged_bundle_eviction() {
        // Use 1000-byte models with 1500-byte max memory
        // This forces eviction after the first model since 2000 > 1500
        let temp = create_test_bundle(&[
            ("model1", vec![1; 1000]),
            ("model2", vec![2; 1000]),
            ("model3", vec![3; 1000]),
        ]);

        // Small max memory to force eviction (1500 = 1.5 models worth)
        let mut bundle = PagedBundle::open(
            temp.path(),
            PagingConfig::new()
                .with_max_memory(1500)
                .with_prefetch(false),
        )
        .expect("Failed to open");

        // Load first model - fits in memory
        let _ = bundle.get_model("model1").expect("Failed");
        assert_eq!(bundle.cached_count(), 1);
        assert_eq!(bundle.memory_used(), 1000);

        // Load second model - should trigger eviction of model1
        // 1000 + 1000 = 2000 > 1500, must evict
        let _ = bundle.get_model("model2").expect("Failed");
        assert!(
            bundle.stats().evictions > 0,
            "Expected evictions > 0, got {}",
            bundle.stats().evictions
        );
        assert!(bundle.memory_used() <= 1500);

        // Load third model - should trigger another eviction
        let _ = bundle.get_model("model3").expect("Failed");
        assert!(bundle.stats().evictions >= 2);
        assert!(bundle.memory_used() <= 1500);
    }

    #[test]
    fn test_paged_bundle_explicit_evict() {
        let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

        let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
            .expect("Failed to open");

        // Load model
        let _ = bundle.get_model("model1").expect("Failed");
        assert!(bundle.is_cached("model1"));

        // Explicitly evict
        let evicted = bundle.evict("model1");
        assert!(evicted);
        assert!(!bundle.is_cached("model1"));
    }

    #[test]
    fn test_paged_bundle_clear_cache() {
        let temp = create_test_bundle(&[("model1", vec![1, 2, 3]), ("model2", vec![4, 5, 6])]);

        let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
            .expect("Failed to open");

        let _ = bundle.get_model("model1").expect("Failed");
        let _ = bundle.get_model("model2").expect("Failed");
        assert_eq!(bundle.cached_count(), 2);

        bundle.clear_cache();
        assert_eq!(bundle.cached_count(), 0);
        assert_eq!(bundle.memory_used(), 0);
    }

    #[test]
    fn test_paged_bundle_prefetch_hint() {
        let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

        let mut bundle =
            PagedBundle::open(temp.path(), PagingConfig::new()).expect("Failed to open");

        // Pre-fetch
        bundle.prefetch_hint("model1").expect("Prefetch failed");
        assert!(bundle.is_cached("model1"));

        // Access should now be a hit
        let _ = bundle.get_model("model1").expect("Failed");
        assert_eq!(bundle.stats().hits, 1);
        assert_eq!(bundle.stats().misses, 0);
    }

    #[test]
    fn test_paged_bundle_nonexistent_model() {
        let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

        let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
            .expect("Failed to open");

        let result = bundle.get_model("nonexistent");
        assert!(result.is_err());
    }
}
