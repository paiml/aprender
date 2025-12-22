//! Model Cache and Registry
//!
//! Hierarchical caching system implementing Toyota Way Just-In-Time principles:
//! - **Right amount**: Cache only what's needed for current inference
//! - **Right time**: Prefetch before access, evict after use
//! - **Right place**: L1 = hot, L2 = warm, L3 = cold storage
//!
//! # Cache Hierarchy
//!
//! ```text
//! L0: Trueno Tensor Cache (SIMD-aligned, in-register)
//! L1: Hot Model Cache (heap-allocated, aligned buffers)
//! L2: Warm Disk Cache (memory-mapped files)
//! L3: Cold Storage (filesystem or network)
//! ```
//!
//! # References
//!
//! - [Megiddo & Modha 2003] ARC: A Self-Tuning, Low Overhead Replacement Cache

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// Eviction policy selection (configurable per deployment)
///
/// # Policy Recommendations
///
/// - **LRU**: Sequential inference, time-series models
/// - **LFU**: Random access, sparse models
/// - **ARC**: Mixed workloads, production deployments
/// - **Clock**: Embedded systems with limited CPU
/// - **Fixed**: Deterministic embedded, NASA Level A
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Least Recently Used (temporal locality)
    /// Best for: Sequential inference, time-series models
    #[default]
    LRU,

    /// Least Frequently Used (frequency locality)
    /// Best for: Random access, sparse models
    LFU,

    /// Adaptive Replacement Cache [Megiddo & Modha 2003]
    /// Best for: Mixed workloads, production deployments
    ARC,

    /// Clock algorithm (efficient approximation of LRU)
    /// Best for: Embedded systems with limited CPU
    Clock,

    /// No eviction (fixed memory pool)
    /// Best for: Deterministic embedded, NASA Level A
    Fixed,
}

impl EvictionPolicy {
    /// Get human-readable description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::LRU => "Least Recently Used - evicts oldest accessed entry",
            Self::LFU => "Least Frequently Used - evicts least accessed entry",
            Self::ARC => "Adaptive Replacement Cache - balances recency and frequency",
            Self::Clock => "Clock algorithm - efficient LRU approximation",
            Self::Fixed => "Fixed - no eviction, deterministic memory",
        }
    }

    /// Check if this policy supports eviction
    #[must_use]
    pub const fn supports_eviction(&self) -> bool {
        !matches!(self, Self::Fixed)
    }

    /// Get recommended use case
    #[must_use]
    pub const fn recommended_use_case(&self) -> &'static str {
        match self {
            Self::LRU => "Sequential inference, time-series models",
            Self::LFU => "Random access, sparse models",
            Self::ARC => "Mixed workloads, production deployments",
            Self::Clock => "Embedded systems with limited CPU",
            Self::Fixed => "Deterministic embedded, NASA Level A",
        }
    }
}

/// Memory budget enforcement (Heijunka principle)
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum pages in memory
    pub max_pages: usize,
    /// High watermark (start eviction)
    pub high_watermark: usize,
    /// Low watermark (stop eviction)
    pub low_watermark: usize,
    /// Reserved pages (never evict)
    pub reserved_pages: HashSet<u64>,
}

impl MemoryBudget {
    /// Create a new memory budget
    #[must_use]
    pub fn new(max_pages: usize) -> Self {
        Self {
            max_pages,
            high_watermark: (max_pages as f64 * 0.9) as usize,
            low_watermark: (max_pages as f64 * 0.7) as usize,
            reserved_pages: HashSet::new(),
        }
    }

    /// Create with custom watermarks
    #[must_use]
    pub fn with_watermarks(max_pages: usize, high_pct: f64, low_pct: f64) -> Self {
        Self {
            max_pages,
            high_watermark: (max_pages as f64 * high_pct) as usize,
            low_watermark: (max_pages as f64 * low_pct) as usize,
            reserved_pages: HashSet::new(),
        }
    }

    /// Reserve a page (won't be evicted)
    pub fn reserve_page(&mut self, page_id: u64) {
        self.reserved_pages.insert(page_id);
    }

    /// Release a reserved page
    pub fn release_page(&mut self, page_id: u64) {
        self.reserved_pages.remove(&page_id);
    }

    /// Check if eviction is needed
    #[must_use]
    pub fn needs_eviction(&self, current_pages: usize) -> bool {
        current_pages >= self.high_watermark
    }

    /// Check if eviction can stop
    #[must_use]
    pub fn can_stop_eviction(&self, current_pages: usize) -> bool {
        current_pages <= self.low_watermark
    }

    /// Check if a page can be evicted
    #[must_use]
    pub fn can_evict(&self, page_id: u64) -> bool {
        !self.reserved_pages.contains(&page_id)
    }
}

/// Access statistics for cache entries
#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    /// Number of cache hits
    pub hit_count: u64,
    /// Number of cache misses
    pub miss_count: u64,
    /// Last access timestamp (monotonic)
    pub last_access: u64,
    /// Total access time in nanoseconds
    pub total_access_time_ns: u64,
    /// Number of prefetch hits
    pub prefetch_hits: u64,
}

impl AccessStats {
    /// Create new access statistics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cache hit
    pub fn record_hit(&mut self, access_time_ns: u64, timestamp: u64) {
        self.hit_count += 1;
        self.total_access_time_ns += access_time_ns;
        self.last_access = timestamp;
    }

    /// Record a cache miss
    pub fn record_miss(&mut self, timestamp: u64) {
        self.miss_count += 1;
        self.last_access = timestamp;
    }

    /// Record a prefetch hit
    pub fn record_prefetch_hit(&mut self) {
        self.prefetch_hits += 1;
    }

    /// Get hit rate (0.0 - 1.0)
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Get average access time in nanoseconds
    #[must_use]
    pub fn avg_access_time_ns(&self) -> f64 {
        if self.hit_count == 0 {
            0.0
        } else {
            self.total_access_time_ns as f64 / self.hit_count as f64
        }
    }

    /// Get prefetch effectiveness (0.0 - 1.0)
    #[must_use]
    pub fn prefetch_effectiveness(&self) -> f64 {
        if self.hit_count == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / self.hit_count as f64
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone)]
pub struct CacheMetadata {
    /// Original .apr file path (for invalidation)
    pub source_path: Option<PathBuf>,
    /// Source file modification time (staleness check)
    pub source_mtime: Option<SystemTime>,
    /// Cache entry creation time
    pub cached_at: SystemTime,
    /// Time-to-live (None = infinite)
    pub ttl: Option<Duration>,
    /// Entry size in bytes
    pub size_bytes: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

impl CacheMetadata {
    /// Create new cache metadata
    #[must_use]
    pub fn new(size_bytes: usize) -> Self {
        Self {
            source_path: None,
            source_mtime: None,
            cached_at: SystemTime::now(),
            ttl: None,
            size_bytes,
            compression_ratio: 1.0,
        }
    }

    /// Create with source path
    #[must_use]
    pub fn with_source(mut self, path: PathBuf, mtime: SystemTime) -> Self {
        self.source_path = Some(path);
        self.source_mtime = Some(mtime);
        self
    }

    /// Set TTL
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set compression ratio
    #[must_use]
    pub fn with_compression_ratio(mut self, ratio: f32) -> Self {
        self.compression_ratio = ratio;
        self
    }

    /// Check if entry is expired
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            if let Ok(elapsed) = self.cached_at.elapsed() {
                return elapsed > ttl;
            }
        }
        false
    }

    /// Check if entry is stale (source modified)
    #[must_use]
    pub fn is_stale(&self, current_mtime: SystemTime) -> bool {
        if let Some(cached_mtime) = self.source_mtime {
            return current_mtime > cached_mtime;
        }
        false
    }

    /// Get age of cache entry
    #[must_use]
    pub fn age(&self) -> Duration {
        self.cached_at.elapsed().unwrap_or(Duration::ZERO)
    }
}

/// Cached data variants
#[derive(Debug, Clone)]
pub enum CacheData {
    /// Compressed data (for L2/L3 tiers)
    Compressed(Vec<u8>),
    /// Decompressed data (for L1 tier)
    Decompressed(Vec<u8>),
    /// Memory-mapped region reference
    Mapped {
        /// File path
        path: PathBuf,
        /// Offset in file
        offset: u64,
        /// Length in bytes
        length: usize,
    },
}

impl CacheData {
    /// Get the data size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Compressed(data) | Self::Decompressed(data) => data.len(),
            Self::Mapped { length, .. } => *length,
        }
    }

    /// Check if data is compressed
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        matches!(self, Self::Compressed(_))
    }

    /// Check if data is memory-mapped
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
    }
}

/// Model type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelType(pub u16);

impl ModelType {
    /// Create a new model type
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }
}

/// Cache entry with comprehensive metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Unique model identifier (SHA-256 of header + metadata)
    pub model_hash: [u8; 32],
    /// Model type for type-safe deserialization
    pub model_type: ModelType,
    /// Cached data (compressed or decompressed based on tier)
    pub data: CacheData,
    /// Entry metadata
    pub metadata: CacheMetadata,
    /// Access statistics for eviction
    pub stats: AccessStats,
}

impl CacheEntry {
    /// Create a new cache entry
    #[must_use]
    pub fn new(model_hash: [u8; 32], model_type: ModelType, data: CacheData) -> Self {
        let size = data.size();
        Self {
            model_hash,
            model_type,
            data,
            metadata: CacheMetadata::new(size),
            stats: AccessStats::new(),
        }
    }

    /// Check if entry is valid (not expired or stale)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.metadata.is_expired()
    }

    /// Get the cache tier based on data type
    #[must_use]
    pub fn tier(&self) -> CacheTier {
        match &self.data {
            CacheData::Decompressed(_) => CacheTier::L1Hot,
            CacheData::Compressed(_) | CacheData::Mapped { .. } => CacheTier::L2Warm,
        }
    }
}

/// Cache tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    /// L1: Hot cache (heap-allocated, decompressed)
    L1Hot,
    /// L2: Warm cache (memory-mapped or compressed)
    L2Warm,
    /// L3: Cold storage (filesystem)
    L3Cold,
}

impl CacheTier {
    /// Get typical latency for this tier
    #[must_use]
    pub const fn typical_latency(&self) -> Duration {
        match self {
            Self::L1Hot => Duration::from_nanos(100),
            Self::L2Warm => Duration::from_micros(1000),
            Self::L3Cold => Duration::from_millis(10),
        }
    }

    /// Get tier name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::L1Hot => "L1 Hot Cache",
            Self::L2Warm => "L2 Warm Cache",
            Self::L3Cold => "L3 Cold Storage",
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum L1 cache size in bytes
    pub l1_max_bytes: usize,
    /// Maximum L2 cache size in bytes
    pub l2_max_bytes: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Default TTL for entries
    pub default_ttl: Option<Duration>,
    /// Enable prefetching
    pub prefetch_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_bytes: 64 * 1024 * 1024,   // 64 MB
            l2_max_bytes: 1024 * 1024 * 1024, // 1 GB
            eviction_policy: EvictionPolicy::LRU,
            default_ttl: None,
            prefetch_enabled: true,
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration for embedded deployment
    #[must_use]
    pub fn embedded(l1_bytes: usize) -> Self {
        Self {
            l1_max_bytes: l1_bytes,
            l2_max_bytes: 0, // No L2 on embedded
            eviction_policy: EvictionPolicy::Fixed,
            default_ttl: None,
            prefetch_enabled: false,
        }
    }

    /// Set L1 cache size
    #[must_use]
    pub fn with_l1_size(mut self, bytes: usize) -> Self {
        self.l1_max_bytes = bytes;
        self
    }

    /// Set L2 cache size
    #[must_use]
    pub fn with_l2_size(mut self, bytes: usize) -> Self {
        self.l2_max_bytes = bytes;
        self
    }

    /// Set eviction policy
    #[must_use]
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Set default TTL
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = Some(ttl);
        self
    }

    /// Enable or disable prefetching
    #[must_use]
    pub fn with_prefetch(mut self, enabled: bool) -> Self {
        self.prefetch_enabled = enabled;
        self
    }
}

/// Model information for registry
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether model is bundled (compile-time)
    pub is_bundled: bool,
    /// Whether model is currently cached
    pub is_cached: bool,
    /// Cache tier if cached
    pub cache_tier: Option<CacheTier>,
}

/// Model registry for bundled and dynamic models
#[derive(Debug)]
pub struct ModelRegistry {
    /// L1 cache entries
    l1_cache: HashMap<String, CacheEntry>,
    /// L2 cache entries
    l2_cache: HashMap<String, CacheEntry>,
    /// Cache configuration
    config: CacheConfig,
    /// Current L1 cache size
    l1_current_bytes: usize,
    /// Current L2 cache size
    l2_current_bytes: usize,
    /// Global access counter for LRU
    access_counter: u64,
    /// Cache creation time
    created_at: Instant,
}

impl ModelRegistry {
    /// Create a new model registry
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            l1_cache: HashMap::new(),
            l2_cache: HashMap::new(),
            config,
            l1_current_bytes: 0,
            l2_current_bytes: 0,
            access_counter: 0,
            created_at: Instant::now(),
        }
    }

    /// Insert a model into L1 cache
    pub fn insert_l1(&mut self, name: String, entry: CacheEntry) {
        let size = entry.data.size();

        // Evict if necessary
        while self.l1_current_bytes + size > self.config.l1_max_bytes && !self.l1_cache.is_empty() {
            if let Some(evict_key) = self.find_eviction_candidate_l1() {
                self.evict_l1(&evict_key);
            } else {
                break;
            }
        }

        if let Some(old) = self.l1_cache.insert(name, entry) {
            self.l1_current_bytes -= old.data.size();
        }
        self.l1_current_bytes += size;
    }

    /// Insert a model into L2 cache
    pub fn insert_l2(&mut self, name: String, entry: CacheEntry) {
        let size = entry.data.size();

        // Evict if necessary
        while self.l2_current_bytes + size > self.config.l2_max_bytes && !self.l2_cache.is_empty() {
            if let Some(evict_key) = self.find_eviction_candidate_l2() {
                self.evict_l2(&evict_key);
            } else {
                break;
            }
        }

        if let Some(old) = self.l2_cache.insert(name, entry) {
            self.l2_current_bytes -= old.data.size();
        }
        self.l2_current_bytes += size;
    }

    /// Get a model by name (checks L1, then L2)
    pub fn get(&mut self, name: &str) -> Option<&CacheEntry> {
        self.access_counter += 1;
        let timestamp = self.access_counter;

        // Check L1 first
        if let Some(entry) = self.l1_cache.get_mut(name) {
            entry.stats.record_hit(100, timestamp);
            return self.l1_cache.get(name);
        }

        // Check L2
        if let Some(entry) = self.l2_cache.get_mut(name) {
            entry.stats.record_hit(1000, timestamp);
            return self.l2_cache.get(name);
        }

        None
    }

    /// Check if a model exists in cache
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.l1_cache.contains_key(name) || self.l2_cache.contains_key(name)
    }

    /// Get cache tier for a model
    #[must_use]
    pub fn get_tier(&self, name: &str) -> Option<CacheTier> {
        if self.l1_cache.contains_key(name) {
            Some(CacheTier::L1Hot)
        } else if self.l2_cache.contains_key(name) {
            Some(CacheTier::L2Warm)
        } else {
            None
        }
    }

    /// Remove a model from cache
    pub fn remove(&mut self, name: &str) {
        if let Some(entry) = self.l1_cache.remove(name) {
            self.l1_current_bytes -= entry.data.size();
        }
        if let Some(entry) = self.l2_cache.remove(name) {
            self.l2_current_bytes -= entry.data.size();
        }
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.l1_cache.clear();
        self.l2_cache.clear();
        self.l1_current_bytes = 0;
        self.l2_current_bytes = 0;
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let l1_entries = self.l1_cache.len();
        let l2_entries = self.l2_cache.len();

        let (l1_hits, l1_misses) = self.l1_cache.values().fold((0, 0), |(h, m), e| {
            (h + e.stats.hit_count, m + e.stats.miss_count)
        });

        let (l2_hits, l2_misses) = self.l2_cache.values().fold((0, 0), |(h, m), e| {
            (h + e.stats.hit_count, m + e.stats.miss_count)
        });

        CacheStats {
            l1_entries,
            l1_bytes: self.l1_current_bytes,
            l1_hits,
            l1_misses,
            l2_entries,
            l2_bytes: self.l2_current_bytes,
            l2_hits,
            l2_misses,
            uptime: self.created_at.elapsed(),
        }
    }

    /// List all cached models
    #[must_use]
    pub fn list(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();

        for (name, entry) in &self.l1_cache {
            models.push(ModelInfo {
                name: name.clone(),
                model_type: entry.model_type,
                size_bytes: entry.data.size(),
                is_bundled: false,
                is_cached: true,
                cache_tier: Some(CacheTier::L1Hot),
            });
        }

        for (name, entry) in &self.l2_cache {
            if !self.l1_cache.contains_key(name) {
                models.push(ModelInfo {
                    name: name.clone(),
                    model_type: entry.model_type,
                    size_bytes: entry.data.size(),
                    is_bundled: false,
                    is_cached: true,
                    cache_tier: Some(CacheTier::L2Warm),
                });
            }
        }

        models
    }

    // Find eviction candidate in L1 based on policy
    fn find_eviction_candidate_l1(&self) -> Option<String> {
        match self.config.eviction_policy {
            EvictionPolicy::Fixed => None,
            EvictionPolicy::LRU | EvictionPolicy::Clock => self
                .l1_cache
                .iter()
                .min_by_key(|(_, e)| e.stats.last_access)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::LFU => self
                .l1_cache
                .iter()
                .min_by_key(|(_, e)| e.stats.hit_count)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::ARC => {
                // Simplified ARC: balance recency and frequency
                self.l1_cache
                    .iter()
                    .min_by_key(|(_, e)| {
                        e.stats.last_access.saturating_add(e.stats.hit_count * 100)
                    })
                    .map(|(k, _)| k.clone())
            }
        }
    }

    // Find eviction candidate in L2
    fn find_eviction_candidate_l2(&self) -> Option<String> {
        match self.config.eviction_policy {
            EvictionPolicy::Fixed => None,
            _ => self
                .l2_cache
                .iter()
                .min_by_key(|(_, e)| e.stats.last_access)
                .map(|(k, _)| k.clone()),
        }
    }

    // Evict from L1
    fn evict_l1(&mut self, key: &str) {
        if let Some(entry) = self.l1_cache.remove(key) {
            self.l1_current_bytes -= entry.data.size();
        }
    }

    // Evict from L2
    fn evict_l2(&mut self, key: &str) {
        if let Some(entry) = self.l2_cache.remove(key) {
            self.l2_current_bytes -= entry.data.size();
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of L1 entries
    pub l1_entries: usize,
    /// L1 cache size in bytes
    pub l1_bytes: usize,
    /// L1 hit count
    pub l1_hits: u64,
    /// L1 miss count
    pub l1_misses: u64,
    /// Number of L2 entries
    pub l2_entries: usize,
    /// L2 cache size in bytes
    pub l2_bytes: usize,
    /// L2 hit count
    pub l2_hits: u64,
    /// L2 miss count
    pub l2_misses: u64,
    /// Cache uptime
    pub uptime: Duration,
}

impl CacheStats {
    /// Get overall hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total_misses = self.l1_misses + self.l2_misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Get total cache size
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.l1_bytes + self.l2_bytes
    }

    /// Get total entries
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.l1_entries + self.l2_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eviction_policy_default() {
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::LRU);
    }

    #[test]
    fn test_eviction_policy_supports_eviction() {
        assert!(EvictionPolicy::LRU.supports_eviction());
        assert!(EvictionPolicy::LFU.supports_eviction());
        assert!(EvictionPolicy::ARC.supports_eviction());
        assert!(EvictionPolicy::Clock.supports_eviction());
        assert!(!EvictionPolicy::Fixed.supports_eviction());
    }

    #[test]
    fn test_memory_budget() {
        let budget = MemoryBudget::new(100);
        assert_eq!(budget.max_pages, 100);
        assert_eq!(budget.high_watermark, 90);
        assert_eq!(budget.low_watermark, 70);
    }

    #[test]
    fn test_memory_budget_eviction_needed() {
        let budget = MemoryBudget::new(100);
        assert!(!budget.needs_eviction(50));
        assert!(!budget.needs_eviction(89));
        assert!(budget.needs_eviction(90));
        assert!(budget.needs_eviction(100));
    }

    #[test]
    fn test_memory_budget_reserved_pages() {
        let mut budget = MemoryBudget::new(100);
        budget.reserve_page(1);
        budget.reserve_page(2);

        assert!(!budget.can_evict(1));
        assert!(!budget.can_evict(2));
        assert!(budget.can_evict(3));

        budget.release_page(1);
        assert!(budget.can_evict(1));
    }

    #[test]
    fn test_access_stats() {
        let mut stats = AccessStats::new();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.record_hit(100, 1);
        stats.record_hit(200, 2);
        stats.record_miss(3);

        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
        assert!((stats.avg_access_time_ns() - 150.0).abs() < 0.01);
    }

    #[test]
    #[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
    fn test_cache_metadata_expiration() {
        let meta = CacheMetadata::new(1024).with_ttl(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(meta.is_expired());
    }

    #[test]
    fn test_cache_metadata_no_expiration() {
        let meta = CacheMetadata::new(1024);
        assert!(!meta.is_expired());
    }

    #[test]
    fn test_cache_data_size() {
        let compressed = CacheData::Compressed(vec![0u8; 100]);
        assert_eq!(compressed.size(), 100);
        assert!(compressed.is_compressed());

        let decompressed = CacheData::Decompressed(vec![0u8; 200]);
        assert_eq!(decompressed.size(), 200);
        assert!(!decompressed.is_compressed());
    }

    #[test]
    fn test_cache_tier_latency() {
        assert!(CacheTier::L1Hot.typical_latency() < CacheTier::L2Warm.typical_latency());
        assert!(CacheTier::L2Warm.typical_latency() < CacheTier::L3Cold.typical_latency());
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.l1_max_bytes, 64 * 1024 * 1024);
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
        assert!(config.prefetch_enabled);
    }

    #[test]
    fn test_cache_config_embedded() {
        let config = CacheConfig::embedded(1024 * 1024);
        assert_eq!(config.l1_max_bytes, 1024 * 1024);
        assert_eq!(config.l2_max_bytes, 0);
        assert_eq!(config.eviction_policy, EvictionPolicy::Fixed);
        assert!(!config.prefetch_enabled);
    }

    #[test]
    fn test_model_registry_basic() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );

        registry.insert_l1("model1".to_string(), entry);
        assert!(registry.contains("model1"));
        assert_eq!(registry.get_tier("model1"), Some(CacheTier::L1Hot));
    }

    #[test]
    fn test_model_registry_get() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );

        registry.insert_l1("model1".to_string(), entry);
        assert!(registry.get("model1").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_model_registry_remove() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );

        registry.insert_l1("model1".to_string(), entry);
        assert!(registry.contains("model1"));

        registry.remove("model1");
        assert!(!registry.contains("model1"));
    }

    #[test]
    fn test_model_registry_eviction() {
        let config = CacheConfig::new().with_l1_size(2048);
        let mut registry = ModelRegistry::new(config);

        // Insert entries that will require eviction
        for i in 0..5 {
            let entry = CacheEntry::new(
                [i as u8; 32],
                ModelType::new(1),
                CacheData::Decompressed(vec![0u8; 1024]),
            );
            registry.insert_l1(format!("model{}", i), entry);
        }

        // Should have evicted some entries
        assert!(registry.l1_current_bytes <= 2048);
    }

    #[test]
    fn test_model_registry_stats() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );

        registry.insert_l1("model1".to_string(), entry);
        let stats = registry.stats();

        assert_eq!(stats.l1_entries, 1);
        assert_eq!(stats.l1_bytes, 1024);
        assert_eq!(stats.total_entries(), 1);
    }

    #[test]
    fn test_model_registry_list() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry1 = CacheEntry::new(
            [1u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );
        let entry2 = CacheEntry::new(
            [2u8; 32],
            ModelType::new(2),
            CacheData::Decompressed(vec![0u8; 2048]),
        );

        registry.insert_l1("model1".to_string(), entry1);
        registry.insert_l1("model2".to_string(), entry2);

        let list = registry.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_model_registry_clear() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );

        registry.insert_l1("model1".to_string(), entry);
        registry.clear();

        assert!(!registry.contains("model1"));
        assert_eq!(registry.l1_current_bytes, 0);
    }

    // Additional tests for coverage

    #[test]
    fn test_eviction_policy_description() {
        assert!(EvictionPolicy::LRU.description().contains("Recently"));
        assert!(EvictionPolicy::LFU.description().contains("Frequently"));
        assert!(EvictionPolicy::ARC.description().contains("Adaptive"));
        assert!(EvictionPolicy::Clock.description().contains("Clock"));
        assert!(EvictionPolicy::Fixed
            .description()
            .contains("deterministic"));
    }

    #[test]
    fn test_eviction_policy_recommended_use_case() {
        assert!(EvictionPolicy::LRU
            .recommended_use_case()
            .contains("Sequential"));
        assert!(EvictionPolicy::LFU
            .recommended_use_case()
            .contains("Random"));
        assert!(EvictionPolicy::ARC.recommended_use_case().contains("Mixed"));
        assert!(EvictionPolicy::Clock
            .recommended_use_case()
            .contains("Embedded"));
        assert!(EvictionPolicy::Fixed
            .recommended_use_case()
            .contains("NASA"));
    }

    #[test]
    fn test_memory_budget_with_watermarks() {
        let budget = MemoryBudget::with_watermarks(100, 0.8, 0.6);
        assert_eq!(budget.max_pages, 100);
        assert_eq!(budget.high_watermark, 80);
        assert_eq!(budget.low_watermark, 60);
    }

    #[test]
    fn test_memory_budget_can_stop_eviction() {
        let budget = MemoryBudget::new(100);
        assert!(budget.can_stop_eviction(70));
        assert!(budget.can_stop_eviction(50));
        assert!(!budget.can_stop_eviction(71));
        assert!(!budget.can_stop_eviction(100));
    }

    #[test]
    fn test_access_stats_prefetch() {
        let mut stats = AccessStats::new();
        stats.record_hit(100, 1);
        stats.record_prefetch_hit();
        stats.record_hit(100, 2);
        stats.record_prefetch_hit();

        assert_eq!(stats.prefetch_hits, 2);
        assert!((stats.prefetch_effectiveness() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_access_stats_zero_hits() {
        let stats = AccessStats::new();
        assert_eq!(stats.avg_access_time_ns(), 0.0);
        assert_eq!(stats.prefetch_effectiveness(), 0.0);
    }

    #[test]
    fn test_cache_metadata_with_source() {
        let path = PathBuf::from("/tmp/model.apr");
        let mtime = SystemTime::now();
        let meta = CacheMetadata::new(1024).with_source(path.clone(), mtime);

        assert_eq!(meta.source_path, Some(path));
        assert_eq!(meta.source_mtime, Some(mtime));
    }

    #[test]
    fn test_cache_metadata_is_stale() {
        let old_mtime = SystemTime::UNIX_EPOCH;
        let path = PathBuf::from("/tmp/model.apr");
        let meta = CacheMetadata::new(1024).with_source(path, old_mtime);

        let new_mtime = SystemTime::now();
        assert!(meta.is_stale(new_mtime));
    }

    #[test]
    fn test_cache_metadata_not_stale() {
        let meta = CacheMetadata::new(1024);
        // No source, so not stale
        assert!(!meta.is_stale(SystemTime::now()));
    }

    #[test]
    #[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
    fn test_cache_metadata_age() {
        let meta = CacheMetadata::new(1024);
        std::thread::sleep(Duration::from_millis(5));
        assert!(meta.age().as_millis() >= 5);
    }

    #[test]
    fn test_cache_metadata_with_compression_ratio() {
        let meta = CacheMetadata::new(1024).with_compression_ratio(0.5);
        assert!((meta.compression_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_data_mapped() {
        let mapped = CacheData::Mapped {
            path: PathBuf::from("/tmp/model.apr"),
            offset: 1024,
            length: 4096,
        };
        assert_eq!(mapped.size(), 4096);
        assert!(mapped.is_mapped());
        assert!(!mapped.is_compressed());
    }

    #[test]
    fn test_cache_entry_tier() {
        let entry_decompressed = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 100]),
        );
        assert_eq!(entry_decompressed.tier(), CacheTier::L1Hot);

        let entry_compressed = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Compressed(vec![0u8; 100]),
        );
        assert_eq!(entry_compressed.tier(), CacheTier::L2Warm);

        let entry_mapped = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Mapped {
                path: PathBuf::from("/tmp/x"),
                offset: 0,
                length: 100,
            },
        );
        assert_eq!(entry_mapped.tier(), CacheTier::L2Warm);
    }

    #[test]
    fn test_cache_entry_is_valid() {
        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 100]),
        );
        assert!(entry.is_valid());
    }

    #[test]
    #[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
    fn test_cache_entry_is_valid_expired() {
        // Entry with expired TTL
        let mut entry_expired = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 100]),
        );
        entry_expired.metadata = CacheMetadata::new(100).with_ttl(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(!entry_expired.is_valid());
    }

    #[test]
    fn test_cache_tier_name() {
        assert_eq!(CacheTier::L1Hot.name(), "L1 Hot Cache");
        assert_eq!(CacheTier::L2Warm.name(), "L2 Warm Cache");
        assert_eq!(CacheTier::L3Cold.name(), "L3 Cold Storage");
    }

    #[test]
    fn test_cache_config_builders() {
        let config = CacheConfig::new()
            .with_l1_size(1024)
            .with_l2_size(2048)
            .with_eviction_policy(EvictionPolicy::LFU)
            .with_ttl(Duration::from_secs(60))
            .with_prefetch(false);

        assert_eq!(config.l1_max_bytes, 1024);
        assert_eq!(config.l2_max_bytes, 2048);
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
        assert_eq!(config.default_ttl, Some(Duration::from_secs(60)));
        assert!(!config.prefetch_enabled);
    }

    #[test]
    fn test_model_registry_l2_operations() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Compressed(vec![0u8; 1024]),
        );

        registry.insert_l2("model1".to_string(), entry);
        assert!(registry.contains("model1"));
        assert_eq!(registry.get_tier("model1"), Some(CacheTier::L2Warm));

        assert!(registry.get("model1").is_some());
    }

    #[test]
    fn test_model_registry_l2_eviction() {
        let config = CacheConfig::new().with_l2_size(2048);
        let mut registry = ModelRegistry::new(config);

        for i in 0..5 {
            let entry = CacheEntry::new(
                [i as u8; 32],
                ModelType::new(1),
                CacheData::Compressed(vec![0u8; 1024]),
            );
            registry.insert_l2(format!("model{}", i), entry);
        }

        assert!(registry.l2_current_bytes <= 2048);
    }

    #[test]
    fn test_model_registry_remove_l2() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry = CacheEntry::new(
            [0u8; 32],
            ModelType::new(1),
            CacheData::Compressed(vec![0u8; 1024]),
        );

        registry.insert_l2("model1".to_string(), entry);
        registry.remove("model1");
        assert!(!registry.contains("model1"));
    }

    #[test]
    fn test_model_registry_lfu_eviction() {
        let config = CacheConfig::new()
            .with_l1_size(2048)
            .with_eviction_policy(EvictionPolicy::LFU);
        let mut registry = ModelRegistry::new(config);

        for i in 0..5 {
            let entry = CacheEntry::new(
                [i as u8; 32],
                ModelType::new(1),
                CacheData::Decompressed(vec![0u8; 1024]),
            );
            registry.insert_l1(format!("model{}", i), entry);
        }

        assert!(registry.l1_current_bytes <= 2048);
    }

    #[test]
    fn test_model_registry_arc_eviction() {
        let config = CacheConfig::new()
            .with_l1_size(2048)
            .with_eviction_policy(EvictionPolicy::ARC);
        let mut registry = ModelRegistry::new(config);

        for i in 0..5 {
            let entry = CacheEntry::new(
                [i as u8; 32],
                ModelType::new(1),
                CacheData::Decompressed(vec![0u8; 1024]),
            );
            registry.insert_l1(format!("model{}", i), entry);
        }

        assert!(registry.l1_current_bytes <= 2048);
    }

    #[test]
    fn test_model_registry_fixed_no_eviction() {
        let config = CacheConfig::new()
            .with_l1_size(1024)
            .with_eviction_policy(EvictionPolicy::Fixed);
        let mut registry = ModelRegistry::new(config);

        // First entry fits
        let entry1 = CacheEntry::new(
            [1u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 512]),
        );
        registry.insert_l1("model1".to_string(), entry1);

        // Second entry also fits
        let entry2 = CacheEntry::new(
            [2u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 256]),
        );
        registry.insert_l1("model2".to_string(), entry2);

        assert!(registry.contains("model1"));
        assert!(registry.contains("model2"));
    }

    #[test]
    fn test_model_registry_get_tier_none() {
        let config = CacheConfig::default();
        let registry = ModelRegistry::new(config);
        assert_eq!(registry.get_tier("nonexistent"), None);
    }

    #[test]
    fn test_model_registry_list_both_caches() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry1 = CacheEntry::new(
            [1u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );
        let entry2 = CacheEntry::new(
            [2u8; 32],
            ModelType::new(2),
            CacheData::Compressed(vec![0u8; 512]),
        );

        registry.insert_l1("model1".to_string(), entry1);
        registry.insert_l2("model2".to_string(), entry2);

        let list = registry.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_cache_stats_hit_rate_zero() {
        let stats = CacheStats {
            l1_entries: 0,
            l1_bytes: 0,
            l1_hits: 0,
            l1_misses: 0,
            l2_entries: 0,
            l2_bytes: 0,
            l2_hits: 0,
            l2_misses: 0,
            uptime: Duration::from_secs(1),
        };
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_total_bytes() {
        let stats = CacheStats {
            l1_entries: 1,
            l1_bytes: 1024,
            l1_hits: 10,
            l1_misses: 5,
            l2_entries: 2,
            l2_bytes: 2048,
            l2_hits: 5,
            l2_misses: 3,
            uptime: Duration::from_secs(1),
        };
        assert_eq!(stats.total_bytes(), 3072);
        assert_eq!(stats.total_entries(), 3);
    }

    #[test]
    fn test_model_type_new() {
        let mt = ModelType::new(42);
        assert_eq!(mt.0, 42);
    }

    #[test]
    fn test_model_registry_insert_replaces() {
        let config = CacheConfig::default();
        let mut registry = ModelRegistry::new(config);

        let entry1 = CacheEntry::new(
            [1u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );
        let entry2 = CacheEntry::new(
            [2u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 2048]),
        );

        registry.insert_l1("model".to_string(), entry1);
        assert_eq!(registry.l1_current_bytes, 1024);

        registry.insert_l1("model".to_string(), entry2);
        assert_eq!(registry.l1_current_bytes, 2048);
    }

    #[test]
    fn test_clone_debug_traits() {
        let policy = EvictionPolicy::LRU;
        let _ = policy.clone();
        let _ = format!("{policy:?}");

        let budget = MemoryBudget::new(100);
        let _ = budget.clone();
        let _ = format!("{budget:?}");

        let stats = AccessStats::new();
        let _ = stats.clone();
        let _ = format!("{stats:?}");

        let meta = CacheMetadata::new(1024);
        let _ = meta.clone();
        let _ = format!("{meta:?}");

        let config = CacheConfig::default();
        let _ = config.clone();
        let _ = format!("{config:?}");
    }
}
