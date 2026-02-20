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

include!("cache_config.rs");
