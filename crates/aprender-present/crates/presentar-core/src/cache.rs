#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
// Data Caching Layer - WASM-first caching for API/network data
//
// Provides:
// - In-memory LRU cache
// - Time-based expiration
// - Cache invalidation patterns
// - Stale-while-revalidate strategy
// - Memory pressure handling

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

/// Unique identifier for a cache entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(u64);

impl CacheKey {
    pub fn from_str(s: &str) -> Self {
        // Simple hash for string keys
        let mut hash: u64 = 0;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        Self(hash)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for CacheKey {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<&str> for CacheKey {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

/// Cache entry state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// Entry is fresh and valid
    Fresh,
    /// Entry is stale but can be used while revalidating
    Stale,
    /// Entry has expired and should not be used
    Expired,
}

/// Cache entry metadata
#[derive(Debug, Clone)]
pub struct CacheMetadata {
    /// When the entry was created
    pub created_at: u64,
    /// When the entry was last accessed
    pub last_accessed: u64,
    /// Time to live in milliseconds
    pub ttl_ms: u64,
    /// Stale time in milliseconds (for stale-while-revalidate)
    pub stale_ms: u64,
    /// Number of times this entry was accessed
    pub access_count: u64,
    /// Size of the cached data in bytes
    pub size_bytes: usize,
    /// Tags for invalidation
    pub tags: Vec<String>,
}

impl CacheMetadata {
    /// Check current state based on timestamp
    pub fn state(&self, now: u64) -> CacheState {
        let age = now.saturating_sub(self.created_at);
        if age <= self.ttl_ms {
            CacheState::Fresh
        } else if age <= self.ttl_ms + self.stale_ms {
            CacheState::Stale
        } else {
            CacheState::Expired
        }
    }

    /// Check if entry is expired
    pub fn is_expired(&self, now: u64) -> bool {
        self.state(now) == CacheState::Expired
    }
}

/// Configuration for cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory in bytes
    pub max_memory: usize,
    /// Default TTL in milliseconds
    pub default_ttl_ms: u64,
    /// Default stale time in milliseconds
    pub default_stale_ms: u64,
    /// Enable LRU eviction
    pub enable_lru: bool,
    /// Cleanup interval in milliseconds
    pub cleanup_interval_ms: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory: 50 * 1024 * 1024,  // 50 MB
            default_ttl_ms: 5 * 60 * 1000, // 5 minutes
            default_stale_ms: 60 * 1000,   // 1 minute stale
            enable_lru: true,
            cleanup_interval_ms: 60 * 1000, // 1 minute
        }
    }
}

/// Cache entry options
#[derive(Debug, Clone, Default)]
pub struct CacheOptions {
    /// Custom TTL (None = use default)
    pub ttl: Option<Duration>,
    /// Custom stale time (None = use default)
    pub stale: Option<Duration>,
    /// Tags for invalidation
    pub tags: Vec<String>,
    /// Priority (higher = less likely to be evicted)
    pub priority: u8,
}

impl CacheOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn with_stale(mut self, stale: Duration) -> Self {
        self.stale = Some(stale);
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Cache event for notifications
#[derive(Debug, Clone)]
pub enum CacheEvent {
    /// Entry was added
    Added(CacheKey),
    /// Entry was accessed
    Hit(CacheKey),
    /// Entry was not found
    Miss(CacheKey),
    /// Entry was evicted
    Evicted(CacheKey),
    /// Entry was invalidated
    Invalidated(CacheKey),
    /// Entries were invalidated by tag
    TagInvalidated(String, usize),
    /// Cache was cleared
    Cleared,
}

/// Callback for cache events
pub type CacheCallback = Arc<dyn Fn(CacheEvent) + Send + Sync>;

/// Generic cache entry
struct CacheEntry<V> {
    value: V,
    metadata: CacheMetadata,
    #[allow(dead_code)]
    priority: u8,
}

/// In-memory data cache
pub struct DataCache<K, V>
where
    K: Eq + Hash + Clone,
{
    config: CacheConfig,
    entries: HashMap<K, CacheEntry<V>>,
    /// Access order for LRU
    access_order: Vec<K>,
    /// Current memory usage
    current_memory: usize,
    /// Current timestamp
    timestamp: u64,
    /// Last cleanup timestamp
    last_cleanup: u64,
    /// Event listeners
    listeners: Vec<CacheCallback>,
    /// Statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub current_entries: usize,
    pub current_memory: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl<K, V> Default for DataCache<K, V>
where
    K: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

impl<K, V> DataCache<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            access_order: Vec::new(),
            current_memory: 0,
            timestamp: 0,
            last_cleanup: 0,
            listeners: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    /// Get a value from cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        // Check if cleanup needed
        self.maybe_cleanup();

        if let Some(entry) = self.entries.get_mut(key) {
            // Check expiration
            if entry.metadata.is_expired(self.timestamp) {
                return None;
            }

            // Update access time
            entry.metadata.last_accessed = self.timestamp;
            entry.metadata.access_count += 1;

            // Update LRU order
            if self.config.enable_lru {
                self.access_order.retain(|k| k != key);
                self.access_order.push(key.clone());
            }

            self.stats.hits += 1;
            Some(&entry.value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get a value and its state
    pub fn get_with_state(&mut self, key: &K) -> Option<(&V, CacheState)> {
        self.maybe_cleanup();

        if let Some(entry) = self.entries.get_mut(key) {
            let state = entry.metadata.state(self.timestamp);

            // Don't return expired entries
            if state == CacheState::Expired {
                return None;
            }

            entry.metadata.last_accessed = self.timestamp;
            entry.metadata.access_count += 1;

            if self.config.enable_lru {
                self.access_order.retain(|k| k != key);
                self.access_order.push(key.clone());
            }

            self.stats.hits += 1;
            Some((&entry.value, state))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check if key exists and is not expired
    pub fn contains(&self, key: &K) -> bool {
        self.entries
            .get(key)
            .is_some_and(|e| !e.metadata.is_expired(self.timestamp))
    }

    /// Insert a value into cache
    pub fn insert(&mut self, key: K, value: V, options: CacheOptions)
    where
        V: CacheSize,
    {
        let size = value.cache_size();
        let ttl = options.ttl.map(|d| d.as_millis() as u64);
        let stale = options.stale.map(|d| d.as_millis() as u64);

        let metadata = CacheMetadata {
            created_at: self.timestamp,
            last_accessed: self.timestamp,
            ttl_ms: ttl.unwrap_or(self.config.default_ttl_ms),
            stale_ms: stale.unwrap_or(self.config.default_stale_ms),
            access_count: 0,
            size_bytes: size,
            tags: options.tags,
        };

        // Remove old entry if exists
        if let Some(old) = self.entries.remove(&key) {
            self.current_memory = self.current_memory.saturating_sub(old.metadata.size_bytes);
            self.access_order.retain(|k| k != &key);
        }

        // Evict if needed
        while self.entries.len() >= self.config.max_entries
            || self.current_memory + size > self.config.max_memory
        {
            if !self.evict_one() {
                break;
            }
        }

        self.current_memory += size;

        let entry = CacheEntry {
            value,
            metadata,
            priority: options.priority,
        };

        self.entries.insert(key.clone(), entry);
        self.access_order.push(key);

        self.stats.current_entries = self.entries.len();
        self.stats.current_memory = self.current_memory;
    }

    /// Insert with default options
    pub fn insert_default(&mut self, key: K, value: V)
    where
        V: CacheSize,
    {
        self.insert(key, value, CacheOptions::default());
    }

    /// Remove a value from cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_memory = self
                .current_memory
                .saturating_sub(entry.metadata.size_bytes);
            self.access_order.retain(|k| k != key);
            self.stats.current_entries = self.entries.len();
            self.stats.current_memory = self.current_memory;
            self.stats.invalidations += 1;
            Some(entry.value)
        } else {
            None
        }
    }

    /// Invalidate entries by tag
    pub fn invalidate_tag(&mut self, tag: &str) -> usize {
        let keys_to_remove: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, e)| e.metadata.tags.iter().any(|t| t == tag))
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.remove(&key);
        }

        self.emit(CacheEvent::TagInvalidated(tag.to_string(), count));
        count
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_memory = 0;
        self.stats.current_entries = 0;
        self.stats.current_memory = 0;
        self.emit(CacheEvent::Cleared);
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_memory
    }

    /// Update timestamp (call each frame or periodically)
    pub fn tick(&mut self, delta_ms: u64) {
        self.timestamp += delta_ms;
        self.maybe_cleanup();
    }

    /// Set timestamp directly
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.timestamp = timestamp;
    }

    /// Get current timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Add event listener
    pub fn on_event(&mut self, callback: CacheCallback) {
        self.listeners.push(callback);
    }

    fn emit(&self, event: CacheEvent) {
        for listener in &self.listeners {
            listener(event.clone());
        }
    }

    fn maybe_cleanup(&mut self) {
        if self.timestamp - self.last_cleanup >= self.config.cleanup_interval_ms {
            self.cleanup_expired();
            self.last_cleanup = self.timestamp;
        }
    }

    fn cleanup_expired(&mut self) {
        let now = self.timestamp;
        let keys_to_remove: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, e)| e.metadata.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_memory = self
                    .current_memory
                    .saturating_sub(entry.metadata.size_bytes);
                self.access_order.retain(|k| k != &key);
                self.stats.evictions += 1;
            }
        }

        self.stats.current_entries = self.entries.len();
        self.stats.current_memory = self.current_memory;
    }

    fn evict_one(&mut self) -> bool {
        if self.config.enable_lru && !self.access_order.is_empty() {
            // LRU eviction
            let key = self.access_order.remove(0);
            if let Some(entry) = self.entries.remove(&key) {
                self.current_memory = self
                    .current_memory
                    .saturating_sub(entry.metadata.size_bytes);
                self.stats.evictions += 1;
                return true;
            }
        } else if !self.entries.is_empty() {
            // Random eviction as fallback
            if let Some(key) = self.entries.keys().next().cloned() {
                if let Some(entry) = self.entries.remove(&key) {
                    self.current_memory = self
                        .current_memory
                        .saturating_sub(entry.metadata.size_bytes);
                    self.access_order.retain(|k| k != &key);
                    self.stats.evictions += 1;
                    return true;
                }
            }
        }
        false
    }
}

/// Trait for getting size of cached values
pub trait CacheSize {
    fn cache_size(&self) -> usize;
}

// Implement CacheSize for common types
impl CacheSize for String {
    fn cache_size(&self) -> usize {
        self.len()
    }
}

impl<T> CacheSize for Vec<T> {
    fn cache_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<T> CacheSize for Box<T> {
    fn cache_size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl CacheSize for () {
    fn cache_size(&self) -> usize {
        0
    }
}

impl CacheSize for i32 {
    fn cache_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl CacheSize for i64 {
    fn cache_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl CacheSize for f32 {
    fn cache_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl CacheSize for f64 {
    fn cache_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Simple string-keyed cache
pub type StringCache<V> = DataCache<String, V>;

/// Builder for cache entries
pub struct CacheBuilder<V> {
    value: V,
    options: CacheOptions,
}

impl<V> CacheBuilder<V> {
    pub fn new(value: V) -> Self {
        Self {
            value,
            options: CacheOptions::default(),
        }
    }

    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.options.ttl = Some(ttl);
        self
    }

    pub fn stale(mut self, stale: Duration) -> Self {
        self.options.stale = Some(stale);
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.options.tags.push(tag.into());
        self
    }

    pub fn priority(mut self, priority: u8) -> Self {
        self.options.priority = priority;
        self
    }

    pub fn build(self) -> (V, CacheOptions) {
        (self.value, self.options)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_from_str() {
        let key1 = CacheKey::from_str("test");
        let key2 = CacheKey::from_str("test");
        let key3 = CacheKey::from_str("different");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_from_u64() {
        let key: CacheKey = 42u64.into();
        assert_eq!(key.as_u64(), 42);
    }

    #[test]
    fn test_cache_default() {
        let cache: DataCache<String, String> = DataCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_insert_get() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key".to_string(), "value".to_string());

        let result = cache.get(&"key".to_string());
        assert_eq!(result, Some(&"value".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache: DataCache<String, String> = DataCache::default();

        let result = cache.get(&"missing".to_string());
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_contains() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key".to_string(), "value".to_string());

        assert!(cache.contains(&"key".to_string()));
        assert!(!cache.contains(&"missing".to_string()));
    }

    #[test]
    fn test_cache_remove() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key".to_string(), "value".to_string());
        let removed = cache.remove(&"key".to_string());

        assert_eq!(removed, Some("value".to_string()));
        assert!(!cache.contains(&"key".to_string()));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key1".to_string(), "value1".to_string());
        cache.insert_default("key2".to_string(), "value2".to_string());

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_expiration() {
        let config = CacheConfig {
            default_ttl_ms: 100,
            default_stale_ms: 0,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        cache.insert_default("key".to_string(), "value".to_string());

        // Should be fresh
        assert!(cache.contains(&"key".to_string()));

        // Advance time past TTL
        cache.set_timestamp(200);

        // Should be expired
        assert!(!cache.contains(&"key".to_string()));
        assert!(cache.get(&"key".to_string()).is_none());
    }

    #[test]
    fn test_cache_stale_while_revalidate() {
        let config = CacheConfig {
            default_ttl_ms: 100,
            default_stale_ms: 50,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        cache.insert_default("key".to_string(), "value".to_string());

        // Should be fresh
        let (_, state) = cache.get_with_state(&"key".to_string()).unwrap();
        assert_eq!(state, CacheState::Fresh);

        // Advance into stale period
        cache.set_timestamp(120);
        let (_, state) = cache.get_with_state(&"key".to_string()).unwrap();
        assert_eq!(state, CacheState::Stale);

        // Advance past stale period
        cache.set_timestamp(200);
        assert!(cache.get_with_state(&"key".to_string()).is_none());
    }

    #[test]
    fn test_cache_custom_ttl() {
        let config = CacheConfig {
            default_ttl_ms: 1000,
            default_stale_ms: 0, // No stale period for this test
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        let options = CacheOptions::new().with_ttl(Duration::from_millis(50));
        cache.insert("key".to_string(), "value".to_string(), options);

        cache.set_timestamp(100);
        assert!(cache.get(&"key".to_string()).is_none());
    }

    #[test]
    fn test_cache_tags() {
        let mut cache: DataCache<String, String> = DataCache::default();

        let options1 = CacheOptions::new().with_tag("user").with_tag("profile");
        let options2 = CacheOptions::new().with_tag("user");
        let options3 = CacheOptions::new().with_tag("settings");

        cache.insert("key1".to_string(), "value1".to_string(), options1);
        cache.insert("key2".to_string(), "value2".to_string(), options2);
        cache.insert("key3".to_string(), "value3".to_string(), options3);

        // Invalidate by tag
        let count = cache.invalidate_tag("user");
        assert_eq!(count, 2);

        assert!(!cache.contains(&"key1".to_string()));
        assert!(!cache.contains(&"key2".to_string()));
        assert!(cache.contains(&"key3".to_string()));
    }

    #[test]
    fn test_cache_lru_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            enable_lru: true,
            ..Default::default()
        };
        let mut cache: DataCache<String, i32> = DataCache::new(config);

        cache.insert_default("key1".to_string(), 1);
        cache.insert_default("key2".to_string(), 2);
        cache.insert_default("key3".to_string(), 3);

        // Access key1 to make it recently used
        cache.get(&"key1".to_string());

        // Insert new entry, should evict key2 (LRU)
        cache.insert_default("key4".to_string(), 4);

        assert!(cache.contains(&"key1".to_string()));
        assert!(!cache.contains(&"key2".to_string())); // Evicted
        assert!(cache.contains(&"key3".to_string()));
        assert!(cache.contains(&"key4".to_string()));
    }

    #[test]
    fn test_cache_memory_limit() {
        let config = CacheConfig {
            max_memory: 20, // Very small
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        cache.insert_default("key1".to_string(), "12345".to_string()); // 5 bytes
        cache.insert_default("key2".to_string(), "12345".to_string()); // 5 bytes
        cache.insert_default("key3".to_string(), "12345".to_string()); // 5 bytes
        cache.insert_default("key4".to_string(), "12345".to_string()); // 5 bytes

        // Should have evicted some entries
        assert!(cache.memory_usage() <= 20);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key".to_string(), "value".to_string());

        cache.get(&"key".to_string()); // Hit
        cache.get(&"key".to_string()); // Hit
        cache.get(&"missing".to_string()); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_tick() {
        let config = CacheConfig {
            cleanup_interval_ms: 100,
            default_ttl_ms: 50,
            default_stale_ms: 0, // No stale period for this test
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        cache.insert_default("key".to_string(), "value".to_string());

        // Tick forward past TTL and cleanup interval
        cache.tick(150);

        // Entry should be cleaned up
        assert!(!cache.contains(&"key".to_string()));
    }

    #[test]
    fn test_cache_metadata_state() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: 100,
            stale_ms: 50,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };

        assert_eq!(meta.state(50), CacheState::Fresh);
        assert_eq!(meta.state(120), CacheState::Stale);
        assert_eq!(meta.state(200), CacheState::Expired);

        assert!(!meta.is_expired(100));
        assert!(meta.is_expired(200));
    }

    #[test]
    fn test_cache_options_builder() {
        let options = CacheOptions::new()
            .with_ttl(Duration::from_secs(60))
            .with_stale(Duration::from_secs(30))
            .with_tag("test")
            .with_priority(5);

        assert_eq!(options.ttl, Some(Duration::from_secs(60)));
        assert_eq!(options.stale, Some(Duration::from_secs(30)));
        assert_eq!(options.tags, vec!["test"]);
        assert_eq!(options.priority, 5);
    }

    #[test]
    fn test_cache_builder() {
        let (value, options) = CacheBuilder::new("test".to_string())
            .ttl(Duration::from_secs(60))
            .stale(Duration::from_secs(30))
            .tag("user")
            .priority(3)
            .build();

        assert_eq!(value, "test");
        assert_eq!(options.ttl, Some(Duration::from_secs(60)));
        assert_eq!(options.tags, vec!["user"]);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 1000);
        assert_eq!(config.max_memory, 50 * 1024 * 1024);
        assert_eq!(config.default_ttl_ms, 5 * 60 * 1000);
        assert_eq!(config.default_stale_ms, 60 * 1000);
        assert!(config.enable_lru);
    }

    #[test]
    fn test_cache_size_string() {
        let s = "hello".to_string();
        assert_eq!(s.cache_size(), 5);
    }

    #[test]
    fn test_cache_size_vec() {
        let v: Vec<u8> = vec![1, 2, 3, 4, 5];
        assert_eq!(v.cache_size(), 5);
    }

    #[test]
    fn test_cache_size_i32() {
        let n: i32 = 42;
        assert_eq!(n.cache_size(), 4);
    }

    #[test]
    fn test_cache_update_entry() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key".to_string(), "value1".to_string());
        cache.insert_default("key".to_string(), "value2".to_string());

        let result = cache.get(&"key".to_string());
        assert_eq!(result, Some(&"value2".to_string()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_event_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut cache: DataCache<String, String> = DataCache::default();
        let event_count = Arc::new(AtomicUsize::new(0));

        let ec = event_count.clone();
        cache.on_event(Arc::new(move |_event| {
            ec.fetch_add(1, Ordering::SeqCst);
        }));

        cache.invalidate_tag("test");
        cache.clear();

        assert!(event_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_cache_state_variants() {
        assert_eq!(CacheState::Fresh, CacheState::Fresh);
        assert_ne!(CacheState::Fresh, CacheState::Stale);
        assert_ne!(CacheState::Stale, CacheState::Expired);
    }

    #[test]
    fn test_stats_hit_rate_empty() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_timestamp() {
        let mut cache: DataCache<String, String> = DataCache::default();
        assert_eq!(cache.timestamp(), 0);

        cache.set_timestamp(1000);
        assert_eq!(cache.timestamp(), 1000);

        cache.tick(500);
        assert_eq!(cache.timestamp(), 1500);
    }

    // ========== Additional CacheKey tests ==========

    #[test]
    fn test_cache_key_empty_string() {
        let key = CacheKey::from_str("");
        assert_eq!(key.as_u64(), 0);
    }

    #[test]
    fn test_cache_key_unicode() {
        let key1 = CacheKey::from_str("日本語");
        let key2 = CacheKey::from_str("日本語");
        let key3 = CacheKey::from_str("中文");
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_long_string() {
        let long_str: String = "a".repeat(10000);
        let key = CacheKey::from_str(&long_str);
        assert!(key.as_u64() > 0);
    }

    #[test]
    fn test_cache_key_from_str_trait() {
        let key: CacheKey = "test".into();
        assert_eq!(key, CacheKey::from_str("test"));
    }

    #[test]
    fn test_cache_key_hash_distribution() {
        // Different short strings should produce different hashes
        let keys: Vec<CacheKey> = (0..100)
            .map(|i| CacheKey::from_str(&format!("key{i}")))
            .collect();
        let unique: std::collections::HashSet<_> = keys.iter().map(|k| k.as_u64()).collect();
        assert_eq!(unique.len(), 100);
    }

    #[test]
    fn test_cache_key_special_chars() {
        let key = CacheKey::from_str("!@#$%^&*()_+-=[]{}|;':\",./<>?");
        assert!(key.as_u64() > 0);
    }

    #[test]
    fn test_cache_key_whitespace() {
        let key1 = CacheKey::from_str("  ");
        let key2 = CacheKey::from_str("\t\n");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_debug() {
        let key = CacheKey::from_str("test");
        let debug = format!("{key:?}");
        assert!(debug.contains("CacheKey"));
    }

    #[test]
    fn test_cache_key_clone() {
        let key1 = CacheKey::from_str("test");
        let key2 = key1;
        assert_eq!(key1, key2);
    }

    // ========== Additional CacheMetadata tests ==========

    #[test]
    fn test_cache_metadata_boundary_fresh() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: 100,
            stale_ms: 50,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };
        // Exactly at ttl boundary should still be fresh
        assert_eq!(meta.state(100), CacheState::Fresh);
    }

    #[test]
    fn test_cache_metadata_boundary_stale() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: 100,
            stale_ms: 50,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };
        // At ttl+1 should be stale
        assert_eq!(meta.state(101), CacheState::Stale);
        // At ttl+stale boundary should still be stale
        assert_eq!(meta.state(150), CacheState::Stale);
    }

    #[test]
    fn test_cache_metadata_zero_ttl() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: 0,
            stale_ms: 0,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };
        assert_eq!(meta.state(0), CacheState::Fresh);
        assert_eq!(meta.state(1), CacheState::Expired);
    }

    #[test]
    fn test_cache_metadata_zero_stale() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: 100,
            stale_ms: 0,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };
        assert_eq!(meta.state(100), CacheState::Fresh);
        assert_eq!(meta.state(101), CacheState::Expired);
    }

    #[test]
    fn test_cache_metadata_large_ttl() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: u64::MAX / 2,
            stale_ms: 1000,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };
        assert_eq!(meta.state(1_000_000), CacheState::Fresh);
    }

    #[test]
    fn test_cache_metadata_created_in_future() {
        let meta = CacheMetadata {
            created_at: 1000,
            last_accessed: 1000,
            ttl_ms: 100,
            stale_ms: 50,
            access_count: 0,
            size_bytes: 0,
            tags: vec![],
        };
        // now < created_at, saturating_sub gives 0
        assert_eq!(meta.state(500), CacheState::Fresh);
    }

    #[test]
    fn test_cache_metadata_with_tags() {
        let meta = CacheMetadata {
            created_at: 0,
            last_accessed: 0,
            ttl_ms: 100,
            stale_ms: 50,
            access_count: 5,
            size_bytes: 1024,
            tags: vec!["user".to_string(), "profile".to_string()],
        };
        assert_eq!(meta.tags.len(), 2);
        assert_eq!(meta.access_count, 5);
        assert_eq!(meta.size_bytes, 1024);
    }

    #[test]
    fn test_cache_metadata_clone() {
        let meta = CacheMetadata {
            created_at: 100,
            last_accessed: 200,
            ttl_ms: 1000,
            stale_ms: 500,
            access_count: 10,
            size_bytes: 256,
            tags: vec!["test".to_string()],
        };
        let cloned = meta;
        assert_eq!(cloned.created_at, 100);
        assert_eq!(cloned.tags, vec!["test"]);
    }

    // ========== Additional CacheState tests ==========

    #[test]
    fn test_cache_state_debug() {
        assert_eq!(format!("{:?}", CacheState::Fresh), "Fresh");
        assert_eq!(format!("{:?}", CacheState::Stale), "Stale");
        assert_eq!(format!("{:?}", CacheState::Expired), "Expired");
    }

    #[test]
    fn test_cache_state_clone() {
        let state = CacheState::Fresh;
        let cloned = state;
        assert_eq!(state, cloned);
    }

    // ========== Additional CacheConfig tests ==========

    #[test]
    fn test_cache_config_custom() {
        let config = CacheConfig {
            max_entries: 500,
            max_memory: 10 * 1024 * 1024,
            default_ttl_ms: 60_000,
            default_stale_ms: 10_000,
            enable_lru: false,
            cleanup_interval_ms: 30_000,
        };
        assert_eq!(config.max_entries, 500);
        assert!(!config.enable_lru);
    }

    #[test]
    fn test_cache_config_clone() {
        let config = CacheConfig::default();
        let cloned = config;
        assert_eq!(cloned.max_entries, 1000);
    }

    // ========== Additional CacheOptions tests ==========

    #[test]
    fn test_cache_options_default() {
        let options = CacheOptions::default();
        assert!(options.ttl.is_none());
        assert!(options.stale.is_none());
        assert!(options.tags.is_empty());
        assert_eq!(options.priority, 0);
    }

    #[test]
    fn test_cache_options_multiple_tags() {
        let options = CacheOptions::new()
            .with_tag("user")
            .with_tag("profile")
            .with_tag("admin");
        assert_eq!(options.tags.len(), 3);
    }

    #[test]
    fn test_cache_options_zero_duration() {
        let options = CacheOptions::new()
            .with_ttl(Duration::ZERO)
            .with_stale(Duration::ZERO);
        assert_eq!(options.ttl, Some(Duration::ZERO));
        assert_eq!(options.stale, Some(Duration::ZERO));
    }

    #[test]
    fn test_cache_options_max_priority() {
        let options = CacheOptions::new().with_priority(255);
        assert_eq!(options.priority, 255);
    }

    #[test]
    fn test_cache_options_clone() {
        let options = CacheOptions::new()
            .with_ttl(Duration::from_secs(60))
            .with_tag("test");
        let cloned = options;
        assert_eq!(cloned.ttl, Some(Duration::from_secs(60)));
        assert_eq!(cloned.tags, vec!["test"]);
    }

    // ========== Additional CacheEvent tests ==========

    #[test]
    fn test_cache_event_all_variants() {
        let key = CacheKey::from_str("test");
        let events = vec![
            CacheEvent::Added(key),
            CacheEvent::Hit(key),
            CacheEvent::Miss(key),
            CacheEvent::Evicted(key),
            CacheEvent::Invalidated(key),
            CacheEvent::TagInvalidated("user".to_string(), 5),
            CacheEvent::Cleared,
        ];
        for event in events {
            let _ = format!("{event:?}");
        }
    }

    #[test]
    fn test_cache_event_clone() {
        let event = CacheEvent::TagInvalidated("test".to_string(), 10);
        let cloned = event;
        if let CacheEvent::TagInvalidated(tag, count) = cloned {
            assert_eq!(tag, "test");
            assert_eq!(count, 10);
        } else {
            panic!("Clone failed");
        }
    }

    // ========== Additional DataCache tests ==========

    #[test]
    fn test_cache_lru_disabled() {
        let config = CacheConfig {
            max_entries: 3,
            enable_lru: false,
            ..Default::default()
        };
        let mut cache: DataCache<String, i32> = DataCache::new(config);

        cache.insert_default("key1".to_string(), 1);
        cache.insert_default("key2".to_string(), 2);
        cache.insert_default("key3".to_string(), 3);
        cache.insert_default("key4".to_string(), 4);

        // Should evict something (random eviction)
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_cache_get_with_state_fresh() {
        let mut cache: DataCache<String, String> = DataCache::default();
        cache.insert_default("key".to_string(), "value".to_string());

        let result = cache.get_with_state(&"key".to_string());
        assert!(result.is_some());
        let (value, state) = result.unwrap();
        assert_eq!(value, "value");
        assert_eq!(state, CacheState::Fresh);
    }

    #[test]
    fn test_cache_get_with_state_miss() {
        let mut cache: DataCache<String, String> = DataCache::default();
        let result = cache.get_with_state(&"missing".to_string());
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_multiple_removes() {
        let mut cache: DataCache<String, String> = DataCache::default();
        cache.insert_default("key".to_string(), "value".to_string());

        let removed1 = cache.remove(&"key".to_string());
        let removed2 = cache.remove(&"key".to_string());

        assert_eq!(removed1, Some("value".to_string()));
        assert_eq!(removed2, None);
    }

    #[test]
    fn test_cache_invalidate_nonexistent_tag() {
        let mut cache: DataCache<String, String> = DataCache::default();
        cache.insert_default("key".to_string(), "value".to_string());

        let count = cache.invalidate_tag("nonexistent");
        assert_eq!(count, 0);
        assert!(cache.contains(&"key".to_string()));
    }

    #[test]
    fn test_cache_clear_empty() {
        let mut cache: DataCache<String, String> = DataCache::default();
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_memory_accounting() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key1".to_string(), "12345".to_string()); // 5 bytes
        assert_eq!(cache.memory_usage(), 5);

        cache.insert_default("key2".to_string(), "12345678".to_string()); // 8 bytes
        assert_eq!(cache.memory_usage(), 13);

        cache.remove(&"key1".to_string());
        assert_eq!(cache.memory_usage(), 8);

        cache.clear();
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_cache_replace_updates_memory() {
        let mut cache: DataCache<String, String> = DataCache::default();

        cache.insert_default("key".to_string(), "12345".to_string()); // 5 bytes
        assert_eq!(cache.memory_usage(), 5);

        cache.insert_default("key".to_string(), "12345678901234567890".to_string()); // 20 bytes
        assert_eq!(cache.memory_usage(), 20);
    }

    #[test]
    fn test_cache_lru_order_updates_on_get() {
        let config = CacheConfig {
            max_entries: 2,
            enable_lru: true,
            ..Default::default()
        };
        let mut cache: DataCache<String, i32> = DataCache::new(config);

        cache.insert_default("key1".to_string(), 1);
        cache.insert_default("key2".to_string(), 2);

        // Access key1 to move it to end of LRU
        cache.get(&"key1".to_string());

        // Insert key3, should evict key2 (least recently used)
        cache.insert_default("key3".to_string(), 3);

        assert!(cache.contains(&"key1".to_string()));
        assert!(!cache.contains(&"key2".to_string()));
        assert!(cache.contains(&"key3".to_string()));
    }

    #[test]
    fn test_cache_access_count_increments() {
        let config = CacheConfig {
            default_ttl_ms: 10000,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);
        cache.insert_default("key".to_string(), "value".to_string());

        for _ in 0..10 {
            cache.get(&"key".to_string());
        }

        assert_eq!(cache.stats().hits, 10);
    }

    #[test]
    fn test_cache_cleanup_triggered_by_tick() {
        let config = CacheConfig {
            cleanup_interval_ms: 50,
            default_ttl_ms: 25,
            default_stale_ms: 0,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        cache.insert_default("key".to_string(), "value".to_string());
        assert_eq!(cache.len(), 1);

        // Tick past TTL but not cleanup interval
        cache.tick(30);
        // Entry exists but expired
        assert!(cache.get(&"key".to_string()).is_none());
        // Entry still in storage until cleanup
        assert_eq!(cache.entries.len(), 1);

        // Tick past cleanup interval
        cache.tick(30);
        // Now entry should be cleaned up
        assert_eq!(cache.entries.len(), 0);
    }

    #[test]
    fn test_cache_multiple_listeners() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut cache: DataCache<String, String> = DataCache::default();
        let count1 = Arc::new(AtomicUsize::new(0));
        let count2 = Arc::new(AtomicUsize::new(0));

        let c1 = count1.clone();
        cache.on_event(Arc::new(move |_| {
            c1.fetch_add(1, Ordering::SeqCst);
        }));

        let c2 = count2.clone();
        cache.on_event(Arc::new(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        }));

        cache.clear();

        assert_eq!(count1.load(Ordering::SeqCst), 1);
        assert_eq!(count2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_cache_eviction_updates_stats() {
        let config = CacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache: DataCache<String, i32> = DataCache::new(config);

        cache.insert_default("key1".to_string(), 1);
        cache.insert_default("key2".to_string(), 2);
        cache.insert_default("key3".to_string(), 3);

        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_contains_expired_entry() {
        let config = CacheConfig {
            default_ttl_ms: 50,
            default_stale_ms: 0,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        cache.insert_default("key".to_string(), "value".to_string());
        assert!(cache.contains(&"key".to_string()));

        cache.set_timestamp(100);
        assert!(!cache.contains(&"key".to_string()));
    }

    #[test]
    fn test_cache_stats_current_entries() {
        let mut cache: DataCache<String, i32> = DataCache::default();

        cache.insert_default("k1".to_string(), 1);
        assert_eq!(cache.stats().current_entries, 1);

        cache.insert_default("k2".to_string(), 2);
        assert_eq!(cache.stats().current_entries, 2);

        cache.remove(&"k1".to_string());
        assert_eq!(cache.stats().current_entries, 1);
    }

    #[test]
    fn test_cache_integer_keys() {
        let mut cache: DataCache<u64, String> = DataCache::default();

        cache.insert_default(1, "one".to_string());
        cache.insert_default(2, "two".to_string());

        assert_eq!(cache.get(&1), Some(&"one".to_string()));
        assert_eq!(cache.get(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_cache_with_custom_stale() {
        let config = CacheConfig {
            default_ttl_ms: 1000,
            default_stale_ms: 500,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        let options = CacheOptions::new().with_stale(Duration::from_millis(100));
        cache.insert("key".to_string(), "value".to_string(), options);

        // At ttl+50, should be stale (custom stale is 100)
        cache.set_timestamp(1050);
        let result = cache.get_with_state(&"key".to_string());
        assert!(result.is_some());
        let (_, state) = result.unwrap();
        assert_eq!(state, CacheState::Stale);

        // At ttl+150, should be expired
        cache.set_timestamp(1150);
        assert!(cache.get_with_state(&"key".to_string()).is_none());
    }

    // ========== Additional CacheStats tests ==========

    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let stats = CacheStats {
            hits: 100,
            misses: 0,
            ..CacheStats::default()
        };
        assert_eq!(stats.hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_misses() {
        let stats = CacheStats {
            hits: 0,
            misses: 100,
            ..CacheStats::default()
        };
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_half() {
        let stats = CacheStats {
            hits: 50,
            misses: 50,
            ..CacheStats::default()
        };
        assert!((stats.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_debug() {
        let stats = CacheStats::default();
        let debug = format!("{stats:?}");
        assert!(debug.contains("hits"));
        assert!(debug.contains("misses"));
    }

    #[test]
    fn test_cache_stats_clone() {
        let stats = CacheStats {
            hits: 42,
            evictions: 5,
            ..CacheStats::default()
        };
        let cloned = stats;
        assert_eq!(cloned.hits, 42);
        assert_eq!(cloned.evictions, 5);
    }

    // ========== Additional CacheSize tests ==========

    #[test]
    fn test_cache_size_unit() {
        let unit = ();
        assert_eq!(unit.cache_size(), 0);
    }

    #[test]
    fn test_cache_size_i64() {
        let n: i64 = 42;
        assert_eq!(n.cache_size(), 8);
    }

    #[test]
    fn test_cache_size_f32() {
        let n: f32 = 1.23;
        assert_eq!(n.cache_size(), 4);
    }

    #[test]
    fn test_cache_size_f64() {
        let n: f64 = 1.23456;
        assert_eq!(n.cache_size(), 8);
    }

    #[test]
    fn test_cache_size_box() {
        let b: Box<i32> = Box::new(42);
        assert_eq!(b.cache_size(), 4);
    }

    #[test]
    fn test_cache_size_empty_string() {
        let s = String::new();
        assert_eq!(s.cache_size(), 0);
    }

    #[test]
    fn test_cache_size_empty_vec() {
        let v: Vec<u8> = Vec::new();
        assert_eq!(v.cache_size(), 0);
    }

    #[test]
    fn test_cache_size_vec_of_structs() {
        #[derive(Clone)]
        struct Data {
            _a: i32,
            _b: i32,
        }
        let v: Vec<Data> = vec![Data { _a: 1, _b: 2 }, Data { _a: 3, _b: 4 }];
        // 2 * size_of::<Data>() = 2 * 8 = 16
        assert_eq!(v.cache_size(), 16);
    }

    // ========== Additional CacheBuilder tests ==========

    #[test]
    fn test_cache_builder_default_options() {
        let (value, options) = CacheBuilder::new(42i32).build();
        assert_eq!(value, 42);
        assert!(options.ttl.is_none());
        assert!(options.tags.is_empty());
    }

    #[test]
    fn test_cache_builder_multiple_tags() {
        let (_, options) = CacheBuilder::new("test".to_string())
            .tag("a")
            .tag("b")
            .tag("c")
            .build();
        assert_eq!(options.tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_cache_builder_chaining() {
        let (value, options) = CacheBuilder::new(vec![1, 2, 3])
            .ttl(Duration::from_secs(120))
            .stale(Duration::from_secs(60))
            .tag("numbers")
            .priority(10)
            .build();

        assert_eq!(value, vec![1, 2, 3]);
        assert_eq!(options.ttl, Some(Duration::from_secs(120)));
        assert_eq!(options.stale, Some(Duration::from_secs(60)));
        assert_eq!(options.tags, vec!["numbers"]);
        assert_eq!(options.priority, 10);
    }

    #[test]
    fn test_cache_builder_with_cache() {
        let mut cache: DataCache<String, String> = DataCache::default();
        let (value, options) = CacheBuilder::new("cached_value".to_string())
            .ttl(Duration::from_secs(300))
            .tag("test")
            .build();

        cache.insert("key".to_string(), value, options);
        assert!(cache.contains(&"key".to_string()));
    }

    // ========== Edge case and stress tests ==========

    #[test]
    fn test_cache_rapid_insert_remove() {
        let mut cache: DataCache<i32, i32> = DataCache::default();

        for i in 0..1000 {
            cache.insert_default(i, i);
            if i % 2 == 0 {
                cache.remove(&i);
            }
        }

        assert_eq!(cache.len(), 500);
    }

    #[test]
    fn test_cache_same_key_multiple_times() {
        let mut cache: DataCache<String, i32> = DataCache::default();

        for i in 0..100 {
            cache.insert_default("key".to_string(), i);
        }

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"key".to_string()), Some(&99));
    }

    #[test]
    fn test_cache_evict_all() {
        let config = CacheConfig {
            max_entries: 5,
            ..Default::default()
        };
        let mut cache: DataCache<i32, i32> = DataCache::new(config);

        // Insert more than max_entries
        for i in 0..10 {
            cache.insert_default(i, i);
        }

        assert_eq!(cache.len(), 5);
        assert!(cache.stats().evictions >= 5);
    }

    #[test]
    fn test_cache_memory_eviction_large_item() {
        let config = CacheConfig {
            max_memory: 100,
            ..Default::default()
        };
        let mut cache: DataCache<String, String> = DataCache::new(config);

        // Insert item larger than max_memory
        cache.insert_default("key".to_string(), "a".repeat(200));

        // Should either not insert or evict everything
        assert!(cache.memory_usage() <= 200);
    }

    #[test]
    fn test_cache_get_updates_lru_order() {
        let config = CacheConfig {
            max_entries: 3,
            enable_lru: true,
            ..Default::default()
        };
        let mut cache: DataCache<String, i32> = DataCache::new(config);

        cache.insert_default("a".to_string(), 1);
        cache.insert_default("b".to_string(), 2);
        cache.insert_default("c".to_string(), 3);

        // Access a, then b
        cache.get(&"a".to_string());
        cache.get(&"b".to_string());

        // Insert d, should evict c (least recently used)
        cache.insert_default("d".to_string(), 4);

        assert!(cache.contains(&"a".to_string()));
        assert!(cache.contains(&"b".to_string()));
        assert!(!cache.contains(&"c".to_string()));
        assert!(cache.contains(&"d".to_string()));
    }

    #[test]
    fn test_cache_invalidate_multiple_tags_same_entry() {
        let mut cache: DataCache<String, String> = DataCache::default();

        let options = CacheOptions::new().with_tag("tag1").with_tag("tag2");
        cache.insert("key".to_string(), "value".to_string(), options);

        // Invalidate by first tag
        let count1 = cache.invalidate_tag("tag1");
        assert_eq!(count1, 1);

        // Second invalidation should find nothing
        let count2 = cache.invalidate_tag("tag2");
        assert_eq!(count2, 0);
    }

    #[test]
    fn test_cache_tick_zero() {
        let mut cache: DataCache<String, String> = DataCache::default();
        cache.tick(0);
        assert_eq!(cache.timestamp(), 0);
    }

    #[test]
    fn test_cache_tick_large_values() {
        let mut cache: DataCache<String, String> = DataCache::default();
        cache.set_timestamp(u64::MAX / 2);
        cache.tick(1000);
        assert_eq!(cache.timestamp(), u64::MAX / 2 + 1000);
    }

    #[test]
    fn test_cache_remove_nonexistent() {
        let mut cache: DataCache<String, String> = DataCache::default();
        let result = cache.remove(&"nonexistent".to_string());
        assert!(result.is_none());
    }

    #[test]
    fn test_string_cache_type_alias() {
        let mut cache: StringCache<i32> = StringCache::default();
        cache.insert_default("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(&42));
    }
}
