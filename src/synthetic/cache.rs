//! Caching for Synthetic Data Generation.
//!
//! Provides memoization of generated synthetic samples to avoid
//! redundant computation during `AutoML` hyperparameter search.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::mem::{size_of, size_of_val};

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;

// ============================================================================
// Cache Key
// ============================================================================

/// Key for cache lookups based on seeds and config.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    /// Hash of seed data.
    seed_hash: u64,
    /// Hash of configuration.
    config_hash: u64,
}

impl CacheKey {
    /// Create a new cache key.
    fn new(seed_hash: u64, config_hash: u64) -> Self {
        Self {
            seed_hash,
            config_hash,
        }
    }
}

// ============================================================================
// Cache Entry
// ============================================================================

/// A cached generation result.
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    /// Generated samples.
    samples: Vec<T>,
    /// Approximate size in bytes.
    size_bytes: usize,
    /// Number of times accessed.
    access_count: usize,
}

impl<T> CacheEntry<T> {
    fn new(samples: Vec<T>, size_bytes: usize) -> Self {
        Self {
            samples,
            size_bytes,
            access_count: 0,
        }
    }
}

// ============================================================================
// Synthetic Cache
// ============================================================================

/// Cache for synthetic data generation results.
///
/// Implements LRU (Least Recently Used) eviction when the cache
/// exceeds the maximum size.
///
/// # Example
///
/// ```
/// use aprender::synthetic::cache::SyntheticCache;
/// use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};
/// use aprender::error::Result;
///
/// #[derive(Clone, Debug)]
/// struct DoubleGen;
///
/// impl SyntheticGenerator for DoubleGen {
///     type Input = i32;
///     type Output = i32;
///
///     fn generate(&self, seeds: &[i32], _: &SyntheticConfig) -> Result<Vec<i32>> {
///         Ok(seeds.iter().map(|x| x * 2).collect())
///     }
///     fn quality_score(&self, _: &i32, _: &i32) -> f32 { 1.0 }
///     fn diversity_score(&self, _: &[i32]) -> f32 { 1.0 }
/// }
///
/// let mut cache = SyntheticCache::<i32>::new(1000);
/// let gen = DoubleGen;
/// let seeds = vec![1, 2, 3];
/// let config = SyntheticConfig::default();
///
/// // First call generates and caches
/// let result1 = cache.get_or_generate(&seeds, &config, &gen).expect("cache get_or_generate should succeed");
/// assert_eq!(result1, &[2, 4, 6]);
///
/// // Second call returns cached result
/// let result2 = cache.get_or_generate(&seeds, &config, &gen).expect("cached result should be available");
/// assert_eq!(result1, result2);
/// ```
#[derive(Debug)]
pub struct SyntheticCache<T> {
    /// Cached entries.
    cache: HashMap<CacheKey, CacheEntry<T>>,
    /// LRU order (front = least recent, back = most recent).
    lru_order: VecDeque<CacheKey>,
    /// Maximum cache size in bytes.
    max_size: usize,
    /// Current cache size in bytes.
    current_size: usize,
    /// Cache statistics.
    stats: CacheStats,
}

/// Cache statistics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of evictions.
    pub evictions: usize,
    /// Total generations performed.
    pub generations: usize,
}

impl CacheStats {
    /// Get the hit rate.
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

impl<T: Clone> SyntheticCache<T> {
    /// Create a new cache with the specified maximum size in bytes.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            max_size: max_size.max(1),
            current_size: 0,
            stats: CacheStats::default(),
        }
    }

    /// Get or generate synthetic samples.
    ///
    /// Returns cached results if available, otherwise generates and caches.
    pub fn get_or_generate<G>(
        &mut self,
        seeds: &[G::Input],
        config: &SyntheticConfig,
        generator: &G,
    ) -> Result<Vec<T>>
    where
        G: SyntheticGenerator<Output = T>,
        G::Input: Hash,
    {
        let key = Self::compute_key(seeds, config);

        // Check cache
        if self.cache.contains_key(&key) {
            self.update_lru(&key);
            self.stats.hits += 1;
            let entry = self.cache.get_mut(&key).expect("key should exist");
            entry.access_count += 1;
            return Ok(entry.samples.clone());
        }

        // Generate new samples
        self.stats.misses += 1;
        self.stats.generations += 1;

        let samples = generator.generate(seeds, config)?;

        // Estimate size (rough approximation)
        let size_bytes = Self::estimate_size(&samples);

        // Evict if necessary
        self.evict_until_fits(size_bytes);

        // Insert into cache
        self.insert(key.clone(), samples.clone(), size_bytes);

        Ok(samples)
    }

    /// Check if a key is in the cache.
    #[must_use]
    pub fn contains<I: Hash>(&self, seeds: &[I], config: &SyntheticConfig) -> bool {
        let key = Self::compute_key(seeds, config);
        self.cache.contains_key(&key)
    }

    /// Get cached samples without generating.
    #[must_use]
    pub fn get<I: Hash>(&mut self, seeds: &[I], config: &SyntheticConfig) -> Option<Vec<T>> {
        let key = Self::compute_key(seeds, config);

        if self.cache.contains_key(&key) {
            self.update_lru(&key);
            self.stats.hits += 1;
            let entry = self.cache.get_mut(&key).expect("key should exist");
            entry.access_count += 1;
            Some(entry.samples.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get cache statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get current cache size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.current_size
    }

    /// Get number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.current_size = 0;
    }

    /// Compute cache key from seeds and config.
    fn compute_key<I: Hash>(seeds: &[I], config: &SyntheticConfig) -> CacheKey {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash seeds
        seeds.hash(&mut hasher);
        let seed_hash = hasher.finish();

        // Hash config (relevant fields)
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        config.augmentation_ratio.to_bits().hash(&mut hasher);
        config.quality_threshold.to_bits().hash(&mut hasher);
        config.diversity_weight.to_bits().hash(&mut hasher);
        config.seed.hash(&mut hasher);
        let config_hash = hasher.finish();

        CacheKey::new(seed_hash, config_hash)
    }

    /// Estimate memory size of samples.
    fn estimate_size(samples: &[T]) -> usize {
        // Rough estimate: size_of_val gives the data size, plus overhead for Vec
        size_of_val(samples) + size_of::<Vec<T>>() + 64
    }

    /// Insert into cache.
    fn insert(&mut self, key: CacheKey, samples: Vec<T>, size_bytes: usize) {
        let entry = CacheEntry::new(samples, size_bytes);
        self.cache.insert(key.clone(), entry);
        self.lru_order.push_back(key);
        self.current_size += size_bytes;
    }

    /// Update LRU order (move to back).
    fn update_lru(&mut self, key: &CacheKey) {
        // Remove from current position
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        // Add to back (most recent)
        self.lru_order.push_back(key.clone());
    }

    /// Evict entries until we have room for `new_size`.
    fn evict_until_fits(&mut self, new_size: usize) {
        while self.current_size + new_size > self.max_size && !self.lru_order.is_empty() {
            self.evict_lru();
        }
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some(key) = self.lru_order.pop_front() {
            if let Some(entry) = self.cache.remove(&key) {
                self.current_size = self.current_size.saturating_sub(entry.size_bytes);
                self.stats.evictions += 1;
            }
        }
    }
}

impl<T: Clone> Default for SyntheticCache<T> {
    fn default() -> Self {
        Self::new(10 * 1024 * 1024) // 10 MB default
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test generator
    #[derive(Debug, Clone)]
    struct DoubleGen;

    impl SyntheticGenerator for DoubleGen {
        type Input = i32;
        type Output = i32;

        fn generate(&self, seeds: &[i32], _config: &SyntheticConfig) -> Result<Vec<i32>> {
            Ok(seeds.iter().map(|x| x * 2).collect())
        }

        fn quality_score(&self, _: &i32, _: &i32) -> f32 {
            1.0
        }

        fn diversity_score(&self, _: &[i32]) -> f32 {
            1.0
        }
    }

    // ========================================================================
    // CacheStats Tests
    // ========================================================================

    #[test]
    fn test_cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.generations, 0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut stats = CacheStats::default();

        // No requests yet
        assert!((stats.hit_rate() - 0.0).abs() < f32::EPSILON);

        // 50% hit rate
        stats.hits = 5;
        stats.misses = 5;
        assert!((stats.hit_rate() - 0.5).abs() < f32::EPSILON);

        // 100% hit rate
        stats.misses = 0;
        assert!((stats.hit_rate() - 1.0).abs() < f32::EPSILON);
    }

    // ========================================================================
    // SyntheticCache Tests
    // ========================================================================

    #[test]
    fn test_cache_new() {
        let cache = SyntheticCache::<i32>::new(1000);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_new_min_size() {
        let cache = SyntheticCache::<i32>::new(0);
        assert_eq!(cache.max_size, 1); // Should be at least 1
    }

    #[test]
    fn test_cache_default() {
        let cache = SyntheticCache::<i32>::default();
        assert_eq!(cache.max_size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_cache_get_or_generate_first_call() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        let result = cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("generation failed");

        assert_eq!(result, vec![2, 4, 6]);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().generations, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_get_or_generate_cached() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        // First call - generates
        let result1 = cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("generation failed");

        // Second call - cached
        let result2 = cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("generation failed");

        assert_eq!(result1, result2);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().generations, 1); // Only one generation
    }

    #[test]
    fn test_cache_different_seeds() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let config = SyntheticConfig::default();

        let seeds1 = vec![1, 2, 3];
        let seeds2 = vec![4, 5, 6];

        let result1 = cache
            .get_or_generate(&seeds1, &config, &gen)
            .expect("failed");
        let result2 = cache
            .get_or_generate(&seeds2, &config, &gen)
            .expect("failed");

        assert_eq!(result1, vec![2, 4, 6]);
        assert_eq!(result2, vec![8, 10, 12]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().generations, 2);
    }

    #[test]
    fn test_cache_different_config() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];

        let config1 = SyntheticConfig::default().with_seed(42);
        let config2 = SyntheticConfig::default().with_seed(99);

        cache
            .get_or_generate(&seeds, &config1, &gen)
            .expect("failed");
        cache
            .get_or_generate(&seeds, &config2, &gen)
            .expect("failed");

        assert_eq!(cache.len(), 2); // Different configs = different entries
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        assert!(!cache.contains(&seeds, &config));

        cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");

        assert!(cache.contains(&seeds, &config));
    }

    #[test]
    fn test_cache_get() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        // Get before insert
        assert!(cache.get(&seeds, &config).is_none());

        // Generate
        cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");

        // Get after insert
        let result = cache.get(&seeds, &config);
        assert!(result.is_some());
        assert_eq!(result.expect("should have value"), vec![2, 4, 6]);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_lru_eviction() {
        // Very small cache that can only hold ~1 entry
        let mut cache = SyntheticCache::<i32>::new(100);
        let gen = DoubleGen;
        let config = SyntheticConfig::default();

        let seeds1 = vec![1];
        let seeds2 = vec![2];
        let seeds3 = vec![3];

        // Add entries that exceed cache size
        cache
            .get_or_generate(&seeds1, &config, &gen)
            .expect("failed");
        cache
            .get_or_generate(&seeds2, &config, &gen)
            .expect("failed");
        cache
            .get_or_generate(&seeds3, &config, &gen)
            .expect("failed");

        // Should have evicted some entries
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_cache_lru_access_order() {
        // Cache that can hold 2 entries
        let mut cache = SyntheticCache::<i32>::new(200);
        let gen = DoubleGen;
        let config = SyntheticConfig::default();

        let seeds1 = vec![1];
        let seeds2 = vec![2];
        let seeds3 = vec![3];

        // Add two entries
        cache
            .get_or_generate(&seeds1, &config, &gen)
            .expect("failed");
        cache
            .get_or_generate(&seeds2, &config, &gen)
            .expect("failed");

        // Access first entry (makes it most recent)
        cache
            .get_or_generate(&seeds1, &config, &gen)
            .expect("failed");

        // Add third entry - should evict seeds2 (least recent)
        cache
            .get_or_generate(&seeds3, &config, &gen)
            .expect("failed");

        // seeds1 should still be cached
        assert!(cache.contains(&seeds1, &config));
        // seeds3 should be cached
        assert!(cache.contains(&seeds3, &config));
    }

    #[test]
    fn test_cache_size_tracking() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let config = SyntheticConfig::default();

        assert_eq!(cache.size(), 0);

        cache
            .get_or_generate(&[1, 2, 3], &config, &gen)
            .expect("failed");

        assert!(cache.size() > 0);

        let size_after_one = cache.size();

        cache
            .get_or_generate(&[4, 5, 6], &config, &gen)
            .expect("failed");

        assert!(cache.size() > size_after_one);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        // First call - miss
        cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");
        assert!((cache.stats().hit_rate() - 0.0).abs() < f32::EPSILON);

        // Second call - hit
        cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");
        assert!((cache.stats().hit_rate() - 0.5).abs() < f32::EPSILON);

        // Third call - hit
        cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");
        // hits=2, misses=1 -> 2/3 = 0.666...
        assert!(cache.stats().hit_rate() > 0.6);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_cache_with_real_generator() {
        use super::super::eda::{EdaConfig, EdaGenerator};

        let mut cache = SyntheticCache::<String>::new(100_000);
        let gen = EdaGenerator::new(EdaConfig::default());
        let seeds = vec!["git status".to_string(), "cargo build".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.3);

        // First generation
        let result1 = cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("generation failed");

        // Cached retrieval
        let result2 = cache
            .get_or_generate(&seeds, &config, &gen)
            .expect("cached retrieval failed");

        assert_eq!(result1, result2);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().generations, 1);
    }

    #[test]
    fn test_cache_determinism() {
        let mut cache1 = SyntheticCache::<i32>::new(10000);
        let mut cache2 = SyntheticCache::<i32>::new(10000);
        let gen = DoubleGen;
        let seeds = vec![1, 2, 3];
        let config = SyntheticConfig::default();

        let result1 = cache1
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");
        let result2 = cache2
            .get_or_generate(&seeds, &config, &gen)
            .expect("failed");

        assert_eq!(result1, result2);
    }
}
