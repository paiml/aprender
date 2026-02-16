pub(crate) use super::*;

// Simple test generator
#[derive(Debug, Clone)]
pub(super) struct DoubleGen;

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
