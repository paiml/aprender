
#[cfg(test)]
mod tests {
    use super::*;

    type TestModelResult = Result<(Model, BPETokenizer), RealizarError>;

    fn create_test_model(vocab_size: usize) -> TestModelResult {
        let config = ModelConfig {
            vocab_size,
            hidden_dim: 16,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 32,
            eps: 1e-5,
        };
        let model = Model::new(config)?;

        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        Ok((model, tokenizer))
    }

    #[test]
    fn test_cache_key_creation() {
        let key = CacheKey::new("model1".to_string());
        assert_eq!(key.id, "model1");
    }

    #[test]
    fn test_cache_key_from_config() {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 2,
            num_layers: 4,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let key = CacheKey::from_config(&config);
        assert_eq!(key.id, "v100_h32_n2_l4_i64");
    }

    #[test]
    fn test_cache_creation() {
        let cache = ModelCache::new(10);
        assert_eq!(cache.capacity, 10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit() {
        let cache = ModelCache::new(10);
        let key = CacheKey::new("test".to_string());

        // First access - cache miss
        let result1 = cache.get_or_load(&key, || create_test_model(50));
        assert!(result1.is_ok());

        let metrics1 = cache.metrics();
        assert_eq!(metrics1.hits, 0);
        assert_eq!(metrics1.misses, 1);
        assert_eq!(metrics1.size, 1);

        // Second access - cache hit
        let result2 = cache.get_or_load(&key, || create_test_model(50));
        assert!(result2.is_ok());

        let metrics2 = cache.metrics();
        assert_eq!(metrics2.hits, 1);
        assert_eq!(metrics2.misses, 1);
        assert_eq!(metrics2.size, 1);
    }

    #[test]
    fn test_cache_miss() {
        let cache = ModelCache::new(10);
        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());

        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 2);
        assert_eq!(metrics.size, 2);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = ModelCache::new(2); // Small capacity

        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());
        let key3 = CacheKey::new("model3".to_string());

        // Fill cache to capacity
        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");

        assert_eq!(cache.len(), 2);

        // Add third model - should evict LRU (model1)
        cache
            .get_or_load(&key3, || create_test_model(70))
            .expect("test");

        assert_eq!(cache.len(), 2);
        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 1);
    }

    #[test]
    fn test_cache_clear() {
        let cache = ModelCache::new(10);

        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());

        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut metrics = CacheMetrics::default();
        assert!(metrics.hit_rate().abs() < 0.1);

        metrics.hits = 7;
        metrics.misses = 3;
        assert!((metrics.hit_rate() - 70.0).abs() < 0.1);

        metrics.hits = 100;
        metrics.misses = 0;
        assert!((metrics.hit_rate() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_cache_entry_access_tracking() {
        let (model, tokenizer) = create_test_model(50).expect("test");
        let mut entry = CacheEntry::new(model, tokenizer);

        assert_eq!(entry.access_count, 0);

        entry.record_access();
        assert_eq!(entry.access_count, 1);

        entry.record_access();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let cache = Arc::new(ModelCache::new(10));
        let key = CacheKey::new("concurrent".to_string());

        // Pre-populate cache to avoid race condition on first load
        cache
            .get_or_load(&key, || create_test_model(50))
            .expect("test");

        // Reset metrics to get clean counts
        let initial_metrics = cache.metrics();
        let initial_hits = initial_metrics.hits;
        let initial_misses = initial_metrics.misses;

        // Spawn multiple threads accessing the same key (all should hit)
        let handles: Vec<_> = (0..5)
            .map(|_| {
                let cache = cache.clone();
                let key = key.clone();
                thread::spawn(move || {
                    cache
                        .get_or_load(&key, || create_test_model(50))
                        .expect("test");
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("test");
        }

        // All threads should have hit the cache
        let metrics = cache.metrics();
        assert_eq!(metrics.size, 1);
        assert_eq!(metrics.hits, initial_hits + 5); // Exactly 5 hits
        assert_eq!(metrics.misses, initial_misses); // No new misses
    }

    // ==========================================================================
    // Additional T-COV-95 Tests
    // ==========================================================================

    #[test]
    fn test_cache_key_equality() {
        let key1 = CacheKey::new("same".to_string());
        let key2 = CacheKey::new("same".to_string());
        let key3 = CacheKey::new("different".to_string());

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_hash() {
        use std::collections::HashSet;

        let key1 = CacheKey::new("model_a".to_string());
        let key2 = CacheKey::new("model_a".to_string());
        let key3 = CacheKey::new("model_b".to_string());

        let mut set = HashSet::new();
        set.insert(key1.clone());
        set.insert(key2.clone()); // Same as key1, should not increase size
        set.insert(key3.clone());

        assert_eq!(set.len(), 2); // Only 2 unique keys
        assert!(set.contains(&key1));
        assert!(set.contains(&key3));
    }

    #[test]
    fn test_cache_key_debug_clone() {
        let key = CacheKey::new("test_model".to_string());
        let cloned = key.clone();
        let debug_str = format!("{:?}", key);

        assert_eq!(key, cloned);
        assert!(debug_str.contains("test_model"));
    }

    #[test]
    fn test_cache_metrics_clone_debug() {
        let metrics = CacheMetrics {
            hits: 10,
            misses: 5,
            evictions: 2,
            size: 3,
        };
        let cloned = metrics.clone();
        let debug_str = format!("{:?}", metrics);

        assert_eq!(cloned.hits, 10);
        assert_eq!(cloned.misses, 5);
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_cache_metrics_default() {
        let metrics = CacheMetrics::default();

        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.evictions, 0);
        assert_eq!(metrics.size, 0);
    }

    #[test]
    fn test_hit_rate_zero_total() {
        let metrics = CacheMetrics::default();
        // When both hits and misses are 0, hit_rate should be 0.0
        assert!((metrics.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_all_misses() {
        let metrics = CacheMetrics {
            hits: 0,
            misses: 100,
            evictions: 0,
            size: 0,
        };
        assert!((metrics.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_capacity_one() {
        let cache = ModelCache::new(1);
        let key1 = CacheKey::new("model1".to_string());
        let key2 = CacheKey::new("model2".to_string());

        cache
            .get_or_load(&key1, || create_test_model(50))
            .expect("test");
        assert_eq!(cache.len(), 1);

        // Adding second model should evict first
        cache
            .get_or_load(&key2, || create_test_model(60))
            .expect("test");
        assert_eq!(cache.len(), 1);

        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 1);
    }

    #[test]
    fn test_cache_loader_error_propagation() {
        let cache = ModelCache::new(10);
        let key = CacheKey::new("error_model".to_string());

        // Loader that returns an error
        let result = cache.get_or_load(&key, || {
            Err(RealizarError::InvalidConfiguration(
                "test error".to_string(),
            ))
        });

        assert!(result.is_err());
        // Cache should not contain the failed entry
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_multiple_evictions() {
        let cache = ModelCache::new(2);

        // Fill beyond capacity multiple times
        for i in 0..5 {
            let key = CacheKey::new(format!("model{i}"));
            cache
                .get_or_load(&key, || create_test_model(50 + i))
                .expect("test");
        }

        // Should have exactly 2 entries (capacity) and 3 evictions
        assert_eq!(cache.len(), 2);
        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 3); // 5 loads - 2 capacity = 3 evictions
    }

    #[test]
    fn test_cache_clear_resets_len() {
        let cache = ModelCache::new(10);

        for i in 0..5 {
            let key = CacheKey::new(format!("model{i}"));
            cache
                .get_or_load(&key, || create_test_model(50))
                .expect("test");
        }

        assert_eq!(cache.len(), 5);
        assert!(!cache.is_empty());

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
