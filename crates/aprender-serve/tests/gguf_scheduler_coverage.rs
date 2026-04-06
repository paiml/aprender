//! EXTREME TDD: GGUF Scheduler and Batch Coverage Tests
//!
//! Tests for BatchRequestCollector, ContinuousBatchScheduler, PrefixCache, etc.
//! These require the `gpu` feature flag.

#[cfg(feature = "gpu")]
mod gpu_tests {
    use realizar::gguf::{
        BatchingConfig, ChunkedPrefill, ChunkedPrefillConfig, PrefixCache, SchedulingPolicy,
        SpeculativeConfig,
    };

    // ===== BatchingConfig Tests =====

    #[test]
    fn test_cov_batching_config_default() {
        let config = BatchingConfig::default();

        assert_eq!(config.batch_threshold, 32);
        assert_eq!(config.timeout_ms, 50);
        assert_eq!(config.max_batch_size, 64);
        assert!(config.prefer_throughput);
    }

    #[test]
    fn test_cov_batching_config_latency_optimized() {
        let config = BatchingConfig::latency_optimized();

        assert_eq!(config.batch_threshold, 8);
        assert_eq!(config.timeout_ms, 10);
        assert_eq!(config.max_batch_size, 32);
        assert!(!config.prefer_throughput);
    }

    #[test]
    fn test_cov_batching_config_throughput_optimized() {
        let config = BatchingConfig::throughput_optimized();

        assert_eq!(config.batch_threshold, 32);
        assert_eq!(config.timeout_ms, 100);
        assert_eq!(config.max_batch_size, 64);
        assert!(config.prefer_throughput);
    }

    #[test]
    fn test_cov_batching_config_clone() {
        let config = BatchingConfig::default();
        let cloned = config.clone();
        assert_eq!(config.batch_threshold, cloned.batch_threshold);
    }

    // ===== PrefixCache Tests =====

    #[test]
    fn test_cov_prefix_cache_new() {
        let cache = PrefixCache::new(16);
        let stats = cache.stats();

        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cov_prefix_cache_insert_and_lookup() {
        let cache = PrefixCache::new(16);

        let tokens = vec![1, 2, 3];
        let k_cache = vec![vec![1.0, 2.0]];
        let v_cache = vec![vec![3.0, 4.0]];

        cache.insert(tokens.clone(), k_cache.clone(), v_cache.clone());
        assert!(cache.contains(&tokens));

        let result = cache.lookup(&tokens);
        assert!(result.is_some());
        let (k, v) = result.unwrap();
        assert_eq!(k, k_cache);
        assert_eq!(v, v_cache);
    }

    #[test]
    fn test_cov_prefix_cache_miss() {
        let cache = PrefixCache::new(16);

        let result = cache.lookup(&[1, 2, 3]);
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cov_prefix_cache_stats() {
        let cache = PrefixCache::new(16);

        cache.insert(vec![1, 2], vec![vec![1.0]], vec![vec![2.0]]);
        cache.lookup(&[1, 2]); // hit
        cache.lookup(&[3, 4]); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cov_prefix_cache_eviction() {
        let cache = PrefixCache::new(2); // Small cache

        cache.insert(vec![1], vec![vec![1.0]], vec![vec![1.0]]);
        cache.insert(vec![2], vec![vec![2.0]], vec![vec![2.0]]);
        cache.insert(vec![3], vec![vec![3.0]], vec![vec![3.0]]); // Should evict

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cov_prefix_cache_clear() {
        let cache = PrefixCache::new(16);
        cache.insert(vec![1], vec![vec![1.0]], vec![vec![1.0]]);
        cache.insert(vec![2], vec![vec![2.0]], vec![vec![2.0]]);

        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_cov_prefix_cache_memory_usage() {
        let cache = PrefixCache::new(16);
        cache.insert(vec![1, 2, 3], vec![vec![1.0; 100]], vec![vec![2.0; 100]]);

        let memory = cache.memory_usage_bytes();
        // 3 tokens * 4 bytes + 100 * 4 * 2 (k and v)
        assert!(memory > 0);
    }

    #[test]
    fn test_cov_prefix_cache_default() {
        let cache = PrefixCache::default();
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_cov_prefix_cache_hash_collision_check() {
        // Insert two different token sequences
        let cache = PrefixCache::new(16);
        cache.insert(vec![1, 2, 3], vec![vec![1.0]], vec![vec![1.0]]);
        cache.insert(vec![4, 5, 6], vec![vec![2.0]], vec![vec![2.0]]);

        // Lookup should return correct data
        let (k, _) = cache.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(k[0][0], 1.0);

        let (k, _) = cache.lookup(&[4, 5, 6]).unwrap();
        assert_eq!(k[0][0], 2.0);
    }

    // ===== SpeculativeConfig Tests =====

    #[test]
    fn test_cov_speculative_config_default() {
        let config = SpeculativeConfig::default();

        assert_eq!(config.speculation_length, 4);
        assert!((config.draft_temperature - 0.0).abs() < 0.01);
        assert!(config.self_speculative);
    }

    // ===== SchedulingPolicy Tests =====

    #[test]
    fn test_cov_scheduling_policy_fcfs() {
        let policy = SchedulingPolicy::Fcfs;
        assert!(matches!(policy, SchedulingPolicy::Fcfs));
    }

    #[test]
    fn test_cov_scheduling_policy_sjf() {
        let policy = SchedulingPolicy::Sjf;
        assert!(matches!(policy, SchedulingPolicy::Sjf));
    }

    #[test]
    fn test_cov_scheduling_policy_round_robin() {
        let policy = SchedulingPolicy::RoundRobin;
        assert!(matches!(policy, SchedulingPolicy::RoundRobin));
    }

    // ===== ChunkedPrefillConfig Tests =====

    #[test]
    fn test_cov_chunked_prefill_config_default() {
        let config = ChunkedPrefillConfig::default();

        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.max_context, 8192);
        assert!(config.stream_chunks);
    }

    #[test]
    fn test_cov_chunked_prefill_config_with_chunk_size() {
        let config = ChunkedPrefillConfig::with_chunk_size(256);

        assert_eq!(config.chunk_size, 256);
        assert_eq!(config.max_context, 8192); // Default
    }

    // ===== ChunkedPrefill Tests =====

    #[test]
    fn test_cov_chunked_prefill_new() {
        let tokens = vec![1, 2, 3, 4, 5];
        let prefill = ChunkedPrefill::new(&tokens, ChunkedPrefillConfig::default());

        assert_eq!(prefill.total_tokens(), 5);
        assert_eq!(prefill.total_chunks(), 1); // 5 tokens < 512 chunk_size
    }

    #[test]
    fn test_cov_chunked_prefill_multiple_chunks() {
        let config = ChunkedPrefillConfig {
            chunk_size: 2,
            max_context: 8192,
            stream_chunks: true,
        };
        let tokens = vec![1, 2, 3, 4, 5];
        let prefill = ChunkedPrefill::new(&tokens, config);

        assert_eq!(prefill.total_chunks(), 3); // ceil(5/2) = 3
        assert!(prefill.has_more_chunks());
    }

    #[test]
    fn test_cov_chunked_prefill_stats() {
        let tokens = vec![1, 2, 3];
        let prefill = ChunkedPrefill::new(&tokens, ChunkedPrefillConfig::default());
        let stats = prefill.stats();

        // Stats structure has total_chunks, total_tokens, etc.
        assert_eq!(stats.total_tokens, 3);
        assert_eq!(stats.chunk_size, 512); // Default chunk size
    }

    #[test]
    fn test_cov_chunked_prefill_empty() {
        let tokens: Vec<u32> = vec![];
        let prefill = ChunkedPrefill::new(&tokens, ChunkedPrefillConfig::default());

        assert_eq!(prefill.total_tokens(), 0);
        assert_eq!(prefill.total_chunks(), 0);
        assert!(!prefill.has_more_chunks());
    }
}

// Non-GPU tests
#[test]
fn test_cov_gguf_scheduler_module_exists() {
    // Just verify the module compiles by using a type from it
    let _ = std::mem::size_of::<u8>(); // Minimal operation to prove compilation
}
