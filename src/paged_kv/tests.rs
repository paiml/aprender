#[cfg(test)]
mod tests {
    use crate::paged_kv::*;

    // --- Test SeqId/PageId equality and hashing ---

    #[test]
    fn test_cov_seq_id_equality() {
        let id1 = SeqId::new();
        let id2 = SeqId::new();

        // Different IDs should not be equal
        assert_ne!(id1, id2);

        // Same ID should equal itself
        let id1_clone = id1;
        assert_eq!(id1, id1_clone);
    }

    #[test]
    fn test_cov_page_id_equality() {
        let id1 = PageId::new(42);
        let id2 = PageId::new(42);
        let id3 = PageId::new(43);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // --- Test KvPage state mutations ---

    #[test]
    fn test_cov_kv_page_full_and_capacity() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);

        // Test at various fill levels
        for tokens in 0..=20 {
            page.num_tokens = tokens;
            let is_full = page.is_full(16);
            let remaining = page.remaining_capacity(16);

            if tokens >= 16 {
                assert!(is_full);
                assert_eq!(remaining, 0);
            } else {
                assert!(!is_full);
                assert_eq!(remaining, 16 - tokens);
            }
        }
    }

    // --- Test PrefixCacheStats computed values ---

    #[test]
    fn test_cov_prefix_cache_stats_various_hit_rates() {
        let stats_100 = PrefixCacheStats {
            hits: 100,
            misses: 0,
            prefixes_cached: 10,
            prefixes_evicted: 0,
            tokens_saved: 1000,
        };
        assert!((stats_100.hit_rate() - 1.0).abs() < 0.001);

        let stats_50 = PrefixCacheStats {
            hits: 50,
            misses: 50,
            prefixes_cached: 5,
            prefixes_evicted: 1,
            tokens_saved: 500,
        };
        assert!((stats_50.hit_rate() - 0.5).abs() < 0.001);

        let stats_0 = PrefixCacheStats {
            hits: 0,
            misses: 100,
            prefixes_cached: 0,
            prefixes_evicted: 0,
            tokens_saved: 0,
        };
        assert!((stats_0.hit_rate() - 0.0).abs() < 0.001);
    }

    // --- Test extend with exactly capacity ---

    #[test]
    fn test_cov_extend_fills_exactly_to_capacity() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq = cache.allocate_sequence(16).expect("alloc"); // 1 page, 16 capacity

        // No tokens used yet
        // Extend by 16 - fits in existing capacity
        cache.extend(seq, 16).expect("extend");

        // Should still be 1 page since capacity wasn't exceeded
        assert_eq!(cache.free_page_count(), 99);
    }

    // --- Test QuantizedKvData memory calculation accuracy ---

    #[test]
    fn test_cov_quantized_kv_data_memory_bytes_accuracy() {
        let block_size = 16;
        let num_heads = 8;
        let head_dim = 64;
        let total_elements = block_size * num_heads * head_dim; // 8192

        let data_fp32 = QuantizedKvData::new(KvQuantType::FP32, block_size, num_heads, head_dim);
        assert_eq!(data_fp32.memory_bytes(), total_elements * 4 * 2); // f32 * 2 (K+V)

        let data_q8 = QuantizedKvData::new(KvQuantType::Q8, block_size, num_heads, head_dim);
        // Q8: (scale + quants) * num_blocks * 2
        assert!(data_q8.memory_bytes() < data_fp32.memory_bytes());

        let data_q4 = QuantizedKvData::new(KvQuantType::Q4, block_size, num_heads, head_dim);
        assert!(data_q4.memory_bytes() < data_q8.memory_bytes());
    }

    // --- Test free_sequence updates stats correctly ---

    #[test]
    fn test_cov_free_sequence_stats_update() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        let seq1 = cache.allocate_sequence(32).expect("s1");
        let seq2 = cache.allocate_sequence(16).expect("s2");

        assert_eq!(cache.stats().sequences_allocated, 2);
        assert_eq!(cache.stats().active_sequences, 2);
        assert_eq!(cache.stats().pages_allocated, 3);

        cache.free_sequence(seq1);
        assert_eq!(cache.stats().sequences_freed, 1);
        assert_eq!(cache.stats().active_sequences, 1);
        assert_eq!(cache.stats().pages_freed, 2);

        cache.free_sequence(seq2);
        assert_eq!(cache.stats().sequences_freed, 2);
        assert_eq!(cache.stats().active_sequences, 0);
        assert_eq!(cache.stats().pages_freed, 3);
    }

    // --- Test QuantizedPagedKvCache free stats ---

    #[test]
    fn test_cov_quantized_free_sequence_stats() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q4);

        let seq = cache.allocate_sequence(32).expect("alloc");
        assert_eq!(cache.stats().active_sequences, 1);
        assert_eq!(cache.stats().used_pages, 2);

        cache.free_sequence(seq);
        assert_eq!(cache.stats().active_sequences, 0);
        assert_eq!(cache.stats().used_pages, 0);
        assert_eq!(cache.stats().pages_freed, 2);
    }
include!("tests_seq.rs");
include!("tests_fragmentation_stats.rs");
include!("tests_quantized.rs");
include!("tests_deep_pkcov.rs");
include!("tests_cov_fragmentation_should.rs");
}
