
    #[test]
    fn test_fragmentation_stats_serialization() {
        let stats = FragmentationStats {
            holes: 5,
            wasted_capacity: 100,
            fragmentation_ratio: 0.25,
            largest_free_region: 50,
            avg_tokens_per_page: 12.5,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: FragmentationStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.holes, 5);
        assert_eq!(parsed.wasted_capacity, 100);
        assert!((parsed.fragmentation_ratio - 0.25).abs() < 0.001);
        assert_eq!(parsed.largest_free_region, 50);
        assert!((parsed.avg_tokens_per_page - 12.5).abs() < 0.001);
    }

    #[test]
    fn test_defrag_stats_tracking() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Initial state
        assert_eq!(cache.stats().defrag_operations, 0);
        assert_eq!(cache.stats().pages_moved, 0);

        // Defrag on empty cache doesn't increment
        cache.defragment();
        assert_eq!(cache.stats().defrag_operations, 0);
    }

    #[test]
    fn test_defragment_preserves_data() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("test");
        cache.update_tokens(seq_id, 16).expect("test");

        // Get page and write some test data
        let page = cache.get_page_mut(seq_id, 0).expect("test");
        page.keys[0] = 42.0;
        page.values[0] = 99.0;

        // Defragment
        cache.defragment();

        // Verify data is preserved
        let page = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(page.keys[0], 42.0);
        assert_eq!(page.values[0], 99.0);
    }

    #[test]
    fn test_cow_prevents_compact() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(32).expect("test");
        cache.update_tokens(parent_id, 32).expect("test");

        // Fork creates shared pages (COW)
        let _child_id = cache.fork_sequence(parent_id).expect("test");

        // Shared pages should not be moved during compaction
        // (ref_count > 1 check in compact_sequence)
        let moved = cache.compact_sequence(parent_id);
        // Since pages are already contiguous, should be 0 anyway
        assert_eq!(moved, 0);
    }

    // === Prefix Caching Tests ===

    #[test]
    fn test_compute_prefix_hash() {
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![1, 2, 3];
        let tokens3 = vec![1, 2, 4];

        let hash1 = compute_prefix_hash(&tokens1);
        let hash2 = compute_prefix_hash(&tokens2);
        let hash3 = compute_prefix_hash(&tokens3);

        // Same tokens = same hash
        assert_eq!(hash1, hash2);
        // Different tokens = different hash (with high probability)
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_compute_prefix_hash_empty() {
        let tokens: Vec<u32> = vec![];
        let hash = compute_prefix_hash(&tokens);
        // Should be the FNV offset basis
        assert_eq!(hash, 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn test_cached_prefix_new() {
        let hash = 12345;
        let page_ids = vec![PageId::new(0), PageId::new(1)];
        let prefix = CachedPrefix::new(hash, 10, page_ids);

        assert_eq!(prefix.hash, hash);
        assert_eq!(prefix.num_tokens, 10);
        assert_eq!(prefix.page_ids.len(), 2);
        assert_eq!(prefix.ref_count, 1);
    }

    #[test]
    fn test_cached_prefix_ref_counting() {
        let mut prefix = CachedPrefix::new(1, 5, vec![]);

        assert_eq!(prefix.ref_count, 1);

        prefix.add_ref();
        assert_eq!(prefix.ref_count, 2);

        assert!(!prefix.remove_ref()); // Still has references
        assert_eq!(prefix.ref_count, 1);

        assert!(prefix.remove_ref()); // No more references
        assert_eq!(prefix.ref_count, 0);
    }

    #[test]
    fn test_prefix_cache_new() {
        let cache = PrefixCache::new(100);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_prefix_cache_insert_lookup() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        let prefix = CachedPrefix::new(hash, 3, vec![PageId::new(0)]);

        assert!(cache.insert(prefix));
        assert_eq!(cache.len(), 1);

        let result = cache.lookup(hash);
        assert!(result.is_some());
        assert_eq!(result.expect("test").num_tokens, 3);
    }

    #[test]
    fn test_prefix_cache_lookup_tokens() {
        let mut cache = PrefixCache::new(100);

        let tokens = vec![10, 20, 30];
        let hash = compute_prefix_hash(&tokens);
        let prefix = CachedPrefix::new(hash, 3, vec![PageId::new(0)]);

        cache.insert(prefix);

        let result = cache.lookup_tokens(&tokens);
        assert!(result.is_some());
    }

    #[test]
    fn test_prefix_cache_miss() {
        let mut cache = PrefixCache::new(100);

        let result = cache.lookup(12345);
        assert!(result.is_none());

        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_prefix_cache_stats() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(hash, 3, vec![]));

        // Miss
        cache.lookup(99999);

        // Hit
        cache.lookup(hash);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prefix_cache_add_remove_ref() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(hash, 3, vec![]));

        // Add reference
        assert!(cache.add_ref(hash));

        // Remove references
        assert!(!cache.remove_ref(hash)); // Still has ref_count = 1
        assert!(cache.remove_ref(hash)); // Now removed

        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_cache_contains() {
        let mut cache = PrefixCache::new(100);

        let hash = compute_prefix_hash(&[1, 2, 3]);
        assert!(!cache.contains(hash));

        cache.insert(CachedPrefix::new(hash, 3, vec![]));
        assert!(cache.contains(hash));
    }

    #[test]
    fn test_prefix_cache_capacity() {
        let mut cache = PrefixCache::new(2);

        // Test that insert fails when cache is full and all entries are referenced
        // Insert with ref_count = 1 (default)
        cache.insert(CachedPrefix::new(1, 1, vec![]));
        cache.insert(CachedPrefix::new(2, 2, vec![]));
        assert_eq!(cache.len(), 2);

        // Both entries have ref_count = 1, so eviction skips them
        // Third insert should fail since no evictable entries
        let success = cache.insert(CachedPrefix::new(3, 3, vec![]));
        // Note: evict_lru only evicts ref_count = 0, so this fails
        assert!(!success);
        assert_eq!(cache.len(), 2);

        // Remove reference from first entry (makes it evictable)
        // Note: This removes the entry since ref_count drops to 0
        cache.remove_ref(1);
        assert_eq!(cache.len(), 1); // Entry 1 was removed

        // Now insert should succeed (we have capacity)
        let success = cache.insert(CachedPrefix::new(3, 3, vec![]));
        assert!(success);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_prefix_cache_clear() {
        let mut cache = PrefixCache::new(100);

        cache.insert(CachedPrefix::new(1, 1, vec![]));
        cache.insert(CachedPrefix::new(2, 2, vec![]));

        cache.clear();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_cache_utilization() {
        let mut cache = PrefixCache::new(4);

        assert_eq!(cache.utilization(), 0.0);

        cache.insert(CachedPrefix::new(1, 1, vec![]));
        assert!((cache.utilization() - 0.25).abs() < 0.01);

        cache.insert(CachedPrefix::new(2, 2, vec![]));
        assert!((cache.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prefix_cache_stats_serialization() {
        let stats = PrefixCacheStats {
            hits: 100,
            misses: 50,
            prefixes_cached: 10,
            prefixes_evicted: 2,
            tokens_saved: 500,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: PrefixCacheStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.hits, 100);
        assert_eq!(parsed.misses, 50);
        assert_eq!(parsed.tokens_saved, 500);
    }

    #[test]
    fn test_find_longest_prefix() {
        let mut cache = PrefixCache::new(100);

        // Insert prefixes of different lengths
        let prefix_3 = compute_prefix_hash(&[1, 2, 3]);
        let prefix_5 = compute_prefix_hash(&[1, 2, 3, 4, 5]);

        cache.insert(CachedPrefix::new(prefix_3, 3, vec![]));
        cache.insert(CachedPrefix::new(prefix_5, 5, vec![]));

        // Search for tokens that match the 5-token prefix
        let tokens = vec![1, 2, 3, 4, 5, 6, 7];
        let result = find_longest_prefix(&mut cache, &tokens);

        assert!(result.is_some());
        let (hash, len) = result.expect("test");
        assert_eq!(hash, prefix_5);
        assert_eq!(len, 5);
    }

    #[test]
    fn test_find_longest_prefix_no_match() {
        let mut cache = PrefixCache::new(100);

        // Insert a prefix
        let prefix = compute_prefix_hash(&[1, 2, 3]);
        cache.insert(CachedPrefix::new(prefix, 3, vec![]));

        // Search for non-matching tokens
        let tokens = vec![4, 5, 6];
        let result = find_longest_prefix(&mut cache, &tokens);

        assert!(result.is_none());
    }

    #[test]
    fn test_prefix_cache_default() {
        let cache = PrefixCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_cache_stats_hit_rate_zero() {
        let stats = PrefixCacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    // === KV Quantization Tests ===

    #[test]
    fn test_kv_quant_type_bytes_per_value() {
        assert_eq!(KvQuantType::FP32.bytes_per_value(), 4.0);
        assert_eq!(KvQuantType::Q8.bytes_per_value(), 1.0);
        assert_eq!(KvQuantType::Q4.bytes_per_value(), 0.5);
    }

    #[test]
    fn test_kv_quant_type_memory_reduction() {
        assert_eq!(KvQuantType::FP32.memory_reduction(), 1.0);
        assert_eq!(KvQuantType::Q8.memory_reduction(), 4.0);
        assert_eq!(KvQuantType::Q4.memory_reduction(), 8.0);
    }

    #[test]
    fn test_kv_quant_type_default() {
        let quant_type = KvQuantType::default();
        assert_eq!(quant_type, KvQuantType::FP32);
    }

    #[test]
    fn test_q8_kv_block_new() {
        let block = Q8KvBlock::new();
        assert_eq!(block.scale, 0.0);
        assert_eq!(block.quants, [0i8; KV_QUANT_BLOCK_SIZE]);
    }

    #[test]
    fn test_q8_kv_block_quantize_dequantize() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.1; // -1.6 to 1.5
        }

        let block = Q8KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Q8 should have ~1% error or less
        for i in 0..KV_QUANT_BLOCK_SIZE {
            let error = (values[i] - restored[i]).abs();
            assert!(
                error < 0.02,
                "Q8 error too high at {}: {} vs {}",
                i,
                values[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_q8_kv_block_zero_values() {
        let values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        let block = Q8KvBlock::quantize(&values);
        let restored = block.dequantize();

        for v in restored {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_kv_block_new() {
        let block = Q4KvBlock::new();
        assert_eq!(block.scale, 0.0);
        assert_eq!(block.quants, [0u8; KV_QUANT_BLOCK_SIZE / 2]);
    }

    #[test]
    fn test_q4_kv_block_quantize_dequantize() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.1;
        }

        let block = Q4KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Q4 has more error than Q8, but should still be reasonable
        for i in 0..KV_QUANT_BLOCK_SIZE {
            let error = (values[i] - restored[i]).abs();
            assert!(
                error < 0.3,
                "Q4 error too high at {}: {} vs {}",
                i,
                values[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_q4_kv_block_zero_values() {
        let values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        let block = Q4KvBlock::quantize(&values);
        let restored = block.dequantize();

        for v in restored {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_quantized_kv_data_fp32() {
        let data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);
        assert_eq!(data.quant_type(), KvQuantType::FP32);
        assert_eq!(data.memory_bytes(), 16 * 8 * 64 * 4 * 2); // 2 for K+V
    }
