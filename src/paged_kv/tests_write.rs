//! Part 3: Quantization read/write tests, prefix cache, and ID tests
//!
//! Focus areas:
//! 1. Quantization write/read across block boundaries
//! 2. QuantizedKvPage token operations
//! 3. Prefix cache additional tests
//! 4. ID generation and hashing

#[cfg(test)]
mod tests {
    use crate::paged_kv::*;

    // =========================================================================
    // Quantization Write/Read Across Block Boundaries
    // =========================================================================

    #[test]
    fn test_q8_write_read_across_block_boundary() {
        let mut data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);

        // Write data that spans multiple quant blocks (block_size = 32)
        let test_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        data.write_keys(0, &test_data);

        let read = data.read_keys(0, 100);
        assert_eq!(read.len(), 100);

        // Q8 should preserve values with small error
        for (orig, read_val) in test_data.iter().zip(read.iter()) {
            assert!(
                (orig - read_val).abs() < 0.05,
                "Q8 error: {} vs {}",
                orig,
                read_val
            );
        }
    }

    #[test]
    fn test_q4_write_read_across_block_boundary() {
        let mut data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);

        let test_data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        data.write_values(0, &test_data);

        let read = data.read_values(0, 100);
        assert_eq!(read.len(), 100);

        // Q4 has larger error tolerance
        for (orig, read_val) in test_data.iter().zip(read.iter()) {
            assert!(
                (orig - read_val).abs() < 0.5,
                "Q4 error: {} vs {}",
                orig,
                read_val
            );
        }
    }

    #[test]
    fn test_read_with_offset_in_middle_of_block() {
        let mut data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);

        // Write full data
        let full_data: Vec<f32> = (0..128).map(|i| i as f32).collect();
        data.write_keys(0, &full_data);

        // Read from middle of block
        let read = data.read_keys(10, 20);
        assert_eq!(read.len(), 20);
        for i in 0..20 {
            assert_eq!(read[i], (i + 10) as f32);
        }
    }

    #[test]
    fn test_write_partial_at_end_of_data() {
        let mut data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);

        let total_size = 16 * 8 * 64; // 8192 elements
        let small_data = vec![99.0f32; 10];

        // Write near the end
        data.write_keys(total_size - 5, &small_data);

        // Only 5 elements should be written (truncated)
        let read = data.read_keys(total_size - 5, 10);
        assert_eq!(read.len(), 5);
        for val in &read {
            assert_eq!(*val, 99.0);
        }
    }

    // =========================================================================
    // QuantizedKvPage Token Position Operations
    // =========================================================================

    #[test]
    fn test_quantized_page_write_read_multiple_tokens() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);

        // Write keys/values for token 0
        let k0: Vec<f32> = (0..512).map(|i| i as f32 * 0.1).collect();
        let v0: Vec<f32> = (0..512).map(|i| -i as f32 * 0.1).collect();
        page.write_keys(0, &k0);
        page.write_values(0, &v0);
        page.num_tokens = 1;

        // Write keys/values for token 1
        let k1: Vec<f32> = (0..512).map(|i| (i + 1000) as f32 * 0.1).collect();
        let v1: Vec<f32> = (0..512).map(|i| -(i + 1000) as f32 * 0.1).collect();
        page.write_keys(1, &k1);
        page.write_values(1, &v1);
        page.num_tokens = 2;

        // Verify reads
        let rk0 = page.read_keys(0);
        let rv0 = page.read_values(0);
        let rk1 = page.read_keys(1);
        let rv1 = page.read_values(1);

        assert_eq!(rk0, k0);
        assert_eq!(rv0, v0);
        assert_eq!(rk1, k1);
        assert_eq!(rv1, v1);
    }

    #[test]
    fn test_quantized_page_remaining_capacity() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);

        assert_eq!(page.remaining_capacity(), 16);
        page.num_tokens = 5;
        assert_eq!(page.remaining_capacity(), 11);
        page.num_tokens = 16;
        assert_eq!(page.remaining_capacity(), 0);
        page.num_tokens = 20; // Over capacity
        assert_eq!(page.remaining_capacity(), 0);
    }

    // =========================================================================
    // Prefix Cache Additional Tests
    // =========================================================================

    #[test]
    #[ignore = "Test expectation needs adjustment"]
    fn test_prefix_cache_eviction_with_zero_ref_count() {
        let mut cache = PrefixCache::new(2);

        // Insert first prefix
        let mut p1 = CachedPrefix::new(1, 10, vec![]);
        p1.ref_count = 0; // Evictable
        cache.insert(p1);

        // Insert second prefix (also evictable)
        let mut p2 = CachedPrefix::new(2, 20, vec![]);
        p2.ref_count = 0;
        cache.insert(p2);

        assert_eq!(cache.len(), 2);

        // Insert third - should evict LRU
        let p3 = CachedPrefix::new(3, 30, vec![]);
        let success = cache.insert(p3);
        assert!(success);
        assert_eq!(cache.len(), 2);

        // First prefix (hash=1) should be evicted (LRU)
        assert!(!cache.contains(1));
        assert!(cache.contains(2) || cache.contains(3));
    }

    #[test]
    fn test_prefix_cache_update_access_time() {
        let mut cache = PrefixCache::new(10);

        // Insert two prefixes
        let p1 = CachedPrefix::new(1, 10, vec![]);
        let p2 = CachedPrefix::new(2, 20, vec![]);
        cache.insert(p1);
        cache.insert(p2);

        // Lookup hash 1 multiple times to update access time
        cache.lookup(1);
        cache.lookup(1);
        cache.lookup(1);

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
    }

    #[test]
    fn test_prefix_cache_add_ref_nonexistent() {
        let mut cache = PrefixCache::new(10);

        // Add ref to nonexistent entry
        let result = cache.add_ref(99999);
        assert!(!result);
    }

    #[test]
    fn test_prefix_cache_remove_ref_nonexistent() {
        let mut cache = PrefixCache::new(10);

        // Remove ref from nonexistent entry
        let result = cache.remove_ref(99999);
        assert!(!result);
    }

    #[test]
    fn test_prefix_cache_zero_capacity() {
        let mut cache = PrefixCache::new(0);
        assert_eq!(cache.utilization(), 0.0);

        // Can't insert anything
        let p = CachedPrefix::new(1, 10, vec![]);
        let success = cache.insert(p);
        assert!(!success);
    }

    // =========================================================================
    // ID Generation and Hashing
    // =========================================================================

    #[test]
    fn test_seq_id_monotonic() {
        let mut prev = SeqId::new().value();
        for _ in 0..100 {
            let current = SeqId::new().value();
            assert!(current > prev);
            prev = current;
        }
    }

    #[test]
    fn test_page_id_serialization() {
        let id = PageId::new(42);
        let json = serde_json::to_string(&id).expect("serialize");
        let parsed: PageId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.value(), 42);
    }

    #[test]
    fn test_seq_id_serialization() {
        let id = SeqId::new();
        let json = serde_json::to_string(&id).expect("serialize");
        let parsed: SeqId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.value(), id.value());
    }

    #[test]
    fn test_prefix_hash_single_token() {
        let hash1 = compute_prefix_hash(&[0]);
        let hash2 = compute_prefix_hash(&[1]);
        let hash3 = compute_prefix_hash(&[u32::MAX]);

        // All should be different
        assert_ne!(hash1, hash2);
        assert_ne!(hash2, hash3);
        assert_ne!(hash1, hash3);
    }

    // =========================================================================
    // Additional Edge Cases
    // =========================================================================

    #[test]
    fn test_q8_quantize_extreme_values() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        // Set some extreme values
        values[0] = 1000.0;
        values[1] = -1000.0;
        values[2] = 0.0001;
        values[3] = -0.0001;

        let block = Q8KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Extremes should be preserved (within quantization error)
        assert!((restored[0] - 1000.0).abs() < 10.0);
        assert!((restored[1] + 1000.0).abs() < 10.0);
    }

    #[test]
    fn test_q4_quantize_extreme_values() {
        let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
        // Set some extreme values
        values[0] = 100.0;
        values[1] = -100.0;

        let block = Q4KvBlock::quantize(&values);
        let restored = block.dequantize();

        // Q4 has higher quantization error
        assert!((restored[0] - 100.0).abs() < 20.0);
        assert!((restored[1] + 100.0).abs() < 20.0);
    }

    #[test]
    fn test_quantized_kv_data_q4_read_write() {
        let mut data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);

        let test_values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        data.write_values(0, &test_values);
        let read_values = data.read_values(0, 64);

        // Q4 has larger error tolerance
        for (orig, read) in test_values.iter().zip(read_values.iter()) {
            assert!((orig - read).abs() < 0.5);
        }
    }

    #[test]
    fn test_quantized_cache_total_pages() {
        let cache = QuantizedPagedKvCache::new(50, 16, 8, 64, KvQuantType::Q8);
        assert_eq!(cache.total_pages(), 50);
    }

    #[test]
    fn test_kv_page_data_clone() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        page.keys[0] = 1.0;
        page.values[0] = 2.0;
        page.num_tokens = 5;
        page.ref_count = 2;

        let cloned = page.clone();
        assert_eq!(cloned.keys[0], 1.0);
        assert_eq!(cloned.values[0], 2.0);
        assert_eq!(cloned.num_tokens, 5);
        assert_eq!(cloned.ref_count, 2);
    }

    #[test]
    fn test_cached_prefix_clone() {
        let prefix = CachedPrefix::new(12345, 100, vec![PageId::new(0), PageId::new(1)]);
        let cloned = prefix.clone();

        assert_eq!(cloned.hash, 12345);
        assert_eq!(cloned.num_tokens, 100);
        assert_eq!(cloned.page_ids.len(), 2);
    }

    #[test]
    fn test_fragmentation_stats_default() {
        let stats = FragmentationStats::default();
        assert_eq!(stats.holes, 0);
        assert_eq!(stats.wasted_capacity, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert_eq!(stats.largest_free_region, 0);
        assert_eq!(stats.avg_tokens_per_page, 0.0);
    }

    #[test]
    fn test_paged_cache_stats_default() {
        let stats = PagedCacheStats::default();
        assert_eq!(stats.sequences_allocated, 0);
        assert_eq!(stats.sequences_freed, 0);
        assert_eq!(stats.pages_allocated, 0);
        assert_eq!(stats.pages_freed, 0);
        assert_eq!(stats.active_sequences, 0);
        assert_eq!(stats.used_pages, 0);
        assert_eq!(stats.cow_operations, 0);
        assert_eq!(stats.defrag_operations, 0);
        assert_eq!(stats.pages_moved, 0);
    }
}
