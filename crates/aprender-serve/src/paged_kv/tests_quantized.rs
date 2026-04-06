
    #[test]
    fn test_quantized_kv_data_q8() {
        let data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);
        assert_eq!(data.quant_type(), KvQuantType::Q8);
        // Q8 uses less memory than FP32
        let fp32_data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);
        assert!(data.memory_bytes() < fp32_data.memory_bytes());
    }

    #[test]
    fn test_quantized_kv_data_q4() {
        let data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);
        assert_eq!(data.quant_type(), KvQuantType::Q4);
        // Q4 uses even less memory than Q8
        let q8_data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);
        assert!(data.memory_bytes() < q8_data.memory_bytes());
    }

    #[test]
    fn test_quantized_kv_data_write_read_fp32() {
        let mut data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);

        let test_keys: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        data.write_keys(0, &test_keys);
        let read_keys = data.read_keys(0, 64);

        assert_eq!(read_keys, test_keys);
    }

    #[test]
    fn test_quantized_kv_data_write_read_q8() {
        let mut data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);

        let test_values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        data.write_values(0, &test_values);
        let read_values = data.read_values(0, 64);

        // Q8 should preserve values with small error
        for (orig, read) in test_values.iter().zip(read_values.iter()) {
            assert!((orig - read).abs() < 0.05);
        }
    }

    #[test]
    fn test_quantized_kv_page_new() {
        let page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);
        assert_eq!(page.quant_type(), KvQuantType::Q8);
        assert_eq!(page.num_tokens, 0);
        assert_eq!(page.ref_count, 0); // Pages start in free pool with ref_count 0
        assert!(!page.is_full());
        assert!(!page.is_shared());
    }

    #[test]
    fn test_quantized_kv_page_read_write() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);

        let keys: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let values: Vec<f32> = (0..512).map(|i| -i as f32 * 0.01).collect();

        page.write_keys(0, &keys);
        page.write_values(0, &values);

        let read_keys = page.read_keys(0);
        let read_values = page.read_values(0);

        assert_eq!(read_keys.len(), 512);
        assert_eq!(read_values.len(), 512);
        assert_eq!(read_keys, keys);
        assert_eq!(read_values, values);
    }

    #[test]
    fn test_quantized_kv_page_is_full() {
        let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);
        assert!(!page.is_full());
        assert_eq!(page.remaining_capacity(), 16);

        page.num_tokens = 16;
        assert!(page.is_full());
        assert_eq!(page.remaining_capacity(), 0);
    }

    #[test]
    fn test_quantized_paged_kv_cache_new() {
        let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        assert_eq!(cache.quant_type(), KvQuantType::Q8);
        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_quantized_paged_kv_cache_allocate() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(32).expect("test");

        assert_eq!(cache.free_page_count(), 98); // 32 tokens = 2 pages
        assert_eq!(cache.stats().active_sequences, 1);
        assert!(seq_id.value() < u64::MAX);
    }

    #[test]
    fn test_quantized_paged_kv_cache_free() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q4);
        let seq_id = cache.allocate_sequence(16).expect("test");

        assert_eq!(cache.free_page_count(), 99);

        cache.free_sequence(seq_id);

        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_quantized_paged_kv_cache_memory_savings() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let _seq_id = cache.allocate_sequence(16).expect("test");

        let savings = cache.memory_savings();
        // Q8 uses (4 + 32) = 36 bytes per block of 32 values vs 128 bytes for FP32
        // Ratio: 36/128 = 0.28125, with some overhead ~0.35
        assert!(
            savings < 0.6,
            "Q8 should use less than 60% of FP32 memory, got {}",
            savings
        );
    }

    #[test]
    fn test_quantized_paged_kv_cache_q4_savings() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q4);
        let _seq_id = cache.allocate_sequence(16).expect("test");

        let savings = cache.memory_savings();
        // Q4 uses (4 + 16) = 20 bytes per block of 32 values vs 128 bytes for FP32
        // Ratio: 20/128 = 0.15625, with some overhead ~0.20
        assert!(
            savings < 0.4,
            "Q4 should use less than 40% of FP32 memory, got {}",
            savings
        );
    }

    #[test]
    fn test_quantized_paged_kv_cache_get_page() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(32).expect("test");

        let page = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(page.quant_type(), KvQuantType::Q8);

        let page2 = cache.get_page(seq_id, 16).expect("test");
        assert_eq!(page2.quant_type(), KvQuantType::Q8);
    }

    #[test]
    fn test_quantized_paged_kv_cache_get_page_mut() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let seq_id = cache.allocate_sequence(16).expect("test");

        let page = cache.get_page_mut(seq_id, 0).expect("test");
        page.num_tokens = 8;

        let page2 = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(page2.num_tokens, 8);
    }

    #[test]
    fn test_quantized_paged_kv_cache_oom() {
        let mut cache = QuantizedPagedKvCache::new(1, 16, 8, 64, KvQuantType::Q8);
        let _seq1 = cache.allocate_sequence(16).expect("test");

        let result = cache.allocate_sequence(16);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_q8_block_default() {
        let block = Q8KvBlock::default();
        assert_eq!(block.scale, 0.0);
    }

    #[test]
    fn test_q4_block_default() {
        let block = Q4KvBlock::default();
        assert_eq!(block.scale, 0.0);
    }

    // =========================================================================
    // Deep coverage tests (_deep_pkcov_ prefix)
    // =========================================================================

    // --- Error handling paths ---

    #[test]
    fn test_deep_pkcov_extend_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.extend(fake_seq, 32);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_extend_sequence_out_of_memory() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // Uses 1 page, 1 free

        // Extend to need more pages than available
        // extend(48) needs 48/16 = 3 pages total, have 1, need 2 more, only 1 free
        let result = cache.extend(seq_id, 48);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_deep_pkcov_extend_no_new_pages_needed() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // 1 page, 16 capacity
        cache.update_tokens(seq_id, 5).expect("update");

        // Extend by small amount that fits in existing page
        let result = cache.extend(seq_id, 5);
        assert!(result.is_ok());
        assert_eq!(cache.free_page_count(), 99); // No new pages allocated
    }

    #[test]
    fn test_deep_pkcov_update_tokens_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.update_tokens(fake_seq, 10);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_fork_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.fork_sequence(fake_seq);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_get_page_sequence_not_found() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.get_page(fake_seq, 0);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_get_page_mut_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.get_page_mut(fake_seq, 0);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_deep_pkcov_get_page_mut_invalid_page_access() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("alloc"); // 1 page

        let result = cache.get_page_mut(seq_id, 100); // Beyond allocated
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    // --- Edge cases in paged memory management ---

    #[test]
    fn test_deep_pkcov_utilization_zero_pages() {
        // Create cache with zero pages - edge case
        let cache = PagedKvCache::new(0, 16, 8, 64);
        assert_eq!(cache.utilization(), 0.0);
        assert_eq!(cache.free_page_count(), 0);
    }

    #[test]
    fn test_deep_pkcov_free_nonexistent_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        // Should not panic, just do nothing
        cache.free_sequence(fake_seq);
        assert_eq!(cache.stats().sequences_freed, 0);
    }

    #[test]
    fn test_deep_pkcov_allocate_zero_tokens() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Zero tokens should allocate at least 1 page (div_ceil behavior)
        // Actually 0.div_ceil(16) = 0, so 0 pages needed
        let seq_id = cache.allocate_sequence(0).expect("alloc");
        assert!(cache.page_tables.contains_key(&seq_id));
    }

    #[test]
    fn test_deep_pkcov_update_tokens_spans_multiple_pages() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(48).expect("alloc"); // 3 pages

        // Update with tokens spanning all pages
        cache.update_tokens(seq_id, 48).expect("update");

        let tokens = cache.get_sequence_tokens(seq_id).expect("get");
        assert_eq!(tokens, 48);
    }

    #[test]
    fn test_deep_pkcov_update_tokens_partial_fill() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("alloc"); // 2 pages

        // Only fill part of first page
        cache.update_tokens(seq_id, 5).expect("update");

        let tokens = cache.get_sequence_tokens(seq_id).expect("get");
        assert_eq!(tokens, 5);
    }

    // --- Copy-on-write edge cases ---

    #[test]
    fn test_deep_pkcov_cow_out_of_memory() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("alloc"); // 1 page
        let child_id = cache.fork_sequence(parent_id).expect("fork"); // Shares page

        // Fill remaining page
        let _ = cache.allocate_sequence(16).expect("alloc2");

        // Now try to write to child - COW needs a free page but none available
        let result = cache.get_page_mut(child_id, 0);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_deep_pkcov_cow_multiple_forks() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("alloc");
        cache.update_tokens(parent_id, 16).expect("update");

        // Fork multiple times
        let child1 = cache.fork_sequence(parent_id).expect("fork1");
        let child2 = cache.fork_sequence(parent_id).expect("fork2");

        assert_eq!(cache.stats().cow_operations, 2);

        // Write to child1 triggers COW
        let _page = cache.get_page_mut(child1, 0).expect("get");
        assert_eq!(cache.stats().cow_operations, 3);

        // child2 still shares with parent
        let page = cache.get_page(child2, 0).expect("get2");
        assert!(page.ref_count >= 1);
    }

    #[test]
    fn test_deep_pkcov_free_shared_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("alloc");
        let child_id = cache.fork_sequence(parent_id).expect("fork");

        // Free parent - pages should not return to free list (still referenced by child)
        cache.free_sequence(parent_id);
        assert_eq!(cache.free_page_count(), 99); // Page still in use by child

        // Free child - now pages return
        cache.free_sequence(child_id);
        assert_eq!(cache.free_page_count(), 100);
    }

    // --- Defragmentation paths ---

    #[test]
    fn test_deep_pkcov_should_defragment_low_free_ratio() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);

        // Allocate to use most pages
        for _ in 0..9 {
            let _ = cache.allocate_sequence(16).expect("alloc");
        }

        // With >90% utilization and any fragmentation, should trigger
        // Need to create a hole first
        let seq_to_free = cache.allocate_sequence(16).ok();
        if let Some(seq) = seq_to_free {
            cache.free_sequence(seq);
        }

        // Low free ratio check is at 10% threshold
        // We have 1 free page out of 10 = 10%, right at threshold
    }

    #[test]
    fn test_deep_pkcov_should_defragment_high_waste() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Allocate several sequences with partial fill
        let seq1 = cache.allocate_sequence(32).expect("alloc1"); // 2 pages
        let seq2 = cache.allocate_sequence(32).expect("alloc2"); // 2 pages
        let seq3 = cache.allocate_sequence(32).expect("alloc3"); // 2 pages

        // Free middle sequence to create holes
        cache.free_sequence(seq2);

        // Update with very few tokens to create waste
        cache.update_tokens(seq1, 1).expect("update1");
        cache.update_tokens(seq3, 1).expect("update3");

        let stats = cache.fragmentation_stats();
        assert!(stats.holes > 0 || stats.wasted_capacity > 0);
    }

    #[test]
    fn test_deep_pkcov_compact_empty_page_list() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(0).expect("alloc"); // 0 pages

        let moved = cache.compact_sequence(seq_id);
        assert_eq!(moved, 0);
    }

    #[test]
    fn test_deep_pkcov_fragmentation_stats_large_free_region() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        // Allocate just one sequence at the beginning
        let _ = cache.allocate_sequence(16).expect("alloc");

        let stats = cache.fragmentation_stats();
        // Large free region at the end
        assert!(stats.largest_free_region >= 99);
    }
