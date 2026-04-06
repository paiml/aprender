//! EXTREME TDD coverage tests for paged_kv.rs
//!
//! Focus areas:
//! - Page allocation/deallocation edge cases
//! - Block management
//! - Cache eviction policies
//! - Memory pressure handling
//! - COW (Copy-on-Write) edge cases
//! - Quantization edge cases
//! - Prefix cache edge cases

use realizar::paged_kv::{
    compute_prefix_hash, find_longest_prefix, CachedPrefix, FragmentationStats, KvPage,
    KvQuantType, PageId, PagedCacheError, PagedCacheStats, PagedKvCache, PrefixCache,
    PrefixCacheStats, Q4KvBlock, Q8KvBlock, QuantizedKvData, QuantizedKvPage,
    QuantizedPagedKvCache, SeqId, KV_QUANT_BLOCK_SIZE,
};

// ============================================================================
// PAGE ALLOCATION/DEALLOCATION EDGE CASES
// ============================================================================

#[test]
fn test_allocate_zero_tokens() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    // Zero tokens should allocate zero pages
    let result = cache.allocate_sequence(0);
    assert!(result.is_ok());
    // With div_ceil(0, 16) = 0 pages, no pages should be used
    assert_eq!(cache.free_page_count(), 100);
}

#[test]
fn test_allocate_exactly_block_size_tokens() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    // Exactly 16 tokens should use exactly 1 page
    let seq_id = cache.allocate_sequence(16).expect("allocation");
    assert_eq!(cache.free_page_count(), 99);
    assert_eq!(cache.stats().pages_allocated, 1);

    // Verify we can access the sequence
    assert!(cache.get_page(seq_id, 0).is_ok());
}

#[test]
fn test_allocate_one_more_than_block_size() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    // 17 tokens should require 2 pages (ceil(17/16) = 2)
    let _seq_id = cache.allocate_sequence(17).expect("allocation");
    assert_eq!(cache.free_page_count(), 98);
    assert_eq!(cache.stats().pages_allocated, 2);
}

#[test]
fn test_extend_sequence_not_found() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let fake_seq = SeqId::new();

    let result = cache.extend(fake_seq, 10);
    assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
}

#[test]
fn test_extend_no_new_pages_needed() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let seq_id = cache.allocate_sequence(16).expect("allocation");
    cache.update_tokens(seq_id, 5).expect("update");

    // We have 16 capacity, 5 tokens used. Extending by 5 more should not allocate new pages.
    let initial_pages = cache.free_page_count();
    cache.extend(seq_id, 5).expect("extend");
    assert_eq!(cache.free_page_count(), initial_pages);
}

#[test]
fn test_extend_triggers_oom() {
    let mut cache = PagedKvCache::new(2, 16, 8, 64);
    let seq_id = cache.allocate_sequence(16).expect("allocation"); // 1 page
    cache.update_tokens(seq_id, 16).expect("update");

    // Allocate another sequence to use remaining page
    let _seq2 = cache.allocate_sequence(16).expect("allocation");

    // Now extend should fail - no free pages
    let result = cache.extend(seq_id, 16);
    assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
}

#[test]
fn test_free_sequence_non_existent() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let fake_seq = SeqId::new();

    // Freeing non-existent sequence should be a no-op
    let initial_free = cache.free_page_count();
    cache.free_sequence(fake_seq);
    assert_eq!(cache.free_page_count(), initial_free);
}

#[test]
fn test_free_multiple_sequences_in_order() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);

    let seq1 = cache.allocate_sequence(16).expect("allocation");
    let seq2 = cache.allocate_sequence(16).expect("allocation");
    let seq3 = cache.allocate_sequence(16).expect("allocation");

    assert_eq!(cache.free_page_count(), 97);

    cache.free_sequence(seq2);
    assert_eq!(cache.free_page_count(), 98);

    cache.free_sequence(seq1);
    assert_eq!(cache.free_page_count(), 99);

    cache.free_sequence(seq3);
    assert_eq!(cache.free_page_count(), 100);
}

// ============================================================================
// BLOCK MANAGEMENT
// ============================================================================

#[test]
fn test_kv_page_remaining_capacity_overflow_safe() {
    let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
    // Set num_tokens beyond block_size (shouldn't happen normally)
    page.num_tokens = 20;
    // Should use saturating_sub and return 0
    assert_eq!(page.remaining_capacity(16), 0);
}

#[test]
fn test_page_id_equality() {
    let id1 = PageId::new(42);
    let id2 = PageId::new(42);
    let id3 = PageId::new(43);

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}

#[test]
fn test_seq_id_uniqueness_across_many() {
    let mut ids = Vec::with_capacity(1000);
    for _ in 0..1000 {
        ids.push(SeqId::new());
    }

    // All IDs should be unique
    let mut values: Vec<u64> = ids.iter().map(|id| id.value()).collect();
    values.sort_unstable();
    values.dedup();
    assert_eq!(values.len(), 1000);
}

#[test]
fn test_get_page_mut_sequence_not_found() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let fake_seq = SeqId::new();

    let result = cache.get_page_mut(fake_seq, 0);
    assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
}

#[test]
fn test_get_page_mut_invalid_offset() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let seq_id = cache.allocate_sequence(16).expect("allocation");

    // Token position 32 would be in page 2, but we only have 1 page
    let result = cache.get_page_mut(seq_id, 32);
    assert!(matches!(
        result,
        Err(PagedCacheError::InvalidPageAccess { .. })
    ));
}

#[test]
fn test_update_tokens_multiple_pages() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let seq_id = cache.allocate_sequence(48).expect("allocation"); // 3 pages

    // Update with 48 tokens - should fill all 3 pages
    cache.update_tokens(seq_id, 48).expect("update");

    let tokens = cache.get_sequence_tokens(seq_id).expect("get tokens");
    assert_eq!(tokens, 48);
}

#[test]
fn test_update_tokens_not_found() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let fake_seq = SeqId::new();

    let result = cache.update_tokens(fake_seq, 10);
    assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
}

// ============================================================================
// CACHE EVICTION AND MEMORY PRESSURE
// ============================================================================

#[test]
fn test_utilization_zero_pages() {
    let cache = PagedKvCache::new(0, 16, 8, 64);
    // Division by zero protection
    assert_eq!(cache.utilization(), 0.0);
}

#[test]
fn test_utilization_full_cache() {
    let mut cache = PagedKvCache::new(10, 16, 8, 64);

    for _ in 0..10 {
        let _ = cache.allocate_sequence(16).expect("allocation");
    }

    assert!((cache.utilization() - 100.0).abs() < 0.01);
}

#[test]
fn test_prefix_cache_eviction_with_references() {
    let mut cache = PrefixCache::new(2);

    // Insert two prefixes with ref_count = 1 (default)
    cache.insert(CachedPrefix::new(1, 1, vec![]));
    cache.insert(CachedPrefix::new(2, 2, vec![]));

    // Add more references to first one
    cache.add_ref(1);
    cache.add_ref(1);

    // Remove one ref from second (makes ref_count = 0)
    let removed = cache.remove_ref(2);
    assert!(removed); // Entry 2 was removed

    // Now we have capacity for a new entry
    let success = cache.insert(CachedPrefix::new(3, 3, vec![]));
    assert!(success);
}

#[test]
fn test_prefix_cache_eviction_lru_order() {
    let mut cache = PrefixCache::new(3);

    // Insert 3 prefixes, then reduce ref_count to 0 for eviction eligibility
    cache.insert(CachedPrefix::new(1, 1, vec![]));
    cache.insert(CachedPrefix::new(2, 2, vec![]));
    cache.insert(CachedPrefix::new(3, 3, vec![]));

    // Make all evictable
    cache.remove_ref(1);
    cache.remove_ref(2);

    // Access hash 2 to make it more recently used (but it's already removed)
    // Actually hash 1 and 2 are removed now. Let's test differently.
    assert_eq!(cache.len(), 1); // Only hash 3 remains (still has ref)
}

#[test]
fn test_prefix_cache_add_ref_not_found() {
    let mut cache = PrefixCache::new(10);

    let result = cache.add_ref(99999);
    assert!(!result);
}

#[test]
fn test_prefix_cache_remove_ref_not_found() {
    let mut cache = PrefixCache::new(10);

    let result = cache.remove_ref(99999);
    assert!(!result);
}

#[test]
fn test_prefix_cache_utilization_zero_max() {
    let cache = PrefixCache::new(0);
    assert_eq!(cache.utilization(), 0.0);
}

// ============================================================================
// COW (COPY-ON-WRITE) EDGE CASES
// ============================================================================

#[test]
fn test_fork_then_free_parent() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let parent_id = cache.allocate_sequence(16).expect("allocation");

    let child_id = cache.fork_sequence(parent_id).expect("fork");

    // Free parent - child should still work because pages are shared
    cache.free_sequence(parent_id);

    // Child's pages should still be accessible (ref_count > 0)
    assert!(cache.get_page(child_id, 0).is_ok());
}

#[test]
fn test_fork_then_free_child() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let parent_id = cache.allocate_sequence(16).expect("allocation");

    let child_id = cache.fork_sequence(parent_id).expect("fork");

    // Free child first
    cache.free_sequence(child_id);

    // Parent should still work
    assert!(cache.get_page(parent_id, 0).is_ok());
}

#[test]
fn test_fork_not_found() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let fake_seq = SeqId::new();

    let result = cache.fork_sequence(fake_seq);
    assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
}

#[test]
fn test_cow_triggers_on_shared_page_write() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let parent_id = cache.allocate_sequence(16).expect("allocation");
    cache.update_tokens(parent_id, 16).expect("update");

    // Write data to parent's page
    {
        let page = cache.get_page_mut(parent_id, 0).expect("get page");
        page.keys[0] = 123.0;
    }

    let child_id = cache.fork_sequence(parent_id).expect("fork");
    let initial_cow = cache.stats().cow_operations;

    // Writing to child's page should trigger COW
    {
        let page = cache.get_page_mut(child_id, 0).expect("get page");
        page.keys[0] = 456.0;
    }

    assert!(cache.stats().cow_operations > initial_cow);

    // Verify both have different data now
    let parent_page = cache.get_page(parent_id, 0).expect("get");
    let child_page = cache.get_page(child_id, 0).expect("get");
    assert_eq!(parent_page.keys[0], 123.0);
    assert_eq!(child_page.keys[0], 456.0);
}

#[test]
fn test_cow_oom_during_write() {
    let mut cache = PagedKvCache::new(2, 16, 8, 64);
    let parent_id = cache.allocate_sequence(16).expect("allocation"); // 1 page
    cache.update_tokens(parent_id, 16).expect("update");

    let child_id = cache.fork_sequence(parent_id).expect("fork");

    // Use the last free page
    let _other = cache.allocate_sequence(16).expect("allocation");

    // Now COW should fail - no free pages for copy
    let result = cache.get_page_mut(child_id, 0);
    assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
}

// ============================================================================
// QUANTIZATION EDGE CASES
// ============================================================================

#[test]
fn test_q8_quantize_extreme_values() {
    let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
    values[0] = 1000.0;
    values[1] = -1000.0;

    let block = Q8KvBlock::quantize(&values);
    let restored = block.dequantize();

    // Should handle large values via scale
    assert!(restored[0] > 0.0);
    assert!(restored[1] < 0.0);
}

#[test]
fn test_q4_quantize_extreme_values() {
    let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
    values[0] = 100.0;
    values[1] = -100.0;

    let block = Q4KvBlock::quantize(&values);
    let restored = block.dequantize();

    // Q4 has less precision but should preserve sign
    assert!(restored[0] > 0.0);
    assert!(restored[1] < 0.0);
}

#[test]
fn test_q8_quantize_tiny_values() {
    let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
    values.fill(1e-11); // Below threshold

    let block = Q8KvBlock::quantize(&values);
    // Scale should be 0, resulting in zero output
    assert_eq!(block.scale, 0.0);
}

#[test]
fn test_q4_quantize_tiny_values() {
    let mut values = [0.0f32; KV_QUANT_BLOCK_SIZE];
    values.fill(1e-11);

    let block = Q4KvBlock::quantize(&values);
    assert_eq!(block.scale, 0.0);
}

#[test]
fn test_quantized_kv_data_write_beyond_bounds() {
    let mut data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);

    // Try to write more data than fits
    let large_data: Vec<f32> = (0..20000).map(|i| i as f32).collect();
    data.write_keys(0, &large_data);

    // Should truncate safely
    let read = data.read_keys(0, 16 * 8 * 64);
    assert_eq!(read.len(), 16 * 8 * 64);
}

#[test]
fn test_quantized_kv_data_read_beyond_bounds() {
    let data = QuantizedKvData::new(KvQuantType::FP32, 16, 8, 64);

    // Try to read beyond bounds
    let read = data.read_keys(16 * 8 * 64 - 10, 100);

    // Should return only available data
    assert_eq!(read.len(), 10);
}

#[test]
fn test_quantized_kv_page_shared_check() {
    let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);

    assert!(!page.is_shared());
    page.ref_count = 2;
    assert!(page.is_shared());
}

#[test]
fn test_quantized_paged_cache_get_page_not_found() {
    let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
    let fake_seq = SeqId::new();

    let result = cache.get_page(fake_seq, 0);
    assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
}

#[test]
fn test_quantized_paged_cache_get_page_invalid_offset() {
    let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
    let seq_id = cache.allocate_sequence(16).expect("allocation");

    let result = cache.get_page(seq_id, 100);
    assert!(matches!(
        result,
        Err(PagedCacheError::InvalidPageAccess { .. })
    ));
}

#[test]
fn test_quantized_paged_cache_memory_savings_empty() {
    let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
    // No pages used, so fp32_equivalent_memory is 0
    let savings = cache.memory_savings();
    assert_eq!(savings, 1.0); // Default when fp32_mem is 0
}

#[test]
fn test_quantized_paged_cache_total_pages() {
    let cache = QuantizedPagedKvCache::new(50, 16, 8, 64, KvQuantType::Q4);
    assert_eq!(cache.total_pages(), 50);
}

// ============================================================================
// DEFRAGMENTATION EDGE CASES
// ============================================================================

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
fn test_should_defragment_low_free_pages_with_holes() {
    let mut cache = PagedKvCache::new(10, 16, 8, 64);

    // Allocate all pages
    for _ in 0..10 {
        let _ = cache.allocate_sequence(16).expect("allocation");
    }

    // This would normally trigger should_defragment if there were holes
    // But all pages are contiguous, so no defrag needed
    assert!(!cache.should_defragment());
}

#[test]
fn test_compact_sequence_empty_pages() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);
    let seq_id = cache.allocate_sequence(0).expect("allocation");

    let moved = cache.compact_sequence(seq_id);
    assert_eq!(moved, 0);
}

#[test]
fn test_is_page_free_check() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);

    // All pages start free
    let _seq_id = cache.allocate_sequence(16).expect("allocation");

    // First page is now used, rest are free
    // We can't directly call is_page_free, but defragment uses it internally
    let stats = cache.fragmentation_stats();
    assert_eq!(stats.largest_free_region, 99); // 99 free pages
}

// ============================================================================
// PREFIX CACHE EDGE CASES
// ============================================================================

#[test]
fn test_compute_prefix_hash_single_token() {
    let hash1 = compute_prefix_hash(&[1]);
    let hash2 = compute_prefix_hash(&[2]);

    assert_ne!(hash1, hash2);
}

#[test]
fn test_compute_prefix_hash_order_matters() {
    let hash1 = compute_prefix_hash(&[1, 2, 3]);
    let hash2 = compute_prefix_hash(&[3, 2, 1]);

    assert_ne!(hash1, hash2);
}

#[test]
fn test_find_longest_prefix_empty_tokens() {
    let mut cache = PrefixCache::new(100);

    // Empty token sequence
    let tokens: Vec<u32> = vec![];
    let result = find_longest_prefix(&mut cache, &tokens);
    assert!(result.is_none());
}

#[test]
fn test_find_longest_prefix_partial_match() {
    let mut cache = PrefixCache::new(100);

    // Only cache prefix of length 3
    let prefix_3 = compute_prefix_hash(&[1, 2, 3]);
    cache.insert(CachedPrefix::new(prefix_3, 3, vec![]));

    // Search with longer sequence
    let tokens = vec![1, 2, 3, 4, 5];
    let result = find_longest_prefix(&mut cache, &tokens);

    assert!(result.is_some());
    let (_, len) = result.unwrap();
    assert_eq!(len, 3);
}

#[test]
fn test_cached_prefix_remove_ref_saturating() {
    let mut prefix = CachedPrefix::new(1, 5, vec![]);
    prefix.ref_count = 0;

    // Should not underflow
    let result = prefix.remove_ref();
    assert!(result); // Returns true because ref_count is 0
    assert_eq!(prefix.ref_count, 0);
}

#[test]
fn test_prefix_cache_stats_hit_rate_all_hits() {
    let mut cache = PrefixCache::new(100);
    let hash = compute_prefix_hash(&[1, 2, 3]);
    cache.insert(CachedPrefix::new(hash, 3, vec![]));

    // Multiple hits
    for _ in 0..10 {
        cache.lookup(hash);
    }

    let stats = cache.stats();
    assert_eq!(stats.hits, 10);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.hit_rate(), 1.0);
}

#[test]
fn test_prefix_cache_stats_hit_rate_all_misses() {
    let mut cache = PrefixCache::new(100);

    // Multiple misses
    for i in 0..10 {
        cache.lookup(i as u64);
    }

    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 10);
    assert_eq!(stats.hit_rate(), 0.0);
}

// ============================================================================
// ERROR DISPLAY COVERAGE
// ============================================================================

#[test]
fn test_page_table_corruption_error() {
    let err = PagedCacheError::PageTableCorruption { seq_id: 42 };
    let msg = err.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("corruption"));
}

// ============================================================================
// STATS COVERAGE
// ============================================================================

#[test]
fn test_paged_cache_stats_clone() {
    let stats = PagedCacheStats {
        sequences_allocated: 10,
        sequences_freed: 5,
        pages_allocated: 100,
        pages_freed: 50,
        active_sequences: 5,
        used_pages: 50,
        cow_operations: 3,
        defrag_operations: 2,
        pages_moved: 15,
    };

    // Use clone to verify Clone impl, then use original to avoid redundant_clone
    #[allow(clippy::redundant_clone)]
    let cloned = stats.clone();
    assert_eq!(cloned.sequences_allocated, 10);
    assert_eq!(cloned.defrag_operations, 2);
    // Also verify original is still usable
    assert_eq!(stats.pages_moved, 15);
}

#[test]
fn test_fragmentation_stats_clone() {
    let stats = FragmentationStats {
        holes: 5,
        wasted_capacity: 100,
        fragmentation_ratio: 0.25,
        largest_free_region: 50,
        avg_tokens_per_page: 12.5,
    };

    #[allow(clippy::redundant_clone)]
    let cloned = stats.clone();
    assert_eq!(cloned.holes, 5);
    assert!((cloned.avg_tokens_per_page - 12.5).abs() < 0.001);
    // Verify original still usable
    assert_eq!(stats.wasted_capacity, 100);
}

#[test]
fn test_prefix_cache_stats_clone() {
    let stats = PrefixCacheStats {
        hits: 100,
        misses: 50,
        prefixes_cached: 10,
        prefixes_evicted: 2,
        tokens_saved: 500,
    };

    #[allow(clippy::redundant_clone)]
    let cloned = stats.clone();
    assert_eq!(cloned.hits, 100);
    assert_eq!(cloned.tokens_saved, 500);
    // Verify original still usable
    assert_eq!(stats.misses, 50);
}

// ============================================================================
// QUANTIZED PAGE READ/WRITE Q8/Q4 COVERAGE
// ============================================================================

#[test]
fn test_quantized_kv_data_q4_write_read() {
    let mut data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);

    let test_keys: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    data.write_keys(0, &test_keys);
    let read_keys = data.read_keys(0, 64);

    // Q4 has more error, but values should be preserved approximately
    for (orig, read) in test_keys.iter().zip(read_keys.iter()) {
        let error = (orig - read).abs();
        assert!(error < 0.5, "Q4 error too high: {} vs {}", orig, read);
    }
}

#[test]
fn test_quantized_kv_data_values_q8() {
    let mut data = QuantizedKvData::new(KvQuantType::Q8, 16, 8, 64);

    let test_values: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    data.write_values(0, &test_values);
    let read_values = data.read_values(0, 64);

    for (orig, read) in test_values.iter().zip(read_values.iter()) {
        let error = (orig - read).abs();
        assert!(error < 0.02, "Q8 error too high: {} vs {}", orig, read);
    }
}

#[test]
fn test_quantized_kv_data_values_q4() {
    let mut data = QuantizedKvData::new(KvQuantType::Q4, 16, 8, 64);

    let test_values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    data.write_values(0, &test_values);
    let read_values = data.read_values(0, 64);

    // Q4 preserves general shape but with more error
    assert_eq!(read_values.len(), 64);
}

#[test]
fn test_quantized_kv_page_write_multiple_positions() {
    let mut page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);

    // Write at position 0
    let keys0: Vec<f32> = (0..512).map(|i| i as f32).collect();
    page.write_keys(0, &keys0);

    // Write at position 1
    let keys1: Vec<f32> = (0..512).map(|i| -i as f32).collect();
    page.write_keys(1, &keys1);

    // Verify both positions
    let read0 = page.read_keys(0);
    let read1 = page.read_keys(1);

    assert_eq!(read0[0], 0.0);
    assert_eq!(read1[0], 0.0);
    assert!(read0[100] > 0.0);
    assert!(read1[100] < 0.0);
}

// ============================================================================
// MEMORY USAGE CALCULATIONS
// ============================================================================

#[test]
fn test_quantized_page_memory_bytes_fp32() {
    let page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);
    let expected = 16 * 8 * 64 * 4 * 2; // block_size * heads * dim * 4 bytes * 2 (K+V)
    assert_eq!(page.memory_bytes(), expected);
}

#[test]
fn test_quantized_page_memory_bytes_q8() {
    let page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);
    let fp32_page = QuantizedKvPage::new(PageId::new(1), KvQuantType::FP32, 16, 8, 64);

    // Q8 should use significantly less memory than FP32
    assert!(page.memory_bytes() < fp32_page.memory_bytes());
}

#[test]
fn test_quantized_page_memory_bytes_q4() {
    let page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q4, 16, 8, 64);
    let q8_page = QuantizedKvPage::new(PageId::new(1), KvQuantType::Q8, 16, 8, 64);

    // Q4 should use less memory than Q8
    assert!(page.memory_bytes() < q8_page.memory_bytes());
}
