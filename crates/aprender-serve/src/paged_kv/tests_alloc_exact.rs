//! Part 2: Page allocation edge cases and memory management tests
//!
//! Focus areas:
//! 1. Page allocation edge cases
//! 2. Memory management

#[cfg(test)]
mod tests {
    use crate::paged_kv::*;

    // =========================================================================
    // Page Allocation Edge Cases
    // =========================================================================

    #[test]
    fn test_alloc_exact_page_boundary() {
        // Allocate exactly block_size tokens - should use exactly 1 page
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let seq = cache.allocate_sequence(16).expect("alloc");
        assert_eq!(cache.free_page_count(), 9);
        assert_eq!(cache.stats().pages_allocated, 1);
        cache.free_sequence(seq);
    }

    #[test]
    fn test_alloc_one_over_boundary() {
        // Allocate block_size + 1 tokens - should use 2 pages
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let seq = cache.allocate_sequence(17).expect("alloc");
        assert_eq!(cache.free_page_count(), 8);
        assert_eq!(cache.stats().pages_allocated, 2);
        cache.free_sequence(seq);
    }

    #[test]
    fn test_alloc_all_pages_then_free() {
        let mut cache = PagedKvCache::new(5, 16, 8, 64);

        // Allocate all pages
        let seqs: Vec<SeqId> = (0..5)
            .map(|_| cache.allocate_sequence(16).expect("alloc"))
            .collect();

        assert_eq!(cache.free_page_count(), 0);
        assert_eq!(cache.stats().pages_allocated, 5);

        // Try to allocate more - should fail
        let result = cache.allocate_sequence(1);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));

        // Free all and reallocate
        for seq in seqs {
            cache.free_sequence(seq);
        }

        assert_eq!(cache.free_page_count(), 5);
        let _new_seq = cache.allocate_sequence(80).expect("reallocate all 5");
        assert_eq!(cache.free_page_count(), 0);
    }

    #[test]
    fn test_alloc_single_token() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let seq = cache.allocate_sequence(1).expect("alloc 1 token");
        assert_eq!(cache.free_page_count(), 9);
        cache.free_sequence(seq);
        assert_eq!(cache.free_page_count(), 10);
    }

    #[test]
    fn test_alloc_large_sequence_spanning_many_pages() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq = cache.allocate_sequence(250).expect("alloc 250 tokens");
        // 250 / 16 = 15.625 -> 16 pages
        assert_eq!(cache.free_page_count(), 84);
        assert_eq!(cache.stats().pages_allocated, 16);
        cache.free_sequence(seq);
        assert_eq!(cache.free_page_count(), 100);
    }

    // =========================================================================
    // Memory Management Tests
    // =========================================================================

    #[test]
    fn test_memory_usage_increases_with_allocation() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        let initial = cache.memory_usage();
        assert_eq!(initial, 0);

        let _seq1 = cache.allocate_sequence(16).expect("s1");
        let usage1 = cache.memory_usage();
        assert!(usage1 > 0);

        let _seq2 = cache.allocate_sequence(32).expect("s2");
        let usage2 = cache.memory_usage();
        assert_eq!(usage2, usage1 * 3); // 1 + 2 = 3 pages total
    }

    #[test]
    fn test_utilization_percentage() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);

        assert_eq!(cache.utilization(), 0.0);

        let _seq1 = cache.allocate_sequence(16).expect("s1"); // 1 page
        assert!((cache.utilization() - 10.0).abs() < 0.01);

        let _seq2 = cache.allocate_sequence(32).expect("s2"); // 2 pages
        assert!((cache.utilization() - 30.0).abs() < 0.01);

        let _seq3 = cache.allocate_sequence(112).expect("s3"); // 7 pages = all 10
        assert!((cache.utilization() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_quantized_memory_reduction_q8() {
        let fp32_cache = QuantizedPagedKvCache::new(10, 16, 8, 64, KvQuantType::FP32);
        let q8_cache = QuantizedPagedKvCache::new(10, 16, 8, 64, KvQuantType::Q8);

        // Compare page memory bytes
        let fp32_page = QuantizedKvPage::new(PageId::new(0), KvQuantType::FP32, 16, 8, 64);
        let q8_page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);

        assert!(q8_page.memory_bytes() < fp32_page.memory_bytes());
        assert_eq!(fp32_cache.quant_type(), KvQuantType::FP32);
        assert_eq!(q8_cache.quant_type(), KvQuantType::Q8);
    }

    #[test]
    fn test_quantized_memory_reduction_q4() {
        let q8_page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q8, 16, 8, 64);
        let q4_page = QuantizedKvPage::new(PageId::new(0), KvQuantType::Q4, 16, 8, 64);

        // Q4 should use less memory than Q8
        assert!(q4_page.memory_bytes() < q8_page.memory_bytes());
    }

    #[test]
    fn test_fp32_equivalent_memory_calculation() {
        let mut cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        let _seq = cache.allocate_sequence(32).expect("alloc"); // 2 pages

        let fp32_equiv = cache.fp32_equivalent_memory();
        let actual = cache.memory_usage();

        // Q8 actual should be less than FP32 equivalent
        assert!(actual < fp32_equiv);
    }

    #[test]
    fn test_memory_savings_empty_cache() {
        let cache = QuantizedPagedKvCache::new(100, 16, 8, 64, KvQuantType::Q8);
        // No pages used, returns 1.0 (no savings)
        assert_eq!(cache.memory_savings(), 1.0);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_oom_error_details() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let _seq = cache.allocate_sequence(16).expect("first alloc");

        // Try to allocate more than available
        let result = cache.allocate_sequence(32); // needs 2 pages, only 1 free
        match result {
            Err(PagedCacheError::OutOfMemory { needed, available }) => {
                assert_eq!(needed, 2);
                assert_eq!(available, 1);
            },
            _ => panic!("Expected OutOfMemory error"),
        }
    }

    #[test]
    fn test_sequence_not_found_in_get_tokens() {
        let cache = PagedKvCache::new(10, 16, 8, 64);
        let fake = SeqId::new();

        let result = cache.get_sequence_tokens(fake);
        match result {
            Err(PagedCacheError::SequenceNotFound(id)) => {
                assert_eq!(id, fake.value());
            },
            _ => panic!("Expected SequenceNotFound error"),
        }
    }

    #[test]
    fn test_invalid_page_access_deep_position() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let seq = cache.allocate_sequence(16).expect("alloc"); // 1 page only

        // Try to access position beyond allocated pages
        let result = cache.get_page(seq, 1000);
        match result {
            Err(PagedCacheError::InvalidPageAccess { page_id, offset }) => {
                assert_eq!(page_id, 62); // 1000 / 16 = 62
                assert_eq!(offset, 1000);
            },
            _ => panic!("Expected InvalidPageAccess error"),
        }
    }

    #[test]
    fn test_cow_oom_during_get_page_mut() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let parent = cache.allocate_sequence(16).expect("parent"); // 1 page
        cache.update_tokens(parent, 16).expect("update");

        // Fork to create shared page (uses COW)
        let child = cache.fork_sequence(parent).expect("fork");

        // Use remaining free page
        let _other = cache.allocate_sequence(16).expect("other"); // takes last free

        // Now try get_page_mut on child - COW needs a new page but none available
        let result = cache.get_page_mut(child, 0);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_extend_oom() {
        let mut cache = PagedKvCache::new(2, 16, 8, 64);
        let seq = cache.allocate_sequence(16).expect("alloc");
        cache.update_tokens(seq, 16).expect("update");

        // Use remaining page
        let _other = cache.allocate_sequence(16).expect("other");

        // Try to extend original - needs new page but none available
        let result = cache.extend(seq, 32);
        match result {
            Err(PagedCacheError::OutOfMemory { needed, available }) => {
                assert!(needed > 0);
                assert_eq!(available, 0);
            },
            _ => panic!("Expected OutOfMemory error"),
        }
    }

    #[test]
    fn test_quantized_cache_sequence_not_found() {
        let cache = QuantizedPagedKvCache::new(10, 16, 8, 64, KvQuantType::Q8);
        let fake = SeqId::new();

        let result = cache.get_page(fake, 0);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    #[test]
    fn test_quantized_cache_invalid_page_access() {
        let mut cache = QuantizedPagedKvCache::new(10, 16, 8, 64, KvQuantType::Q8);
        let seq = cache.allocate_sequence(16).expect("alloc"); // 1 page

        let result = cache.get_page(seq, 100); // beyond allocated
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    // =========================================================================
    // COW (Copy-on-Write) Edge Cases
    // =========================================================================

    #[test]
    fn test_cow_multiple_forks() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let parent = cache.allocate_sequence(16).expect("parent");
        cache.update_tokens(parent, 8).expect("update");

        // Fork multiple times
        let child1 = cache.fork_sequence(parent).expect("fork1");
        let child2 = cache.fork_sequence(parent).expect("fork2");
        let child3 = cache.fork_sequence(child1).expect("fork3");

        assert_eq!(cache.stats().active_sequences, 4);
        assert_eq!(cache.stats().cow_operations, 3);

        // Each child has own SeqId
        assert_ne!(parent, child1);
        assert_ne!(parent, child2);
        assert_ne!(child1, child3);

        // Writing to child should trigger COW
        let _page = cache.get_page_mut(child2, 0).expect("get mut");
        assert_eq!(cache.stats().cow_operations, 4);
    }

    #[test]
    fn test_cow_ref_count_after_free() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let parent = cache.allocate_sequence(16).expect("parent");
        cache.update_tokens(parent, 16).expect("update");

        let child = cache.fork_sequence(parent).expect("fork");

        // Free child first - parent's page should still be valid
        cache.free_sequence(child);
        assert_eq!(cache.stats().active_sequences, 1);

        // Parent still has valid page
        let page = cache.get_page(parent, 0).expect("get");
        assert_eq!(page.ref_count, 1); // Back to single owner
    }

    #[test]
    fn test_cow_preserves_data() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let parent = cache.allocate_sequence(16).expect("parent");
        cache.update_tokens(parent, 16).expect("update");

        // Write data to parent
        {
            let page = cache.get_page_mut(parent, 0).expect("mut");
            page.keys[0] = 123.0;
            page.values[0] = 456.0;
        }

        // Fork
        let child = cache.fork_sequence(parent).expect("fork");

        // Modify child (triggers COW)
        {
            let page = cache.get_page_mut(child, 0).expect("mut child");
            page.keys[0] = 789.0;
        }

        // Parent should have original data
        let parent_page = cache.get_page(parent, 0).expect("get parent");
        assert_eq!(parent_page.keys[0], 123.0);
        assert_eq!(parent_page.values[0], 456.0);

        // Child should have modified data
        let child_page = cache.get_page(child, 0).expect("get child");
        assert_eq!(child_page.keys[0], 789.0);
    }

    // =========================================================================
    // Fragmentation & Defragmentation Edge Cases
    // =========================================================================

    #[test]
    fn test_fragmentation_with_zero_pages() {
        let cache = PagedKvCache::new(0, 16, 8, 64);
        let stats = cache.fragmentation_stats();
        assert_eq!(stats.holes, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_should_defrag_low_free_ratio() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);

        // Fill 9 of 10 pages (90% full)
        let seq = cache.allocate_sequence(144).expect("alloc 144");
        cache.update_tokens(seq, 100).expect("update");

        // At 90% utilization, free_ratio = 0.1
        // No holes since single sequence, so shouldn't trigger
        let should = cache.should_defragment();
        assert!(!should);
    }

    #[test]
    fn test_compact_empty_page_list() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        // compact_sequence with nonexistent seq returns 0
        let fake = SeqId::new();
        let moved = cache.compact_sequence(fake);
        assert_eq!(moved, 0);
    }

    #[test]
    fn test_defrag_increments_stats_only_when_pages_moved() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);
        let _seq = cache.allocate_sequence(32).expect("alloc");

        // Already contiguous, no moves
        let moved = cache.defragment();
        assert_eq!(moved, 0);
        assert_eq!(cache.stats().defrag_operations, 0);
        assert_eq!(cache.stats().pages_moved, 0);
    }
}
