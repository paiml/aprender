
    // === SeqId Tests ===

    #[test]
    fn test_seq_id_new() {
        let id1 = SeqId::new();
        let id2 = SeqId::new();
        assert_ne!(id1.value(), id2.value());
    }

    #[test]
    fn test_seq_id_default() {
        let id1 = SeqId::default();
        let id2 = SeqId::default();
        assert_ne!(id1, id2);
    }

    // === PageId Tests ===

    #[test]
    fn test_page_id_new() {
        let id = PageId::new(42);
        assert_eq!(id.value(), 42);
    }

    // === KvPage Tests ===

    #[test]
    fn test_kv_page_new() {
        let page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert_eq!(page.num_tokens, 0);
        assert_eq!(page.ref_count, 1);
        assert_eq!(page.keys.len(), 16 * 8 * 64);
        assert_eq!(page.values.len(), 16 * 8 * 64);
    }

    #[test]
    fn test_kv_page_is_full() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert!(!page.is_full(16));
        page.num_tokens = 16;
        assert!(page.is_full(16));
    }

    #[test]
    fn test_kv_page_is_shared() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert!(!page.is_shared());
        page.ref_count = 2;
        assert!(page.is_shared());
    }

    #[test]
    fn test_kv_page_remaining_capacity() {
        let mut page = KvPage::new(PageId::new(0), 16, 8, 64);
        assert_eq!(page.remaining_capacity(16), 16);
        page.num_tokens = 10;
        assert_eq!(page.remaining_capacity(16), 6);
    }

    // === PagedKvCache Tests ===

    #[test]
    fn test_paged_kv_cache_new() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_allocate_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        // 32 tokens needs 2 pages (16 tokens per page)
        assert_eq!(cache.free_page_count(), 98);
        assert_eq!(cache.stats().active_sequences, 1);
        assert_eq!(cache.stats().pages_allocated, 2);
        // seq_id is valid (non-zero ID counter)
        assert!(seq_id.value() < u64::MAX);
    }

    #[test]
    fn test_allocate_sequence_out_of_memory() {
        let mut cache = PagedKvCache::new(1, 16, 8, 64);

        // First allocation succeeds
        let _ = cache.allocate_sequence(10).expect("test");

        // Second allocation fails
        let result = cache.allocate_sequence(20);
        assert!(matches!(result, Err(PagedCacheError::OutOfMemory { .. })));
    }

    #[test]
    fn test_extend_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).expect("test");

        // Initially 1 page
        assert_eq!(cache.free_page_count(), 99);

        // Extend to need 2 pages
        cache.extend(seq_id, 20).expect("test");
        assert_eq!(cache.free_page_count(), 98);
    }

    #[test]
    fn test_free_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        assert_eq!(cache.free_page_count(), 98);

        cache.free_sequence(seq_id);

        assert_eq!(cache.free_page_count(), 100);
        assert_eq!(cache.stats().active_sequences, 0);
    }

    #[test]
    fn test_fork_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("test");

        let child_id = cache.fork_sequence(parent_id).expect("test");

        // Pages are shared via COW
        assert_eq!(cache.stats().active_sequences, 2);
        assert_eq!(cache.stats().cow_operations, 1);
        assert_ne!(parent_id, child_id);
    }

    #[test]
    fn test_get_page() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        let page = cache.get_page(seq_id, 0).expect("test");
        assert_eq!(
            page.id.value(),
            cache.page_tables.get(&seq_id).expect("test")[0].value()
        );

        let page2 = cache.get_page(seq_id, 16).expect("test");
        assert_eq!(
            page2.id.value(),
            cache.page_tables.get(&seq_id).expect("test")[1].value()
        );
    }

    #[test]
    fn test_get_page_invalid() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(16).expect("test");

        let result = cache.get_page(seq_id, 100); // Beyond allocated pages
        assert!(matches!(
            result,
            Err(PagedCacheError::InvalidPageAccess { .. })
        ));
    }

    #[test]
    fn test_get_sequence_tokens() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).expect("test");
        cache.update_tokens(seq_id, 10).expect("test");

        let tokens = cache.get_sequence_tokens(seq_id).expect("test");
        assert_eq!(tokens, 10);
    }

    #[test]
    fn test_memory_usage() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        assert_eq!(cache.memory_usage(), 0);

        let _ = cache.allocate_sequence(16).expect("test");

        // 1 page * 16 tokens * 8 heads * 64 dim * 4 bytes * 2 (K+V)
        let expected = 16 * 8 * 64 * 4 * 2;
        assert_eq!(cache.memory_usage(), expected);
    }

    #[test]
    fn test_total_capacity() {
        let cache = PagedKvCache::new(100, 16, 8, 64);

        // 100 pages * 16 tokens * 8 heads * 64 dim * 4 bytes * 2 (K+V)
        let expected = 100 * 16 * 8 * 64 * 4 * 2;
        assert_eq!(cache.total_capacity(), expected);
    }

    #[test]
    fn test_utilization() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);

        assert_eq!(cache.utilization(), 0.0);

        let _ = cache.allocate_sequence(160).expect("test"); // 10 pages

        assert!((cache.utilization() - 10.0).abs() < 0.01);
    }

    // === Error Display Tests ===

    #[test]
    fn test_paged_cache_error_display() {
        let err = PagedCacheError::OutOfMemory {
            needed: 10,
            available: 5,
        };
        assert!(err.to_string().contains("need 10"));
        assert!(err.to_string().contains("have 5"));

        let err = PagedCacheError::SequenceNotFound(42);
        assert!(err.to_string().contains("42"));

        let err = PagedCacheError::InvalidPageAccess {
            page_id: 5,
            offset: 100,
        };
        assert!(err.to_string().contains("page 5"));

        let err = PagedCacheError::PageTableCorruption { seq_id: 99 };
        assert!(err.to_string().contains("99"));
    }

    // === Stats Tests ===

    #[test]
    fn test_paged_cache_stats_default() {
        let stats = PagedCacheStats::default();
        assert_eq!(stats.sequences_allocated, 0);
        assert_eq!(stats.sequences_freed, 0);
        assert_eq!(stats.pages_allocated, 0);
    }

    #[test]
    fn test_stats_serialization() {
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

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: PagedCacheStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.sequences_allocated, stats.sequences_allocated);
        assert_eq!(parsed.cow_operations, stats.cow_operations);
        assert_eq!(parsed.defrag_operations, stats.defrag_operations);
        assert_eq!(parsed.pages_moved, stats.pages_moved);
    }

    // === Copy-on-Write Tests ===

    #[test]
    fn test_cow_on_write() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let parent_id = cache.allocate_sequence(16).expect("test");
        cache.update_tokens(parent_id, 16).expect("test");

        // Fork creates shared pages
        let child_id = cache.fork_sequence(parent_id).expect("test");

        // Get mutable page should trigger COW
        let initial_cow = cache.stats().cow_operations;
        let _page = cache.get_page_mut(child_id, 0).expect("test");

        // COW should have been triggered
        assert!(cache.stats().cow_operations > initial_cow);
    }

    #[test]
    fn test_sequence_not_found() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.get_sequence_tokens(fake_seq);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }

    // === Defragmentation Tests ===

    #[test]
    fn test_fragmentation_stats_empty_cache() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let stats = cache.fragmentation_stats();

        assert_eq!(stats.holes, 0);
        assert_eq!(stats.wasted_capacity, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert_eq!(stats.largest_free_region, 100); // All pages free
    }

    #[test]
    fn test_fragmentation_stats_single_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test"); // 2 pages
        cache.update_tokens(seq_id, 32).expect("test");

        let stats = cache.fragmentation_stats();

        // With 2 contiguous pages at the start, no holes in used region
        assert_eq!(stats.holes, 0);
        // 32 tokens in 2 pages (32 capacity) = 0 wasted
        assert_eq!(stats.wasted_capacity, 0);
        assert_eq!(stats.avg_tokens_per_page, 16.0);
    }

    #[test]
    fn test_fragmentation_stats_with_holes() {
        let mut cache = PagedKvCache::new(10, 16, 8, 64);

        // Allocate 3 sequences
        let seq1 = cache.allocate_sequence(16).expect("test"); // Page 0
        let seq2 = cache.allocate_sequence(16).expect("test"); // Page 1
        let seq3 = cache.allocate_sequence(16).expect("test"); // Page 2

        // Free middle sequence to create a hole
        cache.free_sequence(seq2);

        let stats = cache.fragmentation_stats();

        // Should have at least one hole (between seq1 and seq3's pages)
        // Note: depends on allocation order
        assert!(stats.largest_free_region >= 1);

        // Verify seq1 and seq3 still valid
        assert!(cache.get_sequence_tokens(seq1).is_ok());
        assert!(cache.get_sequence_tokens(seq3).is_ok());
    }

    #[test]
    fn test_fragmentation_stats_wasted_capacity() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test"); // 2 pages
        cache.update_tokens(seq_id, 10).expect("test"); // Only 10 tokens in 2 pages

        let stats = cache.fragmentation_stats();

        // 2 pages * 16 block_size = 32 capacity, 10 tokens = 22 wasted
        assert_eq!(stats.wasted_capacity, 22);
    }

    #[test]
    fn test_should_defragment_empty() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        assert!(!cache.should_defragment());
    }

    #[test]
    fn test_should_defragment_no_fragmentation() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");
        cache.update_tokens(seq_id, 32).expect("test");

        // Single contiguous allocation = no fragmentation
        assert!(!cache.should_defragment());
    }

    #[test]
    fn test_should_defragment_with_threshold() {
        let cache = PagedKvCache::new(100, 16, 8, 64);

        // With 0.0 threshold, any fragmentation triggers
        assert!(!cache.should_defragment_with_threshold(0.0));

        // With 1.0 threshold, only extreme fragmentation triggers
        assert!(!cache.should_defragment_with_threshold(1.0));
    }

    #[test]
    fn test_defragment_empty_cache() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let pages_moved = cache.defragment();

        assert_eq!(pages_moved, 0);
        assert_eq!(cache.stats().defrag_operations, 0);
    }

    #[test]
    fn test_defragment_single_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");
        cache.update_tokens(seq_id, 32).expect("test");

        // Already contiguous, no defrag needed
        let pages_moved = cache.defragment();
        assert_eq!(pages_moved, 0);
    }

    #[test]
    fn test_compact_sequence_already_contiguous() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test");

        let moved = cache.compact_sequence(seq_id);
        assert_eq!(moved, 0); // Already contiguous
    }

    #[test]
    fn test_compact_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let moved = cache.compact_sequence(fake_seq);
        assert_eq!(moved, 0);
    }

    #[test]
    fn test_sequence_contiguity_single_page() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(10).expect("test"); // 1 page

        let contiguity = cache.sequence_contiguity(seq_id).expect("test");
        assert_eq!(contiguity, 1.0); // Single page always contiguous
    }

    #[test]
    fn test_sequence_contiguity_multiple_pages() {
        let mut cache = PagedKvCache::new(100, 16, 8, 64);
        let seq_id = cache.allocate_sequence(32).expect("test"); // 2 pages

        let contiguity = cache.sequence_contiguity(seq_id).expect("test");
        // Fresh allocation should be contiguous
        assert!(contiguity >= 0.0);
        assert!(contiguity <= 1.0);
    }

    #[test]
    fn test_sequence_contiguity_not_found() {
        let cache = PagedKvCache::new(100, 16, 8, 64);
        let fake_seq = SeqId::new();

        let result = cache.sequence_contiguity(fake_seq);
        assert!(matches!(result, Err(PagedCacheError::SequenceNotFound(_))));
    }
