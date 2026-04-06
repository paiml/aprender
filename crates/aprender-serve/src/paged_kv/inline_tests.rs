use super::*;

// =========================================================================
// SeqId Tests
// =========================================================================

#[test]
fn test_seq_id_new() {
    let id1 = SeqId::new();
    let id2 = SeqId::new();
    // Each new ID should be unique (incrementing)
    assert_ne!(id1.value(), id2.value());
}

#[test]
fn test_seq_id_default() {
    let id = SeqId::default();
    assert!(id.value() < u64::MAX);
}

#[test]
fn test_seq_id_value() {
    // SeqId wraps a u64
    let id = SeqId::new();
    let _ = id.value(); // Should not panic
}

#[test]
fn test_seq_id_equality() {
    let id1 = SeqId::new();
    let id2 = id1; // Copy
    assert_eq!(id1, id2);
}

#[test]
fn test_seq_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    let id1 = SeqId::new();
    let id2 = SeqId::new();
    set.insert(id1);
    set.insert(id2);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_seq_id_clone() {
    let id1 = SeqId::new();
    let id2 = id1;
    assert_eq!(id1.value(), id2.value());
}

// =========================================================================
// PageId Tests
// =========================================================================

#[test]
fn test_page_id_new() {
    let id = PageId::new(42);
    assert_eq!(id.value(), 42);
}

#[test]
fn test_page_id_value() {
    let id = PageId::new(100);
    assert_eq!(id.value(), 100);
}

#[test]
fn test_page_id_equality() {
    let id1 = PageId::new(10);
    let id2 = PageId::new(10);
    let id3 = PageId::new(20);
    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}

#[test]
fn test_page_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(PageId::new(1));
    set.insert(PageId::new(2));
    set.insert(PageId::new(1)); // Duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_page_id_clone() {
    let id1 = PageId::new(99);
    let id2 = id1;
    assert_eq!(id1, id2);
}

// =========================================================================
// KvPage Tests
// =========================================================================

#[test]
fn test_kv_page_new() {
    let page = KvPage::new(PageId::new(0), 16, 4, 64);
    assert_eq!(page.id, PageId::new(0));
    assert_eq!(page.num_tokens, 0);
    assert_eq!(page.ref_count, 1);
    // keys size = block_size * num_heads * head_dim = 16 * 4 * 64 = 4096
    assert_eq!(page.keys.len(), 4096);
    assert_eq!(page.values.len(), 4096);
}

#[test]
fn test_kv_page_is_full() {
    let mut page = KvPage::new(PageId::new(0), 16, 4, 64);
    assert!(!page.is_full(16));

    page.num_tokens = 16;
    assert!(page.is_full(16));

    page.num_tokens = 8;
    assert!(!page.is_full(16));
}

#[test]
fn test_kv_page_is_shared() {
    let mut page = KvPage::new(PageId::new(0), 16, 4, 64);
    assert!(!page.is_shared());

    page.ref_count = 2;
    assert!(page.is_shared());

    page.ref_count = 1;
    assert!(!page.is_shared());
}

#[test]
fn test_kv_page_remaining_capacity() {
    let mut page = KvPage::new(PageId::new(0), 16, 4, 64);
    assert_eq!(page.remaining_capacity(16), 16);

    page.num_tokens = 5;
    assert_eq!(page.remaining_capacity(16), 11);

    page.num_tokens = 16;
    assert_eq!(page.remaining_capacity(16), 0);

    page.num_tokens = 20; // Overflow case
    assert_eq!(page.remaining_capacity(16), 0);
}

// =========================================================================
// PagedCacheError Tests
// =========================================================================

#[test]
fn test_error_out_of_memory() {
    let err = PagedCacheError::OutOfMemory {
        needed: 10,
        available: 5,
    };
    let msg = err.to_string();
    assert!(msg.contains("Out of memory"));
    assert!(msg.contains("10"));
    assert!(msg.contains("5"));
}

#[test]
fn test_error_sequence_not_found() {
    let err = PagedCacheError::SequenceNotFound(42);
    let msg = err.to_string();
    assert!(msg.contains("Sequence not found"));
    assert!(msg.contains("42"));
}

#[test]
fn test_error_invalid_page_access() {
    let err = PagedCacheError::InvalidPageAccess {
        page_id: 5,
        offset: 100,
    };
    let msg = err.to_string();
    assert!(msg.contains("Invalid page access"));
    assert!(msg.contains("5"));
    assert!(msg.contains("100"));
}

#[test]
fn test_error_page_table_corruption() {
    let err = PagedCacheError::PageTableCorruption { seq_id: 99 };
    let msg = err.to_string();
    assert!(msg.contains("Page table corruption"));
    assert!(msg.contains("99"));
}

// =========================================================================
// PagedCacheStats Tests
// =========================================================================

#[test]
fn test_paged_cache_stats_default() {
    let stats = PagedCacheStats::default();
    assert_eq!(stats.used_pages, 0);
    assert_eq!(stats.active_sequences, 0);
    assert_eq!(stats.cow_operations, 0);
    assert_eq!(stats.pages_allocated, 0);
    assert_eq!(stats.pages_freed, 0);
}

#[test]
fn test_paged_cache_stats_clone() {
    let stats = PagedCacheStats {
        sequences_allocated: 5,
        sequences_freed: 2,
        pages_allocated: 10,
        pages_freed: 3,
        active_sequences: 3,
        used_pages: 7,
        cow_operations: 1,
        defrag_operations: 0,
        pages_moved: 0,
    };
    let cloned = stats.clone();
    assert_eq!(stats.used_pages, cloned.used_pages);
    assert_eq!(stats.active_sequences, cloned.active_sequences);
    assert_eq!(stats.cow_operations, cloned.cow_operations);
}

// =========================================================================
// PagedKvCache Tests
// =========================================================================

#[test]
fn test_paged_kv_cache_new() {
    let cache = PagedKvCache::new(100, 16, 8, 64);
    assert_eq!(cache.free_page_count(), 100);
}

#[test]
fn test_paged_kv_cache_allocate_sequence() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);

    let result = cache.allocate_sequence(32);
    assert!(result.is_ok());

    let seq_id = result.unwrap();
    assert!(seq_id.value() < u64::MAX);

    // 32 tokens / 16 block_size = 2 pages
    assert!(cache.free_page_count() < 100);
}

#[test]
fn test_paged_kv_cache_free_sequence() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);

    let seq_id = cache.allocate_sequence(32).expect("allocate");
    let free_before = cache.free_page_count();

    cache.free_sequence(seq_id);
    let free_after = cache.free_page_count();

    assert!(free_after > free_before);
}

#[test]
fn test_paged_kv_cache_stats() {
    let cache = PagedKvCache::new(100, 16, 8, 64);
    let stats = cache.stats();
    assert_eq!(stats.used_pages, 0);
    assert_eq!(stats.active_sequences, 0);
}

#[test]
fn test_paged_kv_cache_memory_usage_empty() {
    let cache = PagedKvCache::new(100, 16, 8, 64);
    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn test_paged_kv_cache_allocate_multiple() {
    let mut cache = PagedKvCache::new(100, 16, 8, 64);

    let seq1 = cache.allocate_sequence(16).expect("allocate 1");
    let seq2 = cache.allocate_sequence(32).expect("allocate 2");

    // Both should have unique IDs
    assert_ne!(seq1.value(), seq2.value());

    // Stats should reflect 2 active sequences
    assert_eq!(cache.stats().active_sequences, 2);
}
