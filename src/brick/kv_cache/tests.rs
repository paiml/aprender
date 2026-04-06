use super::*;

// =========================================================================
// KvCacheSlotInfo Tests
// =========================================================================

#[test]
fn test_kv_cache_slot_info_new() {
    let slot = KvCacheSlotInfo::new(5, 42, 2, 8);

    assert_eq!(slot.position, 5);
    assert_eq!(slot.token_id, 42);
    assert_eq!(slot.layer, 2);
    assert_eq!(slot.head, 8);
    assert!(slot.valid);
    assert_eq!(slot.last_access, 0);
}

#[test]
fn test_kv_cache_slot_info_default() {
    let slot = KvCacheSlotInfo::default();

    assert_eq!(slot.position, 0);
    assert_eq!(slot.token_id, 0);
    assert_eq!(slot.layer, 0);
    assert_eq!(slot.head, 0);
    assert!(!slot.valid);
    assert_eq!(slot.last_access, 0);
}

#[test]
fn test_kv_cache_slot_info_touch() {
    let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);

    slot.touch(10);
    assert_eq!(slot.last_access, 10);

    slot.touch(50);
    assert_eq!(slot.last_access, 50);
}

#[test]
fn test_kv_cache_slot_info_invalidate() {
    let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
    assert!(slot.valid);

    slot.invalidate();
    assert!(!slot.valid);
}

#[test]
fn test_kv_cache_slot_info_eviction_priority_valid() {
    let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
    slot.touch(10);

    // Same step as last access = priority 0
    assert_eq!(slot.eviction_priority(10), 0);

    // Steps since last access
    assert_eq!(slot.eviction_priority(20), 10);
    assert_eq!(slot.eviction_priority(100), 90);
}

#[test]
fn test_kv_cache_slot_info_eviction_priority_invalid() {
    let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
    slot.invalidate();

    // Invalid slots have MAX priority (should be evicted first)
    assert_eq!(slot.eviction_priority(0), u64::MAX);
    assert_eq!(slot.eviction_priority(100), u64::MAX);
}

// =========================================================================
// KvCacheManager Tests
// =========================================================================

#[test]
fn test_kv_cache_manager_new() {
    let mgr = KvCacheManager::new(10);

    assert_eq!(mgr.capacity(), 10);
    assert_eq!(mgr.valid_count(), 0);
}

#[test]
fn test_kv_cache_manager_allocate() {
    let mut mgr = KvCacheManager::new(3);

    let idx0 = mgr.allocate(0, 100, 0, 0);
    assert_eq!(idx0, Some(0));
    assert_eq!(mgr.valid_count(), 1);

    let idx1 = mgr.allocate(1, 101, 0, 0);
    assert_eq!(idx1, Some(1));
    assert_eq!(mgr.valid_count(), 2);

    let idx2 = mgr.allocate(2, 102, 0, 0);
    assert_eq!(idx2, Some(2));
    assert_eq!(mgr.valid_count(), 3);

    // Full - no more slots
    let idx3 = mgr.allocate(3, 103, 0, 0);
    assert_eq!(idx3, None);
    assert_eq!(mgr.valid_count(), 3);
}

#[test]
fn test_kv_cache_manager_access() {
    let mut mgr = KvCacheManager::new(3);

    let idx = mgr.allocate(0, 42, 1, 2).unwrap();

    let slot = mgr.access(idx).unwrap();
    assert_eq!(slot.token_id, 42);
    assert_eq!(slot.layer, 1);
    assert_eq!(slot.head, 2);

    // Out of bounds access
    assert!(mgr.access(100).is_none());
}

#[test]
fn test_kv_cache_manager_step() {
    let mut mgr = KvCacheManager::new(2);
    let idx = mgr.allocate(0, 0, 0, 0).unwrap();

    // Step advances counter
    mgr.step();
    mgr.step();

    // Access updates last_access to current step
    let slot = mgr.access(idx).unwrap();
    assert_eq!(slot.last_access, 2);
}

#[test]
fn test_kv_cache_manager_evict_lru() {
    let mut mgr = KvCacheManager::new(3);

    // Allocate with steps between to create LRU ordering
    let idx0 = mgr.allocate(0, 100, 0, 0).unwrap();
    mgr.step();
    let idx1 = mgr.allocate(1, 101, 0, 0).unwrap();
    mgr.step();
    let _idx2 = mgr.allocate(2, 102, 0, 0).unwrap();
    mgr.step();

    // Access idx0 to make it recently used
    mgr.access(idx0);

    // LRU should be idx1 (oldest access without recent touch)
    let evicted = mgr.evict_lru();
    assert_eq!(evicted, Some(idx1));
    assert_eq!(mgr.valid_count(), 2);
}

#[test]
fn test_kv_cache_manager_evict_empty() {
    let mut mgr = KvCacheManager::new(3);

    // No valid slots - nothing to evict
    let evicted = mgr.evict_lru();
    assert_eq!(evicted, None);
}

#[test]
fn test_kv_cache_manager_allocate_after_evict() {
    let mut mgr = KvCacheManager::new(2);

    // Fill up
    mgr.allocate(0, 100, 0, 0).unwrap();
    mgr.step();
    mgr.allocate(1, 101, 0, 0).unwrap();
    assert_eq!(mgr.valid_count(), 2);

    // Full
    assert!(mgr.allocate(2, 102, 0, 0).is_none());

    // Evict one
    mgr.evict_lru();
    assert_eq!(mgr.valid_count(), 1);

    // Now can allocate
    let idx = mgr.allocate(2, 102, 0, 0);
    assert!(idx.is_some());
    assert_eq!(mgr.valid_count(), 2);
}

// =========================================================================
// SequentialBatchOrderer Tests
// =========================================================================

#[test]
fn test_sequential_batch_orderer_new() {
    let orderer = SequentialBatchOrderer::new(5);

    assert_eq!(orderer.remaining(), 5);
    assert!(!orderer.is_done());
}

#[test]
fn test_sequential_batch_orderer_sequential() {
    let mut orderer = SequentialBatchOrderer::new(4);

    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(3));
    assert_eq!(orderer.next_batch(), None);
    assert!(orderer.is_done());
}

#[test]
fn test_sequential_batch_orderer_reversed() {
    let mut orderer = SequentialBatchOrderer::reversed(4);

    assert_eq!(orderer.next_batch(), Some(3));
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), None);
}

#[test]
fn test_sequential_batch_orderer_interleaved_even() {
    let orderer = SequentialBatchOrderer::interleaved(4);
    let order: Vec<_> = orderer.collect();

    // 4 batches: 0, 2, 1, 3
    assert_eq!(order, vec![0, 2, 1, 3]);
}

#[test]
fn test_sequential_batch_orderer_interleaved_odd() {
    let orderer = SequentialBatchOrderer::interleaved(5);
    let order: Vec<_> = orderer.collect();

    // All indices should be present
    assert_eq!(order.len(), 5);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_sequential_batch_orderer_reset() {
    let mut orderer = SequentialBatchOrderer::new(3);

    orderer.next_batch();
    orderer.next_batch();
    assert_eq!(orderer.remaining(), 1);

    orderer.reset();
    assert_eq!(orderer.remaining(), 3);
    assert!(!orderer.is_done());
}

#[test]
fn test_sequential_batch_orderer_remaining() {
    let mut orderer = SequentialBatchOrderer::new(5);

    assert_eq!(orderer.remaining(), 5);
    orderer.next_batch();
    assert_eq!(orderer.remaining(), 4);
    orderer.next_batch();
    assert_eq!(orderer.remaining(), 3);
}

#[test]
fn test_sequential_batch_orderer_iterator() {
    let orderer = SequentialBatchOrderer::new(3);
    let batches: Vec<_> = orderer.collect();

    assert_eq!(batches, vec![0, 1, 2]);
}

#[test]
fn test_sequential_batch_orderer_empty() {
    let mut orderer = SequentialBatchOrderer::new(0);

    assert!(orderer.is_done());
    assert_eq!(orderer.remaining(), 0);
    assert_eq!(orderer.next_batch(), None);
}

// =========================================================================
// FALSIFICATION TESTS
// =========================================================================

/// FALSIFICATION TEST: KvCacheManager must maintain valid_count invariant
///
/// valid_count must always equal the number of slots with valid=true
#[test]
fn test_falsify_kv_cache_valid_count_invariant() {
    let mut mgr = KvCacheManager::new(5);

    // Allocate some
    mgr.allocate(0, 0, 0, 0);
    mgr.allocate(1, 0, 0, 0);
    mgr.allocate(2, 0, 0, 0);

    assert_eq!(
        mgr.valid_count(),
        3,
        "FALSIFICATION FAILED: valid_count should be 3 after 3 allocations"
    );

    // Evict some
    mgr.evict_lru();
    assert_eq!(
        mgr.valid_count(),
        2,
        "FALSIFICATION FAILED: valid_count should be 2 after 1 eviction"
    );

    mgr.evict_lru();
    mgr.evict_lru();
    assert_eq!(
        mgr.valid_count(),
        0,
        "FALSIFICATION FAILED: valid_count should be 0 after all evictions"
    );

    // Can't go negative
    mgr.evict_lru();
    assert_eq!(mgr.valid_count(), 0, "FALSIFICATION FAILED: valid_count should not go negative");
}

/// FALSIFICATION TEST: LRU eviction must evict oldest-accessed slot
///
/// Given slots with different access times, evict_lru must always
/// return the slot with the oldest (smallest) last_access.
#[test]
fn test_falsify_lru_eviction_order() {
    let mut mgr = KvCacheManager::new(4);

    // Allocate 4 slots with step advances
    let idx0 = mgr.allocate(0, 0, 0, 0).unwrap();
    mgr.step();
    let idx1 = mgr.allocate(1, 0, 0, 0).unwrap();
    mgr.step();
    let idx2 = mgr.allocate(2, 0, 0, 0).unwrap();
    mgr.step();
    let idx3 = mgr.allocate(3, 0, 0, 0).unwrap();
    mgr.step();

    // Access slots in specific order to set LRU
    // Touch idx2, then idx0 - making idx1, idx3 the oldest
    mgr.access(idx2);
    mgr.step();
    mgr.access(idx0);
    mgr.step();

    // First eviction should be idx1 (oldest not accessed)
    let evicted1 = mgr.evict_lru().unwrap();
    assert_eq!(evicted1, idx1, "FALSIFICATION FAILED: First eviction should be idx1 (step 1)");

    // Second eviction should be idx3 (next oldest)
    let evicted2 = mgr.evict_lru().unwrap();
    assert_eq!(evicted2, idx3, "FALSIFICATION FAILED: Second eviction should be idx3 (step 3)");
}

/// FALSIFICATION TEST: SequentialBatchOrderer must cover all indices
///
/// After iteration completes, all indices 0..n must have been returned
/// exactly once.
#[test]
fn test_falsify_batch_orderer_covers_all_indices() {
    for n_batches in [1, 2, 3, 5, 10, 100] {
        for orderer in [
            SequentialBatchOrderer::new(n_batches),
            SequentialBatchOrderer::reversed(n_batches),
            SequentialBatchOrderer::interleaved(n_batches),
        ] {
            let batches: Vec<_> = orderer.collect();

            // Must have exactly n_batches elements
            assert_eq!(
                batches.len(),
                n_batches,
                "FALSIFICATION FAILED: orderer returned {} batches, expected {}",
                batches.len(),
                n_batches
            );

            // Sort and verify all indices present
            let mut sorted = batches.clone();
            sorted.sort_unstable();
            let expected: Vec<_> = (0..n_batches).collect();
            assert_eq!(
                sorted, expected,
                "FALSIFICATION FAILED: orderer did not return all indices 0..{}",
                n_batches
            );
        }
    }
}

/// FALSIFICATION TEST: Eviction priority must be monotonic with age
///
/// For valid slots, older access time = higher eviction priority.
#[test]
fn test_falsify_eviction_priority_monotonic() {
    let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
    slot.touch(10);

    let mut prev_priority = slot.eviction_priority(10);
    for current_step in 11..=100 {
        let priority = slot.eviction_priority(current_step);
        assert!(
            priority >= prev_priority,
            "FALSIFICATION FAILED: eviction priority decreased from {} to {} at step {}",
            prev_priority,
            priority,
            current_step
        );
        prev_priority = priority;
    }
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_kv_cache_manager_single_slot() {
    let mut mgr = KvCacheManager::new(1);
    assert_eq!(mgr.capacity(), 1);
    assert_eq!(mgr.valid_count(), 0);

    let idx = mgr.allocate(0, 99, 3, 7).unwrap();
    assert_eq!(idx, 0);
    assert_eq!(mgr.valid_count(), 1);

    // Cannot allocate more
    assert!(mgr.allocate(1, 100, 0, 0).is_none());

    // Evict the only slot
    let evicted = mgr.evict_lru();
    assert_eq!(evicted, Some(0));
    assert_eq!(mgr.valid_count(), 0);

    // Allocate again into the freed slot
    let idx2 = mgr.allocate(1, 100, 0, 0).unwrap();
    assert_eq!(idx2, 0);
}

#[test]
fn test_kv_cache_manager_evict_all_same_step() {
    let mut mgr = KvCacheManager::new(3);

    // All allocated at step 0 (same time)
    mgr.allocate(0, 10, 0, 0);
    mgr.allocate(1, 20, 0, 0);
    mgr.allocate(2, 30, 0, 0);

    // All have same priority, should evict first found (index 0)
    let evicted = mgr.evict_lru();
    assert_eq!(evicted, Some(0));
    assert_eq!(mgr.valid_count(), 2);
}

#[test]
fn test_kv_cache_slot_eviction_priority_saturating() {
    let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
    slot.touch(u64::MAX);

    // current_step < last_access: saturating_sub returns 0
    assert_eq!(slot.eviction_priority(0), 0);
    assert_eq!(slot.eviction_priority(100), 0);
}

#[test]
fn test_kv_cache_manager_allocate_with_layer_head() {
    let mut mgr = KvCacheManager::new(2);

    let idx = mgr.allocate(5, 42, 12, 8).unwrap();
    let slot = mgr.access(idx).unwrap();
    assert_eq!(slot.position, 5);
    assert_eq!(slot.token_id, 42);
    assert_eq!(slot.layer, 12);
    assert_eq!(slot.head, 8);
    assert!(slot.valid);
}

#[test]
fn test_kv_cache_manager_evict_realloc_cycle() {
    let mut mgr = KvCacheManager::new(2);

    // Fill, evict, refill multiple times
    for round in 0..3 {
        let base = round * 2;
        mgr.allocate(base as u32, base as u32, 0, 0);
        mgr.step();
        mgr.allocate((base + 1) as u32, (base + 1) as u32, 0, 0);
        assert_eq!(mgr.valid_count(), 2);

        // Evict both
        mgr.evict_lru();
        mgr.evict_lru();
        assert_eq!(mgr.valid_count(), 0);
    }
}

#[test]
fn test_sequential_batch_orderer_interleaved_one() {
    let orderer = SequentialBatchOrderer::interleaved(1);
    let order: Vec<_> = orderer.collect();
    assert_eq!(order, vec![0]);
}

#[test]
fn test_sequential_batch_orderer_interleaved_two() {
    let orderer = SequentialBatchOrderer::interleaved(2);
    let order: Vec<_> = orderer.collect();
    // mid=1, loop i=0: push 0, push 1
    assert_eq!(order, vec![0, 1]);
}

#[test]
fn test_sequential_batch_orderer_interleaved_three() {
    let orderer = SequentialBatchOrderer::interleaved(3);
    let order: Vec<_> = orderer.collect();
    // mid=1, loop i=0: push 0, push 1, then odd: push 2
    assert_eq!(order.len(), 3);
    let mut sorted = order;
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn test_sequential_batch_orderer_reversed_empty() {
    let orderer = SequentialBatchOrderer::reversed(0);
    let order: Vec<_> = orderer.collect();
    assert!(order.is_empty());
}

#[test]
fn test_sequential_batch_orderer_reversed_single() {
    let mut orderer = SequentialBatchOrderer::reversed(1);
    assert_eq!(orderer.remaining(), 1);
    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), None);
    assert!(orderer.is_done());
}

#[test]
fn test_sequential_batch_orderer_reset_after_done() {
    let mut orderer = SequentialBatchOrderer::new(2);
    // Exhaust
    orderer.next_batch();
    orderer.next_batch();
    assert!(orderer.is_done());
    assert_eq!(orderer.remaining(), 0);

    // Reset and iterate again
    orderer.reset();
    assert!(!orderer.is_done());
    assert_eq!(orderer.remaining(), 2);
    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), None);
}

#[test]
fn test_kv_cache_slot_clone() {
    let slot = KvCacheSlotInfo::new(10, 20, 3, 5);
    let cloned = slot.clone();
    assert_eq!(cloned.position, 10);
    assert_eq!(cloned.token_id, 20);
    assert_eq!(cloned.layer, 3);
    assert_eq!(cloned.head, 5);
    assert!(cloned.valid);
}

#[test]
fn test_sequential_batch_orderer_clone() {
    let mut orderer = SequentialBatchOrderer::new(4);
    orderer.next_batch(); // advance position to 1
    let cloned = orderer.clone();
    assert_eq!(cloned.remaining(), 3);
}
