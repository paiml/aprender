#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! PMAT-014: PagedKvCache Falsification Tests
//!
//! Falsification criteria F411-F420 from cbtop spec §18.
//!
//! # Test Coverage
//!
//! | ID | Claim | Test |
//! |----|-------|------|
//! | F411 | Block allocation succeeds up to GPU memory limit | test_f411_block_allocation_limit |
//! | F412 | Copy-on-write fork works for beam search | test_f412_cow_fork |
//! | F413 | Eviction triggers at memory threshold | test_f413_eviction_threshold |
//! | F414 | LRU eviction correct (oldest access first) | test_f414_lru_eviction |
//! | F415 | Memory utilization reported accurately | test_f415_memory_utilization |
//! | F416 | Cache stats tracked correctly | test_f416_cache_stats |
//! | F417 | No memory leaks on sequence free | test_f417_no_memory_leak |
//! | F418 | Block fragmentation minimized | test_f418_fragmentation |
//! | F419 | Reference counting correct | test_f419_reference_counting |
//! | F420 | StreamingLLM eviction preserves sink tokens | test_f420_streaming_llm |

use cbtop::{EvictionStrategy, PagedKvCache, PagedKvError, SeqId};

/// F411: Block allocation succeeds up to GPU memory limit.
#[test]
fn test_f411_block_allocation_limit() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Allocate sequences until we hit the limit
    let mut allocated = 0;
    for i in 0..1000 {
        match cache.allocate(SeqId(i), 16) {
            Ok(_) => allocated += 1,
            Err(PagedKvError::OutOfMemory { .. }) => break,
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    // Should have allocated exactly 100 sequences (1 block each)
    assert_eq!(allocated, 100);
    assert_eq!(cache.used_block_count(), 100);
    assert_eq!(cache.free_block_count(), 0);
}

/// F411 negative: Allocation beyond limit fails gracefully.
#[test]
fn test_f411_allocation_beyond_limit() {
    let mut cache = PagedKvCache::new(10, 16, 32, 128);

    // Fill the cache
    for i in 0..10 {
        cache.allocate(SeqId(i), 16).unwrap();
    }

    // Next allocation should fail
    let result = cache.allocate(SeqId(100), 16);
    assert!(matches!(result, Err(PagedKvError::OutOfMemory { .. })));
}

/// F412: Copy-on-write fork works for beam search.
#[test]
fn test_f412_cow_fork() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Create source sequence
    let src = SeqId(1);
    cache.allocate(src, 64).unwrap(); // 4 blocks
    let blocks_before = cache.used_block_count();

    // Fork for beam search (multiple hypotheses)
    let beams = vec![SeqId(10), SeqId(11), SeqId(12), SeqId(13)];
    for dst in &beams {
        cache.fork(src, *dst).unwrap();
    }

    // All forked sequences should share blocks (COW)
    assert_eq!(cache.used_block_count(), blocks_before);
    assert_eq!(cache.num_sequences(), 5); // src + 4 beams

    // Each forked sequence has same token count
    for dst in beams {
        let seq = cache.get_sequence(dst).unwrap();
        assert_eq!(seq.num_tokens, 64);
    }
}

/// F412 negative: Fork fails for non-existent source.
#[test]
fn test_f412_fork_nonexistent() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let result = cache.fork(SeqId(999), SeqId(1));
    assert!(matches!(result, Err(PagedKvError::SequenceNotFound(_))));
}

/// F413: Eviction triggers at memory threshold.
#[test]
fn test_f413_eviction_threshold() {
    let mut cache = PagedKvCache::new(10, 16, 32, 128).with_eviction_threshold(0.8);

    // Fill to 90% (above 80% threshold)
    for i in 0..9 {
        cache.allocate(SeqId(i), 16).unwrap();
    }

    assert!(cache.needs_eviction());
    assert!(cache.utilization() >= 0.8);

    // Evict to below threshold
    cache.evict_to_threshold(0.5).unwrap();
    assert!(cache.utilization() <= 0.5);
}

/// F413 negative: No eviction below threshold.
#[test]
fn test_f413_no_eviction_below_threshold() {
    let mut cache = PagedKvCache::new(10, 16, 32, 128).with_eviction_threshold(0.9);

    // Fill to 50% (below 90% threshold)
    for i in 0..5 {
        cache.allocate(SeqId(i), 16).unwrap();
    }

    assert!(!cache.needs_eviction());
}

/// F414: LRU eviction correct (oldest access first).
#[test]
fn test_f414_lru_eviction() {
    let mut cache =
        PagedKvCache::new(100, 16, 32, 128).with_eviction_strategy(EvictionStrategy::LRU);

    // Allocate sequences with different access times
    cache.allocate(SeqId(1), 16).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cache.allocate(SeqId(2), 16).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cache.allocate(SeqId(3), 16).unwrap();

    // Evict should remove SeqId(1) first (oldest)
    let evicted = cache.evict().unwrap();
    assert_eq!(evicted, SeqId(1));

    // Next evict should remove SeqId(2)
    let evicted = cache.evict().unwrap();
    assert_eq!(evicted, SeqId(2));
}

/// F414 negative: LFU evicts least frequently used.
#[test]
fn test_f414_lfu_eviction() {
    let mut cache =
        PagedKvCache::new(100, 16, 32, 128).with_eviction_strategy(EvictionStrategy::LFU);

    // Allocate sequences
    cache.allocate(SeqId(1), 16).unwrap();
    cache.allocate(SeqId(2), 16).unwrap();
    cache.allocate(SeqId(3), 16).unwrap();

    // Access seq 1 multiple times (through append)
    for _ in 0..5 {
        cache.append(SeqId(1), 1).unwrap();
    }

    // Access seq 3 once
    cache.append(SeqId(3), 1).unwrap();

    // SeqId(2) should be evicted (least accessed)
    let target = cache.select_eviction_target();
    assert_eq!(target, Some(SeqId(2)));
}

/// F415: Memory utilization reported accurately.
#[test]
fn test_f415_memory_utilization() {
    let cache = PagedKvCache::new(100, 16, 32, 128);

    // Empty cache should have 0% utilization
    assert!((cache.utilization() - 0.0).abs() < f64::EPSILON);

    let mut cache = PagedKvCache::new(100, 16, 32, 128);
    cache.allocate(SeqId(1), 160).unwrap(); // 10 blocks

    // 10/100 = 10% utilization
    let util = cache.utilization();
    assert!((util - 0.1).abs() < 0.001);

    // Memory calculation should be accurate
    // block_memory = 2 (KV) * 16 (block_size) * 32 (heads) * 128 (dim) * 2 (f16)
    assert_eq!(cache.block_memory_bytes(), 262144); // 256KB per block
    assert_eq!(cache.used_memory_bytes(), 2621440); // 2.5MB for 10 blocks
}

/// F415 negative: Zero blocks means zero utilization.
#[test]
fn test_f415_zero_utilization() {
    let cache = PagedKvCache::new(0, 16, 32, 128);
    assert!((cache.utilization() - 0.0).abs() < f64::EPSILON);
}

/// F416: Cache stats tracked correctly.
#[test]
fn test_f416_cache_stats() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Allocate
    cache.allocate(SeqId(1), 32).unwrap(); // 2 blocks
    cache.allocate(SeqId(2), 16).unwrap(); // 1 block

    // Fork
    cache.fork(SeqId(1), SeqId(3)).unwrap();

    // Free
    cache.free(SeqId(2)).unwrap();

    // Evict
    cache.evict().unwrap();

    let stats = cache.stats();
    assert_eq!(stats.total_allocations, 3); // 2 from allocate + 0 from fork (COW)
    assert_eq!(stats.total_frees, 1); // from free
    assert_eq!(stats.total_forks, 1);
    assert_eq!(stats.total_evictions, 1);
}

/// F417: No memory leaks on sequence free.
#[test]
fn test_f417_no_memory_leak() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Allocate and free repeatedly
    for round in 0..10 {
        for i in 0..10 {
            cache.allocate(SeqId(round * 10 + i), 160).unwrap(); // 10 blocks each
        }
        assert_eq!(cache.used_block_count(), 100);
        assert_eq!(cache.free_block_count(), 0);

        // Free all
        for i in 0..10 {
            cache.free(SeqId(round * 10 + i)).unwrap();
        }
        assert_eq!(cache.used_block_count(), 0);
        assert_eq!(cache.free_block_count(), 100);
    }
}

/// F417 negative: Free non-existent sequence fails.
#[test]
fn test_f417_free_nonexistent() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let result = cache.free(SeqId(999));
    assert!(matches!(result, Err(PagedKvError::SequenceNotFound(_))));
}

/// F418: Block fragmentation minimized.
#[test]
fn test_f418_fragmentation() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Allocate, free, allocate pattern
    for i in 0..50 {
        cache.allocate(SeqId(i), 32).unwrap();
    }
    // Free every other sequence
    for i in (0..50).step_by(2) {
        cache.free(SeqId(i)).unwrap();
    }

    // Should be able to allocate the freed blocks
    let free_before = cache.free_block_count();
    for i in 100..125 {
        cache.allocate(SeqId(i), 32).unwrap();
    }
    let free_after = cache.free_block_count();

    // All freed blocks should be reused
    assert_eq!(free_before - free_after, 50);
}

/// F419: Reference counting correct for COW.
#[test]
fn test_f419_reference_counting() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Create source with 4 blocks
    cache.allocate(SeqId(1), 64).unwrap();
    let blocks_with_one_ref = cache.used_block_count();

    // Fork twice (blocks now have 3 refs each)
    cache.fork(SeqId(1), SeqId(2)).unwrap();
    cache.fork(SeqId(1), SeqId(3)).unwrap();

    // Blocks should still be shared
    assert_eq!(cache.used_block_count(), blocks_with_one_ref);

    // Free one fork - blocks still shared
    cache.free(SeqId(2)).unwrap();
    assert_eq!(cache.used_block_count(), blocks_with_one_ref);

    // Free original - still one reference
    cache.free(SeqId(1)).unwrap();
    assert_eq!(cache.used_block_count(), blocks_with_one_ref);

    // Free last reference - blocks freed
    cache.free(SeqId(3)).unwrap();
    assert_eq!(cache.used_block_count(), 0);
}

/// F420: StreamingLLM eviction preserves sink tokens.
#[test]
fn test_f420_streaming_llm() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128).with_eviction_strategy(
        EvictionStrategy::StreamingLLM { sink_tokens: 4, window_tokens: 16 },
    );

    // Create sequence with many tokens
    cache.allocate(SeqId(1), 160).unwrap(); // 10 blocks
    let original_tokens = 160;

    // Apply StreamingLLM eviction (keep sink + window = 20 tokens)
    let evicted = cache.apply_streaming_llm(SeqId(1), 4, 16).unwrap();

    // Should have evicted 160 - 20 = 140 tokens
    assert_eq!(evicted, original_tokens - 20);

    // Sequence should have 20 tokens remaining
    let seq = cache.get_sequence(SeqId(1)).unwrap();
    assert_eq!(seq.num_tokens, 20);
}

/// F420 negative: StreamingLLM does nothing if under limit.
#[test]
fn test_f420_streaming_llm_under_limit() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    // Create short sequence
    cache.allocate(SeqId(1), 16).unwrap();

    // Try to apply with larger window
    let evicted = cache.apply_streaming_llm(SeqId(1), 4, 32).unwrap();

    // Nothing should be evicted (16 < 4 + 32)
    assert_eq!(evicted, 0);
}

/// Integration test: Full PagedKvCache workflow.
#[test]
fn test_full_paged_kv_workflow() {
    // 1. Create cache with realistic parameters
    // 1024 blocks * 16 tokens/block = 16K tokens capacity
    let mut cache = PagedKvCache::new(1024, 16, 32, 128)
        .with_eviction_strategy(EvictionStrategy::LRU)
        .with_eviction_threshold(0.9);

    // 2. Simulate batch of incoming requests (use fewer blocks per sequence)
    for i in 0..20 {
        cache.allocate(SeqId(i), 256).unwrap(); // 16 blocks each = 320 blocks total
    }

    assert_eq!(cache.num_sequences(), 20);
    assert!(cache.utilization() > 0.0);

    // 3. Some sequences complete, free them
    for i in 0..5 {
        cache.free(SeqId(i)).unwrap();
    }

    // 4. New requests arrive (reuse freed space)
    for i in 100..110 {
        cache.allocate(SeqId(i), 128).unwrap(); // 8 blocks each
    }

    // 5. Beam search on one sequence
    cache.fork(SeqId(10), SeqId(200)).unwrap();
    cache.fork(SeqId(10), SeqId(201)).unwrap();
    cache.fork(SeqId(10), SeqId(202)).unwrap();

    // 6. Continue generation (append)
    for i in 10..15 {
        cache.append(SeqId(i), 32).unwrap();
    }

    // 7. Verify final state
    assert!(cache.num_sequences() > 20);
    let stats = cache.stats();
    assert!(stats.total_allocations > 0);
    assert!(stats.total_forks == 3);

    // 8. Display works
    let display = format!("{}", cache);
    assert!(display.contains("PagedKvCache"));
    assert!(display.contains("LRU"));
}
