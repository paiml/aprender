use super::*;

#[test]
fn test_create_cache() {
    let cache = PagedKvCache::new(1024, 16, 32, 128);
    assert_eq!(cache.total_blocks(), 1024);
    assert_eq!(cache.block_size(), 16);
    assert_eq!(cache.free_block_count(), 1024);
    assert_eq!(cache.used_block_count(), 0);
    assert!((cache.utilization() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_allocate_sequence() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let seq_id = SeqId(1);
    cache.allocate(seq_id, 48).unwrap(); // 3 blocks needed (48/16)

    assert_eq!(cache.used_block_count(), 3);
    assert_eq!(cache.num_sequences(), 1);

    let seq = cache.get_sequence(seq_id).unwrap();
    assert_eq!(seq.num_tokens, 48);
    assert_eq!(seq.num_blocks(), 3);
}

#[test]
fn test_append_tokens() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let seq_id = SeqId(1);
    cache.allocate(seq_id, 8).unwrap(); // 1 block (8 tokens, block_size=16)
    assert_eq!(cache.used_block_count(), 1);

    cache.append(seq_id, 4).unwrap(); // Still fits in 1 block (12 tokens)
    assert_eq!(cache.used_block_count(), 1);

    cache.append(seq_id, 8).unwrap(); // Needs 2nd block (20 tokens > 16)
    assert_eq!(cache.used_block_count(), 2);
}

#[test]
fn test_free_sequence() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let seq_id = SeqId(1);
    cache.allocate(seq_id, 48).unwrap();
    assert_eq!(cache.used_block_count(), 3);

    cache.free(seq_id).unwrap();
    assert_eq!(cache.used_block_count(), 0);
    assert_eq!(cache.num_sequences(), 0);
}

#[test]
fn test_fork_sequence() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let src = SeqId(1);
    let dst = SeqId(2);

    cache.allocate(src, 48).unwrap();
    cache.fork(src, dst).unwrap();

    // Both sequences share blocks
    assert_eq!(cache.num_sequences(), 2);
    // No new blocks allocated (COW)
    assert_eq!(cache.used_block_count(), 3);
    assert_eq!(cache.stats().total_forks, 1);
}

#[test]
fn test_out_of_memory() {
    let mut cache = PagedKvCache::new(10, 16, 32, 128);

    let result = cache.allocate(SeqId(1), 200); // Needs 13 blocks, only 10 available
    assert!(matches!(result, Err(PagedKvError::OutOfMemory { .. })));
}

#[test]
fn test_sequence_not_found() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    let result = cache.free(SeqId(999));
    assert!(matches!(result, Err(PagedKvError::SequenceNotFound(_))));
}

#[test]
fn test_utilization() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    cache.allocate(SeqId(1), 160).unwrap(); // 10 blocks
    let util = cache.utilization();
    assert!((util - 0.1).abs() < 0.001); // 10/100 = 10%

    cache.allocate(SeqId(2), 480).unwrap(); // 30 blocks
    let util = cache.utilization();
    assert!((util - 0.4).abs() < 0.001); // 40/100 = 40%
}

#[test]
fn test_lru_eviction() {
    let mut cache =
        PagedKvCache::new(100, 16, 32, 128).with_eviction_strategy(EvictionStrategy::LRU);

    cache.allocate(SeqId(1), 16).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cache.allocate(SeqId(2), 16).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cache.allocate(SeqId(3), 16).unwrap();

    // Seq 1 should be evicted (oldest)
    let target = cache.select_eviction_target();
    assert_eq!(target, Some(SeqId(1)));
}

#[test]
fn test_longest_first_eviction() {
    let mut cache =
        PagedKvCache::new(100, 16, 32, 128).with_eviction_strategy(EvictionStrategy::LongestFirst);

    cache.allocate(SeqId(1), 16).unwrap();
    cache.allocate(SeqId(2), 64).unwrap(); // Longest
    cache.allocate(SeqId(3), 32).unwrap();

    // Seq 2 should be evicted (longest)
    let target = cache.select_eviction_target();
    assert_eq!(target, Some(SeqId(2)));
}

#[test]
fn test_evict_to_threshold() {
    let mut cache = PagedKvCache::new(10, 16, 32, 128);

    // Fill to 80%
    for i in 0..8 {
        cache.allocate(SeqId(i), 16).unwrap();
    }
    assert!((cache.utilization() - 0.8).abs() < 0.01);

    // Evict to 50%
    let evicted = cache.evict_to_threshold(0.5).unwrap();
    assert!(evicted.len() >= 3); // Need to evict at least 3 to go from 8 to 5
    assert!(cache.utilization() <= 0.5);
}

#[test]
fn test_memory_calculation() {
    // block_size=16, num_heads=32, head_dim=128
    // block_memory = 2 * 16 * 32 * 128 * 2 = 262144 bytes = 256KB
    let cache = PagedKvCache::new(100, 16, 32, 128);
    assert_eq!(cache.block_memory_bytes(), 262144);
    assert_eq!(cache.total_memory_bytes(), 26214400); // 25MB
}

#[test]
fn test_stats_tracking() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    cache.allocate(SeqId(1), 32).unwrap();
    cache.allocate(SeqId(2), 16).unwrap();
    cache.fork(SeqId(1), SeqId(3)).unwrap();
    cache.free(SeqId(2)).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.total_allocations, 3);
    assert_eq!(stats.total_frees, 1);
    assert_eq!(stats.total_forks, 1);
}

#[test]
fn test_eviction_strategy_display() {
    assert_eq!(format!("{}", EvictionStrategy::LRU), "LRU");
    assert_eq!(format!("{}", EvictionStrategy::LFU), "LFU");
    assert_eq!(
        format!("{}", EvictionStrategy::StreamingLLM { sink_tokens: 4, window_tokens: 512 }),
        "StreamingLLM(sink=4, window=512)"
    );
}

#[test]
fn test_duplicate_sequence() {
    let mut cache = PagedKvCache::new(100, 16, 32, 128);

    cache.allocate(SeqId(1), 16).unwrap();
    let result = cache.allocate(SeqId(1), 16);
    assert!(matches!(result, Err(PagedKvError::InvalidOperation(_))));
}

#[test]
fn test_blocks_needed_calculation() {
    let cache = PagedKvCache::new(100, 16, 32, 128);

    // blocks_needed is private, test through allocate behavior
    let mut cache2 = cache;
    cache2.allocate(SeqId(1), 1).unwrap();
    assert_eq!(cache2.used_block_count(), 1);

    let mut cache3 = PagedKvCache::new(100, 16, 32, 128);
    cache3.allocate(SeqId(1), 16).unwrap();
    assert_eq!(cache3.used_block_count(), 1);

    let mut cache4 = PagedKvCache::new(100, 16, 32, 128);
    cache4.allocate(SeqId(1), 17).unwrap();
    assert_eq!(cache4.used_block_count(), 2);
}
