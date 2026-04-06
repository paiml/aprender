use super::super::super::super::*;

// ========================================================================
// Advanced features: graph reuse, KV cache, batch orderer, keep-alive,
// connection state, lazy SIMD, unroll, dual waker, stream capacity,
// wake skip (F191-F215)
// ========================================================================

/// F191: GraphReuseCounter hot detection (LCP-08)
#[test]
fn test_f191_graph_reuse_counter() {
    let mut counter = GraphReuseCounter::new(5);

    assert!(!counter.is_hot());
    assert!(!counter.should_cache());
    assert_eq!(counter.count(), 0);

    // Record uses until hot
    for _ in 0..4 {
        counter.record_use();
    }
    assert!(!counter.is_hot());

    counter.record_use(); // 5th use
    assert!(counter.is_hot());
    assert!(counter.should_cache());

    // Reset clears everything
    counter.reset();
    assert!(!counter.is_hot());
    assert_eq!(counter.count(), 0);
}

/// F192: KvCacheSlotInfo eviction priority (LCP-10)
#[test]
fn test_f192_kv_cache_slot_info() {
    let mut slot = KvCacheSlotInfo::new(0, 42, 0, 0);

    assert!(slot.valid);
    assert_eq!(slot.position, 0);
    assert_eq!(slot.token_id, 42);

    // Touch updates last_access
    slot.touch(10);
    assert_eq!(slot.last_access, 10);

    // Eviction priority
    assert_eq!(slot.eviction_priority(10), 0);
    assert_eq!(slot.eviction_priority(20), 10);

    // Invalidate gives max priority
    slot.invalidate();
    assert!(!slot.valid);
    assert_eq!(slot.eviction_priority(100), u64::MAX);
}

/// F193: KvCacheManager allocation and eviction (LCP-10)
#[test]
fn test_f193_kv_cache_manager() {
    let mut mgr = KvCacheManager::new(3);

    assert_eq!(mgr.capacity(), 3);
    assert_eq!(mgr.valid_count(), 0);

    // Allocate slots
    let idx0 = mgr.allocate(0, 100, 0, 0).unwrap();
    mgr.step();
    let idx1 = mgr.allocate(1, 101, 0, 0).unwrap();
    mgr.step();
    let _idx2 = mgr.allocate(2, 102, 0, 0).unwrap();

    assert_eq!(mgr.valid_count(), 3);
    assert!(mgr.allocate(3, 103, 0, 0).is_none()); // Full

    // Access slot 0 to update its last_access
    mgr.step();
    mgr.access(idx0);

    // Evict LRU (should be slot 1, oldest access)
    let evicted = mgr.evict_lru().unwrap();
    assert_eq!(evicted, idx1);
    assert_eq!(mgr.valid_count(), 2);
}

/// F194: SequentialBatchOrderer iteration (LCP-14)
#[test]
fn test_f194_sequential_batch_orderer() {
    // Sequential order
    let mut orderer = SequentialBatchOrderer::new(4);
    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(3));
    assert_eq!(orderer.next_batch(), None);
    assert!(orderer.is_done());

    // Reversed order
    let mut orderer = SequentialBatchOrderer::reversed(3);
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(0));

    // Reset
    orderer.reset();
    assert_eq!(orderer.remaining(), 3);
}

/// F195: SequentialBatchOrderer interleaved (LCP-14)
#[test]
fn test_f195_batch_orderer_interleaved() {
    // 4 batches: interleaved is 0, 2, 1, 3
    let orderer = SequentialBatchOrderer::interleaved(4);
    let order: Vec<_> = orderer.collect();
    assert_eq!(order, vec![0, 2, 1, 3]);

    // 5 batches: interleaved is 0, 2, 1, 3, 4
    let orderer = SequentialBatchOrderer::interleaved(5);
    let order: Vec<_> = orderer.collect();
    assert_eq!(order.len(), 5);
    // All indices present
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}

/// F196: KeepAliveConfig parsing (AWP-10)
#[test]
fn test_f196_keep_alive_config() {
    // Default config
    let config = KeepAliveConfig::new();
    assert!(config.enabled);
    assert_eq!(config.timeout_secs, 60);
    assert_eq!(config.max_requests, 100);

    // Parse from header
    let config = KeepAliveConfig::from_header("timeout=5, max=50");
    assert_eq!(config.timeout_secs, 5);
    assert_eq!(config.max_requests, 50);

    // Disabled config
    let config = KeepAliveConfig::disabled();
    assert!(!config.enabled);
}

/// F197: KeepAliveConfig should_keep_alive (AWP-10)
#[test]
fn test_f197_keep_alive_should() {
    let config = KeepAliveConfig::new(); // max_requests = 100

    assert!(config.should_keep_alive(0));
    assert!(config.should_keep_alive(99));
    assert!(!config.should_keep_alive(100));
    assert!(!config.should_keep_alive(150));

    // Disabled never keeps alive
    let disabled = KeepAliveConfig::disabled();
    assert!(!disabled.should_keep_alive(0));
}

/// F198: ConnectionState bitflags (AWP-12)
#[test]
fn test_f198_connection_state_flags() {
    let mut state = ConnectionState::new();
    assert_eq!(state.bits(), 0);
    assert!(!state.is_healthy());

    // Set flags
    state.set(ConnectionState::OPEN);
    assert!(state.is_set(ConnectionState::OPEN));
    assert!(!state.is_set(ConnectionState::READABLE));

    state.set(ConnectionState::WRITABLE);
    assert!(state.is_healthy());
    assert!(state.can_write());

    // Clear flags
    state.set(ConnectionState::ERROR);
    assert!(!state.is_healthy());

    state.clear(ConnectionState::ERROR);
    assert!(state.is_healthy());
}

/// F199: ConnectionState open_connection (AWP-12)
#[test]
fn test_f199_connection_state_open() {
    let state = ConnectionState::open_connection();

    assert!(state.is_set(ConnectionState::OPEN));
    assert!(state.is_set(ConnectionState::WRITABLE));
    assert!(!state.is_set(ConnectionState::READABLE));
    assert!(state.is_healthy());
    assert!(state.can_write());
    assert!(!state.can_read());
}

/// F200: ConnectionState closing prevents write (AWP-12)
#[test]
fn test_f200_connection_state_closing() {
    let mut state = ConnectionState::open_connection();
    state.set(ConnectionState::READABLE);

    assert!(state.can_read());
    assert!(state.can_write());

    // Set closing
    state.set(ConnectionState::CLOSING);
    assert!(state.can_read()); // Can still read
    assert!(!state.can_write()); // Cannot write when closing
    assert!(!state.is_healthy());
}

/// F201: LazySimdConfig lazy initialization (LCP-07)
#[test]
fn test_f201_lazy_simd_config() {
    let mut config = LazySimdConfig::new();

    // Starts uninitialized
    assert_eq!(config.state(), SimdBackendState::Uninitialized);

    // First ensure_ready initializes
    let backend = config.ensure_ready().unwrap();
    assert_eq!(config.state(), SimdBackendState::Ready);

    // Second call returns immediately
    let backend2 = config.ensure_ready().unwrap();
    assert_eq!(backend, backend2);

    // Reset works
    config.reset();
    assert_eq!(config.state(), SimdBackendState::Uninitialized);
}

/// F202: UnrollFactor values (LCP-13)
#[test]
fn test_f202_unroll_factor() {
    assert_eq!(UnrollFactor::None.value(), 1);
    assert_eq!(UnrollFactor::X2.value(), 2);
    assert_eq!(UnrollFactor::X4.value(), 4);
    assert_eq!(UnrollFactor::X8.value(), 8);

    // Backend selection
    assert_eq!(UnrollFactor::for_backend(ComputeBackend::Avx512), UnrollFactor::X8);
    assert_eq!(UnrollFactor::for_backend(ComputeBackend::Avx2), UnrollFactor::X4);
    assert_eq!(UnrollFactor::for_backend(ComputeBackend::Scalar), UnrollFactor::None);
}

/// F203: UnrollTailIterator chunks and tail (LCP-13)
#[test]
fn test_f203_unroll_tail_iterator() {
    // 10 elements with X4 unroll: 2 full chunks + 2 tail
    let mut iter = UnrollTailIterator::new(10, UnrollFactor::X4);

    assert_eq!(iter.full_iterations(), 2);
    assert_eq!(iter.tail_size(), 2);
    assert!(iter.has_tail());

    // Get chunks
    assert_eq!(iter.next_chunk(), Some((0, 4)));
    assert_eq!(iter.next_chunk(), Some((4, 8)));
    assert_eq!(iter.next_chunk(), None);

    // Get tail
    assert_eq!(iter.tail_range(), Some((8, 10)));
}

/// F204: unroll_tail_process function (LCP-13)
#[test]
fn test_f204_unroll_tail_process() {
    let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let results = unroll_tail_process(
        &data,
        UnrollFactor::X4,
        |chunk| chunk.iter().sum::<i32>(),
        |&elem| elem,
    );

    // 2 chunks: sum(1,2,3,4)=10, sum(5,6,7,8)=20
    // 2 tail elements: 9, 10
    assert_eq!(results, vec![10, 26, 9, 10]);
}

/// F205: DualWakerState watermarks (AWP-03)
#[test]
fn test_f205_dual_waker_state() {
    let mut state = DualWakerState::new(20, 80);

    assert!(state.can_produce());
    assert!(!state.can_consume());

    // Fill to 50%
    let decision = state.update_fill(50);
    assert_eq!(decision, WakeDecision::None);
    assert!(state.can_produce());
    assert!(state.can_consume());

    // Fill to 80% (high watermark)
    let decision = state.update_fill(80);
    assert_eq!(decision, WakeDecision::PauseProducer);
    assert!(!state.can_produce());

    // Drain to 20% (low watermark)
    let decision = state.update_fill(20);
    assert_eq!(decision, WakeDecision::WakeProducer);
    assert!(state.can_produce());
}

/// F206: DualWakerState consumer wake (AWP-03)
#[test]
fn test_f206_dual_waker_consumer_wake() {
    let mut state = DualWakerState::new(20, 80);

    // Consumer waiting with no data
    state.consumer_wait();
    let decision = state.update_fill(0);
    assert_eq!(decision, WakeDecision::None);

    // Data arrives - should wake consumer
    let decision = state.update_fill(10);
    assert_eq!(decision, WakeDecision::WakeConsumer);
}

/// F207: StreamCapacity flow control (AWP-04)
#[test]
fn test_f207_stream_capacity() {
    let mut cap = StreamCapacity::new();

    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW);
    assert!(!cap.is_blocked());

    // Reserve some capacity
    cap.reserve_send(1000).unwrap();
    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW - 1000);

    // Release capacity
    cap.release_send(1000);
    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW);
}

/// F208: StreamCapacity blocking (AWP-04)
#[test]
fn test_f208_stream_capacity_blocking() {
    let mut cap = StreamCapacity::with_initial_window(100);

    // Try to reserve more than available
    let result = cap.reserve_send(150);
    assert!(result.is_err());
    assert!(cap.is_blocked());

    // Negative reservation should fail
    let result = cap.reserve_send(-10);
    assert!(matches!(result, Err(FlowControlError::NegativeReservation)));
}

/// F209: WakeSkipState optimization (AWP-09)
#[test]
fn test_f209_wake_skip_state() {
    let mut state = WakeSkipState::new(3);

    // No waker - should skip
    assert!(state.should_skip_wake());

    // Register waker, no pending - shouldn't skip (might get work soon)
    state.register_waker();
    assert!(!state.should_skip_wake());

    // Add pending and last poll had work - SHOULD skip (will be polled anyway)
    state.add_pending(1);
    state.record_poll(true);
    assert!(state.should_skip_wake()); // Has work queued, will be polled

    // No pending, last poll had no work - shouldn't skip
    state.remove_pending(1);
    state.record_poll(false);
    assert!(!state.should_skip_wake());

    // Multiple empty polls reach threshold
    state.record_poll(false);
    state.record_poll(false);
    assert!(state.should_skip_wake()); // 3 empty polls
}

/// F210: WakeSkipState needs_wake (AWP-09)
#[test]
fn test_f210_wake_skip_needs_wake() {
    let mut state = WakeSkipState::new(5);

    // No waker, no pending - doesn't need wake
    assert!(!state.needs_wake());

    // Has waker and pending - needs wake
    state.register_waker();
    state.add_pending(1);
    assert!(state.needs_wake());

    // Clear waker - doesn't need wake
    state.clear_waker();
    assert!(!state.needs_wake());

    // Remove pending - doesn't need wake
    state.register_waker();
    state.remove_pending(1);
    assert!(!state.needs_wake());
}
