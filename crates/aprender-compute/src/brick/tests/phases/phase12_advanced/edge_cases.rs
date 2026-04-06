use super::super::super::super::*;

// ========================================================================
// Edge cases and additional method tests (F211-F215)
// ========================================================================

/// F211: LazySimdConfig additional methods
#[test]
fn test_f211_lazy_simd_config_methods() {
    let config = LazySimdConfig::new();

    // best_backend returns detected backend
    let backend = config.best_backend();
    assert!(!format!("{backend:?}").is_empty());

    // has_amx check
    let _amx = config.has_amx(); // Just verify it doesn't panic

    // Default trait
    let config2 = LazySimdConfig::default();
    assert_eq!(config2.state(), SimdBackendState::Uninitialized);
}

/// F212: UnrollTailIterator edge cases
#[test]
fn test_f212_unroll_tail_iterator_edge_cases() {
    // Empty data
    let iter = UnrollTailIterator::new(0, UnrollFactor::X4);
    assert_eq!(iter.full_iterations(), 0);
    assert_eq!(iter.tail_size(), 0);
    assert!(!iter.has_tail());
    assert_eq!(iter.tail_range(), None);

    // Exactly divisible
    let iter = UnrollTailIterator::new(8, UnrollFactor::X4);
    assert_eq!(iter.full_iterations(), 2);
    assert_eq!(iter.tail_size(), 0);
    assert!(!iter.has_tail());

    // No unroll factor
    let mut iter = UnrollTailIterator::new(5, UnrollFactor::None);
    assert_eq!(iter.full_iterations(), 5);
    assert_eq!(iter.tail_size(), 0);
    for i in 0..5 {
        assert_eq!(iter.next_chunk(), Some((i, i + 1)));
    }
    assert_eq!(iter.next_chunk(), None);
}

/// F213: DualWakerState edge cases
#[test]
fn test_f213_dual_waker_state_edge_cases() {
    let mut state = DualWakerState::new(20, 80);

    // Test producer/consumer wait/wake cycle
    state.producer_wait();
    state.producer_woke();
    state.consumer_wait();
    state.consumer_woke();

    // Low fill with consumer waiting should wake consumer
    state.consumer_wait();
    let decision = state.update_fill(30);
    assert_eq!(decision, WakeDecision::WakeConsumer);

    // Empty buffer - can't consume
    state.update_fill(0);
    assert!(!state.can_consume());
}

/// F214: StreamCapacity window operations
#[test]
fn test_f214_stream_capacity_window_ops() {
    let mut cap = StreamCapacity::new();

    // Initial state
    assert_eq!(cap.available_receive(), StreamCapacity::DEFAULT_WINDOW);
    assert!(!cap.needs_window_update());

    // Consume receive window
    cap.consume_receive(50000);
    assert_eq!(cap.available_receive(), StreamCapacity::DEFAULT_WINDOW - 50000);

    // Check if needs window update (when < 50% of initial)
    cap.consume_receive(20000);
    assert!(cap.needs_window_update()); // Below 50% threshold

    // Replenish
    cap.replenish_receive(10000);
    assert_eq!(cap.available_receive(), StreamCapacity::DEFAULT_WINDOW - 60000);

    // Default trait
    let cap2 = StreamCapacity::default();
    assert!(!cap2.is_blocked());
}

/// F215: WakeSkipState tracking
#[test]
fn test_f215_wake_skip_state_tracking() {
    let mut state = WakeSkipState::new(2);
    state.register_waker(); // Must register waker for should_skip_wake to work

    // Pending count
    state.add_pending(5);
    assert_eq!(state.pending(), 5);
    state.add_pending(3);
    assert_eq!(state.pending(), 8);
    state.remove_pending(4);
    assert_eq!(state.pending(), 4);

    // Reset tracking
    state.record_poll(false);
    state.record_poll(false);
    state.reset_tracking();
    // After reset, empty poll count is 0, so should not skip (waker is registered)
    assert!(!state.should_skip_wake()); // Reset clears history
}
