//! Tests for async and buffer patterns.

use super::*;

// ------------------------------------------------------------------------
// AsyncResult Tests (LCP-12)
// ------------------------------------------------------------------------

#[test]
fn test_async_result_states() {
    let async_val: AsyncResult<i32, &str> = AsyncResult::Async(42);
    let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(42);
    let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");

    assert!(async_val.is_async());
    assert!(!async_val.is_sync());
    assert!(!async_val.is_error());

    assert!(!sync_val.is_async());
    assert!(sync_val.is_sync());
    assert!(!sync_val.is_error());

    assert!(err.is_error());
    assert!(!err.is_async());
    assert!(!err.is_sync());

    assert_eq!(async_val.into_result(), Ok(42));
    assert_eq!(sync_val.into_result(), Ok(42));
    assert_eq!(err.into_result(), Err("fail"));
}

#[test]
fn test_async_result_map() {
    let async_val: AsyncResult<i32, &str> = AsyncResult::Async(10);
    let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(10);
    let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");

    let mapped_async = async_val.map(|x| x * 2);
    let mapped_sync = sync_val.map(|x| x * 2);
    let mapped_err = err.map(|x| x * 2);

    assert!(matches!(mapped_async, AsyncResult::Async(20)));
    assert!(matches!(mapped_sync, AsyncResult::Sync(20)));
    assert!(matches!(mapped_err, AsyncResult::Error("fail")));
}

// ------------------------------------------------------------------------
// BoundedQueue Tests (AWP-11)
// ------------------------------------------------------------------------

#[test]
fn test_bounded_queue_basic() {
    let mut queue: BoundedQueue<i32> = BoundedQueue::new(5);

    assert!(queue.is_empty());
    assert!(!queue.is_full());

    queue.try_push(1).unwrap();
    queue.try_push(2).unwrap();
    queue.try_push(3).unwrap();

    assert_eq!(queue.len(), 3);
    assert_eq!(queue.pop(), Some(1));
    assert_eq!(queue.pop(), Some(2));
    assert_eq!(queue.len(), 1);
}

#[test]
fn test_bounded_queue_backpressure() {
    let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);

    assert!(queue.try_push(1).is_ok());
    assert!(queue.try_push(2).is_ok());
    assert!(queue.try_push(3).is_ok());
    assert!(queue.is_full());

    assert!(queue.try_push(4).is_err());

    queue.pop();
    assert!(queue.try_push(4).is_ok());
}

#[test]
fn test_bounded_queue_comprehensive() {
    let mut queue: BoundedQueue<String> = BoundedQueue::new(3);

    // Test peek
    assert!(queue.peek().is_none());
    queue.try_push("first".to_string()).unwrap();
    assert_eq!(queue.peek(), Some(&"first".to_string()));

    // Test remaining
    assert_eq!(queue.remaining(), 2);
    queue.try_push("second".to_string()).unwrap();
    assert_eq!(queue.remaining(), 1);

    // Test clear
    queue.clear();
    assert!(queue.is_empty());
    assert_eq!(queue.remaining(), 3);
}

// ------------------------------------------------------------------------
// ReserveStrategy Tests (AWP-13)
// ------------------------------------------------------------------------

#[test]
fn test_reserve_strategy_variants() {
    assert_eq!(reserve_capacity(100, ReserveStrategy::Exact), 100);
    assert_eq!(reserve_capacity(100, ReserveStrategy::Grow50), 150);
    assert_eq!(reserve_capacity(100, ReserveStrategy::Double), 200);
    assert_eq!(reserve_capacity(100, ReserveStrategy::PowerOfTwo), 128);
}

#[test]
fn test_reserve_capacity_edge_cases() {
    assert_eq!(reserve_capacity(0, ReserveStrategy::Exact), 0);
    assert_eq!(reserve_capacity(0, ReserveStrategy::Double), 0);
    assert_eq!(reserve_capacity(1, ReserveStrategy::PowerOfTwo), 1);
    assert_eq!(reserve_capacity(3, ReserveStrategy::PowerOfTwo), 4);
}

#[test]
fn test_strategic_buffer() {
    let mut buf = StrategicBuffer::with_capacity(10, ReserveStrategy::Double);
    buf.write(b"hello");
    assert_eq!(buf.len(), 5);
    assert_eq!(buf.as_slice(), b"hello");

    buf.write(b" world");
    assert_eq!(buf.len(), 11);
    assert_eq!(buf.as_slice(), b"hello world");

    buf.clear();
    assert!(buf.is_empty());
}

// ------------------------------------------------------------------------
// GraphReuseCounter Tests (LCP-08)
// ------------------------------------------------------------------------

#[test]
fn test_graph_reuse_counter() {
    let mut counter = GraphReuseCounter::new(3);

    assert!(!counter.is_hot());
    assert!(!counter.should_cache());

    counter.record_use();
    counter.record_use();
    assert!(!counter.is_hot());

    counter.record_use();
    assert!(counter.is_hot());
    assert!(counter.should_cache());
    assert_eq!(counter.count(), 3);

    counter.reset();
    assert!(!counter.is_hot());
    assert_eq!(counter.count(), 0);
}

// ------------------------------------------------------------------------
// DualWakerState Tests (AWP-03)
// ------------------------------------------------------------------------

#[test]
fn test_dual_waker_state() {
    let mut state = DualWakerState::new(20, 80);

    assert!(state.can_produce());
    assert!(!state.can_consume());

    // Fill above high watermark
    let decision = state.update_fill(85);
    assert_eq!(decision, WakeDecision::PauseProducer);
    assert!(!state.can_produce());
    assert!(state.can_consume());

    // Drain below low watermark
    let decision = state.update_fill(15);
    assert_eq!(decision, WakeDecision::WakeProducer);
    assert!(state.can_produce());
}

#[test]
fn test_dual_waker_consumer_wake() {
    let mut state = DualWakerState::new(20, 80);
    state.consumer_wait();

    // Add data while consumer waiting
    let decision = state.update_fill(50);
    assert_eq!(decision, WakeDecision::WakeConsumer);
}

#[test]
fn test_dual_waker_edge_cases() {
    let mut state = DualWakerState::new(20, 80);

    state.producer_wait();
    state.consumer_wait();

    state.producer_woke();
    state.consumer_woke();

    // Test clamping
    state.update_fill(200); // Should clamp to 100
    assert!(!state.can_produce());
}

// ------------------------------------------------------------------------
// StreamCapacity Tests (AWP-04)
// ------------------------------------------------------------------------

#[test]
fn test_stream_capacity_basic() {
    let mut cap = StreamCapacity::new();

    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW);
    assert!(!cap.is_blocked());

    cap.reserve_send(1000).unwrap();
    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW - 1000);
}

#[test]
fn test_stream_capacity_blocking() {
    let mut cap = StreamCapacity::with_initial_window(100);

    let result = cap.reserve_send(150);
    assert!(result.is_err());
    assert!(cap.is_blocked());

    cap.release_send(100);
    assert!(!cap.is_blocked());
}

#[test]
fn test_stream_capacity_window_ops() {
    let mut cap = StreamCapacity::new();
    // Default window is 65535, needs_window_update when < 32767

    // Consume more than half to trigger update
    cap.consume_receive(40000);
    assert!(cap.needs_window_update()); // 25535 < 32767

    cap.replenish_receive(20000);
    assert!(!cap.needs_window_update()); // 45535 > 32767
}

#[test]
fn test_flow_control_error() {
    let err1 = FlowControlError::NegativeReservation;
    let err2 = FlowControlError::InsufficientCapacity { requested: 100, available: 50 };

    assert_eq!(err1, FlowControlError::NegativeReservation);
    assert!(matches!(err2, FlowControlError::InsufficientCapacity { .. }));
}

// ------------------------------------------------------------------------
// WakeSkipState Tests (AWP-09)
// ------------------------------------------------------------------------

#[test]
fn test_wake_skip_state() {
    let mut state = WakeSkipState::new(3);

    // No waker = always skip
    assert!(state.should_skip_wake());

    state.register_waker();
    assert!(!state.should_skip_wake());

    // Accumulate empty polls
    state.record_poll(false);
    state.record_poll(false);
    state.record_poll(false);
    assert!(state.should_skip_wake());

    // Reset tracking
    state.reset_tracking();
    assert!(!state.should_skip_wake());
}

#[test]
fn test_wake_skip_needs_wake() {
    let mut state = WakeSkipState::new(3);
    state.register_waker();

    // No pending items
    assert!(!state.needs_wake());

    // Add pending
    state.add_pending(5);
    assert!(state.needs_wake());
    assert_eq!(state.pending(), 5);

    // Remove some
    state.remove_pending(3);
    assert_eq!(state.pending(), 2);
}

#[test]
fn test_wake_skip_tracking() {
    let mut state = WakeSkipState::new(5);
    state.register_waker();

    // Work resets empty poll count
    state.record_poll(false);
    state.record_poll(false);
    state.record_poll(true); // Had work
    state.record_poll(false);

    // Should not skip yet (only 1 empty after work)
    assert!(!state.should_skip_wake());
}

// ------------------------------------------------------------------------
// Falsification Tests
// ------------------------------------------------------------------------

/// FALSIFICATION: BoundedQueue capacity invariant.
/// Queue must never exceed its capacity.
#[test]
fn test_falsify_bounded_queue_capacity_invariant() {
    for cap in [1, 5, 10, 100] {
        let mut queue: BoundedQueue<usize> = BoundedQueue::new(cap);

        // Try to push more than capacity
        for i in 0..cap * 2 {
            let _ = queue.try_push(i);
        }

        assert!(
            queue.len() <= cap,
            "FALSIFICATION FAILED: Queue exceeded capacity {} with len {}",
            cap,
            queue.len()
        );
    }
}

/// FALSIFICATION: ReserveStrategy must always return >= needed.
#[test]
fn test_falsify_reserve_strategy_minimum() {
    for needed in [0, 1, 10, 100, 1000, 10000] {
        for strategy in [
            ReserveStrategy::Exact,
            ReserveStrategy::Grow50,
            ReserveStrategy::Double,
            ReserveStrategy::PowerOfTwo,
        ] {
            let reserved = reserve_capacity(needed, strategy);
            assert!(
                reserved >= needed,
                "FALSIFICATION FAILED: reserve_capacity({}, {:?}) = {} < {}",
                needed,
                strategy,
                reserved,
                needed
            );
        }
    }
}

/// FALSIFICATION: GraphReuseCounter hot transition.
/// Must become hot exactly at threshold.
#[test]
fn test_falsify_graph_reuse_threshold() {
    for threshold in [1, 5, 10, 100] {
        let mut counter = GraphReuseCounter::new(threshold);

        // Before threshold
        for _ in 0..threshold - 1 {
            counter.record_use();
            assert!(
                !counter.is_hot(),
                "FALSIFICATION FAILED: Became hot before threshold {}",
                threshold
            );
        }

        // At threshold
        counter.record_use();
        assert!(counter.is_hot(), "FALSIFICATION FAILED: Not hot at threshold {}", threshold);
    }
}

/// FALSIFICATION: StreamCapacity window consistency.
#[test]
fn test_falsify_stream_capacity_consistency() {
    let mut cap = StreamCapacity::new();
    let initial = cap.available_send();

    // Reserve and release should return to initial
    cap.reserve_send(10000).unwrap();
    cap.release_send(10000);
    assert_eq!(
        cap.available_send(),
        initial,
        "FALSIFICATION FAILED: Window not restored after reserve+release"
    );
}
