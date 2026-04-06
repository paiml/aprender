use super::super::super::super::*;
use std::time::Duration;

// ========================================================================
// Batch split, async result, circuit breaker, connection, queue, reserve
// (F177-F190)
// ========================================================================

/// F177: BatchSplitStrategy variants (LCP-09)
#[test]
fn test_f177_batch_split_strategy() {
    let simple = BatchSplitStrategy::Simple;
    let equal = BatchSplitStrategy::Equal;
    let seq_aware = BatchSplitStrategy::SequenceAware;

    // Verify variants exist and are distinct
    assert!(matches!(simple, BatchSplitStrategy::Simple));
    assert!(matches!(equal, BatchSplitStrategy::Equal));
    assert!(matches!(seq_aware, BatchSplitStrategy::SequenceAware));

    // Default should be Simple
    assert!(matches!(BatchSplitStrategy::default(), BatchSplitStrategy::Simple));
}

/// F178: split_batch correctness (LCP-09)
#[test]
fn test_f178_split_batch() {
    // Simple strategy: 100 items into 4 workers
    let chunks = split_batch(100, 4, BatchSplitStrategy::Simple);
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks.iter().sum::<usize>(), 100);

    // Equal (Balance211): 50 items with 2 workers - guarantees max-min <= 1
    let chunks = split_batch(50, 2, BatchSplitStrategy::Equal);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks.iter().sum::<usize>(), 50);
    // Balance211 property: max - min <= 1
    let max = *chunks.iter().max().unwrap();
    let min = *chunks.iter().min().unwrap();
    assert!(max - min <= 1);

    // SequenceAware: 1000 items with 4 workers
    let chunks = split_batch(1000, 4, BatchSplitStrategy::SequenceAware);
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks.iter().sum::<usize>(), 1000);
}

/// F179: AsyncResult states (LCP-12)
#[test]
fn test_f179_async_result() {
    let async_val: AsyncResult<i32, &str> = AsyncResult::Async(42);
    let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(42);
    let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");

    // Check async/sync detection
    assert!(async_val.is_async());
    assert!(!async_val.is_sync());
    assert!(!async_val.is_error());

    assert!(!sync_val.is_async());
    assert!(sync_val.is_sync());
    assert!(!sync_val.is_error());

    assert!(err.is_error());
    assert!(!err.is_async());
    assert!(!err.is_sync());

    // Extract values using into_result()
    assert_eq!(async_val.into_result(), Ok(42));
    assert_eq!(sync_val.into_result(), Ok(42));
    assert_eq!(err.into_result(), Err("fail"));
}

/// F180: CircuitBreaker initial state (AWP-02)
#[test]
fn test_f180_circuit_breaker_initial() {
    let mut cb = CircuitBreaker::new(3, Duration::from_secs(30));

    // Should start closed
    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.allow_request());
}

/// F181: CircuitBreaker state transitions (AWP-02)
#[test]
fn test_f181_circuit_breaker_transitions() {
    let mut cb = CircuitBreaker::new(3, Duration::from_millis(10));

    // Record failures to open the circuit
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed); // Still closed

    cb.record_failure(); // 3rd failure
    assert_eq!(cb.state(), CircuitState::Open); // Now open
    assert!(!cb.allow_request());

    // Wait for open duration to expire
    std::thread::sleep(Duration::from_millis(15));

    // Now should allow a probe request (half-open)
    assert!(cb.allow_request());
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    // Record success to close
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
}

/// F182: ManagedConnection TTL (AWP-06)
#[test]
fn test_f182_managed_connection_ttl() {
    let conn = ManagedConnection::new(
        "test-conn",
        Duration::from_millis(50), // max lifetime
        Duration::from_millis(20), // max idle
    );

    assert!(conn.is_valid());
    assert!(!conn.is_expired());

    // Wait for expiry
    std::thread::sleep(Duration::from_millis(55));
    assert!(conn.is_expired());
    assert!(!conn.is_valid());
}

/// F183: ManagedConnection health (AWP-06)
#[test]
fn test_f183_managed_connection_health() {
    let mut conn = ManagedConnection::new(42i32, Duration::from_secs(60), Duration::from_secs(30));

    assert_eq!(conn.health_failures(), 0);
    assert!(conn.is_valid());

    // Record some failures
    conn.record_health_failure();
    conn.record_health_failure();
    conn.record_health_failure();
    assert_eq!(conn.health_failures(), 3);
    assert!(!conn.is_valid()); // 3+ failures = invalid

    // Reset health
    conn.reset_health();
    assert_eq!(conn.health_failures(), 0);
    assert!(conn.is_valid());
}

/// F184: BoundedQueue push/pop (AWP-11)
#[test]
fn test_f184_bounded_queue_basic() {
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

/// F185: BoundedQueue back-pressure (AWP-11)
#[test]
fn test_f185_bounded_queue_backpressure() {
    let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);

    // Fill the queue
    assert!(queue.try_push(1).is_ok());
    assert!(queue.try_push(2).is_ok());
    assert!(queue.try_push(3).is_ok());
    assert!(queue.is_full());

    // Back-pressure: can't push more
    assert!(queue.try_push(4).is_err());

    // Pop one, now can push
    queue.pop();
    assert!(queue.try_push(4).is_ok());
}

/// F186: ReserveStrategy variants (AWP-13)
#[test]
fn test_f186_reserve_strategy_variants() {
    let exact = ReserveStrategy::Exact;
    let grow = ReserveStrategy::Grow50;
    let double = ReserveStrategy::Double;
    let power = ReserveStrategy::PowerOfTwo;

    // Verify distinct variants
    assert!(matches!(exact, ReserveStrategy::Exact));
    assert!(matches!(grow, ReserveStrategy::Grow50));
    assert!(matches!(double, ReserveStrategy::Double));
    assert!(matches!(power, ReserveStrategy::PowerOfTwo));
}

/// F187: reserve_capacity correctness (AWP-13)
#[test]
fn test_f187_reserve_capacity() {
    // Exact: returns exactly what's needed
    assert_eq!(reserve_capacity(100, ReserveStrategy::Exact), 100);

    // Grow50: adds 50%
    assert_eq!(reserve_capacity(100, ReserveStrategy::Grow50), 150);

    // Double: 2x
    assert_eq!(reserve_capacity(100, ReserveStrategy::Double), 200);

    // PowerOfTwo: next power of 2
    assert_eq!(reserve_capacity(100, ReserveStrategy::PowerOfTwo), 128);
    assert_eq!(reserve_capacity(128, ReserveStrategy::PowerOfTwo), 128);
    assert_eq!(reserve_capacity(129, ReserveStrategy::PowerOfTwo), 256);
}

/// F188: StrategicBuffer operations (AWP-13)
#[test]
fn test_f188_strategic_buffer() {
    let mut buf = StrategicBuffer::new(ReserveStrategy::Double);

    // Initially empty
    assert!(buf.is_empty());

    // Reserve using strategy
    buf.reserve(10);
    assert!(buf.capacity() >= 10); // Reserved at least 10

    // Write bytes
    buf.write(&[1, 2, 3]);
    assert_eq!(buf.len(), 3);

    // Access inner
    assert_eq!(buf.as_slice(), &[1, 2, 3]);

    // Clear and verify
    buf.clear();
    assert!(buf.is_empty());
}

/// F189: AsyncResult map transform (LCP-12)
#[test]
fn test_f189_async_result_map() {
    let async_val: AsyncResult<i32, &str> = AsyncResult::Async(10);
    let mapped = async_val.map(|x| x * 2);
    assert!(mapped.is_async());
    assert_eq!(mapped.into_result(), Ok(20));

    let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(10);
    let mapped = sync_val.map(|x| x * 2);
    assert!(mapped.is_sync());
    assert_eq!(mapped.into_result(), Ok(20));

    let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");
    let mapped = err.map(|x| x * 2);
    assert!(mapped.is_error());
}

/// F190: split_batch edge cases (LCP-09)
#[test]
fn test_f190_split_batch_edge_cases() {
    // Zero items
    let chunks = split_batch(0, 4, BatchSplitStrategy::Simple);
    assert!(chunks.is_empty());

    // Zero workers
    let chunks = split_batch(100, 0, BatchSplitStrategy::Simple);
    assert!(chunks.is_empty());

    // Single worker gets all items
    let chunks = split_batch(100, 1, BatchSplitStrategy::Simple);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], 100);

    // Exactly divisible: 64 items, 2 workers with Equal strategy
    let chunks = split_batch(64, 2, BatchSplitStrategy::Equal);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks.iter().sum::<usize>(), 64);
    // Both workers get exactly 32
    assert_eq!(chunks[0], 32);
    assert_eq!(chunks[1], 32);
}
