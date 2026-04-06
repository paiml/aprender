/// IMP-072: Failure Isolation
use crate::layers::*;
#[test]
#[cfg(feature = "gpu")]
fn test_imp_072_failure_isolation() {
    use crate::gpu::{FailureIsolator, RequestOutcome};
    use std::sync::Arc;

    // Test 1: Create failure isolator
    let isolator = FailureIsolator::new();
    assert_eq!(
        isolator.active_requests(),
        0,
        "IMP-072: Should start with 0 active"
    );

    // Test 2: Start isolated request
    let request_id = isolator.start_request();
    assert_eq!(
        isolator.active_requests(),
        1,
        "IMP-072: Should have 1 active request"
    );

    // Test 3: Complete request successfully
    isolator.complete_request(request_id, &RequestOutcome::Success);
    assert_eq!(
        isolator.active_requests(),
        0,
        "IMP-072: Should have 0 active after completion"
    );
    assert_eq!(
        isolator.success_count(),
        1,
        "IMP-072: Should have 1 success"
    );

    // Test 4: Handle failed request with cleanup
    let request_id = isolator.start_request();
    let cleanup_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cleanup_flag = cleanup_called.clone();
    isolator.register_cleanup(request_id, move || {
        cleanup_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    isolator.complete_request(
        request_id,
        &RequestOutcome::Failed("test error".to_string()),
    );
    assert!(
        cleanup_called.load(std::sync::atomic::Ordering::SeqCst),
        "IMP-072: Cleanup should be called on failure"
    );
    assert_eq!(
        isolator.failure_count(),
        1,
        "IMP-072: Should have 1 failure"
    );

    // Test 5: Circuit breaker opens after repeated failures
    for _ in 0..5 {
        let req_id = isolator.start_request();
        isolator.complete_request(req_id, &RequestOutcome::Failed("error".to_string()));
    }
    assert!(
        isolator.is_circuit_open(),
        "IMP-072: Circuit should open after repeated failures"
    );

    // Test 6: Circuit breaker rejects new requests when open
    let result = isolator.try_start_request();
    assert!(
        result.is_err(),
        "IMP-072: Should reject requests when circuit open"
    );

    // Test 7: Circuit breaker recovers after timeout
    isolator.reset_circuit();
    assert!(
        !isolator.is_circuit_open(),
        "IMP-072: Circuit should close after reset"
    );
    let result = isolator.try_start_request();
    assert!(
        result.is_ok(),
        "IMP-072: Should accept requests when circuit closed"
    );
}

// ========================================================================
// M30: Connection Pooling & Resource Limits (IMP-073, IMP-074, IMP-075)
// ========================================================================

/// M30: Connection Pool Management (IMP-073)
/// Target: Bounded pool, health checking, warm startup
#[test]
#[cfg(feature = "gpu")]
fn test_imp_073_connection_pool() {
    use crate::gpu::{ConnectionConfig, ConnectionPool, ConnectionState};

    // Test 1: Create pool with configurable limits
    let config = ConnectionConfig::new()
        .with_max_connections(10)
        .with_min_connections(2)
        .with_idle_timeout(std::time::Duration::from_secs(300));
    let pool = ConnectionPool::new(config);
    assert_eq!(
        pool.max_connections(),
        10,
        "IMP-073: Max connections should be configurable"
    );
    assert_eq!(
        pool.min_connections(),
        2,
        "IMP-073: Min connections should be configurable"
    );

    // Test 2: Acquire and release connections
    let conn = pool.acquire();
    assert!(conn.is_ok(), "IMP-073: Should acquire connection from pool");
    assert_eq!(
        pool.active_connections(),
        1,
        "IMP-073: Should track active connections"
    );

    pool.release(conn.expect("test"));
    assert_eq!(
        pool.active_connections(),
        0,
        "IMP-073: Should decrement on release"
    );

    // Test 3: Bounded pool rejects when full
    let mut conns = Vec::new();
    for i in 0..10 {
        let c = pool.acquire();
        assert!(c.is_ok(), "IMP-073: Should acquire connection {}", i);
        conns.push(c.expect("test"));
    }
    let overflow = pool.try_acquire();
    assert!(
        overflow.is_err(),
        "IMP-073: Should reject when pool exhausted"
    );

    // Release all
    for c in conns {
        pool.release(c);
    }

    // Test 4: Connection health checking
    let conn = pool.acquire().expect("test");
    let state = pool.check_health(&conn);
    assert!(
        matches!(state, ConnectionState::Healthy),
        "IMP-073: New connection should be healthy"
    );
    pool.release(conn);

    // Test 5: Warm pool on startup
    let pool2 = ConnectionPool::new(ConnectionConfig::new().with_min_connections(3));
    pool2.warm();
    assert!(
        pool2.idle_connections() >= 3,
        "IMP-073: Should warm pool to min connections"
    );
}

/// M30: Resource Limits (IMP-074)
/// Target: Memory limits, compute time limits, queue depth limits
#[test]
#[cfg(feature = "gpu")]
fn test_imp_074_resource_limits() {
    use crate::gpu::{LimitResult, ResourceConfig, ResourceLimiter};

    // Test 1: Create limiter with configurable limits
    let config = ResourceConfig::new()
        .with_max_memory_per_request(512 * 1024 * 1024) // 512MB
        .with_max_total_memory(4 * 1024 * 1024 * 1024) // 4GB
        .with_max_compute_time(std::time::Duration::from_secs(30))
        .with_max_queue_depth(100);
    let limiter = ResourceLimiter::new(config);

    // Test 2: Check memory limits per request
    let result = limiter.check_memory(256 * 1024 * 1024);
    assert!(
        matches!(result, LimitResult::Allowed),
        "IMP-074: Should allow within limits"
    );

    let result = limiter.check_memory(1024 * 1024 * 1024);
    assert!(
        matches!(result, LimitResult::Denied { .. }),
        "IMP-074: Should deny over per-request limit"
    );

    // Test 3: Track total memory usage
    let alloc1 = limiter.allocate(256 * 1024 * 1024);
    assert!(alloc1.is_ok(), "IMP-074: Should allocate memory");
    assert_eq!(
        limiter.current_memory(),
        256 * 1024 * 1024,
        "IMP-074: Should track allocated"
    );

    limiter.deallocate(256 * 1024 * 1024);
    assert_eq!(
        limiter.current_memory(),
        0,
        "IMP-074: Should track deallocated"
    );

    // Test 4: Queue depth limits with backpressure
    for _ in 0..100 {
        let _ = limiter.enqueue();
    }
    let overflow = limiter.try_enqueue();
    assert!(
        matches!(overflow, LimitResult::Backpressure),
        "IMP-074: Should apply backpressure"
    );

    // Drain queue
    for _ in 0..100 {
        limiter.dequeue();
    }

    // Test 5: Compute time tracking
    let timer = limiter.start_compute();
    assert!(
        timer.elapsed() < std::time::Duration::from_secs(1),
        "IMP-074: Timer should work"
    );
}

/// M30: Resource Monitoring (IMP-075)
/// Target: Real-time memory, GPU utilization, queue metrics
#[test]
#[cfg(feature = "gpu")]
fn test_imp_075_resource_monitoring() {
    use crate::gpu::ResourceMonitor;

    // Test 1: Create monitor
    let monitor = ResourceMonitor::new();

    // Test 2: Track memory usage
    monitor.record_memory_usage(512 * 1024 * 1024);
    let metrics = monitor.current_metrics();
    assert_eq!(
        metrics.memory_bytes,
        512 * 1024 * 1024,
        "IMP-075: Should track memory"
    );

    // Test 3: Track GPU utilization
    monitor.record_gpu_utilization(75.5);
    let metrics = monitor.current_metrics();
    assert!(
        (metrics.gpu_utilization - 75.5).abs() < 0.01,
        "IMP-075: Should track GPU util"
    );

    // Test 4: Track queue depth
    monitor.record_queue_depth(42);
    let metrics = monitor.current_metrics();
    assert_eq!(metrics.queue_depth, 42, "IMP-075: Should track queue depth");

    // Test 5: Track request latency
    monitor.record_latency(std::time::Duration::from_millis(150));
    let metrics = monitor.current_metrics();
    assert_eq!(
        metrics.last_latency_ms, 150,
        "IMP-075: Should track latency"
    );

    // Test 6: Aggregate metrics (min/max/avg)
    for i in 1..=5 {
        monitor.record_latency(std::time::Duration::from_millis(i * 100));
    }
    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 100, "IMP-075: Should track min latency");
    assert_eq!(stats.max_ms, 500, "IMP-075: Should track max latency");
    // 6 values: 150, 100, 200, 300, 400, 500 = 1650 / 6 = 275
    assert_eq!(stats.avg_ms, 275, "IMP-075: Should track avg latency");

    // Test 7: Snapshot for reporting
    let snapshot = monitor.snapshot();
    assert!(
        snapshot.timestamp > 0,
        "IMP-075: Snapshot should have timestamp"
    );
    assert!(
        snapshot.memory_bytes > 0,
        "IMP-075: Snapshot should include memory"
    );
}

// ========================================================================
// M31: Retry Logic & Circuit Breakers (IMP-076, IMP-077, IMP-078)
// ========================================================================

/// M31: Retry Strategy (IMP-076)
/// Target: Configurable retry policies, exponential backoff, max limits
#[test]
#[cfg(feature = "gpu")]
fn test_imp_076_retry_strategy() {
    use crate::gpu::{ErrorCategory, RetryConfig, RetryDecision, RetryPolicy};

    // Test 1: Create retry config with defaults
    let config = RetryConfig::new()
        .with_max_retries(5)
        .with_base_delay(std::time::Duration::from_millis(100))
        .with_max_delay(std::time::Duration::from_secs(30))
        .with_jitter_factor(0.2);
    let policy = RetryPolicy::new(config);
    assert_eq!(
        policy.max_retries(),
        5,
        "IMP-076: Max retries should be configurable"
    );

    // Test 2: Decide retry for transient error
    let decision = policy.should_retry(1, ErrorCategory::Transient);
    assert!(
        matches!(decision, RetryDecision::Retry { .. }),
        "IMP-076: Should retry transient error"
    );

    // Test 3: Don't retry permanent errors
    let decision = policy.should_retry(1, ErrorCategory::Permanent);
    assert!(
        matches!(decision, RetryDecision::Abort { .. }),
        "IMP-076: Should not retry permanent error"
    );

    // Test 4: Exponential backoff calculation
    let delay1 = policy.calculate_delay(1);
    let delay2 = policy.calculate_delay(2);
    let delay3 = policy.calculate_delay(3);
    assert!(
        delay2 > delay1,
        "IMP-076: Delay should increase (exp backoff)"
    );
    assert!(delay3 > delay2, "IMP-076: Delay should continue increasing");

    // Test 5: Max delay capping
    let delay_capped = policy.calculate_delay(100);
    assert!(
        delay_capped <= std::time::Duration::from_secs(30),
        "IMP-076: Should cap at max delay"
    );

    // Test 6: Max retries exceeded
    let decision = policy.should_retry(6, ErrorCategory::Transient);
    assert!(
        matches!(decision, RetryDecision::Abort { .. }),
        "IMP-076: Should abort after max retries"
    );
}

/// M31: Circuit Breaker Pattern (IMP-077)
/// Target: Closed/Open/Half-Open states, failure threshold, timeout probe
#[test]
#[cfg(feature = "gpu")]
fn test_imp_077_circuit_breaker() {
    use crate::gpu::{CircuitBreaker, CircuitConfig, CircuitState};

    // Test 1: Create circuit breaker with config
    let config = CircuitConfig::new()
        .with_failure_threshold(3)
        .with_success_threshold(2)
        .with_timeout(std::time::Duration::from_millis(100));
    let breaker = CircuitBreaker::new(config);
    assert!(
        matches!(breaker.state(), CircuitState::Closed),
        "IMP-077: Should start closed"
    );

    // Test 2: Record failures up to threshold
    breaker.record_failure();
    breaker.record_failure();
    assert!(
        matches!(breaker.state(), CircuitState::Closed),
        "IMP-077: Should stay closed below threshold"
    );

    // Test 3: Open after threshold
    breaker.record_failure();
    assert!(
        matches!(breaker.state(), CircuitState::Open),
        "IMP-077: Should open at threshold"
    );

    // Test 4: Reject requests when open
    assert!(!breaker.allow_request(), "IMP-077: Should reject when open");

    // Test 5: Transition to half-open after timeout
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(
        breaker.allow_request(),
        "IMP-077: Should allow probe after timeout"
    );
    assert!(
        matches!(breaker.state(), CircuitState::HalfOpen),
        "IMP-077: Should be half-open"
    );

    // Test 6: Close on success in half-open
    breaker.record_success();
    breaker.record_success();
    assert!(
        matches!(breaker.state(), CircuitState::Closed),
        "IMP-077: Should close after successes"
    );

    // Test 7: Re-open on failure in half-open
    // First get to half-open state
    for _ in 0..3 {
        breaker.record_failure();
    }
    std::thread::sleep(std::time::Duration::from_millis(150));
    let _ = breaker.allow_request(); // Transition to half-open
    breaker.record_failure();
    assert!(
        matches!(breaker.state(), CircuitState::Open),
        "IMP-077: Should re-open on half-open failure"
    );
}

include!("imp_078.rs");
include!("imp_086.rs");
include!("imp_099.rs");
include!("sliding_window.rs");
