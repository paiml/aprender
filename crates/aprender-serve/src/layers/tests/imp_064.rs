
// =========================================================================
// M27: Request Scheduling & Resource Management Tests (Phase 18)
// =========================================================================

/// IMP-064: Priority request queue
/// Tests priority-based request scheduling.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_064_priority_queue() {
    use crate::gpu::{PriorityRequest, PriorityRequestQueue};

    // Test 1: Create priority queue
    let mut queue = PriorityRequestQueue::new();
    assert!(queue.is_empty(), "IMP-064: New queue should be empty");
    assert_eq!(queue.len(), 0, "IMP-064: New queue length should be 0");

    // Test 2: Enqueue with different priorities (higher = more important)
    queue.enqueue(PriorityRequest::new(1, "low_priority".to_string()));
    queue.enqueue(PriorityRequest::new(3, "high_priority".to_string()));
    queue.enqueue(PriorityRequest::new(2, "medium_priority".to_string()));
    assert_eq!(queue.len(), 3, "IMP-064: Should have 3 requests");

    // Test 3: Dequeue returns highest priority first
    let req = queue.dequeue_highest();
    assert!(req.is_some(), "IMP-064: Should dequeue request");
    assert_eq!(
        req.expect("test").data(),
        "high_priority",
        "IMP-064: Highest priority first"
    );

    let req = queue.dequeue_highest();
    assert_eq!(
        req.expect("test").data(),
        "medium_priority",
        "IMP-064: Medium priority second"
    );

    let req = queue.dequeue_highest();
    assert_eq!(
        req.expect("test").data(),
        "low_priority",
        "IMP-064: Low priority last"
    );

    // Test 4: Dequeue from empty returns None
    assert!(queue.is_empty(), "IMP-064: Queue should be empty");
    assert!(
        queue.dequeue_highest().is_none(),
        "IMP-064: Dequeue empty returns None"
    );

    // Test 5: Same priority maintains FIFO order
    queue.enqueue(PriorityRequest::new(5, "first".to_string()));
    queue.enqueue(PriorityRequest::new(5, "second".to_string()));
    queue.enqueue(PriorityRequest::new(5, "third".to_string()));
    assert_eq!(
        queue.dequeue_highest().expect("test").data(),
        "first",
        "IMP-064: FIFO for same priority"
    );
    assert_eq!(
        queue.dequeue_highest().expect("test").data(),
        "second",
        "IMP-064: FIFO order"
    );
    assert_eq!(
        queue.dequeue_highest().expect("test").data(),
        "third",
        "IMP-064: FIFO order"
    );
}

/// IMP-065: Token rate limiter
/// Tests throughput control with token bucket algorithm.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_065_rate_limiter() {
    use crate::gpu::TokenRateLimiter;
    use std::time::Duration;

    // Test 1: Create rate limiter (10 tokens/sec, burst of 5)
    let mut limiter = TokenRateLimiter::new(10.0, 5);
    assert_eq!(
        limiter.tokens_available(),
        5,
        "IMP-065: Should start with burst capacity"
    );

    // Test 2: Acquire tokens
    assert!(limiter.try_acquire(3), "IMP-065: Should acquire 3 tokens");
    assert_eq!(
        limiter.tokens_available(),
        2,
        "IMP-065: Should have 2 remaining"
    );

    // Test 3: Acquire more than available fails
    assert!(
        !limiter.try_acquire(3),
        "IMP-065: Should fail to acquire 3 when only 2 available"
    );
    assert_eq!(
        limiter.tokens_available(),
        2,
        "IMP-065: Tokens unchanged on failed acquire"
    );

    // Test 4: Acquire exactly available succeeds
    assert!(
        limiter.try_acquire(2),
        "IMP-065: Should acquire remaining 2"
    );
    assert_eq!(
        limiter.tokens_available(),
        0,
        "IMP-065: Should have 0 remaining"
    );

    // Test 5: Refill adds tokens based on elapsed time
    std::thread::sleep(Duration::from_millis(200)); // 0.2 sec at 10 tok/s = 2 tokens
    limiter.refill();
    let available = limiter.tokens_available();
    assert!(
        available >= 1,
        "IMP-065: Should have refilled at least 1 token, got {}",
        available
    );
    assert!(available <= 5, "IMP-065: Should not exceed burst capacity");
}

/// IMP-066: Resource usage tracker
/// Tests memory and compute resource accounting.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_066_resource_tracker() {
    use crate::gpu::ResourceTracker;

    // Test 1: Create resource tracker (1GB memory, 100% compute capacity)
    let mut tracker = ResourceTracker::new(1024 * 1024 * 1024, 100);
    assert_eq!(
        tracker.memory_usage(),
        0,
        "IMP-066: Initial memory usage is 0"
    );
    assert_eq!(
        tracker.compute_usage(),
        0,
        "IMP-066: Initial compute usage is 0"
    );

    // Test 2: Check allocation availability
    assert!(
        tracker.can_allocate(512 * 1024 * 1024, 50),
        "IMP-066: Should be able to allocate 512MB, 50% compute"
    );
    assert!(
        !tracker.can_allocate(2 * 1024 * 1024 * 1024, 50),
        "IMP-066: Cannot allocate more than capacity"
    );

    // Test 3: Allocate resources
    let alloc_id = tracker.allocate(256 * 1024 * 1024, 30);
    assert!(alloc_id.is_some(), "IMP-066: Allocation should succeed");
    assert_eq!(
        tracker.memory_usage(),
        256 * 1024 * 1024,
        "IMP-066: Memory usage updated"
    );
    assert_eq!(
        tracker.compute_usage(),
        30,
        "IMP-066: Compute usage updated"
    );

    // Test 4: Multiple allocations
    let alloc_id_2 = tracker.allocate(128 * 1024 * 1024, 20);
    assert!(
        alloc_id_2.is_some(),
        "IMP-066: Second allocation should succeed"
    );
    assert_eq!(
        tracker.memory_usage(),
        384 * 1024 * 1024,
        "IMP-066: Memory accumulated"
    );
    assert_eq!(tracker.compute_usage(), 50, "IMP-066: Compute accumulated");

    // Test 5: Release resources
    tracker.release(alloc_id.expect("test"));
    assert_eq!(
        tracker.memory_usage(),
        128 * 1024 * 1024,
        "IMP-066: Memory released"
    );
    assert_eq!(tracker.compute_usage(), 20, "IMP-066: Compute released");

    // Test 6: Usage percentage
    let (mem_pct, compute_pct) = tracker.usage_percentage();
    let expected_mem_pct = (128.0 * 1024.0 * 1024.0) / (1024.0 * 1024.0 * 1024.0) * 100.0;
    assert!(
        (mem_pct - expected_mem_pct).abs() < 0.1,
        "IMP-066: Memory percentage correct"
    );
    assert!(
        (compute_pct - 20.0).abs() < 0.1,
        "IMP-066: Compute percentage correct"
    );
}

// =========================================================================
// M28: Metrics & Health Monitoring Tests (Phase 19)
// =========================================================================

/// IMP-067: Inference metrics collector
/// Tests latency histogram and throughput tracking.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_067_inference_metrics() {
    use crate::gpu::InferenceMetrics;
    use std::time::Duration;

    // Test 1: Create metrics collector
    let mut metrics = InferenceMetrics::new();
    assert_eq!(
        metrics.total_inferences(),
        0,
        "IMP-067: No inferences initially"
    );
    assert_eq!(metrics.total_tokens(), 0, "IMP-067: No tokens initially");

    // Test 2: Record inferences
    metrics.record_inference(Duration::from_millis(10), 5); // 10ms, 5 tokens
    metrics.record_inference(Duration::from_millis(20), 10); // 20ms, 10 tokens
    metrics.record_inference(Duration::from_millis(15), 8); // 15ms, 8 tokens
    assert_eq!(
        metrics.total_inferences(),
        3,
        "IMP-067: Should have 3 inferences"
    );
    assert_eq!(metrics.total_tokens(), 23, "IMP-067: Should have 23 tokens");

    // Test 3: Latency percentiles
    let p50 = metrics.latency_percentile(50);
    assert!(p50.is_some(), "IMP-067: Should have p50");
    let p50_ms = p50.expect("test").as_millis();
    assert!(
        p50_ms >= 10 && p50_ms <= 20,
        "IMP-067: p50 should be ~15ms, got {}ms",
        p50_ms
    );

    // Test 4: Throughput calculation
    let throughput = metrics.throughput();
    assert!(throughput > 0.0, "IMP-067: Throughput should be positive");

    // Test 5: Reset metrics
    metrics.reset();
    assert_eq!(metrics.total_inferences(), 0, "IMP-067: Inferences reset");
    assert_eq!(metrics.total_tokens(), 0, "IMP-067: Tokens reset");
}

/// IMP-068: Health checker
/// Tests component health monitoring.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_068_health_checker() {
    use crate::gpu::HealthChecker;

    // Test 1: Create health checker
    let mut checker = HealthChecker::new();
    assert!(
        checker.is_healthy(),
        "IMP-068: Healthy when no checks registered"
    );

    // Test 2: Register healthy check
    checker.register_check("gpu", Box::new(|| true));
    assert_eq!(checker.check_count(), 1, "IMP-068: Should have 1 check");

    // Test 3: Run checks - all healthy
    let results = checker.check_all();
    assert_eq!(results.len(), 1, "IMP-068: Should have 1 result");
    assert!(
        results.get("gpu").copied().unwrap_or(false),
        "IMP-068: GPU should be healthy"
    );
    assert!(checker.is_healthy(), "IMP-068: Overall should be healthy");

    // Test 4: Register unhealthy check
    checker.register_check("memory", Box::new(|| false));
    let results = checker.check_all();
    assert!(
        !results.get("memory").copied().unwrap_or(true),
        "IMP-068: Memory should be unhealthy"
    );
    assert!(
        !checker.is_healthy(),
        "IMP-068: Overall should be unhealthy"
    );

    // Test 5: Clear checks
    checker.clear();
    assert_eq!(checker.check_count(), 0, "IMP-068: No checks after clear");
    assert!(checker.is_healthy(), "IMP-068: Healthy after clear");
}

/// IMP-069: Graceful shutdown coordinator
/// Tests coordinated shutdown with request draining.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_069_graceful_shutdown() {
    use crate::gpu::ShutdownCoordinator;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // Test 1: Create shutdown coordinator
    let mut coordinator = ShutdownCoordinator::new();
    assert!(
        !coordinator.is_shutting_down(),
        "IMP-069: Not shutting down initially"
    );
    assert_eq!(
        coordinator.pending_requests(),
        0,
        "IMP-069: No pending requests"
    );

    // Test 2: Register shutdown handler
    let handler_called = Arc::new(AtomicBool::new(false));
    let handler_called_clone = handler_called.clone();
    coordinator.register_handler(Box::new(move || {
        handler_called_clone.store(true, Ordering::SeqCst);
    }));
    assert_eq!(
        coordinator.handler_count(),
        1,
        "IMP-069: Should have 1 handler"
    );

    // Test 3: Track pending requests
    coordinator.request_started();
    coordinator.request_started();
    assert_eq!(
        coordinator.pending_requests(),
        2,
        "IMP-069: Should have 2 pending"
    );

    // Test 4: Initiate shutdown
    coordinator.initiate_shutdown();
    assert!(
        coordinator.is_shutting_down(),
        "IMP-069: Should be shutting down"
    );
    assert!(
        handler_called.load(Ordering::SeqCst),
        "IMP-069: Handler should be called"
    );

    // Test 5: Complete pending requests
    coordinator.request_completed();
    assert_eq!(
        coordinator.pending_requests(),
        1,
        "IMP-069: Should have 1 pending"
    );
    coordinator.request_completed();
    assert_eq!(
        coordinator.pending_requests(),
        0,
        "IMP-069: Should have 0 pending"
    );

    // Test 6: Check completion
    assert!(
        coordinator.is_complete(),
        "IMP-069: Should be complete when shutdown + no pending"
    );
}
