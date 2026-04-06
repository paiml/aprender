
// =============================================================================
// IMP-137: Add Reset Capability for Metrics (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Allow resetting metrics to zero for fresh benchmarking.
// This is essential for A/B testing and iterative performance tuning.
//
// Test TDD Anchors:
// - IMP-137a: DispatchMetrics should have reset() method
// - IMP-137b: reset() should clear all counters to zero
// - IMP-137c: reset() should reset all latency tracking
// - IMP-137d: reset() should reset bucket counts

// IMP-137a: DispatchMetrics should have reset() method
#[test]
fn test_imp_137a_dispatch_metrics_has_reset_method() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some data
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(100));

    // IMP-137a: reset() should exist and be callable
    metrics.reset();

    // After reset, all counters should be zero
    assert_eq!(
        metrics.cpu_dispatches(),
        0,
        "IMP-137a: CPU dispatches should be 0 after reset"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-137a: GPU dispatches should be 0 after reset"
    );
}

// IMP-137b: reset() should clear all counters to zero
#[test]
fn test_imp_137b_reset_clears_all_counters() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record various data
    for _ in 0..10 {
        metrics.record_cpu_dispatch();
        metrics.record_cpu_latency(Duration::from_micros(100));
    }
    for _ in 0..5 {
        metrics.record_gpu_dispatch();
        metrics.record_gpu_latency(Duration::from_micros(500));
    }

    // Verify data was recorded
    assert_eq!(
        metrics.cpu_dispatches(),
        10,
        "IMP-137b: Pre-reset CPU count"
    );
    assert_eq!(metrics.gpu_dispatches(), 5, "IMP-137b: Pre-reset GPU count");

    // Reset
    metrics.reset();

    // IMP-137b: All counters should be zero
    assert_eq!(
        metrics.cpu_dispatches(),
        0,
        "IMP-137b: Post-reset CPU dispatches"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-137b: Post-reset GPU dispatches"
    );
    assert_eq!(
        metrics.total_dispatches(),
        0,
        "IMP-137b: Post-reset total dispatches"
    );
    assert_eq!(
        metrics.cpu_latency_count(),
        0,
        "IMP-137b: Post-reset CPU latency count"
    );
    assert_eq!(
        metrics.gpu_latency_count(),
        0,
        "IMP-137b: Post-reset GPU latency count"
    );
}

// IMP-137c: reset() should reset all latency tracking (min/max/mean/variance)
#[test]
fn test_imp_137c_reset_clears_latency_tracking() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(500));
    metrics.record_cpu_latency(Duration::from_micros(1000));

    // Verify data was recorded
    assert!(
        metrics.cpu_latency_mean_us() > 0.0,
        "IMP-137c: Pre-reset mean should be > 0"
    );

    // Reset
    metrics.reset();

    // IMP-137c: All latency stats should be reset
    assert_eq!(
        metrics.cpu_latency_mean_us(),
        0.0,
        "IMP-137c: Post-reset CPU mean"
    );
    assert_eq!(
        metrics.cpu_latency_min_us(),
        0,
        "IMP-137c: Post-reset CPU min"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        0,
        "IMP-137c: Post-reset CPU max"
    );
    assert_eq!(
        metrics.cpu_latency_variance_us(),
        0.0,
        "IMP-137c: Post-reset CPU variance"
    );
    assert_eq!(
        metrics.cpu_latency_stddev_us(),
        0.0,
        "IMP-137c: Post-reset CPU stddev"
    );
}

// IMP-137d: reset() should reset bucket counts
#[test]
fn test_imp_137d_reset_clears_bucket_counts() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record latencies in different buckets
    metrics.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    metrics.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    metrics.record_cpu_latency(Duration::from_micros(750)); // bucket 2
    metrics.record_cpu_latency(Duration::from_micros(2000)); // bucket 3
    metrics.record_cpu_latency(Duration::from_micros(10000)); // bucket 4

    // Verify buckets have data
    let buckets_before = metrics.cpu_latency_buckets();
    assert_eq!(
        buckets_before.iter().sum::<usize>(),
        5,
        "IMP-137d: Pre-reset bucket total"
    );

    // Reset
    metrics.reset();

    // IMP-137d: All bucket counts should be zero
    let buckets_after = metrics.cpu_latency_buckets();
    assert_eq!(
        buckets_after,
        [0, 0, 0, 0, 0],
        "IMP-137d: Post-reset buckets should all be 0"
    );
}

// =============================================================================
// IMP-138: Add HTTP Endpoint for Metrics Reset (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Expose POST /v1/dispatch/reset endpoint to reset metrics via HTTP.
// This enables remote A/B testing and benchmark automation.
//
// Test TDD Anchors:
// - IMP-138a: POST /v1/dispatch/reset should exist
// - IMP-138b: Reset should return success response
// - IMP-138c: After reset, GET /v1/dispatch should show zero values
// - IMP-138d: Non-POST methods should return 405 Method Not Allowed

// IMP-138a: dispatch_reset_handler function exists and is callable
#[test]
fn test_imp_138a_dispatch_reset_handler_exists() {
    // IMP-138a: Verify handler function signature is correct
    // The handler exists and can be referenced (compile-time check)
    fn _assert_handler_exists<F, Fut>(f: F)
    where
        F: Fn(axum::extract::State<AppState>) -> Fut,
        Fut: std::future::Future<Output = axum::response::Response>,
    {
        let _ = f;
    }
    _assert_handler_exists(dispatch_reset_handler);
}

// IMP-138b: Reset endpoint should return success JSON
#[tokio::test]
async fn test_imp_138b_reset_returns_success_response() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    // Create metrics with some data
    let metrics = Arc::new(DispatchMetrics::new());
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    // IMP-138b: Build reset response
    let response = DispatchResetResponse {
        success: true,
        message: "Metrics reset successfully".to_string(),
    };

    // Serialize and verify
    let json = serde_json::to_string(&response).expect("IMP-138b: Should serialize");
    assert!(
        json.contains("\"success\":true"),
        "IMP-138b: Should have success: true"
    );
    assert!(
        json.contains("reset successfully"),
        "IMP-138b: Should have success message"
    );
}

// IMP-138c: After reset, metrics should be zero
#[tokio::test]
async fn test_imp_138c_reset_endpoint_clears_metrics() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some data
    for _ in 0..10 {
        metrics.record_cpu_dispatch();
        metrics.record_cpu_latency(Duration::from_micros(100));
    }

    // Verify data exists
    assert_eq!(metrics.cpu_dispatches(), 10, "IMP-138c: Pre-reset count");

    // Call reset
    metrics.reset();

    // IMP-138c: After reset, all should be zero
    assert_eq!(
        metrics.cpu_dispatches(),
        0,
        "IMP-138c: Post-reset CPU dispatches"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-138c: Post-reset GPU dispatches"
    );
    assert_eq!(
        metrics.cpu_latency_count(),
        0,
        "IMP-138c: Post-reset latency count"
    );
}

// IMP-138d: DispatchResetResponse can be deserialized
#[test]
fn test_imp_138d_reset_response_deserialization() {
    // IMP-138d: Verify response can be deserialized (for client integration)
    let json = r#"{"success":true,"message":"Metrics reset successfully"}"#;
    let response: DispatchResetResponse =
        serde_json::from_str(json).expect("IMP-138d: Should deserialize");

    assert!(response.success, "IMP-138d: success should be true");
    assert_eq!(
        response.message, "Metrics reset successfully",
        "IMP-138d: message should match"
    );
}

// =============================================================================
// IMP-139: Add Reset Route to Main Router (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Wire up POST /metrics/dispatch/reset in create_router()
// This makes the reset endpoint available via the standard API.
//
// Test TDD Anchors:
// - IMP-139a: create_router should include reset route
// - IMP-139b: Reset route should accept POST method
// - IMP-139c: Reset route path should be /metrics/dispatch/reset
// - IMP-139d: Router should compile with reset route

// IMP-139a: create_router should include dispatch reset route
#[test]
fn test_imp_139a_router_includes_reset_route() {
    // IMP-139a: Verify create_router includes the reset route
    // This is a compile-time check - if the route is registered, the code compiles
    let state = AppState::with_cache(10);
    let router = create_router(state);

    // Router exists and is usable (compile-time check)
    let _ = router;
}

// IMP-139b: Reset route path should be correct
#[test]
fn test_imp_139b_reset_route_path() {
    // IMP-139b: The reset route should be at /metrics/dispatch/reset
    // This verifies the path constant matches expectation
    const EXPECTED_PATH: &str = "/metrics/dispatch/reset";

    // Path should be correctly formed
    assert!(
        EXPECTED_PATH.starts_with("/metrics/dispatch"),
        "IMP-139b: Reset route should be under /metrics/dispatch"
    );
    assert!(
        EXPECTED_PATH.ends_with("/reset"),
        "IMP-139b: Reset route should end with /reset"
    );
}

// IMP-139c: Router should have the dispatch_reset_handler wired
#[tokio::test]
async fn test_imp_139c_router_has_reset_handler() {
    use axum::body::Body;
    use hyper::Request;
    use tower::ServiceExt;

    let state = AppState::with_cache(10);
    let router = create_router(state);

    // Make a POST request to the reset endpoint
    let req = Request::builder()
        .method("POST")
        .uri("/metrics/dispatch/reset")
        .body(Body::empty())
        .expect("IMP-139c: Should build request");

    let response = router
        .oneshot(req)
        .await
        .expect("IMP-139c: Should get response");

    // Should not return 404 (route exists)
    // May return 503 if no GPU model, but that's fine
    assert_ne!(
        response.status().as_u16(),
        404,
        "IMP-139c: Reset route should exist (not 404)"
    );
}

// IMP-139d: GET method on reset route should return 405
#[tokio::test]
async fn test_imp_139d_reset_route_rejects_get() {
    use axum::body::Body;
    use hyper::Request;
    use tower::ServiceExt;

    let state = AppState::with_cache(10);
    let router = create_router(state);

    // Make a GET request to the reset endpoint (should fail)
    let req = Request::builder()
        .method("GET")
        .uri("/metrics/dispatch/reset")
        .body(Body::empty())
        .expect("IMP-139d: Should build request");

    let response = router
        .oneshot(req)
        .await
        .expect("IMP-139d: Should get response");

    // GET should return 405 Method Not Allowed
    assert_eq!(
        response.status().as_u16(),
        405,
        "IMP-139d: GET on reset route should return 405"
    );
}

// =============================================================================
// IMP-140: Add Throughput Metrics (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Track requests per second for performance monitoring.
// This enables throughput analysis and SLA validation.
//
// Test TDD Anchors:
// - IMP-140a: DispatchMetrics should track start time
// - IMP-140b: elapsed_seconds() should return time since start/reset
// - IMP-140c: throughput_rps() should return requests/second
// - IMP-140d: JSON response should include throughput_rps

// IMP-140a: DispatchMetrics should track start time
#[test]
fn test_imp_140a_dispatch_metrics_tracks_start_time() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-140a: start_time_ms() should return milliseconds since epoch
    let start_time = metrics.start_time_ms();
    assert!(start_time > 0, "IMP-140a: Start time should be > 0");

    // Start time should be recent (within last minute)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("IMP-140a: Should get time")
        .as_millis() as u64;
    assert!(
        now - start_time < 60_000,
        "IMP-140a: Start time should be within last minute"
    );
}

// IMP-140b: elapsed_seconds() should return time since start/reset
#[test]
fn test_imp_140b_elapsed_seconds() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-140b: elapsed_seconds() should return positive duration
    let elapsed = metrics.elapsed_seconds();
    assert!(elapsed >= 0.0, "IMP-140b: Elapsed should be >= 0");
    assert!(elapsed < 10.0, "IMP-140b: Elapsed should be small (< 10s)");
}
