
#[test]
fn test_resource_limiter_check_memory_denied_per_request() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(100)
        .with_max_total_memory(10000);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.check_memory(200);
    assert!(matches!(result, LimitResult::Denied { .. }));
}

#[test]
fn test_resource_limiter_allocate() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(1024)
        .with_max_total_memory(10240);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.allocate(512);
    assert!(result.is_ok());
    assert_eq!(limiter.current_memory(), 512);
}

#[test]
fn test_resource_limiter_deallocate() {
    let config = ResourceConfig::new()
        .with_max_memory_per_request(1024)
        .with_max_total_memory(10240);
    let limiter = ResourceLimiter::new(config);

    let _ = limiter.allocate(512);
    limiter.deallocate(256);

    assert_eq!(limiter.current_memory(), 256);
}

#[test]
fn test_resource_limiter_enqueue() {
    let config = ResourceConfig::new().with_max_queue_depth(10);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.enqueue();
    assert!(matches!(result, LimitResult::Allowed));
}

#[test]
fn test_resource_limiter_enqueue_backpressure() {
    let config = ResourceConfig::new().with_max_queue_depth(2);
    let limiter = ResourceLimiter::new(config);

    let _ = limiter.enqueue();
    let _ = limiter.enqueue();
    let result = limiter.enqueue();

    assert!(matches!(result, LimitResult::Backpressure));
}

#[test]
fn test_resource_limiter_try_enqueue() {
    let config = ResourceConfig::new().with_max_queue_depth(5);
    let limiter = ResourceLimiter::new(config);

    let result = limiter.try_enqueue();
    assert!(matches!(result, LimitResult::Allowed));
}

#[test]
fn test_resource_limiter_dequeue() {
    let config = ResourceConfig::new().with_max_queue_depth(10);
    let limiter = ResourceLimiter::new(config);

    let _ = limiter.enqueue();
    limiter.dequeue();

    // Should not panic on dequeue
}

#[test]
fn test_resource_limiter_start_compute() {
    let config = ResourceConfig::new();
    let limiter = ResourceLimiter::new(config);

    let start = limiter.start_compute();
    // Verify start_compute returns a valid Instant by checking it doesn't panic
    let _ = start.elapsed();
}

// ============================================================================
// ResourceMonitor Tests
// ============================================================================

#[test]
fn test_resource_monitor_new() {
    let monitor = ResourceMonitor::new();
    let metrics = monitor.current_metrics();

    assert_eq!(metrics.memory_bytes, 0);
    assert_eq!(metrics.queue_depth, 0);
}

#[test]
fn test_resource_monitor_default() {
    let monitor = ResourceMonitor::default();
    let metrics = monitor.current_metrics();
    assert_eq!(metrics.memory_bytes, 0);
}

#[test]
fn test_resource_monitor_record_memory_usage() {
    let monitor = ResourceMonitor::new();

    monitor.record_memory_usage(1024);

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.memory_bytes, 1024);
}

#[test]
fn test_resource_monitor_record_gpu_utilization() {
    let monitor = ResourceMonitor::new();

    monitor.record_gpu_utilization(75.5);

    let metrics = monitor.current_metrics();
    assert!((metrics.gpu_utilization - 75.5).abs() < 0.1);
}

#[test]
fn test_resource_monitor_record_queue_depth() {
    let monitor = ResourceMonitor::new();

    monitor.record_queue_depth(42);

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.queue_depth, 42);
}

#[test]
fn test_resource_monitor_record_latency() {
    let monitor = ResourceMonitor::new();

    monitor.record_latency(Duration::from_millis(150));

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.last_latency_ms, 150);
}

#[test]
fn test_resource_monitor_latency_stats_empty() {
    let monitor = ResourceMonitor::new();

    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 0);
    assert_eq!(stats.max_ms, 0);
    assert_eq!(stats.avg_ms, 0);
}

#[test]
fn test_resource_monitor_latency_stats() {
    let monitor = ResourceMonitor::new();

    monitor.record_latency(Duration::from_millis(100));
    monitor.record_latency(Duration::from_millis(200));
    monitor.record_latency(Duration::from_millis(300));

    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 100);
    assert_eq!(stats.max_ms, 300);
    assert_eq!(stats.avg_ms, 200);
}

#[test]
fn test_resource_monitor_snapshot() {
    let monitor = ResourceMonitor::new();

    monitor.record_memory_usage(2048);
    monitor.record_gpu_utilization(50.0);
    monitor.record_queue_depth(10);

    let snapshot = monitor.snapshot();

    assert!(snapshot.timestamp > 0);
    assert_eq!(snapshot.memory_bytes, 2048);
    assert!((snapshot.gpu_utilization - 50.0).abs() < 0.1);
    assert_eq!(snapshot.queue_depth, 10);
}

// ============================================================================
// GgufModelState Tests
// ============================================================================

#[test]
fn test_gguf_model_state_new() {
    let state = GgufModelState::new();

    assert!(!state.is_loaded());
    assert!(!state.is_ready());
    assert!(state.model_name().is_none());
    assert_eq!(state.vocab_size(), 0);
    assert!(state.model().is_none());
}

#[test]
fn test_gguf_model_state_default() {
    let state = GgufModelState::default();

    assert!(!state.is_loaded());
    assert!(!state.is_ready());
}

#[test]
fn test_gguf_model_state_debug() {
    let state = GgufModelState::new();
    let debug_str = format!("{:?}", state);

    assert!(debug_str.contains("GgufModelState"));
    assert!(debug_str.contains("is_loaded"));
}

#[test]
fn test_load_gguf_to_gpu() {
    let result = load_gguf_to_gpu(1000, 256, 4);

    assert!(result.is_ok());

    let state = result.expect("test value should be present");
    assert!(state.is_loaded());
    assert!(state.is_ready());
    assert_eq!(state.vocab_size(), 1000);
    assert!(state.model_name().is_some());
}

#[test]
fn test_load_gguf_to_gpu_model_name() {
    let state = load_gguf_to_gpu(500, 128, 2).expect("test value should be present");

    let name = state.model_name().expect("test value should be present");
    assert!(name.contains("500")); // vocab_size in name
    assert!(name.contains("128")); // hidden_dim in name
}

#[test]
fn test_gguf_model_state_model_mut() {
    let mut state = load_gguf_to_gpu(100, 64, 1).expect("test value should be present");

    let model_ref = state.model_mut();
    assert!(model_ref.is_some());
}

#[test]
fn test_gguf_model_state_small_config() {
    // Test with minimal configuration
    let result = load_gguf_to_gpu(50, 64, 1);
    assert!(result.is_ok());

    let state = result.expect("test value should be present");
    assert!(state.is_ready());
}

#[test]
fn test_gguf_model_state_large_config() {
    // Test with larger configuration
    let result = load_gguf_to_gpu(32000, 4096, 32);
    assert!(result.is_ok());

    let state = result.expect("test value should be present");
    assert!(state.is_ready());
    assert_eq!(state.vocab_size(), 32000);
}
