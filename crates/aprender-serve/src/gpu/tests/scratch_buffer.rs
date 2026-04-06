use crate::gpu::*;
#[test]
fn test_scratch_buffer_new_deep2() {
    let scratch = ScratchBuffer::new(4, 256);
    assert_eq!(scratch.num_layers(), 4);
    assert_eq!(scratch.layer_size(), 256);
    assert_eq!(scratch.total_size(), 1024);
}

#[test]
fn test_scratch_buffer_get_layer_deep2() {
    let scratch = ScratchBuffer::new(2, 100);
    let layer0 = scratch.get_layer(0);
    assert_eq!(layer0.len(), 100);
    let layer1 = scratch.get_layer(1);
    assert_eq!(layer1.len(), 100);
}

#[test]
fn test_scratch_buffer_get_layer_mut_deep2() {
    let mut scratch = ScratchBuffer::new(2, 50);
    {
        let layer = scratch.get_layer_mut(0);
        layer[0] = 42.0;
    }
    assert_eq!(scratch.get_layer(0)[0], 42.0);
}

#[test]
fn test_scratch_buffer_reset_deep2() {
    let mut scratch = ScratchBuffer::new(2, 50);
    scratch.get_layer_mut(0)[0] = 99.0;
    scratch.reset();
    assert_eq!(scratch.get_layer(0)[0], 0.0);
}

// --- GpuBufferPool tests ---
#[test]
fn test_gpu_buffer_pool_new_deep2() {
    let pool = GpuBufferPool::new();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
}

#[test]
fn test_gpu_buffer_pool_acquire_release_deep2() {
    let mut pool = GpuBufferPool::new();
    let buf = pool.acquire(100);
    assert_eq!(buf.len(), 100);
    pool.release(buf);
    let stats = pool.stats();
    assert!(stats.cached_buffers > 0 || stats.cached_bytes > 0);
}

#[test]
fn test_gpu_buffer_pool_clear_deep2() {
    let mut pool = GpuBufferPool::new();
    let buf = pool.acquire(1024);
    pool.release(buf);
    pool.clear();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
}

// --- GpuCompute additional tests ---
#[test]
fn test_gpu_compute_empty_inputs_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let empty: Vec<f32> = vec![];
    let result = compute.relu(&empty);
    assert!(result.is_ok());
    assert!(result.expect("test").is_empty());
}

#[test]
fn test_gpu_compute_large_matmul_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let c = compute.matmul(&a, &b, 64, 64, 64);
    assert!(c.is_ok());
    assert_eq!(c.expect("test").len(), 64 * 64);
}

// --- StreamingKVCache additional tests ---
#[test]
fn test_streaming_kv_cache_get_valid_deep2() {
    let mut cache = StreamingKVCache::new(4, 100, 8, 64);
    let kv_dim = 8 * 64;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Need to append to all layers for each position
    for layer in 0..4 {
        cache.append(layer, &k, &v);
    }

    let (keys, values) = cache.get_valid(0);
    assert!(!keys.is_empty());
    assert!(!values.is_empty());
    assert_eq!(keys.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_get_range_deep2() {
    let mut cache = StreamingKVCache::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Fill 3 positions
    for _ in 0..3 {
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }
    assert_eq!(cache.len(), 3);

    let (keys, values) = cache.get_range(0, 0, 2);
    assert_eq!(keys.len(), 2 * kv_dim);
    assert_eq!(values.len(), 2 * kv_dim);
}

#[test]
fn test_streaming_kv_cache_clear_deep2() {
    let mut cache = StreamingKVCache::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// --- LARGE_VOCAB_THRESHOLD test ---
#[test]
fn test_large_vocab_threshold_constant_cov() {
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

// ========================================================================
// Extended Coverage Tests (_ext_cov suffix)
// ========================================================================

// --- InferenceMetrics Tests ---
#[test]
fn test_inference_metrics_new_ext_cov() {
    let metrics = InferenceMetrics::new();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

#[test]
fn test_inference_metrics_default_ext_cov() {
    let metrics = InferenceMetrics::default();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

#[test]
fn test_inference_metrics_record_inference_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    assert_eq!(metrics.total_inferences(), 1);
    assert_eq!(metrics.total_tokens(), 10);

    metrics.record_inference(std::time::Duration::from_millis(200), 20);
    assert_eq!(metrics.total_inferences(), 2);
    assert_eq!(metrics.total_tokens(), 30);
}

#[test]
fn test_inference_metrics_latency_percentile_empty_ext_cov() {
    let metrics = InferenceMetrics::new();
    assert!(metrics.latency_percentile(50).is_none());
    assert!(metrics.latency_percentile(99).is_none());
}

#[test]
fn test_inference_metrics_latency_percentile_single_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    let p50 = metrics.latency_percentile(50);
    assert!(p50.is_some());
    assert_eq!(
        p50.expect("GPU operation failed"),
        std::time::Duration::from_millis(100)
    );
}

#[test]
fn test_inference_metrics_latency_percentile_multiple_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    metrics.record_inference(std::time::Duration::from_millis(200), 10);
    metrics.record_inference(std::time::Duration::from_millis(300), 10);
    metrics.record_inference(std::time::Duration::from_millis(400), 10);
    metrics.record_inference(std::time::Duration::from_millis(500), 10);

    // p0 should be the minimum
    let p0 = metrics.latency_percentile(0).expect("GPU operation failed");
    assert_eq!(p0, std::time::Duration::from_millis(100));

    // p100 should be near the maximum
    let p100 = metrics
        .latency_percentile(100)
        .expect("GPU operation failed");
    assert_eq!(p100, std::time::Duration::from_millis(500));
}

#[test]
fn test_inference_metrics_throughput_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 100);
    // Throughput should be positive
    let throughput = metrics.throughput();
    assert!(throughput >= 0.0);
}

#[test]
fn test_inference_metrics_reset_ext_cov() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(std::time::Duration::from_millis(100), 10);
    metrics.record_inference(std::time::Duration::from_millis(200), 20);
    assert_eq!(metrics.total_inferences(), 2);
    assert_eq!(metrics.total_tokens(), 30);

    metrics.reset();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

// --- HealthChecker Tests ---
#[test]
fn test_health_checker_new_ext_cov() {
    let checker = HealthChecker::new();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy()); // Empty checks = healthy
}

#[test]
fn test_health_checker_default_ext_cov() {
    let checker = HealthChecker::default();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy());
}

#[test]
fn test_health_checker_register_check_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("test_check", Box::new(|| true));
    assert_eq!(checker.check_count(), 1);
}

#[test]
fn test_health_checker_check_all_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("always_healthy", Box::new(|| true));
    checker.register_check("always_unhealthy", Box::new(|| false));

    let results = checker.check_all();
    assert_eq!(results.len(), 2);
    assert_eq!(results.get("always_healthy"), Some(&true));
    assert_eq!(results.get("always_unhealthy"), Some(&false));
}

#[test]
fn test_health_checker_is_healthy_all_pass_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| true));

    checker.check_all(); // Run checks first
    assert!(checker.is_healthy());
}

#[test]
fn test_health_checker_is_healthy_one_fails_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| false));

    checker.check_all(); // Run checks first
    assert!(!checker.is_healthy());
}

#[test]
fn test_health_checker_clear_ext_cov() {
    let mut checker = HealthChecker::new();
    checker.register_check("check1", Box::new(|| true));
    checker.check_all();

    checker.clear();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy()); // Empty = healthy
}

#[test]
fn test_health_checker_debug_ext_cov() {
    let checker = HealthChecker::new();
    let debug_str = format!("{:?}", checker);
    assert!(debug_str.contains("HealthChecker"));
}

// --- ShutdownCoordinator Tests ---
#[test]
fn test_shutdown_coordinator_new_ext_cov() {
    let coordinator = ShutdownCoordinator::new();
    assert!(!coordinator.is_shutting_down());
    assert_eq!(coordinator.pending_requests(), 0);
    assert_eq!(coordinator.handler_count(), 0);
}

#[test]
fn test_shutdown_coordinator_default_ext_cov() {
    let coordinator = ShutdownCoordinator::default();
    assert!(!coordinator.is_shutting_down());
    assert_eq!(coordinator.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_register_handler_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();
    coordinator.register_handler(Box::new(|| {}));
    assert_eq!(coordinator.handler_count(), 1);

    coordinator.register_handler(Box::new(|| {}));
    assert_eq!(coordinator.handler_count(), 2);
}

#[test]
fn test_shutdown_coordinator_request_lifecycle_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();

    coordinator.request_started();
    assert_eq!(coordinator.pending_requests(), 1);

    coordinator.request_started();
    assert_eq!(coordinator.pending_requests(), 2);

    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 1);

    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_request_completed_saturating_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();

    // Should not underflow
    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 0);

    coordinator.request_completed();
    assert_eq!(coordinator.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_initiate_shutdown_ext_cov() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut coordinator = ShutdownCoordinator::new();
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = Arc::clone(&called);

    coordinator.register_handler(Box::new(move || {
        called_clone.store(true, Ordering::SeqCst);
    }));

    assert!(!coordinator.is_shutting_down());
    coordinator.initiate_shutdown();
    assert!(coordinator.is_shutting_down());
    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn test_shutdown_coordinator_initiate_shutdown_idempotent_ext_cov() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let mut coordinator = ShutdownCoordinator::new();
    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&call_count);

    coordinator.register_handler(Box::new(move || {
        count_clone.fetch_add(1, Ordering::SeqCst);
    }));

    coordinator.initiate_shutdown();
    coordinator.initiate_shutdown(); // Should not call handler again
    coordinator.initiate_shutdown();

    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_shutdown_coordinator_is_complete_ext_cov() {
    let mut coordinator = ShutdownCoordinator::new();

    // Not complete yet - not shutting down
    assert!(!coordinator.is_complete());

    coordinator.request_started();
    coordinator.initiate_shutdown();
    // Not complete - still has pending requests
    assert!(!coordinator.is_complete());

    coordinator.request_completed();
    // Now complete
    assert!(coordinator.is_complete());
}

#[test]
fn test_shutdown_coordinator_debug_ext_cov() {
    let coordinator = ShutdownCoordinator::new();
    let debug_str = format!("{:?}", coordinator);
    assert!(debug_str.contains("ShutdownCoordinator"));
    assert!(debug_str.contains("shutting_down"));
}

// --- StreamingKVCacheFp16 Tests ---
#[test]
fn test_streaming_kv_cache_fp16_new_ext_cov() {
    let cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 10);
}

include!("streaming_02.rs");
include!("hybrid_scheduler_02.rs");
include!("priority_request.rs");
include!("async_gpu.rs");
