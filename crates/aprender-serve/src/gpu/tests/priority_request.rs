
#[test]
fn test_priority_request_queue_fifo_for_same_priority_deep_gcov() {
    // Test FIFO ordering for same priority requests
    let mut queue: PriorityRequestQueue<u32> = PriorityRequestQueue::new();

    // Enqueue with same priority
    queue.enqueue(PriorityRequest::new(5, 1));
    queue.enqueue(PriorityRequest::new(5, 2));
    queue.enqueue(PriorityRequest::new(5, 3));

    // Should dequeue in FIFO order
    assert_eq!(
        queue
            .dequeue_highest()
            .expect("GPU operation failed")
            .into_data(),
        1
    );
    assert_eq!(
        queue
            .dequeue_highest()
            .expect("GPU operation failed")
            .into_data(),
        2
    );
    assert_eq!(
        queue
            .dequeue_highest()
            .expect("GPU operation failed")
            .into_data(),
        3
    );
}

#[test]
fn test_token_rate_limiter_acquire_deep_gcov() {
    // Test token rate limiter acquisition
    let mut limiter = TokenRateLimiter::new(10.0, 5);

    assert_eq!(limiter.tokens_available(), 5);

    // Acquire some tokens
    assert!(limiter.try_acquire(3));
    assert_eq!(limiter.tokens_available(), 2);

    // Try to acquire more than available
    assert!(!limiter.try_acquire(5));
    assert_eq!(limiter.tokens_available(), 2);

    // Acquire remaining
    assert!(limiter.try_acquire(2));
    assert_eq!(limiter.tokens_available(), 0);
}

#[test]
fn test_resource_tracker_allocation_deep_gcov() {
    // Test resource tracker allocation and release
    let mut tracker = ResourceTracker::new(1024, 100);

    // Verify can_allocate
    assert!(tracker.can_allocate(500, 50));
    assert!(!tracker.can_allocate(2000, 50)); // Exceeds memory
    assert!(!tracker.can_allocate(500, 150)); // Exceeds compute

    // Allocate
    let id = tracker.allocate(500, 50);
    assert!(id.is_some());
    let id = id.expect("GPU operation failed");

    assert_eq!(tracker.memory_usage(), 500);
    assert_eq!(tracker.compute_usage(), 50);

    // Try exceeding limits
    let id2 = tracker.allocate(600, 60); // 500+600 > 1024
    assert!(id2.is_none());

    // Release and verify
    tracker.release(id);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_usage_percentage_deep_gcov() {
    // Test resource usage percentage calculation
    let mut tracker = ResourceTracker::new(1000, 100);

    tracker.allocate(250, 25);
    let (mem_pct, compute_pct) = tracker.usage_percentage();

    assert!((mem_pct - 25.0).abs() < 0.1);
    assert!((compute_pct - 25.0).abs() < 0.1);
}

#[test]
fn test_resource_tracker_zero_capacity_deep_gcov() {
    // Test edge case with zero capacity
    let tracker = ResourceTracker::new(0, 0);
    let (mem_pct, compute_pct) = tracker.usage_percentage();

    // Should return 0.0 to avoid division by zero
    assert_eq!(mem_pct, 0.0);
    assert_eq!(compute_pct, 0.0);
}

#[test]
fn test_inference_metrics_percentile_deep_gcov() {
    // Test latency percentile calculation
    use std::time::Duration;

    let mut metrics = InferenceMetrics::new();

    // Add some latencies
    for i in 1..=10 {
        metrics.record_inference(Duration::from_millis(i * 10), 1);
    }

    assert_eq!(metrics.total_inferences(), 10);
    assert_eq!(metrics.total_tokens(), 10);

    // Check percentiles
    let p50 = metrics.latency_percentile(50);
    assert!(p50.is_some());

    let p99 = metrics.latency_percentile(99);
    assert!(p99.is_some());
}

#[test]
fn test_inference_metrics_empty_percentile_deep_gcov() {
    // Test percentile with no data
    let metrics = InferenceMetrics::new();

    assert!(metrics.latency_percentile(50).is_none());
    assert_eq!(metrics.total_inferences(), 0);
}

#[test]
fn test_health_checker_all_pass_deep_gcov() {
    // Test health checker when all checks pass
    let mut checker = HealthChecker::new();

    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| true));

    let results = checker.check_all();
    assert_eq!(results.len(), 2);
    assert!(results["check1"]);
    assert!(results["check2"]);
    assert!(checker.is_healthy());
}

#[test]
fn test_health_checker_some_fail_deep_gcov() {
    // Test health checker when some checks fail
    let mut checker = HealthChecker::new();

    checker.register_check("passing", Box::new(|| true));
    checker.register_check("failing", Box::new(|| false));

    let _ = checker.check_all();
    assert!(!checker.is_healthy());
}

#[test]
fn test_health_checker_empty_is_healthy_deep_gcov() {
    // Test that empty checker is considered healthy
    let checker = HealthChecker::new();
    assert!(checker.is_healthy());
}

#[test]
fn test_cache_aligned_buffer_alignment_deep_gcov() {
    // Test cache aligned buffer actually aligns data
    let buffer = CacheAlignedBuffer::new(256);

    assert_eq!(buffer.len(), 256);
    assert!(!buffer.is_empty());

    // Check alignment (should be 64-byte aligned)
    assert!(buffer.is_aligned(64));
}

#[test]
fn test_cache_aligned_buffer_mut_access_deep_gcov() {
    // Test mutable access to aligned buffer
    let mut buffer = CacheAlignedBuffer::new(100);

    let slice = buffer.as_mut_slice();
    slice.fill(42.0);

    assert!(buffer.as_slice().iter().all(|&x| x == 42.0));
}

#[test]
fn test_contiguous_attention_buffer_views_deep_gcov() {
    // Test attention buffer views
    let mut buffer = ContiguousAttentionBuffer::new(10, 4, 32);

    assert!(buffer.is_contiguous());
    assert_eq!(buffer.max_seq_len(), 10);

    // Get mutable views and modify
    let (q, k, v, o) = buffer.get_views_mut();
    q.fill(1.0);
    k.fill(2.0);
    v.fill(3.0);
    o.fill(4.0);

    // Verify through immutable views
    let (q, k, v, o) = buffer.get_views();
    assert!(q.iter().all(|&x| x == 1.0));
    assert!(k.iter().all(|&x| x == 2.0));
    assert!(v.iter().all(|&x| x == 3.0));
    assert!(o.iter().all(|&x| x == 4.0));

    // Reset and verify
    buffer.reset();
    let (q, _, _, _) = buffer.get_views();
    assert!(q.iter().all(|&x| x == 0.0));
}

#[test]
fn test_batch_embed_out_of_bounds_token_deep_gcov() {
    // Test batch embedding with out-of-bounds tokens
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens, dim 3
    let tokens = vec![0, 1, 999]; // Last token is out of bounds

    let result = batch_embed(&embedding_table, &tokens, 3);

    // First two tokens should be correct, third should be zeros
    assert_eq!(result.len(), 9);
    assert_eq!(&result[0..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&result[3..6], &[4.0, 5.0, 6.0]);
    assert_eq!(&result[6..9], &[0.0, 0.0, 0.0]); // Padded with zeros
}

#[test]
fn test_batch_embed_empty_inputs_deep_gcov() {
    // Test batch embedding with empty inputs
    let embedding_table = vec![1.0, 2.0, 3.0];
    let empty_tokens: Vec<usize> = vec![];

    let result = batch_embed(&embedding_table, &empty_tokens, 3);
    assert!(result.is_empty());

    let empty_table: Vec<f32> = vec![];
    let result = batch_embed(&empty_table, &[0, 1], 3);
    assert!(result.is_empty());
}

#[test]
fn test_sequential_ffn_empty_input_deep_gcov() {
    // Test sequential FFN with empty input
    let result = sequential_ffn(&[], &[0.0; 8], &[0.0; 4], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_parallel_ffn_empty_input_deep_gcov() {
    // Test parallel FFN with empty input
    let result = parallel_ffn(&[], &[0.0; 8], &[0.0; 4], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_softmax_empty_deep_gcov() {
    // Test softmax with empty input
    let result = scalar_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_softmax_empty_deep_gcov() {
    // Test SIMD softmax with empty input
    let result = simd_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_empty_deep_gcov() {
    // Test scalar RoPE with empty/zero inputs
    assert!(scalar_rope(&[], 0, 0, 10000.0).is_empty());
    assert!(scalar_rope(&[1.0], 0, 1, 10000.0).is_empty());
    assert!(scalar_rope(&[1.0], 1, 0, 10000.0).is_empty());
}

#[test]
fn test_simd_rope_empty_deep_gcov() {
    // Test SIMD RoPE with empty/zero inputs
    assert!(simd_rope(&[], 0, 0, 10000.0).is_empty());
    assert!(simd_rope(&[1.0], 0, 1, 10000.0).is_empty());
    assert!(simd_rope(&[1.0], 1, 0, 10000.0).is_empty());
}

#[test]
fn test_fused_layernorm_empty_deep_gcov() {
    // Test fused layernorm with empty input
    let result = fused_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_standard_layernorm_empty_deep_gcov() {
    // Test standard layernorm with empty input
    let result = standard_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_exceeds_gpu_buffer_limit_deep_gcov() {
    // Test GPU buffer limit checking
    let small = 1000;
    let huge = 100_000_000; // 400 MB in f32

    assert!(!exceeds_gpu_buffer_limit(small));
    assert!(exceeds_gpu_buffer_limit(huge));
}

#[test]
fn test_naive_matmul_small_deep_gcov() {
    // Test naive matmul correctness
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let c = naive_matmul(&a, &b, 2, 2, 2);

    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[1] - 22.0).abs() < 1e-5);
    assert!((c[2] - 43.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_deep_gcov() {
    // Test blocked matmul with various block sizes
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    // Block size 1 (maximum blocking)
    let c1 = blocked_matmul(&a, &b, 2, 2, 2, 1);

    // Block size 4 (no blocking effectively)
    let c4 = blocked_matmul(&a, &b, 2, 2, 2, 4);

    // Both should produce same result
    for i in 0..4 {
        assert!((c1[i] - c4[i]).abs() < 1e-5);
    }
}

#[test]
fn test_sum_with_prefetch_deep_gcov() {
    // Test sum with prefetch hints
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let sum1 = sequential_sum(&data);
    let sum2 = sum_with_prefetch(&data, 8);

    assert!((sum1 - sum2).abs() < 1e-3);
}

#[test]
fn test_prefetch_read_bounds_deep_gcov() {
    // Test prefetch with out-of-bounds position
    let data = vec![1.0, 2.0, 3.0];

    // Should not panic for out-of-bounds prefetch
    prefetch_read(&data, 0, 100);
    prefetch_read(&data, 2, 10);
}

#[test]
fn test_gpu_pool_stats_deep_gcov() {
    // Test GPU pool statistics
    let mut pool = GpuBufferPool::new();

    let b1 = pool.acquire(1024);
    let b2 = pool.acquire(2048);
    pool.release(b1);
    pool.release(b2);

    let stats = pool.stats();
    assert!(stats.cached_buffers >= 2);
    assert!(stats.cached_bytes > 0);
}

// --- GpuBufferPool Tests ---
#[test]
fn test_gpu_buffer_pool_new_ext_cov() {
    let pool = GpuBufferPool::new();
    // Pool should be created with default configuration
    assert!(!pool.bucket_sizes().is_empty());
}

#[test]
fn test_gpu_buffer_pool_default_ext_cov() {
    let pool = GpuBufferPool::default();
    // Default should match new()
    assert!(!pool.bucket_sizes().is_empty());
}

#[test]
fn test_gpu_buffer_pool_acquire_new_buffer_ext_cov() {
    let mut pool = GpuBufferPool::new();
    let buffer = pool.acquire(100);
    assert!(buffer.len() >= 100);
}

#[test]
fn test_gpu_buffer_pool_acquire_release_reuse_ext_cov() {
    let mut pool = GpuBufferPool::new();

    // Acquire a buffer
    let buffer = pool.acquire(1024);
    assert!(buffer.len() >= 1024);

    // Release it back
    pool.release(buffer);

    // Acquire again - should reuse
    let buffer2 = pool.acquire(1024);
    assert!(buffer2.len() >= 1024);
}

#[test]
fn test_gpu_buffer_pool_multiple_sizes_ext_cov() {
    let mut pool = GpuBufferPool::new();

    let small = pool.acquire(100);
    let medium = pool.acquire(10000);
    let large = pool.acquire(1000000);

    assert!(small.len() >= 100);
    assert!(medium.len() >= 10000);
    assert!(large.len() >= 1000000);
}

// --- AsyncGpuResult Tests ---
#[test]
fn test_async_gpu_result_pending_ext_cov() {
    let result = AsyncGpuResult::pending();
    assert!(!result.is_ready());
}
