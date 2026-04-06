
#[test]
fn test_simd_softmax_empty_cov() {
    let result = simd_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_empty_cov() {
    let result = scalar_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_empty_cov() {
    let result = simd_rope(&[], 0, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_basic_cov() {
    let embedding_table = vec![1.0f32; 100 * 8]; // 100 tokens, dim 8
    let tokens = vec![0usize, 1, 2];
    let result = batch_embed(&embedding_table, &tokens, 8);
    assert_eq!(result.len(), 3 * 8);
}

#[test]
fn test_sequential_ffn_basic_cov() {
    let hidden = vec![1.0f32; 64];
    let w1 = vec![0.1f32; 64 * 128];
    let w2 = vec![0.1f32; 128 * 64];
    let result = sequential_ffn(&hidden, &w1, &w2, 64, 128);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_parallel_ffn_basic_cov() {
    let hidden = vec![1.0f32; 64];
    let w1 = vec![0.1f32; 64 * 128];
    let w2 = vec![0.1f32; 128 * 64];
    let result = parallel_ffn(&hidden, &w1, &w2, 64, 128);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_standard_layernorm_basic_cov() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
    // Result should be normalized
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_fused_layernorm_basic_cov() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_prefetch_read_cov() {
    let data = vec![1.0f32; 100];
    // Just ensure it doesn't panic
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 20);
    prefetch_read(&data, 90, 10);
}

#[test]
fn test_quantized_dot_q4_basic_cov() {
    let block_a = vec![0u8; 18]; // Q4 block: 2 bytes scale + 16 bytes data
    let block_b = vec![0u8; 18];
    let result = quantized_dot_q4(&block_a, &block_b);
    assert!(result.is_finite());
}

#[test]
fn test_quantized_dot_q8_basic_cov() {
    let block_a = vec![0u8; 34]; // Q8 block: 2 bytes scale + 32 bytes data
    let block_b = vec![0u8; 34];
    let result = quantized_dot_q8(&block_a, &block_b);
    assert!(result.is_finite());
}

#[test]
fn test_quantized_matvec_q4_basic_cov() {
    let rows = 4;
    let cols = 32;
    let weights = vec![0u8; rows * 18]; // Q4 blocks
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q4(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q8_basic_cov() {
    let rows = 4;
    let cols = 32;
    let weights = vec![0u8; rows * 34]; // Q8 blocks
    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q8(&weights, &input, rows, cols);
    assert_eq!(result.len(), rows);
}

#[test]
fn test_large_vocab_threshold_cov() {
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

// =========================================================================
// Extended Coverage Tests for Softmax
// =========================================================================

#[test]
fn test_scalar_softmax_single_element_cov() {
    let result = scalar_softmax(&[1.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_scalar_softmax_uniform_cov() {
    let input = vec![1.0; 4];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 4);
    // Uniform input should give uniform output (0.25 each)
    for &v in &result {
        assert!((v - 0.25).abs() < 1e-6);
    }
}

#[test]
fn test_scalar_softmax_large_values_cov() {
    // Test numerical stability with large values
    let input = vec![1000.0, 1001.0, 1002.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 3);
    // Should still sum to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_scalar_softmax_negative_values_cov() {
    let input = vec![-1.0, -2.0, -3.0];
    let result = scalar_softmax(&input);
    assert_eq!(result.len(), 3);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_single_element_cov() {
    let result = simd_softmax(&[2.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_simd_softmax_uniform_cov() {
    let input = vec![0.0; 8];
    let result = simd_softmax(&input);
    assert_eq!(result.len(), 8);
    for &v in &result {
        assert!((v - 0.125).abs() < 1e-6);
    }
}

#[test]
fn test_simd_softmax_matches_scalar_cov() {
    let input = vec![0.1, 0.5, 0.3, 0.2, 0.8, 0.4, 0.6, 0.7];
    let scalar_result = scalar_softmax(&input);
    let simd_result = simd_softmax(&input);
    assert_eq!(scalar_result.len(), simd_result.len());
    for (s, r) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - r).abs() < 1e-5);
    }
}

// =========================================================================
// Extended Coverage Tests for RoPE
// =========================================================================

#[test]
fn test_scalar_rope_basic_cov() {
    let input = vec![1.0; 16]; // 1 token, 16 hidden
    let result = scalar_rope(&input, 1, 16, 10000.0);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_scalar_rope_multiple_positions_cov() {
    let input = vec![1.0; 64]; // 4 tokens, 16 hidden
    let result = scalar_rope(&input, 4, 16, 10000.0);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_scalar_rope_different_theta_cov() {
    let input = vec![1.0; 32];
    let result1 = scalar_rope(&input, 2, 16, 10000.0);
    let result2 = scalar_rope(&input, 2, 16, 500000.0);
    // Different theta should give different results
    assert_ne!(result1, result2);
}

#[test]
fn test_simd_rope_basic_cov() {
    let input = vec![1.0; 16];
    let result = simd_rope(&input, 1, 16, 10000.0);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_simd_rope_multiple_positions_cov() {
    let input = vec![1.0; 64];
    let result = simd_rope(&input, 4, 16, 10000.0);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_simd_rope_matches_scalar_cov() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let scalar_result = scalar_rope(&input, 2, 16, 10000.0);
    let simd_result = simd_rope(&input, 2, 16, 10000.0);
    assert_eq!(scalar_result.len(), simd_result.len());
    for (s, r) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - r).abs() < 1e-4, "scalar={}, simd={}", s, r);
    }
}

#[test]
fn test_scalar_rope_zero_seq_len_cov() {
    let input = vec![1.0; 16];
    let result = scalar_rope(&input, 0, 16, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_scalar_rope_zero_head_dim_cov() {
    let input = vec![1.0; 16];
    let result = scalar_rope(&input, 1, 0, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_zero_seq_len_cov() {
    let input = vec![1.0; 16];
    let result = simd_rope(&input, 0, 16, 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_simd_rope_zero_head_dim_cov() {
    let input = vec![1.0; 16];
    let result = simd_rope(&input, 1, 0, 10000.0);
    assert!(result.is_empty());
}

// =========================================================================
// Extended Coverage Tests for GPU Compute
// =========================================================================

#[test]
fn test_gpu_compute_dot_empty_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let result = compute.dot(&[], &[]);
    assert!(result.is_err() || result.expect("GPU operation failed").abs() < 1e-10);
}

#[test]
fn test_gpu_compute_relu_empty_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let result = compute.relu(&[]).expect("test");
    assert!(result.is_empty());
}

#[test]
fn test_gpu_compute_sigmoid_multiple_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let input = vec![-100.0, 0.0, 100.0];
    let output = compute.sigmoid(&input).expect("test");
    assert!(output[0] < 0.01); // sigmoid(-100) ≈ 0
    assert!((output[1] - 0.5).abs() < 1e-5);
    assert!(output[2] > 0.99); // sigmoid(100) ≈ 1
}

#[test]
fn test_gpu_compute_matmul_1x1_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let a = vec![3.0];
    let b = vec![4.0];
    let c = compute.matmul(&a, &b, 1, 1, 1).expect("test");
    assert_eq!(c.len(), 1);
    assert!((c[0] - 12.0).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_matmul_large_cov() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");
    let m = 64;
    let k = 64;
    let n = 64;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32 * 0.1).collect();
    let c = compute.matmul(&a, &b, m, k, n).expect("test");
    assert_eq!(c.len(), m * n);
}

// =========================================================================
// Extended Coverage Tests for HybridScheduler
// =========================================================================

#[test]
fn test_hybrid_scheduler_default_threshold_cov() {
    let scheduler = HybridScheduler::new().expect("test");
    // Default threshold should be set
    assert!(scheduler.gpu_threshold() > 0);
}

#[test]
fn test_hybrid_scheduler_has_gpu_cov() {
    let scheduler = HybridScheduler::with_threshold(100).expect("test");
    // has_gpu returns whether GPU backend is active
    let _has_gpu = scheduler.has_gpu();
}

// =========================================================================
// Extended Coverage Tests for GpuBufferPool
// =========================================================================

#[test]
fn test_buffer_pool_multiple_sizes_cov() {
    let mut pool = GpuBufferPool::new();

    // Acquire different sizes
    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(1000);
    let buf3 = pool.acquire(10000);

    assert_eq!(buf1.len(), 100);
    assert_eq!(buf2.len(), 1000);
    assert_eq!(buf3.len(), 10000);

    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);

    // Should have 3 cached buffers
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 3);
}

#[test]
fn test_buffer_pool_stats_bytes_cov() {
    let mut pool = GpuBufferPool::new();

    let buf = pool.acquire(256);
    pool.release(buf);

    let stats = pool.stats();
    assert!(stats.cached_bytes >= 256 * std::mem::size_of::<f32>());
}

// =========================================================================
// Extended Coverage Tests for AsyncGpuResult
// =========================================================================

#[test]
fn test_async_result_set_twice_cov() {
    let mut result = AsyncGpuResult::pending();
    result.set_result(vec![1.0]);
    result.set_result(vec![2.0]); // Second set replaces the first
    assert_eq!(result.wait(), vec![2.0]);
}

#[test]
fn test_async_result_ready_wait_cov() {
    let result = AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    assert!(result.is_ready());
    let data = result.wait();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

// =========================================================================
// Extended Coverage Tests for Constants
// =========================================================================

#[test]
fn test_max_gpu_buffer_bytes_cov() {
    assert_eq!(MAX_GPU_BUFFER_BYTES, 256 * 1024 * 1024);
}

#[test]
fn test_exceeds_limit_edge_cases_cov() {
    // Exactly at limit
    let at_limit = MAX_GPU_BUFFER_BYTES / std::mem::size_of::<f32>();
    assert!(!exceeds_gpu_buffer_limit(at_limit));

    // One over limit
    assert!(exceeds_gpu_buffer_limit(at_limit + 1));
}

// =========================================================================
// Extended Coverage Tests for LayerNorm
// =========================================================================

#[test]
fn test_standard_layernorm_with_bias_cov() {
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let gamma = vec![2.0f32; 4];
    let beta = vec![1.0f32; 4];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_layernorm_with_bias_cov() {
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let gamma = vec![2.0f32; 4];
    let beta = vec![1.0f32; 4];
    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_standard_layernorm_small_eps_cov() {
    let input = vec![1e-10f32; 4];
    let gamma = vec![1.0f32; 4];
    let beta = vec![0.0f32; 4];
    let result = standard_layernorm(&input, &gamma, &beta, 1e-12);
    assert_eq!(result.len(), 4);
    // Should not panic with very small values
}
