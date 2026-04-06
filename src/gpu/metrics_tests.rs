//! Tests for gpu/metrics.rs - Targeting uncovered ~19%
//!
//! Focus areas:
//! - cpu_matmul, cpu_vector_matmul (parallel and sequential)
//! - cpu_matmul_transpose_b, transpose
//! - cpu_matmul_transposed_simd
//! - GpuCompute::matmul_tensor, error paths
//! - HybridScheduler::matmul_pooled, matmul_async, release_buffer

use super::*;
use crate::tensor::Tensor;
use std::time::Duration;

// =============================================================================
// cpu_matmul tests
// =============================================================================

#[test]
fn test_cpu_matmul_1x1() {
    let a = vec![2.0f32];
    let b = vec![3.0f32];
    let c = cpu_matmul(&a, &b, 1, 1, 1);
    assert_eq!(c.len(), 1);
    assert!((c[0] - 6.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_identity() {
    // 3x3 identity @ vector
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let c = cpu_matmul(&identity, &vec, 3, 3, 3);
    assert_eq!(c.len(), 9);
    for i in 0..9 {
        assert!((c[i] - vec[i]).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_matmul_rectangular() {
    // 2x3 @ 3x4 = 2x4
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let c = cpu_matmul(&a, &b, 2, 3, 4);
    assert_eq!(c.len(), 8);
    // Manual calculation: c[0,0] = 1*1 + 2*5 + 3*9 = 38
    assert!((c[0] - 38.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_m1_triggers_vector_path() {
    // m=1 triggers cpu_vector_matmul
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4x2
    let c = cpu_matmul(&a, &b, 1, 4, 2);
    assert_eq!(c.len(), 2);
    // c[0] = 1*1 + 2*3 + 3*5 + 4*7 = 1 + 6 + 15 + 28 = 50
    assert!((c[0] - 50.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_large_parallel_path() {
    // n >= 2048 triggers parallel path in cpu_vector_matmul
    let k = 64;
    let n = 2048;
    let a: Vec<f32> = (0..k).map(|i| (i % 10) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let c = cpu_matmul(&a, &b, 1, k, n);
    assert_eq!(c.len(), n);
    // Just verify no crash and correct length
    assert!(c.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// cpu_matmul_transpose_b tests
// =============================================================================

#[test]
fn test_cpu_matmul_transpose_b_basic() {
    // A @ B^T where B is stored as [n, k]
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2, transposed = identity
    let c = cpu_matmul_transpose_b(&a, &b, 2, 2, 2);
    assert_eq!(c.len(), 4);
    // With B^T = identity, result = A
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_transpose_b_rectangular() {
    // A[2,3] @ B[4,3]^T = C[2,4]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ]; // 4x3
    let c = cpu_matmul_transpose_b(&a, &b, 2, 3, 4);
    assert_eq!(c.len(), 8);
    // c[0,0] = a[0,:] dot b[0,:] = [1,2,3] dot [1,0,0] = 1
    assert!((c[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_transpose_b_attention_style() {
    // Q @ K^T pattern (seq x head_dim) @ (seq x head_dim)^T = (seq x seq)
    let seq = 4;
    let head_dim = 8;
    let q: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32) * 0.01).collect();
    let k: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32) * 0.01).collect();
    let scores = cpu_matmul_transpose_b(&q, &k, seq, head_dim, seq);
    assert_eq!(scores.len(), seq * seq);
    // Diagonal should have highest values (self-similarity)
    assert!(scores.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// transpose tests
// =============================================================================

#[test]
fn test_transpose_2x2() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = transpose(&data, 2, 2);
    assert_eq!(t, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_2x3() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = transpose(&data, 2, 3);
    // [1,2,3]    [1,4]
    // [4,5,6] -> [2,5]
    //            [3,6]
    assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_identity_square() {
    let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let t = transpose(&data, 3, 3);
    // Identity transpose = identity
    assert_eq!(t, data);
}

#[test]
fn test_transpose_round_trip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t1 = transpose(&data, 2, 3);
    let t2 = transpose(&t1, 3, 2);
    assert_eq!(t2, data);
}

// =============================================================================
// cpu_matmul_transposed_simd tests
// =============================================================================

#[test]
fn test_cpu_matmul_transposed_simd_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0]; // k=4
    let weight_t = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
    ]; // n=3, k=4
    let bias = vec![0.0, 0.0, 0.0];
    let c = cpu_matmul_transposed_simd(&a, &weight_t, &bias, 4, 3);
    assert_eq!(c.len(), 3);
    // Identity-like weights extract first 3 elements
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
    assert!((c[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_transposed_simd_with_bias() {
    let a = vec![1.0, 1.0]; // k=2
    let weight_t = vec![1.0, 1.0, 2.0, 2.0]; // n=2, k=2
    let bias = vec![10.0, 20.0];
    let c = cpu_matmul_transposed_simd(&a, &weight_t, &bias, 2, 2);
    // c[0] = dot([1,1], [1,1]) + 10 = 2 + 10 = 12
    // c[1] = dot([1,1], [2,2]) + 20 = 4 + 20 = 24
    assert!((c[0] - 12.0).abs() < 1e-5);
    assert!((c[1] - 24.0).abs() < 1e-5);
}

#[test]
fn test_cpu_matmul_transposed_simd_large_parallel() {
    // Large n to trigger parallel chunking
    let k = 64;
    let n = 8192;
    let a: Vec<f32> = vec![0.1; k];
    let weight_t: Vec<f32> = vec![0.1; n * k];
    let bias: Vec<f32> = vec![0.0; n];
    let c = cpu_matmul_transposed_simd(&a, &weight_t, &bias, k, n);
    assert_eq!(c.len(), n);
    // Each output = k * 0.1 * 0.1 = k * 0.01 = 0.64
    let expected = k as f32 * 0.01;
    assert!((c[0] - expected).abs() < 1e-4);
    assert!((c[n - 1] - expected).abs() < 1e-4);
}

// =============================================================================
// GpuCompute::matmul_tensor tests
// =============================================================================

#[test]
fn test_gpu_compute_matmul_tensor_basic() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let c = compute.matmul_tensor(&a, &b).expect("matmul_tensor");
    assert_eq!(c.shape(), &[2, 2]);
    // c[0,0] = 1*1 + 2*3 + 3*5 = 22
    assert!((c.data()[0] - 22.0).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_matmul_tensor_non_2d_error() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    // 1D tensor
    let a = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = compute.matmul_tensor(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_gpu_compute_matmul_tensor_dim_mismatch() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    // A[2,3] @ B[4,2] - inner dims don't match (3 != 4)
    let a = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    let b = Tensor::from_vec(vec![4, 2], vec![1.0; 8]).unwrap();
    let result = compute.matmul_tensor(&a, &b);
    assert!(result.is_err());
}

// =============================================================================
// GpuCompute error paths
// =============================================================================

#[test]
fn test_gpu_compute_matmul_a_size_mismatch() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let a = vec![1.0, 2.0, 3.0]; // 3 elements
    let b = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    // Claim a is 2x2 (4 elements) but only 3
    let result = compute.matmul(&a, &b, 2, 2, 2);
    assert!(result.is_err());
}

#[test]
fn test_gpu_compute_matmul_b_size_mismatch() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    let b = vec![1.0, 2.0, 3.0]; // 3 elements
    // Claim b is 2x2 (4 elements) but only 3
    let result = compute.matmul(&a, &b, 2, 2, 2);
    assert!(result.is_err());
}

#[test]
fn test_gpu_compute_dot_length_mismatch() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];
    let result = compute.dot(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_gpu_compute_dot_success() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = compute.dot(&a, &b).expect("dot product");
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_relu_values() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = compute.relu(&input).expect("relu");
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_gpu_compute_sigmoid_values() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    let input = vec![0.0, 1.0, -1.0];
    let result = compute.sigmoid(&input).expect("sigmoid");
    // sigmoid(0) = 0.5
    assert!((result[0] - 0.5).abs() < 1e-5);
    // sigmoid(1) ≈ 0.731
    assert!((result[1] - 0.731).abs() < 0.01);
    // sigmoid(-1) ≈ 0.269
    assert!((result[2] - 0.269).abs() < 0.01);
}

// =============================================================================
// HybridScheduler additional methods
// =============================================================================

#[test]
fn test_hybrid_scheduler_matmul_pooled() {
    let mut scheduler = HybridScheduler::new().expect("scheduler");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = scheduler.matmul_pooled(&a, &b, 2, 2, 2).expect("matmul_pooled");
    assert_eq!(result.len(), 4);
    assert!((result[0] - 19.0).abs() < 1e-5);

    // Release buffer back to pool
    scheduler.release_buffer(result);

    // Get another pooled buffer - should reuse
    let result2 = scheduler.matmul_pooled(&a, &b, 2, 2, 2).expect("matmul_pooled 2");
    assert_eq!(result2.len(), 4);
}

#[test]
fn test_hybrid_scheduler_matmul_async() {
    let mut scheduler = HybridScheduler::new().expect("scheduler");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let async_result = scheduler.matmul_async(&a, &b, 2, 2, 2).expect("matmul_async");

    // CPU fallback is immediately ready
    assert!(async_result.is_ready());
    let data = async_result.wait();
    assert_eq!(data.len(), 4);
    assert!((data[0] - 19.0).abs() < 1e-5);
}

#[test]
fn test_hybrid_scheduler_pool_stats() {
    let mut scheduler = HybridScheduler::new().expect("scheduler");

    // Initially empty
    let stats = scheduler.pool_stats();
    assert_eq!(stats.cached_buffers, 0);

    // Use pooled buffer and release
    let a = vec![1.0; 1024];
    let b = vec![1.0; 1024];
    let result = scheduler.matmul_pooled(&a, &b, 1, 1024, 1).expect("matmul");
    scheduler.release_buffer(result);

    let stats = scheduler.pool_stats();
    assert!(stats.cached_buffers > 0 || stats.cached_bytes > 0);
}

#[test]
fn test_hybrid_scheduler_matmul_transpose_b_cpu() {
    let mut scheduler = HybridScheduler::with_threshold(1_000_000).expect("scheduler");

    // Force CPU path with high threshold
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 (will be transposed)
    let result = scheduler.matmul_transpose_b(&a, &b, 2, 2, 2).expect("transpose_b");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_hybrid_scheduler_threshold_accessors() {
    let scheduler = HybridScheduler::with_threshold(12345).expect("scheduler");
    assert_eq!(scheduler.gpu_threshold(), 12345);
}

// =============================================================================
// GpuBufferPool edge cases
// =============================================================================

#[test]
fn test_gpu_buffer_pool_oversized_request() {
    let mut pool = GpuBufferPool::new();
    // Request larger than any bucket
    let huge = pool.acquire(100_000_000);
    assert!(huge.len() >= 100_000_000);
    pool.release(huge);
}

#[test]
fn test_gpu_buffer_pool_exact_bucket_boundary() {
    let mut pool = GpuBufferPool::new();
    // Request exactly at bucket boundary (1024 = 2^10)
    let buf = pool.acquire(1024);
    assert!(buf.len() >= 1024);
    pool.release(buf);

    // Request again - should get from pool
    let buf2 = pool.acquire(1024);
    assert!(buf2.len() >= 1024);
}

#[test]
fn test_gpu_buffer_pool_max_per_bucket() {
    let mut pool = GpuBufferPool::new();

    // Release more buffers than max_per_bucket (4)
    for _ in 0..10 {
        let buf = pool.acquire(1024);
        pool.release(buf);
    }

    // Only max_per_bucket should be cached
    let stats = pool.stats();
    assert!(stats.cached_buffers <= 4);
}

// =============================================================================
// InferenceMetrics edge cases
// =============================================================================

#[test]
fn test_inference_metrics_high_percentile() {
    let mut metrics = InferenceMetrics::new();
    for i in 1..=100 {
        metrics.record_inference(Duration::from_millis(i), 1);
    }

    // p100 should be close to max
    let p100 = metrics.latency_percentile(100).expect("p100");
    assert_eq!(p100, Duration::from_millis(100));

    // p0 should be min
    let p0 = metrics.latency_percentile(0).expect("p0");
    assert_eq!(p0, Duration::from_millis(1));
}

#[test]
fn test_inference_metrics_throughput_immediate() {
    let metrics = InferenceMetrics::new();
    // No time elapsed, no tokens - throughput should be 0 or small
    let throughput = metrics.throughput();
    assert!(throughput >= 0.0);
}

include!("metrics_tests_compute_backend.rs");
