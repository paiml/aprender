//! EXTREME TDD coverage tests for GPU-related code in realizar/src/gguf.rs
//!
//! Tests target:
//! - DispatchMetrics: CPU/GPU dispatch tracking and latency measurement
//! - CudaBackend: CUDA PTX generation and configuration (when cuda feature enabled)
//! - GPU buffer pools and async command queues (when gpu feature enabled)
//! - Batch generation stats and GPU warmup paths
//!
//! Run with:
//! ```bash
//! cargo test --test gguf_gpu_coverage 2>&1 | tail -20
//! # With gpu feature:
//! cargo test --test gguf_gpu_coverage --features gpu 2>&1 | tail -20
//! ```

use std::time::Duration;

// ============================================================================
// DispatchMetrics Tests
// ============================================================================

use realizar::gguf::DispatchMetrics;

#[test]
fn test_dispatch_metrics_creation() {
    let metrics = DispatchMetrics::new();
    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
    assert_eq!(metrics.total_dispatches(), 0);
}

#[test]
fn test_dispatch_metrics_default() {
    let metrics = DispatchMetrics::default();
    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
}

#[test]
fn test_dispatch_metrics_cpu_dispatch() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    assert_eq!(metrics.cpu_dispatches(), 1);
    assert_eq!(metrics.gpu_dispatches(), 0);

    metrics.record_cpu_dispatch();
    assert_eq!(metrics.cpu_dispatches(), 2);
}

#[test]
fn test_dispatch_metrics_gpu_dispatch() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_dispatch();
    assert_eq!(metrics.gpu_dispatches(), 1);
    assert_eq!(metrics.cpu_dispatches(), 0);

    metrics.record_gpu_dispatch();
    assert_eq!(metrics.gpu_dispatches(), 2);
}

#[test]
fn test_dispatch_metrics_total_dispatches() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    assert_eq!(metrics.total_dispatches(), 3);
}

#[test]
fn test_dispatch_metrics_gpu_ratio_empty() {
    let metrics = DispatchMetrics::new();

    let ratio = metrics.gpu_ratio();
    assert!(
        (ratio - 0.0).abs() < 1e-5,
        "Empty metrics should have 0 ratio"
    );
}

#[test]
fn test_dispatch_metrics_gpu_ratio_cpu_only() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();

    let ratio = metrics.gpu_ratio();
    assert!((ratio - 0.0).abs() < 1e-5, "CPU-only should have 0 ratio");
}

#[test]
fn test_dispatch_metrics_gpu_ratio_gpu_only() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_dispatch();
    metrics.record_gpu_dispatch();

    let ratio = metrics.gpu_ratio();
    assert!((ratio - 1.0).abs() < 1e-5, "GPU-only should have 1.0 ratio");
}

#[test]
fn test_dispatch_metrics_gpu_ratio_mixed() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    let ratio = metrics.gpu_ratio();
    assert!(
        (ratio - 0.25).abs() < 0.01,
        "Should be ~0.25, got {}",
        ratio
    );
}

#[test]
fn test_dispatch_metrics_cpu_latency_empty() {
    let metrics = DispatchMetrics::new();

    assert_eq!(metrics.cpu_latency_count(), 0);
    let mean = metrics.cpu_latency_mean_us();
    assert!((mean - 0.0).abs() < 0.001, "Empty mean should be 0");
}

#[test]
fn test_dispatch_metrics_cpu_latency_recording() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    assert_eq!(metrics.cpu_latency_count(), 3);

    let mean = metrics.cpu_latency_mean_us();
    assert!(
        (mean - 200.0).abs() < 1.0,
        "Mean should be ~200us, got {}",
        mean
    );
}

#[test]
fn test_dispatch_metrics_gpu_latency_empty() {
    let metrics = DispatchMetrics::new();

    assert_eq!(metrics.gpu_latency_count(), 0);
    let mean = metrics.gpu_latency_mean_us();
    assert!((mean - 0.0).abs() < 0.001, "Empty GPU mean should be 0");
}

#[test]
fn test_dispatch_metrics_gpu_latency_recording() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(1000));
    metrics.record_gpu_latency(Duration::from_micros(2000));

    assert_eq!(metrics.gpu_latency_count(), 2);

    let mean = metrics.gpu_latency_mean_us();
    assert!(
        (mean - 1500.0).abs() < 1.0,
        "Mean should be ~1500us, got {}",
        mean
    );
}

#[test]
fn test_dispatch_metrics_gpu_latency_sum() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(200));
    metrics.record_gpu_latency(Duration::from_micros(300));

    let sum = metrics.gpu_latency_sum_us();
    assert_eq!(sum, 600, "Sum should be 600us");
}

#[test]
fn test_dispatch_metrics_cpu_latency_sum() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(150));

    let sum = metrics.cpu_latency_sum_us();
    assert_eq!(sum, 200, "Sum should be 200us");
}

#[test]
fn test_dispatch_metrics_gpu_latency_min_max_empty() {
    let metrics = DispatchMetrics::new();

    let min = metrics.gpu_latency_min_us();
    assert_eq!(min, 0, "Empty min should be 0");

    let max = metrics.gpu_latency_max_us();
    assert_eq!(max, 0, "Empty max should be 0");
}

#[test]
fn test_dispatch_metrics_gpu_latency_min_max() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(500));
    metrics.record_gpu_latency(Duration::from_micros(200));

    let min = metrics.gpu_latency_min_us();
    let max = metrics.gpu_latency_max_us();

    assert_eq!(min, 100, "Min should be 100us");
    assert_eq!(max, 500, "Max should be 500us");
}

#[test]
fn test_dispatch_metrics_cpu_latency_min_max() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(300));
    metrics.record_cpu_latency(Duration::from_micros(150));

    let min = metrics.cpu_latency_min_us();
    let max = metrics.cpu_latency_max_us();

    assert_eq!(min, 50, "Min should be 50us");
    assert_eq!(max, 300, "Max should be 300us");
}

#[test]
fn test_dispatch_metrics_gpu_latency_variance_empty() {
    let metrics = DispatchMetrics::new();

    let variance = metrics.gpu_latency_variance_us();
    assert!((variance - 0.0).abs() < 0.001, "Empty variance should be 0");
}

#[test]
fn test_dispatch_metrics_gpu_latency_variance_one_sample() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));

    let variance = metrics.gpu_latency_variance_us();
    assert!(
        (variance - 0.0).abs() < 0.001,
        "Single sample variance should be 0"
    );
}

#[test]
fn test_dispatch_metrics_gpu_latency_variance_same_values() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));

    let variance = metrics.gpu_latency_variance_us();
    assert!(
        variance < 1.0,
        "Same values should have ~0 variance, got {}",
        variance
    );
}

#[test]
fn test_dispatch_metrics_cpu_latency_variance() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(200));

    let variance = metrics.cpu_latency_variance_us();
    assert!(
        variance < 1.0,
        "Same values should have ~0 variance, got {}",
        variance
    );
}

#[test]
fn test_dispatch_metrics_gpu_latency_stddev() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));

    let stddev = metrics.gpu_latency_stddev_us();
    assert!(stddev >= 0.0, "Stddev should be non-negative");
}

#[test]
fn test_dispatch_metrics_cpu_latency_stddev() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(150));

    let stddev = metrics.cpu_latency_stddev_us();
    assert!(stddev >= 0.0, "Stddev should be non-negative");
}

#[test]
fn test_dispatch_metrics_gpu_latency_buckets() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(50));
    metrics.record_gpu_latency(Duration::from_micros(200));
    metrics.record_gpu_latency(Duration::from_micros(750));
    metrics.record_gpu_latency(Duration::from_micros(2000));
    metrics.record_gpu_latency(Duration::from_micros(10000));

    let buckets = metrics.gpu_latency_buckets();

    assert_eq!(buckets.len(), 5, "Should have 5 buckets");
    let total: usize = buckets.iter().sum();
    assert_eq!(total, 5, "Total across buckets should be 5");
}

#[test]
fn test_dispatch_metrics_cpu_latency_buckets() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(250));
    metrics.record_cpu_latency(Duration::from_micros(800));

    let buckets = metrics.cpu_latency_buckets();

    assert_eq!(buckets.len(), 5, "Should have 5 buckets");
    let total: usize = buckets.iter().sum();
    assert_eq!(total, 3, "Total across buckets should be 3");
}

#[test]
fn test_dispatch_metrics_gpu_latency_percentiles() {
    let metrics = DispatchMetrics::new();

    for i in 0..100 {
        metrics.record_gpu_latency(Duration::from_micros(i * 50));
    }

    let p50 = metrics.gpu_latency_p50_us();
    let p95 = metrics.gpu_latency_p95_us();
    let p99 = metrics.gpu_latency_p99_us();

    assert!(p50 <= p95, "p50 ({}) should be <= p95 ({})", p50, p95);
    assert!(p95 <= p99, "p95 ({}) should be <= p99 ({})", p95, p99);
}

#[test]
fn test_dispatch_metrics_cpu_latency_percentiles() {
    let metrics = DispatchMetrics::new();

    for i in 0..50 {
        metrics.record_cpu_latency(Duration::from_micros(i * 10));
    }

    let p50 = metrics.cpu_latency_p50_us();
    let p95 = metrics.cpu_latency_p95_us();
    let p99 = metrics.cpu_latency_p99_us();

    assert!(p50 <= p95, "p50 should be <= p95");
    assert!(p95 <= p99, "p95 should be <= p99");
}

#[test]
fn test_dispatch_metrics_percentiles_empty() {
    let metrics = DispatchMetrics::new();

    let p50 = metrics.gpu_latency_p50_us();
    let p95 = metrics.gpu_latency_p95_us();
    let p99 = metrics.gpu_latency_p99_us();

    assert!((p50 - 0.0).abs() < 0.001, "Empty p50 should be 0");
    assert!((p95 - 0.0).abs() < 0.001, "Empty p95 should be 0");
    assert!((p99 - 0.0).abs() < 0.001, "Empty p99 should be 0");
}

#[test]
fn test_dispatch_metrics_gpu_latency_cv() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(200));
    metrics.record_gpu_latency(Duration::from_micros(150));

    let cv = metrics.gpu_latency_cv();
    assert!(cv >= 0.0, "CV should be non-negative");
}

#[test]
fn test_dispatch_metrics_cpu_latency_cv() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));

    let cv = metrics.cpu_latency_cv();
    assert!(cv >= 0.0, "CV should be non-negative");
}

#[test]
fn test_dispatch_metrics_cv_empty() {
    let metrics = DispatchMetrics::new();

    let cpu_cv = metrics.cpu_latency_cv();
    let gpu_cv = metrics.gpu_latency_cv();

    assert!((cpu_cv - 0.0).abs() < 0.001, "Empty CV should be 0");
    assert!((gpu_cv - 0.0).abs() < 0.001, "Empty CV should be 0");
}

#[test]
fn test_dispatch_metrics_cpu_gpu_speedup() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(1000));
    metrics.record_cpu_latency(Duration::from_micros(1000));

    metrics.record_gpu_latency(Duration::from_micros(500));
    metrics.record_gpu_latency(Duration::from_micros(500));

    let speedup = metrics.cpu_gpu_speedup();
    assert!(
        (speedup - 2.0).abs() < 0.1,
        "Speedup should be ~2.0, got {}",
        speedup
    );
}

#[test]
fn test_dispatch_metrics_cpu_gpu_speedup_no_gpu() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(1000));

    let speedup = metrics.cpu_gpu_speedup();
    assert!(
        (speedup - 0.0).abs() < 0.001,
        "No GPU should return 0 speedup"
    );
}

#[test]
fn test_dispatch_metrics_bucket_boundaries() {
    let metrics = DispatchMetrics::new();

    let boundaries = metrics.bucket_boundaries_us();

    assert_eq!(boundaries.len(), 5, "Should have 5 bucket boundary strings");
    assert!(
        boundaries[0].contains("0-"),
        "First bucket should start with 0"
    );
    assert!(
        boundaries[4].ends_with('+'),
        "Last bucket should end with +"
    );
}

#[test]
fn test_dispatch_metrics_bucket_boundaries_const() {
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES.len(), 4);
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES[0], 100);
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES[1], 500);
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES[2], 1000);
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES[3], 5000);
}

#[test]
fn test_dispatch_metrics_elapsed_seconds() {
    let metrics = DispatchMetrics::new();

    std::thread::sleep(Duration::from_millis(10));

    let elapsed = metrics.elapsed_seconds();
    assert!(
        elapsed >= 0.01,
        "Elapsed should be >= 0.01s, got {}",
        elapsed
    );
}

#[test]
fn test_dispatch_metrics_start_time() {
    let metrics = DispatchMetrics::new();

    let start_time = metrics.start_time_ms();

    let jan_2024_ms = 1704067200000_u64;
    assert!(start_time > jan_2024_ms, "Start time should be recent");
}

#[test]
fn test_dispatch_metrics_throughput() {
    let metrics = DispatchMetrics::new();

    for _ in 0..10 {
        metrics.record_cpu_dispatch();
        metrics.record_gpu_dispatch();
    }

    std::thread::sleep(Duration::from_millis(5));

    let throughput = metrics.throughput_rps();
    assert!(
        throughput > 0.0,
        "Throughput should be > 0, got {}",
        throughput
    );
}

#[test]
fn test_dispatch_metrics_reset() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(200));

    assert!(metrics.total_dispatches() > 0);
    assert!(metrics.cpu_latency_count() > 0);

    metrics.reset();

    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
    assert_eq!(metrics.cpu_latency_count(), 0);
    assert_eq!(metrics.gpu_latency_count(), 0);
    assert_eq!(metrics.cpu_latency_max_us(), 0);
    assert_eq!(metrics.gpu_latency_max_us(), 0);
}

#[test]
fn test_dispatch_metrics_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let metrics = Arc::new(DispatchMetrics::new());
    let mut handles = vec![];

    for i in 0..4 {
        let m = Arc::clone(&metrics);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                if i % 2 == 0 {
                    m.record_cpu_dispatch();
                    m.record_cpu_latency(Duration::from_micros(100));
                } else {
                    m.record_gpu_dispatch();
                    m.record_gpu_latency(Duration::from_micros(200));
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(metrics.total_dispatches(), 400);
    assert_eq!(metrics.cpu_dispatches(), 200);
    assert_eq!(metrics.gpu_dispatches(), 200);
}

// ============================================================================
// CudaBackend Tests (requires cuda feature)
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_creation() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(1024, 1024, 4096, 64);

    assert_eq!(backend.m, 1024);
    assert_eq!(backend.n, 1024);
    assert_eq!(backend.k, 4096);
    assert_eq!(backend.head_dim, 64);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_with_num_heads() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(512, 512, 2048, 128).with_num_heads(16);

    assert_eq!(backend.num_heads, 16);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_with_max_seq_len() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(256, 256, 1024, 64).with_max_seq_len(4096);

    assert_eq!(backend.max_seq_len, 4096);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_q4k_blocks_per_row() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 128, 64);

    assert_eq!(backend.q4k_blocks_per_row(), 4);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_q4k_weight_bytes() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 256, 128, 64);

    assert_eq!(backend.q4k_weight_bytes(), 18432);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_kv_cache_bytes_per_layer() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 1024, 64)
        .with_num_heads(8)
        .with_max_seq_len(512);

    assert_eq!(backend.kv_cache_bytes_per_layer(), 2097152);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_kv_cache_total_bytes() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 1024, 64)
        .with_num_heads(8)
        .with_max_seq_len(512);

    let per_layer = backend.kv_cache_bytes_per_layer();
    let total = backend.kv_cache_total_bytes(32);

    assert_eq!(total, per_layer * 32);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_kv_cache_page_tokens() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 1024, 64);

    assert_eq!(backend.kv_cache_page_tokens(), 64);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_kv_cache_pages_needed() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 1024, 64);

    assert_eq!(backend.kv_cache_pages_needed(100), 2);
    assert_eq!(backend.kv_cache_pages_needed(128), 2);
    assert_eq!(backend.kv_cache_pages_needed(129), 3);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_q4k_gemm_launch_config() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(64, 128, 256, 64);

    let (grid, block) = backend.q4k_gemm_launch_config();

    assert_eq!(grid, (4, 2, 1));
    assert_eq!(block, (1024, 1, 1));
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_flash_attention_launch_config() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 1024, 64).with_num_heads(8);

    let (grid, block) = backend.flash_attention_launch_config(256);

    assert_eq!(grid, (4, 8, 1));
    assert_eq!(block, (4096, 1, 1));
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_validate_dimensions_valid() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 200, 128, 64);

    assert!(backend.validate_dimensions());
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_validate_dimensions_invalid_k() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 200, 100, 64);

    assert!(!backend.validate_dimensions());
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_validate_dimensions_invalid_head_dim() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 200, 128, 65);

    assert!(!backend.validate_dimensions());
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_validate_dimensions_zero() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(0, 200, 128, 64);

    assert!(!backend.validate_dimensions());
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_ptx_target() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 128, 64);

    assert_eq!(backend.ptx_target(), "sm_89");
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_flash_attention_smem_bytes() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 1024, 64);

    let smem = backend.flash_attention_smem_bytes();

    assert_eq!(smem, 49152);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_q4k_gemm_kernel_name() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 128, 64);

    assert_eq!(backend.q4k_gemm_kernel_name(), "q4k_gemm_fused");
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_flash_attention_kernel_name() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(100, 100, 128, 64);

    assert_eq!(
        backend.flash_attention_kernel_name(true),
        "flash_attention_causal"
    );
    assert_eq!(
        backend.flash_attention_kernel_name(false),
        "flash_attention"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_q4k_gemm_ptx_generation() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(32, 64, 128, 64);

    let ptx = backend.q4k_gemm_ptx();

    assert!(!ptx.is_empty(), "PTX should not be empty");
    assert!(
        ptx.contains(".version"),
        "PTX should contain .version directive"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_q4k_gemm_ptx_caching() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(32, 64, 128, 64);

    let ptx1 = backend.q4k_gemm_ptx();
    let ptx2 = backend.q4k_gemm_ptx();

    assert_eq!(ptx1, ptx2);
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_flash_attention_ptx_generation() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(32, 64, 128, 64);

    let ptx = backend.flash_attention_ptx(256, 64, true);

    assert!(!ptx.is_empty(), "PTX should not be empty");
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_flash_attention_causal_ptx_caching() {
    use realizar::gguf::CudaBackend;

    let backend = CudaBackend::new(32, 64, 128, 64);

    let ptx1 = backend.flash_attention_causal_ptx();
    let ptx2 = backend.flash_attention_causal_ptx();

    assert_eq!(ptx1, ptx2);
}

// ============================================================================
// GpuBufferPool Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_buffer_pool_creation() {
    use realizar::gguf::GpuBufferPool;

    let pool = GpuBufferPool::new(256, 1024, 512, 8, 4);
    let stats = pool.stats();

    assert_eq!(stats.borrows, 0);
    assert_eq!(stats.returns, 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_buffer_pool_warmup() {
    use realizar::gguf::GpuBufferPool;

    let pool = GpuBufferPool::new(256, 1024, 512, 8, 4);

    pool.warmup();

    let stats = pool.stats();
    assert!(stats.warmed_up);
    assert!(stats.hidden_available >= 4);
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_buffer_pool_borrow_hidden() {
    use realizar::gguf::GpuBufferPool;

    let pool = GpuBufferPool::new(256, 1024, 512, 8, 4);
    pool.warmup();

    let buf = pool.borrow_hidden();

    assert_eq!(buf.len(), 256);

    let stats = pool.stats();
    assert_eq!(stats.borrows, 1);
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_buffer_pool_return_hidden() {
    use realizar::gguf::GpuBufferPool;

    let pool = GpuBufferPool::new(256, 1024, 512, 8, 4);
    pool.warmup();

    let buf = pool.borrow_hidden();
    pool.return_hidden(buf);

    let stats = pool.stats();
    assert_eq!(stats.returns, 1);
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_buffer_pool_stats() {
    use realizar::gguf::GpuBufferPoolStats;

    let stats = GpuBufferPoolStats {
        borrows: 10,
        returns: 8,
        post_warmup_allocs: 0,
        warmed_up: true,
        hidden_available: 4,
        intermediate_available: 4,
        attention_available: 4,
    };

    assert_eq!(stats.borrows, 10);
    assert_eq!(stats.returns, 8);
    assert!(stats.warmed_up);
}

// ============================================================================
// AsyncQueueStats Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_async_queue_stats_creation() {
    use realizar::gguf::AsyncQueueStats;

    let stats = AsyncQueueStats {
        commands_submitted: 100,
        commands_completed: 95,
        pipeline_stalls: 5,
        in_flight: 5,
        gpu_utilization_percent: 85.5,
    };

    assert_eq!(stats.commands_submitted, 100);
    assert_eq!(stats.commands_completed, 95);
    assert_eq!(stats.pipeline_stalls, 5);
    assert_eq!(stats.in_flight, 5);
    assert!((stats.gpu_utilization_percent - 85.5).abs() < 0.01);
}

// ============================================================================
// BatchGenerationStats Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_batch_generation_stats_creation() {
    use realizar::gguf::BatchGenerationStats;

    let stats = BatchGenerationStats {
        gpu_cache_ready: true,
        cache_memory_gb: 2.5,
        num_layers: 32,
        hidden_dim: 2560,
        intermediate_dim: 10240,
        recommended_batch_size: 16,
        max_batch_size: 64,
    };

    assert!(stats.gpu_cache_ready);
    assert!((stats.cache_memory_gb - 2.5).abs() < 0.01);
    assert_eq!(stats.num_layers, 32);
    assert_eq!(stats.hidden_dim, 2560);
    assert_eq!(stats.intermediate_dim, 10240);
    assert_eq!(stats.recommended_batch_size, 16);
    assert_eq!(stats.max_batch_size, 64);
}

// ============================================================================
// DequantizedWeightCache Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_dequantized_weight_cache_creation() {
    use realizar::gguf::DequantizedWeightCache;

    let cache = DequantizedWeightCache::new(512, 2048, 4);

    assert_eq!(cache.cached_count(), 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_dequantized_weight_cache_warmup() {
    use realizar::gguf::DequantizedWeightCache;

    let cache = DequantizedWeightCache::new(256, 1024, 2);

    cache.warmup(|_layer_idx| {
        let up = vec![1.0f32; 256 * 1024];
        let down = vec![2.0f32; 1024 * 256];
        (up, down)
    });

    assert_eq!(cache.cached_count(), 2);
}

#[test]
#[cfg(feature = "gpu")]
fn test_dequantized_weight_cache_get() {
    use realizar::gguf::DequantizedWeightCache;

    let cache = DequantizedWeightCache::new(4, 8, 2);

    cache.warmup(|layer_idx| {
        let up = vec![(layer_idx as f32) + 1.0; 4 * 8];
        let down = vec![(layer_idx as f32) + 10.0; 8 * 4];
        (up, down)
    });

    let weights0 = cache.get(0);
    assert!(weights0.is_some());
    assert!((weights0.unwrap().up[0] - 1.0).abs() < 0.01);

    let weights1 = cache.get(1);
    assert!(weights1.is_some());
    assert!((weights1.unwrap().up[0] - 2.0).abs() < 0.01);
}

#[test]
#[cfg(feature = "gpu")]
fn test_dequantized_weight_cache_get_out_of_bounds() {
    use realizar::gguf::DequantizedWeightCache;

    let cache = DequantizedWeightCache::new(4, 8, 2);

    let weights = cache.get(5);
    assert!(weights.is_none());
}

#[test]
#[cfg(feature = "gpu")]
fn test_dequantized_weight_cache_memory_bytes() {
    use realizar::gguf::DequantizedWeightCache;

    let cache = DequantizedWeightCache::new(256, 1024, 4);

    cache.warmup(|_| {
        let up = vec![1.0f32; 256 * 1024];
        let down = vec![2.0f32; 1024 * 256];
        (up, down)
    });

    let memory = cache.memory_bytes();
    assert!(memory > 0);
}

// ============================================================================
// BatchingConfig Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_batching_config_default() {
    use realizar::gguf::BatchingConfig;

    let config = BatchingConfig::default();

    assert_eq!(config.batch_threshold, 32);
    assert_eq!(config.timeout_ms, 50);
    assert_eq!(config.max_batch_size, 64);
    assert!(config.prefer_throughput);
}

#[test]
#[cfg(feature = "gpu")]
fn test_batching_config_latency_optimized() {
    use realizar::gguf::BatchingConfig;

    let config = BatchingConfig::latency_optimized();

    assert_eq!(config.batch_threshold, 8);
    assert_eq!(config.timeout_ms, 10);
    assert_eq!(config.max_batch_size, 32);
    assert!(!config.prefer_throughput);
}

#[test]
#[cfg(feature = "gpu")]
fn test_batching_config_throughput_optimized() {
    use realizar::gguf::BatchingConfig;

    let config = BatchingConfig::throughput_optimized();

    assert_eq!(config.batch_threshold, 32);
    assert_eq!(config.timeout_ms, 100);
    assert_eq!(config.max_batch_size, 64);
    assert!(config.prefer_throughput);
}

// ============================================================================
// SlotState Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_slot_state_empty() {
    use realizar::gguf::SlotState;

    let state = SlotState::Empty;

    assert!(state.is_empty());
    assert!(!state.is_active());
    assert!(!state.is_completed());
    assert!(state.request_id().is_none());
}

#[test]
#[cfg(feature = "gpu")]
fn test_slot_state_active() {
    use realizar::gguf::SlotState;

    let state = SlotState::Active {
        request_id: 42,
        prompt_tokens: vec![1, 2, 3],
        generated_tokens: vec![4, 5],
        max_tokens: 10,
        temperature: 0.8,
        top_k: 50,
    };

    assert!(!state.is_empty());
    assert!(state.is_active());
    assert!(!state.is_completed());
    assert_eq!(state.request_id(), Some(42));
}

#[test]
#[cfg(feature = "gpu")]
fn test_slot_state_completed() {
    use realizar::gguf::SlotState;

    let state = SlotState::Completed {
        request_id: 99,
        generated_tokens: vec![1, 2, 3, 4, 5],
    };

    assert!(!state.is_empty());
    assert!(!state.is_active());
    assert!(state.is_completed());
    assert_eq!(state.request_id(), Some(99));
}

// ============================================================================
// SpeculativeConfig Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_speculative_config_default() {
    use realizar::gguf::SpeculativeConfig;

    let config = SpeculativeConfig::default();

    assert_eq!(config.speculation_length, 4);
    assert!((config.draft_temperature - 0.0).abs() < 0.001);
    assert!(config.self_speculative);
}

#[test]
#[cfg(feature = "gpu")]
fn test_speculative_config_custom() {
    use realizar::gguf::SpeculativeConfig;

    let config = SpeculativeConfig {
        speculation_length: 8,
        draft_temperature: 0.5,
        self_speculative: false,
    };

    assert_eq!(config.speculation_length, 8);
    assert!((config.draft_temperature - 0.5).abs() < 0.001);
    assert!(!config.self_speculative);
}

// ============================================================================
// VerificationResult Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_verification_result_all_accepted() {
    use realizar::gguf::VerificationResult;

    let result = VerificationResult {
        accepted_count: 4,
        draft_count: 4,
        accepted_tokens: vec![1, 2, 3, 4],
        all_accepted: true,
    };

    assert_eq!(result.accepted_count, 4);
    assert_eq!(result.draft_count, 4);
    assert_eq!(result.accepted_tokens.len(), 4);
    assert!(result.all_accepted);
}

#[test]
#[cfg(feature = "gpu")]
fn test_verification_result_partial() {
    use realizar::gguf::VerificationResult;

    let result = VerificationResult {
        accepted_count: 2,
        draft_count: 4,
        accepted_tokens: vec![1, 2],
        all_accepted: false,
    };

    assert_eq!(result.accepted_count, 2);
    assert_eq!(result.draft_count, 4);
    assert!(!result.all_accepted);
}

// ============================================================================
// SpeculativeDecoder Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_speculative_decoder_new() {
    use realizar::gguf::SpeculativeDecoder;

    let decoder = SpeculativeDecoder::new();

    assert_eq!(decoder.config.speculation_length, 4);
    assert!((decoder.acceptance_rate() - 0.0).abs() < 0.001);
}

#[test]
#[cfg(feature = "gpu")]
fn test_speculative_decoder_with_config() {
    use realizar::gguf::{SpeculativeConfig, SpeculativeDecoder};

    let config = SpeculativeConfig {
        speculation_length: 8,
        draft_temperature: 0.1,
        self_speculative: true,
    };

    let decoder = SpeculativeDecoder::with_config(config);

    assert_eq!(decoder.config.speculation_length, 8);
}

// ============================================================================
// PrefixCache Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_creation() {
    use realizar::gguf::PrefixCache;

    let cache = PrefixCache::new(1024);

    // Check stats to verify empty state
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_insert_and_contains() {
    use realizar::gguf::PrefixCache;

    let cache = PrefixCache::new(16);

    let tokens = vec![1u32, 2, 3, 4, 5];
    let k_cache = vec![vec![1.0f32; 64]; 4];
    let v_cache = vec![vec![2.0f32; 64]; 4];

    cache.insert(tokens.clone(), k_cache, v_cache);

    assert!(cache.contains(&tokens));
    assert!(!cache.contains(&[9, 8, 7]));
}

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_lookup() {
    use realizar::gguf::PrefixCache;

    let cache = PrefixCache::new(16);

    let tokens = vec![1u32, 2, 3];
    let k_cache = vec![vec![1.0f32; 32]; 2];
    let v_cache = vec![vec![2.0f32; 32]; 2];

    cache.insert(tokens.clone(), k_cache, v_cache);

    let result = cache.lookup(&tokens);
    assert!(result.is_some());

    let (k, v) = result.unwrap();
    assert_eq!(k.len(), 2);
    assert_eq!(v.len(), 2);
}

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_miss() {
    use realizar::gguf::PrefixCache;

    let cache = PrefixCache::new(16);

    let result = cache.lookup(&[1, 2, 3]);
    assert!(result.is_none());

    let stats = cache.stats();
    assert_eq!(stats.misses, 1);
}

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_clear() {
    use realizar::gguf::PrefixCache;

    let cache = PrefixCache::new(16);

    cache.insert(vec![1, 2], vec![vec![1.0]; 1], vec![vec![2.0]; 1]);
    cache.insert(vec![3, 4], vec![vec![3.0]; 1], vec![vec![4.0]; 1]);

    let stats = cache.stats();
    assert_eq!(stats.entries, 2);

    cache.clear();

    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_stats() {
    use realizar::gguf::PrefixCacheStats;

    let stats = PrefixCacheStats {
        hits: 100,
        misses: 20,
        evictions: 5,
        entries: 50,
        hit_rate: 0.833,
    };

    assert_eq!(stats.hits, 100);
    assert_eq!(stats.misses, 20);
    assert_eq!(stats.evictions, 5);
    assert_eq!(stats.entries, 50);
    assert!((stats.hit_rate - 0.833).abs() < 0.01);
}

#[test]
#[cfg(feature = "gpu")]
fn test_prefix_cache_memory_usage() {
    use realizar::gguf::PrefixCache;

    let cache = PrefixCache::new(16);

    // Insert entry with known sizes
    let tokens = vec![1u32, 2, 3, 4]; // 4 * 4 = 16 bytes
    let k_cache = vec![vec![1.0f32; 10]; 2]; // 2 * 10 * 4 = 80 bytes
    let v_cache = vec![vec![2.0f32; 10]; 2]; // 2 * 10 * 4 = 80 bytes

    cache.insert(tokens, k_cache, v_cache);

    let memory = cache.memory_usage_bytes();
    // Should be at least 16 + 80 + 80 = 176 bytes
    assert!(memory >= 176);
}

// ============================================================================
// ChunkedPrefill Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_config_default() {
    use realizar::gguf::ChunkedPrefillConfig;

    let config = ChunkedPrefillConfig::default();

    assert_eq!(config.chunk_size, 512);
    assert_eq!(config.max_context, 8192);
    assert!(config.stream_chunks);
}

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_config_custom() {
    use realizar::gguf::ChunkedPrefillConfig;

    let config = ChunkedPrefillConfig::with_chunk_size(256);

    assert_eq!(config.chunk_size, 256);
}

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_creation() {
    use realizar::gguf::{ChunkedPrefill, ChunkedPrefillConfig};

    let prompt = vec![1u32; 1000];
    let config = ChunkedPrefillConfig::with_chunk_size(256);
    let prefill = ChunkedPrefill::new(&prompt, config);

    assert_eq!(prefill.total_tokens(), 1000);
    assert_eq!(prefill.total_chunks(), 4);
}

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_iteration() {
    use realizar::gguf::{ChunkedPrefill, ChunkedPrefillConfig};

    let prompt = vec![1u32; 500];
    let config = ChunkedPrefillConfig::with_chunk_size(200);
    let mut prefill = ChunkedPrefill::new(&prompt, config);

    let mut chunk_count = 0;
    while let Some(chunk) = prefill.next_chunk() {
        assert!(chunk.len() <= 200);
        prefill.complete_chunk(10.0);
        chunk_count += 1;
    }

    assert_eq!(chunk_count, 3);
}

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_progress() {
    use realizar::gguf::{ChunkedPrefill, ChunkedPrefillConfig};

    let prompt = vec![1u32; 300];
    let config = ChunkedPrefillConfig::with_chunk_size(100);
    let mut prefill = ChunkedPrefill::new(&prompt, config);

    let _ = prefill.next_chunk();
    prefill.complete_chunk(5.0);

    let progress = prefill.progress();
    assert_eq!(progress.tokens_processed, 100);
    assert_eq!(progress.total_tokens, 300);
    assert_eq!(progress.chunk_idx, 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_stats() {
    use realizar::gguf::{ChunkedPrefill, ChunkedPrefillConfig};

    let prompt = vec![1u32; 200];
    let config = ChunkedPrefillConfig::with_chunk_size(100);
    let mut prefill = ChunkedPrefill::new(&prompt, config);

    while prefill.next_chunk().is_some() {
        prefill.complete_chunk(10.0);
    }

    let stats = prefill.stats();
    assert_eq!(stats.total_chunks, 2);
    assert_eq!(stats.chunk_size, 100);
    assert_eq!(stats.total_tokens, 200);
    assert!((stats.total_time_ms - 20.0).abs() < 0.1);
}

#[test]
#[cfg(feature = "gpu")]
fn test_chunked_prefill_stats_struct() {
    use realizar::gguf::ChunkedPrefillStats;

    let stats = ChunkedPrefillStats {
        total_chunks: 4,
        chunk_size: 256,
        total_tokens: 1000,
        total_time_ms: 40.0,
        avg_chunk_time_ms: 10.0,
        ttft_ms: 10.0,
        tokens_per_second: 25000.0,
    };

    assert_eq!(stats.total_chunks, 4);
    assert_eq!(stats.chunk_size, 256);
    assert_eq!(stats.total_tokens, 1000);
}

// ============================================================================
// MultiRequestScheduler Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_multi_request_scheduler_creation() {
    use realizar::gguf::{MultiRequestScheduler, SchedulingPolicy};

    let scheduler = MultiRequestScheduler::new(8, 16, SchedulingPolicy::Fcfs);

    let stats = scheduler.stats();
    assert_eq!(stats.requests_submitted, 0);
    assert_eq!(stats.pending_requests, 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_multi_request_scheduler_submit() {
    use realizar::gguf::{MultiRequestScheduler, SchedulingPolicy};

    let scheduler = MultiRequestScheduler::new(8, 16, SchedulingPolicy::Fcfs);

    let id = scheduler.submit(vec![1, 2, 3], 10);

    assert_eq!(id, 0);

    let stats = scheduler.stats();
    assert_eq!(stats.requests_submitted, 1);
    assert_eq!(stats.pending_requests, 1);
}

#[test]
#[cfg(feature = "gpu")]
fn test_multi_request_scheduler_step() {
    use realizar::gguf::{MultiRequestScheduler, SchedulingPolicy};

    let scheduler = MultiRequestScheduler::new(8, 16, SchedulingPolicy::Fcfs);

    scheduler.step();
    scheduler.step();

    let stats = scheduler.stats();
    assert_eq!(stats.batch_iterations, 2);
}

#[test]
#[cfg(feature = "gpu")]
fn test_multi_request_stats() {
    use realizar::gguf::MultiRequestStats;

    let stats = MultiRequestStats {
        requests_submitted: 100,
        requests_completed: 90,
        tokens_generated: 1800,
        batch_iterations: 200,
        pending_requests: 5,
        active_requests: 5,
        avg_batch_size: 9.0,
    };

    assert_eq!(stats.requests_submitted, 100);
    assert_eq!(stats.requests_completed, 90);
    assert!((stats.avg_batch_size - 9.0).abs() < 0.01);
}

#[test]
#[cfg(feature = "gpu")]
fn test_scheduling_policy_variants() {
    use realizar::gguf::SchedulingPolicy;

    let policies = [
        SchedulingPolicy::Fcfs,
        SchedulingPolicy::Sjf,
        SchedulingPolicy::RoundRobin,
    ];

    assert_eq!(policies.len(), 3);
}

// ============================================================================
// MultiRequestState Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_multi_request_state_variants() {
    use realizar::gguf::MultiRequestState;

    assert_eq!(MultiRequestState::Pending, MultiRequestState::Pending);
    assert_eq!(MultiRequestState::Prefilling, MultiRequestState::Prefilling);
    assert_eq!(MultiRequestState::Decoding, MultiRequestState::Decoding);
    assert_eq!(MultiRequestState::Completed, MultiRequestState::Completed);
    assert_eq!(MultiRequestState::Preempted, MultiRequestState::Preempted);
}

// ============================================================================
// ChunkProgress Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_chunk_progress() {
    use realizar::gguf::ChunkProgress;

    let progress = ChunkProgress {
        chunk_idx: 2,
        total_chunks: 5,
        tokens_processed: 512,
        total_tokens: 1280,
        chunk_time_ms: 8.5,
        cumulative_time_ms: 25.5,
    };

    assert_eq!(progress.chunk_idx, 2);
    assert_eq!(progress.total_chunks, 5);
    assert_eq!(progress.tokens_processed, 512);
    assert_eq!(progress.total_tokens, 1280);
    assert!((progress.chunk_time_ms - 8.5).abs() < 0.01);
}

// ============================================================================
// PendingRequest Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_pending_request_creation() {
    use realizar::gguf::PendingRequest;

    let request = PendingRequest::new(42, vec![1, 2, 3], 10, 0.8, 50);

    assert_eq!(request.id, 42);
    assert_eq!(request.prompt, vec![1, 2, 3]);
    assert_eq!(request.max_tokens, 10);
    assert!((request.temperature - 0.8).abs() < 0.01);
    assert_eq!(request.top_k, 50);
}

// ============================================================================
// CommandSlotState Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_command_slot_state_variants() {
    use realizar::gguf::CommandSlotState;

    let states = [
        CommandSlotState::Empty,
        CommandSlotState::Preparing,
        CommandSlotState::Submitted,
        CommandSlotState::Complete,
    ];

    assert_eq!(states.len(), 4);
}

// ============================================================================
// DequantizedFFNWeights Tests (requires gpu feature)
// ============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_dequantized_ffn_weights() {
    use realizar::gguf::DequantizedFFNWeights;

    let weights = DequantizedFFNWeights {
        up: vec![1.0f32; 256],
        down: vec![2.0f32; 256],
        up_bias: Some(vec![0.1f32; 32]),
        down_bias: None,
    };

    assert_eq!(weights.up.len(), 256);
    assert_eq!(weights.down.len(), 256);
    assert!(weights.up_bias.is_some());
    assert!(weights.down_bias.is_none());
}
