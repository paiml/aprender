//! Deep coverage tests for realizar/src/cuda.rs
//!
//! Tests CUDA functionality on RTX 4090 with fallback for non-CUDA systems.
//! Requires: cargo test --features cuda --test cuda_deep_coverage

#![cfg(feature = "cuda")]

use realizar::cuda::{
    CudaExecutor, CudaKernels, GpuMemoryPool, KernelType, PoolStats, SizeClass, StagingBufferPool,
    StagingPoolStats, TransferMode, WeightQuantType,
};
use serial_test::serial;

// ============================================================================
// Helper: Skip test if CUDA not available
// ============================================================================

fn cuda_available() -> bool {
    CudaExecutor::is_available()
}

macro_rules! skip_if_no_cuda {
    () => {
        if !cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }
    };
}

// ============================================================================
// KernelType Tests (1-20)
// ============================================================================

#[test]
fn test_kernel_type_gemm_naive_debug() {
    let kt = KernelType::GemmNaive {
        m: 64,
        n: 64,
        k: 64,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("GemmNaive"));
}

#[test]
fn test_kernel_type_gemm_tiled_debug() {
    let kt = KernelType::GemmTiled {
        m: 128,
        n: 128,
        k: 128,
        tile_size: 32,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("GemmTiled"));
}

#[test]
fn test_kernel_type_softmax_debug() {
    let kt = KernelType::Softmax { dim: 4096 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Softmax"));
}

#[test]
fn test_kernel_type_layernorm_debug() {
    let kt = KernelType::LayerNorm {
        hidden_size: 2048,
        epsilon: 1e-5,
        affine: true,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("LayerNorm"));
}

#[test]
fn test_kernel_type_attention_debug() {
    let kt = KernelType::Attention {
        seq_len: 512,
        head_dim: 64,
        causal: true,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Attention"));
}

#[test]
fn test_kernel_type_clone() {
    let kt = KernelType::Softmax { dim: 1024 };
    let cloned = kt;
    match cloned {
        KernelType::Softmax { dim } => assert_eq!(dim, 1024),
        _ => panic!("Clone failed"),
    }
}

// ============================================================================
// CudaKernels Tests (21-40)
// ============================================================================

#[test]
fn test_cuda_kernels_new() {
    let kernels = CudaKernels::new();
    let _ = kernels;
}

#[test]
fn test_cuda_kernels_default() {
    let kernels = CudaKernels::default();
    let _ = kernels;
}

#[test]
fn test_cuda_kernels_generate_ptx_gemm_naive() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmNaive {
        m: 32,
        n: 32,
        k: 32,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_generate_ptx_gemm_tiled() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmTiled {
        m: 64,
        n: 64,
        k: 64,
        tile_size: 16,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_generate_ptx_softmax() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 2048 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_generate_ptx_layernorm() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::LayerNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
        affine: false,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_generate_ptx_attention() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Attention {
        seq_len: 1024,
        head_dim: 64,
        causal: false,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_generate_ptx_quantized() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemm {
        m: 1,
        n: 2560,
        k: 2560,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_kernel_name_gemm_naive() {
    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&KernelType::GemmNaive {
        m: 32,
        n: 32,
        k: 32,
    });
    assert!(!name.is_empty());
}

#[test]
fn test_cuda_kernels_kernel_name_softmax() {
    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&KernelType::Softmax { dim: 4096 });
    assert!(!name.is_empty());
}

#[test]
fn test_cuda_kernels_cuda_likely_available() {
    let available = CudaKernels::cuda_likely_available();
    let _ = available;
}

// ============================================================================
// SizeClass Tests (41-55)
// ============================================================================

#[test]
fn test_size_class_for_size_small() {
    let class = SizeClass::for_size(1024);
    assert!(class.is_some());
}

#[test]
fn test_size_class_for_size_medium() {
    let class = SizeClass::for_size(1024 * 1024);
    assert!(class.is_some());
}

#[test]
fn test_size_class_for_size_large() {
    let class = SizeClass::for_size(100 * 1024 * 1024);
    assert!(class.is_some());
}

#[test]
fn test_size_class_bytes() {
    if let Some(class) = SizeClass::for_size(1024) {
        let bytes = class.bytes();
        assert!(bytes >= 1024);
    }
}

#[test]
fn test_size_class_bytes_roundtrip() {
    for size in [1024, 4096, 65536, 1024 * 1024] {
        if let Some(class) = SizeClass::for_size(size) {
            assert!(class.bytes() >= size);
        }
    }
}

// ============================================================================
// GpuMemoryPool Tests (56-75)
// ============================================================================

#[test]
fn test_gpu_memory_pool_new() {
    let pool = GpuMemoryPool::new();
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
}

#[test]
fn test_gpu_memory_pool_default() {
    let pool = GpuMemoryPool::default();
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
}

#[test]
fn test_gpu_memory_pool_with_max_size() {
    let pool = GpuMemoryPool::with_max_size(1024 * 1024 * 100);
    assert_eq!(pool.max_size(), 1024 * 1024 * 100);
}

#[test]
fn test_gpu_memory_pool_try_get_none() {
    let mut pool = GpuMemoryPool::new();
    let handle = pool.try_get(1024);
    assert!(handle.is_none());
}

#[test]
fn test_gpu_memory_pool_record_allocation() {
    let mut pool = GpuMemoryPool::new();
    pool.record_allocation(4096);
    assert_eq!(pool.stats().total_allocated, 4096);
}

#[test]
fn test_gpu_memory_pool_record_deallocation() {
    let mut pool = GpuMemoryPool::new();
    pool.record_allocation(4096);
    pool.record_deallocation(4096);
    // total_allocated stays the same, but we can check no panic
}

#[test]
fn test_gpu_memory_pool_has_capacity() {
    let pool = GpuMemoryPool::with_max_size(1024 * 1024);
    assert!(pool.has_capacity(1024));
}

#[test]
fn test_gpu_memory_pool_max_size() {
    let pool = GpuMemoryPool::with_max_size(50 * 1024 * 1024);
    assert_eq!(pool.max_size(), 50 * 1024 * 1024);
}

#[test]
fn test_gpu_memory_pool_stats() {
    let mut pool = GpuMemoryPool::new();
    pool.record_allocation(1000);
    pool.record_allocation(2000);
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 3000);
}

#[test]
fn test_gpu_memory_pool_clear() {
    let mut pool = GpuMemoryPool::new();
    pool.record_allocation(5000);
    pool.clear();
}

// ============================================================================
// PoolStats Tests (76-85)
// ============================================================================

#[test]
fn test_pool_stats_debug() {
    let stats = PoolStats {
        total_allocated: 1000,
        peak_usage: 800,
        pool_hits: 10,
        pool_misses: 5,
        hit_rate: 0.67,
        free_buffers: 3,
    };
    let debug = format!("{stats:?}");
    assert!(debug.contains("PoolStats"));
}

#[test]
fn test_pool_stats_clone() {
    let stats = PoolStats {
        total_allocated: 2000,
        peak_usage: 1500,
        pool_hits: 20,
        pool_misses: 10,
        hit_rate: 0.67,
        free_buffers: 5,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_allocated, stats.total_allocated);
}

#[test]
fn test_pool_stats_estimated_savings() {
    let stats = PoolStats {
        total_allocated: 10000,
        peak_usage: 8000,
        pool_hits: 30,
        pool_misses: 70,
        hit_rate: 0.3,
        free_buffers: 5,
    };
    let savings = stats.estimated_savings_bytes();
    assert!(savings > 0);
}

#[test]
fn test_pool_stats_zero_hits() {
    let stats = PoolStats {
        total_allocated: 0,
        peak_usage: 0,
        pool_hits: 0,
        pool_misses: 0,
        hit_rate: 0.0,
        free_buffers: 0,
    };
    assert_eq!(stats.estimated_savings_bytes(), 0);
}

// ============================================================================
// StagingPoolStats Tests (86-95)
// ============================================================================

#[test]
fn test_staging_pool_stats_debug() {
    let stats = StagingPoolStats {
        total_allocated: 1000,
        peak_usage: 800,
        pool_hits: 10,
        pool_misses: 5,
        free_buffers: 3,
        hit_rate: 0.67,
    };
    let debug = format!("{stats:?}");
    assert!(debug.contains("StagingPoolStats"));
}

#[test]
fn test_staging_pool_stats_clone() {
    let stats = StagingPoolStats {
        total_allocated: 2000,
        peak_usage: 1500,
        pool_hits: 20,
        pool_misses: 10,
        free_buffers: 5,
        hit_rate: 0.67,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_allocated, stats.total_allocated);
}

// ============================================================================
// TransferMode Tests (96-105)
// ============================================================================

#[test]
fn test_transfer_mode_debug() {
    let mode = TransferMode::Pageable;
    let debug = format!("{mode:?}");
    assert!(debug.contains("Pageable"));
}

#[test]
fn test_transfer_mode_clone() {
    let mode = TransferMode::Pinned;
    let cloned = mode;
    assert!(matches!(cloned, TransferMode::Pinned));
}

#[test]
fn test_transfer_mode_default() {
    let mode = TransferMode::default();
    assert!(matches!(mode, TransferMode::Pageable));
}

#[test]
fn test_transfer_mode_requires_pinned() {
    assert!(!TransferMode::Pageable.requires_pinned());
    assert!(TransferMode::Pinned.requires_pinned());
    assert!(TransferMode::Async.requires_pinned());
}

#[test]
fn test_transfer_mode_estimated_speedup() {
    let pageable = TransferMode::Pageable.estimated_speedup();
    let pinned = TransferMode::Pinned.estimated_speedup();
    let async_mode = TransferMode::Async.estimated_speedup();

    // All speedups should be positive
    assert!(pageable >= 1.0);
    assert!(pinned >= 1.0);
    assert!(async_mode >= 1.0);
}

// ============================================================================
// WeightQuantType Tests (106-125)
// ============================================================================

#[test]
fn test_weight_quant_type_from_ggml_type_q4_0() {
    let qt = WeightQuantType::from_ggml_type(2);
    assert!(qt.is_some());
    assert!(matches!(qt, Some(WeightQuantType::Q4_0)));
}

#[test]
fn test_weight_quant_type_from_ggml_type_q4_1() {
    let qt = WeightQuantType::from_ggml_type(3);
    assert!(qt.is_some());
    assert!(matches!(qt, Some(WeightQuantType::Q4_1)));
}

#[test]
fn test_weight_quant_type_from_ggml_type_q8_0() {
    let qt = WeightQuantType::from_ggml_type(8);
    assert!(qt.is_some());
    assert!(matches!(qt, Some(WeightQuantType::Q8_0)));
}

#[test]
fn test_weight_quant_type_from_ggml_type_q4k() {
    let qt = WeightQuantType::from_ggml_type(12);
    assert!(qt.is_some());
    assert!(matches!(qt, Some(WeightQuantType::Q4K)));
}

#[test]
fn test_weight_quant_type_from_ggml_type_q5k() {
    let qt = WeightQuantType::from_ggml_type(13);
    assert!(qt.is_some());
    assert!(matches!(qt, Some(WeightQuantType::Q5K)));
}

#[test]
fn test_weight_quant_type_from_ggml_type_q6k() {
    let qt = WeightQuantType::from_ggml_type(14);
    assert!(qt.is_some());
    assert!(matches!(qt, Some(WeightQuantType::Q6K)));
}

#[test]
fn test_weight_quant_type_from_ggml_type_invalid() {
    let qt = WeightQuantType::from_ggml_type(255);
    assert!(qt.is_none());
}

#[test]
fn test_weight_quant_type_bytes_per_superblock() {
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);
}

#[test]
fn test_weight_quant_type_bytes_per_block() {
    assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);
    assert_eq!(WeightQuantType::Q4_1.bytes_per_block(), 20);
}

#[test]
fn test_weight_quant_type_explicit_construction() {
    // PMAT-232: WeightQuantType has no Default — must be explicit
    let qt = WeightQuantType::Q4K;
    assert!(matches!(qt, WeightQuantType::Q4K));
}

#[test]
fn test_weight_quant_type_debug() {
    let qt = WeightQuantType::Q4K;
    let debug = format!("{qt:?}");
    assert!(debug.contains("Q4K"));
}

#[test]
fn test_weight_quant_type_clone() {
    let qt = WeightQuantType::Q6K;
    let cloned = qt;
    assert!(matches!(cloned, WeightQuantType::Q6K));
}

// ============================================================================
// CudaExecutor Tests (126-180) - Requires actual CUDA
// ============================================================================

#[test]
fn test_cuda_executor_is_available() {
    let available = CudaExecutor::is_available();
    println!("CUDA available: {available}");
}

#[test]
fn test_cuda_executor_num_devices() {
    let count = CudaExecutor::num_devices();
    println!("CUDA device count: {count}");
    if cuda_available() {
        assert!(count >= 1);
    }
}

#[test]
#[serial]
fn test_cuda_executor_new() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0);
    assert!(executor.is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_device_name() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let name = executor.device_name().expect("device name");
    println!("Device name: {name}");
    assert!(!name.is_empty());
}

#[test]
#[serial]
fn test_cuda_executor_memory_info() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let (free, total) = executor.memory_info().expect("memory info");
    println!("Memory: {free} free / {total} total");
    assert!(total > 0);
    assert!(free <= total);
}

#[test]
#[serial]
fn test_cuda_executor_context() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let _ctx = executor.context();
}

#[test]
#[serial]
fn test_cuda_executor_synchronize() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let result = executor.synchronize();
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_pool_stats() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let stats = executor.pool_stats();
    let _ = stats;
}

#[test]
#[serial]
fn test_cuda_executor_staging_pool_stats() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let stats = executor.staging_pool_stats();
    let _ = stats;
}

#[test]
#[serial]
fn test_cuda_executor_get_staging_buffer() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    let buf = executor.get_staging_buffer(1024);
    assert_eq!(buf.len(), 1024);
}

#[test]
#[serial]
fn test_cuda_executor_return_staging_buffer() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    let buf = executor.get_staging_buffer(512);
    executor.return_staging_buffer(buf);
}

#[test]
#[serial]
fn test_cuda_executor_clear_pool() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    executor.clear_pool();
}

#[test]
#[serial]
fn test_cuda_executor_profiling() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor.enable_profiling();
    assert!(executor.is_profiling_enabled());

    executor.disable_profiling();
    assert!(!executor.is_profiling_enabled());
}

#[test]
#[serial]
fn test_cuda_executor_profiler() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let _profiler = executor.profiler();
}

#[test]
#[serial]
fn test_cuda_executor_profiler_mut() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    let _profiler = executor.profiler_mut();
}

#[test]
#[serial]
fn test_cuda_executor_reset_profiler() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    executor.reset_profiler();
}

#[test]
#[serial]
fn test_cuda_executor_profiler_summary() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let summary = executor.profiler_summary();
    let _ = summary;
}

#[test]
#[serial]
fn test_cuda_executor_graph_tracking() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor.enable_graph_tracking();
    assert!(executor.is_graph_tracking_enabled());

    executor.disable_graph_tracking();
    assert!(!executor.is_graph_tracking_enabled());
}

#[test]
#[serial]
fn test_cuda_executor_execution_graph() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let _graph = executor.execution_graph();
}

#[test]
#[serial]
fn test_cuda_executor_execution_graph_ascii() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let ascii = executor.execution_graph_ascii();
    let _ = ascii;
}

#[test]
#[serial]
fn test_cuda_executor_tile_profiling() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor.enable_tile_profiling();
    assert!(executor.is_tile_profiling_enabled());

    executor.disable_tile_profiling();
    assert!(!executor.is_tile_profiling_enabled());
}

#[test]
#[serial]
fn test_cuda_executor_tile_summary() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let summary = executor.tile_summary();
    let _ = summary;
}

#[test]
#[serial]
fn test_cuda_executor_tile_stats_json() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    let json = executor.tile_stats_json();
    let _ = json;
}

#[test]
#[serial]
fn test_cuda_executor_reset_tile_stats() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    executor.reset_tile_stats();
}

#[test]
#[serial]
fn test_cuda_executor_clear_execution_graph() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    executor.clear_execution_graph();
}

// ============================================================================
// GEMM Tests (181-200)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_gemm_4x4() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];

    let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
    assert!(result.is_ok());

    for val in &c {
        assert!((*val - 4.0).abs() < 1e-4, "Expected 4.0, got {val}");
    }
}

#[test]
#[serial]
fn test_cuda_executor_gemm_32x32() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let a = vec![1.0f32; 32 * 32];
    let b = vec![1.0f32; 32 * 32];
    let mut c = vec![0.0f32; 32 * 32];

    let result = executor.gemm(&a, &b, &mut c, 32, 32, 32);
    assert!(result.is_ok());

    assert!((c[0] - 32.0).abs() < 1e-3, "Expected 32.0, got {}", c[0]);
}

#[test]
#[serial]
fn test_cuda_executor_gemm_64x64() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 64];

    let result = executor.gemm(&a, &b, &mut c, 64, 64, 64);
    assert!(result.is_ok());

    assert!((c[0] - 64.0).abs() < 1e-2, "Expected 64.0, got {}", c[0]);
}

#[test]
#[serial]
fn test_cuda_executor_gemm_non_square() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let m = 4u32;
    let k = 64u32;
    let n = 128u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok());

    assert!((c[0] - 64.0).abs() < 1e-2, "Expected 64.0, got {}", c[0]);
}

#[test]
#[serial]
fn test_cuda_executor_softmax() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let mut input = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut input);
    assert!(result.is_ok());

    let sum: f32 = input.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Softmax sum should be 1.0, got {sum}"
    );
}

// ============================================================================
// Weight Loading Tests (201-220)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_load_weights() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let weights = vec![0.1f32; 1024];
    let result = executor.load_weights("test_weight", &weights);
    assert!(result.is_ok());

    assert!(executor.has_weights("test_weight"));
    assert_eq!(executor.cached_weight_count(), 1);
}

#[test]
#[serial]
fn test_cuda_executor_has_weights() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    assert!(!executor.has_weights("nonexistent"));

    let weights = vec![0.1f32; 100];
    executor.load_weights("exists", &weights).expect("load");

    assert!(executor.has_weights("exists"));
}

#[test]
#[serial]
fn test_cuda_executor_cached_weight_count() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    assert_eq!(executor.cached_weight_count(), 0);

    executor
        .load_weights("w1", &vec![0.1f32; 100])
        .expect("load");
    assert_eq!(executor.cached_weight_count(), 1);

    executor
        .load_weights("w2", &vec![0.1f32; 100])
        .expect("load");
    assert_eq!(executor.cached_weight_count(), 2);
}

#[test]
#[serial]
fn test_cuda_executor_cached_weight_bytes() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let weights = vec![0.1f32; 1000];
    executor.load_weights("w", &weights).expect("load");

    let bytes = executor.cached_weight_bytes();
    assert!(bytes >= 4000);
}

#[test]
#[serial]
fn test_cuda_executor_clear_weights() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor
        .load_weights("w1", &vec![0.1f32; 100])
        .expect("load");
    executor
        .load_weights("w2", &vec![0.1f32; 100])
        .expect("load");

    executor.clear_weights();

    assert_eq!(executor.cached_weight_count(), 0);
    assert!(!executor.has_weights("w1"));
}

// ============================================================================
// Quantized Weight Tests (221-235)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_load_quantized_weights() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let data = vec![0u8; 144 * 16];
    let result = executor.load_quantized_weights("q4k_weight", &data);
    assert!(result.is_ok());

    assert!(executor.has_quantized_weights("q4k_weight"));
}

#[test]
#[serial]
fn test_cuda_executor_has_quantized_weights() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert!(!executor.has_quantized_weights("nonexistent"));
}

#[test]
#[serial]
fn test_cuda_executor_cached_quantized_weight_count() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert_eq!(executor.cached_quantized_weight_count(), 0);
}

#[test]
#[serial]
fn test_cuda_executor_cached_quantized_weight_bytes() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let data = vec![0u8; 1024];
    executor.load_quantized_weights("qw", &data).expect("load");

    let bytes = executor.cached_quantized_weight_bytes();
    assert!(bytes >= 1024);
}

#[test]
#[serial]
fn test_cuda_executor_clear_quantized_weights() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let data = vec![0u8; 512];
    executor.load_quantized_weights("qw", &data).expect("load");

    executor.clear_quantized_weights();

    assert_eq!(executor.cached_quantized_weight_count(), 0);
}

// ============================================================================
// Property Tests (236-250)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_size_class_bytes_ge_input(size in 1usize..10_000_000) {
            if let Some(class) = SizeClass::for_size(size) {
                prop_assert!(class.bytes() >= size);
            }
        }

        #[test]
        fn prop_transfer_mode_speedup_positive(mode in prop_oneof![
            Just(TransferMode::Pageable),
            Just(TransferMode::Pinned),
            Just(TransferMode::Async),
        ]) {
            prop_assert!(mode.estimated_speedup() >= 1.0);
        }

        #[test]
        fn prop_weight_quant_type_roundtrip(type_id in 0u32..20) {
            if let Some(qt) = WeightQuantType::from_ggml_type(type_id) {
                let debug = format!("{qt:?}");
                prop_assert!(!debug.is_empty());
            }
        }

        #[test]
        fn prop_kernel_type_generates_valid_ptx(
            m in 1u32..128,
            n in 1u32..128,
            k in 1u32..128,
        ) {
            let kernels = CudaKernels::new();
            let ptx = kernels.generate_ptx(&KernelType::GemmNaive { m, n, k });
            prop_assert!(ptx.contains(".version"));
        }

        #[test]
        fn prop_pool_stats_savings_non_negative(
            hits in 0usize..1000,
        ) {
            let stats = PoolStats {
                total_allocated: 10000,
                peak_usage: 8000,
                pool_hits: hits,
                pool_misses: 100,
                hit_rate: 0.5,
                free_buffers: 5,
            };
            // estimated_savings_bytes returns usize, always >= 0
            let _ = stats.estimated_savings_bytes();
        }
    }
}

// Remaining unique tests are in the first section

#[test]
#[serial]
fn test_cuda_executor_rmsnorm_host() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let input = vec![1.0f32; 64];
    let gamma = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = executor.rmsnorm_host(&input, &gamma, &mut output, 1e-5);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_residual_add_host() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let a = vec![1.0f32; 64];
    let b = vec![2.0f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = executor.residual_add_host(&a, &b, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!((*val - 3.0).abs() < 1e-5);
    }
}

// ============================================================================
// Activation Function Tests (321-340)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_silu_host() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let input = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = executor.silu_host(&input, &mut output);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_gelu_host() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let input = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = executor.gelu_host(&input, &mut output);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_elementwise_mul_host() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let a = vec![2.0f32; 64];
    let b = vec![3.0f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = executor.elementwise_mul_host(&a, &b, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!((*val - 6.0).abs() < 1e-5);
    }
}

#[test]
#[serial]
fn test_cuda_executor_fused_swiglu_host() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let gate = vec![1.0f32; 64];
    let up = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = executor.fused_swiglu_host(&gate, &up, &mut output);
    assert!(result.is_ok());
}

// ============================================================================
// KV Cache Tests (341-360)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_has_kv_cache_gpu() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert!(!executor.has_kv_cache_gpu());
}

#[test]
#[serial]
fn test_cuda_executor_init_kv_cache_gpu() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    // init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_len)
    let result = executor.init_kv_cache_gpu(4, 8, 4, 64, 128);
    assert!(result.is_ok());
    assert!(executor.has_kv_cache_gpu());
}

#[test]
#[serial]
fn test_cuda_executor_reset_kv_cache_gpu() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor.init_kv_cache_gpu(4, 8, 4, 64, 128).expect("init");
    executor.reset_kv_cache_gpu();
}

#[test]
#[serial]
fn test_cuda_executor_kv_cache_len() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor.init_kv_cache_gpu(4, 8, 4, 64, 128).expect("init");
    let len = executor.kv_cache_len(0);
    assert_eq!(len, 0);
}

#[test]
#[serial]
fn test_cuda_executor_rollback_kv_cache_gpu() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    executor.init_kv_cache_gpu(4, 8, 4, 64, 128).expect("init");
    executor.rollback_kv_cache_gpu(0);
}

// ============================================================================
// RoPE Configuration Tests (361-370)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_set_rope_theta() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    executor.set_rope_theta(10000.0);
}

#[test]
#[serial]
fn test_cuda_executor_set_rope_type() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");
    executor.set_rope_type(0);
    executor.set_rope_type(1);
}

// ============================================================================
// RMSNorm Weights Tests (371-380)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_preload_rmsnorm_weights() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    // preload_rmsnorm_weights takes (num_layers, attn_norms, ffn_norms)
    let gamma1 = vec![1.0f32; 64];
    let gamma2 = vec![1.0f32; 64];
    let attn_norms: Vec<&[f32]> = vec![&gamma1];
    let ffn_norms: Vec<&[f32]> = vec![&gamma2];
    let result = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result.is_ok());

    assert!(executor.has_rmsnorm_weights(0));
}

#[test]
#[serial]
fn test_cuda_executor_has_rmsnorm_weights() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert!(!executor.has_rmsnorm_weights(0));
}

#[test]
#[serial]
fn test_cuda_executor_preload_output_norm() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let gamma = vec![1.0f32; 64];
    let result = executor.preload_output_norm(&gamma);
    assert!(result.is_ok());

    assert!(executor.has_output_norm());
}

#[test]
#[serial]
fn test_cuda_executor_has_output_norm() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert!(!executor.has_output_norm());
}

#[test]
#[serial]
fn test_cuda_executor_cache_rmsnorm_gamma() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let gamma = vec![1.0f32; 64];
    let result = executor.cache_rmsnorm_gamma("test_norm", &gamma);
    assert!(result.is_ok());
}

// ============================================================================
// QKV Bias Tests (381-390)
// ============================================================================

#[test]
#[serial]
fn test_cuda_executor_has_qkv_bias() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert!(!executor.has_qkv_bias(0));
}

#[test]
#[serial]
fn test_cuda_executor_preload_lm_head_bias_some() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let bias = vec![0.1f32; 1000];
    let result = executor.preload_lm_head_bias(Some(&bias));
    assert!(result.is_ok());

    assert!(executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cuda_executor_preload_lm_head_bias_none() {
    skip_if_no_cuda!();
    let mut executor = CudaExecutor::new(0).expect("executor");

    let result = executor.preload_lm_head_bias(None);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_has_lm_head_bias() {
    skip_if_no_cuda!();
    let executor = CudaExecutor::new(0).expect("executor");
    assert!(!executor.has_lm_head_bias());
}

// ============================================================================
// More KernelType Variants Tests (391-450)
// ============================================================================

#[test]
fn test_kernel_type_gemm_tensor_core() {
    let kt = KernelType::GemmTensorCore {
        m: 64,
        n: 64,
        k: 64,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("GemmTensorCore"));
}

#[test]
fn test_kernel_type_gemv() {
    let kt = KernelType::Gemv { k: 1024, n: 256 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Gemv"));
}

#[test]
fn test_kernel_type_coalesced_gemv() {
    let kt = KernelType::CoalescedGemv { k: 1024, n: 256 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("CoalescedGemv"));
}

#[test]
fn test_kernel_type_quantized_gemm() {
    let kt = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("QuantizedGemm"));
}

#[test]
fn test_kernel_type_quantized_gemm_ggml() {
    let kt = KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("QuantizedGemmGgml"));
}

#[test]
fn test_kernel_type_q5k_quantized_gemm() {
    let kt = KernelType::Q5KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q5KQuantizedGemm"));
}

#[test]
fn test_kernel_type_q6k_quantized_gemm() {
    let kt = KernelType::Q6KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q6KQuantizedGemm"));
}

#[test]
fn test_kernel_type_gemm_optimized() {
    let kt = KernelType::GemmOptimized {
        m: 64,
        n: 64,
        k: 64,
        tile_size: 16,
        reg_block: 4,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("GemmOptimized"));
}

#[test]
fn test_kernel_type_gemm_bias_activation() {
    let kt = KernelType::GemmBiasActivation {
        m: 64,
        n: 64,
        k: 64,
        activation: 1,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("GemmBiasActivation"));
}

#[test]
fn test_kernel_type_bias_activation() {
    let kt = KernelType::BiasActivation {
        n: 1024,
        bias_size: 256,
        activation: 2,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BiasActivation"));
}

#[test]
fn test_kernel_type_gemm_fp16_tensor_core() {
    let kt = KernelType::GemmFp16TensorCore {
        m: 64,
        n: 64,
        k: 64,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("GemmFp16TensorCore"));
}

#[test]
fn test_kernel_type_fused_q4q8_dot() {
    let kt = KernelType::FusedQ4Q8Dot { n: 4096 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("FusedQ4Q8Dot"));
}

#[test]
fn test_kernel_type_q4k_gemv() {
    let kt = KernelType::Q4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q4KGemv"));
}

#[test]
fn test_kernel_type_tiled_q4k_gemv() {
    let kt = KernelType::TiledQ4KGemv {
        k: 4096,
        n: 11008,
        outputs_per_block: 4,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("TiledQ4KGemv"));
}

#[test]
fn test_kernel_type_chunked_tiled_q4k_gemv() {
    let kt = KernelType::ChunkedTiledQ4KGemv {
        k: 11008,
        n: 4096,
        outputs_per_block: 4,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("ChunkedTiledQ4KGemv"));
}

#[test]
fn test_kernel_type_coalesced_q4k_gemv() {
    let kt = KernelType::CoalescedQ4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("CoalescedQ4KGemv"));
}

#[test]
fn test_kernel_type_vectorized_q4k_gemv() {
    let kt = KernelType::VectorizedQ4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("VectorizedQ4KGemv"));
}

#[test]
fn test_kernel_type_dp4a_q4k_gemv() {
    let kt = KernelType::Dp4aQ4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Dp4aQ4KGemv"));
}

#[test]
fn test_kernel_type_dp4a_simd_q4k_gemv() {
    let kt = KernelType::Dp4aSIMDQ4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Dp4aSIMDQ4KGemv"));
}

#[test]
fn test_kernel_type_q8_quantize() {
    let kt = KernelType::Q8Quantize { n: 4096 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q8Quantize"));
}

#[test]
fn test_kernel_type_q4k_q8_dot() {
    let kt = KernelType::Q4KQ8Dot { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q4KQ8Dot"));
}

#[test]
fn test_kernel_type_packed_dp4a_q4k_q8() {
    let kt = KernelType::PackedDp4aQ4KQ8 { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("PackedDp4aQ4KQ8"));
}

#[test]
fn test_kernel_type_true_dp4a_q4k_gemv() {
    let kt = KernelType::TrueDp4aQ4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("TrueDp4aQ4KGemv"));
}

#[test]
fn test_kernel_type_tensor_core_q4k_gemm() {
    let kt = KernelType::TensorCoreQ4KGemm {
        m: 16,
        k: 4096,
        n: 4096,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("TensorCoreQ4KGemm"));
}

#[test]
fn test_kernel_type_batched_q4k_gemv() {
    let kt = KernelType::BatchedQ4KGemv {
        m: 8,
        k: 4096,
        n: 11008,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BatchedQ4KGemv"));
}

#[test]
fn test_kernel_type_multi_warp_batched_q4k_gemv() {
    let kt = KernelType::MultiWarpBatchedQ4KGemv {
        k: 4096,
        n: 11008,
        warps: 4,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("MultiWarpBatchedQ4KGemv"));
}

#[test]
fn test_kernel_type_q5k_gemv() {
    let kt = KernelType::Q5KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q5KGemv"));
}

#[test]
fn test_kernel_type_q6k_gemv() {
    let kt = KernelType::Q6KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q6KGemv"));
}

#[test]
fn test_kernel_type_coalesced_q6k_gemv() {
    let kt = KernelType::CoalescedQ6KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("CoalescedQ6KGemv"));
}

#[test]
fn test_kernel_type_batched_q6k_gemv() {
    let kt = KernelType::BatchedQ6KGemv {
        k: 4096,
        n: 11008,
        m: 8,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BatchedQ6KGemv"));
}

#[test]
fn test_kernel_type_fp16_q4k_gemv() {
    let kt = KernelType::Fp16Q4KGemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Fp16Q4KGemv"));
}

#[test]
fn test_kernel_type_q8_0_gemv() {
    let kt = KernelType::Q8_0Gemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q8_0Gemv"));
}

#[test]
fn test_kernel_type_q5_0_gemv() {
    let kt = KernelType::Q5_0Gemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q5_0Gemv"));
}

#[test]
fn test_kernel_type_q4_0_gemv() {
    let kt = KernelType::Q4_0Gemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q4_0Gemv"));
}

#[test]
fn test_kernel_type_q4_1_gemv() {
    let kt = KernelType::Q4_1Gemv { k: 4096, n: 11008 };
    let debug = format!("{kt:?}");
    assert!(debug.contains("Q4_1Gemv"));
}

#[test]
fn test_kernel_type_incremental_attention() {
    let kt = KernelType::IncrementalAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        indirect: false,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("IncrementalAttention"));
}

#[test]
fn test_kernel_type_incremental_attention_indirect() {
    let kt = KernelType::IncrementalAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        indirect: true,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("indirect: true"));
}

#[test]
fn test_kernel_type_multi_warp_attention() {
    let kt = KernelType::MultiWarpAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        num_warps_per_head: 8,
        indirect: false,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("MultiWarpAttention"));
}

#[test]
fn test_kernel_type_kv_cache_scatter() {
    let kt = KernelType::KvCacheScatter {
        num_kv_heads: 8,
        head_dim: 64,
        max_len: 2048,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("KvCacheScatter"));
}

#[test]
fn test_kernel_type_kv_cache_scatter_indirect() {
    let kt = KernelType::KvCacheScatterIndirect {
        num_kv_heads: 8,
        head_dim: 64,
        max_len: 2048,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("KvCacheScatterIndirect"));
}

#[test]
fn test_kernel_type_rms_norm() {
    let kt = KernelType::RmsNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("RmsNorm"));
}

#[test]
fn test_kernel_type_vectorized_rms_norm() {
    let kt = KernelType::VectorizedRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("VectorizedRmsNorm"));
}

#[test]
fn test_kernel_type_batched_vectorized_rms_norm() {
    let kt = KernelType::BatchedVectorizedRmsNorm {
        hidden_size: 4096,
        batch_size: 8,
        epsilon: 1e-5,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BatchedVectorizedRmsNorm"));
}

#[test]
fn test_kernel_type_precise_rms_norm() {
    let kt = KernelType::PreciseRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("PreciseRmsNorm"));
}

#[test]
fn test_kernel_type_batched_rope() {
    let kt = KernelType::BatchedRope {
        num_heads: 32,
        head_dim: 64,
        batch_size: 8,
        theta: 10000.0,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BatchedRope"));
}

#[test]
fn test_kernel_type_batched_residual_add() {
    let kt = KernelType::BatchedResidualAdd {
        n: 4096,
        batch_size: 8,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BatchedResidualAdd"));
}

#[test]
fn test_kernel_type_batched_swiglu() {
    let kt = KernelType::BatchedSwiglu {
        n: 11008,
        batch_size: 8,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("BatchedSwiglu"));
}

#[test]
fn test_kernel_type_multi_head_attention() {
    let kt = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: true,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("MultiHeadAttention"));
}

#[test]
fn test_kernel_type_attention_tensor_core() {
    let kt = KernelType::AttentionTensorCore {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: true,
    };
    let debug = format!("{kt:?}");
    assert!(debug.contains("AttentionTensorCore"));
}

// ============================================================================
// PinnedHostBuffer Tests (521-540)
// ============================================================================

#[test]
fn test_pinned_host_buffer_new() {
    use realizar::cuda::PinnedHostBuffer;
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(1024);
    assert_eq!(buf.len(), 1024);
    assert!(!buf.is_empty());
}

#[test]
fn test_pinned_host_buffer_as_slice() {
    use realizar::cuda::PinnedHostBuffer;
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
    let slice = buf.as_slice();
    assert_eq!(slice.len(), 100);
}

#[test]
fn test_pinned_host_buffer_as_mut_slice() {
    use realizar::cuda::PinnedHostBuffer;
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
    let slice = buf.as_mut_slice();
    slice[0] = 42.0;
    assert_eq!(buf.as_slice()[0], 42.0);
}

#[test]
fn test_pinned_host_buffer_size_bytes() {
    use realizar::cuda::PinnedHostBuffer;
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
    assert_eq!(buf.size_bytes(), 400);
}

#[test]
fn test_pinned_host_buffer_copy_from_slice() {
    use realizar::cuda::PinnedHostBuffer;
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(4);
    buf.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(buf.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_pinned_host_buffer_is_pinned() {
    use realizar::cuda::PinnedHostBuffer;
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
    let _ = buf.is_pinned();
}

// ============================================================================
// StagingBufferPool Additional Tests (541-550)
// ============================================================================

#[test]
fn test_staging_buffer_pool_with_max_size() {
    let pool = StagingBufferPool::with_max_size(1024 * 1024);
    let _ = pool;
}

#[test]
fn test_staging_buffer_pool_get_put() {
    let mut pool = StagingBufferPool::new();
    let buf = pool.get(256);
    assert!(buf.len() >= 256);
    pool.put(buf);
}

#[test]
fn test_staging_buffer_pool_clear() {
    let mut pool = StagingBufferPool::new();
    let buf = pool.get(256);
    pool.put(buf);
    pool.clear();
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 0);
}

// ============================================================================
// Additional Property Tests (561-570)
// ============================================================================

#[cfg(test)]
mod additional_proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_weight_quant_type_matches_size_consistency(
            n_rows in 256u32..1024,
            n_cols in 256u32..1024,
        ) {
            let n_rows = (n_rows / 256) * 256;
            let n_cols = (n_cols / 256) * 256;

            let q4k_size = (n_rows as usize * n_cols as usize / 256) * 144;
            let qt = WeightQuantType::Q4K;

            if n_rows > 0 && n_cols > 0 {
                prop_assert!(qt.matches_size(q4k_size, n_rows as usize, n_cols as usize));
            }
        }

        #[test]
        fn prop_kernel_type_clone_equal(m in 1u32..128, n in 1u32..128, k in 1u32..128) {
            let kt = KernelType::GemmNaive { m, n, k };
            let cloned = kt.clone();
            match (&kt, &cloned) {
                (KernelType::GemmNaive { m: m1, n: n1, k: k1 },
                 KernelType::GemmNaive { m: m2, n: n2, k: k2 }) => {
                    prop_assert_eq!(m1, m2);
                    prop_assert_eq!(n1, n2);
                    prop_assert_eq!(k1, k2);
                }
                _ => prop_assert!(false, "Clone should produce same variant"),
            }
        }
    }
}
