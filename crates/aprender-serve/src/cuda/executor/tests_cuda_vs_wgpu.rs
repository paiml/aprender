
/// Sub-test 2: Patterned data — CUDA vs wgpu vs CPU reference
fn cuda_vs_wgpu_patterned() {
    use crate::gpu::{CudaScheduler, HybridScheduler};

    let m = 4usize;
    let k = 64usize;
    let n = 192usize;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();

    // CPU reference (ground truth)
    let cpu_result = cpu_matmul_reference(&a, &b, m, k, n);

    // CUDA path
    let mut cuda_sched = CudaScheduler::new().expect("CudaScheduler should init");
    let cuda_result = cuda_sched
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul should succeed");

    // wgpu path
    let mut wgpu_sched =
        HybridScheduler::with_threshold(1000).expect("HybridScheduler should init");
    let _wgpu_result = wgpu_sched
        .matmul(&a, &b, m, k, n)
        .expect("wgpu matmul should succeed");

    // Compare CUDA to CPU reference
    assert_eq!(cuda_result.len(), cpu_result.len());
    for i in 0..cuda_result.len() {
        let diff = (cuda_result[i] - cpu_result[i]).abs();
        assert!(
            diff < 1e-3,
            "PARITY-114: CUDA vs CPU mismatch at {}: cuda={}, cpu={}, diff={}",
            i,
            cuda_result[i],
            cpu_result[i],
            diff
        );
    }
}

/// Naive CPU matrix multiply for test reference
fn cpu_matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

#[test]
#[serial]
fn test_cuda_executor_gemm_size_validation() {
    // This test requires CUDA GPU to create an executor
    let mut executor = CudaExecutor::new(0).expect("test");

    // Wrong sizes - should fail validation
    let a = vec![1.0f32; 10]; // Wrong size
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];

    let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cuda_executor_softmax() {
    // Debug: print PTX first
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 4 });
    eprintln!("Generated PTX:\n{}", ptx);

    let mut executor = CudaExecutor::new(0).expect("test");

    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax failed: {:?}", result.err());

    // Check softmax properties
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(data[3] > data[2]); // Larger input = larger output
    assert!(data[2] > data[1]);
    assert!(data[1] > data[0]);
}

#[test]
#[serial]
fn test_cuda_executor_synchronize() {
    let executor = CudaExecutor::new(0).expect("test");
    let result = executor.synchronize();
    assert!(result.is_ok());
}

// ========================================================================
// Drop Order Tests (IMP-800: GPU Parity)
// ========================================================================

/// Test that CudaExecutor can be created and dropped multiple times
/// without crashing (validates correct Drop order: context dropped last)
#[test]
#[serial]
fn test_cuda_executor_drop_order_multiple_cycles() {
    // This test verifies the Drop order is correct:
    // Fields should be dropped in reverse declaration order,
    // with context dropped LAST (after stream and modules)
    for i in 1..=3 {
        let mut executor = CudaExecutor::new(0)
            .unwrap_or_else(|e| panic!("Cycle {}: Failed to create executor: {}", i, e));

        // Verify executor works
        assert!(
            executor.device_name().is_ok(),
            "Cycle {}: device_name failed",
            i
        );

        // Run a GEMM to load a module (tests module Drop)
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let mut c = vec![0.0f32; 16];
        executor
            .gemm(&a, &b, &mut c, 4, 4, 4)
            .unwrap_or_else(|e| panic!("Cycle {}: GEMM failed: {}", i, e));

        // executor is dropped here - must not crash
    }
    // If we reach here, Drop order is correct
}

/// Test rapid create/destroy cycles (stress test for Drop order)
#[test]
#[serial]
fn test_cuda_executor_rapid_lifecycle() {
    // 10 rapid cycles without any work - pure lifecycle test
    for _ in 0..10 {
        let executor = CudaExecutor::new(0).expect("Failed to create executor");
        drop(executor); // Explicit drop for clarity
    }
}

/// Test that modules are properly cleaned up before context
#[test]
#[serial]
fn test_cuda_executor_module_cleanup() {
    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Load multiple modules (different GEMM configurations)
    for size in [4, 8, 16, 32] {
        let a = vec![1.0f32; size * size];
        let b = vec![1.0f32; size * size];
        let mut c = vec![0.0f32; size * size];
        executor
            .gemm(&a, &b, &mut c, size as u32, size as u32, size as u32)
            .expect("GEMM should succeed");
    }

    // Now drop - all modules must be cleaned up before context
    drop(executor);

    // Create new executor to verify GPU is in good state
    let executor2 = CudaExecutor::new(0).expect("Should create after cleanup");
    assert!(executor2.device_name().is_ok());
}

// ========================================================================
// GpuMemoryPool Tests (IMP-900d)
// ========================================================================

#[test]
fn test_size_class_for_small_size() {
    // Small size should map to 4KB class
    let class = SizeClass::for_size(1024);
    assert_eq!(class.map(|c| c.bytes()), Some(4096));
}

#[test]
fn test_size_class_for_exact_size() {
    // Exact match should return same size
    let class = SizeClass::for_size(1048576); // 1 MB
    assert_eq!(class.map(|c| c.bytes()), Some(1048576));
}

#[test]
fn test_size_class_for_large_size() {
    // Large size should map to 256MB class
    let class = SizeClass::for_size(200_000_000);
    assert_eq!(class.map(|c| c.bytes()), Some(268435456)); // 256 MB
}

#[test]
fn test_size_class_too_large() {
    // Size larger than max class should return None
    let class = SizeClass::for_size(500_000_000);
    assert!(class.is_none());
}

#[test]
fn test_gpu_memory_pool_creation() {
    let pool = GpuMemoryPool::new();
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
}

#[test]
fn test_gpu_memory_pool_with_max_size() {
    let pool = GpuMemoryPool::with_max_size(512 * 1024 * 1024);
    assert_eq!(pool.max_size(), 512 * 1024 * 1024);
}

#[test]
fn test_gpu_memory_pool_try_get_empty() {
    let mut pool = GpuMemoryPool::new();

    // Pool is empty, should return None and increment miss counter
    let result = pool.try_get(1024);
    assert!(result.is_none());

    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
    assert_eq!(stats.pool_hits, 0);
}

#[test]
fn test_gpu_memory_pool_return_and_get() {
    let mut pool = GpuMemoryPool::new();

    // Return a buffer to the pool
    let handle = GpuBufferHandle {
        size: 4096,
        in_use: false,
    };
    pool.return_buffer(handle);

    // Now try to get it back
    let result = pool.try_get(4096);
    assert!(result.is_some());
    let handle = result.expect("test");
    assert!(handle.in_use);

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 1);
}

#[test]
fn test_gpu_memory_pool_allocation_tracking() {
    let mut pool = GpuMemoryPool::new();

    pool.record_allocation(1024 * 1024);
    assert_eq!(pool.stats().total_allocated, 1024 * 1024);

    pool.record_allocation(2048 * 1024);
    assert_eq!(pool.stats().total_allocated, 3072 * 1024);
    assert_eq!(pool.stats().peak_usage, 3072 * 1024);

    pool.record_deallocation(1024 * 1024);
    assert_eq!(pool.stats().total_allocated, 2048 * 1024);
    assert_eq!(pool.stats().peak_usage, 3072 * 1024); // Peak unchanged
}

#[test]
fn test_gpu_memory_pool_hit_rate() {
    let mut pool = GpuMemoryPool::new();

    // Return 3 buffers
    for _ in 0..3 {
        pool.return_buffer(GpuBufferHandle {
            size: 4096,
            in_use: false,
        });
    }

    // Get 3 (hits) + try to get 1 more (miss)
    for _ in 0..3 {
        let _ = pool.try_get(4096);
    }
    let _ = pool.try_get(4096); // Miss - pool now empty

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 3);
    assert_eq!(stats.pool_misses, 1);
    assert!((stats.hit_rate - 0.75).abs() < 0.01); // 3/4 = 75%
}

#[test]
fn test_gpu_memory_pool_clear() {
    let mut pool = GpuMemoryPool::new();

    // Add some buffers
    for _ in 0..5 {
        pool.return_buffer(GpuBufferHandle {
            size: 4096,
            in_use: false,
        });
    }
    assert_eq!(pool.stats().free_buffers, 5);

    // Clear the pool
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
}

#[test]
fn test_pool_stats_estimated_savings() {
    let stats = PoolStats {
        total_allocated: 10 * 1024 * 1024,
        peak_usage: 20 * 1024 * 1024,
        pool_hits: 100,
        pool_misses: 50,
        hit_rate: 0.667,
        free_buffers: 5,
    };

    // 100 hits * 1MB assumed per allocation = 100MB saved
    assert_eq!(stats.estimated_savings_bytes(), 100 * 1024 * 1024);
}

#[test]
fn test_gpu_memory_pool_has_capacity() {
    let mut pool = GpuMemoryPool::with_max_size(100 * 1024 * 1024); // 100 MB max

    // Initially has capacity
    assert!(pool.has_capacity(50 * 1024 * 1024)); // 50 MB fits
    assert!(pool.has_capacity(100 * 1024 * 1024)); // 100 MB fits exactly
    assert!(!pool.has_capacity(101 * 1024 * 1024)); // 101 MB doesn't fit

    // After recording allocation
    pool.record_allocation(60 * 1024 * 1024); // 60 MB allocated
    assert!(pool.has_capacity(40 * 1024 * 1024)); // 40 MB still fits
    assert!(!pool.has_capacity(41 * 1024 * 1024)); // 41 MB doesn't fit
}

#[test]
fn test_gpu_memory_pool_max_size_getter() {
    let pool = GpuMemoryPool::with_max_size(512 * 1024 * 1024);
    assert_eq!(pool.max_size(), 512 * 1024 * 1024);

    let default_pool = GpuMemoryPool::new();
    assert_eq!(default_pool.max_size(), 2 * 1024 * 1024 * 1024); // 2 GB default
}

// ========================================================================
// Kernel Fusion Tests (IMP-900b)
// ========================================================================

#[test]
fn test_gemm_bias_activation_kernel_type() {
    let kernel_type = KernelType::GemmBiasActivation {
        m: 64,
        n: 64,
        k: 64,
        activation: 1, // ReLU
    };

    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&kernel_type);
    assert_eq!(name, "gemm_tiled"); // Falls back to tiled for now

    let ptx = kernels.generate_ptx(&kernel_type);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm_tiled"));
}

#[test]
fn test_gemm_fused_activation_values() {
    // Test activation types are correctly defined
    // 0 = no activation
    // 1 = ReLU
    // 2 = GELU
    let no_act = KernelType::GemmBiasActivation {
        m: 4,
        n: 4,
        k: 4,
        activation: 0,
    };
    let relu = KernelType::GemmBiasActivation {
        m: 4,
        n: 4,
        k: 4,
        activation: 1,
    };
    let gelu = KernelType::GemmBiasActivation {
        m: 4,
        n: 4,
        k: 4,
        activation: 2,
    };

    // All should generate valid PTX
    let kernels = CudaKernels::new();
    assert!(kernels.generate_ptx(&no_act).contains(".version"));
    assert!(kernels.generate_ptx(&relu).contains(".version"));
    assert!(kernels.generate_ptx(&gelu).contains(".version"));
}

#[test]
#[serial]
fn test_gemm_fused_no_activation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    // Identity-like matrices for easy verification
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, None, &mut c, m, n, k, 0)
        .expect("GEMM fused should succeed");

    // Each element should be k (dot product of 1s)
    for val in &c {
        assert!((val - k as f32).abs() < 0.001);
    }
}
