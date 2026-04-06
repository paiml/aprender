
// =============================================================================
// ComputeBackend tests
// =============================================================================

#[test]
fn test_compute_backend_debug() {
    let gpu = ComputeBackend::Gpu;
    let cpu = ComputeBackend::Cpu;
    let auto = ComputeBackend::Auto;

    assert!(format!("{:?}", gpu).contains("Gpu"));
    assert!(format!("{:?}", cpu).contains("Cpu"));
    assert!(format!("{:?}", auto).contains("Auto"));
}

#[test]
fn test_compute_backend_copy() {
    let original = ComputeBackend::Gpu;
    let copied = original;
    assert_eq!(original, copied);
}

// =============================================================================
// GpuCompute is_gpu tests
// =============================================================================

#[test]
fn test_gpu_compute_is_gpu_cpu_backend() {
    let compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU backend");
    assert!(!compute.is_gpu());
    assert_eq!(compute.backend(), ComputeBackend::Cpu);
}

#[test]
fn test_gpu_compute_auto_backend_type() {
    let compute = GpuCompute::auto().expect("auto backend");
    // Auto picks GPU if available, otherwise CPU
    let backend = compute.backend();
    assert!(backend == ComputeBackend::Gpu || backend == ComputeBackend::Cpu);
}

// =============================================================================
// HybridScheduler::should_use_gpu edge cases
// =============================================================================

#[test]
fn test_should_use_gpu_m0() {
    let scheduler = HybridScheduler::with_threshold(1).expect("scheduler");
    // m=0 should not use GPU
    assert!(!scheduler.should_use_gpu(0, 100, 100));
}

#[test]
fn test_should_use_gpu_threshold_boundary() {
    let scheduler = HybridScheduler::with_threshold(1000).expect("scheduler");
    // Exactly at threshold: 10*10*10 = 1000
    let at_threshold = scheduler.should_use_gpu(10, 10, 10);
    // Just below: 9*10*10 = 900 < 1000
    let below = scheduler.should_use_gpu(9, 10, 10);

    // m > 1 and at threshold should consider GPU (if available)
    // but below threshold should not
    assert!(!below || !scheduler.has_gpu());
    // At threshold depends on GPU availability
    assert!(at_threshold == scheduler.has_gpu());
}

// =============================================================================
// AsyncGpuResult edge cases
// =============================================================================

#[test]
fn test_async_gpu_result_set_result_overwrites() {
    let mut result = AsyncGpuResult::ready(vec![1.0, 2.0]);
    result.set_result(vec![3.0, 4.0, 5.0]);
    assert!(result.is_ready());
    assert_eq!(result.try_get().expect("try_get"), &vec![3.0, 4.0, 5.0]);
}

// =============================================================================
// GpuPoolStats tests
// =============================================================================

#[test]
fn test_gpu_pool_stats_debug() {
    let stats = GpuPoolStats {
        cached_buffers: 5,
        cached_bytes: 10240,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("cached_buffers"));
    assert!(debug.contains("5"));
}

#[test]
fn test_gpu_pool_stats_clone() {
    let stats = GpuPoolStats {
        cached_buffers: 3,
        cached_bytes: 4096,
    };
    let cloned = stats;
    assert_eq!(cloned.cached_buffers, 3);
    assert_eq!(cloned.cached_bytes, 4096);
}
