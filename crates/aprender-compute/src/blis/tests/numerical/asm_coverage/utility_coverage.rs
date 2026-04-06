use crate::blis::*;

// ========================================================================
// Coverage Tests: Utility Types
// ========================================================================

#[test]
fn test_jidoka_error_display() {
    // NumericalDeviation
    let err = JidokaError::NumericalDeviation { computed: 1.5, expected: 1.0, relative_error: 0.5 };
    let display = format!("{}", err);
    assert!(display.contains("numerical deviation"));
    assert!(display.contains("1.5"));
    assert!(display.contains('1'));
    assert!(display.contains("0.5"));

    // NaNDetected
    let err = JidokaError::NaNDetected { location: "test_loc" };
    let display = format!("{}", err);
    assert!(display.contains("NaN"));
    assert!(display.contains("test_loc"));

    // InfDetected
    let err = JidokaError::InfDetected { location: "inf_loc" };
    let display = format!("{}", err);
    assert!(display.contains("Inf"));
    assert!(display.contains("inf_loc"));

    // DimensionMismatch
    let err = JidokaError::DimensionMismatch { expected: (10, 20, 30), actual: (5, 10, 15) };
    let display = format!("{}", err);
    assert!(display.contains("dimension mismatch"));
}

#[test]
fn test_jidoka_guard_check_input() {
    let guard = JidokaGuard::strict();

    // Valid input passes
    assert!(guard.check_input(1.0, "test").is_ok());

    // NaN input fails
    assert!(matches!(
        guard.check_input(f32::NAN, "nan_loc"),
        Err(JidokaError::NaNDetected { location: "nan_loc" })
    ));

    // Inf input fails
    assert!(matches!(
        guard.check_input(f32::INFINITY, "inf_loc"),
        Err(JidokaError::InfDetected { location: "inf_loc" })
    ));

    // Negative Inf input fails
    assert!(matches!(
        guard.check_input(f32::NEG_INFINITY, "neg_inf"),
        Err(JidokaError::InfDetected { location: "neg_inf" })
    ));
}

#[test]
fn test_jidoka_guard_check_special_disabled() {
    let guard = JidokaGuard { epsilon: 1e-6, check_special: false, sample_rate: 1 };

    // With check_special disabled, NaN/Inf should pass check_input
    assert!(guard.check_input(f32::NAN, "test").is_ok());
    assert!(guard.check_input(f32::INFINITY, "test").is_ok());
}

#[test]
fn test_kaizen_metrics_record_and_gflops() {
    let mut metrics = KaizenMetrics::default();

    // Initially zero
    assert_eq!(metrics.gflops(), 0.0);
    assert_eq!(metrics.flops, 0);
    assert_eq!(metrics.samples, 0);

    // Record a 10x10x10 GEMM (2*10*10*10 = 2000 FLOPs)
    metrics.record(10, 10, 10, std::time::Duration::from_nanos(1000));
    assert_eq!(metrics.flops, 2000);
    assert_eq!(metrics.samples, 1);
    assert!((metrics.gflops() - 2.0).abs() < 0.01); // 2000 flops / 1000 ns = 2 GFLOP/s

    // Record another
    metrics.record(10, 10, 10, std::time::Duration::from_nanos(1000));
    assert_eq!(metrics.flops, 4000);
    assert_eq!(metrics.samples, 2);

    // Reset
    metrics.reset();
    assert_eq!(metrics.flops, 0);
    assert_eq!(metrics.samples, 0);
    assert_eq!(metrics.gflops(), 0.0);
}

#[test]
fn test_blis_level_stats() {
    let mut stats = BlisLevelStats::default();

    // Initially zero
    assert_eq!(stats.avg_us(), 0.0);
    assert_eq!(stats.gflops(), 0.0);
    assert_eq!(stats.count, 0);

    // Record some data: 1000 ns, 1000 FLOPs
    stats.record(1000, 1000);
    assert_eq!(stats.count, 1);
    assert!((stats.avg_us() - 1.0).abs() < 0.01); // 1000 ns = 1 us
    assert!((stats.gflops() - 1.0).abs() < 0.01); // 1000 flops / 1000 ns = 1 GFLOP/s

    // Record more: 2000 ns, 2000 FLOPs
    stats.record(2000, 2000);
    assert_eq!(stats.count, 2);
    assert!((stats.avg_us() - 1.5).abs() < 0.01); // (1000+2000)/2/1000 = 1.5 us
    assert!((stats.gflops() - 1.0).abs() < 0.01); // 3000 flops / 3000 ns = 1 GFLOP/s
}

#[test]
fn test_blis_profiler_disabled() {
    let mut profiler = BlisProfiler::new();
    assert!(!profiler.enabled);

    // Recording when disabled should not change anything
    profiler.record(BlisProfileLevel::Macro, 1000, 1000);
    assert_eq!(profiler.macro_stats.count, 0);
}

#[test]
fn test_blis_profiler_enabled() {
    let mut profiler = BlisProfiler::enabled();
    assert!(profiler.enabled);

    // Record at each level
    profiler.record(BlisProfileLevel::Macro, 1000, 1000);
    profiler.record(BlisProfileLevel::Midi, 500, 500);
    profiler.record(BlisProfileLevel::Micro, 100, 100);
    profiler.record(BlisProfileLevel::Pack, 200, 0);

    assert_eq!(profiler.macro_stats.count, 1);
    assert_eq!(profiler.midi_stats.count, 1);
    assert_eq!(profiler.micro_stats.count, 1);
    assert_eq!(profiler.pack_stats.count, 1);

    // Total GFLOP/s based on macro level
    assert!((profiler.total_gflops() - 1.0).abs() < 0.01);
}

#[test]
fn test_blis_profiler_summary() {
    let mut profiler = BlisProfiler::enabled();
    profiler.record(BlisProfileLevel::Macro, 1000000, 1000000); // 1 GFLOP in 1ms
    profiler.record(BlisProfileLevel::Midi, 100000, 100000);
    profiler.record(BlisProfileLevel::Micro, 10000, 10000);
    profiler.record(BlisProfileLevel::Pack, 5000, 0);

    let summary = profiler.summary();
    assert!(summary.contains("BLIS Profiler Summary"));
    assert!(summary.contains("Macro:"));
    assert!(summary.contains("Midi:"));
    assert!(summary.contains("Micro:"));
    assert!(summary.contains("Pack:"));
    assert!(summary.contains("Total:"));
}

#[test]
fn test_blis_profiler_reset() {
    let mut profiler = BlisProfiler::enabled();
    profiler.record(BlisProfileLevel::Macro, 1000, 1000);
    profiler.record(BlisProfileLevel::Midi, 500, 500);

    profiler.reset();

    assert_eq!(profiler.macro_stats.count, 0);
    assert_eq!(profiler.midi_stats.count, 0);
    assert_eq!(profiler.micro_stats.count, 0);
    assert_eq!(profiler.pack_stats.count, 0);
}

#[test]
fn test_heijunka_scheduler_partition() {
    let scheduler = HeijunkaScheduler { num_threads: 4, variance_threshold: 0.05 };

    // Test partitioning with M=100, MC=32
    let partitions = scheduler.partition_m(100, 32);
    // Should get partitions for workers
    assert!(!partitions.is_empty());

    // Total should cover all M
    let total: usize = partitions.iter().map(|r| r.len()).sum();
    assert_eq!(total, 100);

    // Each partition should be non-empty
    for p in &partitions {
        assert!(!p.is_empty());
    }
}

#[test]
fn test_heijunka_scheduler_small_m() {
    let scheduler = HeijunkaScheduler { num_threads: 4, variance_threshold: 0.05 };

    // Test with M smaller than MC
    let partitions = scheduler.partition_m(10, 32);
    // Should still partition among workers
    let total: usize = partitions.iter().map(|r| r.len()).sum();
    assert_eq!(total, 10);
}

#[test]
fn test_heijunka_scheduler_default() {
    let scheduler = HeijunkaScheduler::default();
    assert!(scheduler.num_threads >= 1);
    assert!(scheduler.variance_threshold > 0.0);
}

#[test]
fn test_backend_cost_model_select() {
    let model = BackendCostModel {
        pcie_bandwidth_gbps: 15.75,
        gpu_peak_tflops: 10.0,
        cpu_peak_gflops: 400.0,
        gpu_min_elements: 1_000_000,
    };

    // Small matrix - should use CPU (or Scalar)
    let backend = model.select_backend(16, 16, 16);
    assert!(matches!(backend, ComputeBackend::Cpu | ComputeBackend::Scalar));

    // Large matrix - may use GPU if feature enabled, otherwise CPU
    let backend = model.select_backend(4096, 4096, 4096);
    assert!(matches!(
        backend,
        ComputeBackend::Gpu | ComputeBackend::Cpu | ComputeBackend::Scalar | ComputeBackend::Wgpu
    ));
}

#[test]
fn test_backend_cost_model_estimate_time() {
    let model = BackendCostModel::default();

    // CPU estimate for small matrix
    let cpu_time = model.estimate_time_us(32, 32, 32, ComputeBackend::Cpu);
    assert!(cpu_time > 0.0);

    // GPU estimate for same matrix
    let gpu_time = model.estimate_time_us(32, 32, 32, ComputeBackend::Gpu);
    assert!(gpu_time > 0.0);

    // Scalar estimate
    let scalar_time = model.estimate_time_us(32, 32, 32, ComputeBackend::Scalar);
    assert!(scalar_time > 0.0);

    // Wgpu estimate
    let wgpu_time = model.estimate_time_us(32, 32, 32, ComputeBackend::Wgpu);
    assert!(wgpu_time > 0.0);
}

#[test]
fn test_roofline_result() {
    // Compute-bound result
    let compute = RooflineResult::ComputeBound { ai: 100.0, ridge_point: 50.0 };
    assert!(compute.is_compute_bound());
    assert!((compute.arithmetic_intensity() - 100.0).abs() < 0.01);

    // Memory-bound result
    let memory = RooflineResult::MemoryBound { ai: 2.0, ridge_point: 50.0 };
    assert!(!memory.is_compute_bound());
    assert!((memory.arithmetic_intensity() - 2.0).abs() < 0.01);
}

#[test]
fn test_unified_brick_profiler() {
    let mut profiler = UnifiedBrickProfiler::new();

    // Record some selections
    profiler.record_selection(100, 100, 100, ComputeBackend::Cpu);
    profiler.record_selection(1000, 1000, 1000, ComputeBackend::Gpu);

    // Check roofline analysis
    let result = profiler.roofline_analysis(512, 512, 512);
    // Should return a valid result
    match result {
        RooflineResult::ComputeBound { .. } | RooflineResult::MemoryBound { .. } => {}
    }

    // Summary should work
    let summary = profiler.summary();
    assert!(!summary.is_empty());
}

#[test]
fn test_transpose() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let mut b = vec![0.0; 6]; // 3x2 matrix

    transpose(2, 3, &a, &mut b).unwrap();

    // [1 2 3]T = [1 4]
    // [4 5 6]    [2 5]
    //            [3 6]
    assert_eq!(b, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_size_mismatch() {
    let a = vec![1.0, 2.0, 3.0];
    let mut b = vec![0.0; 6];

    // Wrong input size
    let result = transpose(2, 3, &a, &mut b);
    assert!(result.is_err());
}

#[test]
fn test_gemm_function() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    gemm(2, 2, 2, &a, &b, &mut c).unwrap();

    // Should give same result as gemm_reference
    let mut c_ref = vec![0.0; 4];
    gemm_reference(2, 2, 2, &a, &b, &mut c_ref).unwrap();

    for (i, (val, expected)) in c.iter().zip(c_ref.iter()).enumerate() {
        assert!((val - expected).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, val, expected);
    }
}

#[test]
fn test_gemm_auto() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    gemm_auto(2, 2, 2, &a, &b, &mut c, None).unwrap();

    // Check correctness
    let mut c_ref = vec![0.0; 4];
    gemm_reference(2, 2, 2, &a, &b, &mut c_ref).unwrap();

    for (val, expected) in c.iter().zip(c_ref.iter()) {
        assert!((val - expected).abs() < 1e-5);
    }
}

#[test]
fn test_gemm_auto_selection_history() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    let mut profiler = UnifiedBrickProfiler::new();

    gemm_auto(2, 2, 2, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    // Profiler should have recorded the selection
    assert!(!profiler.selection_history.is_empty());
}

#[test]
fn test_gemm_profiled() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    let mut profiler = BlisProfiler::enabled();

    gemm_profiled(2, 2, 2, &a, &b, &mut c, &mut profiler).unwrap();

    // Profiler should be enabled
    assert!(profiler.enabled);
}

#[test]
fn test_packed_sizes() {
    // Test packed_a_size
    let a_size = packed_a_size(72, 256);
    // Should be MC * KC rounded up
    assert!(a_size >= 72 * 256);

    // Test packed_b_size
    let b_size = packed_b_size(256, 4096);
    // Should be KC * NC rounded up
    assert!(b_size >= 256 * 4096);
}

#[test]
fn test_compute_backend_variants() {
    // Test equality
    assert_eq!(ComputeBackend::Cpu, ComputeBackend::Cpu);
    assert_ne!(ComputeBackend::Cpu, ComputeBackend::Gpu);

    // Test debug
    let debug = format!("{:?}", ComputeBackend::Gpu);
    assert!(debug.contains("Gpu"));
}

#[test]
fn test_brick_level_variants() {
    // Test all variants
    let levels = [BrickLevel::Nano, BrickLevel::Micro, BrickLevel::Meso];

    for level in &levels {
        let debug = format!("{:?}", level);
        assert!(!debug.is_empty());
    }

    // Test equality
    assert_eq!(BrickLevel::Nano, BrickLevel::Nano);
    assert_ne!(BrickLevel::Nano, BrickLevel::Micro);
}
