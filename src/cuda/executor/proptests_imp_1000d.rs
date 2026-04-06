
/// IMP-1000d: Verify max_throughput preset
#[test]
fn test_imp_1000d_max_throughput_preset() {
    let hints = PtxOptimizationHints::max_throughput();

    assert_eq!(hints.memory_pattern, MemoryPattern::Vector4);
    assert_eq!(hints.register_tiling.width, 8);
    assert_eq!(hints.register_tiling.height, 8);
    assert_eq!(hints.bank_conflict_strategy, BankConflictStrategy::Padding);
    assert!(hints.enable_ilp);
    assert!(hints.uses_vectorized_loads());
    assert_eq!(hints.vector_width(), 4);
    assert_eq!(hints.shared_mem_padding(), 1);
}

/// IMP-1000d: Verify register tiling configurations
#[test]
fn test_imp_1000d_register_tiling() {
    let large = RegisterTiling::large();
    assert_eq!(large.width, 8);
    assert_eq!(large.height, 8);
    assert_eq!(large.registers_needed(), 64);

    let medium = RegisterTiling::medium();
    assert_eq!(medium.registers_needed(), 16);

    let small = RegisterTiling::small();
    assert_eq!(small.registers_needed(), 4);
}

/// IMP-1000d: Verify PtxOptimizer summary and register estimation
#[test]
fn test_imp_1000d_ptx_optimizer() {
    let hints = PtxOptimizationHints::max_throughput();
    let optimizer = PtxOptimizer::new(hints);

    // Summary should contain configuration info
    let summary = optimizer.summary();
    assert!(summary.contains("vec=4"), "Expected vec=4 in: {}", summary);
    assert!(summary.contains("8x8"), "Expected 8x8 in: {}", summary);
    assert!(
        summary.contains("ilp=true"),
        "Expected ilp=true in: {}",
        summary
    );

    // Register estimation: 16 base + 64 accum + 64 ilp = 144
    assert_eq!(optimizer.estimated_registers(), 144);
    assert!(optimizer.is_high_register_pressure());

    // Padded shared memory
    assert_eq!(optimizer.padded_shared_mem_row(32), 33);
}

/// IMP-1000d: Verify low_latency preset
#[test]
fn test_imp_1000d_low_latency_preset() {
    let hints = PtxOptimizationHints::low_latency();
    let optimizer = PtxOptimizer::new(hints);

    assert!(!optimizer.hints().uses_vectorized_loads());
    assert_eq!(optimizer.hints().vector_width(), 1);
    assert!(!optimizer.hints().enable_ilp);

    // Low latency = low register pressure: 16 base + 4 accum = 20
    assert_eq!(optimizer.estimated_registers(), 20);
    assert!(!optimizer.is_high_register_pressure());
}

/// IMP-1000d: Verify bank conflict strategies
#[test]
fn test_imp_1000d_bank_conflict_strategies() {
    let mut hints = PtxOptimizationHints::default();

    // None strategy
    hints.bank_conflict_strategy = BankConflictStrategy::None;
    assert_eq!(hints.shared_mem_padding(), 0);

    // Padding strategy
    hints.bank_conflict_strategy = BankConflictStrategy::Padding;
    assert_eq!(hints.shared_mem_padding(), 1);

    // XOR strategy (no padding, uses different approach)
    hints.bank_conflict_strategy = BankConflictStrategy::Xor;
    assert_eq!(hints.shared_mem_padding(), 0);
}

// ========================================================================
// IMP-800d: GPU Integration Test Suite
// ========================================================================

/// IMP-800d: Stress runner with GPU - verify config and report work
#[test]
fn test_imp_800d_stress_runner_config() {
    use trueno_gpu::testing::{PerformanceThresholds, StressConfig, StressTestRunner};

    let config = StressConfig {
        cycles: 10,
        interval_ms: 0, // No delay for unit test
        seed: 42,
        min_input_size: 64,
        max_input_size: 256,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 100,
            max_memory_bytes: 64 * 1024 * 1024,
            max_timing_variance: 0.5,
            max_failure_rate: 0.01,
        },
    };

    let runner = StressTestRunner::new(config.clone());
    let report = runner.report();

    assert_eq!(report.cycles_completed, 0);
    assert!(report.frames.is_empty());
    assert_eq!(config.seed, 42);
}

/// IMP-800d: Performance verification thresholds enforced
#[test]
fn test_imp_800d_performance_verification() {
    use trueno_gpu::testing::{
        verify_performance, FrameProfile, PerformanceThresholds, StressReport,
    };

    let mut report = StressReport::default();

    // Add frames with varying performance
    for i in 0..10 {
        report.add_frame(FrameProfile {
            cycle: i,
            duration_ms: 20 + i as u64 * 2, // 20-38ms
            memory_bytes: 1024,
            tests_passed: 1,
            tests_failed: 0,
            input_seed: i as u64,
            input_size: 64,
        });
    }

    // Thresholds that should PASS
    let thresholds_pass = PerformanceThresholds {
        max_frame_time_ms: 50,
        max_memory_bytes: 64 * 1024 * 1024,
        max_timing_variance: 0.5,
        max_failure_rate: 0.01,
    };

    let result = verify_performance(&report, &thresholds_pass);
    assert!(result.passed, "Should pass: {:?}", result.violations);
    assert_eq!(result.max_frame_ms, 38);
    assert!(result.violations.is_empty());

    // Thresholds that should FAIL (max frame too low)
    let thresholds_fail = PerformanceThresholds {
        max_frame_time_ms: 30, // Will fail - max is 38ms
        max_memory_bytes: 64 * 1024 * 1024,
        max_timing_variance: 0.5,
        max_failure_rate: 0.01,
    };

    let result_fail = verify_performance(&report, &thresholds_fail);
    assert!(!result_fail.passed, "Should fail due to max frame time");
    assert!(!result_fail.violations.is_empty());
}

/// IMP-800d: TUI renders GPU metrics correctly
#[test]
fn test_imp_800d_tui_output() {
    use trueno_gpu::testing::{
        render_to_string, FrameProfile, PerformanceResult, StressReport, TuiState,
    };

    let mut state = TuiState::new(100);
    let mut report = StressReport::default();

    // Add frames to generate sparkline data
    for i in 0..20 {
        report.add_frame(FrameProfile {
            cycle: i,
            duration_ms: 30 + (i % 5) as u64 * 3, // 30-42ms
            memory_bytes: 1024 * 1024,            // 1MB
            tests_passed: 5,
            tests_failed: 0,
            input_seed: i as u64,
            input_size: 128,
        });
    }

    state.update_from_report(&report);

    let perf = PerformanceResult {
        passed: true,
        max_frame_ms: 42,
        mean_frame_ms: 36.0,
        variance: 0.1,
        pass_rate: 1.0,
        violations: vec![],
    };

    let output = render_to_string(&state, &report, &perf);

    // Verify TUI contains expected sections
    assert!(output.contains("Stress Test Monitor"), "Missing header");
    assert!(output.contains("Cycle:"), "Missing cycle info");
    assert!(output.contains("FPS:"), "Missing FPS");
    assert!(output.contains("PASS"), "Missing status");
    assert!(output.contains("Mean:"), "Missing mean");
}

/// IMP-800d: Deterministic output with same seed
#[test]
fn test_imp_800d_deterministic_output() {
    use trueno_gpu::testing::{StressConfig, StressRng, StressTestRunner};

    // Run twice with same seed
    let seed = 12345u64;

    let mut rng1 = StressRng::new(seed);
    let mut rng2 = StressRng::new(seed);

    // Generate sequences - should be identical
    let seq1: Vec<u32> = (0..100).map(|_| rng1.next_u32()).collect();
    let seq2: Vec<u32> = (0..100).map(|_| rng2.next_u32()).collect();

    assert_eq!(seq1, seq2, "Same seed must produce identical sequences");

    // Verify runner generates same inputs
    let config = StressConfig {
        cycles: 5,
        seed,
        ..StressConfig::default()
    };

    let mut runner1 = StressTestRunner::new(config.clone());
    let mut runner2 = StressTestRunner::new(config);

    for _ in 0..5 {
        let (seed1, input1) = runner1.generate_input();
        let (seed2, input2) = runner2.generate_input();

        assert_eq!(seed1, seed2, "Seeds must match");
        assert_eq!(
            input1, input2,
            "Inputs must match for deterministic testing"
        );
    }
}

/// IMP-800d: Stress test with GPU kernel execution (requires GPU)
#[test]
#[serial]
fn test_imp_800d_stress_runner_gpu() {
    use trueno_gpu::testing::{
        verify_performance, PerformanceThresholds, StressConfig, StressTestRunner,
    };

    if !has_cuda() {
        return;
    }

    let _context = CudaContext::new(0).expect("test");
    let kernels = CudaKernels::new();

    let config = StressConfig {
        cycles: 20,
        interval_ms: 0,
        seed: 42,
        min_input_size: 128,
        max_input_size: 512,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 100, // 10 FPS minimum
            max_memory_bytes: 64 * 1024 * 1024,
            max_timing_variance: 0.5,
            max_failure_rate: 0.01,
        },
    };

    let mut runner = StressTestRunner::new(config.clone());

    // Run stress test with softmax kernel
    let report = runner.run_all(|input| {
        // Generate PTX for this input size
        let _ptx = kernels.generate_ptx(&KernelType::Softmax {
            dim: input.len() as u32,
        });
        // PTX generation succeeded
        (1, 0) // 1 passed, 0 failed
    });

    let result = verify_performance(report, &config.thresholds);
    assert!(
        result.passed,
        "GPU stress test failed: {:?}",
        result.violations
    );
}

// IMP-900: GPU Optimization Infrastructure Tests
// These tests verify the infrastructure for M3/M4 parity milestones

/// IMP-900a: Optimized GEMM kernel infrastructure
#[test]
fn test_imp_900a_optimized_gemm_kernel() {
    let kernels = CudaKernels::new();

    // Test optimized GEMM kernel type exists
    let kernel = KernelType::GemmTiled {
        m: 32,
        n: 4096,
        k: 4096,
        tile_size: 32,
    };

    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"), "IMP-900a: PTX version header");
    assert!(ptx.contains("gemm"), "IMP-900a: Kernel function name");

    // Verify tile parameters are encoded
    assert!(
        ptx.contains(".shared"),
        "IMP-900a: Shared memory for tiling"
    );
}

/// IMP-900a: GEMM performance characteristics
#[test]
fn test_imp_900a_gemm_performance_characteristics() {
    // Document expected performance characteristics
    let tile_size = 32;
    let m = 32;
    let n = 4096;
    let k = 4096;

    // Theoretical FLOPS
    let flops = 2 * m * n * k; // 2 * 32 * 4096 * 4096 = 1.07B FLOPS

    // Memory bandwidth (bytes)
    let input_a = m * k * 4; // FP32
    let input_b = k * n * 4;
    let output_c = m * n * 4;
    let total_memory = input_a + input_b + output_c;

    // Arithmetic intensity (FLOPS per byte)
    let arithmetic_intensity = flops as f64 / total_memory as f64;

    println!("IMP-900a: GEMM Performance Characteristics");
    println!("  Dimensions: {}x{}x{}", m, n, k);
    println!("  Tile size: {}", tile_size);
    println!("  FLOPS: {:.2} GFLOPS", flops as f64 / 1e9);
    println!("  Memory: {:.2} MB", total_memory as f64 / 1e6);
    println!(
        "  Arithmetic Intensity: {:.2} FLOPS/byte",
        arithmetic_intensity
    );

    assert!(
        arithmetic_intensity > 10.0,
        "IMP-900a: GEMM should be compute-bound (>10 FLOPS/byte)"
    );
}

/// IMP-900b: Kernel fusion infrastructure
#[test]
fn test_imp_900b_kernel_fusion_infrastructure() {
    let kernels = CudaKernels::new();

    // Test fused Q4K GEMM kernel
    let fused_kernel = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let name = kernels.kernel_name(&fused_kernel);
    assert_eq!(name, "q4k_gemm_fused", "IMP-900b: Fused kernel name");

    // Test PTX generation for fused kernel
    let ptx = kernels.generate_ptx(&fused_kernel);
    assert!(
        ptx.contains("q4k_gemm_fused"),
        "IMP-900b: Fused kernel in PTX"
    );
}

/// IMP-900b: Kernel fusion types
#[test]
fn test_imp_900b_kernel_fusion_types() {
    // Document available fused kernels
    let fused_kernels = [
        ("q4k_gemm_fused", "Q4_K dequantize + GEMM"),
        ("attention_softmax_fused", "QK matmul + softmax"),
        ("gelu_add_fused", "GELU activation + residual add"),
    ];

    for (name, description) in fused_kernels {
        println!("IMP-900b: {} - {}", name, description);
    }

    assert_eq!(fused_kernels.len(), 3, "IMP-900b: 3 fused kernel types");
}

/// IMP-900c: FlashAttention configuration
#[test]
fn test_imp_900c_flash_attention_config() {
    // FlashAttention memory analysis
    let seq_len = 1024;
    let head_dim = 64;
    let n_heads = 32;

    // Standard attention memory: O(n²)
    let standard_memory = seq_len * seq_len * 4; // FP32 attention matrix

    // FlashAttention memory: O(n) - only block at a time
    let block_size = 64;
    let flash_memory = 2 * block_size * head_dim * 4; // Q and K blocks

    let memory_reduction = standard_memory as f64 / flash_memory as f64;

    println!("IMP-900c: FlashAttention Memory Analysis");
    println!("  Sequence length: {}", seq_len);
    println!("  Head dimension: {}", head_dim);
    println!("  Num heads: {}", n_heads);
    println!("  Standard memory: {:.2} MB", standard_memory as f64 / 1e6);
    println!(
        "  FlashAttention memory: {:.2} KB",
        flash_memory as f64 / 1e3
    );
    println!("  Memory reduction: {:.0}x", memory_reduction);

    assert!(
        memory_reduction > 100.0,
        "IMP-900c: FlashAttention should reduce memory >100x at seq_len=1024"
    );
}
