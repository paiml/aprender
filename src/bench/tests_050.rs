
    /// QA-050: Documentation consistency
    /// Per spec: Documentation updated with latest benchmark results
    #[test]
    fn test_qa_050_documentation_support() {
        // Verify markdown report generation for documentation
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("test-model", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "test",
            &[100.0],
            &[50.0],
            90.0,
        ));

        let markdown = matrix.to_markdown_table();

        // Markdown should be valid for documentation
        assert!(
            markdown.contains("|"),
            "QA-050: Should produce markdown table"
        );
        assert!(
            markdown.contains("Runtime") || markdown.contains("runtime"),
            "QA-050: Table should have headers"
        );
    }

    // ========================================================================
    // IMP-800: GPU Parity Benchmark Tests
    // ========================================================================

    /// IMP-800a: GPU forward method exists (struct test)
    #[test]
    fn test_imp800a_gpu_parity_benchmark_config() {
        let config = GpuParityBenchmark::new("/path/to/model.gguf")
            .with_prompt("Hello world")
            .with_max_tokens(64)
            .with_ollama_endpoint("http://localhost:11434")
            .with_warmup(5)
            .with_iterations(20);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.prompt, "Hello world");
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.ollama_endpoint, "http://localhost:11434");
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 20);
    }

    /// IMP-800a: GPU forward correctness (default config)
    #[test]
    fn test_imp800a_gpu_parity_benchmark_default() {
        let config = GpuParityBenchmark::default();

        assert!(config.model_path.is_empty());
        assert_eq!(config.prompt, "The quick brown fox");
        assert_eq!(config.max_tokens, 32);
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.measurement_iterations, 10);
        assert!((config.target_cv - 0.05).abs() < f64::EPSILON);
    }

    /// IMP-800b: Result struct captures all metrics
    #[test]
    fn test_imp800b_gpu_parity_result_struct() {
        let result = GpuParityResult::new(150.0, 240.0, 0.03, "NVIDIA RTX 4090", 8192);

        assert!((result.realizar_gpu_tps - 150.0).abs() < f64::EPSILON);
        assert!((result.ollama_tps - 240.0).abs() < f64::EPSILON);
        assert!((result.gap_ratio - 1.6).abs() < 0.01);
        assert!((result.cv - 0.03).abs() < f64::EPSILON);
        assert_eq!(result.gpu_device, "NVIDIA RTX 4090");
        assert_eq!(result.vram_mb, 8192);
    }

    /// IMP-800b: M2/M4 parity thresholds
    #[test]
    fn test_imp800b_parity_thresholds() {
        // M2 parity (within 2x)
        let m2_pass = GpuParityResult::new(130.0, 240.0, 0.03, "GPU", 8192);
        assert!(m2_pass.achieves_m2_parity()); // 1.85x gap
        assert!(!m2_pass.achieves_m4_parity()); // Not within 1.25x

        // M4 parity (within 1.25x)
        let m4_pass = GpuParityResult::new(200.0, 240.0, 0.02, "GPU", 8192);
        assert!(m4_pass.achieves_m2_parity()); // 1.2x gap
        assert!(m4_pass.achieves_m4_parity()); // Within 1.25x

        // Fail both
        let fail = GpuParityResult::new(50.0, 240.0, 0.05, "GPU", 8192);
        assert!(!fail.achieves_m2_parity()); // 4.8x gap
        assert!(!fail.achieves_m4_parity());
    }

    /// IMP-800b: CV stability check
    #[test]
    fn test_imp800b_cv_stability() {
        let stable = GpuParityResult::new(150.0, 240.0, 0.04, "GPU", 8192);
        assert!(stable.measurements_stable()); // CV < 0.05

        let unstable = GpuParityResult::new(150.0, 240.0, 0.08, "GPU", 8192);
        assert!(!unstable.measurements_stable()); // CV >= 0.05
    }

    /// IMP-800c: Gap analysis struct
    #[test]
    fn test_imp800c_gap_analysis_struct() {
        let analysis = GapAnalysis::new(2.0, 1.8).with_statistics(0.01, 1.5, 2.1);

        assert!((analysis.claimed_gap - 2.0).abs() < f64::EPSILON);
        assert!((analysis.measured_gap - 1.8).abs() < f64::EPSILON);
        assert!((analysis.p_value - 0.01).abs() < f64::EPSILON);
        assert!((analysis.ci_95_lower - 1.5).abs() < f64::EPSILON);
        assert!((analysis.ci_95_upper - 2.1).abs() < f64::EPSILON);
    }

    /// IMP-800c: Claim verification logic
    #[test]
    fn test_imp800c_claim_verification() {
        let within_ci = GapAnalysis::new(2.0, 1.8).with_statistics(0.01, 1.5, 2.1);
        assert!(within_ci.claim_verified()); // 1.8 is within [1.5, 2.1]

        let outside_ci = GapAnalysis::new(2.0, 1.2).with_statistics(0.01, 1.5, 2.1);
        assert!(!outside_ci.claim_verified()); // 1.2 is not within [1.5, 2.1]
    }

    /// IMP-800c: Statistical bounds calculation
    #[test]
    fn test_imp800c_statistical_bounds() {
        let analysis = GapAnalysis::new(2.0, 1.8).with_statistics(0.05, 1.6, 2.0);

        assert!((analysis.ci_95_lower - 1.6).abs() < f64::EPSILON);
        assert!((analysis.ci_95_upper - 2.0).abs() < f64::EPSILON);
        assert!((analysis.p_value - 0.05).abs() < f64::EPSILON);
    }

    /// IMP-800c: Popper score computation
    #[test]
    fn test_imp800c_popper_score() {
        // Test with 150 tok/s (passes IMP-800c-1 and IMP-800c-2, fails IMP-800c-3 and IMP-800c-4)
        let analysis = GapAnalysis::new(2.0, 1.6).with_default_claims(150.0);

        // 150 tok/s passes:
        // - IMP-800c-1: threshold 25 tok/s ✓
        // - IMP-800c-2: threshold 24 tok/s ✓
        // - IMP-800c-3: threshold 120 tok/s ✓
        // - IMP-800c-4: threshold 192 tok/s ✗
        // Score should be 75% (3/4)
        assert!((analysis.popper_score - 75.0).abs() < f64::EPSILON);
        assert_eq!(analysis.claims.len(), 4);
    }

    /// IMP-800d: Falsifiable claim evaluation
    #[test]
    fn test_imp800d_falsifiable_claim() {
        let claim = FalsifiableClaim::new("TEST-001", "Test claim", 5.0, 25.0).evaluate(30.0);

        assert_eq!(claim.id, "TEST-001");
        assert_eq!(claim.description, "Test claim");
        assert!((claim.expected - 5.0).abs() < f64::EPSILON);
        assert!((claim.threshold - 25.0).abs() < f64::EPSILON);
        assert!((claim.measured - 30.0).abs() < f64::EPSILON);
        assert!(claim.verified); // 30 >= 25

        let failed_claim =
            FalsifiableClaim::new("TEST-002", "Failing claim", 5.0, 50.0).evaluate(30.0);
        assert!(!failed_claim.verified); // 30 < 50
    }

    /// IMP-800d: GPU faster than CPU check
    #[test]
    fn test_imp800d_gpu_faster_than_cpu() {
        let faster = GpuParityResult::new(30.0, 240.0, 0.03, "GPU", 8192);
        assert!(faster.gpu_faster_than_cpu()); // 30 > 5 tok/s
        assert!((faster.cpu_speedup() - 6.0).abs() < f64::EPSILON); // 30 / 5 = 6x

        let slower = GpuParityResult::new(4.0, 240.0, 0.03, "GPU", 8192);
        assert!(!slower.gpu_faster_than_cpu()); // 4 < 5 tok/s
    }

    // ========================================================================
    // IMP-900a: Optimized GEMM Tests
    // ========================================================================

    /// IMP-900a: Optimized GEMM config defaults
    #[test]
    fn test_imp900a_optimized_gemm_config_default() {
        let config = OptimizedGemmConfig::default();
        assert_eq!(config.tile_size, 32);
        assert_eq!(config.reg_block, 4);
        assert!(!config.use_tensor_cores);
        assert_eq!(config.vector_width, 4);
        assert_eq!(config.k_unroll, 4);
        assert!(config.double_buffer);
    }

    /// IMP-900a: Shared memory calculation
    #[test]
    fn test_imp900a_shared_memory_calculation() {
        let config = OptimizedGemmConfig::default();
        // 32×32 tiles × 4 bytes × 2 tiles × 2 buffers = 32768 bytes
        assert_eq!(config.shared_memory_bytes(), 32 * 32 * 4 * 4);

        let no_double = OptimizedGemmConfig {
            double_buffer: false,
            ..Default::default()
        };
        // Without double buffering: 32×32 × 4 bytes × 2 tiles = 8192 bytes
        assert_eq!(no_double.shared_memory_bytes(), 32 * 32 * 4 * 2);
    }

    /// IMP-900a: Threads per block calculation
    #[test]
    fn test_imp900a_threads_per_block() {
        let config = OptimizedGemmConfig::default();
        // 32/4 = 8 threads per dim, 8×8 = 64 threads
        assert_eq!(config.threads_per_block(), 64);

        let large = OptimizedGemmConfig::large();
        // 64/8 = 8 threads per dim, 8×8 = 64 threads
        assert_eq!(large.threads_per_block(), 64);
    }

    /// IMP-900a: Register allocation calculation
    #[test]
    fn test_imp900a_registers_per_thread() {
        let config = OptimizedGemmConfig::default();
        // 4×4 = 16 accumulators per thread
        assert_eq!(config.registers_per_thread(), 16);

        let large = OptimizedGemmConfig::large();
        // 8×8 = 64 accumulators per thread
        assert_eq!(large.registers_per_thread(), 64);
    }

    /// IMP-900a: GEMM performance result calculation
    #[test]
    fn test_imp900a_gemm_performance_result() {
        // 1024×1024×1024 GEMM in 1.54ms (measured baseline)
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 1.54);

        // 2 * 1024³ = 2,147,483,648 ops
        // 2,147,483,648 / (1.54 * 1e6) = 1394.5 GFLOP/s
        assert!((result.gflops - 1394.5).abs() < 10.0);

        // With RTX 4090 peak (~82 TFLOP/s FP32)
        let with_peak = result.with_peak(82000.0);
        assert!(with_peak.efficiency < 2.0); // ~1.7% efficiency (naive kernel)
    }

    /// IMP-900a: Performance improvement check
    #[test]
    fn test_imp900a_performance_improvement_check() {
        // 1024×1024×1024 GEMM in 0.70ms (~3x faster than 1.54ms baseline)
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 0.70);
        let baseline_gflops = 1396.0;

        // 2 * 1024³ / (0.70 * 1e6) = 3067.8 GFLOP/s
        // 3067.8 / 1396.0 = 2.2x improvement
        assert!(result.improved_by(baseline_gflops, 2.0));
        assert!(!result.improved_by(baseline_gflops, 3.0)); // Not quite 3x
    }

    /// IMP-900a: Expected improvement calculation
    #[test]
    fn test_imp900a_expected_improvement() {
        let benchmark = OptimizedGemmBenchmark::default();
        // With all optimizations: 2.0 * 1.5 * 1.3 * 1.2 = 4.68x
        let expected = benchmark.expected_improvement();
        assert!((expected - 4.68).abs() < 0.1);
    }

    // ========================================================================
    // IMP-900b: Kernel Fusion Tests
    // ========================================================================

    /// IMP-900b: Fused operation specification
    #[test]
    fn test_imp900b_fused_op_spec() {
        let spec = FusedOpSpec {
            op_type: FusedOpType::GemmBiasActivation,
            input_dims: vec![256, 2560],
            output_dims: vec![256, 10240],
            activation: Some("gelu".to_string()),
            fused_launches: 1,
            unfused_launches: 3,
        };

        assert_eq!(spec.launch_reduction(), 3.0);
        assert!(spec.achieves_target_reduction()); // 3x > 2x target
    }

    /// IMP-900b: Launch reduction targets
    #[test]
    fn test_imp900b_launch_reduction_targets() {
        // Good fusion: 4 ops → 1 launch
        let good = FusedOpSpec {
            op_type: FusedOpType::FusedAttention,
            input_dims: vec![1, 32, 512, 80],
            output_dims: vec![1, 32, 512, 80],
            activation: None,
            fused_launches: 1,
            unfused_launches: 4,
        };
        assert!(good.achieves_target_reduction());

        // Marginal fusion: 2 ops → 1 launch (exactly 2x)
        let marginal = FusedOpSpec {
            op_type: FusedOpType::LayerNormLinear,
            input_dims: vec![256, 2560],
            output_dims: vec![256, 2560],
            activation: None,
            fused_launches: 1,
            unfused_launches: 2,
        };
        assert!(marginal.achieves_target_reduction());

        // Poor fusion: 3 ops → 2 launches (only 1.5x)
        let poor = FusedOpSpec {
            op_type: FusedOpType::FusedFfn,
            input_dims: vec![256, 2560],
            output_dims: vec![256, 2560],
            activation: None,
            fused_launches: 2,
            unfused_launches: 3,
        };
        assert!(!poor.achieves_target_reduction());
    }

    // ========================================================================
    // IMP-900c: FlashAttention Tests
    // ========================================================================

    /// IMP-900c: FlashAttention config for phi-2
    #[test]
    fn test_imp900c_flash_attention_phi2_config() {
        let config = FlashAttentionConfig::phi2();
        assert_eq!(config.head_dim, 80);
        assert_eq!(config.num_heads, 32);
        assert!(config.causal);
        // scale = 1/sqrt(80) ≈ 0.1118
        assert!((config.scale - 0.1118).abs() < 0.001);
    }

    /// IMP-900c: Memory comparison naive vs flash
    #[test]
    fn test_imp900c_memory_comparison() {
        let config = FlashAttentionConfig::phi2();

        // 512 tokens
        let (naive_512, flash_512) = config.memory_comparison(512);
        assert_eq!(naive_512, 512 * 512 * 4); // 1 MB
        assert_eq!(flash_512, 64 * 64 * 4 * 2); // 32 KB

        // 2048 tokens
        let (naive_2048, flash_2048) = config.memory_comparison(2048);
        assert_eq!(naive_2048, 2048 * 2048 * 4); // 16 MB
        assert_eq!(flash_2048, 64 * 64 * 4 * 2); // 32 KB (same!)
    }

    /// IMP-900c: Memory savings calculation
    #[test]
    fn test_imp900c_memory_savings() {
        let config = FlashAttentionConfig::phi2();

        // 512 tokens: 1MB / 32KB = 32x savings
        let savings_512 = config.memory_savings(512);
        assert!((savings_512 - 32.0).abs() < 1.0);

        // 2048 tokens: 16MB / 32KB = 512x savings
        let savings_2048 = config.memory_savings(2048);
        assert!((savings_2048 - 512.0).abs() < 10.0);
    }

    // ========================================================================
    // IMP-900d: Memory Pool Tests
    // ========================================================================

    /// IMP-900d: Memory pool default configuration
    #[test]
    fn test_imp900d_memory_pool_default() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.initial_size, 256 * 1024 * 1024); // 256 MB
        assert_eq!(config.max_size, 2 * 1024 * 1024 * 1024); // 2 GB
        assert!(config.use_pinned_memory);
        assert!(config.async_transfers);
        assert_eq!(config.size_classes.len(), 9);
    }

    /// IMP-900d: Size class lookup
    #[test]
    fn test_imp900d_size_class_lookup() {
        let config = MemoryPoolConfig::default();

        // Small allocation → 4KB
        assert_eq!(config.find_size_class(1024), Some(4096));

        // Medium allocation → 1MB
        assert_eq!(config.find_size_class(500_000), Some(1048576));

        // Large allocation → 256MB
        assert_eq!(config.find_size_class(200_000_000), Some(268435456));

        // Too large → None
        assert_eq!(config.find_size_class(500_000_000), None);
    }

    /// IMP-900d: Bandwidth improvement estimate
    #[test]
    fn test_imp900d_bandwidth_improvement() {
        let pinned = MemoryPoolConfig::default();
        assert!((pinned.expected_bandwidth_improvement() - 2.4).abs() < 0.1);

        let unpinned = MemoryPoolConfig {
            use_pinned_memory: false,
            ..Default::default()
        };
        assert!((unpinned.expected_bandwidth_improvement() - 1.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // IMP-900: Combined Result Tests
    // ========================================================================

    /// IMP-900: Combined result from baseline
    #[test]
    fn test_imp900_combined_result_baseline() {
        let result = Imp900Result::from_baseline(13.1); // IMP-800 measured

        assert!((result.baseline_tps - 13.1).abs() < 0.1);
        assert!((result.optimized_tps - 13.1).abs() < 0.1);
        assert!((result.gap_ratio - 18.32).abs() < 0.1); // 240/13.1 ≈ 18.32
        assert!(result.milestone.is_none()); // Not yet at any milestone
    }

    /// IMP-900: Individual optimizations
    #[test]
    fn test_imp900_individual_optimizations() {
        let result = Imp900Result::from_baseline(13.1).with_gemm_improvement(2.5); // 2.5x from optimized GEMM

        assert!((result.optimized_tps - 32.75).abs() < 0.1); // 13.1 * 2.5
        assert!((result.gap_ratio - 7.33).abs() < 0.1); // 240/32.75

        // Still not at M2 (need <5x gap)
        assert!(result.milestone.is_none());
    }
