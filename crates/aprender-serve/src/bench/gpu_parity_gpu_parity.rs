
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GpuParityBenchmark Tests
    // =========================================================================

    #[test]
    fn test_gpu_parity_benchmark_default() {
        let bench = GpuParityBenchmark::default();
        assert!(bench.model_path.is_empty());
        assert_eq!(bench.prompt, "The quick brown fox");
        assert_eq!(bench.max_tokens, 32);
        assert_eq!(bench.ollama_endpoint, "http://localhost:11434");
        assert_eq!(bench.warmup_iterations, 3);
        assert_eq!(bench.measurement_iterations, 10);
    }

    #[test]
    fn test_gpu_parity_benchmark_new() {
        let bench = GpuParityBenchmark::new("/path/to/model.gguf");
        assert_eq!(bench.model_path, "/path/to/model.gguf");
    }

    #[test]
    fn test_gpu_parity_benchmark_builder() {
        let bench = GpuParityBenchmark::new("model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(64)
            .with_ollama_endpoint("http://other:11434")
            .with_warmup(5)
            .with_iterations(20);

        assert_eq!(bench.prompt, "Test prompt");
        assert_eq!(bench.max_tokens, 64);
        assert_eq!(bench.ollama_endpoint, "http://other:11434");
        assert_eq!(bench.warmup_iterations, 5);
        assert_eq!(bench.measurement_iterations, 20);
    }

    // =========================================================================
    // GpuParityResult Tests
    // =========================================================================

    #[test]
    fn test_gpu_parity_result_new() {
        let result = GpuParityResult::new(120.0, 240.0, 0.03, "RTX 4090", 2000);
        assert!((result.realizar_gpu_tps - 120.0).abs() < 0.01);
        assert!((result.ollama_tps - 240.0).abs() < 0.01);
        assert!((result.gap_ratio - 2.0).abs() < 0.01);
        assert_eq!(result.gpu_device, "RTX 4090");
        assert_eq!(result.vram_mb, 2000);
    }

    #[test]
    fn test_gpu_parity_result_gap_ratio_zero_realizar() {
        let result = GpuParityResult::new(0.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.gap_ratio.is_infinite());
    }

    #[test]
    fn test_gpu_parity_result_achieves_m2_parity() {
        let result = GpuParityResult::new(120.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.achieves_m2_parity()); // 2.0x

        let result2 = GpuParityResult::new(100.0, 240.0, 0.03, "GPU", 1000);
        assert!(!result2.achieves_m2_parity()); // 2.4x
    }

    #[test]
    fn test_gpu_parity_result_achieves_m4_parity() {
        let result = GpuParityResult::new(200.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.achieves_m4_parity()); // 1.2x

        let result2 = GpuParityResult::new(150.0, 240.0, 0.03, "GPU", 1000);
        assert!(!result2.achieves_m4_parity()); // 1.6x
    }

    #[test]
    fn test_gpu_parity_result_gpu_faster_than_cpu() {
        let result = GpuParityResult::new(10.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.gpu_faster_than_cpu()); // 10 > 5

        let result2 = GpuParityResult::new(4.0, 240.0, 0.03, "GPU", 1000);
        assert!(!result2.gpu_faster_than_cpu()); // 4 <= 5
    }

    #[test]
    fn test_gpu_parity_result_measurements_stable() {
        let result = GpuParityResult::new(100.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.measurements_stable()); // 0.03 < 0.05

        let result2 = GpuParityResult::new(100.0, 240.0, 0.06, "GPU", 1000);
        assert!(!result2.measurements_stable()); // 0.06 >= 0.05
    }

    #[test]
    fn test_gpu_parity_result_cpu_speedup() {
        let result = GpuParityResult::new(25.0, 240.0, 0.03, "GPU", 1000);
        assert!((result.cpu_speedup() - 5.0).abs() < 0.01); // 25 / 5 = 5x
    }

    // =========================================================================
    // FalsifiableClaim Tests
    // =========================================================================

    #[test]
    fn test_falsifiable_claim_new() {
        let claim = FalsifiableClaim::new("C1", "GPU > 5x CPU", 5.0, 25.0);
        assert_eq!(claim.id, "C1");
        assert_eq!(claim.description, "GPU > 5x CPU");
        assert!((claim.expected - 5.0).abs() < 0.01);
        assert!((claim.threshold - 25.0).abs() < 0.01);
        assert!(!claim.verified);
    }

    #[test]
    fn test_falsifiable_claim_evaluate_verified() {
        let claim = FalsifiableClaim::new("C1", "test", 5.0, 25.0).evaluate(30.0);
        assert!(claim.verified);
        assert!((claim.measured - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_falsifiable_claim_evaluate_not_verified() {
        let claim = FalsifiableClaim::new("C1", "test", 5.0, 25.0).evaluate(20.0);
        assert!(!claim.verified);
    }

    // =========================================================================
    // GapAnalysis Tests
    // =========================================================================

    #[test]
    fn test_gap_analysis_new() {
        let analysis = GapAnalysis::new(18.0, 10.0);
        assert!((analysis.claimed_gap - 18.0).abs() < 0.01);
        assert!((analysis.measured_gap - 10.0).abs() < 0.01);
        assert!(analysis.claims.is_empty());
    }

    #[test]
    fn test_gap_analysis_with_statistics() {
        let analysis = GapAnalysis::new(18.0, 10.0).with_statistics(0.01, 8.0, 12.0);
        assert!((analysis.p_value - 0.01).abs() < 0.001);
        assert!((analysis.ci_95_lower - 8.0).abs() < 0.01);
        assert!((analysis.ci_95_upper - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_gap_analysis_calculate_popper_score() {
        let mut analysis = GapAnalysis::new(18.0, 10.0);
        analysis.add_claim(FalsifiableClaim::new("C1", "test", 5.0, 25.0).evaluate(30.0)); // verified
        analysis.add_claim(FalsifiableClaim::new("C2", "test", 5.0, 25.0).evaluate(20.0)); // not verified
        analysis.calculate_popper_score();
        assert!((analysis.popper_score - 50.0).abs() < 0.01); // 1/2 = 50%
    }

    #[test]
    fn test_gap_analysis_calculate_popper_score_empty() {
        let mut analysis = GapAnalysis::new(18.0, 10.0);
        analysis.calculate_popper_score();
        assert_eq!(analysis.popper_score, 0.0);
    }

    #[test]
    fn test_gap_analysis_claim_verified() {
        let analysis = GapAnalysis::new(18.0, 10.0).with_statistics(0.01, 8.0, 12.0);
        assert!(analysis.claim_verified()); // 10 is within [8, 12]

        let analysis2 = GapAnalysis::new(18.0, 15.0).with_statistics(0.01, 8.0, 12.0);
        assert!(!analysis2.claim_verified()); // 15 is outside [8, 12]
    }

    #[test]
    fn test_gap_analysis_with_default_claims() {
        let analysis = GapAnalysis::new(18.0, 10.0).with_default_claims(30.0);
        assert_eq!(analysis.claims.len(), 4);
        // Claim IMP-800c-1: threshold 25 tok/s, measured 30 -> verified
        assert!(analysis.claims[0].verified);
        // Claim IMP-800c-2: threshold 24 tok/s, measured 30 -> verified
        assert!(analysis.claims[1].verified);
    }

    // =========================================================================
    // OptimizedGemmConfig Tests
    // =========================================================================

    #[test]
    fn test_optimized_gemm_config_default() {
        let config = OptimizedGemmConfig::default();
        assert_eq!(config.tile_size, 32);
        assert_eq!(config.reg_block, 4);
        assert!(!config.use_tensor_cores);
        assert_eq!(config.vector_width, 4);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_optimized_gemm_config_small() {
        let config = OptimizedGemmConfig::small();
        assert_eq!(config.tile_size, 16);
        assert_eq!(config.reg_block, 2);
        assert!(!config.double_buffer);
    }

    #[test]
    fn test_optimized_gemm_config_large() {
        let config = OptimizedGemmConfig::large();
        assert_eq!(config.tile_size, 64);
        assert_eq!(config.reg_block, 8);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_optimized_gemm_config_shared_memory_bytes() {
        let config = OptimizedGemmConfig::default();
        // 32 * 32 * 4 = 4096 bytes per tile
        // 2 tiles * 2 buffers = 4 * 4096 = 16384
        assert_eq!(config.shared_memory_bytes(), 16384);

        let config_no_double = OptimizedGemmConfig::small();
        // 16 * 16 * 4 = 1024 bytes per tile
        // 2 tiles = 2048
        assert_eq!(config_no_double.shared_memory_bytes(), 2048);
    }

    #[test]
    fn test_optimized_gemm_config_threads_per_block() {
        let config = OptimizedGemmConfig::default();
        // 32 / 4 = 8 threads per dim
        // 8 * 8 = 64 threads
        assert_eq!(config.threads_per_block(), 64);
    }

    #[test]
    fn test_optimized_gemm_config_registers_per_thread() {
        let config = OptimizedGemmConfig::default();
        // 4 * 4 = 16 registers
        assert_eq!(config.registers_per_thread(), 16);
    }

    // =========================================================================
    // GemmPerformanceResult Tests
    // =========================================================================

    #[test]
    fn test_gemm_performance_result_new() {
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 10.0);
        // ops = 2 * 1024^3 = 2147483648
        // gflops = 2147483648 / (10 * 1e6) = 214.7
        assert!(result.gflops > 200.0);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_gemm_performance_result_with_peak() {
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 10.0).with_peak(300.0);
        // efficiency = (gflops / 300) * 100
        assert!(result.efficiency > 0.0 && result.efficiency <= 100.0);
    }

    #[test]
    fn test_gemm_performance_result_improved_by() {
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 10.0);
        assert!(result.improved_by(100.0, 2.0)); // 214+ >= 200
        assert!(!result.improved_by(100.0, 10.0)); // 214 < 1000
    }

    // =========================================================================
    // OptimizedGemmBenchmark Tests
    // =========================================================================

    #[test]
    fn test_optimized_gemm_benchmark_default() {
        let bench = OptimizedGemmBenchmark::default();
        assert_eq!(bench.warmup_iterations, 5);
        assert_eq!(bench.measurement_iterations, 20);
    }

    #[test]
    fn test_optimized_gemm_benchmark_with_config() {
        let config = OptimizedGemmConfig::large();
        let bench = OptimizedGemmBenchmark::with_config(config);
        assert_eq!(bench.config.tile_size, 64);
    }

    #[test]
    fn test_optimized_gemm_benchmark_expected_improvement() {
        let bench = OptimizedGemmBenchmark::default();
        let improvement = bench.expected_improvement();
        // With default config: 2 * 1.5 * 1.3 * 1.2 = 4.68
        assert!(improvement > 4.0 && improvement < 5.0);
    }

    // =========================================================================
    // FusedOpType and FusedOpSpec Tests
    // =========================================================================

    #[test]
    fn test_fused_op_type_eq() {
        assert_eq!(
            FusedOpType::GemmBiasActivation,
            FusedOpType::GemmBiasActivation
        );
        assert_ne!(FusedOpType::FusedFfn, FusedOpType::FusedAttention);
    }

    #[test]
    fn test_fused_op_spec_launch_reduction() {
        let spec = FusedOpSpec {
            op_type: FusedOpType::GemmBiasActivation,
            input_dims: vec![1024, 1024],
            output_dims: vec![1024, 1024],
            activation: Some("relu".to_string()),
            fused_launches: 1,
            unfused_launches: 3,
        };
        assert!((spec.launch_reduction() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_op_spec_achieves_target_reduction() {
        let spec = FusedOpSpec {
            op_type: FusedOpType::FusedFfn,
            input_dims: vec![1024],
            output_dims: vec![1024],
            activation: None,
            fused_launches: 1,
            unfused_launches: 3,
        };
        assert!(spec.achieves_target_reduction()); // 3x >= 2x

        let spec2 = FusedOpSpec {
            op_type: FusedOpType::FusedFfn,
            input_dims: vec![1024],
            output_dims: vec![1024],
            activation: None,
            fused_launches: 2,
            unfused_launches: 3,
        };
        assert!(!spec2.achieves_target_reduction()); // 1.5x < 2x
    }

    // =========================================================================
    // FlashAttentionConfig Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_config_phi2() {
        let config = FlashAttentionConfig::phi2();
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.head_dim, 80);
        assert_eq!(config.num_heads, 32);
        assert!(config.causal);
    }

    #[test]
    fn test_flash_attention_config_memory_comparison() {
        let config = FlashAttentionConfig::phi2();
        let (naive, flash) = config.memory_comparison(1024);
        // naive = 1024 * 1024 * 4 = 4MB
        assert_eq!(naive, 4 * 1024 * 1024);
        // flash = 64 * 64 * 4 * 2 = 32KB
        assert!(flash < naive);
    }

    #[test]
    fn test_flash_attention_config_memory_savings() {
        let config = FlashAttentionConfig::phi2();
        let savings = config.memory_savings(2048);
        // Should be significant for large sequences
        assert!(savings > 100.0);
    }

    // =========================================================================
    // MemoryPoolConfig Tests
    // =========================================================================

    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.initial_size, 256 * 1024 * 1024);
        assert!(config.use_pinned_memory);
        assert!(config.async_transfers);
    }

    #[test]
    fn test_memory_pool_config_find_size_class() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.find_size_class(1000), Some(4096));
        assert_eq!(config.find_size_class(5000), Some(16384));
        assert_eq!(config.find_size_class(100_000_000), Some(268_435_456));
        assert_eq!(config.find_size_class(500_000_000), None); // Larger than max class
    }

    #[test]
    fn test_memory_pool_config_expected_bandwidth_improvement() {
        let config = MemoryPoolConfig::default();
        assert!((config.expected_bandwidth_improvement() - 2.4).abs() < 0.01);

        let mut config_no_pinned = MemoryPoolConfig::default();
        config_no_pinned.use_pinned_memory = false;
        assert!((config_no_pinned.expected_bandwidth_improvement() - 1.0).abs() < 0.01);
    }

    // =========================================================================
    // Imp900Result Tests
    // =========================================================================

    #[test]
    fn test_imp900_result_from_baseline() {
        let result = Imp900Result::from_baseline(13.1);
        assert!((result.baseline_tps - 13.1).abs() < 0.01);
        assert!((result.optimized_tps - 13.1).abs() < 0.01);
        assert!(result.milestone.is_none());
    }

    #[test]
    fn test_imp900_result_with_improvements() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(2.0)
            .with_fusion_improvement(1.5)
            .with_flash_attention_improvement(1.3)
            .with_memory_improvement(1.2);

        // 13.1 * 2 * 1.5 * 1.3 * 1.2 = 61.2
        assert!(result.optimized_tps > 60.0);
        assert_eq!(result.milestone, Some("M2".to_string())); // Within 5x of 240
    }

    #[test]
    fn test_imp900_result_achieves_m3() {
        let result = Imp900Result::from_baseline(13.1).with_gemm_improvement(4.0);
        // 13.1 * 4 = 52.4 tok/s, gap = 240/52.4 = 4.6x
        assert!(result.achieves_m3());
    }

    #[test]
    fn test_imp900_result_achieves_m4() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(5.0)
            .with_fusion_improvement(3.0);
        // 13.1 * 5 * 3 = 196.5 tok/s, gap = 240/196.5 = 1.22x
        assert!(result.achieves_m4());
    }

    #[test]
    fn test_imp900_result_total_improvement() {
        let result = Imp900Result::from_baseline(10.0)
            .with_gemm_improvement(2.0)
            .with_fusion_improvement(2.0);
        // optimized = 10 * 2 * 2 = 40
        assert!((result.total_improvement() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_imp900_result_milestones() {
        // from_baseline doesn't calculate milestone; use with_* to trigger recalculate

        // No milestone: gap > 5x (10 tok/s -> gap = 24x)
        let result1 = Imp900Result::from_baseline(10.0).with_gemm_improvement(1.0);
        assert!(result1.milestone.is_none());

        // M2: gap <= 5x (50 tok/s -> gap = 4.8x)
        let result2 = Imp900Result::from_baseline(50.0).with_gemm_improvement(1.0);
        assert_eq!(result2.milestone, Some("M2".to_string()));

        // M3: gap <= 2x (130 tok/s -> gap = 1.85x)
        let result3 = Imp900Result::from_baseline(130.0).with_gemm_improvement(1.0);
        assert_eq!(result3.milestone, Some("M3".to_string()));

        // M4: gap <= 1.25x (200 tok/s -> gap = 1.2x)
        let result4 = Imp900Result::from_baseline(200.0).with_gemm_improvement(1.0);
        assert_eq!(result4.milestone, Some("M4".to_string()));
    }
}
