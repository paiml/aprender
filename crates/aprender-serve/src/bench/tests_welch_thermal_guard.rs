
    #[test]
    fn test_welch_t_test_alpha_levels() {
        // Different alpha levels affect significance
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let sample_b = vec![11.0, 12.0, 11.5, 11.2, 11.8];
        let result_strict = welch_t_test(&sample_a, &sample_b, 0.01);
        let result_lenient = welch_t_test(&sample_a, &sample_b, 0.10);
        // Lenient alpha should be at least as likely to find significance
        if result_strict.significant {
            assert!(result_lenient.significant);
        }
    }

    // BENCH-009: ThermalGuard Tests (TDD RED)
    #[test]
    fn test_thermal_guard_struct_fields() {
        // Per spec: ThermalGuard has max_temp_c, cooldown_threshold_c, cooldown_sleep_ms, temp_variance_c
        let guard = ThermalGuard::new(80.0, 70.0, 10_000, 2.0);
        assert_eq!(guard.max_temp_c, 80.0);
        assert_eq!(guard.cooldown_threshold_c, 70.0);
        assert_eq!(guard.cooldown_sleep_ms, 10_000);
        assert_eq!(guard.temp_variance_c, 2.0);
    }

    #[test]
    fn test_thermal_guard_default() {
        // Default should use spec values: 80°C, 70°C, 10000ms, 2°C
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp_c, 80.0);
        assert_eq!(guard.cooldown_threshold_c, 70.0);
        assert_eq!(guard.cooldown_sleep_ms, 10_000);
        assert_eq!(guard.temp_variance_c, 2.0);
    }

    #[test]
    fn test_thermal_validity_valid() {
        // Low variance temps should be valid
        let guard = ThermalGuard::default();
        let temps = vec![75.0, 76.0, 75.5, 76.5, 75.2]; // Variance < 2°C
        let result = guard.validate_run(&temps);
        assert!(matches!(result, ThermalValidity::Valid));
    }

    #[test]
    fn test_thermal_validity_invalid_high_variance() {
        // High variance temps should be invalid
        let guard = ThermalGuard::default();
        let temps = vec![60.0, 80.0, 65.0, 85.0, 70.0]; // High variance
        let result = guard.validate_run(&temps);
        assert!(matches!(result, ThermalValidity::Invalid(_)));
    }

    #[test]
    fn test_thermal_needs_cooldown_above_max() {
        // Above max temp should need cooldown
        let guard = ThermalGuard::default();
        assert!(guard.needs_cooldown(85.0)); // 85 > 80
    }

    #[test]
    fn test_thermal_needs_cooldown_below_max() {
        // Below max temp should not need cooldown
        let guard = ThermalGuard::default();
        assert!(!guard.needs_cooldown(75.0)); // 75 < 80
    }

    // BENCH-010: KL-Divergence Quality Validation Tests (TDD RED)
    #[test]
    fn test_quality_result_pass() {
        // QualityResult::Pass should contain kl_divergence
        let result = QualityResult::Pass {
            kl_divergence: 0.001,
        };
        match result {
            QualityResult::Pass { kl_divergence } => assert!(kl_divergence < 0.01),
            QualityResult::Fail { .. } => panic!("Expected Pass"),
        }
    }

    #[test]
    fn test_quality_result_fail() {
        // QualityResult::Fail should contain kl_divergence, threshold, message
        let result = QualityResult::Fail {
            kl_divergence: 0.1,
            threshold: 0.05,
            message: "Degradation detected",
        };
        match result {
            QualityResult::Fail {
                kl_divergence,
                threshold,
                message,
            } => {
                assert!(kl_divergence > threshold);
                assert!(!message.is_empty());
            },
            QualityResult::Pass { .. } => panic!("Expected Fail"),
        }
    }

    #[test]
    fn test_validate_quantization_identical() {
        // Identical logits should pass with kl_div ~= 0
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.01);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    #[test]
    fn test_validate_quantization_slight_difference() {
        // Small difference should still pass
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![1.01, 2.01, 3.01, 4.01]; // ~1% off
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.05);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    #[test]
    fn test_validate_quantization_large_difference() {
        // Large difference should fail
        let fp32_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let quant_logits: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0]; // Reversed distribution
        let result = validate_quantization_quality(&fp32_logits, &quant_logits, 0.01);
        assert!(matches!(result, QualityResult::Fail { .. }));
    }

    #[test]
    fn test_softmax_basic() {
        // Test softmax via validate_quantization_quality
        // Softmax should produce probability distribution
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        // Sum should be ~1.0
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    // =========================================================================
    // OllamaBackend Tests (EXTREME TDD - REAL HTTP Integration)
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_creation() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Ollama);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_info() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "phi2:2.7b".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Ollama);
        assert!(info.supports_streaming);
        assert_eq!(info.loaded_model, Some("phi2:2.7b".to_string()));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_connection_error() {
        // Invalid port should fail
        let config = OllamaConfig {
            base_url: "http://localhost:59999".to_string(),
            model: "test".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        assert!(result.is_err());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama2");
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_ollama_backend_with_custom_client() {
        use crate::http_client::ModelHttpClient;

        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
        };
        let client = ModelHttpClient::with_timeout(30);
        let backend = OllamaBackend::with_client(config, client);

        // Should create without panicking
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Ollama);
    }

    // Integration test - requires running Ollama server
    #[cfg(feature = "bench-http")]
    #[test]
    #[ignore = "Requires Ollama server at localhost:11434"]
    fn test_ollama_backend_real_inference() {
        let config = OllamaConfig {
            base_url: "http://localhost:11434".to_string(),
            model: "phi2:2.7b".to_string(),
        };
        let backend = OllamaBackend::new(config);
        let request = InferenceRequest::new("What is 2+2?")
            .with_max_tokens(20)
            .with_temperature(0.1);

        let result = backend.inference(&request);

        // MUST succeed with real server
        let response = result.expect("Ollama inference failed - is server running?");

        // Verify REAL data
        assert!(
            response.ttft_ms > 0.0,
            "TTFT must be positive (real latency)"
        );
        assert!(response.total_time_ms > 0.0, "Total time must be positive");
        assert!(response.tokens_generated > 0, "Must generate tokens");
        assert!(!response.text.is_empty(), "Must get actual text");

        println!("Ollama Real Inference via Backend:");
        println!("  TTFT: {:.2}ms", response.ttft_ms);
        println!("  Total: {:.2}ms", response.total_time_ms);
        println!("  Tokens: {}", response.tokens_generated);
        println!("  Text: {}", response.text);
    }

    // ========================================================================
    // Distributed Benchmark Suite Tests
    // ========================================================================

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_default() {
        let config = DistributedBenchConfig::default();
        assert_eq!(config.gpu_counts, vec![1, 2, 4, 8]);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup, 10);
        assert_eq!(config.model_params, 7_000_000_000);
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.batch_size, 1);
        assert!((config.efficiency_threshold - 0.85).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_small_model() {
        let config = DistributedBenchConfig::for_small_model();
        assert_eq!(config.gpu_counts, vec![1, 2]);
        assert_eq!(config.model_params, 125_000_000);
        assert!((config.efficiency_threshold - 0.80).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_config_large_model() {
        let config = DistributedBenchConfig::for_large_model();
        assert_eq!(config.gpu_counts, vec![2, 4, 8]);
        assert_eq!(config.model_params, 70_000_000_000);
        assert_eq!(config.seq_len, 4096);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_suite_new() {
        let config = DistributedBenchConfig::default();
        let suite = DistributedBenchSuite::new(config.clone());
        assert_eq!(suite.config().gpu_counts, config.gpu_counts);
        assert!(suite.scaling_results().is_empty());
        assert!(suite.tp_results().is_empty());
        assert!(suite.pp_results().is_empty());
        assert!(suite.comm_results().is_empty());
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_scaling() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_scaling_benchmark();

        let results = suite.scaling_results();
        assert_eq!(results.len(), 4); // 1, 2, 4, 8 GPUs

        // First result should be 1 GPU (baseline)
        assert_eq!(results[0].gpu_count, 1);
        assert!((results[0].efficiency - 1.0).abs() < 0.001);
        assert!(results[0].comm_overhead_ms.abs() < 0.001);

        // Multi-GPU should have lower efficiency due to overhead
        for result in results.iter().skip(1) {
            assert!(result.efficiency < 1.0);
            assert!(result.efficiency > 0.0); // Efficiency is always positive
            assert!(result.comm_overhead_ms > 0.0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.latency_p50_ms > 0.0);
            assert!(result.latency_p99_ms > result.latency_p50_ms);
        }

        // 2 GPUs should be >85% efficient (spec target for 2-8 GPUs)
        let gpu2 = results.iter().find(|r| r.gpu_count == 2).expect("test");
        assert!(gpu2.efficiency > 0.85, "2-GPU efficiency should be >85%");
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_scaling_efficiency_result_meets_threshold() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.90,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        assert!(result.meets_threshold(0.85));
        assert!(result.meets_threshold(0.90));
        assert!(!result.meets_threshold(0.95));
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_scaling_efficiency_parallel_fraction() {
        let result = ScalingEfficiencyResult {
            gpu_count: 4,
            throughput_tps: 400.0,
            latency_p50_ms: 2.5,
            latency_p99_ms: 3.75,
            efficiency: 0.85,
            comm_overhead_ms: 0.5,
            theoretical_speedup: 3.6,
            achieved_speedup: 3.4,
        };

        let parallel = result.parallel_fraction();
        assert!(parallel > 0.8); // Should be highly parallelizable
        assert!(parallel <= 1.0);

        // Single GPU case
        let single = ScalingEfficiencyResult {
            gpu_count: 1,
            throughput_tps: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 15.0,
            efficiency: 1.0,
            comm_overhead_ms: 0.0,
            theoretical_speedup: 1.0,
            achieved_speedup: 1.0,
        };
        assert!((single.parallel_fraction() - 1.0).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_tensor_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_tensor_parallel_benchmark();

        let results = suite.tp_results();
        assert!(!results.is_empty());

        // Check that TP=1 has no communication overhead
        let tp1 = results.iter().find(|r| r.tp_degree == 1).expect("test");
        assert!(tp1.all_reduce_ms.abs() < 0.001);
        assert!(tp1.comm_overhead_pct.abs() < 0.001);

        // Check that higher TP degrees have communication overhead
        for result in results.iter().filter(|r| r.tp_degree > 1) {
            assert!(result.all_reduce_ms > 0.0);
            assert!(result.comm_overhead_pct > 0.0);
            assert!(result.memory_per_gpu_mb > 0.0);
            assert!(result.effective_tflops > 0.0);
        }
    }

    #[test]
    #[cfg(feature = "distributed-bench")]
    fn test_distributed_bench_pipeline_parallel() {
        let config = DistributedBenchConfig::default();
        let mut suite = DistributedBenchSuite::new(config);
        suite.run_pipeline_parallel_benchmark();

        let results = suite.pp_results();
        assert!(!results.is_empty());

        // Check PP=1 has no bubble
        let pp1 = results.iter().find(|r| r.pp_degree == 1).expect("test");
        assert!(pp1.bubble_ratio.abs() < 0.001);
        assert!(pp1.inter_stage_ms.abs() < 0.001);

        // Check higher PP degrees have bubble and inter-stage latency
        for result in results.iter().filter(|r| r.pp_degree > 1) {
            assert!(result.bubble_ratio > 0.0);
            assert!(result.bubble_ratio < 1.0); // Should be <100%
            assert!(result.inter_stage_ms > 0.0);
            assert!(result.micro_batches > 0);
            assert!(result.throughput_tps > 0.0);
            assert!(result.memory_per_stage_mb > 0.0);
        }
    }
