
    #[test]
    fn test_convoy_test_result_fail_hol_blocking() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![11.0; 100]; // 10% increase - acceptable
        let hol = vec![600.0; 100]; // 600ms > 500ms threshold
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with HOL blocking > 500ms");
        assert!(result.failure_reasons.iter().any(|r| r.contains("HOL")));
    }

    #[test]
    fn test_convoy_test_result_fail_kv_fragmentation() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![11.0; 100];
        let hol = vec![50.0; 100];
        let kv_frag = 20.0; // 20% > 15% threshold

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with KV fragmentation > 15%");
        assert!(result.failure_reasons.iter().any(|r| r.contains("KV")));
    }

    // ========================================================================
    // SaturationTestConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_saturation_config_default() {
        let config = SaturationTestConfig::default();
        assert_eq!(config.cpu_load_pct, 50);
        assert!((config.max_throughput_degradation_pct - 30.0).abs() < 0.01);
        assert!((config.max_p99_increase_pct - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // SaturationTestResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_saturation_test_result_pass() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let stressed_throughput = vec![85.0, 87.0, 83.0, 86.0, 84.0]; // ~15% degradation
        let baseline_latency = vec![10.0, 12.0, 11.0, 10.5, 11.5];
        let stressed_latency = vec![15.0, 17.0, 16.0, 15.5, 16.5]; // ~50% increase

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(result.passed, "Should pass with acceptable degradation");
        assert!(result.throughput_degradation_pct < 30.0);
        assert!(result.p99_increase_pct < 100.0);
    }

    #[test]
    fn test_saturation_test_result_fail_throughput() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0; 100];
        let stressed_throughput = vec![50.0; 100]; // 50% degradation > 30%
        let baseline_latency = vec![10.0; 100];
        let stressed_latency = vec![15.0; 100]; // 50% increase - acceptable

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(
            !result.passed,
            "Should fail with 50% throughput degradation"
        );
        assert!(result
            .failure_reasons
            .iter()
            .any(|r| r.contains("Throughput")));
    }

    #[test]
    fn test_saturation_test_result_fail_p99() {
        let config = SaturationTestConfig::default();
        let baseline_throughput = vec![100.0; 100];
        let stressed_throughput = vec![90.0; 100]; // 10% degradation - acceptable
        let baseline_latency = vec![10.0; 100];
        let stressed_latency = vec![25.0; 100]; // 150% increase > 100%

        let result = SaturationTestResult::new(
            &config,
            &baseline_throughput,
            &stressed_throughput,
            &baseline_latency,
            &stressed_latency,
        );

        assert!(!result.passed, "Should fail with 150% p99 increase");
        assert!(result.failure_reasons.iter().any(|r| r.contains("P99")));
    }

    // ========================================================================
    // HardwareSpec Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_hardware_spec_default() {
        let spec = HardwareSpec::default();
        assert_eq!(spec.cpu, "Unknown");
        assert!(spec.gpu.is_none());
        assert_eq!(spec.memory_gb, 0);
        assert_eq!(spec.storage, "Unknown");
    }

    // ========================================================================
    // SamplingConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.method, "dynamic_cv");
        assert!((config.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(config.warmup_iterations, 100);
    }

    // ========================================================================
    // ThermalInfo Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_thermal_info_default() {
        let info = ThermalInfo::default();
        assert!(info.valid);
        assert!((info.temp_variance_c - 0.0).abs() < 0.001);
        assert!((info.max_temp_c - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // FullBenchmarkResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_full_benchmark_result_from_benchmark_result() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0, 25.0, 23.0],
            itl_ms: vec![10.0, 11.0, 10.5, 11.5, 10.2],
            generation_tok_s: vec![140.0, 142.0, 141.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let hardware = HardwareSpec {
            cpu: "Apple M3 Max".to_string(),
            gpu: Some("Apple M3 Max (40 cores)".to_string()),
            memory_gb: 128,
            storage: "NVMe".to_string(),
        };

        let temps = vec![72.0, 73.0, 72.5, 73.5, 72.0];
        let kl_div = 0.031;

        let full_result =
            FullBenchmarkResult::from_benchmark_result(&result, hardware, &temps, kl_div);

        assert_eq!(full_result.version, "1.1");
        assert!(full_result.timestamp.contains("1970")); // Simple timestamp format
        assert_eq!(full_result.config.model, "test");
        assert_eq!(full_result.hardware.cpu, "Apple M3 Max");
        assert_eq!(full_result.sampling.actual_iterations, 500);
        assert!(full_result.thermal.valid);
        assert!((full_result.quality.kl_divergence_vs_fp32 - 0.031).abs() < 0.001);
    }

    #[test]
    fn test_full_benchmark_result_json_roundtrip() {
        let result = BenchmarkResult {
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.2.3".to_string(),
            },
            cold_start_ms: 100.0,
            model_load_ms: 50.0,
            ttft_ms: vec![20.0, 22.0, 21.0],
            itl_ms: vec![10.0, 11.0, 10.5],
            generation_tok_s: vec![140.0, 142.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let full_result =
            FullBenchmarkResult::from_benchmark_result(&result, HardwareSpec::default(), &[], 0.0);

        let json = full_result.to_json().expect("Should serialize");
        let parsed: FullBenchmarkResult =
            FullBenchmarkResult::from_json(&json).expect("Should parse");

        assert_eq!(parsed.version, "1.1");
        assert_eq!(parsed.config.model, "test");
        assert_eq!(parsed.sampling.actual_iterations, 500);
    }

    // ========================================================================
    // BenchmarkComparison Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_benchmark_comparison_realizar_wins() {
        let baseline = create_test_full_result("llama.cpp", 40.0, 100.0, 1500, 0.06);
        let current = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);

        assert_eq!(comparison.winner, "realizar");
        assert!(comparison.ttft_p99_change_pct < 0.0); // Improvement
        assert!(comparison.throughput_change_pct > 0.0); // Improvement
        assert!(comparison.memory_change_pct < 0.0); // Improvement
        assert!(comparison.energy_change_pct < 0.0); // Improvement
    }

    #[test]
    fn test_benchmark_comparison_tie() {
        let baseline = create_test_full_result("runtime_a", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("runtime_b", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);

        assert_eq!(comparison.winner, "tie");
    }

    // ========================================================================
    // RegressionResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_regression_result_no_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 29.0, 145.0, 1150, 0.038);

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(!regression.regression_detected);
        assert!(regression.regressed_metrics.is_empty());
    }

    #[test]
    fn test_regression_result_ttft_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 35.0, 140.0, 1200, 0.04); // 16.7% worse TTFT

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("ttft")));
    }

    #[test]
    fn test_regression_result_throughput_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 30.0, 120.0, 1200, 0.04); // 14.3% worse throughput

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("throughput")));
    }

    #[test]
    fn test_regression_result_memory_regression() {
        let baseline = create_test_full_result("realizar", 30.0, 140.0, 1200, 0.04);
        let current = create_test_full_result("realizar", 30.0, 140.0, 1400, 0.04); // 16.7% worse memory

        let regression = RegressionResult::check(&baseline, &current, 5.0);

        assert!(regression.regression_detected);
        assert!(regression
            .regressed_metrics
            .iter()
            .any(|m| m.contains("memory")));
    }

    /// Helper function to create test FullBenchmarkResult
    fn create_test_full_result(
        runtime: &str,
        ttft_p99: f64,
        throughput: f64,
        memory_mb: u64,
        token_joules: f64,
    ) -> FullBenchmarkResult {
        FullBenchmarkResult {
            version: "1.1".to_string(),
            timestamp: "2025-12-09T12:00:00Z".to_string(),
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: runtime.to_string(),
                runtime_version: "1.0.0".to_string(),
            },
            hardware: HardwareSpec::default(),
            sampling: SamplingConfig::default(),
            thermal: ThermalInfo::default(),
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: ttft_p99 * 0.7,
                    p95: ttft_p99 * 0.9,
                    p99: ttft_p99,
                    p999: ttft_p99 * 1.2,
                },
                itl_ms: ItlResults {
                    median: 10.0,
                    std_dev: 2.0,
                    p99: 15.0,
                },
                throughput_tok_s: ThroughputResults {
                    median: throughput,
                    ci_95: (throughput * 0.95, throughput * 1.05),
                },
                memory_mb: MemoryResults {
                    model_mb: memory_mb / 2,
                    peak_rss_mb: memory_mb,
                    kv_waste_pct: 3.0,
                },
                energy: EnergyResults {
                    total_joules: 50.0,
                    token_joules,
                    idle_watts: 8.0,
                },
                cold_start_ms: ColdStartResults {
                    median: 100.0,
                    p99: 150.0,
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: 0.03,
                perplexity_wikitext2: Some(5.89),
            },
        }
    }

    // ========================================================================
    // Additional Coverage Tests (Phase 3 - 95% Target)
    // ========================================================================

    #[test]
    fn test_dynamic_sampler_current_cv_empty() {
        let sampler = DynamicSampler::default();
        let cv = sampler.current_cv(&[]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_dynamic_sampler_current_cv_single_value() {
        let sampler = DynamicSampler::default();
        let cv = sampler.current_cv(&[100.0]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_dynamic_sampler_current_cv_constant_values() {
        let sampler = DynamicSampler::default();
        let data: Vec<f64> = vec![50.0; 100];
        let cv = sampler.current_cv(&data);
        assert!(cv.abs() < 1e-10, "CV of constant should be ~0");
    }

    #[test]
    fn test_dynamic_sampler_current_cv_varied_window() {
        let sampler = DynamicSampler {
            cv_window: 10,
            ..Default::default()
        };
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 % 10.0)).collect();
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.0 && cv < 1.0);
    }

    #[test]
    fn test_dynamic_sampler_current_cv_small_window() {
        let sampler = DynamicSampler {
            cv_window: 5,
            ..Default::default()
        };
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = sampler.current_cv(&data);
        assert!(cv > 0.4 && cv < 0.6);
    }

    #[test]
    fn test_dynamic_sampler_default_values() {
        let sampler = DynamicSampler::default();
        assert_eq!(sampler.min_samples, 100);
        assert_eq!(sampler.max_samples, 10_000);
        assert!((sampler.cv_threshold - 0.05).abs() < 0.001);
        assert_eq!(sampler.cv_window, 50);
        assert_eq!(sampler.stability_count, 3);
        // stable_streak is private, tested via should_continue
    }

    #[test]
    fn test_thermal_guard_temp_variance_empty() {
        let guard = ThermalGuard::default();
        let variance = guard.temp_variance(&[]);
        assert!((variance - 0.0).abs() < 0.001);
    }
