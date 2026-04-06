
    #[test]
    fn test_thermal_guard_temp_variance_single() {
        let guard = ThermalGuard::default();
        let variance = guard.temp_variance(&[75.0]);
        assert!((variance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_constant() {
        let guard = ThermalGuard::default();
        let temps = vec![72.0; 100];
        let variance = guard.temp_variance(&temps);
        assert!(variance < 0.001);
    }

    #[test]
    fn test_thermal_guard_temp_variance_varied() {
        let guard = ThermalGuard::default();
        let temps = vec![70.0, 72.0, 74.0, 76.0, 78.0];
        let variance = guard.temp_variance(&temps);
        assert!(variance > 2.0 && variance < 4.0);
    }

    #[test]
    fn test_thermal_guard_max_temp_empty() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp(&[]), 0.0);
    }

    #[test]
    fn test_thermal_guard_max_temp_single() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.max_temp(&[82.5]), 82.5);
    }

    #[test]
    fn test_thermal_guard_cooldown_not_needed() {
        let guard = ThermalGuard::default();
        // Should not sleep when temp is below max
        guard.cooldown_if_needed(70.0);
        // Test passes if no timeout
    }

    #[test]
    fn test_chrono_timestamp_format() {
        let ts = chrono_timestamp();
        assert!(ts.contains("1970"));
        assert!(ts.contains("T"));
        assert!(ts.contains("Z"));
        assert!(ts.contains("+"));
        assert!(ts.contains("s"));
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let (lower, upper) = bootstrap_ci(&[], 0.95, 1000);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 0.0);
    }

    #[test]
    fn test_bootstrap_ci_single_value() {
        let (lower, upper) = bootstrap_ci(&[42.0], 0.95, 1000);
        assert!((lower - 42.0).abs() < 0.01);
        assert!((upper - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_ci_varied_data() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);
        // Mean is 50.5, CI should contain mean
        assert!(lower < 55.0);
        assert!(upper > 45.0);
        assert!(lower < upper);
    }

    #[test]
    fn test_bootstrap_ci_narrow_confidence() {
        let data = vec![100.0; 50];
        let (lower, upper) = bootstrap_ci(&data, 0.50, 100);
        // Even narrow CI should be close to 100 for constant data
        assert!((lower - 100.0).abs() < 0.1);
        assert!((upper - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[42.0], 99.0), 42.0);
    }

    #[test]
    fn test_compute_std_dev_constant() {
        let data = vec![100.0; 50];
        let std_dev = compute_std_dev(&data);
        assert!(std_dev < 0.001);
    }

    #[test]
    fn test_compute_std_dev_empty() {
        let std_dev = compute_std_dev(&[]);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_compute_variance_empty() {
        assert_eq!(compute_variance(&[]), 0.0);
    }

    #[test]
    fn test_compute_variance_single() {
        assert_eq!(compute_variance(&[100.0]), 0.0);
    }

    #[test]
    fn test_compute_cv_single_value() {
        let cv = compute_cv(&[100.0]);
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_compute_cv_zero_mean() {
        // Mix of positive and negative that averages to near zero
        let data = vec![-1.0, 1.0, -1.0, 1.0];
        let cv = compute_cv(&data);
        // Mean is 0, CV should be infinite
        assert!(cv.is_infinite());
    }

    #[test]
    fn test_energy_metrics_tokens_per_joule_zero_joules() {
        let metrics = EnergyMetrics::new(0.0, 10.0, 50.0, 1000);
        assert_eq!(metrics.tokens_per_joule(), 0.0);
    }

    #[test]
    fn test_energy_metrics_very_small_joules() {
        let metrics = EnergyMetrics::new(1e-15, 10.0, 50.0, 1000);
        assert_eq!(metrics.tokens_per_joule(), 0.0);
    }

    #[test]
    fn test_itl_metrics_single_value() {
        let metrics = ItlMetrics::from_measurements(&[15.0]);
        assert_eq!(metrics.median_ms, 15.0);
        assert_eq!(metrics.p99_ms, 15.0);
        assert_eq!(metrics.p999_ms, 15.0);
        assert_eq!(metrics.std_dev_ms, 0.0);
    }

    #[test]
    fn test_itl_metrics_two_values() {
        let metrics = ItlMetrics::from_measurements(&[10.0, 20.0]);
        assert_eq!(metrics.median_ms, 15.0);
        assert!(metrics.std_dev_ms > 0.0);
    }

    #[test]
    fn test_convoy_test_result_empty_hol() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 10];
        let convoy = vec![11.0; 10];
        let hol: Vec<f64> = vec![];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);
        assert_eq!(result.avg_hol_blocking_ms, 0.0);
        assert_eq!(result.max_hol_blocking_ms, 0.0);
    }

    #[test]
    fn test_convoy_test_result_zero_baseline() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![0.0; 10];
        let convoy = vec![10.0; 10];
        let hol = vec![50.0; 10];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);
        assert_eq!(result.p99_increase_pct, 0.0);
    }

    #[test]
    fn test_saturation_test_result_empty_data() {
        let config = SaturationTestConfig::default();
        let result = SaturationTestResult::new(&config, &[], &[], &[], &[]);

        assert_eq!(result.baseline_throughput, 0.0);
        assert_eq!(result.stressed_throughput, 0.0);
        assert_eq!(result.throughput_degradation_pct, 0.0);
    }

    #[test]
    fn test_saturation_test_result_zero_baseline() {
        let config = SaturationTestConfig::default();
        let result =
            SaturationTestResult::new(&config, &[0.0; 10], &[50.0; 10], &[0.0; 10], &[10.0; 10]);

        assert_eq!(result.throughput_degradation_pct, 0.0);
        assert_eq!(result.p99_increase_pct, 0.0);
    }

    #[test]
    fn test_benchmark_comparison_zero_baselines() {
        let baseline = create_test_full_result("baseline", 0.0, 0.0, 0, 0.0);
        let current = create_test_full_result("current", 30.0, 140.0, 1200, 0.04);

        let comparison = BenchmarkComparison::compare(&baseline, &current);
        assert_eq!(comparison.ttft_p99_change_pct, 0.0);
        assert_eq!(comparison.throughput_change_pct, 0.0);
        assert_eq!(comparison.memory_change_pct, 0.0);
        assert_eq!(comparison.energy_change_pct, 0.0);
    }

    #[test]
    fn test_regression_result_zero_baselines() {
        let baseline = create_test_full_result("test", 0.0, 0.0, 0, 0.0);
        let current = create_test_full_result("test", 30.0, 140.0, 1200, 0.04);

        let regression = RegressionResult::check(&baseline, &current, 5.0);
        // No regression detected because baseline is zero
        assert!(!regression.regression_detected);
    }

    #[test]
    fn test_benchmark_result_zero_tokens() {
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
            ttft_ms: vec![20.0],
            itl_ms: vec![10.0],
            generation_tok_s: vec![140.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 0,
            actual_iterations: 100,
            cv_at_stop: 0.04,
            timestamp: 12345,
        };

        let summary = result.summary();
        assert_eq!(summary.token_joules, 0.0);
    }

    #[test]
    fn test_kv_cache_used_more_than_allocated() {
        // Edge case: used > allocated (shouldn't happen but test boundary)
        let metrics = KvCacheMetrics::new(1000, 1500);
        // saturating_sub gives 0 waste
        assert_eq!(metrics.fragmentation_pct, 0.0);
    }

    #[test]
    fn test_softmax_single_value() {
        let probs = softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-5.0, -3.0, -1.0, 0.0, 1.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Should still be monotonic
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_full_benchmark_result_invalid_thermal() {
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
            ttft_ms: vec![20.0],
            itl_ms: vec![10.0],
            generation_tok_s: vec![140.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 100,
            cv_at_stop: 0.04,
            timestamp: 12345,
        };

        // High variance temps that should be invalid
        let temps = vec![60.0, 70.0, 80.0, 65.0, 85.0];
        let full_result = FullBenchmarkResult::from_benchmark_result(
            &result,
            HardwareSpec::default(),
            &temps,
            0.03,
        );

        assert!(!full_result.thermal.valid);
        assert!(full_result.thermal.temp_variance_c > 2.0);
    }

    #[test]
    fn test_benchmark_comparison_baseline_wins() {
        let baseline = create_test_full_result("baseline", 25.0, 160.0, 1000, 0.03);
        let current = create_test_full_result("current", 40.0, 100.0, 1500, 0.06);

        let comparison = BenchmarkComparison::compare(&baseline, &current);
        assert_eq!(comparison.winner, "baseline");
    }

    #[test]
    fn test_thermal_validity_debug() {
        let valid = ThermalValidity::Valid;
        let invalid = ThermalValidity::Invalid("test".to_string());
        // Test Debug derive
        assert!(format!("{valid:?}").contains("Valid"));
        assert!(format!("{invalid:?}").contains("Invalid"));
    }

    #[test]
    fn test_quality_result_debug() {
        let pass = QualityResult::Pass {
            kl_divergence: 0.01,
        };
        let fail = QualityResult::Fail {
            kl_divergence: 0.5,
            threshold: 0.1,
            message: "test",
        };
        // Test Debug derive
        assert!(format!("{pass:?}").contains("Pass"));
        assert!(format!("{fail:?}").contains("Fail"));
    }

    #[test]
    fn test_workload_type_equality() {
        assert_eq!(WorkloadType::ShortQa, WorkloadType::ShortQa);
        assert_eq!(WorkloadType::LongContext, WorkloadType::LongContext);
        assert_ne!(WorkloadType::ShortQa, WorkloadType::LongContext);
    }

    #[test]
    fn test_workload_type_copy() {
        let wt = WorkloadType::ShortQa;
        let wt_copy = wt;
        assert_eq!(wt, wt_copy);
    }

    // ========================================================================
    // BENCH-002: RuntimeBackend trait tests (TDD RED -> GREEN)
    // ========================================================================

    #[test]
    fn test_runtime_type_display() {
        assert_eq!(RuntimeType::Realizar.as_str(), "realizar");
        assert_eq!(RuntimeType::LlamaCpp.as_str(), "llama-cpp");
        assert_eq!(RuntimeType::Vllm.as_str(), "vllm");
        assert_eq!(RuntimeType::Ollama.as_str(), "ollama");
    }

    #[test]
    fn test_runtime_type_from_str() {
        assert_eq!(RuntimeType::parse("realizar"), Some(RuntimeType::Realizar));
        assert_eq!(RuntimeType::parse("llama-cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llama.cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("vllm"), Some(RuntimeType::Vllm));
        assert_eq!(RuntimeType::parse("ollama"), Some(RuntimeType::Ollama));
        assert_eq!(RuntimeType::parse("unknown"), None);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_request_default() {
        let req = InferenceRequest::default();
        assert_eq!(req.prompt, "");
        assert_eq!(req.max_tokens, 100);
        assert!(req.temperature > 0.0);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_request_builder() {
        let req = InferenceRequest::new("Hello, world!")
            .with_max_tokens(50)
            .with_temperature(0.5);
        assert_eq!(req.prompt, "Hello, world!");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 0.5).abs() < 0.001);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_response_tokens_per_second() {
        let response = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 50.0,
            total_time_ms: 1000.0,
            itl_ms: vec![10.0, 10.0, 10.0],
        };
        assert!((response.tokens_per_second() - 100.0).abs() < 0.1);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_inference_response_tokens_per_second_zero_time() {
        let response = InferenceResponse {
            text: String::new(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: 0.0,
            itl_ms: vec![],
        };
        assert_eq!(response.tokens_per_second(), 0.0);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_mock_backend_inference() {
        let backend = MockBackend::new(42.0, 150.0);
        let req = InferenceRequest::new("test prompt");
        let response = backend.inference(&req);

        assert!(response.is_ok());
        let resp = response.expect("test");
        assert!((resp.ttft_ms - 42.0).abs() < 0.001);
        assert!(resp.tokens_generated > 0);
    }
