
    // ========================================================================
    // DynamicSampler Tests
    // ========================================================================

    #[test]
    fn test_dynamic_sampler_continues_until_min_samples() {
        let mut dyn_sampler = DynamicSampler::new(100, 10_000, 0.05);
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        assert!(dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_stops_at_max_samples() {
        let mut dyn_sampler = DynamicSampler::new(10, 100, 0.05);

        // Generate 100 data points with high variance
        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64 * 10.0).collect();

        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_stops_when_cv_stable() {
        let mut dyn_sampler = DynamicSampler::new(10, 10_000, 0.05);
        dyn_sampler.stability_count = 1; // Stop after 1 stable check

        // Generate 100 data points with very low variance (CV ~= 0)
        let data: Vec<f64> = vec![100.0; 100];

        // Should stop because CV = 0 < 0.05
        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_requires_stability_streak() {
        let mut dyn_sampler = DynamicSampler::new(10, 10_000, 0.05);
        dyn_sampler.stability_count = 3;

        // Stable data points
        let data: Vec<f64> = vec![100.0; 100];

        // First check - streak = 1
        assert!(dyn_sampler.should_continue(&data));
        // Second check - streak = 2
        assert!(dyn_sampler.should_continue(&data));
        // Third check - streak = 3, should stop
        assert!(!dyn_sampler.should_continue(&data));
    }

    #[test]
    fn test_dynamic_sampler_reset() {
        let mut sampler = DynamicSampler::new(10, 10_000, 0.05);
        sampler.stable_streak = 5;
        sampler.reset();
        assert_eq!(sampler.stable_streak, 0);
    }

    #[test]
    fn test_compute_cv_constant_values() {
        let data = vec![100.0; 50];
        let cv = compute_cv(&data);
        assert!(cv.abs() < 1e-10, "CV of constant values should be ~0");
    }

    #[test]
    fn test_compute_cv_varied_values() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let cv = compute_cv(&data);
        // CV = std_dev / mean = 15.81 / 30 ~= 0.527
        assert!(cv > 0.5 && cv < 0.6, "CV should be ~0.527, got {cv}");
    }

    #[test]
    fn test_compute_cv_empty_data() {
        let data: Vec<f64> = vec![];
        let cv = compute_cv(&data);
        assert!(cv.is_infinite());
    }

    // ========================================================================
    // ThermalGuard Tests
    // ========================================================================

    #[test]
    fn test_thermal_guard_valid_low_variance() {
        let guard = ThermalGuard::default();
        let temps = vec![75.0, 75.5, 74.8, 75.2, 75.1];

        assert_eq!(guard.validate_run(&temps), ThermalValidity::Valid);
    }

    #[test]
    fn test_thermal_guard_invalid_high_variance() {
        let guard = ThermalGuard::default();
        // Variance std_dev > 2°C
        let temps = vec![70.0, 75.0, 80.0, 72.0, 78.0];

        match guard.validate_run(&temps) {
            ThermalValidity::Invalid(msg) => {
                assert!(msg.contains("exceeds threshold"));
            },
            ThermalValidity::Valid => panic!("Expected Invalid"),
        }
    }

    #[test]
    fn test_thermal_guard_empty_temps() {
        let guard = ThermalGuard::default();
        assert_eq!(guard.validate_run(&[]), ThermalValidity::Valid);
    }

    #[test]
    fn test_thermal_guard_max_temp() {
        let guard = ThermalGuard::default();
        let temps = vec![70.0, 75.0, 85.0, 72.0];
        assert_eq!(guard.max_temp(&temps), 85.0);
    }

    // ========================================================================
    // KvCacheMetrics Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_metrics_no_waste() {
        let metrics = KvCacheMetrics::new(1000, 1000);
        assert_eq!(metrics.fragmentation_pct, 0.0);
        assert!(metrics.is_acceptable(10.0));
    }

    #[test]
    fn test_kv_cache_metrics_with_waste() {
        let metrics = KvCacheMetrics::new(1000, 800);
        assert!((metrics.fragmentation_pct - 20.0).abs() < 0.01);
        assert!(!metrics.is_acceptable(10.0));
        assert!(metrics.is_acceptable(25.0));
    }

    #[test]
    fn test_kv_cache_metrics_zero_allocated() {
        let metrics = KvCacheMetrics::new(0, 0);
        assert_eq!(metrics.fragmentation_pct, 0.0);
    }

    #[test]
    fn test_kv_cache_metrics_mb_conversion() {
        let metrics = KvCacheMetrics::new(1024 * 1024 * 100, 1024 * 1024 * 80);
        assert!((metrics.allocated_mb() - 100.0).abs() < 0.01);
        assert!((metrics.used_mb() - 80.0).abs() < 0.01);
    }

    // ========================================================================
    // EnergyMetrics Tests
    // ========================================================================

    #[test]
    fn test_energy_metrics_joules_per_token() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 1000);
        assert!((metrics.joules_per_token() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_energy_metrics_zero_tokens() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 0);
        assert_eq!(metrics.joules_per_token(), 0.0);
    }

    #[test]
    fn test_energy_metrics_tokens_per_joule() {
        let metrics = EnergyMetrics::new(100.0, 10.0, 50.0, 1000);
        assert!((metrics.tokens_per_joule() - 10.0).abs() < 0.001);
    }

    // ========================================================================
    // ItlMetrics Tests
    // ========================================================================

    #[test]
    fn test_itl_metrics_from_measurements() {
        let itl = vec![10.0, 12.0, 11.0, 15.0, 13.0, 14.0, 11.0, 12.0, 13.0, 10.0];
        let metrics = ItlMetrics::from_measurements(&itl);

        // Median should be around 12
        assert!(metrics.median_ms > 11.0 && metrics.median_ms < 13.0);
        // Std dev should be small
        assert!(metrics.std_dev_ms < 5.0);
        // p99 should be around 15
        assert!(metrics.p99_ms >= 14.0);
    }

    #[test]
    fn test_itl_metrics_empty() {
        let metrics = ItlMetrics::from_measurements(&[]);
        assert_eq!(metrics.median_ms, 0.0);
        assert_eq!(metrics.std_dev_ms, 0.0);
    }

    #[test]
    fn test_itl_metrics_low_jitter() {
        let itl = vec![10.0; 100];
        let metrics = ItlMetrics::from_measurements(&itl);
        assert!(metrics.is_low_jitter(1.0));
    }

    #[test]
    fn test_itl_metrics_high_jitter() {
        let itl: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let metrics = ItlMetrics::from_measurements(&itl);
        assert!(!metrics.is_low_jitter(5.0));
    }

    // ========================================================================
    // KL-Divergence Tests
    // ========================================================================

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_quantization_quality(&logits, &logits, 0.01);

        match result {
            QualityResult::Pass { kl_divergence } => {
                assert!(kl_divergence < 1e-10, "KL should be ~0 for identical");
            },
            QualityResult::Fail { .. } => panic!("Expected Pass for identical"),
        }
    }

    #[test]
    fn test_kl_divergence_slightly_different() {
        let fp32 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quant = vec![1.01, 2.01, 3.01, 4.01, 5.01];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);

        match result {
            QualityResult::Pass { kl_divergence } => {
                assert!(kl_divergence < 0.001, "KL should be very small");
            },
            QualityResult::Fail { .. } => panic!("Expected Pass for small diff"),
        }
    }

    #[test]
    fn test_kl_divergence_very_different() {
        let fp32 = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let quant = vec![0.0, 0.0, 0.0, 0.0, 10.0];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);

        match result {
            QualityResult::Fail { kl_divergence, .. } => {
                assert!(kl_divergence > 1.0, "KL should be large for opposite");
            },
            QualityResult::Pass { .. } => panic!("Expected Fail for very different"),
        }
    }

    #[test]
    fn test_kl_divergence_mismatched_lengths() {
        let fp32 = vec![1.0, 2.0, 3.0];
        let quant = vec![1.0, 2.0];

        let result = validate_quantization_quality(&fp32, &quant, 0.01);
        assert!(matches!(result, QualityResult::Fail { .. }));
    }

    #[test]
    fn test_kl_divergence_empty() {
        let result = validate_quantization_quality(&[], &[], 0.01);
        assert!(matches!(result, QualityResult::Pass { .. }));
    }

    // ========================================================================
    // BenchmarkResult Tests
    // ========================================================================

    #[test]
    fn test_benchmark_result_summary() {
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
            ttft_ms: vec![20.0, 22.0, 21.0, 25.0, 23.0, 24.0, 22.0, 21.0, 20.0, 26.0],
            itl_ms: vec![10.0, 11.0, 10.5, 11.5, 10.2, 10.8, 11.2, 10.3, 10.7, 11.0],
            generation_tok_s: vec![140.0, 142.0, 141.0, 143.0, 139.0],
            peak_memory_mb: 1024,
            kv_cache_waste_pct: 3.5,
            energy_joules: 50.0,
            tokens_generated: 1000,
            actual_iterations: 500,
            cv_at_stop: 0.045,
            timestamp: 12345,
        };

        let summary = result.summary();

        // Check percentiles are reasonable
        assert!(summary.ttft_p50 > 20.0 && summary.ttft_p50 < 25.0);
        assert!(summary.ttft_p99 >= summary.ttft_p50);
        assert!(summary.ttft_p999 >= summary.ttft_p99);

        // Check ITL
        assert!(summary.itl_median > 10.0 && summary.itl_median < 12.0);
        assert!(summary.itl_std_dev < 2.0);

        // Check throughput
        assert!(summary.throughput_median > 139.0 && summary.throughput_median < 144.0);

        // Check energy
        assert!((summary.token_joules - 0.05).abs() < 0.001);

        // Check metadata
        assert_eq!(summary.iterations, 500);
        assert!((summary.cv_final - 0.045).abs() < 0.001);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!(percentile(&data, 50.0) >= 5.0 && percentile(&data, 50.0) <= 6.0);
        assert!(percentile(&data, 90.0) >= 9.0);
        assert_eq!(percentile(&data, 100.0), 10.0);
    }

    #[test]
    fn test_bootstrap_ci() {
        let data = vec![100.0; 100];
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);

        // For constant data, CI should be tight around 100
        assert!((lower - 100.0).abs() < 0.01);
        assert!((upper - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // Softmax Tests
    // ========================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_monotonic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);

        // Higher logits should have higher probabilities
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large logits shouldn't overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // ========================================================================
    // WorkloadType Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_workload_type_short_qa() {
        let workload = WorkloadType::ShortQa;
        assert_eq!(workload.input_tokens(), 32);
        assert_eq!(workload.output_tokens(), 64);
    }

    #[test]
    fn test_workload_type_long_context() {
        let workload = WorkloadType::LongContext;
        assert_eq!(workload.input_tokens(), 2048);
        assert_eq!(workload.output_tokens(), 512);
    }

    // ========================================================================
    // ConvoyTestConfig Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_convoy_config_default() {
        let config = ConvoyTestConfig::default();
        assert_eq!(config.long_requests, 10);
        assert_eq!(config.short_requests, 100);
        assert!((config.max_p99_increase_pct - 50.0).abs() < 0.01);
        assert!((config.max_hol_blocking_ms - 500.0).abs() < 0.01);
        assert!((config.max_kv_fragmentation_pct - 15.0).abs() < 0.01);
    }

    // ========================================================================
    // ConvoyTestResult Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_convoy_test_result_pass() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0, 12.0, 11.0, 13.0, 10.5]; // p99 ~= 13
        let convoy = vec![12.0, 14.0, 13.0, 15.0, 12.5]; // p99 ~= 15 (15% increase)
        let hol = vec![50.0, 100.0, 75.0, 80.0, 60.0];
        let kv_frag = 10.0; // 10% < 15% threshold

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(result.passed, "Should pass with acceptable metrics");
        assert!(result.failure_reasons.is_empty());
        assert!(result.p99_increase_pct < 50.0);
        assert!(result.max_hol_blocking_ms < 500.0);
    }

    #[test]
    fn test_convoy_test_result_fail_p99() {
        let config = ConvoyTestConfig::default();
        let baseline = vec![10.0; 100];
        let convoy = vec![20.0; 100]; // 100% increase > 50% threshold
        let hol = vec![50.0; 100];
        let kv_frag = 5.0;

        let result = ConvoyTestResult::new(&config, &baseline, &convoy, &hol, kv_frag);

        assert!(!result.passed, "Should fail with 100% p99 increase");
        assert!(result.failure_reasons.iter().any(|r| r.contains("P99")));
    }
