
    #[cfg(feature = "bench-http")]
    #[test]
    fn test_mock_backend_info() {
        let backend = MockBackend::new(30.0, 140.0);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Realizar);
        assert!(!info.version.is_empty());
        assert!(info.supports_streaming);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::new();
        assert!(registry.get(RuntimeType::Realizar).is_none());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_backend_registry_register_and_get() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(MockBackend::new(30.0, 140.0));
        registry.register(RuntimeType::Realizar, backend);

        assert!(registry.get(RuntimeType::Realizar).is_some());
        assert!(registry.get(RuntimeType::LlamaCpp).is_none());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_backend_registry_list() {
        let mut registry = BackendRegistry::new();
        registry.register(
            RuntimeType::Realizar,
            Box::new(MockBackend::new(30.0, 140.0)),
        );
        registry.register(
            RuntimeType::LlamaCpp,
            Box::new(MockBackend::new(35.0, 130.0)),
        );

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&RuntimeType::Realizar));
        assert!(list.contains(&RuntimeType::LlamaCpp));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_config_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.binary_path, "llama-cli");
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.ctx_size, 2048);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_config_builder() {
        let config = LlamaCppConfig::new("/usr/bin/llama-cli")
            .with_model("/models/test.gguf")
            .with_gpu_layers(32)
            .with_ctx_size(4096);

        assert_eq!(config.binary_path, "/usr/bin/llama-cli");
        assert_eq!(config.model_path, Some("/models/test.gguf".to_string()));
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.ctx_size, 4096);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.base_url, "http://localhost:8000");
        assert_eq!(config.api_version, "v1");
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_config_builder() {
        let config = VllmConfig::new("http://gpu-server:8080")
            .with_model("meta-llama/Llama-2-7b")
            .with_api_key("test-key");

        assert_eq!(config.base_url, "http://gpu-server:8080");
        assert_eq!(config.model, Some("meta-llama/Llama-2-7b".to_string()));
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }

    // =========================================================================
    // LlamaCppBackend Tests (BENCH-002: Runtime Backend Integration)
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_creation() {
        let config = LlamaCppConfig::new("llama-cli");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.version.is_empty());
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_info() {
        let config = LlamaCppConfig::new("llama-cli").with_model("test.gguf");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.supports_streaming); // CLI doesn't support streaming
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_missing_binary() {
        let config = LlamaCppConfig::new("/nonexistent/llama-cli");
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        // Should return error for missing binary
        assert!(result.is_err());
    }

    // =========================================================================
    // VllmBackend Tests (BENCH-003: HTTP Client Integration)
    // Requires bench-http feature for HTTP client
    // =========================================================================

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_creation() {
        let config = VllmConfig::new("http://localhost:8000");
        let backend = VllmBackend::new(config);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Vllm);
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_info() {
        let config = VllmConfig::new("http://localhost:8000").with_model("meta-llama/Llama-2-7b");
        let backend = VllmBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::Vllm);
        assert!(info.supports_streaming); // vLLM supports streaming
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_vllm_backend_connection_error() {
        let config = VllmConfig::new("http://localhost:99999"); // Invalid port
        let backend = VllmBackend::new(config);
        let request = InferenceRequest::new("test");
        let result = backend.inference(&request);

        // Should return error for connection failure
        assert!(result.is_err());
    }

    // =========================================================================
    // BENCH-004: MeasurementProtocol Tests (TDD RED)
    // =========================================================================

    #[test]
    fn test_measurement_protocol_default() {
        let protocol = MeasurementProtocol::default();
        assert_eq!(protocol.latency_samples, 100);
        assert_eq!(
            protocol.latency_percentiles,
            vec![50.0, 90.0, 95.0, 99.0, 99.9]
        );
        assert_eq!(protocol.throughput_duration.as_secs(), 60);
        assert_eq!(protocol.throughput_ramp_up.as_secs(), 10);
        assert_eq!(protocol.memory_samples, 10);
    }

    #[test]
    fn test_measurement_protocol_builder() {
        let protocol = MeasurementProtocol::new()
            .with_latency_samples(200)
            .with_percentiles(vec![50.0, 95.0, 99.0])
            .with_throughput_duration(Duration::from_secs(120))
            .with_memory_samples(20);

        assert_eq!(protocol.latency_samples, 200);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 95.0, 99.0]);
        assert_eq!(protocol.throughput_duration.as_secs(), 120);
        assert_eq!(protocol.memory_samples, 20);
    }

    // =========================================================================
    // BENCH-005: LatencyStatistics Tests (TDD RED)
    // =========================================================================

    #[test]
    fn test_latency_statistics_from_samples() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 5);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.mean, Duration::from_millis(30));
    }

    #[test]
    fn test_latency_statistics_percentiles() {
        // 100 samples from 1ms to 100ms
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        // p50 should be around 50ms
        assert!(stats.p50 >= Duration::from_millis(49));
        assert!(stats.p50 <= Duration::from_millis(51));

        // p95 should be around 95ms
        assert!(stats.p95 >= Duration::from_millis(94));
        assert!(stats.p95 <= Duration::from_millis(96));

        // p99 should be around 99ms
        assert!(stats.p99 >= Duration::from_millis(98));
        assert!(stats.p99 <= Duration::from_millis(100));
    }

    #[test]
    fn test_latency_statistics_confidence_interval() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        // 95% CI should contain the mean
        let (lower, upper) = stats.confidence_interval_95;
        assert!(lower < stats.mean);
        assert!(upper > stats.mean);
    }

    #[test]
    fn test_latency_statistics_std_dev() {
        // Uniform samples should have non-zero std dev
        let samples: Vec<Duration> = (1..=10).map(|i| Duration::from_millis(i * 10)).collect();
        let stats = LatencyStatistics::from_samples(&samples);

        assert!(stats.std_dev > Duration::ZERO);
    }

    // ==========================================
    // BENCH-006: OutlierDetector Tests (MAD-based)
    // ==========================================

    #[test]
    fn test_outlier_detector_no_outliers() {
        // Normal distribution with no outliers
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3];
        let outliers = detect_outliers(&samples, 3.5); // Standard threshold
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_outlier_detector_single_outlier() {
        // One clear outlier at position 8 (value 100.0)
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3, 100.0];
        let outliers = detect_outliers(&samples, 3.5);
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 8);
    }

    #[test]
    fn test_outlier_detector_multiple_outliers() {
        // Two outliers: one high, one low
        let samples = vec![0.1, 10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 100.0];
        let outliers = detect_outliers(&samples, 3.5);
        assert_eq!(outliers.len(), 2);
        assert!(outliers.contains(&0)); // 0.1 is an outlier
        assert!(outliers.contains(&8)); // 100.0 is an outlier
    }

    #[test]
    fn test_outlier_detector_threshold_sensitivity() {
        // Lower threshold should catch more outliers
        let samples = vec![10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 15.0];
        let strict_outliers = detect_outliers(&samples, 2.0);
        let lenient_outliers = detect_outliers(&samples, 5.0);
        assert!(strict_outliers.len() >= lenient_outliers.len());
    }

    // ==========================================
    // BENCH-007: RegressionDetector Tests
    // ==========================================

    #[test]
    fn test_regression_detector_default() {
        let detector = RegressionDetector::default();
        assert_eq!(detector.warning_threshold, 0.02); // 2%
        assert_eq!(detector.failure_threshold, 0.05); // 5%
    }

    #[test]
    fn test_regression_detector_no_regression() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 101.0, // 1% increase - within warning
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert!(report.regressions.is_empty());
    }

    #[test]
    fn test_regression_detector_warning() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 103.0, // 3% increase - warning
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed); // Warnings don't fail
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn test_regression_detector_failure() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 110.0, // 10% increase - failure
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(!report.passed);
        assert_eq!(report.regressions.len(), 1);
    }

    #[test]
    fn test_regression_detector_improvement() {
        let baseline = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 100.0,
            std_dev: 5.0,
            samples: 100,
        };
        let current = BenchmarkMetrics {
            name: "latency".to_string(),
            mean: 90.0, // 10% decrease - improvement!
            std_dev: 5.0,
            samples: 100,
        };
        let detector = RegressionDetector::default();
        let report = detector.compare(&baseline, &current);
        assert!(report.passed);
        assert_eq!(report.improvements.len(), 1);
    }

    // ==========================================
    // BENCH-008: Welch's t-test Tests
    // ==========================================

    #[test]
    fn test_welch_t_test_result_fields() {
        // Verify result struct has all required fields
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let sample_b = vec![20.0, 21.0, 20.5, 20.2, 20.8];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // Result should have t_statistic, degrees_of_freedom, p_value, significant
        assert!(result.t_statistic.is_finite());
        assert!(result.degrees_of_freedom > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // These are clearly different - should be significant
        assert!(result.significant);
    }

    #[test]
    fn test_welch_t_test_identical_samples() {
        // Identical samples should NOT be significant
        let sample_a = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let sample_b = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        assert!(!result.significant);
        assert!(result.t_statistic.abs() < 1e-10 || result.p_value > 0.05);
    }

    #[test]
    fn test_welch_t_test_clearly_different() {
        // Clearly different samples should be significant
        let sample_a = vec![10.0, 11.0, 10.5, 10.2, 10.8, 10.3, 10.7, 10.1];
        let sample_b = vec![50.0, 51.0, 50.5, 50.2, 50.8, 50.3, 50.7, 50.1];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        assert!(result.significant);
        assert!(result.p_value < 0.001); // Very significant
    }

    #[test]
    fn test_welch_t_test_unequal_variance() {
        // Welch's t-test handles unequal variances correctly
        let sample_a = vec![10.0, 10.1, 10.0, 10.1, 10.0]; // Low variance
        let sample_b = vec![10.0, 15.0, 5.0, 20.0, 0.0]; // High variance, same mean
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // Same mean, different variance - should NOT be significant
        assert!(!result.significant);
    }

    #[test]
    fn test_welch_t_test_small_samples() {
        // Small samples require larger differences
        let sample_a = vec![10.0, 11.0, 12.0];
        let sample_b = vec![12.0, 13.0, 14.0];
        let result = welch_t_test(&sample_a, &sample_b, 0.05);
        // With only 3 samples each, difference may not be significant
        assert!(result.degrees_of_freedom > 0.0);
    }
