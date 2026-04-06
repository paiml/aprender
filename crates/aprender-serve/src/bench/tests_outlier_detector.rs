
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
