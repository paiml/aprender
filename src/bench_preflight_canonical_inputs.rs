
    // =========================================================================
    // Canonical Inputs Tests
    // =========================================================================

    #[test]
    fn test_canonical_inputs_version_is_semver() {
        let version = canonical_inputs::VERSION;
        let parts: Vec<&str> = version.split('.').collect();
        assert_eq!(
            parts.len(),
            3,
            "Version should be semver (major.minor.patch)"
        );
        for part in parts {
            assert!(
                part.parse::<u32>().is_ok(),
                "Version part '{}' should be numeric",
                part
            );
        }
    }

    #[test]
    fn test_canonical_inputs_prompt_not_empty() {
        // Verify prompt has reasonable length for benchmarking
        let prompt_len = canonical_inputs::LATENCY_PROMPT.len();
        assert!(
            prompt_len >= 10,
            "Latency prompt should have at least 10 chars, got {}",
            prompt_len
        );
    }

    #[test]
    fn test_canonical_inputs_tokens_not_empty() {
        // Verify we have enough tokens for throughput testing
        let token_count = canonical_inputs::THROUGHPUT_TOKENS.len();
        assert!(
            token_count >= 4,
            "Throughput tokens should have at least 4 tokens, got {}",
            token_count
        );
    }

    #[test]
    fn test_canonical_inputs_max_tokens_reasonable() {
        // Verify max tokens is in a sensible range
        let max_tokens = canonical_inputs::MAX_TOKENS;
        assert!(
            max_tokens > 0,
            "Max tokens should be positive, got {}",
            max_tokens
        );
        assert!(
            max_tokens <= 1000,
            "Max tokens should be <= 1000, got {}",
            max_tokens
        );
    }

    // =========================================================================
    // DeterministicInferenceConfig Tests
    // =========================================================================

    #[test]
    fn test_deterministic_config_default_is_deterministic() {
        let config = DeterministicInferenceConfig::default();
        assert!(
            config.validate_determinism().is_ok(),
            "Default config should be deterministic"
        );
    }

    #[test]
    fn test_deterministic_config_default_values() {
        let config = DeterministicInferenceConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.seed, 42);
        assert_eq!(config.top_k, 1);
        assert_eq!(config.top_p, 1.0);
    }

    #[test]
    fn test_deterministic_config_with_seed() {
        let config = DeterministicInferenceConfig::with_seed(12345);
        assert_eq!(config.seed, 12345);
        assert!(config.validate_determinism().is_ok());
    }

    #[test]
    fn test_deterministic_config_rejects_nonzero_temperature() {
        let config = DeterministicInferenceConfig {
            temperature: 0.7,
            ..Default::default()
        };
        let result = config.validate_determinism();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PreflightError::ConfigError { .. }));
    }

    #[test]
    fn test_deterministic_config_rejects_topk_not_one() {
        let config = DeterministicInferenceConfig {
            top_k: 50,
            ..Default::default()
        };
        let result = config.validate_determinism();
        assert!(result.is_err());
    }

    // =========================================================================
    // CvStoppingCriterion Tests
    // =========================================================================

    #[test]
    fn test_cv_criterion_default_values() {
        let criterion = CvStoppingCriterion::default();
        assert_eq!(criterion.min_samples, 5);
        assert_eq!(criterion.max_samples, 30);
        assert!((criterion.cv_threshold - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_cv_criterion_continues_below_min_samples() {
        let criterion = CvStoppingCriterion::new(5, 30, 0.05);
        let samples = vec![100.0, 100.0, 100.0]; // Only 3 samples
        assert_eq!(criterion.should_stop(&samples), StopDecision::Continue);
    }

    #[test]
    fn test_cv_criterion_stops_at_max_samples() {
        let criterion = CvStoppingCriterion::new(5, 10, 0.01); // Very tight CV
        let samples: Vec<f64> = (1..=10).map(|x| x as f64 * 10.0).collect();
        assert_eq!(
            criterion.should_stop(&samples),
            StopDecision::Stop(StopReason::MaxSamples)
        );
    }

    #[test]
    fn test_cv_criterion_converges_on_identical_values() {
        let criterion = CvStoppingCriterion::new(5, 30, 0.05);
        let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let cv = criterion.calculate_cv(&samples);
        assert!(
            cv < 0.001,
            "CV of identical values should be ~0, got {}",
            cv
        );

        match criterion.should_stop(&samples) {
            StopDecision::Stop(StopReason::CvConverged(cv)) => {
                assert!(cv < 0.05);
            },
            other => panic!("Expected CvConverged, got {:?}", other),
        }
    }

    #[test]
    fn test_cv_criterion_continues_on_high_variance() {
        let criterion = CvStoppingCriterion::new(5, 30, 0.05);
        // High variance: 10, 100, 10, 100, 10 - CV >> 0.05
        let samples = vec![10.0, 100.0, 10.0, 100.0, 10.0];
        assert_eq!(criterion.should_stop(&samples), StopDecision::Continue);
    }

    #[test]
    fn test_cv_calculation_single_value() {
        let criterion = CvStoppingCriterion::default();
        let samples = vec![100.0];
        let cv = criterion.calculate_cv(&samples);
        assert_eq!(cv, f64::MAX);
    }

    #[test]
    fn test_cv_calculation_empty() {
        let criterion = CvStoppingCriterion::default();
        let samples: Vec<f64> = vec![];
        let cv = criterion.calculate_cv(&samples);
        assert_eq!(cv, f64::MAX);
    }

    #[test]
    fn test_cv_calculation_known_values() {
        let criterion = CvStoppingCriterion::default();
        // Mean = 100, values deviate by ~7.9, so CV ~0.079
        let samples = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let cv = criterion.calculate_cv(&samples);
        assert!(cv > 0.07 && cv < 0.09, "Expected CV ~0.079, got {}", cv);
    }

    // =========================================================================
    // OutlierDetector Tests
    // =========================================================================

    #[test]
    fn test_outlier_detector_default_k_factor() {
        let detector = OutlierDetector::default();
        assert!((detector.k_factor - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_outlier_detector_no_outliers_uniform() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 101.0, 99.0, 100.5, 99.5];
        let outliers = detector.detect(&samples);
        assert!(
            !outliers.iter().any(|&x| x),
            "Uniform samples should have no outliers"
        );
    }

    #[test]
    fn test_outlier_detector_finds_extreme_outlier() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 101.0, 99.0, 100.0, 1000.0]; // 1000 is extreme
        let outliers = detector.detect(&samples);
        assert!(outliers[4], "1000.0 should be detected as outlier");
    }

    #[test]
    fn test_outlier_detector_filter_removes_outliers() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 101.0, 99.0, 100.0, 1000.0];
        let filtered = detector.filter(&samples);
        assert!(
            !filtered.contains(&1000.0),
            "Filtered should not contain outlier"
        );
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_outlier_detector_handles_small_samples() {
        let detector = OutlierDetector::default();
        let samples = vec![100.0, 200.0]; // Only 2 samples
        let outliers = detector.detect(&samples);
        assert_eq!(
            outliers,
            vec![false, false],
            "Should not detect outliers with < 3 samples"
        );
    }

    #[test]
    fn test_outlier_detector_percentile() {
        // Uses nearest-rank method: idx = round((p/100) * (n-1))
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // p50 of 10 elements: idx = round(0.5 * 9) = round(4.5) = 5 → value 6.0
        let p50 = OutlierDetector::percentile(&samples, 50.0);
        assert!(
            (p50 - 5.5).abs() < 1.0,
            "p50 should be ~5.5 (nearest rank gives 6), got {}",
            p50
        );

        // p99: idx = round(0.99 * 9) = round(8.91) = 9 → value 10.0
        let p99 = OutlierDetector::percentile(&samples, 99.0);
        assert!((p99 - 10.0).abs() < 0.5, "p99 should be ~10.0, got {}", p99);
    }

    // =========================================================================
    // QualityMetrics Tests
    // =========================================================================

    #[test]
    fn test_quality_metrics_default() {
        let metrics = QualityMetrics::default();
        assert_eq!(metrics.cv_at_stop, f64::MAX);
        assert!(!metrics.cv_converged);
        assert_eq!(metrics.outliers_detected, 0);
        assert!(metrics.preflight_checks_passed.is_empty());
    }

    #[test]
    fn test_quality_metrics_serialization() {
        let metrics = QualityMetrics {
            cv_at_stop: 0.03,
            cv_converged: true,
            outliers_detected: 2,
            outliers_excluded: 1,
            preflight_checks_passed: vec!["server_check".to_string()],
        };
        let json = serde_json::to_string(&metrics).expect("serialization");
        assert!(json.contains("0.03"));
        assert!(json.contains("server_check"));
    }

    // =========================================================================
    // DeterminismCheck Tests
    // =========================================================================

    #[test]
    fn test_determinism_check_trait_impl() {
        let config = DeterministicInferenceConfig::default();
        let check = DeterminismCheck::new(config);
        assert_eq!(check.name(), "determinism_check");
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_determinism_check_fails_on_bad_config() {
        let config = DeterministicInferenceConfig {
            temperature: 0.5,
            ..Default::default()
        };
        let check = DeterminismCheck::new(config);
        assert!(check.validate().is_err());
    }

    // =========================================================================
    // PreflightError Tests
    // =========================================================================

    #[test]
    fn test_preflight_error_display() {
        let err = PreflightError::ModelNotFound {
            requested: "phi".to_string(),
            available: vec!["phi2:2.7b".to_string(), "llama2".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("phi"));
        assert!(msg.contains("phi2:2.7b"));
    }

    #[test]
    fn test_preflight_error_schema_mismatch() {
        let err = PreflightError::SchemaMismatch {
            missing_field: "eval_count".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("eval_count"));
    }

    #[test]
    fn test_preflight_error_type_mismatch() {
        let err = PreflightError::FieldTypeMismatch {
            field: "tokens".to_string(),
            expected: "number".to_string(),
            actual: "string".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("tokens"));
        assert!(msg.contains("number"));
    }

    // =========================================================================
    // ServerAvailabilityCheck Tests
    // =========================================================================

    #[test]
    fn test_server_check_llama_cpp_defaults() {
        let check = ServerAvailabilityCheck::llama_cpp(8082);
        assert_eq!(check.health_url(), "http://127.0.0.1:8082/health");
        assert_eq!(check.name(), "server_availability_check");
    }

    #[test]
    fn test_server_check_ollama_defaults() {
        let check = ServerAvailabilityCheck::ollama(11434);
        assert_eq!(check.health_url(), "http://127.0.0.1:11434/api/tags");
    }

    #[test]
    fn test_server_check_validates_url_format() {
        let check = ServerAvailabilityCheck::new("invalid-url".to_string(), "/health".to_string());
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_server_check_rejects_empty_url() {
        let check = ServerAvailabilityCheck::new(String::new(), "/health".to_string());
        let result = check.validate();
        assert!(matches!(result, Err(PreflightError::ConfigError { .. })));
    }

    #[test]
    fn test_server_check_requires_health_status() {
        let check = ServerAvailabilityCheck::llama_cpp(8082);
        // No health status set yet
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_server_check_accepts_200_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(200);
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_server_check_accepts_204_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(204); // No Content is valid
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_server_check_rejects_500_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(500);
        let result = check.validate();
        assert!(matches!(
            result,
            Err(PreflightError::HealthCheckFailed { status: 500, .. })
        ));
    }

    #[test]
    fn test_server_check_rejects_404_status() {
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);
        check.set_health_status(404);
        let result = check.validate();
        assert!(result.is_err());
    }

    // =========================================================================
    // ModelAvailabilityCheck Tests
    // =========================================================================

    #[test]
    fn test_model_check_finds_exact_match() {
        let mut check = ModelAvailabilityCheck::new("phi2:2.7b".to_string());
        check.set_available_models(vec!["phi2:2.7b".to_string(), "llama2".to_string()]);
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_model_check_finds_partial_match() {
        let mut check = ModelAvailabilityCheck::new("phi2".to_string());
        check.set_available_models(vec!["phi2:2.7b".to_string(), "llama2".to_string()]);
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_model_check_fails_on_missing_model() {
        let mut check = ModelAvailabilityCheck::new("gpt4".to_string());
        check.set_available_models(vec!["phi2:2.7b".to_string(), "llama2".to_string()]);
        let result = check.validate();
        assert!(matches!(result, Err(PreflightError::ModelNotFound { .. })));
    }
