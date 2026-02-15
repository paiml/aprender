
    #[test]
    fn test_filter_results_no_matching_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![Hotspot {
                name: "custom_op".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 10,
                avg_us: 100.0,
                min_us: 90.0,
                max_us: 110.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        // No attention-related hotspots
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        assert!(filtered.hotspots.is_empty());
    }

    #[test]
    fn test_filter_results_empty_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 100.0,
            throughput_tok_s: 10000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        for focus in [
            ProfileFocus::All,
            ProfileFocus::Attention,
            ProfileFocus::Mlp,
            ProfileFocus::Matmul,
            ProfileFocus::Embedding,
        ] {
            let filtered = filter_results_by_focus(&results, focus);
            assert!(filtered.hotspots.is_empty());
        }
    }

    // ========================================================================
    // CiProfileReport from_results Extended Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_all_assertions_combined() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 20000.0, // 20ms
            throughput_tok_s: 200.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(30.0),
            max_memory_mb: None,
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
        assert_eq!(report.assertions.len(), 3);
        assert!(report.assertions.iter().all(|a| a.passed));
        assert_eq!(report.throughput_tok_s, 200.0);
        // 20000us / 1000 = 20ms
        assert!((report.latency_p50_ms - 20.0).abs() < 0.01);
        assert!((report.latency_p99_ms - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_report_zero_throughput() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        assert!(!report.assertions[0].passed);
    }

    #[test]
    fn test_ci_profile_report_latency_p50_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 60000.0, // 60ms
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            max_p50_ms: Some(50.0), // 60ms > 50ms => fail
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        assert_eq!(report.assertions.len(), 1);
        assert!(!report.assertions[0].passed);
        assert_eq!(report.assertions[0].name, "latency_p50");
    }

    #[test]
    fn test_ci_profile_report_mixed_pass_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 30000.0, // 30ms
            throughput_tok_s: 150.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0), // 150 >= 100 => pass
            max_p99_ms: Some(20.0),      // 30ms > 20ms => fail
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed); // One failure means overall fail
        assert_eq!(report.assertions.len(), 2);
        assert!(report.assertions[0].passed); // throughput passed
        assert!(!report.assertions[1].passed); // latency failed
    }

    #[test]
    fn test_ci_profile_report_assertion_format_strings() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 150.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(25.0),
            max_memory_mb: None,
        };
        let report = CiProfileReport::from_results(&results, &assertions);

        // Check format of assertion strings
        let throughput_assertion = &report.assertions[0];
        assert_eq!(throughput_assertion.name, "throughput");
        assert!(throughput_assertion.expected.contains("tok/s"));
        assert!(throughput_assertion.actual.contains("tok/s"));

        let p99_assertion = &report.assertions[1];
        assert_eq!(p99_assertion.name, "latency_p99");
        assert!(p99_assertion.expected.contains("ms"));
        assert!(p99_assertion.actual.contains("ms"));

        let p50_assertion = &report.assertions[2];
        assert_eq!(p50_assertion.name, "latency_p50");
        assert!(p50_assertion.expected.contains("ms"));
    }

    // ========================================================================
    // DiffBenchmarkReport Tests
    // ========================================================================

    #[test]
    fn test_diff_benchmark_report_construction() {
        let report = DiffBenchmarkReport {
            model_a: "model_a.gguf".to_string(),
            model_b: "model_b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 150.0,
            throughput_delta_pct: 50.0,
            latency_a_ms: 10.0,
            latency_b_ms: 7.0,
            latency_delta_pct: -30.0,
            winner: "Model B (50.0% faster)".to_string(),
            regressions: vec![],
            improvements: vec!["Throughput: 50.0% faster".to_string()],
        };
        assert_eq!(report.model_a, "model_a.gguf");
        assert_eq!(report.throughput_delta_pct, 50.0);
        assert!(report.regressions.is_empty());
        assert_eq!(report.improvements.len(), 1);
    }

    #[test]
    fn test_diff_benchmark_report_debug() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 90.0,
            throughput_delta_pct: -10.0,
            latency_a_ms: 10.0,
            latency_b_ms: 11.0,
            latency_delta_pct: 10.0,
            winner: "Model A".to_string(),
            regressions: vec!["Throughput regression".to_string()],
            improvements: vec![],
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("DiffBenchmarkReport"));
    }

    #[test]
    fn test_diff_benchmark_report_clone() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 100.0,
            throughput_delta_pct: 0.0,
            latency_a_ms: 10.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 0.0,
            winner: "Tie".to_string(),
            regressions: vec![],
            improvements: vec![],
        };
        let cloned = report.clone();
        assert_eq!(cloned.model_a, report.model_a);
        assert_eq!(cloned.throughput_delta_pct, report.throughput_delta_pct);
        assert_eq!(cloned.winner, report.winner);
    }

    #[test]
    fn test_diff_benchmark_report_with_regressions() {
        let report = DiffBenchmarkReport {
            model_a: "baseline.gguf".to_string(),
            model_b: "candidate.gguf".to_string(),
            throughput_a: 200.0,
            throughput_b: 100.0,
            throughput_delta_pct: -50.0,
            latency_a_ms: 5.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 100.0,
            winner: "Model A (50.0% faster)".to_string(),
            regressions: vec![
                "Throughput: 50.0% slower".to_string(),
                "Latency: 100.0% slower".to_string(),
            ],
            improvements: vec![],
        };
        assert_eq!(report.regressions.len(), 2);
        assert!(report.improvements.is_empty());
    }

    #[test]
    fn test_diff_benchmark_report_with_improvements() {
        let report = DiffBenchmarkReport {
            model_a: "old.gguf".to_string(),
            model_b: "new.gguf".to_string(),
            throughput_a: 50.0,
            throughput_b: 200.0,
            throughput_delta_pct: 300.0,
            latency_a_ms: 20.0,
            latency_b_ms: 5.0,
            latency_delta_pct: -75.0,
            winner: "Model B (300.0% faster)".to_string(),
            regressions: vec![],
            improvements: vec![
                "Throughput: 300.0% faster".to_string(),
                "Latency: 75.0% faster".to_string(),
            ],
        };
        assert!(report.regressions.is_empty());
        assert_eq!(report.improvements.len(), 2);
    }

    // ========================================================================
    // CiProfileReport print_json Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_print_json_no_assertions() {
        // Just verify it doesn't panic
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        report.print_json();
    }

    #[test]
    fn test_ci_profile_report_print_json_with_assertions() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: false,
            throughput_tok_s: 50.0,
            latency_p50_ms: 100.0,
            latency_p99_ms: 200.0,
            assertions: vec![
                AssertionResult {
                    name: "throughput".to_string(),
                    expected: ">= 100.0 tok/s".to_string(),
                    actual: "50.0 tok/s".to_string(),
                    passed: false,
                },
                AssertionResult {
                    name: "latency_p99".to_string(),
                    expected: "<= 50.0 ms".to_string(),
                    actual: "200.00 ms".to_string(),
                    passed: false,
                },
            ],
        };
        report.print_json();
    }

    #[test]
    fn test_ci_profile_report_print_json_single_assertion() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 200.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 10.0,
            assertions: vec![AssertionResult {
                name: "throughput".to_string(),
                expected: ">= 100.0 tok/s".to_string(),
                actual: "200.0 tok/s".to_string(),
                passed: true,
            }],
        };
        report.print_json();
    }

    // ========================================================================
    // CiProfileReport print_human Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_print_human_passed() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 150.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![AssertionResult {
                name: "throughput".to_string(),
                expected: ">= 100.0 tok/s".to_string(),
                actual: "150.0 tok/s".to_string(),
                passed: true,
            }],
        };
        report.print_human();
    }
