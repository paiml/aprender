
    // ========================================================================
    // print_flamegraph Tests
    // ========================================================================

    #[test]
    fn test_print_flamegraph_stdout() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "op_a".to_string(),
                    time_us: 600.0,
                    percent: 60.0,
                    count: 1,
                    avg_us: 600.0,
                    min_us: 600.0,
                    max_us: 600.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "op_b".to_string(),
                    time_us: 400.0,
                    percent: 40.0,
                    count: 1,
                    avg_us: 400.0,
                    min_us: 400.0,
                    max_us: 400.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_flamegraph_to_file() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "forward".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 1000.0,
                max_us: 1000.0,
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
        let file = NamedTempFile::with_suffix(".svg").expect("create temp file");
        let result = print_flamegraph(&results, Some(file.path()));
        assert!(result.is_ok());
        // Verify file was written
        let content = std::fs::read_to_string(file.path()).expect("read svg");
        assert!(content.contains("<svg"));
        assert!(content.contains("forward"));
    }

    #[test]
    fn test_print_flamegraph_empty_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 0,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_flamegraph_zero_total_time() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 0,
            hotspots: vec![Hotspot {
                name: "op".to_string(),
                time_us: 0.0,
                percent: 0.0,
                count: 0,
                avg_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
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
        let result = print_flamegraph(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_flamegraph_invalid_output_path() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, Some(Path::new("/nonexistent/dir/file.svg")));
        assert!(result.is_err());
    }

    // ========================================================================
    // detect_format Extended Tests
    // ========================================================================

    #[test]
    fn test_detect_format_dot_only() {
        assert_eq!(detect_format(Path::new(".")), "unknown");
    }

    #[test]
    fn test_detect_format_multiple_dots() {
        assert_eq!(detect_format(Path::new("model.v2.gguf")), "gguf");
        assert_eq!(
            detect_format(Path::new("model.q4_k_m.safetensors")),
            "safetensors"
        );
    }

    #[test]
    fn test_detect_format_absolute_paths() {
        assert_eq!(detect_format(Path::new("/models/latest/model.apr")), "apr");
        assert_eq!(
            detect_format(Path::new("/tmp/downloads/model.gguf")),
            "gguf"
        );
    }

    // ========================================================================
    // Additional Edge Case Tests
    // ========================================================================

    #[test]
    fn test_real_profile_results_debug() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let debug = format!("{results:?}");
        assert!(debug.contains("RealProfileResults"));
        assert!(debug.contains("test.gguf"));
    }

    #[test]
    fn test_real_profile_results_clone() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "op".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 1000.0,
                max_us: 1000.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![250.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let cloned = results.clone();
        assert_eq!(cloned.model_path, results.model_path);
        assert_eq!(cloned.hotspots.len(), results.hotspots.len());
        assert_eq!(cloned.per_layer_us, results.per_layer_us);
    }

    // ========================================================================
    // OutputFormat parse error message Tests
    // ========================================================================

    #[test]
    fn test_output_format_error_message() {
        let err = "invalid".parse::<OutputFormat>().unwrap_err();
        assert!(err.contains("invalid"));
    }

    #[test]
    fn test_profile_focus_error_message() {
        let err = "invalid".parse::<ProfileFocus>().unwrap_err();
        assert!(err.contains("invalid"));
    }

    // ========================================================================
    // Comprehensive Boundary Tests for CI Report
    // ========================================================================

    #[test]
    fn test_ci_profile_report_very_high_throughput() {
        let results = RealProfileResults {
            model_path: "fast.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 10,
            total_inference_us: 1.0, // 0.001ms
            throughput_tok_s: 1_000_000.0,
            tokens_per_pass: 1,
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
            max_p99_ms: Some(1.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
    }

    #[test]
    fn test_ci_profile_report_very_slow() {
        let results = RealProfileResults {
            model_path: "slow.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 128,
            vocab_size: 128000,
            hidden_dim: 16384,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 10_000_000.0, // 10 seconds
            throughput_tok_s: 0.1,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(1.0),
            max_p99_ms: Some(100.0),
            max_p50_ms: Some(50.0),
            max_memory_mb: None,
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        // All 3 assertions should fail
        assert_eq!(report.assertions.len(), 3);
        assert!(report.assertions.iter().all(|a| !a.passed));
    }

    // ========================================================================
    // Attention Focus Variant Keywords
    // ========================================================================

    #[test]
    fn test_filter_attention_qkv_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "qkv_projection".to_string(),
                time_us: 500.0,
                percent: 50.0,
                count: 1,
                avg_us: 500.0,
                min_us: 500.0,
                max_us: 500.0,
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
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        assert_eq!(filtered.hotspots.len(), 1);
        assert_eq!(filtered.hotspots[0].name, "qkv_projection");
    }
