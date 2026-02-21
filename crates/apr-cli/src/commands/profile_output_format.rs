
    // ========================================================================
    // OutputFormat Tests
    // ========================================================================

    #[test]
    fn test_output_format_parse() {
        assert!(matches!(
            "json".parse::<OutputFormat>().unwrap(),
            OutputFormat::Json
        ));
        assert!(matches!(
            "human".parse::<OutputFormat>().unwrap(),
            OutputFormat::Human
        ));
        assert!(matches!(
            "flamegraph".parse::<OutputFormat>().unwrap(),
            OutputFormat::Flamegraph
        ));
    }

    #[test]
    fn test_output_format_parse_text() {
        assert!(matches!(
            "text".parse::<OutputFormat>().unwrap(),
            OutputFormat::Human
        ));
    }

    #[test]
    fn test_output_format_parse_svg() {
        assert!(matches!(
            "svg".parse::<OutputFormat>().unwrap(),
            OutputFormat::Flamegraph
        ));
    }

    #[test]
    fn test_output_format_parse_case_insensitive() {
        assert!(matches!(
            "JSON".parse::<OutputFormat>().unwrap(),
            OutputFormat::Json
        ));
        assert!(matches!(
            "HUMAN".parse::<OutputFormat>().unwrap(),
            OutputFormat::Human
        ));
    }

    #[test]
    fn test_output_format_parse_invalid() {
        assert!("invalid".parse::<OutputFormat>().is_err());
        assert!("xml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert!(matches!(format, OutputFormat::Human));
    }

    #[test]
    fn test_output_format_debug() {
        let format = OutputFormat::Json;
        let debug = format!("{format:?}");
        assert!(debug.contains("Json"));
    }

    #[test]
    fn test_output_format_clone() {
        let format = OutputFormat::Flamegraph;
        let cloned = format;
        assert!(matches!(cloned, OutputFormat::Flamegraph));
    }

    // ========================================================================
    // ProfileFocus Tests
    // ========================================================================

    #[test]
    fn test_profile_focus_parse() {
        assert!(matches!(
            "attention".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Attention
        ));
        assert!(matches!(
            "mlp".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Mlp
        ));
        assert!(matches!(
            "all".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::All
        ));
    }

    #[test]
    fn test_profile_focus_parse_attn() {
        assert!(matches!(
            "attn".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Attention
        ));
    }

    #[test]
    fn test_profile_focus_parse_ffn() {
        assert!(matches!(
            "ffn".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Mlp
        ));
    }

    #[test]
    fn test_profile_focus_parse_matmul() {
        assert!(matches!(
            "matmul".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Matmul
        ));
        assert!(matches!(
            "gemm".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Matmul
        ));
    }

    #[test]
    fn test_profile_focus_parse_embedding() {
        assert!(matches!(
            "embedding".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Embedding
        ));
        assert!(matches!(
            "embed".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Embedding
        ));
    }

    #[test]
    fn test_profile_focus_parse_invalid() {
        assert!("invalid".parse::<ProfileFocus>().is_err());
        assert!("unknown".parse::<ProfileFocus>().is_err());
    }

    #[test]
    fn test_profile_focus_default() {
        let focus = ProfileFocus::default();
        assert!(matches!(focus, ProfileFocus::All));
    }

    #[test]
    fn test_profile_focus_debug() {
        let focus = ProfileFocus::Attention;
        let debug = format!("{focus:?}");
        assert!(debug.contains("Attention"));
    }

    // ========================================================================
    // detect_format Tests
    // ========================================================================

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("model.apr")), "apr");
        assert_eq!(detect_format(Path::new("model.safetensors")), "safetensors");
        assert_eq!(detect_format(Path::new("model.gguf")), "gguf");
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format(Path::new("model.xyz")), "unknown");
        assert_eq!(detect_format(Path::new("model.pt")), "unknown");
    }

    #[test]
    fn test_detect_format_pytorch() {
        assert_eq!(detect_format(Path::new("model.bin")), "pytorch");
    }

    #[test]
    fn test_detect_format_no_extension() {
        assert_eq!(detect_format(Path::new("model")), "unknown");
    }

    #[test]
    fn test_detect_format_case() {
        // Extensions are case-sensitive in typical implementations
        assert_eq!(detect_format(Path::new("model.APR")), "unknown");
    }

    // ========================================================================
    // CiAssertions Tests
    // ========================================================================

    #[test]
    fn test_ci_assertions_default() {
        let assertions = CiAssertions::default();
        assert!(assertions.min_throughput.is_none());
        assert!(assertions.max_p99_ms.is_none());
        assert!(assertions.max_p50_ms.is_none());
    }

    #[test]
    fn test_ci_assertions_with_throughput() {
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            ..Default::default()
        };
        assert_eq!(assertions.min_throughput.unwrap(), 100.0);
    }

    #[test]
    fn test_ci_assertions_with_latency() {
        let assertions = CiAssertions {
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(25.0),
            ..Default::default()
        };
        assert_eq!(assertions.max_p99_ms.unwrap(), 50.0);
        assert_eq!(assertions.max_p50_ms.unwrap(), 25.0);
    }

    #[test]
    fn test_ci_assertions_debug() {
        let assertions = CiAssertions::default();
        let debug = format!("{assertions:?}");
        assert!(debug.contains("CiAssertions"));
    }

    #[test]
    fn test_ci_assertions_clone() {
        let assertions = CiAssertions {
            min_throughput: Some(50.0),
            ..Default::default()
        };
        let cloned = assertions.clone();
        assert_eq!(cloned.min_throughput, assertions.min_throughput);
    }

    // ========================================================================
    // AssertionResult Tests
    // ========================================================================

    #[test]
    fn test_assertion_result_passed() {
        let result = AssertionResult {
            name: "throughput".to_string(),
            expected: ">= 100.0 tok/s".to_string(),
            actual: "150.0 tok/s".to_string(),
            passed: true,
        };
        assert!(result.passed);
        assert_eq!(result.name, "throughput");
    }

    #[test]
    fn test_assertion_result_failed() {
        let result = AssertionResult {
            name: "latency_p99".to_string(),
            expected: "<= 50.0 ms".to_string(),
            actual: "75.0 ms".to_string(),
            passed: false,
        };
        assert!(!result.passed);
    }

    #[test]
    fn test_assertion_result_debug() {
        let result = AssertionResult {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
            passed: true,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("AssertionResult"));
    }

    #[test]
    fn test_assertion_result_clone() {
        let result = AssertionResult {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
            passed: true,
        };
        let cloned = result.clone();
        assert_eq!(cloned.name, result.name);
        assert_eq!(cloned.passed, result.passed);
    }

    // ========================================================================
    // CiProfileReport Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_from_results_no_assertions() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
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
        let assertions = CiAssertions::default();
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
        assert!(report.assertions.is_empty());
    }

    #[test]
    fn test_ci_profile_report_throughput_pass() {
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
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
        assert_eq!(report.assertions.len(), 1);
        assert!(report.assertions[0].passed);
    }

    #[test]
    fn test_ci_profile_report_throughput_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 50.0, // Below threshold
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
    fn test_ci_profile_report_latency_assertions() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 25000.0, // 25ms
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
            max_p50_ms: Some(50.0),
            max_p99_ms: Some(100.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed); // 25ms is below both thresholds
        assert_eq!(report.assertions.len(), 2);
    }

    #[test]
    fn test_ci_profile_report_debug() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("CiProfileReport"));
    }

    #[test]
    fn test_ci_profile_report_clone() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        let cloned = report.clone();
        assert_eq!(cloned.model_path, report.model_path);
        assert_eq!(cloned.passed, report.passed);
    }
