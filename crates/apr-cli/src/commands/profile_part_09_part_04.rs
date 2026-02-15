
    #[test]
    fn test_real_profile_results_synthetic_data() {
        let results = RealProfileResults {
            model_path: "synthetic.gguf".to_string(),
            architecture: "mock".to_string(),
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
            is_real_data: false,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        assert!(!results.is_real_data);
    }

    // ========================================================================
    // Edge Cases for Formats
    // ========================================================================

    #[test]
    fn test_detect_format_apr_path() {
        let path = Path::new("/some/deep/path/model.apr");
        assert_eq!(detect_format(path), "apr");
    }

    #[test]
    fn test_detect_format_safetensors_path() {
        let path = Path::new("models/v1.0/model.safetensors");
        assert_eq!(detect_format(path), "safetensors");
    }

    #[test]
    fn test_detect_format_gguf_with_version() {
        let path = Path::new("qwen2.5-coder-1.5b-q4_k_m.gguf");
        assert_eq!(detect_format(path), "gguf");
    }

    #[test]
    fn test_detect_format_empty_extension() {
        let path = Path::new("model.");
        assert_eq!(detect_format(path), "unknown");
    }

    // ========================================================================
    // CI Assertions Edge Cases
    // ========================================================================

    #[test]
    fn test_ci_assertions_all_set() {
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(25.0),
            max_memory_mb: Some(1024.0),
        };
        assert!(assertions.min_throughput.is_some());
        assert!(assertions.max_p99_ms.is_some());
        assert!(assertions.max_p50_ms.is_some());
    }

    #[test]
    fn test_ci_profile_report_multiple_assertions_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 100000.0, // 100ms
            throughput_tok_s: 10.0,       // Low
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
            min_throughput: Some(100.0), // Will fail
            max_p99_ms: Some(50.0),      // Will fail (100ms > 50ms)
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        // Both assertions should fail
        assert_eq!(report.assertions.iter().filter(|a| !a.passed).count(), 2);
    }

    #[test]
    fn test_ci_profile_report_boundary_values() {
        // Exactly at threshold
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 50000.0, // Exactly 50ms
            throughput_tok_s: 100.0,     // Exactly at threshold
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
            min_throughput: Some(100.0), // Exactly at threshold
            max_p99_ms: Some(50.0),      // Exactly at threshold
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        // Exactly at threshold should pass for >= and <=
        assert!(report.passed);
    }

    // ========================================================================
    // OutputFormat and ProfileFocus Exhaustive Tests
    // ========================================================================

    #[test]
    fn test_output_format_copy_trait() {
        let format = OutputFormat::Json;
        let copied = format;
        assert!(matches!(format, OutputFormat::Json));
        assert!(matches!(copied, OutputFormat::Json));
    }

    #[test]
    fn test_profile_focus_copy_trait() {
        let focus = ProfileFocus::Mlp;
        let copied = focus;
        assert!(matches!(focus, ProfileFocus::Mlp));
        assert!(matches!(copied, ProfileFocus::Mlp));
    }

    #[test]
    fn test_output_format_parse_mixed_case() {
        assert!(matches!(
            "Json".parse::<OutputFormat>().unwrap(),
            OutputFormat::Json
        ));
        assert!(matches!(
            "FLAMEGRAPH".parse::<OutputFormat>().unwrap(),
            OutputFormat::Flamegraph
        ));
    }

    #[test]
    fn test_profile_focus_parse_mixed_case() {
        assert!(matches!(
            "ATTENTION".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Attention
        ));
        assert!(matches!(
            "Mlp".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Mlp
        ));
    }

    // ========================================================================
    // filter_results_by_focus Tests
    // ========================================================================

    fn make_test_results_with_hotspots() -> RealProfileResults {
        RealProfileResults {
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
            hotspots: vec![
                Hotspot {
                    name: "attention_qkv".to_string(),
                    time_us: 3000.0,
                    percent: 30.0,
                    count: 10,
                    avg_us: 300.0,
                    min_us: 280.0,
                    max_us: 320.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "mlp_gate_up".to_string(),
                    time_us: 2500.0,
                    percent: 25.0,
                    count: 10,
                    avg_us: 250.0,
                    min_us: 230.0,
                    max_us: 270.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "matmul_q4k".to_string(),
                    time_us: 2000.0,
                    percent: 20.0,
                    count: 10,
                    avg_us: 200.0,
                    min_us: 180.0,
                    max_us: 220.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "embedding_lookup".to_string(),
                    time_us: 1000.0,
                    percent: 10.0,
                    count: 10,
                    avg_us: 100.0,
                    min_us: 90.0,
                    max_us: 110.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "softmax".to_string(),
                    time_us: 800.0,
                    percent: 8.0,
                    count: 10,
                    avg_us: 80.0,
                    min_us: 70.0,
                    max_us: 90.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "ffn_down_proj".to_string(),
                    time_us: 500.0,
                    percent: 5.0,
                    count: 10,
                    avg_us: 50.0,
                    min_us: 40.0,
                    max_us: 60.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "lm_head".to_string(),
                    time_us: 200.0,
                    percent: 2.0,
                    count: 10,
                    avg_us: 20.0,
                    min_us: 15.0,
                    max_us: 25.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "linear_proj".to_string(),
                    time_us: 100.0,
                    percent: 1.0,
                    count: 10,
                    avg_us: 10.0,
                    min_us: 8.0,
                    max_us: 12.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "gemm_f16".to_string(),
                    time_us: 50.0,
                    percent: 0.5,
                    count: 5,
                    avg_us: 10.0,
                    min_us: 8.0,
                    max_us: 12.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![312.5; 32],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_filter_results_by_focus_all() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::All);
        assert_eq!(filtered.hotspots.len(), results.hotspots.len());
    }

    #[test]
    fn test_filter_results_by_focus_attention() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        // Should match: attention_qkv, softmax
        assert_eq!(
            filtered.hotspots.len(),
            2,
            "Expected 2 attention hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered.hotspots.iter().any(|h| h.name == "attention_qkv"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "softmax"));
    }

    #[test]
    fn test_filter_results_by_focus_mlp() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Mlp);
        // Should match: mlp_gate_up, ffn_down_proj, gate (in mlp_gate_up)
        assert!(
            filtered.hotspots.len() >= 2,
            "Expected at least 2 MLP hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered.hotspots.iter().any(|h| h.name == "mlp_gate_up"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "ffn_down_proj"));
    }

    #[test]
    fn test_filter_results_by_focus_matmul() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Matmul);
        // Should match: matmul_q4k, linear_proj, gemm_f16
        assert!(
            filtered.hotspots.len() >= 3,
            "Expected at least 3 matmul hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered.hotspots.iter().any(|h| h.name == "matmul_q4k"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "linear_proj"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "gemm_f16"));
    }

    #[test]
    fn test_filter_results_by_focus_embedding() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Embedding);
        // Should match: embedding_lookup, lm_head
        assert_eq!(
            filtered.hotspots.len(),
            2,
            "Expected 2 embedding hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered
            .hotspots
            .iter()
            .any(|h| h.name == "embedding_lookup"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "lm_head"));
    }

    #[test]
    fn test_filter_results_preserves_metadata() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        // All metadata should be preserved
        assert_eq!(filtered.model_path, results.model_path);
        assert_eq!(filtered.architecture, results.architecture);
        assert_eq!(filtered.num_layers, results.num_layers);
        assert_eq!(filtered.vocab_size, results.vocab_size);
        assert_eq!(filtered.hidden_dim, results.hidden_dim);
        assert_eq!(filtered.warmup_passes, results.warmup_passes);
        assert_eq!(filtered.measure_passes, results.measure_passes);
        assert_eq!(filtered.total_inference_us, results.total_inference_us);
        assert_eq!(filtered.throughput_tok_s, results.throughput_tok_s);
        assert_eq!(filtered.tokens_per_pass, results.tokens_per_pass);
        assert_eq!(filtered.per_layer_us.len(), results.per_layer_us.len());
        assert_eq!(filtered.is_real_data, results.is_real_data);
    }
