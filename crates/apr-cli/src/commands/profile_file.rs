
    // ========================================================================
    // run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        // run(path, granular, format, focus, detect_naive, naive_threshold,
        //     compare_hf, energy, perf_grade, callgraph, fail_on_naive, output_path)
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            false, // granular
            OutputFormat::Human,
            ProfileFocus::All,
            false, // detect_naive
            0.5,   // naive_threshold
            None,  // compare_hf
            false, // energy
            false, // perf_grade
            false, // callgraph
            false, // fail_on_naive
            None,  // output_path
            32,    // tokens
            false, // ollama
            true,  // no_gpu
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Human,
            ProfileFocus::All,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid GGUF or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_format() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Json,
            ProfileFocus::All,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_granular_mode() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            true, // granular
            OutputFormat::Human,
            ProfileFocus::All,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_attention_focus() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Human,
            ProfileFocus::Attention,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file) but tests focus path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_matmul_focus() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Human,
            ProfileFocus::Matmul,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file) but tests focus path
        assert!(result.is_err());
    }

    // ========================================================================
    // run_ci Command Tests
    // ========================================================================

    #[test]
    fn test_run_ci_file_not_found() {
        let result = run_ci(
            Path::new("/nonexistent/model.gguf"),
            OutputFormat::Human,
            &CiAssertions::default(),
            3,  // warmup
            10, // measure
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_ci_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run_ci(
            file.path(),
            OutputFormat::Json,
            &CiAssertions {
                min_throughput: Some(100.0),
                ..Default::default()
            },
            1,
            1,
        );
        // Should fail (invalid file or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_ci_with_latency_assertions() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run_ci(
            file.path(),
            OutputFormat::Human,
            &CiAssertions {
                max_p99_ms: Some(50.0),
                max_p50_ms: Some(25.0),
                ..Default::default()
            },
            1,
            1,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    // ========================================================================
    // run_diff_benchmark Tests
    // ========================================================================

    #[test]
    fn test_run_diff_benchmark_model_a_not_found() {
        let model_b = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run_diff_benchmark(
            Path::new("/nonexistent/model_a.gguf"),
            model_b.path(),
            OutputFormat::Human,
            1,
            1,
            0.05,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_benchmark_model_b_not_found() {
        let mut model_a = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_a.write_all(b"not valid").expect("write");

        let result = run_diff_benchmark(
            model_a.path(),
            Path::new("/nonexistent/model_b.gguf"),
            OutputFormat::Human,
            1,
            1,
            0.05,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_benchmark_both_invalid() {
        let mut model_a = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_a.write_all(b"not valid a").expect("write");
        let mut model_b = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_b.write_all(b"not valid b").expect("write");

        let result = run_diff_benchmark(
            model_a.path(),
            model_b.path(),
            OutputFormat::Json,
            1,
            1,
            0.05,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    // ========================================================================
    // Additional Hotspot Tests
    // ========================================================================

    #[test]
    fn test_hotspot_debug() {
        let hotspot = Hotspot {
            name: "matmul".to_string(),
            time_us: 1000.0,
            percent: 50.0,
            count: 10,
            avg_us: 100.0,
            min_us: 80.0,
            max_us: 120.0,
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        let debug = format!("{:?}", hotspot);
        assert!(debug.contains("Hotspot"));
        assert!(debug.contains("matmul"));
    }

    #[test]
    fn test_hotspot_clone() {
        let hotspot = Hotspot {
            name: "attention".to_string(),
            time_us: 500.0,
            percent: 25.0,
            count: 5,
            avg_us: 100.0,
            min_us: 90.0,
            max_us: 110.0,
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        let cloned = hotspot.clone();
        assert_eq!(cloned.name, hotspot.name);
        assert_eq!(cloned.time_us, hotspot.time_us);
    }

    #[test]
    fn test_hotspot_zero_count() {
        let hotspot = Hotspot {
            name: "empty".to_string(),
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
        };
        assert_eq!(hotspot.count, 0);
        assert_eq!(hotspot.avg_us, 0.0);
    }

    #[test]
    fn test_hotspot_high_variance() {
        let hotspot = Hotspot {
            name: "variable_op".to_string(),
            time_us: 10000.0,
            percent: 100.0,
            count: 100,
            avg_us: 100.0,
            min_us: 10.0,
            max_us: 500.0, // High max vs avg
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        assert!(hotspot.max_us > hotspot.avg_us * 4.0);
    }

    // ========================================================================
    // Additional RealProfileResults Tests
    // ========================================================================

    #[test]
    fn test_real_profile_results_construction() {
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
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        assert_eq!(results.model_path, "test.gguf");
        assert_eq!(results.architecture, "llama");
        assert!(results.is_real_data);
    }

    #[test]
    fn test_real_profile_results_with_hotspots() {
        let hotspots = vec![
            Hotspot {
                name: "matmul".to_string(),
                time_us: 5000.0,
                percent: 50.0,
                count: 10,
                avg_us: 500.0,
                min_us: 450.0,
                max_us: 550.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
            Hotspot {
                name: "attention".to_string(),
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
        ];
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
            hotspots,
            per_layer_us: vec![100.0; 32],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        assert_eq!(results.hotspots.len(), 2);
        assert_eq!(results.per_layer_us.len(), 32);
    }
