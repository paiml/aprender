
    #[test]
    fn test_ci_profile_report_print_human_failed() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: false,
            throughput_tok_s: 50.0,
            latency_p50_ms: 100.0,
            latency_p99_ms: 200.0,
            assertions: vec![AssertionResult {
                name: "throughput".to_string(),
                expected: ">= 100.0 tok/s".to_string(),
                actual: "50.0 tok/s".to_string(),
                passed: false,
            }],
        };
        report.print_human();
    }

    #[test]
    fn test_ci_profile_report_print_human_no_assertions() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        report.print_human();
    }

    // ========================================================================
    // DiffBenchmarkReport print Tests
    // ========================================================================

    #[test]
    fn test_diff_benchmark_report_print_human() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 150.0,
            throughput_delta_pct: 50.0,
            latency_a_ms: 10.0,
            latency_b_ms: 7.0,
            latency_delta_pct: -30.0,
            winner: "Model B".to_string(),
            regressions: vec![],
            improvements: vec!["Throughput +50%".to_string()],
        };
        report.print_human();
    }

    #[test]
    fn test_diff_benchmark_report_print_human_with_regressions() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 200.0,
            throughput_b: 100.0,
            throughput_delta_pct: -50.0,
            latency_a_ms: 5.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 100.0,
            winner: "Model A".to_string(),
            regressions: vec!["Throughput -50%".to_string()],
            improvements: vec![],
        };
        report.print_human();
    }

    #[test]
    fn test_diff_benchmark_report_print_json() {
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
        report.print_json();
    }

    #[test]
    fn test_diff_benchmark_report_print_json_with_regressions_and_improvements() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 150.0,
            throughput_delta_pct: 50.0,
            latency_a_ms: 5.0,
            latency_b_ms: 7.0,
            latency_delta_pct: 40.0,
            winner: "Mixed".to_string(),
            regressions: vec!["Latency regression".to_string()],
            improvements: vec![
                "Throughput improvement".to_string(),
                "Memory improvement".to_string(),
            ],
        };
        report.print_json();
    }

    // ========================================================================
    // print_human_results Tests
    // ========================================================================

    #[test]
    fn test_print_human_results_basic() {
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
                name: "forward_pass".to_string(),
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
        let result = print_human_results(&results, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_granular() {
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
                name: "forward_pass".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 900.0,
                max_us: 1100.0,
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
        let result = print_human_results(&results, true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_simulated_data() {
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
            is_real_data: false,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_zero_total_time() {
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
        let result = print_human_results(&results, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_many_hotspots() {
        let hotspots: Vec<Hotspot> = (0..10)
            .map(|i| Hotspot {
                name: format!("op_{i}"),
                time_us: (10 - i) as f64 * 100.0,
                percent: (10 - i) as f64 * 10.0,
                count: 10,
                avg_us: (10 - i) as f64 * 10.0,
                min_us: (10 - i) as f64 * 8.0,
                max_us: (10 - i) as f64 * 12.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            })
            .collect();
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 10,
            total_inference_us: 5500.0,
            throughput_tok_s: 1818.0,
            tokens_per_pass: 10,
            hotspots,
            per_layer_us: vec![1375.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_granular_zero_max_layer() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 2,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![0.0, 0.0], // Zero layer times
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, true, false, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // print_json_results Tests
    // ========================================================================

    #[test]
    fn test_print_json_results_basic() {
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
        let result = print_json_results(&results);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_json_results_with_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 5,
            total_inference_us: 5000.0,
            throughput_tok_s: 200.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "op_a".to_string(),
                    time_us: 3000.0,
                    percent: 60.0,
                    count: 5,
                    avg_us: 600.0,
                    min_us: 550.0,
                    max_us: 650.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "op_b".to_string(),
                    time_us: 2000.0,
                    percent: 40.0,
                    count: 5,
                    avg_us: 400.0,
                    min_us: 350.0,
                    max_us: 450.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![1250.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_json_results(&results);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_json_results_single_hotspot() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 1,
            vocab_size: 100,
            hidden_dim: 32,
            warmup_passes: 0,
            measure_passes: 1,
            total_inference_us: 100.0,
            throughput_tok_s: 10000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "forward".to_string(),
                time_us: 100.0,
                percent: 100.0,
                count: 1,
                avg_us: 100.0,
                min_us: 100.0,
                max_us: 100.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![100.0],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_json_results(&results);
        assert!(result.is_ok());
    }
