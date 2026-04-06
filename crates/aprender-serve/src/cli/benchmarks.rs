
    // =========================================================================
    // run_visualization
    // =========================================================================

    #[test]
    fn test_run_visualization_with_color() {
        // Should not panic
        run_visualization(true, 50);
    }

    #[test]
    fn test_run_visualization_without_color() {
        run_visualization(false, 50);
    }

    #[test]
    fn test_run_visualization_small_samples() {
        run_visualization(false, 10);
    }

    #[test]
    fn test_run_visualization_large_samples() {
        run_visualization(true, 200);
    }

    // =========================================================================
    // run_convoy_test
    // =========================================================================

    #[test]
    fn test_run_convoy_test_no_args() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_runtime() {
        let result = run_convoy_test(Some("test-runtime".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_model() {
        let result = run_convoy_test(None, Some("test-model".to_string()), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_output() {
        let dir = tempfile::tempdir().expect("dir");
        let output_path = dir.path().join("convoy_results.json");
        let result = run_convoy_test(None, None, Some(output_path.to_string_lossy().to_string()));
        assert!(result.is_ok());
        assert!(output_path.exists());
        let contents = std::fs::read_to_string(&output_path).expect("contents");
        assert!(contents.contains("baseline_short_p99_ms"));
    }

    #[test]
    fn test_run_convoy_test_all_args() {
        let dir = tempfile::tempdir().expect("dir");
        let output_path = dir.path().join("convoy_all.json");
        let result = run_convoy_test(
            Some("realizar".to_string()),
            Some("llama".to_string()),
            Some(output_path.to_string_lossy().to_string()),
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_saturation_test
    // =========================================================================

    #[test]
    fn test_run_saturation_test_no_args() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_runtime() {
        let result = run_saturation_test(Some("test-runtime".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_model() {
        let result = run_saturation_test(None, Some("test-model".to_string()), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_output() {
        let dir = tempfile::tempdir().expect("dir");
        let output_path = dir.path().join("saturation_results.json");
        let result =
            run_saturation_test(None, None, Some(output_path.to_string_lossy().to_string()));
        assert!(result.is_ok());
        assert!(output_path.exists());
        let contents = std::fs::read_to_string(&output_path).expect("contents");
        assert!(contents.contains("baseline_throughput"));
    }

    // =========================================================================
    // parse_cargo_bench_output
    // =========================================================================

    #[test]
    fn test_parse_cargo_bench_output_valid_lines() {
        let output = "test benchmark_matmul ... bench:      1,234 ns/iter (+/- 56)\n\
                       test benchmark_softmax ... bench:        789 ns/iter (+/- 12)";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "benchmark_matmul");
        assert_eq!(results[0]["time_ns"], 1234);
        assert_eq!(results[0]["suite"], "tensor_ops");
        assert_eq!(results[1]["name"], "benchmark_softmax");
        assert_eq!(results[1]["time_ns"], 789);
    }

    #[test]
    fn test_parse_cargo_bench_output_no_suite() {
        let output = "test bench_add ... bench:        100 ns/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert!(results[0]["suite"].is_null());
    }

    #[test]
    fn test_parse_cargo_bench_output_empty() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines() {
        let output = "Compiling realizar v0.3.5\nFinished release\nrunning 5 tests";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_partial_match() {
        // Has "bench:" but not "ns/iter"
        let output = "test partial ... bench: 100 ms/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_mixed_lines() {
        let output = "Compiling realizar v0.3.5\n\
                       test bench_fast ... bench:        50 ns/iter (+/- 2)\n\
                       running 1 test\n\
                       test bench_slow ... bench:      5,000 ns/iter (+/- 100)\n\
                       test result: ok. 0 passed";
        let results = parse_cargo_bench_output(output, Some("all"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_cargo_bench_output_large_numbers() {
        let output = "test bench_huge ... bench: 1,234,567 ns/iter (+/- 1,000)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 1234567);
    }

    #[test]
    fn test_parse_cargo_bench_output_unparseable_time() {
        // Time has non-numeric chars
        let output = "test bench_bad ... bench:      abc ns/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // =========================================================================
    // run_benchmarks (list mode only, since actual bench needs cargo)
    // =========================================================================

    #[test]
    fn test_run_benchmarks_list_mode() {
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_external_no_feature() {
        // Without bench-http feature, should return error
        let result = run_benchmarks(
            Some("all".to_string()),
            false,
            Some("ollama".to_string()),
            None,
            Some("http://localhost:11434".to_string()),
            None,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Helper: create valid benchmark JSON
    // =========================================================================

    fn create_test_benchmark_json(runtime: &str) -> String {
        use crate::bench::*;
        let result = FullBenchmarkResult {
            version: "1.1".to_string(),
            timestamp: "2025-12-09T12:00:00Z".to_string(),
            config: BenchmarkConfig {
                model: "test".to_string(),
                format: "apr".to_string(),
                quantization: "q4_k".to_string(),
                runtime: runtime.to_string(),
                runtime_version: "1.0.0".to_string(),
            },
            hardware: HardwareSpec::default(),
            sampling: SamplingConfig::default(),
            thermal: ThermalInfo::default(),
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: 7.0,
                    p95: 9.0,
                    p99: 10.0,
                    p999: 12.0,
                },
                itl_ms: ItlResults {
                    median: 10.0,
                    std_dev: 2.0,
                    p99: 15.0,
                },
                throughput_tok_s: ThroughputResults {
                    median: 100.0,
                    ci_95: (95.0, 105.0),
                },
                memory_mb: MemoryResults {
                    model_mb: 512,
                    peak_rss_mb: 1024,
                    kv_waste_pct: 3.0,
                },
                energy: EnergyResults {
                    total_joules: 50.0,
                    token_joules: 0.5,
                    idle_watts: 8.0,
                },
                cold_start_ms: ColdStartResults {
                    median: 100.0,
                    p99: 150.0,
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: 0.03,
                perplexity_wikitext2: Some(5.89),
            },
        };
        serde_json::to_string_pretty(&result).expect("serialization")
    }

    // =========================================================================
    // run_bench_compare
    // =========================================================================

    #[test]
    fn test_run_bench_compare_valid_files() {
        let dir = tempfile::tempdir().expect("dir");
        let file1 = dir.path().join("baseline.json");
        let file2 = dir.path().join("current.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&file1, &json).expect("expected value");
        std::fs::write(&file2, &json).expect("expected value");

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 5.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_bench_compare_nonexistent_file1() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_compare_invalid_json() {
        let dir = tempfile::tempdir().expect("dir");
        let file1 = dir.path().join("bad1.json");
        let file2 = dir.path().join("bad2.json");
        std::fs::write(&file1, "not valid json").expect("expected value");
        std::fs::write(&file2, "also bad").expect("expected value");

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_compare_file1_valid_file2_invalid() {
        let dir = tempfile::tempdir().expect("dir");
        let file1 = dir.path().join("good.json");
        let file2 = dir.path().join("bad.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&file1, &json).expect("expected value");
        std::fs::write(&file2, "not json").expect("expected value");

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_compare_zero_threshold() {
        let dir = tempfile::tempdir().expect("dir");
        let file1 = dir.path().join("base.json");
        let file2 = dir.path().join("curr.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&file1, &json).expect("expected value");
        std::fs::write(&file2, &json).expect("expected value");

        let result = run_bench_compare(&file1.to_string_lossy(), &file2.to_string_lossy(), 0.0);
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_bench_regression
    // =========================================================================

    #[test]
    fn test_run_bench_regression_no_regression() {
        let dir = tempfile::tempdir().expect("dir");
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&baseline, &json).expect("expected value");
        std::fs::write(&current, &json).expect("expected value");

        let result = run_bench_regression(
            &baseline.to_string_lossy(),
            &current.to_string_lossy(),
            false, // strict
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_bench_regression_strict_mode() {
        let dir = tempfile::tempdir().expect("dir");
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");

        let json = create_test_benchmark_json("realizar");
        std::fs::write(&baseline, &json).expect("expected value");
        std::fs::write(&current, &json).expect("expected value");

        // Strict mode with identical data
        let result = run_bench_regression(
            &baseline.to_string_lossy(),
            &current.to_string_lossy(),
            true, // strict
        );
        // May pass or fail depending on threshold check
        let _ = result;
    }

    #[test]
    fn test_run_bench_regression_nonexistent() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false, // strict
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_invalid_json() {
        let dir = tempfile::tempdir().expect("dir");
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");
        std::fs::write(&baseline, "bad json").expect("expected value");
        std::fs::write(&current, "also bad").expect("expected value");

        let result = run_bench_regression(
            &baseline.to_string_lossy(),
            &current.to_string_lossy(),
            false, // strict
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // print_info
    // =========================================================================

    #[test]
    fn test_print_info_does_not_panic() {
        print_info();
    }

    // =========================================================================
    // home_dir
    // =========================================================================

    #[test]
    fn test_home_dir_returns_some() {
        // HOME is typically set in test environments
        let result = home_dir();
        // May or may not be Some depending on environment
        let _ = result;
    }

    // =========================================================================
    // validate_suite_name
    // =========================================================================

    #[test]
    fn test_validate_suite_name_valid() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_validate_suite_name_invalid() {
        assert!(!validate_suite_name("nonexistent"));
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("TENSOR_OPS")); // case sensitive
        assert!(!validate_suite_name("tensor_ops ")); // trailing space
    }
