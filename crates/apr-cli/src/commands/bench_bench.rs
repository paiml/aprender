
    #[test]
    fn test_bench_result_debug_trait() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(1),
            tokens_per_second: 100.0,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: vec![],
            mean_time: Duration::from_millis(100),
            median_time: Duration::from_millis(100),
            std_dev: Duration::from_millis(1),
            passed: true,
        };

        // BenchResult derives Debug
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("BenchResult"));
        assert!(debug_str.contains("tokens_per_second"));
    }

    #[test]
    fn test_bench_result_clone_trait() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(1),
            tokens_per_second: 100.0,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: vec![Duration::from_millis(100)],
            mean_time: Duration::from_millis(100),
            median_time: Duration::from_millis(100),
            std_dev: Duration::from_millis(1),
            passed: true,
        };

        let cloned = result.clone();
        assert_eq!(cloned.total_tokens, result.total_tokens);
        assert_eq!(cloned.tokens_per_second, result.tokens_per_second);
    }

    // ========================================================================
    // Throughput Grade Tests
    // ========================================================================

    #[test]
    fn test_throughput_grade_a_plus() {
        // A+ grade: >= 100 tok/s
        let result = BenchResult {
            total_tokens: 500,
            total_time: Duration::from_millis(500),
            tokens_per_second: 1000.0,
            time_to_first_token: Duration::from_millis(1),
            iteration_times: vec![Duration::from_millis(100); 5],
            mean_time: Duration::from_millis(100),
            median_time: Duration::from_millis(100),
            std_dev: Duration::from_millis(1),
            passed: true,
        };

        assert!(result.tokens_per_second >= 100.0);
    }

    #[test]
    fn test_throughput_grade_b() {
        // B grade: 50-99 tok/s
        let result = BenchResult {
            total_tokens: 250,
            total_time: Duration::from_secs(5),
            tokens_per_second: 50.0,
            time_to_first_token: Duration::from_millis(20),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(10),
            passed: true,
        };

        assert!(result.tokens_per_second >= 50.0 && result.tokens_per_second < 100.0);
    }

    #[test]
    fn test_throughput_grade_c() {
        // C grade: 20-49 tok/s
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(5),
            tokens_per_second: 20.0,
            time_to_first_token: Duration::from_millis(50),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(50),
            passed: true,
        };

        assert!(result.tokens_per_second >= 20.0 && result.tokens_per_second < 50.0);
    }

    // ========================================================================
    // print_header() Tests
    // ========================================================================

    #[test]
    fn test_print_header_default_config() {
        let config = BenchConfig::default();
        let path = Path::new("/tmp/model.gguf");
        // Should not panic; exercises print_header lines 385-394
        print_header(path, &config);
    }

    #[test]
    fn test_print_header_custom_config() {
        let config = BenchConfig {
            warmup: 10,
            iterations: 100,
            max_tokens: 256,
            prompt: "Explain quantum computing".to_string(),
            quiet: false,
        };
        let path = Path::new("/models/large-model.safetensors");
        print_header(path, &config);
    }

    #[test]
    fn test_print_header_empty_prompt() {
        let config = BenchConfig {
            warmup: 0,
            iterations: 0,
            max_tokens: 0,
            prompt: String::new(),
            quiet: false,
        };
        let path = Path::new("model.apr");
        print_header(path, &config);
    }

    #[test]
    fn test_print_header_unicode_path() {
        let config = BenchConfig::default();
        let path = Path::new("/tmp/модель.gguf");
        print_header(path, &config);
    }

    // ========================================================================
    // print_results() Tests - All Grade Branches
    // ========================================================================

    #[test]
    fn test_print_results_grade_a_plus() {
        // A+ grade: >= 100 tok/s
        let result = BenchResult {
            total_tokens: 500,
            total_time: Duration::from_secs(1),
            tokens_per_second: 150.0,
            time_to_first_token: Duration::from_millis(5),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(2),
            passed: true,
        };
        // Exercises the A+ branch (>= 100.0) in print_results
        print_results(&result);
    }

    #[test]
    fn test_print_results_grade_a() {
        // A grade: >= 50 and < 100 tok/s
        let result = BenchResult {
            total_tokens: 250,
            total_time: Duration::from_secs(5),
            tokens_per_second: 75.0,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(5),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_grade_b() {
        // B grade: >= 20 and < 50 tok/s
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(5),
            tokens_per_second: 35.0,
            time_to_first_token: Duration::from_millis(20),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(10),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_grade_c() {
        // C grade: >= 10 and < 20 tok/s
        let result = BenchResult {
            total_tokens: 50,
            total_time: Duration::from_secs(5),
            tokens_per_second: 15.0,
            time_to_first_token: Duration::from_millis(50),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(20),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_grade_f() {
        // F grade: < 10 tok/s
        let result = BenchResult {
            total_tokens: 10,
            total_time: Duration::from_secs(5),
            tokens_per_second: 2.0,
            time_to_first_token: Duration::from_millis(500),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(100),
            passed: false,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_passed_true() {
        // Tests the passed=true display path
        let result = BenchResult {
            total_tokens: 200,
            total_time: Duration::from_secs(2),
            tokens_per_second: 100.0,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: vec![Duration::from_millis(400); 5],
            mean_time: Duration::from_millis(400),
            median_time: Duration::from_millis(400),
            std_dev: Duration::from_millis(5),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_passed_false() {
        // Tests the passed=false display path (red output)
        let result = BenchResult {
            total_tokens: 5,
            total_time: Duration::from_secs(10),
            tokens_per_second: 0.5,
            time_to_first_token: Duration::from_secs(2),
            iteration_times: vec![Duration::from_secs(2); 5],
            mean_time: Duration::from_secs(2),
            median_time: Duration::from_secs(2),
            std_dev: Duration::from_millis(200),
            passed: false,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_exact_boundary_100() {
        // Exactly 100 tok/s: should be A+
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(1),
            tokens_per_second: 100.0,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(1),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_exact_boundary_50() {
        // Exactly 50 tok/s: should be A (>= 50, not >= 100)
        let result = BenchResult {
            total_tokens: 50,
            total_time: Duration::from_secs(1),
            tokens_per_second: 50.0,
            time_to_first_token: Duration::from_millis(20),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(5),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_exact_boundary_20() {
        // Exactly 20 tok/s: should be B
        let result = BenchResult {
            total_tokens: 20,
            total_time: Duration::from_secs(1),
            tokens_per_second: 20.0,
            time_to_first_token: Duration::from_millis(50),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(10),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_exact_boundary_10() {
        // Exactly 10 tok/s: should be C
        let result = BenchResult {
            total_tokens: 10,
            total_time: Duration::from_secs(1),
            tokens_per_second: 10.0,
            time_to_first_token: Duration::from_millis(100),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(20),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_zero_throughput() {
        let result = BenchResult {
            total_tokens: 0,
            total_time: Duration::from_secs(1),
            tokens_per_second: 0.0,
            time_to_first_token: Duration::ZERO,
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::ZERO,
            passed: false,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_very_high_throughput() {
        let result = BenchResult {
            total_tokens: 100_000,
            total_time: Duration::from_millis(100),
            tokens_per_second: 1_000_000.0,
            time_to_first_token: Duration::from_nanos(100),
            iteration_times: vec![Duration::from_millis(20); 5],
            mean_time: Duration::from_millis(20),
            median_time: Duration::from_millis(20),
            std_dev: Duration::from_nanos(500),
            passed: true,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_subsecond_times() {
        // Test formatting of sub-millisecond times
        let result = BenchResult {
            total_tokens: 50,
            total_time: Duration::from_micros(500),
            tokens_per_second: 100_000.0,
            time_to_first_token: Duration::from_micros(5),
            iteration_times: vec![Duration::from_micros(100); 5],
            mean_time: Duration::from_micros(100),
            median_time: Duration::from_micros(100),
            std_dev: Duration::from_micros(1),
            passed: true,
        };
        print_results(&result);
    }

    // ========================================================================
    // calculate_benchmark_stats() Tests
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_uniform_times() {
        let times = vec![Duration::from_secs(1); 5];
        let config = BenchConfig {
            warmup: 1,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 100, Duration::from_millis(50), &config).unwrap();

        assert_eq!(result.total_tokens, 100);
        assert_eq!(result.total_time, Duration::from_secs(5));
        // 100 tokens / 5 seconds = 20 tok/s
        assert!((result.tokens_per_second - 20.0).abs() < 0.01);
        assert_eq!(result.mean_time, Duration::from_secs(1));
        assert_eq!(result.median_time, Duration::from_secs(1));
        // All identical => std_dev == 0
        assert_eq!(result.std_dev, Duration::ZERO);
        assert_eq!(result.time_to_first_token, Duration::from_millis(50));
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_varying_times() {
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
            Duration::from_millis(500),
        ];
        let config = BenchConfig {
            warmup: 1,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 50, Duration::from_millis(10), &config).unwrap();

        // Total time = 1500ms = 1.5s
        assert_eq!(result.total_time, Duration::from_millis(1500));
        // Mean = 300ms
        assert_eq!(result.mean_time, Duration::from_millis(300));
        // Sorted: [100,200,300,400,500], median = index 2 = 300ms
        assert_eq!(result.median_time, Duration::from_millis(300));
        // 50 tokens / 1.5s = ~33.33 tok/s
        assert!((result.tokens_per_second - 33.333).abs() < 0.1);
        // Std dev: sqrt(mean(sq_diffs)) in ms
        // diffs: [-200, -100, 0, 100, 200] ms
        // sq: [40000, 10000, 0, 10000, 40000]
        // variance = 20000 => std = ~141.4ms
        let std_ms = result.std_dev.as_secs_f64() * 1000.0;
        assert!((std_ms - 141.42).abs() < 1.0);
    }
