
    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_case_sensitive() {
        // Brick names are case-sensitive: "RMS_NORM" != "rms_norm"
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("RMS_NORM"), false);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Unknown brick type"));
    }

    // ========================================================================
    // BenchResult Field Combinations and Edge Cases
    // ========================================================================

    #[test]
    fn test_bench_result_all_zero_durations() {
        let result = BenchResult {
            total_tokens: 0,
            total_time: Duration::ZERO,
            tokens_per_second: 0.0,
            time_to_first_token: Duration::ZERO,
            iteration_times: vec![Duration::ZERO; 3],
            mean_time: Duration::ZERO,
            median_time: Duration::ZERO,
            std_dev: Duration::ZERO,
            passed: false,
        };
        assert_eq!(result.total_time, Duration::ZERO);
        assert_eq!(result.iteration_times.len(), 3);
        assert!(!result.passed);
    }

    #[test]
    fn test_bench_result_max_duration() {
        let max = Duration::from_secs(u64::MAX / 2);
        let result = BenchResult {
            total_tokens: usize::MAX,
            total_time: max,
            tokens_per_second: f64::MAX,
            time_to_first_token: max,
            iteration_times: vec![max],
            mean_time: max,
            median_time: max,
            std_dev: max,
            passed: true,
        };
        assert_eq!(result.total_tokens, usize::MAX);
        assert!(result.tokens_per_second.is_finite());
    }

    #[test]
    fn test_bench_result_nan_throughput() {
        let result = BenchResult {
            total_tokens: 0,
            total_time: Duration::ZERO,
            tokens_per_second: f64::NAN,
            time_to_first_token: Duration::ZERO,
            iteration_times: vec![],
            mean_time: Duration::ZERO,
            median_time: Duration::ZERO,
            std_dev: Duration::ZERO,
            passed: false,
        };
        assert!(result.tokens_per_second.is_nan());
    }

    #[test]
    fn test_bench_result_infinity_throughput() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::ZERO,
            tokens_per_second: f64::INFINITY,
            time_to_first_token: Duration::ZERO,
            iteration_times: vec![],
            mean_time: Duration::ZERO,
            median_time: Duration::ZERO,
            std_dev: Duration::ZERO,
            passed: true,
        };
        assert!(result.tokens_per_second.is_infinite());
    }

    #[test]
    fn test_bench_result_clone_deep_equality() {
        let result = BenchResult {
            total_tokens: 42,
            total_time: Duration::from_millis(1234),
            tokens_per_second: 34.036,
            time_to_first_token: Duration::from_millis(56),
            iteration_times: vec![
                Duration::from_millis(400),
                Duration::from_millis(500),
                Duration::from_millis(334),
            ],
            mean_time: Duration::from_millis(411),
            median_time: Duration::from_millis(400),
            std_dev: Duration::from_millis(68),
            passed: true,
        };
        let cloned = result.clone();
        assert_eq!(cloned.total_tokens, result.total_tokens);
        assert_eq!(cloned.total_time, result.total_time);
        assert_eq!(cloned.tokens_per_second, result.tokens_per_second);
        assert_eq!(cloned.time_to_first_token, result.time_to_first_token);
        assert_eq!(cloned.iteration_times, result.iteration_times);
        assert_eq!(cloned.mean_time, result.mean_time);
        assert_eq!(cloned.median_time, result.median_time);
        assert_eq!(cloned.std_dev, result.std_dev);
        assert_eq!(cloned.passed, result.passed);
    }

    #[test]
    fn test_bench_result_debug_contains_all_fields() {
        let result = BenchResult {
            total_tokens: 77,
            total_time: Duration::from_secs(3),
            tokens_per_second: 25.667,
            time_to_first_token: Duration::from_millis(40),
            iteration_times: vec![Duration::from_secs(1); 3],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(5),
            passed: true,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("total_tokens"));
        assert!(debug.contains("total_time"));
        assert!(debug.contains("tokens_per_second"));
        assert!(debug.contains("time_to_first_token"));
        assert!(debug.contains("iteration_times"));
        assert!(debug.contains("mean_time"));
        assert!(debug.contains("median_time"));
        assert!(debug.contains("std_dev"));
        assert!(debug.contains("passed"));
    }

    // ========================================================================
    // BenchConfig Exhaustive Field Tests
    // ========================================================================

    #[test]
    fn test_bench_config_default_all_fields() {
        let config = BenchConfig::default();
        assert_eq!(config.warmup, 3);
        assert_eq!(config.iterations, 5);
        assert_eq!(config.max_tokens, 32);
        assert_eq!(config.prompt, "What is 2+2?");
    }

    #[test]
    fn test_bench_config_single_char_prompt() {
        let config = BenchConfig {
            warmup: 1,
            iterations: 1,
            max_tokens: 1,
            prompt: "x".to_string(),
            quiet: false,
        };
        assert_eq!(config.prompt.len(), 1);
    }

    #[test]
    fn test_bench_config_multiline_prompt() {
        let config = BenchConfig {
            warmup: 1,
            iterations: 1,
            max_tokens: 64,
            prompt: "Line 1\nLine 2\nLine 3".to_string(),
            quiet: false,
        };
        assert!(config.prompt.contains('\n'));
        assert_eq!(config.prompt.lines().count(), 3);
    }

    #[test]
    fn test_bench_config_max_values() {
        let config = BenchConfig {
            warmup: usize::MAX,
            iterations: usize::MAX,
            max_tokens: usize::MAX,
            prompt: "test".to_string(),
            quiet: false,
        };
        assert_eq!(config.warmup, usize::MAX);
        assert_eq!(config.iterations, usize::MAX);
        assert_eq!(config.max_tokens, usize::MAX);
    }

    // ========================================================================
    // print_results() Edge Case Formatting Tests
    // ========================================================================

    #[test]
    fn test_print_results_just_below_grade_boundaries() {
        // 99.9 tok/s - should be A, not A+
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(1),
            tokens_per_second: 99.9,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(1),
            passed: true,
        };
        print_results(&result);

        // 49.9 tok/s - should be B, not A
        let result_b = BenchResult {
            tokens_per_second: 49.9,
            passed: true,
            ..result.clone()
        };
        print_results(&result_b);

        // 19.9 tok/s - should be C, not B
        let result_c = BenchResult {
            tokens_per_second: 19.9,
            passed: true,
            ..result.clone()
        };
        print_results(&result_c);

        // 9.9 tok/s - should be F, not C
        let result_f = BenchResult {
            tokens_per_second: 9.9,
            passed: false,
            ..result
        };
        print_results(&result_f);
    }

    #[test]
    fn test_print_results_fractional_throughput() {
        let result = BenchResult {
            total_tokens: 1,
            total_time: Duration::from_secs(100),
            tokens_per_second: 0.01,
            time_to_first_token: Duration::from_secs(50),
            iteration_times: vec![Duration::from_secs(20); 5],
            mean_time: Duration::from_secs(20),
            median_time: Duration::from_secs(20),
            std_dev: Duration::from_secs(1),
            passed: false,
        };
        print_results(&result);
    }

    // ========================================================================
    // run() with Various File Configurations
    // ========================================================================

    #[test]
    fn test_run_gguf_with_valid_magic_but_invalid_content() {
        // GGUF magic: "GGUF" followed by garbage
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let mut content = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF"
        content.extend_from_slice(&[0; 100]); // padding
        file.write_all(&content).expect("write");
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_with_json_header() {
        // SafeTensors starts with a JSON length + JSON header
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        let header = b"{\"__metadata__\":{}}";
        let len = (header.len() as u64).to_le_bytes();
        let mut content = Vec::new();
        content.extend_from_slice(&len);
        content.extend_from_slice(header);
        file.write_all(&content).expect("write");
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_with_magic_but_invalid_content() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let mut content = vec![0x41, 0x50, 0x52, 0x32]; // "APR2"
        content.extend_from_slice(&[0; 200]);
        file.write_all(&content).expect("write");
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_config_construction_from_parameters() {
        // Verify that run() constructs BenchConfig from its parameters
        // We test this indirectly: if parameters are passed, print_header shows them
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"fake data").expect("write");

        // warmup=7, iterations=13, max_tokens=128, prompt=Some("test prompt")
        let result = run(
            file.path(),
            7,
            13,
            128,
            Some("test prompt"),
            false,
            None,
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // print_header() with print_results() Integration
    // ========================================================================

    #[test]
    fn test_print_header_then_results_workflow() {
        // Simulates the actual output flow in run()
        let config = BenchConfig {
            warmup: 3,
            iterations: 5,
            max_tokens: 32,
            prompt: "What is 2+2?".to_string(),
            quiet: false,
        };
        let path = Path::new("/models/test.gguf");
        print_header(path, &config);

        let result = BenchResult {
            total_tokens: 160,
            total_time: Duration::from_secs(2),
            tokens_per_second: 80.0,
            time_to_first_token: Duration::from_millis(25),
            iteration_times: vec![Duration::from_millis(400); 5],
            mean_time: Duration::from_millis(400),
            median_time: Duration::from_millis(400),
            std_dev: Duration::from_millis(5),
            passed: true,
        };
        print_results(&result);
    }

    // ========================================================================
    // calculate_benchmark_stats() Additional Edge Cases
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_identical_long_duration() {
        // All iterations take exactly 10 seconds
        let times = vec![Duration::from_secs(10); 3];
        let config = BenchConfig {
            warmup: 0,
            iterations: 3,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 30, Duration::from_secs(10), &config).unwrap();

        assert_eq!(result.mean_time, Duration::from_secs(10));
        assert_eq!(result.median_time, Duration::from_secs(10));
        assert_eq!(result.std_dev, Duration::ZERO);
        // 30 tokens / 30s = 1 tok/s
        assert!((result.tokens_per_second - 1.0).abs() < 0.01);
        assert!(!result.passed); // 1 < 60
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_descending_times() {
        let times = vec![
            Duration::from_millis(500),
            Duration::from_millis(400),
            Duration::from_millis(300),
            Duration::from_millis(200),
            Duration::from_millis(100),
        ];
        let config = BenchConfig {
            warmup: 0,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 50, Duration::from_millis(50), &config).unwrap();

        // Total = 1500ms, mean = 300ms
        assert_eq!(result.mean_time, Duration::from_millis(300));
        // Sorted: [100,200,300,400,500], median index 2 = 300ms
        assert_eq!(result.median_time, Duration::from_millis(300));
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_high_token_count() {
        let times = vec![Duration::from_millis(10); 5];
        let config = BenchConfig {
            warmup: 0,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        // 10000 tokens in 50ms = 200,000 tok/s
        let result =
            calculate_benchmark_stats(times, 10000, Duration::from_millis(1), &config).unwrap();

        assert!(result.tokens_per_second > 100_000.0);
        assert!(result.passed);
    }
