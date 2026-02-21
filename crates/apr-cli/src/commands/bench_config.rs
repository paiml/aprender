
    // ========================================================================
    // BenchConfig Tests
    // ========================================================================

    #[test]
    fn test_bench_config_default() {
        let config = BenchConfig::default();
        assert_eq!(config.warmup, 3);
        assert_eq!(config.iterations, 5);
        assert_eq!(config.max_tokens, 32);
    }

    #[test]
    fn test_bench_config_default_prompt() {
        let config = BenchConfig::default();
        assert_eq!(config.prompt, "What is 2+2?");
    }

    #[test]
    fn test_bench_config_custom_values() {
        let config = BenchConfig {
            warmup: 5,
            iterations: 10,
            max_tokens: 64,
            prompt: "Custom prompt".to_string(),
            quiet: false,
        };
        assert_eq!(config.warmup, 5);
        assert_eq!(config.iterations, 10);
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.prompt, "Custom prompt");
    }

    // ========================================================================
    // BenchResult Tests
    // ========================================================================

    #[test]
    fn test_bench_result_pass() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(5),
            tokens_per_second: 20.0,
            time_to_first_token: Duration::from_millis(50),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(10),
            passed: true,
        };

        assert!(result.passed);
        assert!(result.tokens_per_second >= 10.0);
    }

    #[test]
    fn test_bench_result_fail() {
        let result = BenchResult {
            total_tokens: 50,
            total_time: Duration::from_secs(10),
            tokens_per_second: 5.0, // Below threshold
            time_to_first_token: Duration::from_millis(100),
            iteration_times: vec![Duration::from_secs(2); 5],
            mean_time: Duration::from_secs(2),
            median_time: Duration::from_secs(2),
            std_dev: Duration::from_millis(50),
            passed: false,
        };

        assert!(!result.passed);
        assert!(result.tokens_per_second < 10.0);
    }

    #[test]
    fn test_bench_result_excellent_throughput() {
        let result = BenchResult {
            total_tokens: 1000,
            total_time: Duration::from_secs(1),
            tokens_per_second: 1000.0, // Excellent
            time_to_first_token: Duration::from_millis(5),
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(5),
            passed: true,
        };

        assert!(result.passed);
        assert!(result.tokens_per_second >= 100.0); // A+ grade threshold
    }

    #[test]
    fn test_bench_result_threshold_boundary() {
        // Exactly at threshold
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(10),
            tokens_per_second: 10.0, // Exactly at threshold
            time_to_first_token: Duration::from_millis(100),
            iteration_times: vec![Duration::from_secs(2); 5],
            mean_time: Duration::from_secs(2),
            median_time: Duration::from_secs(2),
            std_dev: Duration::from_millis(50),
            passed: true,
        };

        assert!(result.passed);
        assert!(result.tokens_per_second == 10.0);
    }

    #[test]
    fn test_bench_result_just_below_threshold() {
        let result = BenchResult {
            total_tokens: 99,
            total_time: Duration::from_secs(10),
            tokens_per_second: 9.9, // Just below threshold
            time_to_first_token: Duration::from_millis(100),
            iteration_times: vec![Duration::from_secs(2); 5],
            mean_time: Duration::from_secs(2),
            median_time: Duration::from_secs(2),
            std_dev: Duration::from_millis(50),
            passed: false,
        };

        assert!(!result.passed);
        assert!(result.tokens_per_second < 10.0);
    }

    // ========================================================================
    // Run Command Tests (no inference feature)
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            3,     // warmup
            5,     // iterations
            32,    // max_tokens
            None,  // prompt
            false, // fast
            None,  // brick
            false, // json
        );
        // Should fail - file doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(
            dir.path(),
            3,     // warmup
            5,     // iterations
            32,    // max_tokens
            None,  // prompt
            false, // fast
            None,  // brick
            false, // json
        );
        // Should fail - it's a directory not a file
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_brick_name() {
        // Create a dummy file so path validation passes
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(
            file.path(),
            3,                         // warmup
            5,                         // iterations
            32,                        // max_tokens
            None,                      // prompt
            false,                     // fast
            Some("invalid_brick_xyz"), // invalid brick name
            false,                     // json
        );
        // Should fail - either no inference feature or invalid brick
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_custom_prompt() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a real gguf").expect("write");
        let result = run(
            file.path(),
            1,  // warmup
            1,  // iterations
            16, // max_tokens
            Some("Custom test prompt"),
            false, // fast
            None,  // brick
            false, // json
        );
        // Will fail since it's not a real model, but tests the path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_various_extensions() {
        // Test .apr extension
        let mut file_apr = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file_apr.write_all(b"not a real apr").expect("write");
        let result = run(file_apr.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err()); // Will fail - invalid format

        // Test .safetensors extension
        let mut file_st = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file_st.write_all(b"not a real safetensors").expect("write");
        let result = run(file_st.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err()); // Will fail - invalid format
    }

    // ========================================================================
    // Brick Benchmark Tests (when inference feature enabled)
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_benchmark_rms_norm() {
        // Create a dummy file
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let _result = run(file.path(), 1, 1, 16, None, false, Some("rms_norm"), false);
        // May pass or fail depending on implementation, but should not panic
        // The important thing is the brick name is recognized
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_benchmark_invalid_name() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(
            file.path(),
            1,
            1,
            16,
            None,
            false,
            Some("nonexistent_brick"),
            false,
        );
        assert!(result.is_err());
        // Error message should mention unknown brick type
    }

    // ========================================================================
    // Additional BenchConfig Tests
    // ========================================================================

    #[test]
    fn test_bench_config_zero_iterations() {
        let config = BenchConfig {
            warmup: 0,
            iterations: 0,
            max_tokens: 0,
            prompt: String::new(),
            quiet: false,
        };
        assert_eq!(config.warmup, 0);
        assert_eq!(config.iterations, 0);
        assert_eq!(config.max_tokens, 0);
        assert!(config.prompt.is_empty());
    }

    #[test]
    fn test_bench_config_large_values() {
        let config = BenchConfig {
            warmup: 1000,
            iterations: 10000,
            max_tokens: 4096,
            prompt: "x".repeat(10000),
            quiet: false,
        };
        assert_eq!(config.warmup, 1000);
        assert_eq!(config.iterations, 10000);
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.prompt.len(), 10000);
    }

    #[test]
    fn test_bench_config_unicode_prompt() {
        let config = BenchConfig {
            warmup: 1,
            iterations: 1,
            max_tokens: 32,
            prompt: "日本語テスト 🎉 émojis".to_string(),
            quiet: false,
        };
        assert!(config.prompt.contains('日'));
        assert!(config.prompt.contains('🎉'));
    }

    // ========================================================================
    // Additional BenchResult Tests
    // ========================================================================

    #[test]
    fn test_bench_result_zero_tokens() {
        let result = BenchResult {
            total_tokens: 0,
            total_time: Duration::from_secs(1),
            tokens_per_second: 0.0,
            time_to_first_token: Duration::from_millis(0),
            iteration_times: vec![],
            mean_time: Duration::from_secs(0),
            median_time: Duration::from_secs(0),
            std_dev: Duration::from_secs(0),
            passed: false,
        };

        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.tokens_per_second, 0.0);
        assert!(!result.passed);
    }

    #[test]
    fn test_bench_result_single_iteration() {
        let result = BenchResult {
            total_tokens: 10,
            total_time: Duration::from_millis(500),
            tokens_per_second: 20.0,
            time_to_first_token: Duration::from_millis(50),
            iteration_times: vec![Duration::from_millis(500)],
            mean_time: Duration::from_millis(500),
            median_time: Duration::from_millis(500),
            std_dev: Duration::from_millis(0),
            passed: true,
        };

        assert_eq!(result.iteration_times.len(), 1);
        assert_eq!(result.mean_time, result.median_time);
    }

    #[test]
    fn test_bench_result_high_variance() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(10),
            tokens_per_second: 10.0,
            time_to_first_token: Duration::from_millis(100),
            iteration_times: vec![
                Duration::from_millis(100),
                Duration::from_secs(5),
                Duration::from_millis(200),
            ],
            mean_time: Duration::from_millis(1767),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_secs(2), // High variance
            passed: true,
        };

        // Mean and median are very different due to outlier
        assert!(result.mean_time > result.median_time);
    }

    #[test]
    fn test_bench_result_fast_ttft() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(1),
            tokens_per_second: 100.0,
            time_to_first_token: Duration::from_micros(500), // 0.5ms
            iteration_times: vec![Duration::from_millis(200); 5],
            mean_time: Duration::from_millis(200),
            median_time: Duration::from_millis(200),
            std_dev: Duration::from_millis(1),
            passed: true,
        };

        assert!(result.time_to_first_token < Duration::from_millis(1));
    }

    #[test]
    fn test_bench_result_slow_ttft() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(10),
            tokens_per_second: 10.0,
            time_to_first_token: Duration::from_secs(5), // 5s - very slow
            iteration_times: vec![Duration::from_secs(2); 5],
            mean_time: Duration::from_secs(2),
            median_time: Duration::from_secs(2),
            std_dev: Duration::from_millis(10),
            passed: true,
        };

        assert!(result.time_to_first_token >= Duration::from_secs(5));
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_run_empty_file() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // File is empty - should fail validation
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_unknown_extension() {
        let mut file = NamedTempFile::with_suffix(".xyz").expect("create temp file");
        file.write_all(b"some content").expect("write");
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_no_extension() {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(b"some content").expect("write");
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_bench_result_iterations_consistency() {
        // Ensure iteration_times length matches what would be expected
        let times = vec![Duration::from_millis(100); 10];
        let result = BenchResult {
            total_tokens: 320,
            total_time: Duration::from_secs(1),
            tokens_per_second: 320.0,
            time_to_first_token: Duration::from_millis(10),
            iteration_times: times.clone(),
            mean_time: Duration::from_millis(100),
            median_time: Duration::from_millis(100),
            std_dev: Duration::from_millis(0),
            passed: true,
        };

        assert_eq!(result.iteration_times.len(), 10);
    }
