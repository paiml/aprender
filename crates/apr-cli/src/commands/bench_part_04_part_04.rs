
    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_single_iteration() {
        let times = vec![Duration::from_millis(500)];
        let config = BenchConfig {
            warmup: 0,
            iterations: 1,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 10, Duration::from_millis(50), &config).unwrap();

        assert_eq!(result.total_time, Duration::from_millis(500));
        assert_eq!(result.mean_time, Duration::from_millis(500));
        assert_eq!(result.median_time, Duration::from_millis(500));
        // Single value => std_dev = 0
        assert_eq!(result.std_dev, Duration::ZERO);
        // 10 tokens / 0.5s = 20 tok/s
        assert!((result.tokens_per_second - 20.0).abs() < 0.01);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_passed_threshold_high() {
        // High throughput should pass (>= 60 tok/s per spec Z5/Z6)
        let times = vec![Duration::from_millis(100); 5];
        let config = BenchConfig {
            warmup: 1,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        // 500 tokens / 0.5s = 1000 tok/s
        let result =
            calculate_benchmark_stats(times, 500, Duration::from_millis(5), &config).unwrap();

        assert!(result.passed); // 1000 >= 60
        assert!(result.tokens_per_second >= 60.0);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_passed_threshold_low() {
        // Low throughput should fail (< 60 tok/s)
        let times = vec![Duration::from_secs(2); 5];
        let config = BenchConfig {
            warmup: 1,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        // 10 tokens / 10s = 1 tok/s
        let result =
            calculate_benchmark_stats(times, 10, Duration::from_millis(200), &config).unwrap();

        assert!(!result.passed); // 1 < 60
        assert!(result.tokens_per_second < 60.0);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_passed_threshold_exactly_60() {
        // Exactly at the 60 tok/s threshold
        let times = vec![Duration::from_secs(1); 5];
        let config = BenchConfig {
            warmup: 1,
            iterations: 5,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        // 300 tokens / 5s = 60 tok/s
        let result =
            calculate_benchmark_stats(times, 300, Duration::from_millis(10), &config).unwrap();

        assert!(result.passed); // 60 >= 60
        assert!((result.tokens_per_second - 60.0).abs() < 0.01);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_preserves_iteration_times() {
        let original_times = vec![
            Duration::from_millis(100),
            Duration::from_millis(300),
            Duration::from_millis(200),
        ];
        let config = BenchConfig {
            warmup: 0,
            iterations: 3,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result = calculate_benchmark_stats(
            original_times.clone(),
            30,
            Duration::from_millis(10),
            &config,
        )
        .unwrap();

        // iteration_times should be preserved as-is (not sorted)
        assert_eq!(result.iteration_times, original_times);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_median_even_count() {
        // With even count, median picks the index iterations/2
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
        ];
        let config = BenchConfig {
            warmup: 0,
            iterations: 4,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 40, Duration::from_millis(25), &config).unwrap();

        // sorted: [100, 200, 300, 400], index 4/2=2 => 300ms
        assert_eq!(result.median_time, Duration::from_millis(300));
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_unsorted_input() {
        // Verify median works even when input is not sorted
        let times = vec![
            Duration::from_millis(500),
            Duration::from_millis(100),
            Duration::from_millis(300),
        ];
        let config = BenchConfig {
            warmup: 0,
            iterations: 3,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 30, Duration::from_millis(10), &config).unwrap();

        // sorted: [100, 300, 500], median index 3/2=1 => 300ms
        assert_eq!(result.median_time, Duration::from_millis(300));
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_very_fast_iterations() {
        let times = vec![Duration::from_nanos(100); 10];
        let config = BenchConfig {
            warmup: 0,
            iterations: 10,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 1000, Duration::from_nanos(10), &config).unwrap();

        // 1000 tokens / 1 microsecond = 1e9 tok/s
        assert!(result.tokens_per_second > 1e6);
        assert!(result.passed);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_large_outlier() {
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_secs(10), // outlier
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

        // Total time = 10400ms
        // Mean = 2080ms
        // Median: sorted [100, 100, 100, 100, 10000], index 2 = 100ms
        assert_eq!(result.median_time, Duration::from_millis(100));
        // Mean should be much larger than median due to outlier
        assert!(result.mean_time > result.median_time);
        // Std dev should be large
        assert!(result.std_dev > Duration::from_secs(1));
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_two_iterations() {
        let times = vec![Duration::from_millis(200), Duration::from_millis(800)];
        let config = BenchConfig {
            warmup: 0,
            iterations: 2,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result =
            calculate_benchmark_stats(times, 20, Duration::from_millis(50), &config).unwrap();

        // Mean = 500ms
        assert_eq!(result.mean_time, Duration::from_millis(500));
        // Median: sorted [200, 800], index 2/2=1 => 800ms
        assert_eq!(result.median_time, Duration::from_millis(800));
        // Std dev: diffs = [-300, 300], sq = [90000, 90000], var = 90000, std = 300ms
        let std_ms = result.std_dev.as_secs_f64() * 1000.0;
        assert!((std_ms - 300.0).abs() < 1.0);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_zero_tokens() {
        let times = vec![Duration::from_millis(100); 3];
        let config = BenchConfig {
            warmup: 0,
            iterations: 3,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let result = calculate_benchmark_stats(times, 0, Duration::ZERO, &config).unwrap();

        assert_eq!(result.total_tokens, 0);
        // 0 / 0.3 = 0.0
        assert_eq!(result.tokens_per_second, 0.0);
        assert!(!result.passed);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_calculate_stats_first_token_time_preserved() {
        let times = vec![Duration::from_millis(100); 3];
        let config = BenchConfig {
            warmup: 0,
            iterations: 3,
            max_tokens: 32,
            prompt: "test".to_string(),
            quiet: false,
        };
        let ttft = Duration::from_millis(42);
        let result = calculate_benchmark_stats(times, 30, ttft, &config).unwrap();

        assert_eq!(result.time_to_first_token, ttft);
    }

    // ========================================================================
    // run() Branch Coverage Tests
    // ========================================================================

    #[test]
    fn test_run_prompt_none_uses_default() {
        // When prompt is None, run() should use "What is 2+2?" as default
        // This exercises the prompt.unwrap_or("What is 2+2?") branch at line 109
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"fake gguf data").expect("write");
        let result = run(file.path(), 1, 1, 16, None, false, None, false);
        // Will error because it's not a real model, but exercises the None prompt path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_prompt_some_uses_custom() {
        // When prompt is Some, run() should use the provided prompt
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"fake gguf data").expect("write");
        let result = run(
            file.path(),
            1,
            1,
            16,
            Some("Hello world"),
            false,
            None,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fast_flag_deprecated() {
        // fast=true should not change behavior (deprecated parameter)
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"fake gguf data").expect("write");
        let result = run(file.path(), 1, 1, 16, None, true, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_zero_warmup_iterations() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"fake gguf data").expect("write");
        let result = run(file.path(), 0, 0, 0, None, false, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_large_max_tokens() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"fake gguf data").expect("write");
        let result = run(file.path(), 1, 1, 100_000, None, false, None, false);
        assert!(result.is_err());
    }

    // ========================================================================
    // Brick Name Validation Tests (exercises the match arms in run_brick_benchmark)
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_rms_norm_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // rms_norm is a valid brick name - should not return "Unknown brick" error
        let result = run(file.path(), 1, 3, 16, None, false, Some("rms_norm"), false);
        // Either succeeds or fails with a non-"Unknown brick" error
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_qkv_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("qkv"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_rope_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("rope"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_attn_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("attn"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_attention_alias_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("attention"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_o_proj_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("o_proj"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_ffn_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("ffn"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_layer_valid() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some("layer"), false);
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(!msg.contains("Unknown brick type"));
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_unknown_returns_error() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(
            file.path(),
            1,
            3,
            16,
            None,
            false,
            Some("unknown_thing"),
            false,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Unknown brick type"));
        assert!(msg.contains("unknown_thing"));
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_brick_name_empty_string_returns_error() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run(file.path(), 1, 3, 16, None, false, Some(""), false);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Unknown brick type"));
    }
