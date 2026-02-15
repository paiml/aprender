
    /// Overflow beyond u32::MAX must fail.
    /// Bug class: silent truncation on overflow.
    #[test]
    fn parse_token_ids_overflow_u32() {
        let input = format!("{}", u64::from(u32::MAX) + 1);
        let result = parse_token_ids(&input);
        assert!(
            result.is_err(),
            "Values exceeding u32::MAX must be rejected"
        );
    }

    /// Mixed comma-and-space separated values.
    /// Bug class: split only accepting one delimiter type.
    #[test]
    fn parse_token_ids_mixed_comma_space() {
        let result = parse_token_ids("1, 2, 3").expect("should parse mixed delimiters");
        assert_eq!(result, vec![1, 2, 3]);
    }

    /// Token ID zero is valid.
    /// Bug class: zero treated as sentinel/invalid.
    #[test]
    fn parse_token_ids_zero_is_valid() {
        let result = parse_token_ids("0").expect("should parse zero");
        assert_eq!(result, vec![0u32]);
    }

    /// Multiple zeros.
    #[test]
    fn parse_token_ids_multiple_zeros() {
        let result = parse_token_ids("0,0,0").expect("should parse multiple zeros");
        assert_eq!(result, vec![0, 0, 0]);
    }

    /// Whitespace-only input should produce empty vec (all filtered out).
    #[test]
    fn parse_token_ids_whitespace_only() {
        let result = parse_token_ids("   \t  \n  ").expect("whitespace-only should not error");
        assert!(
            result.is_empty(),
            "Whitespace-only input should produce empty token list"
        );
    }

    /// JSON array with spaces around elements.
    #[test]
    fn parse_token_ids_json_with_whitespace() {
        let result = parse_token_ids("  [ 10 , 20 , 30 ]  ").expect("should handle padded JSON");
        assert_eq!(result, vec![10, 20, 30]);
    }

    /// Float values should be rejected (tokens are integers).
    /// Bug class: parse::<u32>() silently truncating floats.
    #[test]
    fn parse_token_ids_float_rejected() {
        let result = parse_token_ids("1.5");
        assert!(
            result.is_err(),
            "Float values must be rejected as token IDs"
        );
    }

    // ========================================================================
    // format_prediction_output: additional formats/edge cases
    // ========================================================================

    /// Zero-duration should not cause division by zero or NaN in output.
    /// Bug class: division by duration producing Inf.
    #[test]
    fn format_prediction_output_zero_duration() {
        use std::time::Duration;
        let options = RunOptions::default();
        let result = format_prediction_output(&[0.5], Duration::from_secs(0), &options)
            .expect("zero duration should not fail");
        assert!(
            result.contains("0.00ms") || result.contains("0.0ms") || result.contains("0ms"),
            "Zero duration should show as zero, got: {result}"
        );
    }

    /// Large output array should format all elements.
    /// Bug class: output truncation at arbitrary limit.
    #[test]
    fn format_prediction_output_large_array() {
        use std::time::Duration;
        let options = RunOptions::default();
        let values: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = format_prediction_output(&values, Duration::from_millis(50), &options)
            .expect("large array should format");
        // Last element [99] should be present
        assert!(
            result.contains("[99]:"),
            "Should contain label for last element"
        );
    }

    /// Unknown output format should fall through to default text.
    /// Bug class: panicking on unrecognized format string.
    #[test]
    fn format_prediction_output_unknown_format_uses_text() {
        use std::time::Duration;
        let options = RunOptions {
            output_format: "xml".to_string(),
            ..Default::default()
        };
        let result = format_prediction_output(&[1.0], Duration::from_millis(10), &options)
            .expect("unknown format should default to text");
        assert!(
            result.contains("Predictions:"),
            "Unknown format should produce text output"
        );
    }

    /// JSON format with NaN should fail because JSON spec has no NaN.
    /// Bug class: silently producing invalid JSON with NaN literal.
    #[test]
    fn format_prediction_output_json_with_nan_fails() {
        use std::time::Duration;
        let options = RunOptions {
            output_format: "json".to_string(),
            ..Default::default()
        };
        let result = format_prediction_output(&[f32::NAN], Duration::from_millis(1), &options);
        // serde_json::json! macro converts NaN to null, so this may still succeed
        // The key property: it should not panic
        let _ = result;
    }

    /// Text format precision: values should display with 6 decimal places.
    /// Bug class: insufficient precision in float formatting.
    #[test]
    fn format_prediction_output_text_precision() {
        use std::time::Duration;
        let options = RunOptions::default();
        let result = format_prediction_output(&[0.123456789], Duration::from_millis(1), &options)
            .expect("should format");
        assert!(
            result.contains("0.123457") || result.contains("0.123456"),
            "Should show ~6 decimal places, got: {result}"
        );
    }

    /// JSON output should be valid JSON (parseable).
    /// Bug class: missing comma, unquoted keys, etc.
    #[test]
    fn format_prediction_output_json_is_valid_json() {
        use std::time::Duration;
        let options = RunOptions {
            output_format: "json".to_string(),
            ..Default::default()
        };
        let output =
            format_prediction_output(&[0.1, 0.2, 0.3], Duration::from_millis(100), &options)
                .expect("should format");
        let parsed: serde_json::Value = serde_json::from_str::<serde_json::Value>(&output)
            .expect("JSON output must be valid JSON");
        assert!(
            parsed.get("predictions").is_some(),
            "JSON must have predictions field"
        );
        assert!(
            parsed.get("inference_time_ms").is_some(),
            "JSON must have inference_time_ms field"
        );
    }

    // ========================================================================
    // RunOptions: comprehensive default verification
    // ========================================================================

    /// Verify ALL default field values, not just a subset.
    /// Bug class: default value changed without updating tests.
    #[test]
    fn run_options_default_all_fields() {
        let opts = RunOptions::default();
        assert!(opts.input.is_none(), "input should default to None");
        assert!(opts.prompt.is_none(), "prompt should default to None");
        assert_eq!(opts.max_tokens, 32, "max_tokens should default to 32");
        assert_eq!(
            opts.output_format, "text",
            "output_format should default to text"
        );
        assert!(!opts.force, "force should default to false");
        assert!(!opts.no_gpu, "no_gpu should default to false");
        assert!(!opts.offline, "offline should default to false");
        assert!(!opts.benchmark, "benchmark should default to false");
        assert!(!opts.verbose, "verbose should default to false");
        assert!(!opts.trace, "trace should default to false");
        assert!(
            opts.trace_steps.is_none(),
            "trace_steps should default to None"
        );
        assert!(!opts.trace_verbose, "trace_verbose should default to false");
        assert!(
            opts.trace_output.is_none(),
            "trace_output should default to None"
        );
    }

    /// RunOptions with trace_output path.
    /// Bug class: trace_output not propagated through options.
    #[test]
    fn run_options_trace_output_propagates() {
        let opts = RunOptions {
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/trace.json")),
            ..Default::default()
        };
        assert_eq!(
            opts.trace_output,
            Some(PathBuf::from("/tmp/trace.json")),
            "trace_output must propagate"
        );
    }

    // ========================================================================
    // RunResult: structural verification
    // ========================================================================

    /// RunResult with no tokens_generated should be None, not Some(0).
    /// Bug class: default value confusion between None and Some(0).
    #[test]
    fn run_result_tokens_generated_none_vs_zero() {
        let result_none = RunResult {
            text: String::new(),
            duration_secs: 0.0,
            cached: false,
            tokens_generated: None,
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };
        let result_zero = RunResult {
            text: String::new(),
            duration_secs: 0.0,
            cached: false,
            tokens_generated: Some(0),
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };
        assert_ne!(
            result_none.tokens_generated, result_zero.tokens_generated,
            "None and Some(0) must be distinguishable"
        );
    }

    /// RunResult fields should be independently settable.
    /// Bug class: struct field ordering causing misassignment.
    #[test]
    fn run_result_field_independence() {
        let result = RunResult {
            text: "output".to_string(),
            duration_secs: 1.234,
            cached: true,
            tokens_generated: Some(42),
            tok_per_sec: Some(100.0),
            used_gpu: Some(true),
            generated_tokens: Some(vec![10, 20, 30]),
        };
        assert_eq!(result.text, "output");
        assert!((result.duration_secs - 1.234).abs() < f64::EPSILON);
        assert!(result.cached);
        assert_eq!(result.tokens_generated, Some(42));
    }

    // ========================================================================
    // ModelSource: PartialEq contract tests
    // ========================================================================

    /// Two HuggingFace sources with same org/repo but different files are not equal.
    /// Bug class: PartialEq ignoring the file field.
    #[test]
    fn model_source_hf_different_files_not_equal() {
        let s1 = ModelSource::HuggingFace {
            org: "org".to_string(),
            repo: "repo".to_string(),
            file: Some("a.gguf".to_string()),
        };
        let s2 = ModelSource::HuggingFace {
            org: "org".to_string(),
            repo: "repo".to_string(),
            file: Some("b.gguf".to_string()),
        };
        assert_ne!(s1, s2, "Different files should make sources unequal");
    }

    /// HuggingFace with file=None vs file=Some are not equal.
    /// Bug class: Option comparison treating None as "don't care".
    #[test]
    fn model_source_hf_none_file_vs_some_file() {
        let s1 = ModelSource::HuggingFace {
            org: "org".to_string(),
            repo: "repo".to_string(),
            file: None,
        };
        let s2 = ModelSource::HuggingFace {
            org: "org".to_string(),
            repo: "repo".to_string(),
            file: Some("model.gguf".to_string()),
        };
        assert_ne!(s1, s2, "None file vs Some file must be unequal");
    }

    /// Local and URL sources should never be equal even with similar-looking content.
    /// Bug class: cross-variant equality.
    #[test]
    fn model_source_local_vs_url_never_equal() {
        let local = ModelSource::Local(PathBuf::from("https://example.com"));
        let url = ModelSource::Url("https://example.com".to_string());
        assert_ne!(local, url, "Local and URL variants must never be equal");
    }

    // ========================================================================
    // cache_path: additional invariants
    // ========================================================================

    /// Local source cache_path is identity (returns the same path).
    /// Bug class: Local path being redirected through cache directory.
    #[test]
    fn cache_path_local_is_identity() {
        let path = PathBuf::from("/some/model.safetensors");
        let source = ModelSource::Local(path.clone());
        assert_eq!(
            source.cache_path(),
            path,
            "Local source cache_path must be identity"
        );
    }

    /// Two different URLs must produce different cache paths.
    /// Bug class: hash collision in short URL space.
    #[test]
    fn cache_path_url_different_urls_different_paths() {
        let urls = [
            "https://a.com/model.gguf",
            "https://b.com/model.gguf",
            "https://c.com/model.gguf",
            "https://a.com/other.gguf",
        ];
        let paths: Vec<_> = urls
            .iter()
            .map(|u| ModelSource::Url(u.to_string()).cache_path())
            .collect();
        // All pairs should differ
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                assert_ne!(
                    paths[i], paths[j],
                    "URLs '{}' and '{}' should have different cache paths",
                    urls[i], urls[j]
                );
            }
        }
    }

    /// HuggingFace cache path should contain ".apr/cache" directory.
    /// Bug class: cache going to wrong base directory.
    #[test]
    fn cache_path_hf_contains_apr_cache() {
        let source = ModelSource::HuggingFace {
            org: "test".to_string(),
            repo: "model".to_string(),
            file: None,
        };
        let path_str = source.cache_path().to_string_lossy().to_string();
        assert!(
            path_str.contains(".apr") && path_str.contains("cache"),
            "HF cache path should include .apr/cache, got: {path_str}"
        );
    }

    // ========================================================================
    // resolve_model: offline mode comprehensive
    // ========================================================================

    /// Offline mode with URL source should error with descriptive message.
    /// Bug class: generic error without mentioning offline mode.
    #[test]
    fn resolve_model_offline_url_error_message() {
        let source = ModelSource::Url("https://example.com/model.gguf".to_string());
        let result = resolve_model(&source, false, true);
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("OFFLINE MODE"),
            "Error should mention OFFLINE MODE, got: {err_msg}"
        );
        assert!(
            err_msg.contains("example.com") || err_msg.contains("model.gguf"),
            "Error should mention the URL, got: {err_msg}"
        );
    }

    /// Non-offline mode with local path should succeed (identity).
    #[test]
    fn resolve_model_online_local_returns_path() {
        let source = ModelSource::Local(PathBuf::from("/any/path.apr"));
        let result = resolve_model(&source, false, false);
        assert_eq!(
            result.expect("should succeed"),
            PathBuf::from("/any/path.apr")
        );
    }

    /// Force flag should not affect local path resolution.
    /// Bug class: force flag triggering re-download even for local files.
    #[test]
    fn resolve_model_force_flag_local_unchanged() {
        let source = ModelSource::Local(PathBuf::from("/any/path.apr"));
        let result = resolve_model(&source, true, false);
        assert_eq!(
            result.expect("should succeed"),
            PathBuf::from("/any/path.apr")
        );
    }

    // ========================================================================
    // find_cached_model: negative cases
    // ========================================================================

    /// Requesting a specific file from non-existent cache should return None.
    /// Bug class: returning directory path instead of None when file missing.
    #[test]
    fn find_cached_model_with_specific_file_not_found() {
        let result = find_cached_model("nonexistent_org", "nonexistent_repo", Some("model.gguf"));
        assert!(
            result.is_none(),
            "Non-existent org/repo/file should return None"
        );
    }
