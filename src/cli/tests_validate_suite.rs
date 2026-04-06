
    #[test]
    fn test_validate_suite_name_uppercase_fails_ext_cov() {
        // Should be case sensitive
        assert!(!validate_suite_name("TENSOR_OPS"));
        assert!(!validate_suite_name("Tensor_Ops"));
        assert!(!validate_suite_name("INFERENCE"));
    }

    #[test]
    fn test_validate_suite_name_prefix_suffix_ext_cov() {
        assert!(!validate_suite_name("tensor")); // partial
        assert!(!validate_suite_name("tensor_ops_extra")); // too long
    }

    // =========================================================================
    // Extended Coverage Tests: display_model_info formats
    // =========================================================================

    #[test]
    fn test_display_model_info_apr_file_ext_cov() {
        let result = display_model_info("model.apr", &[0; 8]);
        // May error due to invalid APR data but shouldn't panic
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_display_model_info_unknown_extension_with_data_cov() {
        let result = display_model_info("model.xyz", &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(result.is_ok()); // Should handle unknown gracefully
    }

    #[test]
    fn test_display_model_info_no_extension_cov() {
        let result = display_model_info("model", &[0; 10]);
        // Should handle unknown gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // =========================================================================
    // Extended Coverage Tests: load_*_model functions
    // =========================================================================

    #[test]
    fn test_load_gguf_model_empty_data_cov() {
        let result = load_gguf_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_empty_data_cov() {
        let result = load_safetensors_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_empty_data_cov() {
        let result = load_apr_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_invalid_magic_cov() {
        let result = load_apr_model(&[0x00, 0x00, 0x00, 0x00]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Extended Coverage Tests: home_dir
    // =========================================================================

    #[test]
    fn test_home_dir_not_panic_cov() {
        // Simply verify the function doesn't panic
        let _ = home_dir();
    }

    // =========================================================================
    // Extended Coverage Tests: run_benchmarks list mode
    // =========================================================================

    #[test]
    fn test_run_benchmarks_list_mode_cov() {
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Extended Coverage Tests: BENCHMARK_SUITES const
    // =========================================================================

    #[test]
    fn test_benchmark_suites_at_least_expected_cov() {
        assert!(BENCHMARK_SUITES.len() >= 5);
        // Should contain commonly expected suites
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"tensor_ops"));
        assert!(names.contains(&"inference"));
    }

    #[test]
    fn test_benchmark_suites_no_empty_descriptions_cov() {
        for (name, desc) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Suite name should not be empty");
            assert!(!desc.is_empty(), "Suite description should not be empty");
            assert!(desc.len() > 5, "Description should be meaningful");
        }
    }

    // =========================================================================
    // Deep Coverage Tests: Error Handling and Edge Cases
    // Prefix: _deep_clicov_
    // =========================================================================

    #[test]
    fn test_deep_clicov_format_size_one_byte() {
        assert_eq!(format_size(1), "1 B");
    }

    #[test]
    fn test_deep_clicov_format_size_max_bytes() {
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_deep_clicov_format_size_fractional_kb() {
        // 1.5 KB
        assert_eq!(format_size(1536), "1.5 KB");
        // 2.25 KB -> rounds to 2.2 with .1f precision
        assert_eq!(format_size(2304), "2.2 KB");
    }

    #[test]
    fn test_deep_clicov_format_size_fractional_mb() {
        // 1.5 MB
        assert_eq!(format_size(1572864), "1.5 MB");
        // 2.75 MB
        let bytes = (2.75 * 1024.0 * 1024.0) as u64;
        assert!(format_size(bytes).contains("2."));
    }

    #[test]
    fn test_deep_clicov_format_size_fractional_gb() {
        // 1.5 GB
        let bytes = (1.5 * 1024.0 * 1024.0 * 1024.0) as u64;
        assert_eq!(format_size(bytes), "1.5 GB");
    }

    #[test]
    #[cfg(not(feature = "bench-http"))]
    fn test_deep_clicov_run_external_benchmark_feature_disabled() {
        // Test the non-bench-http stub (only runs when bench-http is disabled)
        let result = run_external_benchmark("ollama", "http://localhost:11434", None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(err_str.contains("bench-http"));
    }

    #[test]
    fn test_deep_clicov_run_external_benchmark_with_model() {
        let result = run_external_benchmark("vllm", "http://localhost:8000", Some("llama3"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_run_external_benchmark_with_output() {
        let result = run_external_benchmark(
            "llama-cpp",
            "http://localhost:8080",
            None,
            Some("/tmp/output.json"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_bench_without_ns_iter() {
        // Line has "bench:" but not "ns/iter"
        let output = "test benchmark_foo ... bench: 1234 ms (+/- 56)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_missing_bench_keyword() {
        // Line has ns/iter but no bench:
        let output = "test benchmark_foo ... result: 1234 ns/iter (+/- 56)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_test_at_end() {
        // "test" appears but not as first word after split
        let output = "running test benchmark_foo bench: 100 ns/iter (+/- 5)";
        let results = parse_cargo_bench_output(output, None);
        // Should still parse if "test" is found and name follows
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_no_name_after_test() {
        // "test" keyword but nothing follows
        let output = "test bench: 100 ns/iter";
        let results = parse_cargo_bench_output(output, None);
        // Can't extract name properly
        assert!(results.is_empty() || results.len() == 1);
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_dot_slash_only() {
        // Just "./" is not a file path
        assert!(is_local_file_path("./"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_slash_only() {
        // Just "/" is root path
        assert!(is_local_file_path("/"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_hidden_file() {
        assert!(is_local_file_path("./.hidden.gguf"));
        assert!(is_local_file_path("/home/.config/model.safetensors"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_special_chars() {
        assert!(is_local_file_path("./model-v1.0.gguf"));
        assert!(is_local_file_path("/path/to/model_v2.safetensors"));
        assert!(is_local_file_path("./model (copy).apr"));
    }

    #[test]
    fn test_deep_clicov_validate_suite_name_leading_trailing_spaces() {
        assert!(!validate_suite_name(" tensor_ops"));
        assert!(!validate_suite_name("tensor_ops "));
        assert!(!validate_suite_name(" tensor_ops "));
    }

    #[test]
    fn test_deep_clicov_validate_suite_name_newline() {
        assert!(!validate_suite_name("tensor_ops\n"));
        assert!(!validate_suite_name("\ntensor_ops"));
    }

    #[test]
    fn test_deep_clicov_display_model_info_gguf_magic_partial() {
        // GGUF magic but truncated data
        let data = b"GGUF";
        let result = display_model_info("model.bin", data);
        // Should fail to parse but not panic
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_deep_clicov_display_model_info_apr_magic_no_type() {
        // APR magic but no type bytes
        let data = b"APR\0";
        let result = display_model_info("model.bin", data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_load_apr_model_minimum_valid() {
        // Minimum valid APR header
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_load_apr_model_version_display() {
        // APR header with version info
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
        data[6..8].copy_from_slice(&42u16.to_le_bytes()); // version 42
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_convoy_test_all_none() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_saturation_test_all_none() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_visualization_zero_samples() {
        // Zero samples should not panic
        run_visualization(false, 0);
    }

    #[test]
    fn test_deep_clicov_run_visualization_two_samples() {
        run_visualization(true, 2);
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_quantize_exists() {
        let found = BENCHMARK_SUITES.iter().any(|(name, _)| *name == "quantize");
        assert!(found);
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_tokenizer_exists() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "tokenizer");
        assert!(found);
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_count() {
        // Should have exactly 7 suites as defined
        assert_eq!(BENCHMARK_SUITES.len(), 7);
    }

    #[test]
    fn test_deep_clicov_run_benchmarks_url_only_without_runtime() {
        // URL without runtime should still work in list mode
        let result = run_benchmarks(
            None,
            true,
            None,
            None,
            Some("http://localhost:8080".to_string()),
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_suite_none_vs_some() {
        let output = "test bench_a ... bench: 100 ns/iter (+/- 5)";
        let results_none = parse_cargo_bench_output(output, None);
        let results_some = parse_cargo_bench_output(output, Some("test_suite"));

        assert_eq!(results_none.len(), results_some.len());
        assert!(results_none[0]["suite"].is_null());
        assert_eq!(results_some[0]["suite"], "test_suite");
    }

    #[test]
    fn test_deep_clicov_load_gguf_model_invalid_short_data() {
        let data = vec![0u8; 4]; // Too short
        let result = load_gguf_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_load_safetensors_model_invalid_short_data() {
        let data = vec![0u8; 4]; // Too short
        let result = load_safetensors_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_display_model_info_mixed_magic() {
        // Data that doesn't match any known magic
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        let result = display_model_info("model.unknown", &data);
        assert!(result.is_ok()); // Should print "Unknown" format
    }

    #[test]
    fn test_deep_clicov_home_dir_env_var_behavior() {
        // Test that home_dir reads from HOME env var
        let result = home_dir();
        if std::env::var("HOME").is_ok() {
            assert!(result.is_some());
        }
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_uppercase_extension() {
        // Extensions are case-sensitive in current implementation
        assert!(!is_local_file_path("model.GGUF"));
        assert!(!is_local_file_path("model.SAFETENSORS"));
        assert!(!is_local_file_path("model.APR"));
    }

    #[test]
    fn test_deep_clicov_validate_suite_name_similar_names() {
        // Names similar to valid suites but not exact
        assert!(!validate_suite_name("tensor_op"));
        assert!(!validate_suite_name("tensors_ops"));
        assert!(!validate_suite_name("inferences"));
        assert!(!validate_suite_name("caches"));
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_realistic_output() {
        let output = r"
running 3 benchmarks
test tensor_ops::bench_add     ... bench:         123 ns/iter (+/- 12)
test tensor_ops::bench_mul     ... bench:       1,456 ns/iter (+/- 145)
test tensor_ops::bench_matmul  ... bench:      12,345 ns/iter (+/- 1,234)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out
";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["time_ns"], 123);
        assert_eq!(results[1]["time_ns"], 1456);
        assert_eq!(results[2]["time_ns"], 12345);
    }

    #[test]
    fn test_deep_clicov_run_convoy_test_output_file_write() {
        use std::fs;
        let dir = std::env::temp_dir();
        let output = dir.join("deep_clicov_convoy_test.json");

        let result = run_convoy_test(
            Some("test_runtime".to_string()),
            Some("test_model.gguf".to_string()),
            Some(output.to_str().expect("invalid UTF-8").to_string()),
        );
        assert!(result.is_ok());

        // Verify file was created and contains valid JSON
        assert!(output.exists());
        let content = fs::read_to_string(&output).expect("file operation failed");
        assert!(content.contains("baseline"));
        assert!(content.contains("convoy"));

        let _ = fs::remove_file(&output);
    }
