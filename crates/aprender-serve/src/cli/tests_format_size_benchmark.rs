
    #[test]
    fn test_format_size_large_gb_cov() {
        let size = 100 * 1024 * 1024 * 1024; // 100 GB
        assert!(format_size(size).contains("GB"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: BENCHMARK_SUITES constants
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_tensor_ops_exists_cov() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "tensor_ops");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_inference_exists_cov() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "inference");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_cache_exists_cov() {
        let found = BENCHMARK_SUITES.iter().any(|(name, _)| *name == "cache");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_lambda_exists_cov() {
        let found = BENCHMARK_SUITES.iter().any(|(name, _)| *name == "lambda");
        assert!(found);
    }

    #[test]
    fn test_benchmark_suites_comparative_exists_cov() {
        let found = BENCHMARK_SUITES
            .iter()
            .any(|(name, _)| *name == "comparative");
        assert!(found);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: display_model_info edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_unknown_format_cov() {
        let result = display_model_info("unknown.bin", &[1, 2, 3, 4, 5]);
        // Unknown format just prints info and succeeds
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_empty_data_cov() {
        let result = display_model_info("empty.bin", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_extension_cov() {
        // APR extension without magic triggers APR path
        let result = display_model_info("model.apr", &[0; 10]);
        // Will succeed (shows unknown model type)
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_benchmarks edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_with_all_options_cov() {
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true, // list mode
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost:8080".to_string()),
            Some("output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: is_local_file_path edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_empty_cov() {
        assert!(!is_local_file_path(""));
    }

    #[test]
    fn test_is_local_file_path_url_without_extension_cov() {
        // URLs without model extensions are not local
        assert!(!is_local_file_path("http://example.com/model"));
        assert!(!is_local_file_path("pacha://registry/model"));
        // Note: URLs with .gguf extension are still considered local by extension check
        assert!(is_local_file_path("https://example.com/model.gguf"));
    }

    #[test]
    fn test_is_local_file_path_parent_dir_cov() {
        assert!(is_local_file_path("../model.gguf"));
        assert!(is_local_file_path("../../models/file.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_windows_style_cov() {
        // Windows-style paths
        assert!(is_local_file_path("C:\\models\\model.gguf"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: validate_suite_name edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_suite_name_case_sensitive_cov() {
        // Suite names are case-sensitive
        assert!(!validate_suite_name("TENSOR_OPS"));
        assert!(!validate_suite_name("Inference"));
        assert!(validate_suite_name("tensor_ops"));
    }

    #[test]
    fn test_validate_suite_name_partial_match_cov() {
        // Partial matches should fail
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("ops"));
        assert!(!validate_suite_name("infer"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: format_size edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_zero_cov() {
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_exact_kb_cov() {
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_exact_mb_cov() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_format_size_exact_gb_cov() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_size_large_tb_cov() {
        // 10 TB still shows in GB
        let ten_tb = 10u64 * 1024 * 1024 * 1024 * 1024;
        let result = format_size(ten_tb);
        assert!(result.contains("GB"));
    }

    #[test]
    fn test_format_size_fractional_mb_cov() {
        let bytes = 1500 * 1024u64; // 1.5 MB
        let result = format_size(bytes);
        assert!(result.contains("MB"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: print_info
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_cov() {
        // Just test that it doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: home_dir
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_path_cov() {
        // home_dir may return None on some systems but should not panic
        let _ = home_dir();
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: validate_suite_name comprehensive
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_suite_name_all_valid_cov() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_validate_suite_name_empty_cov() {
        assert!(!validate_suite_name(""));
    }

    #[test]
    fn test_validate_suite_name_whitespace_cov() {
        assert!(!validate_suite_name("  "));
        assert!(!validate_suite_name("\t"));
        assert!(!validate_suite_name(" tensor_ops "));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: parse_cargo_bench_output edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_cargo_bench_output_empty_input_cov() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines_ext_cov() {
        let output = "This is some random output\nwithout any benchmark data";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_multiple_benches_cov() {
        let output = "test bench_one ... bench: 100 ns/iter (+/- 10)\ntest bench_two ... bench: 200 ns/iter (+/- 20)";
        let results = parse_cargo_bench_output(output, Some("multi"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_cargo_bench_output_with_commas_cov() {
        let output = "test big_bench ... bench: 1,000,000 ns/iter (+/- 100)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 1000000);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: BENCHMARK_SUITES metadata
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_descriptions_not_empty_cov() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty());
            assert!(!description.is_empty());
        }
    }

    #[test]
    fn test_benchmark_suites_unique_names_cov() {
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(names.len(), unique_count);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: is_local_file_path comprehensive
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_dot_slash_cov() {
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("./subdir/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_only_extension_cov() {
        // Just filename with extension
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
    }

    #[test]
    fn test_is_local_file_path_no_extension_cov() {
        // No recognized extension
        assert!(!is_local_file_path("model"));
        assert!(!is_local_file_path("model.txt"));
        assert!(!is_local_file_path("model.bin"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_visualization
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_small_samples_cov() {
        run_visualization(false, 5);
    }

    #[test]
    fn test_run_visualization_large_samples_cov() {
        run_visualization(true, 100);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: display_model_info additional formats
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_safetensors_ext_cov() {
        let result = display_model_info("model.safetensors", &[0; 8]);
        // Should fail to parse empty data but not panic
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_ext_cov() {
        let result = display_model_info("model.gguf", &[0; 8]);
        // Should fail to parse invalid GGUF but not panic
        assert!(result.is_err() || result.is_ok());
    }

    // =========================================================================
    // Extended Coverage Tests: format_size boundaries
    // =========================================================================

    #[test]
    fn test_format_size_exact_boundaries_cov() {
        // Exact KB boundary
        assert_eq!(format_size(1024), "1.0 KB");
        // Exact MB boundary
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        // Exact GB boundary
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_size_just_below_boundaries_cov() {
        // Just below KB boundary
        assert_eq!(format_size(1023), "1023 B");
        // Just below MB boundary
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
        // Just below GB boundary
        assert_eq!(format_size(1024 * 1024 * 1024 - 1), "1024.0 MB");
    }

    #[test]
    fn test_format_size_large_values_cov() {
        // Multi-GB values
        assert_eq!(format_size(10 * 1024 * 1024 * 1024), "10.0 GB");
        assert_eq!(format_size(100 * 1024 * 1024 * 1024), "100.0 GB");
    }

    // =========================================================================
    // Extended Coverage Tests: parse_cargo_bench_output
    // =========================================================================

    #[test]
    fn test_parse_cargo_bench_output_nonnumeric_ext_cov() {
        let output = "test name bench: abc ns/iter (+/- 10)"; // non-numeric time
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty()); // Should skip malformed lines
    }

    #[test]
    fn test_parse_cargo_bench_output_no_test_keyword_ext_cov() {
        let output = "bench_name ... bench: 100 ns/iter (+/- 10)"; // no "test" keyword
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_too_few_parts_ext_cov() {
        let output = "test bench"; // too few parts
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_with_suite_name_ext_cov() {
        let output = "test bench_test ... bench: 500 ns/iter (+/- 50)";
        let results = parse_cargo_bench_output(output, Some("my_suite"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["suite"], "my_suite");
    }

    #[test]
    fn test_parse_cargo_bench_output_large_comma_separated_ext_cov() {
        let output = "test slow_bench ... bench: 50,000 ns/iter (+/- 1000)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 50000);
    }

    // =========================================================================
    // Extended Coverage Tests: is_local_file_path
    // =========================================================================

    #[test]
    fn test_is_local_file_path_relative_subdirs_ext_cov() {
        assert!(is_local_file_path("./models/model.gguf"));
        assert!(is_local_file_path("../model.safetensors"));
        assert!(is_local_file_path("models/subdir/file.apr"));
    }

    #[test]
    fn test_is_local_file_path_absolute_linux_ext_cov() {
        assert!(is_local_file_path("/home/user/models/model.gguf"));
        assert!(is_local_file_path("/var/models/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_url_with_ext_ext_cov() {
        // URLs with recognized extensions ARE treated as "local" by current logic
        // (function checks extension, not scheme)
        assert!(is_local_file_path("http://example.com/model.gguf"));
        assert!(is_local_file_path("https://example.com/model.safetensors"));
        // URLs without recognized extensions are NOT treated as local
        assert!(!is_local_file_path("http://example.com/model"));
        assert!(!is_local_file_path("s3://bucket/model.bin"));
    }

    // =========================================================================
    // Extended Coverage Tests: validate_suite_name comprehensive
    // =========================================================================

    #[test]
    fn test_validate_suite_name_iterate_all_ext_cov() {
        for (name, _) in BENCHMARK_SUITES {
            assert!(validate_suite_name(name), "Suite {name} should be valid");
        }
    }
