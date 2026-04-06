
    #[test]
    fn test_load_apr_model_unknown_type() {
        // Valid magic but unknown model type
        // APR header: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0xFFFEu16.to_le_bytes()); // Unknown type (not 0x00FF)
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = load_apr_model(&data);
        // Should succeed (shows "Unknown" type)
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_empty_data() {
        let result = load_apr_model(&[]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Parse Cargo Bench Output Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_cargo_bench_output_empty() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines() {
        let output = "running 5 tests\ntest test_foo ... ok\ntest test_bar ... ok\n";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_valid_bench_line() {
        let output = "test benchmark_matmul ... bench:      1,234 ns/iter (+/- 56)";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "benchmark_matmul");
        assert_eq!(results[0]["time_ns"], 1234);
        assert_eq!(results[0]["suite"], "tensor_ops");
    }

    #[test]
    fn test_parse_cargo_bench_output_multiple_lines() {
        let output = "\
test benchmark_add ... bench:         100 ns/iter (+/- 5)
test benchmark_mul ... bench:         200 ns/iter (+/- 10)
test benchmark_div ... bench:       1,500 ns/iter (+/- 50)
";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_parse_cargo_bench_output_mixed_content() {
        let output = "\
running 3 benchmarks
test benchmark_foo ... bench:         500 ns/iter (+/- 25)
some other line
test tests::unit_test ... ok
test benchmark_bar ... bench:         750 ns/iter (+/- 30)
";
        let results = parse_cargo_bench_output(output, Some("test_suite"));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "benchmark_foo");
        assert_eq!(results[1]["name"], "benchmark_bar");
    }

    #[test]
    fn test_parse_cargo_bench_output_invalid_time() {
        // Line has "bench:" but no parseable number
        let output = "test benchmark_bad ... bench: invalid ns/iter (+/- 0)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // -------------------------------------------------------------------------
    // Run Benchmarks Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_list_all_params() {
        // List mode with all params populated
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true,
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost:8080".to_string()),
            Some("/tmp/output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Run Visualization Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_large_samples() {
        // Test with larger sample count
        run_visualization(true, 100);
    }

    #[test]
    fn test_run_visualization_single_sample() {
        // Edge case: single sample
        run_visualization(false, 1);
    }

    // -------------------------------------------------------------------------
    // Format Size Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_boundary_kb() {
        // Exactly 1 byte under KB threshold
        assert_eq!(format_size(1023), "1023 B");
        // Exactly at KB threshold
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_boundary_mb() {
        // Exactly 1 byte under MB threshold
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
        // Exactly at MB threshold
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_format_size_boundary_gb() {
        // Exactly at GB threshold
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        // Large GB value
        assert_eq!(format_size(100 * 1024 * 1024 * 1024), "100.0 GB");
    }

    // -------------------------------------------------------------------------
    // Is Local File Path Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_relative() {
        assert!(is_local_file_path("../models/test.gguf"));
        assert!(is_local_file_path("./relative.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_no_extension() {
        // Path without known extension is NOT considered local
        // (must have .gguf, .safetensors, .apr, or start with / or ./)
        assert!(!is_local_file_path("model_without_ext"));
        // But absolute path without extension IS local
        assert!(is_local_file_path("/home/user/model_without_ext"));
        // And relative path starting with ./ IS local
        assert!(is_local_file_path("./model_without_ext"));
    }

    // -------------------------------------------------------------------------
    // Validate Suite Name Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_suite_name_all_suites() {
        // Verify all listed suites are valid
        for (name, _) in BENCHMARK_SUITES {
            assert!(
                validate_suite_name(name),
                "Suite '{}' should be valid",
                name
            );
        }
    }

    #[test]
    fn test_validate_suite_name_case_sensitivity() {
        // Verify case sensitivity
        assert!(!validate_suite_name("TENSOR_OPS"));
        assert!(!validate_suite_name("Tensor_Ops"));
        assert!(!validate_suite_name("TenSor_OpS"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: home_dir
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_value() {
        // home_dir should return Some on most systems
        let result = home_dir();
        // Don't assert it's Some since CI may not have HOME set
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_home_dir_path_is_absolute() {
        if let Some(path) = home_dir() {
            // If HOME is set, the path should be absolute
            assert!(
                path.is_absolute() || path.to_string_lossy().starts_with('/'),
                "Home dir should be absolute path"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: print_info
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_no_panic() {
        // Just verify print_info doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: display_model_info
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_random_bytes() {
        // Random bytes should print "Unknown" format
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let result = display_model_info("model.bin", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_empty_data() {
        let data: Vec<u8> = vec![];
        let result = display_model_info("empty.bin", &data);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: load_apr_model error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_apr_model_wrong_format() {
        // GGUF magic instead of APR
        let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00]; // GGUF magic
        let result = load_apr_model(&data);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_bench_compare
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_missing_files() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 0.1);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_bench_regression
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_regression_missing_files() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false, // strict
        );
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: is_local_file_path edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_apr_extension() {
        assert!(is_local_file_path("model.apr"));
        assert!(is_local_file_path("path/to/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_safetensors_extension() {
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("/absolute/path/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_gguf_extension() {
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("./relative/model.gguf"));
    }

    #[test]
    fn test_is_local_file_path_absolute() {
        assert!(is_local_file_path("/usr/local/models/test"));
        assert!(is_local_file_path("/home/user/model"));
    }

    #[test]
    fn test_is_local_file_path_relative_dot() {
        assert!(is_local_file_path("./model"));
        assert!(is_local_file_path("./subdir/model"));
    }

    #[test]
    fn test_is_local_file_path_hf_style() {
        // HuggingFace style refs should NOT be local
        assert!(!is_local_file_path("meta-llama/Llama-2-7b"));
        assert!(!is_local_file_path("openai/whisper-tiny"));
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: parse_cargo_bench_output
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_cargo_bench_output_empty_cov() {
        let results = parse_cargo_bench_output("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_no_bench_lines_cov() {
        let output = "Running tests...\nAll tests passed\n";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_single_bench_cov() {
        let output = "test tensor_add ... bench: 1,234 ns/iter (+/- 56)";
        let results = parse_cargo_bench_output(output, Some("tensor_ops"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "tensor_add");
        assert_eq!(results[0]["time_ns"], 1234);
        assert_eq!(results[0]["suite"], "tensor_ops");
    }

    #[test]
    fn test_parse_cargo_bench_output_multiple_bench_cov() {
        let output = "test bench_add ... bench: 100 ns/iter (+/- 5)\n\
                      test bench_mul ... bench: 200 ns/iter (+/- 10)";
        let results = parse_cargo_bench_output(output, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parse_cargo_bench_output_malformed_line_cov() {
        // Line that contains bench: but is malformed
        let output = "bench: broken line";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_visualization
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_color_cov() {
        // Run visualization with color enabled
        run_visualization(true, 10);
    }

    #[test]
    fn test_run_visualization_no_color_cov() {
        // Run visualization without color
        run_visualization(false, 10);
    }

    #[test]
    fn test_run_visualization_many_samples_cov() {
        run_visualization(false, 100);
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_convoy_test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_default_runtime_cov() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_convoy_test_with_model_cov() {
        let result = run_convoy_test(
            Some("custom_runtime".to_string()),
            Some("custom_model.gguf".to_string()),
            None,
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: run_saturation_test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_saturation_test_default_runtime_cov() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_with_model_cov() {
        let result = run_saturation_test(
            Some("custom_runtime".to_string()),
            Some("model.gguf".to_string()),
            None,
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Coverage Tests: format_size boundary cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_boundary_kb_cov() {
        // Exactly at KB boundary
        assert_eq!(format_size(1024 - 1), "1023 B");
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_boundary_mb_cov() {
        // Just below MB boundary
        let just_under_mb = 1024 * 1024 - 1;
        assert!(format_size(just_under_mb).contains("KB"));
    }

    #[test]
    fn test_format_size_boundary_gb_cov() {
        // Just below GB boundary
        let just_under_gb = 1024 * 1024 * 1024 - 1;
        assert!(format_size(just_under_gb).contains("MB"));
    }
