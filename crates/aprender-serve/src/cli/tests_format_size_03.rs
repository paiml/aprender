
    // -------------------------------------------------------------------------
    // Format Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(10 * 1024), "10.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.0 MB");
        assert_eq!(format_size(512 * 1024 * 1024), "512.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(7 * 1024 * 1024 * 1024), "7.0 GB");
    }

    // -------------------------------------------------------------------------
    // Benchmark Suite Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_benchmark_suites_not_empty() {
        // BENCHMARK_SUITES is a static const array, verify it has entries
        let suites_len = BENCHMARK_SUITES.len();
        assert!(suites_len > 0, "BENCHMARK_SUITES should not be empty");
        assert!(suites_len >= 5, "Should have at least 5 benchmark suites");
    }

    #[test]
    fn test_benchmark_suites_have_descriptions() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Benchmark name should not be empty");
            assert!(
                !description.is_empty(),
                "Benchmark description should not be empty"
            );
        }
    }

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
        assert!(!validate_suite_name("unknown"));
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("TENSOR_OPS"));
    }

    // -------------------------------------------------------------------------
    // Is Local File Path Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_local_file_path_true() {
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("/home/user/model.gguf"));
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
    }

    #[test]
    fn test_is_local_file_path_false() {
        assert!(!is_local_file_path("llama3:8b"));
        assert!(!is_local_file_path("pacha://model:v1"));
        assert!(!is_local_file_path("hf://meta-llama/Llama-3"));
    }

    // -------------------------------------------------------------------------
    // Home Dir Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_home_dir_returns_path() {
        // This test depends on HOME being set, which it usually is
        let home = home_dir();
        // Just check it doesn't panic - may be None in some environments
        if let Some(path) = home {
            assert!(path.is_absolute() || path.to_string_lossy().starts_with('/'));
        }
    }

    // -------------------------------------------------------------------------
    // Display Model Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_unknown_format() {
        // Empty data with unknown extension
        let result = display_model_info("model.bin", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_extension_but_invalid() {
        // .gguf extension but not valid GGUF data
        let result = display_model_info("model.gguf", &[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Print Info Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_print_info_does_not_panic() {
        // Just ensure it doesn't panic
        print_info();
    }

    // -------------------------------------------------------------------------
    // Run Visualization Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_visualization_does_not_panic() {
        // Run with minimal samples to keep test fast
        run_visualization(false, 10);
        run_visualization(true, 10);
    }

    // -------------------------------------------------------------------------
    // Load Model Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_gguf_model_invalid() {
        let result = load_gguf_model(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_invalid() {
        let result = load_safetensors_model(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Convoy and Saturation Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_no_output() {
        // Just verify it runs without panic
        let result = run_convoy_test(Some("test".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_no_output() {
        let result = run_saturation_test(Some("test".to_string()), None, None);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Benchmark Compare/Regression Tests (file not found cases)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_file_not_found() {
        let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_file_not_found() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            false, // strict
        );
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // run_benchmarks Tests (list mode only - doesn't run cargo bench)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_benchmarks_list_mode() {
        // List mode should succeed without running cargo bench
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_runtime() {
        // List mode with runtime specified
        let result = run_benchmarks(None, true, Some("realizar".to_string()), None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_model() {
        // List mode with model specified
        let result = run_benchmarks(None, true, None, Some("model.gguf".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_with_output() {
        // List mode with output specified
        let result = run_benchmarks(
            None,
            true,
            None,
            None,
            None,
            Some("/tmp/output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Temp File Tests for bench compare/regression (error paths)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_bench_compare_second_file_not_found() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let file1 = dir.join("bench_compare_one.json");

        let mut f1 = std::fs::File::create(&file1).expect("test");
        f1.write_all(b"{}").expect("test");

        let result = run_bench_compare(
            file1.to_str().expect("test"),
            "/nonexistent/file2.json",
            5.0,
        );

        let _ = std::fs::remove_file(&file1);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_bench_regression_second_file_not_found() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let baseline = dir.join("bench_regress_base.json");

        let mut f1 = std::fs::File::create(&baseline).expect("test");
        f1.write_all(b"{}").expect("test");

        let result = run_bench_regression(
            baseline.to_str().expect("test"),
            "/nonexistent/current.json",
            false, // strict
        );

        let _ = std::fs::remove_file(&baseline);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Convoy/Saturation with output file tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_convoy_test_with_output() {
        let dir = std::env::temp_dir();
        let output = dir.join("convoy_output.json");

        let result = run_convoy_test(
            Some("test".to_string()),
            Some("model.gguf".to_string()),
            Some(output.to_str().expect("test").to_string()),
        );

        assert!(result.is_ok());
        assert!(output.exists());

        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_run_saturation_test_with_output() {
        let dir = std::env::temp_dir();
        let output = dir.join("saturation_output.json");

        let result = run_saturation_test(
            Some("test".to_string()),
            Some("model.gguf".to_string()),
            Some(output.to_str().expect("test").to_string()),
        );

        assert!(result.is_ok());
        assert!(output.exists());

        let _ = std::fs::remove_file(&output);
    }

    // -------------------------------------------------------------------------
    // Display Model Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_with_gguf_magic() {
        // Create minimal GGUF data with magic header
        let data = b"GGUF\x03\x00\x00\x00"; // GGUF magic + version 3
        let result = display_model_info("test.gguf", data);
        // Will fail to parse but exercises the GGUF path
        assert!(result.is_err());
    }

    #[test]
    fn test_display_model_info_safetensors_extension() {
        let result = display_model_info("test.safetensors", &[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // APR Format Support Tests (EXTREME TDD)
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_model_info_apr_extension() {
        // Create minimal APR data with magic header + model type
        // APR header: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = display_model_info("test.apr", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_magic() {
        // Test detection via magic bytes, not extension
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = display_model_info("model.bin", &data); // Unknown extension
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_zero_bytes_unknown() {
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00"; // All zeros (not SafeTensors either)
        let result = display_model_info("test.bin", data);
        // Should not error, just show "Unknown"
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // load_apr_model Tests (EXTREME TDD)
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_apr_model_valid() {
        // Valid APR data with magic and model type
        // APR header: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version 1
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_all_recognized_types() {
        // Test recognized APR model types per model_loader mapping
        let type_codes = [
            0x0001u16, // LinearRegression
            0x0002,    // LogisticRegression
            0x0003,    // DecisionTree
            0x0004,    // RandomForest
            0x0005,    // GradientBoosting
            0x0006,    // KMeans
            0x0007,    // PCA
            0x0008,    // NaiveBayes
            0x0009,    // KNN
            0x000A,    // SVM
            0x0010,    // NgramLM
            0x0011,    // TFIDF
            0x0012,    // CountVectorizer
            0x0020,    // NeuralSequential
            0x0021,    // NeuralCustom
            0x0030,    // ContentRecommender
            0x0040,    // MixtureOfExperts
            0x00FF,    // Custom
        ];

        for type_code in type_codes {
            let mut data = vec![0u8; 16];
            data[0..4].copy_from_slice(b"APR\0");
            data[4..6].copy_from_slice(&type_code.to_le_bytes());
            data[6..8].copy_from_slice(&1u16.to_le_bytes());
            let result = load_apr_model(&data);
            assert!(result.is_ok(), "Failed for type code 0x{:04X}", type_code);
        }
    }

    #[test]
    fn test_load_apr_model_invalid_magic() {
        // Wrong magic bytes
        let data = b"GGUFxxxxxxxxxxxxxxxx";
        let result = load_apr_model(data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Expected APR format"));
    }

    #[test]
    fn test_load_apr_model_too_short() {
        // Data too short for format detection
        let data = b"APR"; // Only 3 bytes
        let result = load_apr_model(data);
        assert!(result.is_err());
    }
