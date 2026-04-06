//! EXTREME TDD: CLI Module Coverage Tests
//!
//! Additional tests for cli.rs to increase coverage to 85%+.
//! Tests format_size, display_model_info, benchmark functions, and utility functions.

use realizar::cli::{
    display_model_info, format_size, home_dir, is_local_file_path, print_info, run_bench_compare,
    run_bench_regression, run_benchmarks, run_convoy_test, run_saturation_test, run_visualization,
    validate_suite_name, BENCHMARK_SUITES,
};

// ===== format_size Tests =====

#[test]
fn test_cov_format_size_exact_boundaries() {
    // Exact KB boundary
    assert_eq!(format_size(1024), "1.0 KB");

    // Exact MB boundary
    assert_eq!(format_size(1024 * 1024), "1.0 MB");

    // Exact GB boundary
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
}

#[test]
fn test_cov_format_size_fractional() {
    // 1.5 KB
    assert_eq!(format_size(1536), "1.5 KB");

    // 2.5 MB
    assert_eq!(format_size(2621440), "2.5 MB");

    // 3.5 GB
    assert_eq!(format_size(3758096384), "3.5 GB");
}

#[test]
fn test_cov_format_size_large_values() {
    // 100 GB
    assert_eq!(format_size(107374182400), "100.0 GB");

    // 1 TB (displayed as GB)
    assert_eq!(format_size(1099511627776), "1024.0 GB");
}

#[test]
fn test_cov_format_size_small_values() {
    assert_eq!(format_size(0), "0 B");
    assert_eq!(format_size(1), "1 B");
    assert_eq!(format_size(100), "100 B");
    assert_eq!(format_size(1023), "1023 B");
}

// ===== BENCHMARK_SUITES Tests =====

#[test]
fn test_cov_benchmark_suites_contains_expected() {
    let suite_names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();

    assert!(suite_names.contains(&"tensor_ops"));
    assert!(suite_names.contains(&"inference"));
    assert!(suite_names.contains(&"cache"));
    assert!(suite_names.contains(&"tokenizer"));
    assert!(suite_names.contains(&"quantize"));
}

#[test]
fn test_cov_benchmark_suites_unique_names() {
    let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
    let mut unique_names = names.clone();
    unique_names.sort_unstable();
    unique_names.dedup();
    assert_eq!(
        names.len(),
        unique_names.len(),
        "Benchmark names should be unique"
    );
}

#[test]
fn test_cov_benchmark_suites_descriptions_not_empty() {
    for (name, description) in BENCHMARK_SUITES {
        assert!(!name.is_empty(), "Suite name should not be empty");
        assert!(
            !description.is_empty(),
            "Suite '{}' description should not be empty",
            name
        );
        assert!(
            description.len() > 5,
            "Suite '{}' description should be meaningful",
            name
        );
    }
}

// ===== validate_suite_name Tests =====

#[test]
fn test_cov_validate_suite_name_all_valid() {
    for (name, _) in BENCHMARK_SUITES {
        assert!(
            validate_suite_name(name),
            "Suite '{}' should be valid",
            name
        );
    }
}

#[test]
fn test_cov_validate_suite_name_invalid_cases() {
    assert!(!validate_suite_name(""));
    assert!(!validate_suite_name(" "));
    assert!(!validate_suite_name("TENSOR_OPS")); // case sensitive
    assert!(!validate_suite_name("tensor-ops")); // hyphen vs underscore
    assert!(!validate_suite_name("nonexistent"));
    assert!(!validate_suite_name("inference ")); // trailing space
    assert!(!validate_suite_name(" inference")); // leading space
}

// ===== is_local_file_path Tests =====

#[test]
fn test_cov_is_local_file_path_gguf() {
    assert!(is_local_file_path("model.gguf"));
    assert!(is_local_file_path("./models/model.gguf"));
    assert!(is_local_file_path("/absolute/path/model.gguf"));
    assert!(is_local_file_path("../relative/model.gguf"));
}

#[test]
fn test_cov_is_local_file_path_safetensors() {
    assert!(is_local_file_path("model.safetensors"));
    assert!(is_local_file_path("./model.safetensors"));
    assert!(is_local_file_path("/path/to/model.safetensors"));
}

#[test]
fn test_cov_is_local_file_path_apr() {
    assert!(is_local_file_path("model.apr"));
    assert!(is_local_file_path("./model.apr"));
    assert!(is_local_file_path("/path/to/model.apr"));
}

#[test]
fn test_cov_is_local_file_path_relative() {
    assert!(is_local_file_path("./model.bin"));
    // ../model.bin doesn't match the function's criteria (doesn't start with "./" or "/")
    // models/subdir/file.gguf returns true because it ends with .gguf
    assert!(is_local_file_path("models/subdir/file.gguf"));
}

#[test]
fn test_cov_is_local_file_path_absolute() {
    assert!(is_local_file_path("/home/user/model.gguf"));
    assert!(is_local_file_path("/tmp/model.safetensors"));
}

#[test]
fn test_cov_is_local_file_path_remote_schemes() {
    // Model references that don't look like local files (no extension, no leading ./ or /)
    assert!(!is_local_file_path("llama3:8b"));
    assert!(!is_local_file_path("pacha://model:v1"));
    assert!(!is_local_file_path("hf://meta-llama/Llama-3"));
    // Note: URLs ending in .gguf/.safetensors/.apr will return true due to extension check
    // This is the actual behavior of the function
    assert!(is_local_file_path("http://example.com/model.gguf")); // ends with .gguf
}

// ===== home_dir Tests =====

#[test]
fn test_cov_home_dir_returns_option() {
    let home = home_dir();
    // May be None in some environments, but should not panic
    if let Some(path) = home {
        // If present, should be a valid path
        let path_str = path.to_string_lossy();
        assert!(!path_str.is_empty());
    }
}

// ===== display_model_info Tests =====

#[test]
fn test_cov_display_model_info_empty_data() {
    let result = display_model_info("model.bin", &[]);
    assert!(result.is_ok());
}

#[test]
fn test_cov_display_model_info_random_data() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let result = display_model_info("model.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_display_model_info_apr_extension() {
    // APR magic is "APR\0"
    let data = b"APR\0invalid".to_vec();
    let result = display_model_info("model.apr", &data);
    // May fail parsing, but shouldn't panic
    let _ = result;
}

#[test]
fn test_cov_display_model_info_safetensors_extension_invalid() {
    let result = display_model_info("model.safetensors", &[0, 1, 2, 3]);
    assert!(result.is_err()); // Invalid safetensors format
}

// ===== print_info Tests =====

#[test]
fn test_cov_print_info_no_panic() {
    // Just verify it doesn't panic
    print_info();
}

// ===== run_visualization Tests =====

#[test]
fn test_cov_run_visualization_minimal() {
    run_visualization(false, 5);
}

#[test]
fn test_cov_run_visualization_with_color() {
    run_visualization(true, 5);
}

#[test]
fn test_cov_run_visualization_larger() {
    run_visualization(false, 100);
}

// ===== run_benchmarks Tests =====

#[test]
fn test_cov_run_benchmarks_list_only() {
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_benchmarks_list_with_suite() {
    let result = run_benchmarks(Some("tensor_ops".to_string()), true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_benchmarks_list_with_all_options() {
    let result = run_benchmarks(
        Some("inference".to_string()),
        true,
        Some("realizar".to_string()),
        Some("model.gguf".to_string()),
        Some("http://localhost:8080".to_string()),
        Some("/tmp/output.json".to_string()),
    );
    assert!(result.is_ok());
}

// ===== run_convoy_test Tests =====

#[test]
fn test_cov_run_convoy_test_minimal() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_url() {
    let result = run_convoy_test(Some("http://localhost:8080".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_model() {
    let result = run_convoy_test(None, Some("model.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_output() {
    let output_path = std::env::temp_dir().join("convoy_test_output.json");
    let result = run_convoy_test(
        Some("test".to_string()),
        Some("model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output_path);
}

// ===== run_saturation_test Tests =====

#[test]
fn test_cov_run_saturation_test_minimal() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_url() {
    let result = run_saturation_test(Some("http://localhost:8080".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_model() {
    let result = run_saturation_test(None, Some("model.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_output() {
    let output_path = std::env::temp_dir().join("saturation_test_output.json");
    let result = run_saturation_test(
        Some("test".to_string()),
        Some("model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output_path);
}

// ===== run_bench_compare Tests =====

#[test]
fn test_cov_run_bench_compare_both_missing() {
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 5.0);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_first_exists_second_missing() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_compare_first.json");

    let mut f = std::fs::File::create(&file1).expect("test");
    f.write_all(b"{}").expect("test");

    let result = run_bench_compare(
        file1.to_str().expect("test"),
        "/nonexistent/second.json",
        5.0,
    );

    let _ = std::fs::remove_file(&file1);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_invalid_json() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_inv1.json");
    let file2 = dir.join("bench_cmp_inv2.json");

    let mut f1 = std::fs::File::create(&file1).expect("test");
    f1.write_all(b"{}").expect("test");
    let mut f2 = std::fs::File::create(&file2).expect("test");
    f2.write_all(b"{}").expect("test");

    // Empty JSON is invalid for FullBenchmarkResult - should error
    let result = run_bench_compare(
        file1.to_str().expect("test"),
        file2.to_str().expect("test"),
        10.0,
    );

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);
    // Invalid JSON format should return error
    assert!(result.is_err());
}

// ===== run_bench_regression Tests =====

#[test]
fn test_cov_run_bench_regression_both_missing() {
    let result = run_bench_regression(
        "/nonexistent/baseline.json",
        "/nonexistent/current.json",
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_invalid_json_strict() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_inv_strict.json");
    let current = dir.join("bench_regress_inv_curr.json");

    let mut f1 = std::fs::File::create(&baseline).expect("test");
    f1.write_all(b"{}").expect("test");
    let mut f2 = std::fs::File::create(&current).expect("test");
    f2.write_all(b"{}").expect("test");

    // Empty JSON is invalid - should error
    let result = run_bench_regression(
        baseline.to_str().expect("test"),
        current.to_str().expect("test"),
        true, // strict mode
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_invalid_json_non_strict() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_inv_non.json");
    let current = dir.join("bench_regress_inv_curr_non.json");

    let mut f1 = std::fs::File::create(&baseline).expect("test");
    f1.write_all(b"{}").expect("test");
    let mut f2 = std::fs::File::create(&current).expect("test");
    f2.write_all(b"{}").expect("test");

    // Empty JSON is invalid - should error
    let result = run_bench_regression(
        baseline.to_str().expect("test"),
        current.to_str().expect("test"),
        false, // non-strict mode
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);
    assert!(result.is_err());
}

// ===== Edge Cases =====

#[test]
fn test_cov_format_size_just_under_kb() {
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn test_cov_format_size_just_under_mb() {
    // 1024 * 1024 - 1 = 1048575
    let result = format_size(1048575);
    assert!(result.contains("KB"));
}

#[test]
fn test_cov_format_size_just_under_gb() {
    // 1024 * 1024 * 1024 - 1
    let result = format_size(1073741823);
    assert!(result.contains("MB"));
}

// ===== load_gguf_model Tests =====

#[test]
fn test_cov_load_gguf_model_invalid_magic() {
    // Invalid magic bytes
    let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let result = realizar::cli::load_gguf_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_gguf_model_truncated_header() {
    // GGUF magic but truncated after magic
    // GGUF magic is "GGUF" (0x47, 0x47, 0x55, 0x46)
    let data = vec![0x47, 0x47, 0x55, 0x46]; // Just magic, no version
    let result = realizar::cli::load_gguf_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_gguf_model_empty() {
    let result = realizar::cli::load_gguf_model(&[]);
    assert!(result.is_err());
}

// ===== load_safetensors_model Tests =====

#[test]
fn test_cov_load_safetensors_model_invalid() {
    // Random bytes - not valid safetensors
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
    let result = realizar::cli::load_safetensors_model(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_safetensors_model_empty() {
    let result = realizar::cli::load_safetensors_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_cov_load_safetensors_model_small_header_invalid_json() {
    // SafeTensors header size (small) followed by invalid JSON
    let mut data = vec![0u8; 32];
    // Header size = 16 bytes (reasonable)
    data[0..8].copy_from_slice(&16u64.to_le_bytes());
    // Invalid JSON content after header size
    data[8..24].copy_from_slice(b"not valid json!!");
    let result = realizar::cli::load_safetensors_model(&data);
    assert!(result.is_err());
}

// ===== load_apr_model Tests =====

#[test]
fn test_cov_load_apr_model_valid_linear_regression() {
    // APR magic is "APR\0"
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version
    let result = realizar::cli::load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_load_apr_model_valid_decision_tree() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
    data[6..8].copy_from_slice(&2u16.to_le_bytes()); // version 2
    let result = realizar::cli::load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_load_apr_model_wrong_magic() {
    let data = b"GGUF\x00\x00\x00\x00\x00\x00\x00\x00"; // GGUF magic instead
    let result = realizar::cli::load_apr_model(data);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Expected APR format"));
}

// ===== run_external_benchmark (stub) Tests =====

// Note: Without the bench-http feature, this should return an error
// This test validates the stub behavior
#[test]
#[cfg(not(feature = "bench-http"))]
fn test_cov_run_benchmarks_external_without_feature() {
    // When calling with runtime and url but without bench-http feature,
    // it should error with UnsupportedOperation
    let result = run_benchmarks(
        None,
        false, // not list mode
        Some("ollama".to_string()),
        None,
        Some("http://localhost:11434".to_string()),
        None,
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("bench-http"));
}

// ===== display_model_info Additional Tests =====

#[test]
fn test_cov_display_model_info_apr_valid() {
    // Valid APR with magic
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = display_model_info("model.apr", &data);
    assert!(result.is_ok());
}

#[test]
fn test_cov_display_model_info_gguf_extension_invalid_data() {
    // .gguf extension but random data (not GGUF magic)
    let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let result = display_model_info("model.gguf", &data);
    assert!(result.is_err());
}

#[test]
fn test_cov_display_model_info_safetensors_extension_invalid_data() {
    // .safetensors extension with small header size but invalid JSON after
    let mut data = vec![0u8; 32];
    // Small header size (16 bytes)
    data[0..8].copy_from_slice(&16u64.to_le_bytes());
    // Invalid JSON
    data[8..24].copy_from_slice(b"not valid json!!");
    let result = display_model_info("model.safetensors", &data);
    assert!(result.is_err());
}

#[test]
fn test_cov_display_model_info_unknown_extension_unknown_data() {
    // Unknown extension and unknown data - should show "Unknown" and succeed
    let data = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = display_model_info("model.xyz", &data);
    assert!(result.is_ok());
}

// ===== run_visualization Additional Edge Cases =====

#[test]
fn test_cov_run_visualization_zero_samples() {
    // Zero samples edge case
    run_visualization(false, 0);
}

#[test]
fn test_cov_run_visualization_single_sample() {
    run_visualization(true, 1);
}

#[test]
fn test_cov_run_visualization_many_samples() {
    run_visualization(false, 500);
}

// ===== run_benchmarks Additional Tests =====

#[test]
fn test_cov_run_benchmarks_list_with_url_only() {
    // List mode with url but no runtime
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

// ===== Convoy Test Additional Edge Cases =====

#[test]
fn test_cov_run_convoy_test_all_none() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_convoy_test_with_all_params() {
    let output_path = std::env::temp_dir().join("convoy_all_params.json");
    let result = run_convoy_test(
        Some("realizar".to_string()),
        Some("test-model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    // Verify output file was created
    assert!(output_path.exists());
    let _ = std::fs::remove_file(&output_path);
}

// ===== Saturation Test Additional Edge Cases =====

#[test]
fn test_cov_run_saturation_test_all_none() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_cov_run_saturation_test_with_all_params() {
    let output_path = std::env::temp_dir().join("saturation_all_params.json");
    let result = run_saturation_test(
        Some("realizar".to_string()),
        Some("test-model.gguf".to_string()),
        Some(output_path.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    assert!(output_path.exists());
    let _ = std::fs::remove_file(&output_path);
}

// ===== Bench Compare and Regression Additional Error Path Tests =====
// Note: Valid JSON schema tests are covered in src/bench.rs unit tests
// Here we test the error paths and file handling

#[test]
fn test_cov_run_bench_compare_with_zero_threshold() {
    // Test threshold edge case - both files missing
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 0.0);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_with_large_threshold() {
    // Test threshold edge case - both files missing
    let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 100.0);
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_compare_files_exist_but_invalid_schema() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_invalid_schema1.json");
    let file2 = dir.join("bench_cmp_invalid_schema2.json");

    // Valid JSON but wrong schema
    let json = r#"{"foo": "bar"}"#;

    let mut f1 = std::fs::File::create(&file1).expect("create file1");
    f1.write_all(json.as_bytes()).expect("write file1");

    let mut f2 = std::fs::File::create(&file2).expect("create file2");
    f2.write_all(json.as_bytes()).expect("write file2");

    let result = run_bench_compare(
        file1.to_str().expect("path1"),
        file2.to_str().expect("path2"),
        5.0,
    );

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);

    // Should error because JSON doesn't match schema
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_files_exist_but_invalid_schema() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_inv_schema_base.json");
    let current = dir.join("bench_regress_inv_schema_curr.json");

    // Valid JSON but wrong schema
    let json = r#"{"version": "1.0", "data": []}"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create baseline");
    f1.write_all(json.as_bytes()).expect("write baseline");

    let mut f2 = std::fs::File::create(&current).expect("create current");
    f2.write_all(json.as_bytes()).expect("write current");

    let result = run_bench_regression(
        baseline.to_str().expect("baseline path"),
        current.to_str().expect("current path"),
        false,
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);

    // Should error because JSON doesn't match schema
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_strict_vs_non_strict_threshold() {
    // Test both strict and non-strict modes error on missing files
    let result_strict =
        run_bench_regression("/nonexistent/base.json", "/nonexistent/curr.json", true);
    assert!(result_strict.is_err());

    let result_non_strict =
        run_bench_regression("/nonexistent/base.json", "/nonexistent/curr.json", false);
    assert!(result_non_strict.is_err());
}

#[test]
fn test_cov_run_bench_compare_first_file_parse_error() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_first_parse_err.json");
    let file2 = dir.join("bench_cmp_second_ok.json");

    // First file has invalid JSON
    let json1 = r"not valid json";
    let json2 = r#"{"test": true}"#;

    let mut f1 = std::fs::File::create(&file1).expect("create file1");
    f1.write_all(json1.as_bytes()).expect("write file1");

    let mut f2 = std::fs::File::create(&file2).expect("create file2");
    f2.write_all(json2.as_bytes()).expect("write file2");

    let result = run_bench_compare(
        file1.to_str().expect("path1"),
        file2.to_str().expect("path2"),
        5.0,
    );

    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);

    // Should error because first file has invalid JSON
    assert!(result.is_err());
}

#[test]
fn test_cov_run_bench_regression_first_file_parse_error() {
    use std::io::Write;

    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_regress_base_parse_err.json");
    let current = dir.join("bench_regress_curr_ok.json");

    // First file has invalid JSON
    let json1 = r"{invalid";
    let json2 = r#"{"test": true}"#;

    let mut f1 = std::fs::File::create(&baseline).expect("create baseline");
    f1.write_all(json1.as_bytes()).expect("write baseline");

    let mut f2 = std::fs::File::create(&current).expect("create current");
    f2.write_all(json2.as_bytes()).expect("write current");

    let result = run_bench_regression(
        baseline.to_str().expect("baseline path"),
        current.to_str().expect("current path"),
        false,
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&current);

    // Should error because baseline file has invalid JSON
    assert!(result.is_err());
}

// ===== is_local_file_path Additional Edge Cases =====

#[test]
fn test_cov_is_local_file_path_empty_string() {
    assert!(!is_local_file_path(""));
}

#[test]
fn test_cov_is_local_file_path_just_dot() {
    // Just "." - doesn't match criteria
    assert!(!is_local_file_path("."));
}

#[test]
fn test_cov_is_local_file_path_just_slash() {
    // Just "/" - starts with /
    assert!(is_local_file_path("/"));
}

#[test]
fn test_cov_is_local_file_path_windows_style() {
    // Windows-style path with .gguf extension returns true because of .gguf ending
    // This is the actual function behavior
    assert!(is_local_file_path("C:\\Users\\model.gguf"));
    // Without .gguf extension - not local
    assert!(!is_local_file_path("C:\\Users\\model.bin"));
}

// ===== validate_suite_name Edge Cases =====

#[test]
fn test_cov_validate_suite_name_with_whitespace() {
    assert!(!validate_suite_name(" tensor_ops"));
    assert!(!validate_suite_name("tensor_ops "));
    assert!(!validate_suite_name(" tensor_ops "));
    assert!(!validate_suite_name("\ttensor_ops"));
    assert!(!validate_suite_name("tensor_ops\n"));
}

#[test]
fn test_cov_validate_suite_name_partial_match() {
    // Partial names should not match
    assert!(!validate_suite_name("tensor"));
    assert!(!validate_suite_name("ops"));
    assert!(!validate_suite_name("infer"));
    assert!(!validate_suite_name("cach"));
}

// ===== home_dir Edge Cases =====

#[test]
fn test_cov_home_dir_path_format() {
    if let Some(path) = home_dir() {
        // Home directory should not be empty
        assert!(!path.as_os_str().is_empty());
        // Should be a valid path string
        let path_str = path.to_string_lossy();
        assert!(!path_str.is_empty());
    }
}

// ===== BENCHMARK_SUITES Comprehensive Tests =====

#[test]
fn test_cov_benchmark_suites_count() {
    // Verify we have the expected number of suites
    assert!(
        BENCHMARK_SUITES.len() >= 7,
        "Should have at least 7 benchmark suites"
    );
}

#[test]
fn test_cov_benchmark_suites_specific_entries() {
    // Verify specific suites exist
    let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();

    assert!(names.contains(&"lambda"), "Should have lambda suite");
    assert!(
        names.contains(&"comparative"),
        "Should have comparative suite"
    );
}

// ===== print_info Tests =====

#[test]
fn test_cov_print_info_multiple_calls() {
    // Verify multiple calls don't cause issues
    print_info();
    print_info();
    print_info();
}

// =============================================================================
// COMPREHENSIVE CLI COVERAGE TESTS
// =============================================================================

// ===== Benchmark Result JSON Schema Tests =====

/// Tests for FullBenchmarkResult JSON serialization/deserialization
/// Schema v1.1 per Appendix B:
/// - hardware: {cpu, gpu, memory_gb, storage}
/// - sampling: {method, cv_threshold, cv_at_stop, actual_iterations, warmup_iterations}
/// - thermal: {valid, max_temp_c, temp_variance_c}
/// - cold_start_ms: {median, p99}
mod benchmark_json_schema_tests {
    use realizar::bench::{
        BenchmarkConfig, BenchmarkResults, ColdStartResults, EnergyResults, FullBenchmarkResult,
        HardwareSpec, ItlResults, MemoryResults, QualityValidation, SamplingConfig, ThermalInfo,
        ThroughputResults, TtftResults,
    };

    fn create_test_full_benchmark_result() -> FullBenchmarkResult {
        FullBenchmarkResult {
            version: "1.1".to_string(),
            timestamp: "2025-01-15T12:00:00Z".to_string(),
            config: BenchmarkConfig {
                model: "tinyllama-1.1b".to_string(),
                format: "gguf".to_string(),
                quantization: "Q4_K_M".to_string(),
                runtime: "realizar".to_string(),
                runtime_version: "0.3.5".to_string(),
            },
            hardware: HardwareSpec {
                cpu: "AMD Ryzen 9 5900X".to_string(),
                gpu: Some("NVIDIA RTX 4090".to_string()),
                memory_gb: 64,
                storage: "NVMe SSD".to_string(),
            },
            sampling: SamplingConfig {
                method: "dynamic_cv".to_string(),
                cv_threshold: 0.05,
                actual_iterations: 1000,
                cv_at_stop: 0.03,
                warmup_iterations: 100,
            },
            thermal: ThermalInfo {
                valid: true,
                max_temp_c: 72.5,
                temp_variance_c: 2.3,
            },
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: 25.0,
                    p95: 35.0,
                    p99: 45.0,
                    p999: 60.0,
                },
                itl_ms: ItlResults {
                    median: 8.5,
                    std_dev: 1.2,
                    p99: 15.0,
                },
                throughput_tok_s: ThroughputResults {
                    median: 150.0,
                    ci_95: (145.0, 155.0),
                },
                memory_mb: MemoryResults {
                    model_mb: 512,
                    peak_rss_mb: 1024,
                    kv_waste_pct: 5.0,
                },
                energy: EnergyResults {
                    total_joules: 100.0,
                    token_joules: 0.5,
                    idle_watts: 15.0,
                },
                cold_start_ms: ColdStartResults {
                    median: 150.0,
                    p99: 200.0,
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: 0.02,
                perplexity_wikitext2: Some(8.5),
            },
        }
    }

    #[test]
    fn test_full_benchmark_result_serialization_roundtrip() {
        let original = create_test_full_benchmark_result();
        let json = original.to_json().expect("serialization should succeed");
        let deserialized =
            FullBenchmarkResult::from_json(&json).expect("deserialization should succeed");

        assert_eq!(original.version, deserialized.version);
        assert_eq!(original.timestamp, deserialized.timestamp);
        assert_eq!(original.config.model, deserialized.config.model);
        assert_eq!(original.hardware.cpu, deserialized.hardware.cpu);
        assert_eq!(original.hardware.gpu, deserialized.hardware.gpu);
        assert_eq!(original.hardware.memory_gb, deserialized.hardware.memory_gb);
        assert_eq!(original.hardware.storage, deserialized.hardware.storage);
    }

    #[test]
    fn test_hardware_spec_schema() {
        let hardware = HardwareSpec {
            cpu: "Intel Core i9-13900K".to_string(),
            gpu: Some("NVIDIA RTX 4090".to_string()),
            memory_gb: 128,
            storage: "NVMe SSD".to_string(),
        };

        let json = serde_json::to_string(&hardware).expect("serialization");
        assert!(json.contains("\"cpu\":"));
        assert!(json.contains("\"gpu\":"));
        assert!(json.contains("\"memory_gb\":"));
        assert!(json.contains("\"storage\":"));

        let parsed: HardwareSpec = serde_json::from_str(&json).expect("deserialization");
        assert_eq!(parsed.cpu, hardware.cpu);
        assert_eq!(parsed.gpu, hardware.gpu);
        assert_eq!(parsed.memory_gb, hardware.memory_gb);
        assert_eq!(parsed.storage, hardware.storage);
    }

    #[test]
    fn test_hardware_spec_without_gpu() {
        let hardware = HardwareSpec {
            cpu: "Apple M3 Max".to_string(),
            gpu: None,
            memory_gb: 96,
            storage: "Apple SSD".to_string(),
        };

        let json = serde_json::to_string(&hardware).expect("serialization");
        let parsed: HardwareSpec = serde_json::from_str(&json).expect("deserialization");
        assert!(parsed.gpu.is_none());
    }

    #[test]
    fn test_sampling_config_schema() {
        let sampling = SamplingConfig {
            method: "dynamic_cv".to_string(),
            cv_threshold: 0.05,
            cv_at_stop: 0.03,
            actual_iterations: 500,
            warmup_iterations: 50,
        };

        let json = serde_json::to_string(&sampling).expect("serialization");
        assert!(json.contains("\"method\":"));
        assert!(json.contains("\"cv_threshold\":"));
        assert!(json.contains("\"cv_at_stop\":"));
        assert!(json.contains("\"actual_iterations\":"));
        assert!(json.contains("\"warmup_iterations\":"));

        let parsed: SamplingConfig = serde_json::from_str(&json).expect("deserialization");
        assert_eq!(parsed.method, sampling.method);
        assert!((parsed.cv_threshold - sampling.cv_threshold).abs() < f64::EPSILON);
        assert!((parsed.cv_at_stop - sampling.cv_at_stop).abs() < f64::EPSILON);
        assert_eq!(parsed.actual_iterations, sampling.actual_iterations);
        assert_eq!(parsed.warmup_iterations, sampling.warmup_iterations);
    }

    #[test]
    fn test_thermal_info_schema() {
        let thermal = ThermalInfo {
            valid: true,
            max_temp_c: 85.0,
            temp_variance_c: 3.5,
        };

        let json = serde_json::to_string(&thermal).expect("serialization");
        assert!(json.contains("\"valid\":"));
        assert!(json.contains("\"max_temp_c\":"));
        assert!(json.contains("\"temp_variance_c\":"));

        let parsed: ThermalInfo = serde_json::from_str(&json).expect("deserialization");
        assert_eq!(parsed.valid, thermal.valid);
        assert!((parsed.max_temp_c - thermal.max_temp_c).abs() < f64::EPSILON);
        assert!((parsed.temp_variance_c - thermal.temp_variance_c).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_info_invalid_run() {
        let thermal = ThermalInfo {
            valid: false,
            max_temp_c: 95.0,
            temp_variance_c: 15.0,
        };

        let json = serde_json::to_string(&thermal).expect("serialization");
        let parsed: ThermalInfo = serde_json::from_str(&json).expect("deserialization");
        assert!(!parsed.valid);
    }

    #[test]
    fn test_cold_start_results_schema() {
        let cold_start = ColdStartResults {
            median: 120.0,
            p99: 180.0,
        };

        let json = serde_json::to_string(&cold_start).expect("serialization");
        assert!(json.contains("\"median\":"));
        assert!(json.contains("\"p99\":"));

        let parsed: ColdStartResults = serde_json::from_str(&json).expect("deserialization");
        assert!((parsed.median - cold_start.median).abs() < f64::EPSILON);
        assert!((parsed.p99 - cold_start.p99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_full_benchmark_result_from_json_invalid() {
        let invalid_json = r#"{"not": "valid"}"#;
        let result = FullBenchmarkResult::from_json(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_benchmark_result_from_json_empty() {
        let result = FullBenchmarkResult::from_json("{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_full_benchmark_result_from_json_malformed() {
        let result = FullBenchmarkResult::from_json("{invalid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_full_benchmark_result_all_fields_present() {
        let result = create_test_full_benchmark_result();
        let json = result.to_json().expect("serialization");

        // Verify all top-level fields are present
        assert!(json.contains("\"version\":"));
        assert!(json.contains("\"timestamp\":"));
        assert!(json.contains("\"config\":"));
        assert!(json.contains("\"hardware\":"));
        assert!(json.contains("\"sampling\":"));
        assert!(json.contains("\"thermal\":"));
        assert!(json.contains("\"results\":"));
        assert!(json.contains("\"quality\":"));
    }

    #[test]
    fn test_benchmark_config_serialization() {
        let config = BenchmarkConfig {
            model: "llama-2-7b".to_string(),
            format: "safetensors".to_string(),
            quantization: "F16".to_string(),
            runtime: "realizar".to_string(),
            runtime_version: "0.4.0".to_string(),
        };

        let json = serde_json::to_string(&config).expect("serialization");
        let parsed: BenchmarkConfig = serde_json::from_str(&json).expect("deserialization");

        assert_eq!(parsed.model, config.model);
        assert_eq!(parsed.format, config.format);
        assert_eq!(parsed.quantization, config.quantization);
        assert_eq!(parsed.runtime, config.runtime);
        assert_eq!(parsed.runtime_version, config.runtime_version);
    }

    #[test]
    fn test_ttft_results_all_percentiles() {
        let ttft = TtftResults {
            p50: 20.0,
            p95: 30.0,
            p99: 40.0,
            p999: 55.0,
        };

        let json = serde_json::to_string(&ttft).expect("serialization");
        assert!(json.contains("\"p50\":"));
        assert!(json.contains("\"p95\":"));
        assert!(json.contains("\"p99\":"));
        assert!(json.contains("\"p999\":"));

        let parsed: TtftResults = serde_json::from_str(&json).expect("deserialization");
        assert!((parsed.p50 - ttft.p50).abs() < f64::EPSILON);
        assert!((parsed.p95 - ttft.p95).abs() < f64::EPSILON);
        assert!((parsed.p99 - ttft.p99).abs() < f64::EPSILON);
        assert!((parsed.p999 - ttft.p999).abs() < f64::EPSILON);
    }

    #[test]
    fn test_itl_results_schema() {
        let itl = ItlResults {
            median: 10.0,
            std_dev: 2.0,
            p99: 18.0,
        };

        let json = serde_json::to_string(&itl).expect("serialization");
        let parsed: ItlResults = serde_json::from_str(&json).expect("deserialization");

        assert!((parsed.median - itl.median).abs() < f64::EPSILON);
        assert!((parsed.std_dev - itl.std_dev).abs() < f64::EPSILON);
        assert!((parsed.p99 - itl.p99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_results_with_confidence_interval() {
        let throughput = ThroughputResults {
            median: 200.0,
            ci_95: (190.0, 210.0),
        };

        let json = serde_json::to_string(&throughput).expect("serialization");
        let parsed: ThroughputResults = serde_json::from_str(&json).expect("deserialization");

        assert!((parsed.median - throughput.median).abs() < f64::EPSILON);
        assert!((parsed.ci_95.0 - throughput.ci_95.0).abs() < f64::EPSILON);
        assert!((parsed.ci_95.1 - throughput.ci_95.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_results_schema() {
        let memory = MemoryResults {
            model_mb: 1024,
            peak_rss_mb: 2048,
            kv_waste_pct: 3.5,
        };

        let json = serde_json::to_string(&memory).expect("serialization");
        let parsed: MemoryResults = serde_json::from_str(&json).expect("deserialization");

        assert_eq!(parsed.model_mb, memory.model_mb);
        assert_eq!(parsed.peak_rss_mb, memory.peak_rss_mb);
        assert!((parsed.kv_waste_pct - memory.kv_waste_pct).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_results_schema() {
        let energy = EnergyResults {
            total_joules: 500.0,
            token_joules: 0.25,
            idle_watts: 20.0,
        };

        let json = serde_json::to_string(&energy).expect("serialization");
        let parsed: EnergyResults = serde_json::from_str(&json).expect("deserialization");

        assert!((parsed.total_joules - energy.total_joules).abs() < f64::EPSILON);
        assert!((parsed.token_joules - energy.token_joules).abs() < f64::EPSILON);
        assert!((parsed.idle_watts - energy.idle_watts).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_validation_with_perplexity() {
        let quality = QualityValidation {
            kl_divergence_vs_fp32: 0.015,
            perplexity_wikitext2: Some(7.8),
        };

        let json = serde_json::to_string(&quality).expect("serialization");
        let parsed: QualityValidation = serde_json::from_str(&json).expect("deserialization");

        assert!(
            (parsed.kl_divergence_vs_fp32 - quality.kl_divergence_vs_fp32).abs() < f64::EPSILON
        );
        assert_eq!(parsed.perplexity_wikitext2, quality.perplexity_wikitext2);
    }

    #[test]
    fn test_quality_validation_without_perplexity() {
        let quality = QualityValidation {
            kl_divergence_vs_fp32: 0.03,
            perplexity_wikitext2: None,
        };

        let json = serde_json::to_string(&quality).expect("serialization");
        let parsed: QualityValidation = serde_json::from_str(&json).expect("deserialization");

        assert!(parsed.perplexity_wikitext2.is_none());
    }

    #[test]
    fn test_hardware_spec_default() {
        let default_hw = HardwareSpec::default();
        assert_eq!(default_hw.cpu, "Unknown");
        assert!(default_hw.gpu.is_none());
        assert_eq!(default_hw.memory_gb, 0);
        assert_eq!(default_hw.storage, "Unknown");
    }

    #[test]
    fn test_sampling_config_default() {
        let default_sampling = SamplingConfig::default();
        assert_eq!(default_sampling.method, "dynamic_cv");
        assert!((default_sampling.cv_threshold - 0.05).abs() < f64::EPSILON);
        assert_eq!(default_sampling.actual_iterations, 0);
        assert!((default_sampling.cv_at_stop - 0.0).abs() < f64::EPSILON);
        assert_eq!(default_sampling.warmup_iterations, 100);
    }

    #[test]
    fn test_thermal_info_default() {
        let default_thermal = ThermalInfo::default();
        assert!(default_thermal.valid);
        assert!((default_thermal.temp_variance_c - 0.0).abs() < f64::EPSILON);
        assert!((default_thermal.max_temp_c - 0.0).abs() < f64::EPSILON);
    }
}

// ===== Model Path Validation Tests =====

mod model_path_validation_tests {
    use realizar::cli::is_local_file_path;

    #[test]
    fn test_local_gguf_paths() {
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("../model.gguf"));
        assert!(is_local_file_path("/absolute/path/model.gguf"));
        assert!(is_local_file_path("./subdir/model.gguf"));
        assert!(is_local_file_path("/home/user/models/tinyllama.gguf"));
    }

    #[test]
    fn test_local_safetensors_paths() {
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("./model.safetensors"));
        assert!(is_local_file_path("/path/to/model.safetensors"));
        assert!(is_local_file_path("dir/model.safetensors"));
    }

    #[test]
    fn test_local_apr_paths() {
        assert!(is_local_file_path("model.apr"));
        assert!(is_local_file_path("./classifier.apr"));
        assert!(is_local_file_path("/models/linear_regression.apr"));
    }

    #[test]
    fn test_absolute_paths_any_extension() {
        assert!(is_local_file_path("/home/user/model"));
        assert!(is_local_file_path("/tmp/test.bin"));
        assert!(is_local_file_path("/var/models/custom.weights"));
    }

    #[test]
    fn test_relative_dot_paths_any_extension() {
        assert!(is_local_file_path("./model"));
        assert!(is_local_file_path("./subdir/model.bin"));
        assert!(is_local_file_path("./custom.weights"));
    }

    #[test]
    fn test_remote_model_references() {
        // Model references without local path indicators
        assert!(!is_local_file_path("llama3:8b"));
        assert!(!is_local_file_path("mistral:7b-instruct"));
        assert!(!is_local_file_path("phi3:mini"));
        assert!(!is_local_file_path("hf://meta-llama/Llama-2-7b"));
        assert!(!is_local_file_path("pacha://model-repo:v1.0"));
    }

    #[test]
    fn test_edge_cases() {
        assert!(!is_local_file_path("")); // empty string
        assert!(!is_local_file_path("model")); // no extension, no path prefix
        assert!(!is_local_file_path("just-a-name")); // no extension, no path prefix
        assert!(is_local_file_path("/")); // root path
    }

    #[test]
    fn test_paths_with_spaces() {
        assert!(is_local_file_path("./my model.gguf"));
        assert!(is_local_file_path("/path with spaces/model.safetensors"));
    }

    #[test]
    fn test_paths_with_special_characters() {
        assert!(is_local_file_path("./model-v1.0.gguf"));
        assert!(is_local_file_path("./model_test.gguf"));
        assert!(is_local_file_path("/path/model.v2.gguf"));
    }
}

// ===== Configuration Validation Tests =====

mod config_validation_tests {
    use realizar::cli::{validate_suite_name, BENCHMARK_SUITES};

    #[test]
    fn test_all_defined_suites_are_valid() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(
                validate_suite_name(name),
                "Suite '{}' should be valid",
                name
            );
            assert!(
                !description.is_empty(),
                "Suite '{}' should have a description",
                name
            );
        }
    }

    #[test]
    fn test_invalid_suite_names() {
        assert!(!validate_suite_name("nonexistent"));
        assert!(!validate_suite_name("TENSOR_OPS")); // case matters
        assert!(!validate_suite_name("tensor-ops")); // hyphen vs underscore
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("   "));
    }

    #[test]
    fn test_suite_name_trimming_not_applied() {
        // Whitespace is NOT trimmed by validate_suite_name
        assert!(!validate_suite_name(" tensor_ops"));
        assert!(!validate_suite_name("tensor_ops "));
        assert!(!validate_suite_name(" tensor_ops "));
    }

    #[test]
    fn test_core_suites_exist() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
    }

    #[test]
    fn test_extended_suites_exist() {
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_benchmark_suites_not_empty() {
        assert!(!BENCHMARK_SUITES.is_empty());
        assert!(BENCHMARK_SUITES.len() >= 7);
    }

    #[test]
    fn test_benchmark_suites_unique_names() {
        let mut names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        let original_len = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            original_len,
            "All benchmark suite names should be unique"
        );
    }
}

// ===== CLI Command Error Handling Tests =====

mod cli_error_handling_tests {
    use realizar::cli::{
        display_model_info, load_apr_model, load_gguf_model, load_safetensors_model,
        run_bench_compare, run_bench_regression,
    };

    #[test]
    fn test_load_gguf_model_empty_data() {
        let result = load_gguf_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_model_invalid_magic() {
        let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let result = load_gguf_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_model_truncated() {
        // GGUF magic but nothing else
        let data = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF"
        let result = load_gguf_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_empty_data() {
        let result = load_safetensors_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_invalid_header() {
        // Use a reasonable header size that won't cause capacity overflow
        // Header claims 100 bytes but only 8 bytes provided
        let mut data = vec![0u8; 16];
        data[0..8].copy_from_slice(&100u64.to_le_bytes());
        let result = load_safetensors_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_model_valid_size_invalid_json() {
        // Valid 8-byte header size, but invalid JSON follows
        let mut data = vec![0u8; 32];
        data[0..8].copy_from_slice(&16u64.to_le_bytes());
        data[8..24].copy_from_slice(b"not valid json!!");
        let result = load_safetensors_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_empty_data() {
        let result = load_apr_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_wrong_magic() {
        let data = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = load_apr_model(data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Expected APR format"));
    }

    #[test]
    fn test_load_apr_model_valid_magic_all_types() {
        // APR model type codes
        let type_codes: &[u16] = &[
            0x0001, // LinearRegression
            0x0002, // LogisticRegression
            0x0003, // DecisionTree
            0x0004, // RandomForest
            0x0005, // GradientBoosting
        ];

        for &type_code in type_codes {
            let mut data = vec![0u8; 16];
            data[0..4].copy_from_slice(b"APR\0");
            data[4..6].copy_from_slice(&type_code.to_le_bytes());
            data[6..8].copy_from_slice(&1u16.to_le_bytes());
            let result = load_apr_model(&data);
            assert!(
                result.is_ok(),
                "APR type 0x{:04X} should load successfully",
                type_code
            );
        }
    }

    #[test]
    fn test_display_model_info_empty_file() {
        let result = display_model_info("empty.bin", &[]);
        assert!(result.is_ok()); // Unknown format, but should not error
    }

    #[test]
    fn test_display_model_info_gguf_extension_invalid_content() {
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let result = display_model_info("model.gguf", &data);
        assert!(result.is_err()); // Extension says GGUF but content is not
    }

    #[test]
    fn test_display_model_info_safetensors_extension_invalid_content() {
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let result = display_model_info("model.safetensors", &data);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_model_info_apr_extension_valid_content() {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("model.apr", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bench_compare_missing_files() {
        let result = run_bench_compare("/nonexistent/a.json", "/nonexistent/b.json", 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_bench_regression_missing_files() {
        let result = run_bench_regression(
            "/nonexistent/baseline.json",
            "/nonexistent/current.json",
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_bench_compare_invalid_json_content() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let file1 = dir.join("test_compare_invalid1.json");
        let file2 = dir.join("test_compare_invalid2.json");

        let mut f1 = std::fs::File::create(&file1).expect("create file1");
        f1.write_all(b"{\"invalid\": \"schema\"}")
            .expect("write file1");

        let mut f2 = std::fs::File::create(&file2).expect("create file2");
        f2.write_all(b"{\"also\": \"invalid\"}")
            .expect("write file2");

        let result = run_bench_compare(
            file1.to_str().expect("path1"),
            file2.to_str().expect("path2"),
            5.0,
        );

        let _ = std::fs::remove_file(&file1);
        let _ = std::fs::remove_file(&file2);

        assert!(result.is_err());
    }
}

// ===== Benchmark Functions Additional Tests =====

mod benchmark_function_tests {
    use realizar::cli::{run_benchmarks, run_convoy_test, run_saturation_test, run_visualization};

    #[test]
    fn test_run_benchmarks_list_mode_all_params() {
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true, // list mode
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost:8080".to_string()),
            Some("/tmp/output.json".to_string()),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_benchmarks_list_mode_no_suite() {
        let result = run_benchmarks(None, true, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_visualization_edge_cases() {
        // Zero samples
        run_visualization(false, 0);

        // One sample
        run_visualization(true, 1);

        // Large samples
        run_visualization(false, 1000);
    }

    #[test]
    fn test_run_convoy_test_output_file_created() {
        let output_path = std::env::temp_dir().join("convoy_test_coverage.json");
        let result = run_convoy_test(
            Some("realizar".to_string()),
            Some("test.gguf".to_string()),
            Some(output_path.to_string_lossy().to_string()),
        );

        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify JSON is valid
        let content = std::fs::read_to_string(&output_path).expect("read file");
        let json: serde_json::Value = serde_json::from_str(&content).expect("parse json");
        assert!(json.is_object());

        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_run_saturation_test_output_file_created() {
        let output_path = std::env::temp_dir().join("saturation_test_coverage.json");
        let result = run_saturation_test(
            Some("realizar".to_string()),
            Some("test.gguf".to_string()),
            Some(output_path.to_string_lossy().to_string()),
        );

        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify JSON is valid
        let content = std::fs::read_to_string(&output_path).expect("read file");
        let json: serde_json::Value = serde_json::from_str(&content).expect("parse json");
        assert!(json.is_object());

        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_run_convoy_test_no_output() {
        let result = run_convoy_test(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_saturation_test_no_output() {
        let result = run_saturation_test(None, None, None);
        assert!(result.is_ok());
    }
}

// ===== Format Size Edge Cases =====

mod format_size_tests {
    use realizar::cli::format_size;

    #[test]
    fn test_format_size_zero() {
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_one_byte() {
        assert_eq!(format_size(1), "1 B");
    }

    #[test]
    fn test_format_size_max_bytes() {
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_exact_kb() {
        assert_eq!(format_size(1024), "1.0 KB");
    }

    #[test]
    fn test_format_size_exact_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_format_size_exact_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_size_fractional_kb() {
        assert_eq!(format_size(1536), "1.5 KB"); // 1.5 KB
        assert_eq!(format_size(2048), "2.0 KB"); // 2.0 KB
    }

    #[test]
    fn test_format_size_fractional_mb() {
        assert_eq!(format_size(1572864), "1.5 MB"); // 1.5 MB
        assert_eq!(format_size(5242880), "5.0 MB"); // 5.0 MB
    }

    #[test]
    fn test_format_size_fractional_gb() {
        assert_eq!(format_size(1610612736), "1.5 GB"); // 1.5 GB
        assert_eq!(format_size(10737418240), "10.0 GB"); // 10.0 GB
    }

    #[test]
    fn test_format_size_large_gb() {
        // 100 GB
        assert_eq!(format_size(107374182400), "100.0 GB");
        // 1 TB (shown as 1024 GB)
        assert_eq!(format_size(1099511627776), "1024.0 GB");
    }

    #[test]
    fn test_format_size_boundary_kb_to_mb() {
        // Just under 1 MB
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
    }

    #[test]
    fn test_format_size_boundary_mb_to_gb() {
        // Just under 1 GB
        assert_eq!(format_size(1024 * 1024 * 1024 - 1), "1024.0 MB");
    }
}

// ===== Home Directory Tests =====

mod home_dir_tests {
    use realizar::cli::home_dir;

    #[test]
    fn test_home_dir_no_panic() {
        // Should not panic regardless of environment
        let _ = home_dir();
    }

    #[test]
    fn test_home_dir_returns_path_if_set() {
        if let Some(path) = home_dir() {
            let path_str = path.to_string_lossy();
            assert!(!path_str.is_empty());
            // On Unix, should start with /
            #[cfg(unix)]
            assert!(path_str.starts_with('/'));
        }
    }
}

// ===== External Benchmark Stub Tests =====

#[test]
#[cfg(not(feature = "bench-http"))]
fn test_external_benchmark_requires_feature() {
    use realizar::cli::run_benchmarks;

    let result = run_benchmarks(
        None,
        false, // not list mode
        Some("ollama".to_string()),
        Some("llama3.2".to_string()),
        Some("http://localhost:11434".to_string()),
        None,
    );

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("bench-http"));
}

// ===== Print Info Tests =====

mod print_info_tests {
    use realizar::cli::print_info;

    #[test]
    fn test_print_info_no_panic() {
        print_info();
    }

    #[test]
    fn test_print_info_can_be_called_multiple_times() {
        for _ in 0..5 {
            print_info();
        }
    }
}
