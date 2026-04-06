//! Deep coverage tests for realizar/src/cli.rs
//!
//! This module provides additional coverage for CLI functions
//! and error paths not covered by existing tests.

use realizar::cli::{
    display_model_info, format_size, home_dir, is_local_file_path, load_apr_model, load_gguf_model,
    load_safetensors_model, print_info, run_bench_compare, run_bench_regression, run_benchmarks,
    run_convoy_test, run_saturation_test, run_visualization, validate_suite_name, BENCHMARK_SUITES,
};

// ============================================================================
// Test 1-10: format_size edge cases
// ============================================================================

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
fn test_format_size_exactly_1kb() {
    assert_eq!(format_size(1024), "1.0 KB");
}

#[test]
fn test_format_size_half_kb() {
    // 512 bytes is 0.5 KB but still shows as bytes
    assert_eq!(format_size(512), "512 B");
}

#[test]
fn test_format_size_max_kb() {
    // 1MB - 1 byte = max KB display
    let max_kb = 1024 * 1024 - 1;
    assert_eq!(format_size(max_kb), "1024.0 KB");
}

#[test]
fn test_format_size_exactly_1mb() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
}

#[test]
fn test_format_size_half_gb() {
    let half_gb = 512 * 1024 * 1024;
    assert_eq!(format_size(half_gb), "512.0 MB");
}

#[test]
fn test_format_size_exactly_1gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
}

#[test]
fn test_format_size_large_gb() {
    let large = 100 * 1024 * 1024 * 1024;
    assert_eq!(format_size(large), "100.0 GB");
}

// ============================================================================
// Test 11-20: is_local_file_path patterns
// ============================================================================

#[test]
fn test_is_local_dot_slash_prefix() {
    assert!(is_local_file_path("./model"));
    assert!(is_local_file_path("./"));
    assert!(is_local_file_path("./a/b/c"));
}

#[test]
fn test_is_local_absolute_path() {
    assert!(is_local_file_path("/model"));
    assert!(is_local_file_path("/a/b/c"));
    assert!(is_local_file_path("/"));
}

#[test]
fn test_is_local_gguf_extension() {
    assert!(is_local_file_path("model.gguf"));
    assert!(is_local_file_path("path/to/model.gguf"));
    assert!(is_local_file_path(".gguf"));
}

#[test]
fn test_is_local_safetensors_extension() {
    assert!(is_local_file_path("model.safetensors"));
    assert!(is_local_file_path("deep/path/model.safetensors"));
}

#[test]
fn test_is_local_apr_extension() {
    assert!(is_local_file_path("model.apr"));
    assert!(is_local_file_path("dir/model.apr"));
}

#[test]
fn test_is_not_local_ollama_ref() {
    assert!(!is_local_file_path("llama3:8b"));
    assert!(!is_local_file_path("mistral:latest"));
}

#[test]
fn test_is_not_local_hf_ref() {
    assert!(!is_local_file_path("hf://meta-llama/Llama-3"));
    assert!(!is_local_file_path("meta-llama/Llama-2-7b"));
}

#[test]
fn test_is_not_local_pacha_ref() {
    assert!(!is_local_file_path("pacha://model:v1"));
}

#[test]
fn test_is_local_relative_with_extension() {
    assert!(is_local_file_path("../models/test.gguf"));
    assert!(is_local_file_path("subdir/model.safetensors"));
}

#[test]
fn test_is_not_local_plain_name() {
    // Plain name without known extension or path prefix
    assert!(!is_local_file_path("model"));
    assert!(!is_local_file_path("llama2"));
}

// ============================================================================
// Test 21-30: validate_suite_name
// ============================================================================

#[test]
fn test_validate_suite_tensor_ops() {
    assert!(validate_suite_name("tensor_ops"));
}

#[test]
fn test_validate_suite_inference() {
    assert!(validate_suite_name("inference"));
}

#[test]
fn test_validate_suite_cache() {
    assert!(validate_suite_name("cache"));
}

#[test]
fn test_validate_suite_tokenizer() {
    assert!(validate_suite_name("tokenizer"));
}

#[test]
fn test_validate_suite_quantize() {
    assert!(validate_suite_name("quantize"));
}

#[test]
fn test_validate_suite_lambda() {
    assert!(validate_suite_name("lambda"));
}

#[test]
fn test_validate_suite_comparative() {
    assert!(validate_suite_name("comparative"));
}

#[test]
fn test_validate_suite_invalid() {
    assert!(!validate_suite_name("invalid"));
    assert!(!validate_suite_name(""));
    assert!(!validate_suite_name("TENSOR_OPS"));
}

#[test]
fn test_validate_suite_all_defined_are_valid() {
    for (name, _) in BENCHMARK_SUITES {
        assert!(
            validate_suite_name(name),
            "Suite '{}' should be valid",
            name
        );
    }
}

#[test]
fn test_benchmark_suites_has_entries() {
    assert!(BENCHMARK_SUITES.len() >= 5);
}

// ============================================================================
// Test 31-40: home_dir and print_info
// ============================================================================

#[test]
fn test_home_dir_returns_option() {
    let _home = home_dir();
    // Just verify it doesn't panic
}

#[test]
fn test_home_dir_path_if_set() {
    if let Some(path) = home_dir() {
        // Path should be absolute-ish
        let path_str = path.to_string_lossy();
        assert!(!path_str.is_empty());
    }
}

#[test]
fn test_print_info_no_panic() {
    print_info();
}

#[test]
fn test_run_visualization_with_color() {
    run_visualization(true, 10);
}

#[test]
fn test_run_visualization_without_color() {
    run_visualization(false, 10);
}

#[test]
fn test_run_visualization_minimal_samples() {
    run_visualization(false, 1);
}

#[test]
fn test_run_visualization_many_samples() {
    run_visualization(true, 100);
}

#[test]
fn test_run_visualization_zero_samples() {
    run_visualization(false, 0);
}

#[test]
fn test_benchmark_suites_have_descriptions() {
    for (name, desc) in BENCHMARK_SUITES {
        assert!(!name.is_empty());
        assert!(!desc.is_empty());
    }
}

#[test]
fn test_benchmark_suites_unique_names() {
    let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
    let mut unique_names = names.clone();
    unique_names.sort_unstable();
    unique_names.dedup();
    assert_eq!(
        names.len(),
        unique_names.len(),
        "Duplicate suite names found"
    );
}

// ============================================================================
// Test 41-50: display_model_info edge cases
// ============================================================================

#[test]
fn test_display_model_info_empty_data() {
    let result = display_model_info("test.bin", &[]);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_unknown_format() {
    let data = vec![0u8; 100];
    let result = display_model_info("test.unknown", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_gguf_extension_invalid_data() {
    let data = vec![0, 1, 2, 3]; // Not valid GGUF magic
    let result = display_model_info("test.gguf", &data);
    assert!(result.is_err());
}

#[test]
fn test_display_model_info_safetensors_extension_invalid_data() {
    let data = vec![0, 1, 2, 3];
    let result = display_model_info("test.safetensors", &data);
    assert!(result.is_err());
}

#[test]
fn test_display_model_info_apr_extension_valid() {
    // APR magic: "APR\0"
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = display_model_info("test.apr", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_apr_magic_different_extension() {
    // APR magic but .bin extension
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0002u16.to_le_bytes());
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = display_model_info("test.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_gguf_magic_different_extension() {
    // GGUF magic but .bin extension - will fail to parse since data is too short
    let data = b"GGUF\x03\x00\x00\x00";
    let result = display_model_info("test.bin", data);
    assert!(result.is_err()); // Not enough data for full GGUF
}

#[test]
fn test_display_model_info_random_large_data() {
    let data = vec![42u8; 1000];
    let result = display_model_info("random.dat", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_all_zeros() {
    let data = vec![0u8; 1000];
    let result = display_model_info("zeros.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_all_ones() {
    let data = vec![0xFFu8; 1000];
    let result = display_model_info("ones.bin", &data);
    assert!(result.is_ok());
}

// ============================================================================
// Test 51-60: load_*_model error paths
// ============================================================================

#[test]
fn test_load_gguf_model_empty() {
    let result = load_gguf_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_load_gguf_model_too_short() {
    let result = load_gguf_model(&[0, 1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_load_gguf_model_wrong_magic() {
    let data = b"XXXX\x03\x00\x00\x00";
    let result = load_gguf_model(data);
    assert!(result.is_err());
}

#[test]
fn test_load_safetensors_model_empty() {
    let result = load_safetensors_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_load_safetensors_model_invalid_json() {
    // SafeTensors expects a u64 length prefix then JSON
    let data = b"\x05\x00\x00\x00\x00\x00\x00\x00{{{{{";
    let result = load_safetensors_model(data);
    assert!(result.is_err());
}

#[test]
fn test_load_apr_model_empty() {
    let result = load_apr_model(&[]);
    assert!(result.is_err());
}

#[test]
fn test_load_apr_model_wrong_magic() {
    let data = b"GGUF\x00\x00\x00\x00";
    let result = load_apr_model(data);
    assert!(result.is_err());
}

#[test]
fn test_load_apr_model_valid_linear_regression() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_valid_logistic_regression() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_valid_decision_tree() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

// ============================================================================
// Test 61-70: run_benchmarks list mode
// ============================================================================

#[test]
fn test_run_benchmarks_list_mode() {
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_benchmarks_list_with_suite() {
    let result = run_benchmarks(Some("tensor_ops".to_string()), true, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_benchmarks_list_with_runtime() {
    let result = run_benchmarks(None, true, Some("realizar".to_string()), None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_benchmarks_list_with_model() {
    let result = run_benchmarks(None, true, None, Some("test.gguf".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_benchmarks_list_with_url() {
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
fn test_run_benchmarks_list_with_output() {
    let result = run_benchmarks(
        None,
        true,
        None,
        None,
        None,
        Some("/tmp/test.json".to_string()),
    );
    assert!(result.is_ok());
}

#[test]
fn test_run_benchmarks_list_all_params() {
    let result = run_benchmarks(
        Some("inference".to_string()),
        true,
        Some("ollama".to_string()),
        Some("llama3:8b".to_string()),
        Some("http://localhost:11434".to_string()),
        Some("/tmp/results.json".to_string()),
    );
    assert!(result.is_ok());
}

#[test]
fn test_run_benchmarks_external_without_feature() {
    // When bench-http feature is not enabled, external benchmark returns error
    let result = run_benchmarks(
        None,
        false,
        Some("ollama".to_string()),
        None,
        Some("http://localhost:11434".to_string()),
        None,
    );
    // This should fail because bench-http feature is not enabled
    assert!(result.is_err());
}

#[test]
fn test_run_benchmarks_external_vllm_without_feature() {
    let result = run_benchmarks(
        None,
        false,
        Some("vllm".to_string()),
        None,
        Some("http://localhost:8000".to_string()),
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_benchmarks_external_llama_cpp_without_feature() {
    let result = run_benchmarks(
        None,
        false,
        Some("llama-cpp".to_string()),
        None,
        Some("http://localhost:8080".to_string()),
        None,
    );
    assert!(result.is_err());
}

// ============================================================================
// Test 71-80: run_convoy_test
// ============================================================================

#[test]
fn test_run_convoy_test_no_params() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_convoy_test_with_runtime() {
    let result = run_convoy_test(Some("realizar".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_convoy_test_with_model() {
    let result = run_convoy_test(None, Some("test.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_run_convoy_test_with_output() {
    let dir = std::env::temp_dir();
    let output = dir.join("cli_convoy_test.json");
    let result = run_convoy_test(None, None, Some(output.to_string_lossy().to_string()));
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output);
}

#[test]
fn test_run_convoy_test_all_params() {
    let dir = std::env::temp_dir();
    let output = dir.join("cli_convoy_all.json");
    let result = run_convoy_test(
        Some("test_runtime".to_string()),
        Some("model.gguf".to_string()),
        Some(output.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output);
}

// ============================================================================
// Test 81-90: run_saturation_test
// ============================================================================

#[test]
fn test_run_saturation_test_no_params() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_saturation_test_with_runtime() {
    let result = run_saturation_test(Some("realizar".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_saturation_test_with_model() {
    let result = run_saturation_test(None, Some("test.gguf".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_run_saturation_test_with_output() {
    let dir = std::env::temp_dir();
    let output = dir.join("cli_saturation_test.json");
    let result = run_saturation_test(None, None, Some(output.to_string_lossy().to_string()));
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output);
}

#[test]
fn test_run_saturation_test_all_params() {
    let dir = std::env::temp_dir();
    let output = dir.join("cli_saturation_all.json");
    let result = run_saturation_test(
        Some("test_runtime".to_string()),
        Some("model.gguf".to_string()),
        Some(output.to_string_lossy().to_string()),
    );
    assert!(result.is_ok());
    let _ = std::fs::remove_file(&output);
}

// ============================================================================
// Test 91-100: run_bench_compare and run_bench_regression errors
// ============================================================================

#[test]
fn test_run_bench_compare_file1_not_found() {
    let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_file2_not_found() {
    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_f1.json");
    std::fs::write(&file1, "{}").expect("write");
    let result = run_bench_compare(file1.to_str().unwrap(), "/nonexistent/file2.json", 5.0);
    let _ = std::fs::remove_file(&file1);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_invalid_json_file1() {
    let dir = std::env::temp_dir();
    let file1 = dir.join("bench_cmp_invalid1.json");
    let file2 = dir.join("bench_cmp_invalid2.json");
    std::fs::write(&file1, "not valid json").expect("write");
    std::fs::write(&file2, "{}").expect("write");
    let result = run_bench_compare(file1.to_str().unwrap(), file2.to_str().unwrap(), 5.0);
    let _ = std::fs::remove_file(&file1);
    let _ = std::fs::remove_file(&file2);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_file1_not_found() {
    let result = run_bench_regression(
        "/nonexistent/baseline.json",
        "/nonexistent/current.json",
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_file2_not_found() {
    let dir = std::env::temp_dir();
    let baseline = dir.join("bench_reg_baseline.json");
    std::fs::write(&baseline, "{}").expect("write");
    let result = run_bench_regression(
        baseline.to_str().unwrap(),
        "/nonexistent/current.json",
        false,
    );
    let _ = std::fs::remove_file(&baseline);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_strict_mode() {
    let result = run_bench_regression("/nonexistent/a.json", "/nonexistent/b.json", true);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_normal_mode() {
    let result = run_bench_regression("/nonexistent/a.json", "/nonexistent/b.json", false);
    assert!(result.is_err());
}

// ============================================================================
// Test 101-110: Additional APR model types
// ============================================================================

#[test]
fn test_load_apr_model_random_forest() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0004u16.to_le_bytes()); // RandomForest
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_gradient_boosting() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0005u16.to_le_bytes()); // GradientBoosting
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_kmeans() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0006u16.to_le_bytes()); // KMeans
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_pca() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0007u16.to_le_bytes()); // PCA
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_naive_bayes() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0008u16.to_le_bytes()); // NaiveBayes
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_knn() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0009u16.to_le_bytes()); // KNN
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_svm() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x000Au16.to_le_bytes()); // SVM
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_ngram_lm() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0010u16.to_le_bytes()); // NgramLM
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_tfidf() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0011u16.to_le_bytes()); // TFIDF
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_neural_sequential() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0020u16.to_le_bytes()); // NeuralSequential
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

// ============================================================================
// Test 111-120: More edge cases
// ============================================================================

#[test]
fn test_load_apr_model_moe() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0040u16.to_le_bytes()); // MixtureOfExperts
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_custom() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x00FFu16.to_le_bytes()); // Custom
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_load_apr_model_unknown_type() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0xFFFEu16.to_le_bytes()); // Unknown type
    data[6..8].copy_from_slice(&1u16.to_le_bytes());
    let result = load_apr_model(&data);
    assert!(result.is_ok()); // Should show "Unknown" type
}

#[test]
fn test_load_apr_model_version_2() {
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
    data[6..8].copy_from_slice(&2u16.to_le_bytes()); // version 2
    let result = load_apr_model(&data);
    assert!(result.is_ok());
}

#[test]
fn test_format_size_exact_boundaries() {
    // Test exact boundary conditions
    assert_eq!(format_size(1024 - 1), "1023 B");
    assert_eq!(format_size(1024), "1.0 KB");
    assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
}

#[test]
fn test_is_local_file_path_dots() {
    // Paths with dots
    assert!(is_local_file_path("./model.v2.gguf"));
    assert!(is_local_file_path("/path/to/model.v1.0.safetensors"));
}

#[test]
fn test_is_local_file_path_spaces() {
    // Paths with spaces
    assert!(is_local_file_path("./my model.gguf"));
    assert!(is_local_file_path("/home/user/my models/test.apr"));
}

#[test]
fn test_is_local_file_path_unicode() {
    // Unicode in paths
    assert!(is_local_file_path("./模型.gguf"));
    assert!(is_local_file_path("/путь/к/модели.safetensors"));
}

#[test]
fn test_validate_suite_name_whitespace() {
    assert!(!validate_suite_name(" tensor_ops"));
    assert!(!validate_suite_name("tensor_ops "));
    assert!(!validate_suite_name(" tensor_ops "));
}

#[test]
fn test_run_convoy_with_different_runtimes() {
    for runtime in ["realizar", "ollama", "vllm", "llama-cpp"] {
        let result = run_convoy_test(Some(runtime.to_string()), None, None);
        assert!(result.is_ok());
    }
}
