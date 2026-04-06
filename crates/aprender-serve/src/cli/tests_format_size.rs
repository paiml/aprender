//! T-COV-95 Deep Coverage Bridge: cli/mod.rs remaining paths
//!
//! Targets: parse_cargo_bench_output, run_convoy_test, run_saturation_test,
//! run_bench_compare, run_bench_regression, print_info, load_gguf_model,
//! load_safetensors_model, load_apr_model, run_visualization, format_size,
//! is_local_file_path, validate_suite_name, display_model_info, home_dir,
//! BENCHMARK_SUITES.
//!
//! Refs PMAT-802: Protocol T-COV-95

use super::*;

// ============================================================================
// format_size edge cases
// ============================================================================

#[test]
fn test_format_size_bytes() {
    assert_eq!(format_size(0), "0 B");
    assert_eq!(format_size(1), "1 B");
    assert_eq!(format_size(100), "100 B");
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn test_format_size_kilobytes() {
    assert_eq!(format_size(1024), "1.0 KB");
    assert_eq!(format_size(2048), "2.0 KB");
    assert_eq!(format_size(1024 * 1023), "1023.0 KB");
}

#[test]
fn test_format_size_megabytes() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
    assert_eq!(format_size(1024 * 1024 * 500), "500.0 MB");
}

#[test]
fn test_format_size_gigabytes() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(format_size(7 * 1024 * 1024 * 1024), "7.0 GB");
}

// ============================================================================
// is_local_file_path
// ============================================================================

#[test]
fn test_is_local_file_path_gguf() {
    assert!(is_local_file_path("model.gguf"));
    assert!(is_local_file_path("/path/to/model.gguf"));
}

#[test]
fn test_is_local_file_path_safetensors() {
    assert!(is_local_file_path("model.safetensors"));
}

#[test]
fn test_is_local_file_path_apr() {
    assert!(is_local_file_path("model.apr"));
}

#[test]
fn test_is_local_file_path_relative() {
    assert!(is_local_file_path("./model.bin"));
}

#[test]
fn test_is_local_file_path_absolute() {
    assert!(is_local_file_path("/home/user/model"));
}

#[test]
fn test_is_local_file_path_registry_uri() {
    assert!(!is_local_file_path("llama3:8b"));
}

#[test]
fn test_is_local_file_path_plain_name() {
    assert!(!is_local_file_path("llama3"));
}

// ============================================================================
// validate_suite_name
// ============================================================================

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
    assert!(!validate_suite_name("nonexistent"));
    assert!(!validate_suite_name(""));
    assert!(!validate_suite_name("TENSOR_OPS")); // case sensitive
}

// ============================================================================
// BENCHMARK_SUITES constant
// ============================================================================

#[test]
fn test_benchmark_suites_not_empty() {
    // Verify BENCHMARK_SUITES has content
    assert!(BENCHMARK_SUITES.len() >= 5);
}

#[test]
fn test_benchmark_suites_have_descriptions() {
    for (name, description) in BENCHMARK_SUITES {
        assert!(!name.is_empty());
        assert!(!description.is_empty());
    }
}

// ============================================================================
// home_dir
// ============================================================================

#[test]
fn test_home_dir() {
    let home = home_dir();
    // HOME is set on Linux/Mac
    assert!(home.is_some());
}

// ============================================================================
// print_info (just verify it doesn't panic)
// ============================================================================

#[test]
fn test_print_info_no_panic() {
    print_info();
}

// ============================================================================
// display_model_info
// ============================================================================

#[test]
fn test_display_model_info_gguf() {
    use crate::gguf::test_factory::build_minimal_llama_gguf;
    let file_data = build_minimal_llama_gguf(8, 4, 8, 1, 1);
    let result = display_model_info("test.gguf", &file_data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_unknown_format() {
    let file_data = b"not a valid model format at all";
    let result = display_model_info("test.bin", file_data);
    assert!(result.is_ok()); // Unknown format still succeeds (prints info)
}

#[test]
fn test_display_model_info_by_extension() {
    use crate::gguf::test_factory::build_minimal_llama_gguf;
    let file_data = build_minimal_llama_gguf(8, 4, 8, 1, 1);
    // Even with different extension, magic bytes should be detected
    let result = display_model_info("model.bin", &file_data);
    // GGUF magic present → formats as GGUF
    assert!(result.is_ok());
}

// ============================================================================
// load_gguf_model
// ============================================================================

#[test]
fn test_load_gguf_model_valid() {
    use crate::gguf::test_factory::build_minimal_llama_gguf;
    let file_data = build_minimal_llama_gguf(8, 4, 8, 1, 1);
    let result = load_gguf_model(&file_data);
    assert!(result.is_ok());
}

#[test]
fn test_load_gguf_model_invalid() {
    let result = load_gguf_model(&[0u8; 4]);
    assert!(result.is_err());
}

#[test]
fn test_load_gguf_model_with_many_tensors() {
    use crate::gguf::test_factory::*;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 4)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("t1", &[4], &[1.0; 4])
        .add_f32_tensor("t2", &[4], &[2.0; 4])
        .add_f32_tensor("t3", &[4], &[3.0; 4])
        .add_f32_tensor("t4", &[4], &[4.0; 4])
        .add_f32_tensor("t5", &[4], &[5.0; 4])
        .add_f32_tensor("t6", &[4], &[6.0; 4])
        .add_f32_tensor("t7", &[4], &[7.0; 4])
        .add_f32_tensor("t8", &[4], &[8.0; 4])
        .add_f32_tensor("t9", &[4], &[9.0; 4])
        .add_f32_tensor("t10", &[4], &[10.0; 4])
        .add_f32_tensor("t11", &[4], &[11.0; 4])
        .build();
    let result = load_gguf_model(&data);
    assert!(result.is_ok());
}

// ============================================================================
// load_apr_model
// ============================================================================

#[test]
fn test_load_apr_model_invalid() {
    let result = load_apr_model(&[0u8; 32]);
    assert!(result.is_err());
}

#[test]
fn test_load_apr_model_non_apr_magic() {
    // GGUF data should fail APR check
    use crate::gguf::test_factory::build_minimal_llama_gguf;
    let file_data = build_minimal_llama_gguf(8, 4, 8, 1, 1);
    let result = load_apr_model(&file_data);
    assert!(result.is_err());
}

// ============================================================================
// parse_cargo_bench_output (private function accessible through module)
// ============================================================================

#[test]
fn test_parse_bench_output_valid() {
    let output = "test bench_tensor_add ... bench:       123 ns/iter (+/- 45)\ntest bench_matmul    ... bench:     4,567 ns/iter (+/- 890)\n";
    let results = parse_cargo_bench_output(output, Some("tensor_ops"));
    assert_eq!(results.len(), 2);
}

#[test]
fn test_parse_bench_output_empty() {
    let results = parse_cargo_bench_output("", None);
    assert!(results.is_empty());
}

#[test]
fn test_parse_bench_output_no_bench_lines() {
    let output = "Compiling realizarr v0.3.5\nRunning tests\n";
    let results = parse_cargo_bench_output(output, None);
    assert!(results.is_empty());
}

#[test]
fn test_parse_bench_output_single_result() {
    let output = "test benchmark_name ... bench:       500 ns/iter (+/- 10)\n";
    let results = parse_cargo_bench_output(output, Some("test"));
    assert_eq!(results.len(), 1);
}

// ============================================================================
// run_convoy_test
// ============================================================================

#[test]
fn test_run_convoy_test_default() {
    let result = run_convoy_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_convoy_test_with_runtime() {
    let result = run_convoy_test(Some("ollama".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_convoy_test_with_model() {
    let result = run_convoy_test(None, Some("llama3.2".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_run_convoy_test_with_output() {
    let output_path = "/tmp/test_convoy_cov95.json";
    let result = run_convoy_test(None, None, Some(output_path.to_string()));
    let _ = std::fs::remove_file(output_path);
    assert!(result.is_ok());
}

// ============================================================================
// run_saturation_test
// ============================================================================

#[test]
fn test_run_saturation_test_default() {
    let result = run_saturation_test(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_saturation_test_with_runtime() {
    let result = run_saturation_test(Some("vllm".to_string()), None, None);
    assert!(result.is_ok());
}

#[test]
fn test_run_saturation_test_with_model() {
    let result = run_saturation_test(None, Some("phi-2".to_string()), None);
    assert!(result.is_ok());
}

#[test]
fn test_run_saturation_test_with_output() {
    let output_path = "/tmp/test_saturation_cov95.json";
    let result = run_saturation_test(None, None, Some(output_path.to_string()));
    let _ = std::fs::remove_file(output_path);
    assert!(result.is_ok());
}

// ============================================================================
// run_bench_compare
// ============================================================================

fn make_test_bench_json() -> String {
    serde_json::json!({
        "version": "1.1",
        "timestamp": "2026-01-30T00:00:00Z",
        "config": {
            "model_name": "test",
            "model_path": "test.gguf",
            "quantization": "Q4_K",
            "context_length": 2048,
            "batch_size": 1,
            "prompt": "test",
            "max_tokens": 50
        },
        "hardware": {
            "cpu": "test",
            "gpu": null,
            "ram_gb": 32,
            "os": "linux"
        },
        "sampling": {
            "method": "dynamic_cv",
            "cv_threshold": 0.05,
            "actual_iterations": 100,
            "cv_at_stop": 0.03,
            "warmup_iterations": 10
        },
        "thermal": {
            "valid": true,
            "temp_variance_c": 1.0,
            "max_temp_c": 65.0
        },
        "results": {
            "ttft_ms": {"p50": 50.0, "p95": 80.0, "p99": 100.0, "p999": 120.0},
            "itl_ms": {"median": 10.0, "std_dev": 2.0, "p99": 20.0},
            "throughput_tok_s": {"median": 100.0, "ci_95": [95.0, 105.0]},
            "memory_mb": {"model_mb": 500, "peak_rss_mb": 1000, "kv_waste_pct": 5.0},
            "energy": {"total_joules": 10.0, "token_joules": 0.1, "idle_watts": 50.0},
            "cold_start_ms": {"median": 200.0, "p99": 300.0}
        },
        "quality": {
            "kl_divergence_vs_fp32": 0.01,
            "perplexity_wikitext2": null
        }
    })
    .to_string()
}

#[test]
fn test_run_bench_compare_file_not_found() {
    let result = run_bench_compare("/nonexistent/file1.json", "/nonexistent/file2.json", 5.0);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_invalid_json() {
    let path1 = "/tmp/test_bench_cmp1_cov95.json";
    let path2 = "/tmp/test_bench_cmp2_cov95.json";

    std::fs::write(path1, "not json").unwrap();
    std::fs::write(path2, "not json").unwrap();

    let result = run_bench_compare(path1, path2, 5.0);
    let _ = std::fs::remove_file(path1);
    let _ = std::fs::remove_file(path2);

    // Invalid JSON should error
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_first_file_missing() {
    let path2 = "/tmp/test_bench_cmp2b_cov95.json";
    std::fs::write(path2, "{}").unwrap();
    let result = run_bench_compare("/nonexistent/first.json", path2, 5.0);
    let _ = std::fs::remove_file(path2);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_compare_second_file_missing() {
    let path1 = "/tmp/test_bench_cmp1b_cov95.json";
    std::fs::write(path1, "{}").unwrap();
    let result = run_bench_compare(path1, "/nonexistent/second.json", 5.0);
    let _ = std::fs::remove_file(path1);
    assert!(result.is_err());
}

// ============================================================================
// run_bench_regression
// ============================================================================

#[test]
fn test_run_bench_regression_file_not_found() {
    let result = run_bench_regression(
        "/nonexistent/baseline.json",
        "/nonexistent/current.json",
        false, // strict
    );
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_invalid_json() {
    let path1 = "/tmp/test_bench_reg_inv_base_cov95.json";
    let path2 = "/tmp/test_bench_reg_inv_curr_cov95.json";

    std::fs::write(path1, "not json").unwrap();
    std::fs::write(path2, "not json").unwrap();

    let result = run_bench_regression(path1, path2, false);
    let _ = std::fs::remove_file(path1);
    let _ = std::fs::remove_file(path2);

    assert!(result.is_err());
}

include!("benchmarks_1.rs");
