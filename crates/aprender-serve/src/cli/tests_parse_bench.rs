
// ============================================================================
// parse_bench_line / parse_cargo_bench_output extended tests (GH-219)
// ============================================================================

#[test]
fn test_parse_bench_line_valid_simple_p18() {
    let line = "test tensor_add         ... bench:      12345 ns/iter (+/- 100)";
    let result = parse_bench_line(line, None);
    assert!(result.is_some());
    let obj = result.unwrap();
    assert_eq!(obj["name"], "tensor_add");
    assert_eq!(obj["time_ns"], 12345);
    assert!(obj["suite"].is_null());
}

#[test]
fn test_parse_bench_line_suite_propagation_p18() {
    let line = "test matmul_128         ... bench:     145300 ns/iter (+/- 2000)";
    let result = parse_bench_line(line, Some("matrix_ops"));
    assert!(result.is_some());
    let obj = result.unwrap();
    assert_eq!(obj["name"], "matmul_128");
    assert_eq!(obj["time_ns"], 145300);
    assert_eq!(obj["suite"], "matrix_ops");
}

#[test]
fn test_parse_bench_line_comma_separated_ns_p18() {
    let line = "test large_matmul       ... bench:   1,234,567 ns/iter (+/- 50000)";
    let result = parse_bench_line(line, None);
    assert!(result.is_some());
    let obj = result.unwrap();
    assert_eq!(obj["name"], "large_matmul");
    assert_eq!(obj["time_ns"], 1234567);
}

#[test]
fn test_parse_bench_line_no_bench_marker_p18() {
    let line = "test tensor_add         ... ok";
    assert!(parse_bench_line(line, None).is_none());
}

#[test]
fn test_parse_bench_line_missing_ns_iter_p18() {
    let line = "test tensor_add         ... bench:      12345 us/iter";
    assert!(parse_bench_line(line, None).is_none());
}

#[test]
fn test_parse_bench_line_empty_p18() {
    assert!(parse_bench_line("", None).is_none());
}

#[test]
fn test_parse_bench_line_short_line_p18() {
    assert!(parse_bench_line("a b c", None).is_none());
}

#[test]
fn test_parse_bench_line_no_test_keyword_p18() {
    let line = "bench: 12345 ns/iter";
    assert!(parse_bench_line(line, None).is_none());
}

#[test]
fn test_parse_bench_line_nonnumeric_time_p18() {
    let line = "test tensor_add         ... bench:      abc ns/iter";
    assert!(parse_bench_line(line, None).is_none());
}

#[test]
fn test_parse_bench_line_zero_ns_p18() {
    let line = "test noop               ... bench:          0 ns/iter (+/- 0)";
    let result = parse_bench_line(line, None);
    assert!(result.is_some());
    assert_eq!(result.unwrap()["time_ns"], 0);
}

#[test]
fn test_parse_cargo_bench_output_mixed_p18() {
    let output = "\
running 3 tests
test tensor_add         ... bench:      12345 ns/iter (+/- 100)
test tensor_mul         ... bench:      23456 ns/iter (+/- 200)
test ignored            ... ok
test matmul_128         ... bench:     145300 ns/iter (+/- 2000)

test result: ok. 3 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out
";
    let results = parse_cargo_bench_output(output, Some("all"));
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["name"], "tensor_add");
    assert_eq!(results[0]["time_ns"], 12345);
    assert_eq!(results[0]["suite"], "all");
    assert_eq!(results[1]["name"], "tensor_mul");
    assert_eq!(results[2]["name"], "matmul_128");
    assert_eq!(results[2]["time_ns"], 145300);
}

#[test]
fn test_parse_cargo_bench_output_empty_p18() {
    assert!(parse_cargo_bench_output("", None).is_empty());
}

#[test]
fn test_parse_cargo_bench_output_no_bench_lines_p18() {
    let output = "\
running 2 tests
test test_foo ... ok
test test_bar ... ok

test result: ok. 2 passed; 0 failed
";
    assert!(parse_cargo_bench_output(output, None).is_empty());
}

#[test]
fn test_parse_cargo_bench_output_single_p18() {
    let output = "test dot_product        ... bench:       5678 ns/iter (+/- 50)";
    let results = parse_cargo_bench_output(output, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["name"], "dot_product");
    assert_eq!(results[0]["time_ns"], 5678);
}

// ============================================================================
// validate_suite_or_error extended tests (GH-219)
// ============================================================================

#[test]
fn test_validate_suite_or_error_all_valid_p18() {
    for (name, _) in BENCHMARK_SUITES {
        assert!(validate_suite_or_error(name), "Suite '{}' should be valid", name);
    }
}

#[test]
fn test_validate_suite_or_error_invalid_names_p18() {
    assert!(!validate_suite_or_error("nonexistent_suite_xyz"));
    assert!(!validate_suite_or_error(""));
    assert!(!validate_suite_or_error("TENSOR_OPS")); // case sensitive
    assert!(!validate_suite_or_error("tensor ops")); // spaces
}

// ============================================================================
// print_bench_config smoke tests (GH-219)
// ============================================================================

#[test]
fn test_print_bench_config_all_params_p18() {
    print_bench_config("realizar", None, None, None);
    print_bench_config("ollama", Some("llama3.2"), None, None);
    print_bench_config("vllm", Some("model"), Some("http://localhost:8000"), None);
    print_bench_config(
        "realizar",
        Some("model.gguf"),
        Some("http://localhost"),
        Some("results.json"),
    );
}

// ============================================================================
// print_bench_usage smoke test (GH-219)
// ============================================================================

#[test]
fn test_print_bench_usage_no_panic_p18() {
    print_bench_usage();
}

// ============================================================================
// execute_cargo_bench tests (GH-219)
// ============================================================================

#[test]
fn test_execute_cargo_bench_nonexistent_binary_p18() {
    use std::process::Command;
    let mut cmd = Command::new("__nonexistent_binary_p18__");
    let result = execute_cargo_bench(&mut cmd, true);
    assert!(result.is_err());
}

#[test]
fn test_execute_cargo_bench_capture_mode_p18() {
    use std::process::Command;
    let mut cmd = Command::new("echo");
    cmd.arg("test output");
    let result = execute_cargo_bench(&mut cmd, true);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.is_some());
    assert!(output.unwrap().status.success());
}

#[test]
fn test_execute_cargo_bench_stream_mode_p18() {
    use std::process::Command;
    let mut cmd = Command::new("true");
    let result = execute_cargo_bench(&mut cmd, false);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_execute_cargo_bench_stream_failure_p18() {
    use std::process::Command;
    let mut cmd = Command::new("false");
    let result = execute_cargo_bench(&mut cmd, false);
    assert!(result.is_err());
}

// ============================================================================
// display_model_info tests (GH-219)
// ============================================================================

#[test]
fn test_display_model_info_unknown_bytes_p18() {
    let data = vec![0u8; 100];
    let result = display_model_info("test.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_small_data_p18() {
    let data = vec![0u8; 4];
    let result = display_model_info("test.bin", &data);
    assert!(result.is_ok());
}

#[test]
fn test_display_model_info_gguf_extension_p18() {
    // Fake bytes, will fail format parsing but exercises the branch
    let data = vec![0u8; 100];
    let result = display_model_info("model.gguf", &data);
    // May error due to invalid GGUF magic — that's fine, exercises the branch
    let _ = result;
}

#[test]
fn test_display_model_info_safetensors_extension_p18() {
    let data = vec![0u8; 100];
    let result = display_model_info("model.safetensors", &data);
    let _ = result;
}

#[test]
fn test_display_model_info_apr_extension_p18() {
    let data = vec![0u8; 100];
    let result = display_model_info("model.apr", &data);
    let _ = result;
}

// ============================================================================
// run_visualization smoke test (GH-219)
// ============================================================================

#[test]
fn test_run_visualization_color_false_p18() {
    run_visualization(false, 10);
}

#[test]
fn test_run_visualization_color_true_p18() {
    run_visualization(true, 50);
}
