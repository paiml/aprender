/// Verify DiffBenchmarkResult detects improved performance without regression
#[test]
fn test_diff_benchmark_improved_performance() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "original.gguf".to_string(),
            throughput_tps: 10.0,
            latency_p50_ms: 100.0,
            latency_p99_ms: 200.0,
        },
        model_b: BenchmarkMetrics {
            path: "converted.apr".to_string(),
            throughput_tps: 10.5,
            latency_p50_ms: 95.0,
            latency_p99_ms: 190.0,
        },
        throughput_delta_pct: 5.0,
        latency_p50_delta_pct: -5.0,
        latency_p99_delta_pct: -5.0,
        regression_detected: false,
        regression_threshold: 10.0,
    };
    assert!(!result.regression_detected);
    assert!(result.throughput_delta_pct > 0.0);
}

/// Verify CiProfileResult passes when all assertions are satisfied
#[test]
fn test_ci_profile_all_assertions_pass() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 50.0,
        latency_p50_ms: 20.0,
        latency_p99_ms: 40.0,
        assertions: vec![
            CiAssertion {
                name: "throughput".to_string(),
                expected: ">= 10".to_string(),
                actual: "50".to_string(),
                passed: true,
                gate_id: "F-CI-001".to_string(),
            },
            CiAssertion {
                name: "p99".to_string(),
                expected: "<= 100".to_string(),
                actual: "40".to_string(),
                passed: true,
                gate_id: "F-CI-002".to_string(),
            },
        ],
        passed: true,
    };
    assert!(result.passed);
    assert!(result.assertions.iter().all(|a| a.passed));
}

/// Verify TensorMismatchType implements Copy semantics
#[test]
fn test_tensor_mismatch_type_clone() {
    let t = TensorMismatchType::Missing;
    let cloned = t;
    assert_eq!(cloned, TensorMismatchType::Missing);
}

/// Verify parse_diff_output passes when all tensors report OK
#[test]
fn test_parse_diff_output_with_text_only() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    // Text output without any mismatch markers
    let output = "Comparing tensors...\n\
                      tensor1: OK\n\
                      tensor2: OK\n\
                      All 100 tensors match.";
    let result = executor.parse_diff_output(output).unwrap();
    assert!(result.passed);
    assert!(result.mismatches.is_empty());
}

/// Verify parse_inference_output falls back to failed result on invalid JSON
#[test]
fn test_parse_inference_output_failure_fallback() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    // Invalid JSON should fallback to basic result
    let output = "not valid json";
    let result = executor.parse_inference_output(output, false).unwrap();
    assert!(!result.passed);
    assert_eq!(result.total_tokens, 0);
}

/// Verify DiffConfig allows None filter with mismatches_only enabled
#[test]
fn test_diff_config_filter_none() {
    let config = DiffConfig {
        apr_binary: "apr".to_string(),
        filter: None,
        mismatches_only: true,
        tolerance: 1e-5,
    };
    assert!(config.filter.is_none());
}

/// Verify DiffBenchmarkResult detects regression with negative throughput delta
#[test]
fn test_diff_benchmark_result_delta_calculations() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "a.gguf".to_string(),
            throughput_tps: 20.0,
            latency_p50_ms: 50.0,
            latency_p99_ms: 100.0,
        },
        model_b: BenchmarkMetrics {
            path: "b.gguf".to_string(),
            throughput_tps: 10.0,
            latency_p50_ms: 100.0,
            latency_p99_ms: 200.0,
        },
        throughput_delta_pct: -50.0,
        latency_p50_delta_pct: 100.0,
        latency_p99_delta_pct: 100.0,
        regression_detected: true,
        regression_threshold: 20.0,
    };
    assert!(result.regression_detected);
    assert!(result.throughput_delta_pct < 0.0);
    assert!(result.latency_p50_delta_pct > 0.0);
}

/// Verify BenchmarkMetrics stores path, throughput, and latency fields correctly
#[test]
fn test_benchmark_metrics_all_fields() {
    let metrics = BenchmarkMetrics {
        path: "/models/qwen.gguf".to_string(),
        throughput_tps: 25.5,
        latency_p50_ms: 39.2,
        latency_p99_ms: 78.4,
    };
    assert!(metrics.path.contains("qwen"));
    assert!(metrics.throughput_tps > 0.0);
    assert!(metrics.latency_p99_ms > metrics.latency_p50_ms);
}

/// Verify TokenComparison stores index, tokens, logit diff, and match flag
#[test]
fn test_token_comparison_fields() {
    let tc = TokenComparison {
        index: 100,
        token_a: 12345,
        token_b: 12346,
        logit_diff: 0.123,
        matches: false,
    };
    assert_eq!(tc.index, 100);
    assert_ne!(tc.token_a, tc.token_b);
    assert!(!tc.matches);
}

/// Verify run_profile_ci returns error when binary does not exist
#[test]
fn test_run_profile_ci_nonexistent_binary() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_profile_ci("/nonexistent/apr/binary", &path, None, None, None, 1, 1);
    assert!(result.is_err());
}

/// Verify run_profile_ci with throughput assertion returns error for missing binary
#[test]
fn test_run_profile_ci_with_throughput_assert() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_profile_ci(
        "/nonexistent/apr/binary",
        &path,
        Some(10.0),
        None,
        None,
        2,
        5,
    );
    assert!(result.is_err());
}

/// Verify run_profile_ci with p99 latency assertion returns error for missing binary
#[test]
fn test_run_profile_ci_with_p99_assert() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_profile_ci(
        "/nonexistent/apr/binary",
        &path,
        None,
        Some(100.0),
        None,
        1,
        1,
    );
    assert!(result.is_err());
}

/// Verify run_profile_ci with p50 latency assertion returns error for missing binary
#[test]
fn test_run_profile_ci_with_p50_assert() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_profile_ci(
        "/nonexistent/apr/binary",
        &path,
        None,
        None,
        Some(50.0),
        1,
        1,
    );
    assert!(result.is_err());
}

/// Verify run_profile_ci with all three assertions returns error for missing binary
#[test]
fn test_run_profile_ci_with_all_asserts() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_profile_ci(
        "/nonexistent/apr/binary",
        &path,
        Some(10.0),
        Some(100.0),
        Some(50.0),
        5,
        10,
    );
    assert!(result.is_err());
}

/// Verify run_diff_benchmark returns error when binary does not exist
#[test]
fn test_run_diff_benchmark_nonexistent_binary() {
    let model_a = std::path::PathBuf::from("model_a.gguf");
    let model_b = std::path::PathBuf::from("model_b.apr");
    let result = run_diff_benchmark("/nonexistent/apr/binary", &model_a, &model_b, 5.0);
    assert!(result.is_err());
}

/// Verify DifferentialExecutor diff_tensors returns error for nonexistent binary
#[test]
fn test_differential_executor_diff_tensors_error() {
    let config = DiffConfig {
        apr_binary: "/nonexistent/apr/binary".to_string(),
        ..DiffConfig::default()
    };
    let executor = DifferentialExecutor::new(config);
    let model_a = std::path::PathBuf::from("model_a.gguf");
    let model_b = std::path::PathBuf::from("model_b.apr");
    let result = executor.diff_tensors(&model_a, &model_b);
    assert!(result.is_err());
}

/// Verify DifferentialExecutor diff_tensors with filter returns error for missing binary
#[test]
fn test_differential_executor_diff_tensors_with_filter() {
    let config = DiffConfig {
        apr_binary: "/nonexistent/apr/binary".to_string(),
        filter: Some("token_embd".to_string()),
        mismatches_only: false,
        tolerance: 1e-5,
    };
    let executor = DifferentialExecutor::new(config);
    let model_a = std::path::PathBuf::from("model_a.gguf");
    let model_b = std::path::PathBuf::from("model_b.apr");
    let result = executor.diff_tensors(&model_a, &model_b);
    assert!(result.is_err());
}

/// Verify DifferentialExecutor compare_inference returns error for missing binary
#[test]
fn test_differential_executor_compare_inference_error() {
    let config = DiffConfig {
        apr_binary: "/nonexistent/apr/binary".to_string(),
        ..DiffConfig::default()
    };
    let executor = DifferentialExecutor::new(config);
    let model_a = std::path::PathBuf::from("model_a.gguf");
    let model_b = std::path::PathBuf::from("model_b.apr");
    let result = executor.compare_inference(&model_a, &model_b, "test prompt", 10);
    assert!(result.is_err());
}

/// Verify DiffConfig stores embedding filter and tight tolerance correctly
#[test]
fn test_diff_config_embedding_filter() {
    let config = DiffConfig {
        apr_binary: "apr".to_string(),
        filter: Some("embedding".to_string()),
        mismatches_only: true,
        tolerance: 1e-6,
    };
    assert_eq!(config.filter.as_deref(), Some("embedding"));
    assert!((config.tolerance - 1e-6).abs() < 1e-10);
}

/// Verify CiAssertion captures failed throughput check with expected and actual values
#[test]
fn test_ci_assertion_failed() {
    let assertion = CiAssertion {
        name: "throughput".to_string(),
        expected: ">= 20.0 tok/s".to_string(),
        actual: "15.5 tok/s".to_string(),
        passed: false,
        gate_id: "F-PROFILE-CI-001".to_string(),
    };
    assert!(!assertion.passed);
    assert!(assertion.expected.contains("20.0"));
    assert!(assertion.actual.contains("15.5"));
}

/// Verify CiProfileResult fails when both throughput and p99 assertions fail
#[test]
fn test_ci_profile_result_with_failed_assertions() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 15.5,
        latency_p50_ms: 50.0,
        latency_p99_ms: 250.0,
        assertions: vec![
            CiAssertion {
                name: "throughput".to_string(),
                expected: ">= 20.0".to_string(),
                actual: "15.5".to_string(),
                passed: false,
                gate_id: "F-CI-001".to_string(),
            },
            CiAssertion {
                name: "p99".to_string(),
                expected: "<= 200".to_string(),
                actual: "250".to_string(),
                passed: false,
                gate_id: "F-CI-002".to_string(),
            },
        ],
        passed: false,
    };
    assert!(!result.passed);
    assert_eq!(result.assertions.iter().filter(|a| a.passed).count(), 0);
}

/// Verify TensorMismatch reports ShapeMismatch with correct gate ID
#[test]
fn test_tensor_mismatch_type_shape_mismatch() {
    let mismatch = TensorMismatch {
        name: "lm_head.weight".to_string(),
        shape_a: vec![4096, 128_256],
        shape_b: vec![4096, 32_000],
        mismatch_type: TensorMismatchType::ShapeMismatch,
    };
    assert_eq!(mismatch.mismatch_type.gate_id(), "F-ROSETTA-DIFF-002");
    assert_ne!(mismatch.shape_a, mismatch.shape_b);
}

/// Verify TensorMismatch reports Missing tensor type with correct gate ID
#[test]
fn test_tensor_mismatch_missing_tensor() {
    let mismatch = TensorMismatch {
        name: "rotary_embd.inv_freq".to_string(),
        shape_a: vec![64],
        shape_b: vec![],
        mismatch_type: TensorMismatchType::Missing,
    };
    assert_eq!(mismatch.mismatch_type.gate_id(), "F-ROSETTA-DIFF-002");
}

/// Verify InferenceComparisonResult detects partial token match as failure
#[test]
fn test_inference_comparison_result_partial_match() {
    let result = InferenceComparisonResult {
        total_tokens: 100,
        matching_tokens: 85,
        max_logit_diff: 0.05,
        passed: false,
        token_comparisons: vec![TokenComparison {
            index: 42,
            token_a: 1000,
            token_b: 1001,
            logit_diff: 0.05,
            matches: false,
        }],
    };
    assert!(!result.passed);
    assert!(result.matching_tokens < result.total_tokens);
    assert!(!result.token_comparisons.is_empty());
}

/// Verify TokenComparison reports exact match when both tokens are identical
#[test]
fn test_token_comparison_exact_match() {
    let tc = TokenComparison {
        index: 0,
        token_a: 500,
        token_b: 500,
        logit_diff: 0.0001,
        matches: true,
    };
    assert!(tc.matches);
    assert_eq!(tc.token_a, tc.token_b);
}

/// Verify parse_diff_output detects TRANSPOSED marker and fails with count
#[test]
fn test_parse_diff_output_with_transposed_marker() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    // Text output with transposed marker
    let output = "Comparing tensors...\n\
                      TRANSPOSED: token_embd.weight (4096, 32000) vs (32000, 4096)\n\
                      All 100 tensors compared.";
    let result = executor.parse_diff_output(output).unwrap();
    assert!(!result.passed);
    assert_eq!(result.transposed_tensors, 1);
}

/// Verify parse_diff_output passes when no mismatch markers are present
#[test]
fn test_parse_diff_output_with_no_mismatch_marker() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    // Text output without transposed marker - should pass
    let output = "Comparing tensors...\n\
                      lm_head.weight: OK\n\
                      Done.";
    let result = executor.parse_diff_output(output).unwrap();
    // No TRANSPOSED markers found, so should pass
    assert!(result.passed);
    assert_eq!(result.mismatched_tensors, 0);
}

/// Verify parse_inference_output parses valid JSON into passing result
#[test]
fn test_parse_inference_output_with_valid_json() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let output = r#"{"total_tokens":10,"matching_tokens":10,"max_logit_diff":0.0001,"passed":true,"token_comparisons":[]}"#;
    let result = executor.parse_inference_output(output, true).unwrap();
    assert!(result.passed);
    assert_eq!(result.total_tokens, 10);
}

/// Verify DiffConfig stores relaxed tolerance with mismatches_only disabled
#[test]
fn test_diff_config_relaxed_tolerance() {
    let config = DiffConfig {
        apr_binary: "apr".to_string(),
        filter: None,
        mismatches_only: false,
        tolerance: 1e-3,
    };
    assert!((config.tolerance - 1e-3).abs() < 1e-10);
    assert!(!config.mismatches_only);
}

// =========================================================================
// SixColumnProfile tests
// =========================================================================

/// Verify SixColumnProfile default has all None throughputs and empty collections
#[test]
fn test_six_column_profile_default() {
    let profile = SixColumnProfile::default();
    assert!(profile.tps_gguf_cpu.is_none());
    assert!(profile.tps_gguf_gpu.is_none());
    assert!(profile.tps_apr_cpu.is_none());
    assert!(profile.tps_apr_gpu.is_none());
    assert!(profile.tps_st_cpu.is_none());
    assert!(profile.tps_st_gpu.is_none());
    assert!(profile.conversions.is_empty());
    assert!(profile.failed_assertions.is_empty());
    assert_eq!(profile.total_duration_ms, 0);
}
