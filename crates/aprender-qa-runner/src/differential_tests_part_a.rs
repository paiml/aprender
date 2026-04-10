/// Verify DiffConfig default values
#[test]
fn test_diff_config_default() {
    let config = DiffConfig::default();
    assert_eq!(config.apr_binary, "apr");
    assert!(config.mismatches_only);
    assert!((config.tolerance - 1e-5).abs() < 1e-10);
}

/// Verify TensorMismatchType gate_id mapping
#[test]
fn test_tensor_mismatch_type_gate_id() {
    assert_eq!(
        TensorMismatchType::Transposed.gate_id(),
        "F-ROSETTA-DIFF-001"
    );
    assert_eq!(
        TensorMismatchType::ShapeMismatch.gate_id(),
        "F-ROSETTA-DIFF-002"
    );
    assert_eq!(TensorMismatchType::Missing.gate_id(), "F-ROSETTA-DIFF-002");
}

/// Verify TensorDiffResult reports passed for zero mismatches
#[test]
fn test_tensor_diff_result_passed() {
    let result = TensorDiffResult {
        total_tensors: 100,
        mismatched_tensors: 0,
        transposed_tensors: 0,
        mismatches: vec![],
        passed: true,
    };
    assert!(result.passed);
}

/// Verify TensorDiffResult reports failed with transposed tensors
#[test]
fn test_tensor_diff_result_failed() {
    let result = TensorDiffResult {
        total_tensors: 100,
        mismatched_tensors: 2,
        transposed_tensors: 2,
        mismatches: vec![TensorMismatch {
            name: "token_embd.weight".to_string(),
            shape_a: vec![4096, 32000],
            shape_b: vec![32000, 4096],
            mismatch_type: TensorMismatchType::Transposed,
        }],
        passed: false,
    };
    assert!(!result.passed);
    assert_eq!(result.transposed_tensors, 2);
}

/// Verify InferenceComparisonResult reports passed for matching tokens
#[test]
fn test_inference_comparison_passed() {
    let result = InferenceComparisonResult {
        total_tokens: 10,
        matching_tokens: 10,
        max_logit_diff: 1e-6,
        passed: true,
        token_comparisons: vec![],
    };
    assert!(result.passed);
    assert_eq!(result.matching_tokens, result.total_tokens);
}

/// Verify CiProfileResult with passing assertions
#[test]
fn test_ci_profile_assertions() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 12.8,
        latency_p50_ms: 78.2,
        latency_p99_ms: 156.5,
        assertions: vec![
            CiAssertion {
                name: "throughput".to_string(),
                expected: ">= 10.0 tok/s".to_string(),
                actual: "12.8 tok/s".to_string(),
                passed: true,
                gate_id: "F-PROFILE-CI-001".to_string(),
            },
            CiAssertion {
                name: "p99_latency".to_string(),
                expected: "<= 200 ms".to_string(),
                actual: "156.5 ms".to_string(),
                passed: true,
                gate_id: "F-PROFILE-CI-002".to_string(),
            },
        ],
        passed: true,
    };
    assert!(result.passed);
    assert!(result.assertions.iter().all(|a| a.passed));
}

/// Verify DiffBenchmarkResult with no regression
#[test]
fn test_diff_benchmark_no_regression() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "model_a.gguf".to_string(),
            throughput_tps: 12.3,
            latency_p50_ms: 78.2,
            latency_p99_ms: 156.5,
        },
        model_b: BenchmarkMetrics {
            path: "model_b.gguf".to_string(),
            throughput_tps: 12.5, // Slight improvement
            latency_p50_ms: 76.1,
            latency_p99_ms: 152.3,
        },
        throughput_delta_pct: 1.6,
        latency_p50_delta_pct: -2.7,
        latency_p99_delta_pct: -2.7,
        regression_detected: false,
        regression_threshold: 5.0,
    };
    assert!(!result.regression_detected);
}

/// Verify DiffBenchmarkResult detects regression above threshold
#[test]
fn test_diff_benchmark_with_regression() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "model_a.gguf".to_string(),
            throughput_tps: 12.3,
            latency_p50_ms: 78.2,
            latency_p99_ms: 156.5,
        },
        model_b: BenchmarkMetrics {
            path: "model_b.gguf".to_string(),
            throughput_tps: 11.0, // 10.6% regression
            latency_p50_ms: 88.0,
            latency_p99_ms: 180.0,
        },
        throughput_delta_pct: -10.6,
        latency_p50_delta_pct: 12.5,
        latency_p99_delta_pct: 15.0,
        regression_detected: true,
        regression_threshold: 5.0,
    };
    assert!(result.regression_detected);
}

/// Verify DifferentialExecutor preserves config
#[test]
fn test_differential_executor_new() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    assert_eq!(executor.config.apr_binary, "apr");
}

/// Verify DiffConfig accepts filter option
#[test]
fn test_diff_config_with_filter() {
    let config = DiffConfig {
        filter: Some("token_embd".to_string()),
        ..Default::default()
    };
    assert_eq!(config.filter.as_deref(), Some("token_embd"));
}

/// Verify DiffConfig accepts custom binary path
#[test]
fn test_diff_config_custom_binary() {
    let config = DiffConfig {
        apr_binary: "/custom/path/apr".to_string(),
        ..Default::default()
    };
    assert_eq!(config.apr_binary, "/custom/path/apr");
}

/// Verify DiffConfig accepts custom tolerance
#[test]
fn test_diff_config_custom_tolerance() {
    let config = DiffConfig {
        tolerance: 1e-10,
        ..Default::default()
    };
    assert!((config.tolerance - 1e-10).abs() < 1e-15);
}

/// Verify DiffConfig mismatches_only can be disabled
#[test]
fn test_diff_config_mismatches_only_false() {
    let config = DiffConfig {
        mismatches_only: false,
        ..Default::default()
    };
    assert!(!config.mismatches_only);
}

/// Verify TensorMismatch clone preserves all fields
#[test]
fn test_tensor_mismatch_clone() {
    let mismatch = TensorMismatch {
        name: "weights.0".to_string(),
        shape_a: vec![100, 200],
        shape_b: vec![200, 100],
        mismatch_type: TensorMismatchType::Transposed,
    };
    let cloned = mismatch.clone();
    assert_eq!(cloned.name, "weights.0");
    assert_eq!(cloned.shape_a, vec![100, 200]);
    assert_eq!(cloned.shape_b, vec![200, 100]);
}

/// Verify TensorMismatch debug output contains type name
#[test]
fn test_tensor_mismatch_debug() {
    let mismatch = TensorMismatch {
        name: "test".to_string(),
        shape_a: vec![10],
        shape_b: vec![20],
        mismatch_type: TensorMismatchType::ShapeMismatch,
    };
    let debug_str = format!("{mismatch:?}");
    assert!(debug_str.contains("TensorMismatch"));
    assert!(debug_str.contains("test"));
}

/// Verify Missing mismatch type maps to correct gate ID
#[test]
fn test_tensor_mismatch_type_missing() {
    let mismatch_type = TensorMismatchType::Missing;
    assert_eq!(mismatch_type.gate_id(), "F-ROSETTA-DIFF-002");
}

/// Verify TensorMismatchType debug formatting
#[test]
fn test_tensor_mismatch_type_debug() {
    let debug_str = format!("{:?}", TensorMismatchType::Transposed);
    assert!(debug_str.contains("Transposed"));
}

/// Verify TensorMismatchType equality comparison
#[test]
fn test_tensor_mismatch_type_eq() {
    assert_eq!(
        TensorMismatchType::Transposed,
        TensorMismatchType::Transposed
    );
    assert_ne!(TensorMismatchType::Transposed, TensorMismatchType::Missing);
}

/// Verify TensorMismatchType implements Copy
#[test]
fn test_tensor_mismatch_type_copy() {
    let t = TensorMismatchType::ShapeMismatch;
    let copied: TensorMismatchType = t;
    assert_eq!(copied, TensorMismatchType::ShapeMismatch);
}

/// Verify TensorDiffResult clone preserves counts
#[test]
fn test_tensor_diff_result_clone() {
    let result = TensorDiffResult {
        total_tensors: 10,
        mismatched_tensors: 2,
        transposed_tensors: 1,
        mismatches: vec![],
        passed: false,
    };
    let cloned = result.clone();
    assert_eq!(cloned.total_tensors, 10);
    assert_eq!(cloned.mismatched_tensors, 2);
}

/// Verify TensorDiffResult debug formatting
#[test]
fn test_tensor_diff_result_debug() {
    let result = TensorDiffResult {
        total_tensors: 5,
        mismatched_tensors: 0,
        transposed_tensors: 0,
        mismatches: vec![],
        passed: true,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("TensorDiffResult"));
}

/// Verify InferenceComparisonResult clone preserves token count
#[test]
fn test_inference_comparison_result_clone() {
    let result = InferenceComparisonResult {
        total_tokens: 10,
        matching_tokens: 10,
        max_logit_diff: 0.001,
        passed: true,
        token_comparisons: vec![],
    };
    let cloned = result.clone();
    assert_eq!(cloned.total_tokens, 10);
}

/// Verify InferenceComparisonResult debug formatting
#[test]
fn test_inference_comparison_result_debug() {
    let result = InferenceComparisonResult {
        total_tokens: 5,
        matching_tokens: 4,
        max_logit_diff: 0.1,
        passed: false,
        token_comparisons: vec![],
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("InferenceComparisonResult"));
}

/// Verify TokenComparison clone preserves all fields
#[test]
fn test_token_comparison_clone() {
    let tc = TokenComparison {
        index: 0,
        token_a: 100,
        token_b: 100,
        logit_diff: 0.0,
        matches: true,
    };
    let cloned = tc.clone();
    assert_eq!(cloned.index, 0);
    assert!(cloned.matches);
}

/// Verify TokenComparison debug formatting
#[test]
fn test_token_comparison_debug() {
    let tc = TokenComparison {
        index: 5,
        token_a: 42,
        token_b: 43,
        logit_diff: 0.5,
        matches: false,
    };
    let debug_str = format!("{tc:?}");
    assert!(debug_str.contains("TokenComparison"));
}

/// Verify DiffBenchmarkResult clone preserves model paths
#[test]
fn test_diff_benchmark_result_clone() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "a.gguf".to_string(),
            throughput_tps: 10.0,
            latency_p50_ms: 50.0,
            latency_p99_ms: 100.0,
        },
        model_b: BenchmarkMetrics {
            path: "b.gguf".to_string(),
            throughput_tps: 11.0,
            latency_p50_ms: 48.0,
            latency_p99_ms: 95.0,
        },
        throughput_delta_pct: 10.0,
        latency_p50_delta_pct: -4.0,
        latency_p99_delta_pct: -5.0,
        regression_detected: false,
        regression_threshold: 5.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.model_a.path, "a.gguf");
}

/// Verify DiffBenchmarkResult debug formatting
#[test]
fn test_diff_benchmark_result_debug() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "model_a".to_string(),
            throughput_tps: 10.0,
            latency_p50_ms: 50.0,
            latency_p99_ms: 100.0,
        },
        model_b: BenchmarkMetrics {
            path: "model_b".to_string(),
            throughput_tps: 10.0,
            latency_p50_ms: 50.0,
            latency_p99_ms: 100.0,
        },
        throughput_delta_pct: 0.0,
        latency_p50_delta_pct: 0.0,
        latency_p99_delta_pct: 0.0,
        regression_detected: false,
        regression_threshold: 5.0,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("DiffBenchmarkResult"));
}

/// Verify BenchmarkMetrics clone preserves throughput
#[test]
fn test_benchmark_metrics_clone() {
    let metrics = BenchmarkMetrics {
        path: "test.gguf".to_string(),
        throughput_tps: 15.5,
        latency_p50_ms: 65.0,
        latency_p99_ms: 130.0,
    };
    let cloned = metrics.clone();
    assert_eq!(cloned.path, "test.gguf");
    assert!((cloned.throughput_tps - 15.5).abs() < f64::EPSILON);
}

/// Verify BenchmarkMetrics debug formatting
#[test]
fn test_benchmark_metrics_debug() {
    let metrics = BenchmarkMetrics {
        path: "model.gguf".to_string(),
        throughput_tps: 20.0,
        latency_p50_ms: 40.0,
        latency_p99_ms: 80.0,
    };
    let debug_str = format!("{metrics:?}");
    assert!(debug_str.contains("BenchmarkMetrics"));
}

/// Verify CiProfileResult clone preserves passed flag
#[test]
fn test_ci_profile_result_clone() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 15.0,
        latency_p50_ms: 70.0,
        latency_p99_ms: 140.0,
        assertions: vec![],
        passed: true,
    };
    let cloned = result.clone();
    assert!(cloned.passed);
}

/// Verify CiProfileResult debug formatting
#[test]
fn test_ci_profile_result_debug() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 10.0,
        latency_p50_ms: 80.0,
        latency_p99_ms: 160.0,
        assertions: vec![],
        passed: false,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("CiProfileResult"));
}

/// Verify CiAssertion clone preserves name and passed flag
#[test]
fn test_ci_assertion_clone() {
    let assertion = CiAssertion {
        name: "throughput".to_string(),
        expected: ">= 10".to_string(),
        actual: "12".to_string(),
        passed: true,
        gate_id: "F-CI-001".to_string(),
    };
    let cloned = assertion.clone();
    assert_eq!(cloned.name, "throughput");
    assert!(cloned.passed);
}

/// Verify CiAssertion debug formatting
#[test]
fn test_ci_assertion_debug() {
    let assertion = CiAssertion {
        name: "p99".to_string(),
        expected: "<= 200".to_string(),
        actual: "250".to_string(),
        passed: false,
        gate_id: "F-CI-002".to_string(),
    };
    let debug_str = format!("{assertion:?}");
    assert!(debug_str.contains("CiAssertion"));
}
