#[test]
fn test_diff_config_clone() {
    let config = DiffConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.apr_binary, config.apr_binary);
    assert_eq!(cloned.mismatches_only, config.mismatches_only);
}

#[test]
fn test_diff_config_debug() {
    let config = DiffConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("DiffConfig"));
}

#[test]
fn test_differential_executor_with_custom_config() {
    let config = DiffConfig {
        apr_binary: "custom-apr".to_string(),
        filter: Some("embed".to_string()),
        mismatches_only: false,
        tolerance: 1e-8,
    };
    let executor = DifferentialExecutor::new(config);
    assert_eq!(executor.config.apr_binary, "custom-apr");
    assert_eq!(executor.config.filter.as_deref(), Some("embed"));
    assert!(!executor.config.mismatches_only);
}

#[test]
fn test_parse_diff_output_empty_json() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    // Text output with no mismatches
    let output = "All tensors match";
    let result = executor.parse_diff_output(output).unwrap();
    assert!(result.passed);
    assert!(result.mismatches.is_empty());
}

#[test]
fn test_parse_diff_output_with_transposed() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let output = "token_embd.weight: [4096, 32000] vs [32000, 4096] ⚠️ TRANSPOSED\n";
    let result = executor.parse_diff_output(output).unwrap();
    assert!(!result.passed);
    assert_eq!(result.transposed_tensors, 1);
}

#[test]
fn test_parse_diff_output_valid_json() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let json = r#"{"total_tensors":100,"mismatched_tensors":0,"transposed_tensors":0,"mismatches":[],"passed":true}"#;
    let result = executor.parse_diff_output(json).unwrap();
    assert!(result.passed);
    assert_eq!(result.total_tensors, 100);
}

#[test]
fn test_parse_inference_output_success() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let result = executor
        .parse_inference_output("some output", true)
        .unwrap();
    assert!(result.passed);
}

#[test]
fn test_parse_inference_output_failure() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let result = executor.parse_inference_output("error", false).unwrap();
    assert!(!result.passed);
}

#[test]
fn test_parse_inference_output_valid_json() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let json = r#"{"total_tokens":10,"matching_tokens":10,"max_logit_diff":0.0,"passed":true,"token_comparisons":[]}"#;
    let result = executor.parse_inference_output(json, true).unwrap();
    assert!(result.passed);
    assert_eq!(result.total_tokens, 10);
}

#[test]
fn test_tensor_diff_result_with_mismatches() {
    let result = TensorDiffResult {
        total_tensors: 50,
        mismatched_tensors: 3,
        transposed_tensors: 2,
        mismatches: vec![
            TensorMismatch {
                name: "layer.0.weight".to_string(),
                shape_a: vec![768, 768],
                shape_b: vec![768, 768],
                mismatch_type: TensorMismatchType::ShapeMismatch,
            },
            TensorMismatch {
                name: "layer.1.weight".to_string(),
                shape_a: vec![768, 3072],
                shape_b: vec![3072, 768],
                mismatch_type: TensorMismatchType::Transposed,
            },
        ],
        passed: false,
    };
    assert!(!result.passed);
    assert_eq!(result.mismatches.len(), 2);
}

#[test]
fn test_inference_comparison_with_token_details() {
    let result = InferenceComparisonResult {
        total_tokens: 5,
        matching_tokens: 4,
        max_logit_diff: 0.05,
        passed: false,
        token_comparisons: vec![
            TokenComparison {
                index: 0,
                token_a: 1,
                token_b: 1,
                logit_diff: 0.0,
                matches: true,
            },
            TokenComparison {
                index: 1,
                token_a: 5,
                token_b: 6,
                logit_diff: 0.05,
                matches: false,
            },
        ],
    };
    assert!(!result.passed);
    assert_eq!(result.token_comparisons.len(), 2);
    assert!(!result.token_comparisons[1].matches);
}

#[test]
fn test_ci_profile_with_multiple_assertions() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 12.0,
        latency_p50_ms: 80.0,
        latency_p99_ms: 180.0,
        assertions: vec![
            CiAssertion {
                name: "throughput".to_string(),
                expected: ">= 10 tok/s".to_string(),
                actual: "12.0 tok/s".to_string(),
                passed: true,
                gate_id: "F-PROFILE-CI-001".to_string(),
            },
            CiAssertion {
                name: "p50".to_string(),
                expected: "<= 100 ms".to_string(),
                actual: "80.0 ms".to_string(),
                passed: true,
                gate_id: "F-PROFILE-CI-002".to_string(),
            },
            CiAssertion {
                name: "p99".to_string(),
                expected: "<= 150 ms".to_string(),
                actual: "180.0 ms".to_string(),
                passed: false,
                gate_id: "F-PROFILE-CI-003".to_string(),
            },
        ],
        passed: false,
    };
    assert!(!result.passed);
    assert_eq!(result.assertions.len(), 3);
    assert!(!result.assertions[2].passed);
}

// Additional tests for edge cases and coverage

#[test]
fn test_parse_diff_output_multiple_transposed() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let output = "tensor1: [100, 200] vs [200, 100] ⚠️ TRANSPOSED\n\
                      tensor2: [50, 100] vs [100, 50] TRANSPOSED\n\
                      tensor3: [32, 64] vs [64, 32] ⚠️";
    let result = executor.parse_diff_output(output).unwrap();
    assert!(!result.passed);
    assert_eq!(result.transposed_tensors, 3);
    assert_eq!(result.mismatched_tensors, 3);
}

#[test]
fn test_parse_diff_output_no_colon() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    // Line with TRANSPOSED but no colon should be skipped
    let output = "tensor TRANSPOSED without colon\n\
                      valid_tensor: [10, 20] TRANSPOSED";
    let result = executor.parse_diff_output(output).unwrap();
    assert_eq!(result.transposed_tensors, 1);
}

#[test]
fn test_tensor_mismatch_shape_fields() {
    let mismatch = TensorMismatch {
        name: "layer.0.attn.weight".to_string(),
        shape_a: vec![768, 768, 12],
        shape_b: vec![768, 12, 768],
        mismatch_type: TensorMismatchType::Transposed,
    };
    assert_eq!(mismatch.shape_a.len(), 3);
    assert_eq!(mismatch.shape_b.len(), 3);
}

#[test]
fn test_tensor_diff_result_serialization() {
    let result = TensorDiffResult {
        total_tensors: 50,
        mismatched_tensors: 2,
        transposed_tensors: 1,
        mismatches: vec![TensorMismatch {
            name: "test".to_string(),
            shape_a: vec![10, 20],
            shape_b: vec![20, 10],
            mismatch_type: TensorMismatchType::Transposed,
        }],
        passed: false,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: TensorDiffResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.total_tensors, 50);
    assert_eq!(parsed.transposed_tensors, 1);
}

#[test]
fn test_inference_comparison_result_serialization() {
    let result = InferenceComparisonResult {
        total_tokens: 20,
        matching_tokens: 18,
        max_logit_diff: 0.05,
        passed: false,
        token_comparisons: vec![TokenComparison {
            index: 5,
            token_a: 100,
            token_b: 101,
            logit_diff: 0.05,
            matches: false,
        }],
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: InferenceComparisonResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.total_tokens, 20);
    assert_eq!(parsed.token_comparisons.len(), 1);
}

#[test]
fn test_benchmark_metrics_serialization() {
    let metrics = BenchmarkMetrics {
        path: "/path/to/model.gguf".to_string(),
        throughput_tps: 15.5,
        latency_p50_ms: 65.0,
        latency_p99_ms: 130.0,
    };
    let json = serde_json::to_string(&metrics).unwrap();
    let parsed: BenchmarkMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.path, "/path/to/model.gguf");
}

#[test]
fn test_ci_assertion_serialization() {
    let assertion = CiAssertion {
        name: "throughput".to_string(),
        expected: ">= 10".to_string(),
        actual: "15".to_string(),
        passed: true,
        gate_id: "F-CI-001".to_string(),
    };
    let json = serde_json::to_string(&assertion).unwrap();
    let parsed: CiAssertion = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "throughput");
    assert!(parsed.passed);
}

#[test]
fn test_ci_profile_result_serialization() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 20.0,
        latency_p50_ms: 50.0,
        latency_p99_ms: 100.0,
        assertions: vec![],
        passed: true,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: CiProfileResult = serde_json::from_str(&json).unwrap();
    assert!(parsed.passed);
}

#[test]
fn test_diff_benchmark_result_serialization() {
    let result = DiffBenchmarkResult {
        model_a: BenchmarkMetrics {
            path: "a.gguf".to_string(),
            throughput_tps: 10.0,
            latency_p50_ms: 50.0,
            latency_p99_ms: 100.0,
        },
        model_b: BenchmarkMetrics {
            path: "b.gguf".to_string(),
            throughput_tps: 12.0,
            latency_p50_ms: 45.0,
            latency_p99_ms: 90.0,
        },
        throughput_delta_pct: 20.0,
        latency_p50_delta_pct: -10.0,
        latency_p99_delta_pct: -10.0,
        regression_detected: false,
        regression_threshold: 5.0,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: DiffBenchmarkResult = serde_json::from_str(&json).unwrap();
    assert!(!parsed.regression_detected);
}

#[test]
fn test_diff_config_all_fields() {
    let config = DiffConfig {
        apr_binary: "/custom/path/to/apr".to_string(),
        filter: Some("embedding".to_string()),
        mismatches_only: false,
        tolerance: 1e-8,
    };
    assert_eq!(config.apr_binary, "/custom/path/to/apr");
    assert_eq!(config.filter.as_deref(), Some("embedding"));
    assert!(!config.mismatches_only);
}

#[test]
fn test_token_comparison_matching() {
    let tc = TokenComparison {
        index: 10,
        token_a: 42,
        token_b: 42,
        logit_diff: 0.0,
        matches: true,
    };
    assert!(tc.matches);
    assert_eq!(tc.token_a, tc.token_b);
}

#[test]
fn test_tensor_mismatch_type_serialization() {
    let types = [
        TensorMismatchType::Transposed,
        TensorMismatchType::ShapeMismatch,
        TensorMismatchType::Missing,
    ];
    for t in types {
        let json = serde_json::to_string(&t).unwrap();
        let parsed: TensorMismatchType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }
}

#[test]
fn test_tensor_mismatch_serialization() {
    let mismatch = TensorMismatch {
        name: "layer.weight".to_string(),
        shape_a: vec![1, 2, 3],
        shape_b: vec![3, 2, 1],
        mismatch_type: TensorMismatchType::ShapeMismatch,
    };
    let json = serde_json::to_string(&mismatch).unwrap();
    let parsed: TensorMismatch = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "layer.weight");
    assert_eq!(parsed.mismatch_type, TensorMismatchType::ShapeMismatch);
}

#[test]
fn test_diff_config_default_tolerance() {
    let config = DiffConfig::default();
    assert!((config.tolerance - 1e-5).abs() < 1e-10);
}

#[test]
fn test_parse_inference_output_with_token_data() {
    let config = DiffConfig::default();
    let executor = DifferentialExecutor::new(config);
    let json = r#"{
            "total_tokens": 5,
            "matching_tokens": 4,
            "max_logit_diff": 0.02,
            "passed": false,
            "token_comparisons": [
                {"index": 0, "token_a": 1, "token_b": 1, "logit_diff": 0.0, "matches": true},
                {"index": 1, "token_a": 2, "token_b": 3, "logit_diff": 0.02, "matches": false}
            ]
        }"#;
    let result = executor.parse_inference_output(json, true).unwrap();
    assert_eq!(result.total_tokens, 5);
    assert_eq!(result.token_comparisons.len(), 2);
}

#[test]
fn test_tensor_diff_result_with_empty_mismatches() {
    let result = TensorDiffResult {
        total_tensors: 100,
        mismatched_tensors: 0,
        transposed_tensors: 0,
        mismatches: vec![],
        passed: true,
    };
    assert!(result.passed);
    assert!(result.mismatches.is_empty());
}

#[test]
fn test_inference_comparison_all_matching() {
    let result = InferenceComparisonResult {
        total_tokens: 10,
        matching_tokens: 10,
        max_logit_diff: 0.0,
        passed: true,
        token_comparisons: vec![],
    };
    assert!(result.passed);
    assert_eq!(result.total_tokens, result.matching_tokens);
}
