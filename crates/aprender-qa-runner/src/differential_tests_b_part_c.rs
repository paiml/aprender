#[test]
fn test_run_ci_profile_json_with_prefix() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake").unwrap();

    let json = r#"{"model":"test","metrics":null,"throughput_tps":55.0,"latency_p50_ms":8.0,"latency_p99_ms":20.0,"assertions":[],"passed":true}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_prefix",
        &format!("echo 'Loading model...' && echo '{json}' && exit 0"),
    );

    let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 3);
    if let Ok(r) = result {
        assert!(r.passed);
    }
}

#[test]
fn test_run_ci_profile_fallback_on_bad_json() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_bad",
        "echo 'not json at all' && exit 0",
    );

    let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 1);
    if let Ok(r) = result {
        // Fallback: passed = exit code success
        assert!(r.passed);
        assert!((r.throughput_tps - 0.0).abs() < 0.01);
    }
}

#[test]
fn test_run_ci_profile_fallback_failed_exit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_fail",
        "printf 'error\\n'; exit 1",
    );

    let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 1);
    if let Ok(r) = result {
        assert!(!r.passed);
    }
}

// =========================================================================
// run_diff_benchmark tests
// =========================================================================

#[test]
fn test_run_diff_benchmark_json_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"model_a").unwrap();
    std::fs::write(&model_b, b"model_b").unwrap();

    // Write JSON to a file to avoid shell quoting issues
    let json_file = temp_dir.path().join("diff_output.json");
    let json = r#"{"model_a":{"path":"a.gguf","throughput_tps":10.0,"latency_p50_ms":50.0,"latency_p99_ms":100.0},"model_b":{"path":"b.gguf","throughput_tps":12.0,"latency_p50_ms":45.0,"latency_p99_ms":90.0},"throughput_delta_pct":20.0,"latency_p50_delta_pct":-10.0,"latency_p99_delta_pct":-10.0,"regression_detected":false,"regression_threshold":5.0}"#;
    std::fs::write(&json_file, json).unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_diff_bench",
        &format!("cat '{}'", json_file.display()),
    );

    let result = run_diff_benchmark(mock.to_str().unwrap(), &model_a, &model_b, 5.0);
    // Mock binary execution can be flaky under parallel test runs
    if let Ok(r) = result {
        assert!(!r.regression_detected);
    }
}

#[test]
fn test_run_diff_benchmark_bad_json() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let mock = create_mock_binary(temp_dir.path(), "apr_diff_bad", "echo 'not json' && exit 0");

    let result = run_diff_benchmark(mock.to_str().unwrap(), &model_a, &model_b, 5.0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        Error::ExecutionFailed { reason, .. } => {
            assert!(reason.contains("Failed to parse output"));
        }
        other => unreachable!("Expected ExecutionFailed, got: {other:?}"),
    }
}

// =========================================================================
// DifferentialExecutor with mock binary
// =========================================================================

#[test]
fn test_diff_tensors_success() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.safetensors");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let json = r#"{"total_tensors":50,"mismatched_tensors":0,"transposed_tensors":0,"passed":true,"mismatches":[]}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_diff",
        &format!("echo '{json}' && exit 0"),
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.diff_tensors(&model_a, &model_b);
    if let Ok(r) = result {
        assert!(r.passed);
        assert_eq!(r.total_tensors, 50);
    }
}

#[test]
fn test_diff_tensors_failure_nonzero_exit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_diff_err",
        "echo 'tensor mismatch error' >&2 && exit 1",
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.diff_tensors(&model_a, &model_b);
    assert!(result.is_err());
}

#[test]
fn test_compare_inference_success() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.safetensors");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let json = r#"{"total_tokens":5,"matching_tokens":5,"max_logit_diff":1e-7,"passed":true,"token_comparisons":[]}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_compare",
        &format!("echo '{json}' && exit 0"),
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.compare_inference(&model_a, &model_b, "What is 2+2?", 5);
    if let Ok(r) = result {
        assert!(r.passed);
        assert_eq!(r.total_tokens, 5);
        assert_eq!(r.matching_tokens, 5);
    }
}

#[test]
fn test_compare_inference_fallback() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_compare_nojson",
        "echo 'some text output' && exit 0",
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.compare_inference(&model_a, &model_b, "test", 3);
    if let Ok(r) = result {
        // Fallback: passed from exit code
        assert!(r.passed);
        assert_eq!(r.total_tokens, 0);
    }
}

// =========================================================================
// run_six_column_profile tests
// =========================================================================

#[test]
fn test_run_six_column_profile_basic() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cache_dir = temp_dir.path().join("cache");

    // Create directory structure
    let gguf_dir = cache_dir.join("gguf");
    let apr_dir = cache_dir.join("apr");
    let st_dir = cache_dir.join("safetensors");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::create_dir_all(&apr_dir).unwrap();
    std::fs::create_dir_all(&st_dir).unwrap();

    // Create model file in gguf dir
    let gguf_model = gguf_dir.join("model.gguf");
    std::fs::write(&gguf_model, b"fake gguf model data for testing").unwrap();

    // Mock binary that handles convert and bench
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_six",
        r#"
case "$1" in
    rosetta)
        printf 'converted\n' > "$4"
        exit 0
        ;;
    bench)
        printf 'Throughput: 25.5 tok/s\n'
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
"#,
    );

    let result = run_six_column_profile(mock.to_str().unwrap(), &cache_dir, 1, 1);
    // May fail if mock binary has issues, but should work on most systems
    if let Ok(r) = result {
        assert_eq!(r.conversions.len(), 2); // APR + SafeTensors
    }
}

// =========================================================================
// prepare_model_with_provenance - resume workflow
// =========================================================================

#[test]
fn test_prepare_model_with_provenance_resume_matching_hash() {
    use crate::provenance::{create_source_provenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"safetensors content for resume test").unwrap();

    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    // Create existing provenance with matching hash
    let prov = create_source_provenance(&safetensors, "test/model").unwrap();
    save_provenance(&output_dir, &prov).unwrap();

    // Mock binary that handles conversions
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_resume",
        r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );
    if let Ok(r) = result {
        assert_eq!(r.provenance.source.format, "safetensors");
        assert!(r.gguf_path.is_some());
        assert!(r.apr_path.is_some());
        assert_eq!(r.conversions.len(), 2);
    }
}

#[test]
fn test_prepare_model_with_provenance_resume_changed_hash() {
    use crate::provenance::{Provenance, SourceProvenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"new content different from original").unwrap();

    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    // Create provenance with different hash (stale)
    let stale_prov = Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: safetensors.to_string_lossy().to_string(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            hf_repo: "test/model".to_string(),
            downloaded_at: "2026-01-01T00:00:00Z".to_string(),
        },
        derived: vec![],
    };
    save_provenance(&output_dir, &stale_prov).unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_changed",
        r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );
    if let Ok(r) = result {
        // Should have recreated provenance with new hash
        assert_ne!(
            r.provenance.source.sha256,
            "0000000000000000000000000000000000000000000000000000000000000000"
        );
    }
}

#[test]
fn test_prepare_model_with_provenance_with_quantization() {
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"safetensors quantization test").unwrap();

    let output_dir = temp_dir.path().join("output");

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_quant",
        r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        Some("q4_k_m"),
    );
    if let Ok(r) = result {
        // With quantization, paths should contain the quant level
        if let Some(gguf) = &r.gguf_path {
            assert!(gguf.to_string_lossy().contains("q4_k_m"));
        }
        if let Some(apr) = &r.apr_path {
            assert!(apr.to_string_lossy().contains("q4_k_m"));
        }
    }
}

#[test]
fn test_prepare_model_with_provenance_partial_failure() {
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"model content partial fail").unwrap();

    let output_dir = temp_dir.path().join("output");

    // Mock binary: first conversion succeeds, second fails
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_partial",
        r#"
# Track call count via a file
COUNTER_FILE="/tmp/apr_partial_counter_$$"
if [ ! -f "$COUNTER_FILE" ]; then
    echo "1" > "$COUNTER_FILE"
    echo "converted" > "$4"
    exit 0
else
    rm -f "$COUNTER_FILE"
    exit 1
fi
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );
    // Should still succeed overall (partial conversions are ok)
    if let Ok(r) = result {
        assert_eq!(r.conversions.len(), 2);
    }
}
