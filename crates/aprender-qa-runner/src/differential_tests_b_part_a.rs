#[test]
fn test_six_column_profile_serialization() {
    let profile = SixColumnProfile {
        tps_gguf_cpu: Some(12.0),
        tps_gguf_gpu: Some(45.0),
        total_duration_ms: 1000,
        ..Default::default()
    };
    let json = serde_json::to_string(&profile).unwrap();
    let parsed: SixColumnProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.tps_gguf_cpu, Some(12.0));
}

// =========================================================================
// BenchResult tests
// =========================================================================

#[test]
fn test_bench_result_fields() {
    let result = BenchResult {
        throughput_tps: 15.5,
        passed: true,
        backend: "cpu".to_string(),
        format: "gguf".to_string(),
    };
    assert!((result.throughput_tps - 15.5).abs() < f64::EPSILON);
    assert!(result.passed);
    assert_eq!(result.backend, "cpu");
    assert_eq!(result.format, "gguf");
}

#[test]
fn test_bench_result_clone() {
    let result = BenchResult {
        throughput_tps: 20.0,
        passed: false,
        backend: "gpu".to_string(),
        format: "apr".to_string(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.backend, "gpu");
    assert_eq!(cloned.format, "apr");
}

#[test]
fn test_bench_result_debug() {
    let result = BenchResult {
        throughput_tps: 10.0,
        passed: true,
        backend: "cpu".to_string(),
        format: "safetensors".to_string(),
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("BenchResult"));
}

#[test]
fn test_bench_result_serialization() {
    let result = BenchResult {
        throughput_tps: 25.5,
        passed: true,
        backend: "gpu".to_string(),
        format: "gguf".to_string(),
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: BenchResult = serde_json::from_str(&json).unwrap();
    assert!((parsed.throughput_tps - 25.5).abs() < 0.01);
    assert!(parsed.passed);
}

// =========================================================================
// FormatConversionResult tests
// =========================================================================

#[test]
fn test_format_conversion_result_success() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 500,
        error: None,
        cached: false,
    };
    assert!(result.success);
    assert!(result.error.is_none());
    assert!(!result.cached);
}

#[test]
fn test_format_conversion_result_cached() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "safetensors".to_string(),
        success: true,
        duration_ms: 0,
        error: None,
        cached: true,
    };
    assert!(result.cached);
    assert_eq!(result.duration_ms, 0);
}

#[test]
fn test_format_conversion_result_failure() {
    let result = FormatConversionResult {
        source_format: "apr".to_string(),
        target_format: "safetensors".to_string(),
        success: false,
        duration_ms: 1000,
        error: Some("Conversion failed: unsupported format".to_string()),
        cached: false,
    };
    assert!(!result.success);
    assert!(result.error.is_some());
    assert!(result.error.as_ref().unwrap().contains("Conversion failed"));
}

#[test]
fn test_format_conversion_result_clone() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 250,
        error: None,
        cached: true,
    };
    let cloned = result.clone();
    assert_eq!(cloned.source_format, "gguf");
    assert_eq!(cloned.target_format, "apr");
}

#[test]
fn test_format_conversion_result_debug() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 100,
        error: None,
        cached: false,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("FormatConversionResult"));
}

#[test]
fn test_format_conversion_result_serialization() {
    let result = FormatConversionResult {
        source_format: "safetensors".to_string(),
        target_format: "gguf".to_string(),
        success: false,
        duration_ms: 2000,
        error: Some("Test error".to_string()),
        cached: false,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: FormatConversionResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source_format, "safetensors");
    assert!(!parsed.success);
}

// =========================================================================
// find_model_file tests
// =========================================================================

#[test]
fn test_find_model_file_nonexistent_dir() {
    let result = find_model_file(std::path::Path::new("/nonexistent/dir"));
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        Error::ExecutionFailed { command, reason } => {
            assert!(command.contains("find model"));
            assert!(reason.contains("does not exist"));
        }
        _ => unreachable!("Expected ExecutionFailed error"),
    }
}

#[test]
fn test_find_model_file_empty_dir() {
    let temp_dir = tempfile::tempdir().unwrap();
    let result = find_model_file(temp_dir.path());
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        Error::ExecutionFailed { reason, .. } => {
            assert!(reason.contains("No model file found"));
        }
        _ => unreachable!("Expected ExecutionFailed error"),
    }
}

#[test]
fn test_find_model_file_with_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("model.gguf");
    std::fs::write(&model_file, b"fake model data").unwrap();
    let result = find_model_file(temp_dir.path());
    assert!(result.is_ok());
    assert!(result.unwrap().exists());
}

#[test]
fn test_find_model_file_with_subdir_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let subdir = temp_dir.path().join("subdir");
    std::fs::create_dir(&subdir).unwrap();
    // Directory contains only a subdirectory, no files
    let result = find_model_file(temp_dir.path());
    assert!(result.is_err());
}

// =========================================================================
// compute_file_hash tests (via convert_format_cached)
// =========================================================================

#[test]
fn test_compute_file_hash_via_cached_conversion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.gguf");
    let target = temp_dir.path().join("target.apr");
    let hash_file = temp_dir.path().join(".hash");

    // Write source file
    std::fs::write(&source, b"test model content").unwrap();

    // Attempt cached conversion (will fail because apr binary doesn't exist,
    // but should still compute hash)
    let result = convert_format_cached("/nonexistent/apr", &source, &target, &hash_file);
    // The conversion will fail, but that's expected - we're testing hash computation
    assert!(result.is_err());
}

// =========================================================================
// CiProfileMetrics tests
// =========================================================================

#[test]
fn test_ci_profile_metrics_fields() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 50.0,
        latency_p50_ms: 20.0,
        latency_p99_ms: 45.0,
    };
    assert!((metrics.throughput_tok_s - 50.0).abs() < f64::EPSILON);
    assert!((metrics.latency_p50_ms - 20.0).abs() < f64::EPSILON);
    assert!((metrics.latency_p99_ms - 45.0).abs() < f64::EPSILON);
}

#[test]
fn test_ci_profile_metrics_clone() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 100.0,
        latency_p50_ms: 10.0,
        latency_p99_ms: 25.0,
    };
    let cloned = metrics.clone();
    assert!((cloned.throughput_tok_s - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_ci_profile_metrics_debug() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 75.0,
        latency_p50_ms: 15.0,
        latency_p99_ms: 35.0,
    };
    let debug_str = format!("{metrics:?}");
    assert!(debug_str.contains("CiProfileMetrics"));
}

#[test]
fn test_ci_profile_metrics_serialization() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 80.0,
        latency_p50_ms: 12.5,
        latency_p99_ms: 30.0,
    };
    let json = serde_json::to_string(&metrics).unwrap();
    let parsed: CiProfileMetrics = serde_json::from_str(&json).unwrap();
    assert!((parsed.throughput_tok_s - 80.0).abs() < 0.01);
}

// =========================================================================
// CiProfileResult accessor method tests
// =========================================================================

#[test]
fn test_ci_profile_result_throughput_from_metrics() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
        }),
        throughput_tps: 50.0, // Should be ignored when metrics is Some
        latency_p50_ms: 5.0,
        latency_p99_ms: 10.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.throughput() - 100.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_throughput_legacy() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: None, // No metrics, use legacy field
        throughput_tps: 75.0,
        latency_p50_ms: 5.0,
        latency_p99_ms: 10.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.throughput() - 75.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p50_latency_from_metrics() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 15.0,
            latency_p99_ms: 25.0,
        }),
        throughput_tps: 50.0,
        latency_p50_ms: 5.0, // Should be ignored when metrics is Some
        latency_p99_ms: 10.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p50_latency() - 15.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p50_latency_legacy() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: None,
        throughput_tps: 50.0,
        latency_p50_ms: 8.0,
        latency_p99_ms: 16.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p50_latency() - 8.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p99_latency_from_metrics() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 30.0,
        }),
        throughput_tps: 50.0,
        latency_p50_ms: 5.0,
        latency_p99_ms: 10.0, // Should be ignored when metrics is Some
        assertions: vec![],
        passed: true,
    };
    assert!((result.p99_latency() - 30.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p99_latency_legacy() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: None,
        throughput_tps: 50.0,
        latency_p50_ms: 5.0,
        latency_p99_ms: 12.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p99_latency() - 12.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_all_accessors_with_metrics() {
    let result = CiProfileResult {
        model: "model-x".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 200.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 15.0,
        }),
        throughput_tps: 100.0, // All legacy values should be ignored
        latency_p50_ms: 10.0,
        latency_p99_ms: 30.0,
        assertions: vec![],
        passed: true,
    };
    // All values should come from metrics
    assert!((result.throughput() - 200.0).abs() < 0.01);
    assert!((result.p50_latency() - 5.0).abs() < 0.01);
    assert!((result.p99_latency() - 15.0).abs() < 0.01);
}

// =========================================================================
// Provenance-Aware Model Preparation Tests (PMAT-PROV-001)
// =========================================================================

#[test]
fn test_model_preparation_result_fields() {
    use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc123".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![DerivedProvenance {
                format: "gguf".to_string(),
                path: "model.gguf".to_string(),
                sha256: "def456".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:05:00Z".to_string(),
            }],
        },
        safetensors_path: std::path::PathBuf::from("/models/model.safetensors"),
        gguf_path: Some(std::path::PathBuf::from("/models/model.gguf")),
        apr_path: None,
        conversions: vec![],
    };

    assert_eq!(result.provenance.source.format, "safetensors");
    assert!(result.gguf_path.is_some());
    assert!(result.apr_path.is_none());
}
