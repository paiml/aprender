// ── CiProfileResult accessor tests with nested metrics ─────────────────────

/// Verify CiProfileResult::throughput() returns nested metrics value when present
#[test]
fn test_ci_profile_result_throughput_with_metrics() {
    let result = CiProfileResult {
        model: "test/model".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 42.5,
            latency_p50_ms: 10.0,
            latency_p99_ms: 25.0,
        }),
        throughput_tps: 0.0, // legacy field should be ignored
        latency_p50_ms: 0.0,
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.throughput() - 42.5).abs() < f64::EPSILON);
}

/// Verify CiProfileResult::throughput() falls back to legacy field when metrics absent
#[test]
fn test_ci_profile_result_throughput_legacy_fallback() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 33.3,
        latency_p50_ms: 0.0,
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.throughput() - 33.3).abs() < f64::EPSILON);
}

/// Verify CiProfileResult::p50_latency() returns nested metrics value when present
#[test]
fn test_ci_profile_result_p50_with_metrics() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 10.0,
            latency_p50_ms: 18.7,
            latency_p99_ms: 50.0,
        }),
        throughput_tps: 0.0,
        latency_p50_ms: 999.0, // legacy should be ignored
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p50_latency() - 18.7).abs() < f64::EPSILON);
}

/// Verify CiProfileResult::p50_latency() falls back to legacy field when metrics absent
#[test]
fn test_ci_profile_result_p50_legacy_fallback() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 0.0,
        latency_p50_ms: 22.5,
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p50_latency() - 22.5).abs() < f64::EPSILON);
}

/// Verify CiProfileResult::p99_latency() returns nested metrics value when present
#[test]
fn test_ci_profile_result_p99_with_metrics() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 10.0,
            latency_p50_ms: 15.0,
            latency_p99_ms: 88.8,
        }),
        throughput_tps: 0.0,
        latency_p50_ms: 0.0,
        latency_p99_ms: 999.0, // legacy should be ignored
        assertions: vec![],
        passed: true,
    };
    assert!((result.p99_latency() - 88.8).abs() < f64::EPSILON);
}

/// Verify CiProfileResult::p99_latency() falls back to legacy field when metrics absent
#[test]
fn test_ci_profile_result_p99_legacy_fallback() {
    let result = CiProfileResult {
        model: String::new(),
        metrics: None,
        throughput_tps: 0.0,
        latency_p50_ms: 0.0,
        latency_p99_ms: 45.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p99_latency() - 45.0).abs() < f64::EPSILON);
}

/// Verify CiProfileMetrics serialization roundtrip preserves all values
#[test]
fn test_ci_profile_metrics_serialization_roundtrip() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 55.5,
        latency_p50_ms: 12.3,
        latency_p99_ms: 45.6,
    };
    let json = serde_json::to_string(&metrics).unwrap();
    let parsed: CiProfileMetrics = serde_json::from_str(&json).unwrap();
    assert!((parsed.throughput_tok_s - 55.5).abs() < f64::EPSILON);
    assert!((parsed.latency_p50_ms - 12.3).abs() < f64::EPSILON);
    assert!((parsed.latency_p99_ms - 45.6).abs() < f64::EPSILON);
}

/// Verify CiProfileResult with nested metrics serializes to JSON containing metrics key
#[test]
fn test_ci_profile_result_with_metrics_serialization() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 20.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 30.0,
        }),
        throughput_tps: 0.0,
        latency_p50_ms: 0.0,
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: true,
    };
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("metrics"));
    assert!(json.contains("throughput_tok_s"));
}

/// Verify FormatConversionResult stores all fields including cached flag
#[test]
fn test_format_conversion_result_fields() {
    let result = FormatConversionResult {
        source_format: "safetensors".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 1500,
        error: None,
        cached: false,
    };
    assert!(result.success);
    assert_eq!(result.duration_ms, 1500);
    assert!(!result.cached);
}

/// Verify FormatConversionResult stores error message on failure
#[test]
fn test_format_conversion_result_failure() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: false,
        duration_ms: 500,
        error: Some("Layout mismatch".to_string()),
        cached: false,
    };
    assert!(!result.success);
    assert!(result.error.as_deref().unwrap().contains("Layout"));
}

/// Verify BenchResult stores throughput and backend fields correctly
#[test]
fn test_bench_result_fields() {
    let result = BenchResult {
        throughput_tps: 65.5,
        passed: true,
        backend: "gpu".to_string(),
        format: "gguf".to_string(),
    };
    assert!(result.passed);
    assert_eq!(result.backend, "gpu");
    assert_eq!(result.format, "gguf");
}
