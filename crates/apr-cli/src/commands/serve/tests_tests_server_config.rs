
#[test]
fn test_server_config_with_host_string_type() {
    let config = ServerConfig::default().with_host(String::from("10.0.0.1"));
    assert_eq!(config.host, "10.0.0.1");
}

#[test]
fn test_server_config_bind_addr_default() {
    let config = ServerConfig::default();
    assert_eq!(config.bind_addr(), "127.0.0.1:8080");
}

#[test]
fn test_server_config_bind_addr_custom() {
    let config = ServerConfig::default().with_port(443).with_host("0.0.0.0");
    assert_eq!(config.bind_addr(), "0.0.0.0:443");
}

#[test]
fn test_server_config_default_gpu_flags() {
    let config = ServerConfig::default();
    assert!(!config.gpu);
    assert!(!config.no_gpu);
    assert!(!config.batch);
}

#[test]
fn test_server_config_default_trace_fields() {
    let config = ServerConfig::default();
    assert!(!config.trace);
    assert_eq!(config.trace_level, "basic");
    assert!(!config.profile);
}

#[test]
fn test_server_config_clone() {
    let config = ServerConfig::default().with_port(9090);
    let cloned = config.clone();
    assert_eq!(cloned.port, 9090);
    assert_eq!(cloned.host, config.host);
    assert_eq!(cloned.metrics, config.metrics);
}

#[test]
fn test_server_config_debug_impl() {
    let config = ServerConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ServerConfig"));
    assert!(debug.contains("port"));
    assert!(debug.contains("8080"));
}

// ========================================================================
// J. ServerMetrics Comprehensive Tests
// ========================================================================

#[test]
fn test_server_metrics_default_zeros() {
    let metrics = ServerMetrics::new();
    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.requests_client_error.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.requests_server_error.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.model_memory_bytes.load(Ordering::Relaxed), 0);
}

#[test]
fn test_server_metrics_record_success_request() {
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 10, 50);

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_server_error.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 10);
    assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 50);
}

#[test]
fn test_server_metrics_record_failure_request() {
    let metrics = ServerMetrics::new();
    metrics.record_request(false, 0, 100);

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.requests_server_error.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 100);
}

#[test]
fn test_server_metrics_record_client_error_increments_both() {
    let metrics = ServerMetrics::new();
    metrics.record_client_error();

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_client_error.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 0);
}

#[test]
fn test_server_metrics_cumulative_tokens() {
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 50, 10);
    metrics.record_request(true, 75, 20);
    metrics.record_request(true, 25, 15);

    assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 150);
    assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 45);
}

#[test]
fn test_server_metrics_cumulative_inference_time() {
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 10, 100);
    metrics.record_request(true, 10, 200);
    metrics.record_request(true, 10, 300);

    assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 600);
}

#[test]
fn test_server_metrics_mixed_success_error_counting() {
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 10, 50); // success
    metrics.record_request(false, 0, 10); // server error
    metrics.record_client_error(); // client error

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 3);
    assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_server_error.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_client_error.load(Ordering::Relaxed), 1);
}

#[test]
fn test_server_metrics_uptime_starts_at_zero_or_more() {
    let metrics = ServerMetrics::new();
    // Uptime should be 0 or very small immediately after creation
    let uptime = metrics.uptime_seconds();
    assert!(
        uptime <= 1,
        "Uptime should be ~0 immediately after creation"
    );
}

#[test]
fn test_server_metrics_prometheus_output_format() {
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 42, 500);
    metrics.record_request(false, 0, 10);
    metrics.record_client_error();
    metrics
        .model_memory_bytes
        .store(1_000_000, Ordering::Relaxed);

    let output = metrics.prometheus_output();

    // Verify HELP/TYPE pairs
    assert!(output.contains("# HELP apr_requests_total"));
    assert!(output.contains("# TYPE apr_requests_total counter"));
    assert!(output.contains("# HELP apr_requests_success"));
    assert!(output.contains("# TYPE apr_requests_success counter"));
    assert!(output.contains("# HELP apr_requests_client_error"));
    assert!(output.contains("# TYPE apr_requests_client_error counter"));
    assert!(output.contains("# HELP apr_requests_server_error"));
    assert!(output.contains("# TYPE apr_requests_server_error counter"));
    assert!(output.contains("# HELP apr_tokens_generated_total"));
    assert!(output.contains("# TYPE apr_tokens_generated_total counter"));
    assert!(output.contains("# HELP apr_inference_duration_seconds_total"));
    assert!(output.contains("# TYPE apr_inference_duration_seconds_total counter"));
    assert!(output.contains("# HELP apr_memory_bytes"));
    assert!(output.contains("# TYPE apr_memory_bytes gauge"));
    assert!(output.contains("# HELP apr_uptime_seconds"));
    assert!(output.contains("# TYPE apr_uptime_seconds gauge"));

    // Verify actual metric values
    assert!(output.contains("apr_requests_total 3"));
    assert!(output.contains("apr_requests_success 1"));
    assert!(output.contains("apr_requests_client_error 1"));
    assert!(output.contains("apr_requests_server_error 1"));
    assert!(output.contains("apr_tokens_generated_total 42"));
}

#[test]
fn test_server_metrics_prometheus_inference_duration_seconds() {
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 10, 1500); // 1500ms = 1.5 seconds

    let output = metrics.prometheus_output();
    // 1500ms = 1.5 seconds, formatted as {:.3}
    assert!(output.contains("apr_inference_duration_seconds_total 1.500"));
}

// ========================================================================
// K. ErrorResponse Tests
// ========================================================================

#[test]
fn test_error_response_new_basic() {
    let err = ErrorResponse::new("not_found", "Resource not found");
    assert_eq!(err.error, "not_found");
    assert_eq!(err.message, "Resource not found");
    assert!(err.request_id.is_none());
}

#[test]
fn test_error_response_with_request_id_builder() {
    let err = ErrorResponse::new("timeout", "Request timed out").with_request_id("req-abc-123");
    assert_eq!(err.error, "timeout");
    assert_eq!(err.message, "Request timed out");
    assert_eq!(err.request_id.as_deref(), Some("req-abc-123"));
}

#[test]
fn test_error_response_serialization_without_request_id() {
    let err = ErrorResponse::new("error", "msg");
    let json = serde_json::to_string(&err).expect("should serialize");
    // request_id should be skipped (skip_serializing_if = "Option::is_none")
    assert!(!json.contains("request_id"));
}

#[test]
fn test_error_response_serialization_with_request_id() {
    let err = ErrorResponse::new("error", "msg").with_request_id("req-1");
    let json = serde_json::to_string(&err).expect("should serialize");
    assert!(json.contains("\"request_id\":\"req-1\""));
}

#[test]
fn test_error_response_roundtrip() {
    let original = ErrorResponse::new("test_error", "A message").with_request_id("id-42");
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.error, "test_error");
    assert_eq!(parsed.message, "A message");
    assert_eq!(parsed.request_id.as_deref(), Some("id-42"));
}

#[test]
fn test_error_response_with_string_types() {
    // Test that Into<String> works with both &str and String
    let err = ErrorResponse::new(String::from("err"), String::from("msg"));
    assert_eq!(err.error, "err");
    assert_eq!(err.message, "msg");
}

// ========================================================================
// L. HealthStatus Tests
// ========================================================================

#[test]
fn test_health_status_deserialization() {
    let healthy: HealthStatus = serde_json::from_str("\"healthy\"").unwrap();
    assert_eq!(healthy, HealthStatus::Healthy);

    let degraded: HealthStatus = serde_json::from_str("\"degraded\"").unwrap();
    assert_eq!(degraded, HealthStatus::Degraded);

    let unhealthy: HealthStatus = serde_json::from_str("\"unhealthy\"").unwrap();
    assert_eq!(unhealthy, HealthStatus::Unhealthy);
}

#[test]
fn test_health_status_equality() {
    assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
    assert_ne!(HealthStatus::Healthy, HealthStatus::Degraded);
    assert_ne!(HealthStatus::Degraded, HealthStatus::Unhealthy);
}

#[test]
fn test_health_status_copy_clone() {
    let status = HealthStatus::Healthy;
    let copied = status;
    let cloned = status.clone();
    assert_eq!(copied, HealthStatus::Healthy);
    assert_eq!(cloned, HealthStatus::Healthy);
}

#[test]
fn test_health_status_debug() {
    let debug = format!("{:?}", HealthStatus::Degraded);
    assert_eq!(debug, "Degraded");
}

// ========================================================================
// M. HealthResponse Roundtrip Tests
// ========================================================================

#[test]
fn test_health_response_full_roundtrip() {
    let original = HealthResponse {
        status: HealthStatus::Degraded,
        model_id: "qwen2-7b".to_string(),
        version: "0.25.1".to_string(),
        uptime_seconds: 3600,
        requests_total: 50000,
        gpu_available: true,
    };

    let json = serde_json::to_string(&original).unwrap();
    let parsed: HealthResponse = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.status, HealthStatus::Degraded);
    assert_eq!(parsed.model_id, "qwen2-7b");
    assert_eq!(parsed.version, "0.25.1");
    assert_eq!(parsed.uptime_seconds, 3600);
    assert_eq!(parsed.requests_total, 50000);
    assert!(parsed.gpu_available);
}

// ========================================================================
// N. health_check Logic Tests
// ========================================================================

#[test]
fn test_health_check_not_ready_is_unhealthy() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    // Not calling set_ready()
    let health = health_check(&state);
    assert_eq!(health.status, HealthStatus::Unhealthy);
}

#[test]
fn test_health_check_ready_zero_requests_is_healthy() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    state.set_ready();
    let health = health_check(&state);
    assert_eq!(health.status, HealthStatus::Healthy);
}

#[test]
fn test_health_check_ready_low_latency_is_healthy() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    state.set_ready();
    state.metrics.record_request(true, 10, 100); // 100ms avg

    let health = health_check(&state);
    assert_eq!(health.status, HealthStatus::Healthy);
}

#[test]
fn test_health_check_ready_high_latency_is_degraded() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    state.set_ready();
    state.metrics.record_request(true, 10, 5000); // 5000ms avg > 1000ms threshold

    let health = health_check(&state);
    assert_eq!(health.status, HealthStatus::Degraded);
}

#[test]
fn test_health_check_includes_version() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    state.set_ready();

    let health = health_check(&state);
    assert!(!health.version.is_empty());
    // Should be semver format
    let parts: Vec<&str> = health.version.split('.').collect();
    assert!(parts.len() >= 2);
}

#[test]
fn test_health_check_includes_requests_total() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    state.set_ready();
    state.metrics.record_request(true, 5, 10);
    state.metrics.record_request(true, 5, 10);
    state.metrics.record_request(true, 5, 10);

    let health = health_check(&state);
    assert_eq!(health.requests_total, 3);
}

// ========================================================================
// O. ServerState Tests
// ========================================================================

#[test]
fn test_server_state_mmap_threshold_exact() {
    // 50MB exactly should NOT use mmap (threshold is >50MB)
    let mut file = NamedTempFile::new().unwrap();
    let data = vec![0u8; 50 * 1024 * 1024];
    file.write_all(&data).unwrap();

    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    assert!(
        !state.uses_mmap,
        "Exactly 50MB should not use mmap (threshold is >50MB)"
    );
}

#[test]
fn test_server_state_ready_toggle() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();

    assert!(!state.is_ready());
    state.set_ready();
    assert!(state.is_ready());
    // Setting ready again should be fine
    state.set_ready();
    assert!(state.is_ready());
}

#[test]
fn test_server_state_model_id_no_extension() {
    let mut file = NamedTempFile::with_suffix(".gguf").unwrap();
    file.write_all(b"data").unwrap();

    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
    assert!(!state.model_id.contains(".gguf"));
    assert!(!state.model_id.is_empty());
}

#[test]
fn test_server_state_preserves_config() {
    let file = create_test_model();
    let config = ServerConfig::default().with_port(9999).with_host("0.0.0.0");
    let state = ServerState::new(file.path().to_path_buf(), config).unwrap();

    assert_eq!(state.config.port, 9999);
    assert_eq!(state.config.host, "0.0.0.0");
}

#[test]
fn test_server_state_metrics_initialized() {
    let file = create_test_model();
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();

    // Metrics should be freshly initialized with model size already stored
    assert!(state.metrics.model_memory_bytes.load(Ordering::Relaxed) > 0);
    assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 0);
}
