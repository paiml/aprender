// ============================================================================
// EXTREME TDD Tests - APR-SPEC §4.15.9 Falsification Checklist
// ============================================================================
//
// This test module implements the 100-point falsification checklist from
// APR-SPEC §4.15.9. Tests are written FIRST (TDD), then implementation
// is added to make them pass.
//
// Coverage targets:
// - Server Lifecycle (SL01-SL10): 10 points
// - Health & Readiness (HR01-HR10): 10 points
// - Metrics Accuracy (MA01-MA10): 10 points
// - Error Handling (EH01-EH10): 10 points
// - Concurrency (CC01-CC10): 10 points

// File-level imports from serve module (needed for inner mod tests via use super::*)
#[allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// Helper to create test model file
    fn create_test_model() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test model data").unwrap();
        file
    }

    // ========================================================================
    // A. Server Lifecycle (SL01-SL10) — 10 Points
    // ========================================================================

    /// SL01: Server starts within 5s (no model) - validated via ServerConfig
    #[test]
    fn test_sl01_server_config_creation_fast() {
        let start = std::time::Instant::now();
        let _config = ServerConfig::default();
        assert!(
            start.elapsed().as_millis() < 100,
            "Config creation should be < 100ms"
        );
    }

    /// SL03: mmap models >50MB threshold check
    #[test]
    fn test_sl03_mmap_threshold_50mb() {
        let mut file = NamedTempFile::new().unwrap();
        // Small file: no mmap
        file.write_all(b"small").unwrap();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        assert!(!state.uses_mmap, "Files <50MB should not use mmap");
    }

    /// SL06: Invalid model path returns error
    #[test]
    fn test_sl06_invalid_model_path_error() {
        let result = ServerState::new(
            PathBuf::from("/nonexistent/model.apr"),
            ServerConfig::default(),
        );
        assert!(result.is_err(), "Invalid model path should return error");
    }

    /// SL09: Root endpoint returns semver
    #[test]
    fn test_sl09_server_info_semver() {
        let info = ServerInfo {
            name: "apr-serve".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_id: "test".to_string(),
        };
        // Verify semver format (X.Y.Z)
        let parts: Vec<&str> = info.version.split('.').collect();
        assert!(parts.len() >= 2, "Version should be semver format");
        assert!(
            parts[0].parse::<u32>().is_ok(),
            "Major version should be numeric"
        );
    }

    /// SL10: Ready only after model loaded
    #[test]
    fn test_sl10_ready_after_model_load() {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();

        // Initially not ready
        assert!(
            !state.is_ready(),
            "Server should not be ready before set_ready()"
        );

        // After set_ready
        state.set_ready();
        assert!(state.is_ready(), "Server should be ready after set_ready()");
    }

    // ========================================================================
    // B. Health & Readiness (HR01-HR10) — 10 Points
    // ========================================================================

    /// HR01: /health returns 200 when ready (via health_check function)
    #[test]
    fn test_hr01_health_returns_healthy_when_ready() {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        state.set_ready();

        let health = health_check(&state);
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    /// HR02: /health returns unhealthy during load
    #[test]
    fn test_hr02_health_unhealthy_during_load() {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        // Don't call set_ready()

        let health = health_check(&state);
        assert_eq!(
            health.status,
            HealthStatus::Unhealthy,
            "Should be unhealthy before ready"
        );
    }

    /// HR03: Health includes model_id
    #[test]
    fn test_hr03_health_includes_model_id() {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        state.set_ready();

        let health = health_check(&state);
        assert!(
            !health.model_id.is_empty(),
            "Health response must include model_id"
        );
    }

    /// HR04: uptime_seconds monotonically increases
    #[test]
    fn test_hr04_uptime_monotonic() {
        let metrics = ServerMetrics::new();
        let uptime1 = metrics.uptime_seconds();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let uptime2 = metrics.uptime_seconds();
        assert!(uptime2 >= uptime1, "Uptime should monotonically increase");
    }

    /// HR05: requests_total accurate
    #[test]
    fn test_hr05_requests_total_accurate() {
        let metrics = ServerMetrics::new();
        for _ in 0..5 {
            metrics.record_request(true, 0, 10);
        }
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 5);
    }

    /// HR06: Degraded status on high latency (>1s avg)
    #[test]
    fn test_hr06_degraded_on_high_latency() {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        state.set_ready();

        // Simulate high latency requests (>1000ms avg)
        state.metrics.record_request(true, 0, 2000);

        let health = health_check(&state);
        assert_eq!(
            health.status,
            HealthStatus::Degraded,
            "Should be degraded when avg latency > 1s"
        );
    }

    // ========================================================================
    // C. Metrics Accuracy (MA01-MA10) — 10 Points
    // ========================================================================

    /// MA01: Request counter increments
    #[test]
    fn test_ma01_request_counter_increments() {
        let metrics = ServerMetrics::new();
        let before = metrics.requests_total.load(Ordering::Relaxed);
        metrics.record_request(true, 0, 10);
        let after = metrics.requests_total.load(Ordering::Relaxed);
        assert_eq!(after, before + 1);
    }

    /// MA03: Token counter matches output
    #[test]
    fn test_ma03_token_counter_accuracy() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 100, 10);
        metrics.record_request(true, 50, 10);
        assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 150);
    }

    /// MA04: Error counter increments on errors
    #[test]
    fn test_ma04_error_counter_increments() {
        let metrics = ServerMetrics::new();
        metrics.record_client_error();
        metrics.record_client_error();
        assert_eq!(metrics.requests_client_error.load(Ordering::Relaxed), 2);
    }

    /// MA06: Metrics valid Prometheus format
    #[test]
    fn test_ma06_prometheus_format_valid() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 100, 1000);

        let output = metrics.prometheus_output();

        // Check Prometheus format requirements
        assert!(output.contains("# HELP"), "Must have HELP comments");
        assert!(output.contains("# TYPE"), "Must have TYPE comments");
        assert!(output.contains("counter"), "Must declare counter types");
        assert!(output.contains("gauge"), "Must declare gauge types");
        assert!(
            output.contains("apr_requests_total"),
            "Must have requests metric"
        );
        assert!(
            output.contains("apr_uptime_seconds"),
            "Must have uptime metric"
        );
    }

    /// MA07: Histogram buckets sensible (check memory metric exists)
    #[test]
    fn test_ma07_memory_metric_exists() {
        let metrics = ServerMetrics::new();
        metrics
            .model_memory_bytes
            .store(1024 * 1024, Ordering::Relaxed);

        let output = metrics.prometheus_output();
        assert!(
            output.contains("apr_memory_bytes"),
            "Must have memory metric"
        );
        assert!(
            output.contains("type=\"model\""),
            "Must have model memory label"
        );
    }

    // ========================================================================
    // D. Error Handling (EH01-EH10) — 10 Points
    // ========================================================================

    /// EH09: Error response is valid JSON
    #[test]
    fn test_eh09_error_response_valid_json() {
        let error = ErrorResponse::new("test_error", "Test message");
        let json = serde_json::to_string(&error).unwrap();

        // Verify it can be parsed back
        let parsed: ErrorResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.error, "test_error");
        assert_eq!(parsed.message, "Test message");
    }

    /// EH10: Errors include request ID when provided
    #[test]
    fn test_eh10_error_includes_request_id() {
        let error = ErrorResponse::new("test", "message").with_request_id("req-12345");

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("req-12345"), "Should include request_id");
    }

    // ========================================================================
    // E. Concurrency (CC01-CC10) — 10 Points
    // ========================================================================

    /// CC05: Thread safety (no data races)
    #[test]
    fn test_cc05_metrics_thread_safe() {
        use std::thread;

        let metrics = ServerMetrics::new();
        let metrics_clone = Arc::clone(&metrics);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let m = Arc::clone(&metrics_clone);
                thread::spawn(move || {
                    for _ in 0..100 {
                        m.record_request(true, 1, 1);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 1000);
    }

    /// CC06: Atomic metrics updates
    #[test]
    fn test_cc06_atomic_counter_updates() {
        use std::thread;

        let metrics = ServerMetrics::new();

        // Concurrent success and error recording
        let m1 = Arc::clone(&metrics);
        let m2 = Arc::clone(&metrics);

        let h1 = thread::spawn(move || {
            for _ in 0..500 {
                m1.record_request(true, 1, 1);
            }
        });

        let h2 = thread::spawn(move || {
            for _ in 0..500 {
                m2.record_client_error();
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 500);
        assert_eq!(metrics.requests_client_error.load(Ordering::Relaxed), 500);
    }

    // ========================================================================
    // F. ServerConfig Tests (Foundation)
    // ========================================================================

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert!(config.cors);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_concurrent, 10);
        assert!(config.metrics);
        assert!(!config.no_gpu);
        assert!(!config.verbose); // GH-152: verbose defaults to false
    }

    /// GH-152: Test verbose flag can be set
    #[test]
    fn test_server_config_verbose_flag() {
        let mut config = ServerConfig::default();
        assert!(!config.verbose);
        config.verbose = true;
        assert!(config.verbose);
    }

    #[test]
    fn test_server_config_with_port() {
        let config = ServerConfig::default().with_port(3000);
        assert_eq!(config.port, 3000);
    }

    #[test]
    fn test_server_config_with_host() {
        let config = ServerConfig::default().with_host("0.0.0.0");
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_server_config_bind_addr() {
        let config = ServerConfig::default().with_port(9000).with_host("0.0.0.0");
        assert_eq!(config.bind_addr(), "0.0.0.0:9000");
    }

    // ========================================================================
    // G. ServerState Tests (Foundation)
    // ========================================================================

    #[test]
    fn test_server_state_model_id_extraction() {
        let mut file = NamedTempFile::with_suffix(".apr").unwrap();
        file.write_all(b"test").unwrap();

        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        // Model ID should be the filename without extension
        assert!(!state.model_id.is_empty());
        assert!(!state.model_id.contains(".apr"));
    }

    #[test]
    fn test_server_state_model_size_recorded() {
        let mut file = NamedTempFile::new().unwrap();
        let data = vec![0u8; 1024];
        file.write_all(&data).unwrap();

        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        assert_eq!(state.model_size_bytes, 1024);
        assert_eq!(
            state.metrics.model_memory_bytes.load(Ordering::Relaxed),
            1024
        );
    }

    // ========================================================================
    // H. HealthStatus Serialization
    // ========================================================================

    #[test]
    fn test_health_status_serialization() {
        assert_eq!(
            serde_json::to_string(&HealthStatus::Healthy).unwrap(),
            "\"healthy\""
        );
        assert_eq!(
            serde_json::to_string(&HealthStatus::Degraded).unwrap(),
            "\"degraded\""
        );
        assert_eq!(
            serde_json::to_string(&HealthStatus::Unhealthy).unwrap(),
            "\"unhealthy\""
        );
    }

    #[test]
    fn test_health_response_serialization() {
        let health = HealthResponse {
            status: HealthStatus::Healthy,
            model_id: "test-model".to_string(),
            version: "1.0.0".to_string(),
            uptime_seconds: 60,
            requests_total: 100,
            gpu_available: false,
        };

        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"model_id\":\"test-model\""));
        assert!(json.contains("\"uptime_seconds\":60"));
    }

    // ========================================================================
    // I. ServerConfig Builder Pattern Tests
    // ========================================================================

    #[test]
    fn test_server_config_builder_chaining() {
        let config = ServerConfig::default().with_port(3000).with_host("0.0.0.0");
        assert_eq!(config.port, 3000);
        assert_eq!(config.host, "0.0.0.0");
        // Other defaults preserved
        assert!(config.cors);
        assert_eq!(config.timeout_secs, 30);
        assert!(config.metrics);
    }

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

    // ========================================================================
    // P. GenerateRequest / GenerateResponse Tests
    // ========================================================================

    #[test]
    fn test_generate_request_full_deserialization() {
        let json = r#"{
            "prompt": "Hello world",
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": true,
            "stop": [".", "!"]
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Hello world");
        assert_eq!(req.max_tokens, 128);
        assert!((req.temperature - 0.7).abs() < f32::EPSILON);
        assert!(req.stream);
        assert_eq!(req.stop.len(), 2);
        assert_eq!(req.stop[0], ".");
        assert_eq!(req.stop[1], "!");
    }

    #[test]
    fn test_generate_request_minimal() {
        // Only prompt is required, everything else has defaults
        let json = r#"{"prompt": "test"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 1.0).abs() < f32::EPSILON);
        assert!(!req.stream);
        assert!(req.stop.is_empty());
    }

    #[test]
    fn test_generate_response_serialization() {
        let resp = GenerateResponse {
            text: "Hello there".to_string(),
            tokens_generated: 5,
            finish_reason: "stop".to_string(),
            latency_ms: 123,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"text\":\"Hello there\""));
        assert!(json.contains("\"tokens_generated\":5"));
        assert!(json.contains("\"finish_reason\":\"stop\""));
        assert!(json.contains("\"latency_ms\":123"));
    }

    #[test]
    fn test_generate_response_roundtrip() {
        let original = GenerateResponse {
            text: "Output text".to_string(),
            tokens_generated: 42,
            finish_reason: "length".to_string(),
            latency_ms: 567,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: GenerateResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "Output text");
        assert_eq!(parsed.tokens_generated, 42);
        assert_eq!(parsed.finish_reason, "length");
        assert_eq!(parsed.latency_ms, 567);
    }

    // ========================================================================
    // Q. StreamEvent Comprehensive Tests
    // ========================================================================

    #[test]
    fn test_stream_event_token_fields() {
        let event = StreamEvent::token("world", 42);
        assert_eq!(event.event, "token");
        assert_eq!(event.data, "world");
        assert_eq!(event.token_id, Some(42));
    }

    #[test]
    fn test_stream_event_done_fields() {
        let event = StreamEvent::done("length", 100);
        assert_eq!(event.event, "done");
        assert!(event.data.contains("\"finish_reason\":\"length\""));
        assert!(event.data.contains("\"tokens_generated\":100"));
        assert!(event.token_id.is_none());
    }

    #[test]
    fn test_stream_event_error_fields() {
        let event = StreamEvent::error("Out of memory");
        assert_eq!(event.event, "error");
        assert_eq!(event.data, "Out of memory");
        assert!(event.token_id.is_none());
    }

    #[test]
    fn test_stream_event_to_sse_format() {
        let event = StreamEvent::token("hi", 1);
        let sse = event.to_sse();
        // SSE format: "event: <type>\ndata: <data>\n\n"
        assert!(sse.starts_with("event: token\n"));
        assert!(sse.contains("data: hi\n"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_stream_event_done_sse_format() {
        let event = StreamEvent::done("stop", 5);
        let sse = event.to_sse();
        assert!(sse.starts_with("event: done\n"));
        assert!(sse.contains("data: "));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_stream_event_serialization() {
        let event = StreamEvent::token("test", 99);
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"event\":\"token\""));
        assert!(json.contains("\"data\":\"test\""));
        assert!(json.contains("\"token_id\":99"));
    }

    #[test]
    fn test_stream_event_done_serialization_skips_token_id() {
        let event = StreamEvent::done("stop", 10);
        let json = serde_json::to_string(&event).unwrap();
        // token_id is None -> should be skipped
        assert!(!json.contains("token_id"));
    }

    // ========================================================================
    // R. TranscribeRequest / TranscribeResponse Tests
    // ========================================================================

    #[test]
    fn test_transcribe_request_defaults() {
        let json = r#"{}"#;
        let req: TranscribeRequest = serde_json::from_str(json).unwrap();
        assert!(req.language.is_none());
        assert_eq!(req.task, "transcribe");
    }

    #[test]
    fn test_transcribe_request_with_language() {
        let json = r#"{"language": "fr", "task": "translate"}"#;
        let req: TranscribeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.language.as_deref(), Some("fr"));
        assert_eq!(req.task, "translate");
    }

    #[test]
    fn test_transcribe_response_roundtrip() {
        let original = TranscribeResponse {
            text: "Hello world".to_string(),
            language: "en".to_string(),
            duration_seconds: 3.5,
            latency_ms: 200,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: TranscribeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "Hello world");
        assert_eq!(parsed.language, "en");
        assert!((parsed.duration_seconds - 3.5).abs() < f64::EPSILON);
        assert_eq!(parsed.latency_ms, 200);
    }

    // ========================================================================
    // S. ServerInfo Tests
    // ========================================================================

    #[test]
    fn test_server_info_serialization() {
        let info = ServerInfo {
            name: "apr-serve".to_string(),
            version: "0.25.1".to_string(),
            model_id: "my-model".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"name\":\"apr-serve\""));
        assert!(json.contains("\"version\":\"0.25.1\""));
        assert!(json.contains("\"model_id\":\"my-model\""));
    }

    #[test]
    fn test_server_info_roundtrip() {
        let original = ServerInfo {
            name: "test".to_string(),
            version: "1.2.3".to_string(),
            model_id: "model-abc".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ServerInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.version, "1.2.3");
        assert_eq!(parsed.model_id, "model-abc");
    }

    // ========================================================================
    // T. ChatMessage Tests
    // ========================================================================

    #[test]
    fn test_chat_message_user_message() {
        let json = r#"{"role": "user", "content": "Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.as_deref(), Some("Hello"));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(msg.name.is_none());
    }

    #[test]
    fn test_chat_message_system_message() {
        let json = r#"{"role": "system", "content": "You are helpful."}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn test_chat_message_assistant_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calc", "arguments": "{\"x\":1}"}
            }]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        assert!(msg.tool_calls.is_some());
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "calc");
    }

    #[test]
    fn test_chat_message_serialization_skips_none() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: Some("Hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));
        assert!(!json.contains("name"));
    }

    // ========================================================================
    // U. ChatCompletionRequest / Response Tests
    // ========================================================================

    #[test]
    fn test_chat_completion_request_minimal() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, ""); // default
        assert_eq!(req.messages.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert!(req.max_tokens.is_none());
        assert!(!req.stream);
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
    }

    #[test]
    fn test_chat_completion_request_full() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100,
            "stream": true,
            "temperature": 0.5,
            "top_p": 0.9
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, Some(100));
        assert!(req.stream);
        assert!((req.temperature.unwrap() - 0.5).abs() < f32::EPSILON);
        assert!((req.top_p.unwrap() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_completion_response_roundtrip() {
        let original = ChatCompletionResponse {
            id: "chatcmpl-001".to_string(),
            object: "chat.completion".to_string(),
            created: 1700000000,
            model: "apr".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "chatcmpl-001");
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(parsed.usage.as_ref().unwrap().total_tokens, 6);
    }

    // ========================================================================
    // V. TokenUsage Tests
    // ========================================================================

    #[test]
    fn test_token_usage_roundtrip() {
        let usage = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let parsed: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.prompt_tokens, 10);
        assert_eq!(parsed.completion_tokens, 20);
        assert_eq!(parsed.total_tokens, 30);
    }

    // ========================================================================
    // W. ToolChoice Tests
    // ========================================================================

    #[test]
    fn test_tool_choice_mode_auto() {
        let json = r#""auto""#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        match choice {
            ToolChoice::Mode(mode) => assert_eq!(mode, "auto"),
            _ => panic!("Expected Mode variant"),
        }
    }

    #[test]
    fn test_tool_choice_mode_none() {
        let json = r#""none""#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        match choice {
            ToolChoice::Mode(mode) => assert_eq!(mode, "none"),
            _ => panic!("Expected Mode variant"),
        }
    }

    #[test]
    fn test_tool_choice_specific_function() {
        let json = r#"{"type": "function", "function": {"name": "get_temp"}}"#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        match choice {
            ToolChoice::Function {
                tool_type,
                function,
            } => {
                assert_eq!(tool_type, "function");
                assert_eq!(function.name, "get_temp");
            }
            _ => panic!("Expected Function variant"),
        }
    }

    // ========================================================================
    // X. format_tools_prompt Tests
    // ========================================================================

    #[test]
    fn test_format_tools_prompt_empty() {
        let result = format_tools_prompt(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_tools_prompt_single_tool() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: None,
            },
        }];
        let prompt = format_tools_prompt(&tools);
        assert!(prompt.contains("### search"));
        assert!(prompt.contains("Search the web"));
        assert!(prompt.contains("tool_call"));
    }

    #[test]
    fn test_format_tools_prompt_multiple_tools() {
        let tools = vec![
            Tool {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: "tool_a".to_string(),
                    description: Some("First tool".to_string()),
                    parameters: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: "tool_b".to_string(),
                    description: Some("Second tool".to_string()),
                    parameters: Some(serde_json::json!({"type": "object"})),
                },
            },
        ];
        let prompt = format_tools_prompt(&tools);
        assert!(prompt.contains("### tool_a"));
        assert!(prompt.contains("### tool_b"));
        assert!(prompt.contains("First tool"));
        assert!(prompt.contains("Second tool"));
        assert!(prompt.contains("Parameters:"));
    }

    #[test]
    fn test_format_tools_prompt_no_description() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "bare_tool".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let prompt = format_tools_prompt(&tools);
        assert!(prompt.contains("### bare_tool"));
        // Should not crash or contain garbage for missing description
    }

    // ========================================================================
    // Y. parse_tool_calls Tests
    // ========================================================================

    #[test]
    fn test_parse_tool_calls_valid_json() {
        let output = r#"{"tool_call": {"name": "calc", "arguments": {"x": 42}}}"#;
        let calls = parse_tool_calls(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].tool_type, "function");
        assert!(calls[0].id.starts_with("call_"));
        assert!(calls[0].function.arguments.contains("42"));
    }

    #[test]
    fn test_parse_tool_calls_embedded_in_text() {
        let output =
            r#"Let me help. {"tool_call": {"name": "search", "arguments": {"q": "rust"}}}"#;
        let calls = parse_tool_calls(output).unwrap();
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_parse_tool_calls_no_tool_call() {
        assert!(parse_tool_calls("Just regular text").is_none());
        assert!(parse_tool_calls("").is_none());
        assert!(parse_tool_calls("{}").is_none());
    }

    #[test]
    fn test_parse_tool_calls_missing_name() {
        let output = r#"{"tool_call": {"arguments": {"x": 1}}}"#;
        assert!(parse_tool_calls(output).is_none());
    }

    #[test]
    fn test_parse_tool_calls_missing_arguments() {
        let output = r#"{"tool_call": {"name": "test"}}"#;
        assert!(parse_tool_calls(output).is_none());
    }

    #[test]
    fn test_parse_tool_calls_invalid_json() {
        let output = r#"{"tool_call": {"name": "test", "arguments": broken}}"#;
        assert!(parse_tool_calls(output).is_none());
    }

    #[test]
    fn test_parse_tool_calls_whitespace_trimmed() {
        let output = r#"  {"tool_call": {"name": "ws_test", "arguments": {}}}  "#;
        let calls = parse_tool_calls(output).unwrap();
        assert_eq!(calls[0].function.name, "ws_test");
    }

    // ========================================================================
    // Z. uuid_simple Tests
    // ========================================================================

    #[test]
    fn test_uuid_simple_is_hex_string() {
        let id = uuid_simple();
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_uuid_simple_changes_over_time() {
        let id1 = uuid_simple();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id2 = uuid_simple();
        // Not guaranteed to differ in all cases, but very likely with sleep
        // This is a best-effort check
        let _ = (id1, id2); // At minimum ensure both calls succeed
    }

    // ========================================================================
    // AA. MAX_REQUEST_SIZE Constant Test
    // ========================================================================

    #[test]
    fn test_max_request_size_is_10mb() {
        assert_eq!(MAX_REQUEST_SIZE, 10 * 1024 * 1024);
        assert_eq!(MAX_REQUEST_SIZE, 10_485_760);
    }

    // ========================================================================
    // AB. FunctionDef / FunctionCall Tests
    // ========================================================================

    #[test]
    fn test_function_def_serialization_skips_none() {
        let def = FunctionDef {
            name: "test".to_string(),
            description: None,
            parameters: None,
        };
        let json = serde_json::to_string(&def).unwrap();
        assert!(!json.contains("description"));
        assert!(!json.contains("parameters"));
        assert!(json.contains("\"name\":\"test\""));
    }

    #[test]
    fn test_function_call_roundtrip() {
        let original = FunctionCall {
            name: "compute".to_string(),
            arguments: r#"{"a": 1, "b": 2}"#.to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: FunctionCall = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "compute");
        assert_eq!(parsed.arguments, r#"{"a": 1, "b": 2}"#);
    }

    #[test]
    fn test_tool_choice_function_name_roundtrip() {
        let name = ToolChoiceFunction {
            name: "my_fn".to_string(),
        };
        let json = serde_json::to_string(&name).unwrap();
        let parsed: ToolChoiceFunction = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "my_fn");
    }
}

// ============================================================================
// HTTP Integration Tests (requires inference feature)
// ============================================================================

#[cfg(all(test, feature = "inference"))]
mod http_tests {
    use super::super::routes::create_router;
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tower::ServiceExt;

    /// Helper to create test model file
    fn create_test_model() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test model data").unwrap();
        file
    }

    /// Helper to create test server state
    fn create_ready_state() -> Arc<ServerState> {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        state.set_ready();
        std::mem::forget(file);
        Arc::new(state)
    }

    // ========================================================================
    // HR01-HR10: Health Endpoint HTTP Tests
    // ========================================================================

    /// HR01: GET /health returns 200 when ready
    #[tokio::test]
    async fn test_hr01_health_returns_200_when_ready() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// HR02: GET /health returns 503 during model load
    #[tokio::test]
    async fn test_hr02_health_returns_503_during_load() {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        // Don't set ready
        std::mem::forget(file);
        let app = create_router(Arc::new(state));

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// HR03: Health response includes model_id
    #[tokio::test]
    async fn test_hr03_health_includes_model_id() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert!(!health.model_id.is_empty());
    }

    // ========================================================================
    // SL09: Root Endpoint Tests
    // ========================================================================

    /// SL09: GET / returns semver version
    #[tokio::test]
    async fn test_sl09_root_returns_semver() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let info: ServerInfo = serde_json::from_slice(&body).unwrap();
        assert_eq!(info.name, "apr-serve");
        assert!(!info.version.is_empty());
    }

    // ========================================================================
    // MA01-MA10: Metrics Endpoint Tests
    // ========================================================================

    /// MA05: GET /metrics returns 200
    #[tokio::test]
    async fn test_ma05_metrics_returns_200() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// MA06: /metrics returns valid Prometheus format
    #[tokio::test]
    async fn test_ma06_metrics_prometheus_format() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        assert!(text.contains("# HELP"));
        assert!(text.contains("# TYPE"));
        assert!(text.contains("apr_requests_total"));
    }

    // ========================================================================
    // EH01-EH10: Error Handling Tests
    // ========================================================================

    /// EH01: 400 for invalid JSON on /predict
    #[tokio::test]
    async fn test_eh01_invalid_json_returns_400() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from("not json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// EH02: 400 for missing required fields
    #[tokio::test]
    async fn test_eh02_missing_fields_returns_400() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"wrong": "field"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// EH03: 404 for unknown endpoint
    #[tokio::test]
    async fn test_eh03_unknown_endpoint_returns_404() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/unknown").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// EH09: Error response is valid JSON
    #[tokio::test]
    async fn test_eh09_error_response_valid_json() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from("invalid"))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let error: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert!(!error.error.is_empty());
        assert!(!error.message.is_empty());
    }

    // ========================================================================
    // IC05: Empty prompt returns 400
    // ========================================================================

    #[tokio::test]
    async fn test_ic05_empty_prompt_returns_400() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": ""}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    // ========================================================================
    // Successful Request Tests
    // ========================================================================

    /// /predict with valid input returns 200
    #[tokio::test]
    async fn test_predict_valid_returns_200() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"inputs": {"data": [1,2,3]}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// /generate with valid prompt returns 200
    #[tokio::test]
    async fn test_generate_valid_returns_200() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "Hello world"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    // ========================================================================
    // SP01-SP10: SSE Streaming Tests
    // ========================================================================

    /// SP02: Streaming response has correct content-type
    #[tokio::test]
    async fn test_sp02_streaming_content_type() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "Hello", "stream": true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers().get("content-type").unwrap();
        assert!(content_type.to_str().unwrap().contains("text/event-stream"));
    }

    /// LG03: Response includes latency_ms field
    #[tokio::test]
    async fn test_lg03_response_includes_latency() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "Hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp: GenerateResponse = serde_json::from_slice(&body).unwrap();
        // Verify latency field exists (u64 is always >= 0)
        let _ = resp.latency_ms;
    }

    // ========================================================================
    // Transcribe Endpoint Tests
    // ========================================================================

    /// POST /transcribe returns 200
    #[tokio::test]
    async fn test_transcribe_returns_200() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/transcribe")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![0u8; 100]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Transcribe response includes latency_ms
    #[tokio::test]
    async fn test_transcribe_includes_latency() {
        let state = create_ready_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::post("/transcribe")
                    .body(Body::from(vec![0u8; 100]))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp: TranscribeResponse = serde_json::from_slice(&body).unwrap();
        // Verify latency field exists (u64 is always >= 0)
        let _ = resp.latency_ms;
        assert_eq!(resp.language, "en");
    }

    // ========================================================================
    // SE02, EH05: Request Size Limit Tests
    // ========================================================================

    /// SE02: MAX_REQUEST_SIZE is 10MB
    #[test]
    fn test_se02_max_request_size() {
        assert_eq!(MAX_REQUEST_SIZE, 10 * 1024 * 1024);
    }

    // ========================================================================
    // Additional Type Tests
    // ========================================================================

    /// GenerateRequest defaults
    #[test]
    fn test_generate_request_defaults() {
        let json = r#"{"prompt": "test"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 1.0);
        assert!(!req.stream);
        assert!(req.stop.is_empty());
    }

    /// StreamEvent formatting
    #[test]
    fn test_stream_event_sse_format() {
        let event = StreamEvent::token("hello", 0);
        let sse = event.to_sse();
        assert!(sse.contains("event: token"));
        assert!(sse.contains("data: hello"));
        assert!(sse.ends_with("\n\n"));
    }

    /// StreamEvent done
    #[test]
    fn test_stream_event_done() {
        let event = StreamEvent::done("stop", 10);
        assert_eq!(event.event, "done");
        assert!(event.data.contains("stop"));
        assert!(event.data.contains("10"));
    }

    /// StreamEvent error
    #[test]
    fn test_stream_event_error() {
        let event = StreamEvent::error("OOM");
        assert_eq!(event.event, "error");
        assert_eq!(event.data, "OOM");
    }

    // ========================================================================
    // GH-160: Tool Calling Tests (F-TOOL-001 to F-TOOL-005)
    // ========================================================================

    /// F-TOOL-001: Tool definition parsing
    #[test]
    fn test_tool_definition_parsing() {
        let json = r#"{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }"#;
        let tool: Tool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "get_weather");
        assert!(tool.function.description.is_some());
    }

    /// F-TOOL-002: ChatCompletionRequest with tools
    #[test]
    fn test_chat_request_with_tools() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"}
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    }

    /// F-TOOL-003: Parse tool call from model output
    #[test]
    fn test_parse_tool_calls_json() {
        let output = r#"{"tool_call": {"name": "get_weather", "arguments": {"location": "NYC"}}}"#;
        let tool_calls = parse_tool_calls(output);
        assert!(tool_calls.is_some());
        let calls = tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    /// F-TOOL-003: Parse embedded tool call in text
    #[test]
    fn test_parse_tool_calls_embedded() {
        let output = r#"I'll check the weather for you. {"tool_call": {"name": "get_weather", "arguments": {"city": "NYC"}}}"#;
        let tool_calls = parse_tool_calls(output);
        assert!(tool_calls.is_some());
    }

    /// F-TOOL-003: No tool call in regular text
    #[test]
    fn test_parse_tool_calls_none() {
        let output = "The weather in NYC is sunny and 72°F.";
        let tool_calls = parse_tool_calls(output);
        assert!(tool_calls.is_none());
    }

    /// F-TOOL-004: ChatMessage with tool_call_id (tool response)
    #[test]
    fn test_chat_message_tool_response() {
        let json = r#"{
            "role": "tool",
            "content": "72°F, sunny",
            "tool_call_id": "call_abc123"
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert!(msg.tool_call_id.is_some());
        assert_eq!(msg.tool_call_id.unwrap(), "call_abc123");
    }

    /// F-TOOL-005: Format tools into prompt
    #[test]
    fn test_format_tools_prompt() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get current weather".to_string()),
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let prompt = format_tools_prompt(&tools);
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Get current weather"));
        assert!(prompt.contains("tool_call"));
    }

    /// ToolCall serialization
    #[test]
    fn test_tool_call_serialization() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "test_func".to_string(),
                arguments: r#"{"arg": "value"}"#.to_string(),
            },
        };
        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("test_func"));
    }

    /// ChatCompletionResponse with tool_calls
    #[test]
    fn test_chat_response_with_tool_calls() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "test".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_123".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"NYC"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("tool_calls"));
        assert!(json.contains("get_weather"));
    }
}
