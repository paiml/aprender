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

include!("tests_tests_server_config.rs");
include!("tests_tests_generate_request.rs");
include!("validation.rs");
