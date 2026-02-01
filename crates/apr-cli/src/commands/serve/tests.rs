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
