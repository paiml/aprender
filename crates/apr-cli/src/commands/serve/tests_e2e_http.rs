// ============================================================================
// E2E HTTP Integration Tests — axum-test
// ============================================================================
//
// Tests the actual HTTP layer via create_router(). Every endpoint is exercised
// through real HTTP requests using axum-test's TestServer.
//
// Requires: inference feature (create_router is cfg-gated)

#[cfg(feature = "inference")]
mod e2e {
    use crate::commands::serve::routes::create_router;
    use crate::commands::serve::types::*;
    use axum_test::TestServer;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// Create a test server backed by a temp model file.
    fn test_server() -> (TestServer, NamedTempFile) {
        let mut file = NamedTempFile::with_suffix(".apr").unwrap();
        file.write_all(b"fake model data for testing").unwrap();

        let state =
            ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        state.set_ready();

        let router = create_router(Arc::new(state));
        let server = TestServer::new(router).unwrap();
        (server, file)
    }

    /// Create a test server in "not ready" state (model still loading).
    fn test_server_not_ready() -> (TestServer, NamedTempFile) {
        let mut file = NamedTempFile::with_suffix(".apr").unwrap();
        file.write_all(b"fake model data for testing").unwrap();

        let state =
            ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        // Do NOT call set_ready()

        let router = create_router(Arc::new(state));
        let server = TestServer::new(router).unwrap();
        (server, file)
    }

    // ====================================================================
    // GET / — Root endpoint (SL09)
    // ====================================================================

    #[tokio::test]
    async fn e2e_root_returns_server_info() {
        let (server, _f) = test_server();
        let resp = server.get("/").await;

        resp.assert_status_ok();
        let info: ServerInfo = resp.json();
        assert_eq!(info.name, "apr-serve");
        assert!(!info.version.is_empty());
        assert!(!info.model_id.is_empty());
    }

    #[tokio::test]
    async fn e2e_root_version_is_semver() {
        let (server, _f) = test_server();
        let info: ServerInfo = server.get("/").await.json();
        let parts: Vec<&str> = info.version.split('.').collect();
        assert!(parts.len() >= 2, "version should be semver: {}", info.version);
        assert!(parts[0].parse::<u32>().is_ok());
    }

    // ====================================================================
    // GET /health — Health check (HR01-HR10)
    // ====================================================================

    #[tokio::test]
    async fn e2e_health_returns_200_when_ready() {
        let (server, _f) = test_server();
        let resp = server.get("/health").await;

        resp.assert_status_ok();
        let health: HealthResponse = resp.json();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(!health.model_id.is_empty());
        assert!(!health.version.is_empty());
    }

    #[tokio::test]
    async fn e2e_health_returns_503_when_not_ready() {
        let (server, _f) = test_server_not_ready();
        let resp = server.get("/health").await;

        resp.assert_status(axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let health: HealthResponse = resp.json();
        assert_eq!(health.status, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn e2e_health_includes_uptime() {
        let (server, _f) = test_server();
        let health: HealthResponse = server.get("/health").await.json();
        // Uptime should be a valid number (0+ seconds)
        assert!(health.uptime_seconds < 60, "test uptime should be < 60s");
    }

    // ====================================================================
    // GET /metrics — Prometheus metrics (MA01-MA10)
    // ====================================================================

    #[tokio::test]
    async fn e2e_metrics_returns_prometheus_format() {
        let (server, _f) = test_server();
        let resp = server.get("/metrics").await;

        resp.assert_status_ok();
        let body = resp.text();
        assert!(body.contains("# HELP apr_requests_total"));
        assert!(body.contains("# TYPE apr_requests_total counter"));
        assert!(body.contains("apr_uptime_seconds"));
    }

    #[tokio::test]
    async fn e2e_metrics_content_type_is_text() {
        let (server, _f) = test_server();
        let resp = server.get("/metrics").await;

        let ct = resp
            .header("content-type")
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            ct.contains("text/plain"),
            "metrics content-type should be text/plain, got: {ct}"
        );
    }

    // ====================================================================
    // POST /predict — Prediction endpoint (IC01-IC15)
    // ====================================================================

    #[tokio::test]
    async fn e2e_predict_valid_request() {
        let (server, _f) = test_server();
        let resp = server
            .post("/predict")
            .json(&serde_json::json!({"inputs": {"text": "hello"}}))
            .await;

        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert!(body.get("outputs").is_some());
        assert!(body.get("latency_ms").is_some());
    }

    #[tokio::test]
    async fn e2e_predict_missing_inputs_returns_400() {
        let (server, _f) = test_server();
        let resp = server
            .post("/predict")
            .json(&serde_json::json!({"text": "no inputs field"}))
            .await;

        resp.assert_status(axum::http::StatusCode::BAD_REQUEST);
        let err: ErrorResponse = resp.json();
        assert_eq!(err.error, "missing_field");
    }

    #[tokio::test]
    async fn e2e_predict_invalid_json_returns_400() {
        let (server, _f) = test_server();
        let resp = server
            .post("/predict")
            .text("not json")
            .await;

        resp.assert_status(axum::http::StatusCode::BAD_REQUEST);
        let err: ErrorResponse = resp.json();
        assert_eq!(err.error, "invalid_json");
    }

    // ====================================================================
    // POST /generate — Text generation (SP01-SP10, LG03)
    // ====================================================================

    #[tokio::test]
    async fn e2e_generate_non_streaming() {
        let (server, _f) = test_server();
        let resp = server
            .post("/generate")
            .json(&serde_json::json!({
                "prompt": "Hello world",
                "max_tokens": 10,
                "stream": false
            }))
            .await;

        resp.assert_status_ok();
        let gen: GenerateResponse = resp.json();
        assert_eq!(gen.finish_reason, "stop");
    }

    #[tokio::test]
    async fn e2e_generate_streaming_returns_sse() {
        let (server, _f) = test_server();
        let resp = server
            .post("/generate")
            .json(&serde_json::json!({
                "prompt": "Hello world",
                "max_tokens": 10,
                "stream": true
            }))
            .await;

        resp.assert_status_ok();
        let ct = resp
            .header("content-type")
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            ct.contains("text/event-stream"),
            "streaming should return SSE content-type, got: {ct}"
        );
    }

    #[tokio::test]
    async fn e2e_generate_empty_prompt_returns_400() {
        let (server, _f) = test_server();
        let resp = server
            .post("/generate")
            .json(&serde_json::json!({
                "prompt": "",
                "max_tokens": 10
            }))
            .await;

        resp.assert_status(axum::http::StatusCode::BAD_REQUEST);
        let err: ErrorResponse = resp.json();
        assert_eq!(err.error, "empty_prompt");
    }

    #[tokio::test]
    async fn e2e_generate_invalid_json_returns_400() {
        let (server, _f) = test_server();
        let resp = server
            .post("/generate")
            .text("garbage")
            .await;

        resp.assert_status(axum::http::StatusCode::BAD_REQUEST);
    }

    // ====================================================================
    // POST /transcribe — Audio transcription
    // ====================================================================

    #[tokio::test]
    async fn e2e_transcribe_returns_response() {
        let (server, _f) = test_server();
        let resp = server
            .post("/transcribe")
            .text("fake audio bytes")
            .await;

        resp.assert_status_ok();
        let tr: TranscribeResponse = resp.json();
        assert_eq!(tr.language, "en");
    }

    // ====================================================================
    // Method not allowed (EH04)
    // ====================================================================

    #[tokio::test]
    async fn e2e_get_predict_returns_405() {
        let (server, _f) = test_server();
        let resp = server.get("/predict").await;

        resp.assert_status(axum::http::StatusCode::METHOD_NOT_ALLOWED);
        let err: ErrorResponse = resp.json();
        assert_eq!(err.error, "method_not_allowed");
    }

    #[tokio::test]
    async fn e2e_get_generate_returns_405() {
        let (server, _f) = test_server();
        let resp = server.get("/generate").await;

        resp.assert_status(axum::http::StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn e2e_get_transcribe_returns_405() {
        let (server, _f) = test_server();
        let resp = server.get("/transcribe").await;

        resp.assert_status(axum::http::StatusCode::METHOD_NOT_ALLOWED);
    }

    // ====================================================================
    // Fallback 404 (EH03)
    // ====================================================================

    #[tokio::test]
    async fn e2e_unknown_endpoint_returns_404() {
        let (server, _f) = test_server();
        let resp = server.get("/nonexistent").await;

        resp.assert_status(axum::http::StatusCode::NOT_FOUND);
        let err: ErrorResponse = resp.json();
        assert_eq!(err.error, "not_found");
    }

    #[tokio::test]
    async fn e2e_post_unknown_endpoint_returns_404() {
        let (server, _f) = test_server();
        let resp = server
            .post("/v99/imaginary")
            .json(&serde_json::json!({}))
            .await;

        resp.assert_status(axum::http::StatusCode::NOT_FOUND);
    }

    // ====================================================================
    // Metrics increment after requests (MA01)
    // ====================================================================

    #[tokio::test]
    async fn e2e_metrics_increment_after_predict() {
        let (server, _f) = test_server();

        // Make a successful predict request
        server
            .post("/predict")
            .json(&serde_json::json!({"inputs": {"text": "test"}}))
            .await
            .assert_status_ok();

        // Check metrics reflect the request
        let body = server.get("/metrics").await.text();
        assert!(
            body.contains("apr_requests_total 1"),
            "should have 1 total request, got:\n{body}"
        );
        assert!(
            body.contains("apr_requests_success 1"),
            "should have 1 success, got:\n{body}"
        );
    }

    #[tokio::test]
    async fn e2e_metrics_increment_client_errors() {
        let (server, _f) = test_server();

        // Make an invalid predict request (missing inputs)
        server
            .post("/predict")
            .json(&serde_json::json!({"bad": true}))
            .await;

        let body = server.get("/metrics").await.text();
        // 1 from the bad predict + 1 from the metrics GET itself (metrics doesn't increment)
        assert!(
            body.contains("apr_requests_client_error 1"),
            "should have 1 client error, got:\n{body}"
        );
    }

    // ====================================================================
    // Size limit middleware (SE02)
    // ====================================================================

    #[tokio::test]
    async fn e2e_oversized_content_length_rejected() {
        let (server, _f) = test_server();

        // Send with Content-Length exceeding 10MB
        let resp = server
            .post("/predict")
            .add_header(
                axum::http::header::CONTENT_LENGTH,
                "20000000",
            )
            .text("small body")
            .await;

        resp.assert_status(axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }
}
