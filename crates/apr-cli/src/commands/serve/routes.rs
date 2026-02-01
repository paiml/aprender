//! HTTP route definitions and middleware for APR serve command

#![allow(unused_imports)]
#![allow(unused_variables)]

use super::health_check;
use super::types::{
    ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse, HealthStatus, ServerInfo,
    ServerState, StreamEvent, TranscribeResponse, MAX_REQUEST_SIZE,
};
use std::sync::Arc;
use std::time::Instant;

/// Create the inference server router
///
/// This function creates an axum Router that can be used for both production
/// and testing. All endpoints implement APR-SPEC §4.15.8.3 REST API spec.
#[cfg(feature = "inference")]
pub fn create_router(state: Arc<ServerState>) -> axum::Router {
    use axum::{
        body::Body,
        extract::{Request, State},
        http::{header, Method, StatusCode},
        middleware::{self, Next},
        response::{IntoResponse, Response, Sse},
        routing::{get, post},
        Json, Router,
    };
    use futures_util::stream;
    use std::convert::Infallible;

    // Middleware: Request size limit (SE02, EH05)
    async fn size_limit_middleware(request: Request, next: Next) -> Response {
        let content_length = request
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);

        if content_length > MAX_REQUEST_SIZE {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorResponse::new(
                    "payload_too_large",
                    format!("Request body exceeds {} bytes limit", MAX_REQUEST_SIZE),
                )),
            )
                .into_response();
        }

        next.run(request).await
    }

    // Handler: GET / (SL09: Root endpoint returns semver)
    async fn root_handler(State(state): State<Arc<ServerState>>) -> Json<ServerInfo> {
        Json(ServerInfo {
            name: "apr-serve".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_id: state.model_id.clone(),
        })
    }

    // Handler: GET /health (HR01-HR10)
    async fn health_handler(
        State(state): State<Arc<ServerState>>,
    ) -> (StatusCode, Json<HealthResponse>) {
        let health = health_check(&state);

        // GH-152: Verbose health check logging
        if state.config.verbose {
            eprintln!(
                "[VERBOSE] GET /health: status={:?}, uptime={}s",
                health.status, health.uptime_seconds
            );
        }

        // HR02: Return 503 during model load
        let status_code = match health.status {
            HealthStatus::Healthy => StatusCode::OK,
            HealthStatus::Degraded => StatusCode::OK,
            HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
        };

        (status_code, Json(health))
    }

    // Handler: GET /metrics (MA01-MA10)
    async fn metrics_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
        (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            state.metrics.prometheus_output(),
        )
    }

    // Handler: POST /predict (IC01-IC15)
    async fn predict_handler(
        State(state): State<Arc<ServerState>>,
        body: axum::body::Bytes,
    ) -> impl IntoResponse {
        let start = Instant::now();

        // EH05: Check body size
        if body.len() > MAX_REQUEST_SIZE {
            state.metrics.record_client_error();
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorResponse::new(
                    "payload_too_large",
                    "Request body too large",
                )),
            )
                .into_response();
        }

        // EH01: 400 for invalid JSON
        let request: serde_json::Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => {
                state.metrics.record_client_error();
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse::new(
                        "invalid_json",
                        format!("Invalid JSON: {e}"),
                    )),
                )
                    .into_response();
            }
        };

        // EH02: 400 for missing required fields
        if request.get("inputs").is_none() {
            state.metrics.record_client_error();
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "missing_field",
                    "Missing required field: inputs",
                )),
            )
                .into_response();
        }

        // Record successful request (placeholder inference)
        let duration_ms = start.elapsed().as_millis() as u64;
        state.metrics.record_request(true, 0, duration_ms);

        // Placeholder response (LG03: latency_ms included)
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "outputs": {},
                "latency_ms": duration_ms
            })),
        )
            .into_response()
    }

    // Handler: POST /generate with SSE streaming (SP01-SP10)
    async fn generate_handler(
        State(state): State<Arc<ServerState>>,
        body: axum::body::Bytes,
    ) -> Response {
        let start = Instant::now();

        // EH05: Check body size
        if body.len() > MAX_REQUEST_SIZE {
            state.metrics.record_client_error();
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorResponse::new(
                    "payload_too_large",
                    "Request body too large",
                )),
            )
                .into_response();
        }

        // EH01: 400 for invalid JSON
        let request: GenerateRequest = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => {
                state.metrics.record_client_error();
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse::new(
                        "invalid_json",
                        format!("Invalid JSON: {e}"),
                    )),
                )
                    .into_response();
            }
        };

        // GH-152: Verbose request logging
        if state.config.verbose {
            let prompt_preview = if request.prompt.len() > 100 {
                format!("{}...", &request.prompt[..100])
            } else {
                request.prompt.clone()
            };
            eprintln!(
                "[VERBOSE] POST /generate: prompt={:?}, max_tokens={}, stream={}",
                prompt_preview, request.max_tokens, request.stream
            );
        }

        // IC05: Empty prompt → 400 error
        if request.prompt.is_empty() {
            state.metrics.record_client_error();
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("empty_prompt", "Prompt cannot be empty")),
            )
                .into_response();
        }

        // Check if streaming requested
        if request.stream {
            // SP01-SP10: SSE streaming response
            let metrics = state.metrics.clone();
            let max_tokens = request.max_tokens;

            // Create SSE stream
            let stream = stream::iter((0..3).map(move |i| {
                let event = if i < 2 {
                    StreamEvent::token(&format!("token{}", i), i)
                } else {
                    StreamEvent::done("stop", 2)
                };
                Ok::<_, Infallible>(
                    axum::response::sse::Event::default()
                        .event(&event.event)
                        .data(&event.data),
                )
            }));

            // Record streaming request
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_request(true, 2, duration_ms);

            // GH-152: Verbose streaming response logging
            if state.config.verbose {
                eprintln!(
                    "[VERBOSE] POST /generate streaming: started, latency_ms={}",
                    duration_ms
                );
            }

            return Sse::new(stream)
                .keep_alive(axum::response::sse::KeepAlive::default())
                .into_response();
        }

        // Non-streaming response
        let duration_ms = start.elapsed().as_millis() as u64;
        state.metrics.record_request(true, 0, duration_ms);

        // GH-152: Verbose response logging
        if state.config.verbose {
            eprintln!(
                "[VERBOSE] POST /generate response: tokens=0, latency_ms={}, finish_reason=stop",
                duration_ms
            );
        }

        // LG03: latency_ms included
        (
            StatusCode::OK,
            Json(GenerateResponse {
                text: String::new(),
                tokens_generated: 0,
                finish_reason: "stop".to_string(),
                latency_ms: duration_ms,
            }),
        )
            .into_response()
    }

    // Handler: POST /transcribe (audio transcription)
    async fn transcribe_handler(
        State(state): State<Arc<ServerState>>,
        body: axum::body::Bytes,
    ) -> impl IntoResponse {
        let start = Instant::now();

        // EH05: Check body size
        if body.len() > MAX_REQUEST_SIZE {
            state.metrics.record_client_error();
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorResponse::new(
                    "payload_too_large",
                    "Request body too large",
                )),
            )
                .into_response();
        }

        // For now, return placeholder (audio processing requires additional setup)
        let duration_ms = start.elapsed().as_millis() as u64;
        state.metrics.record_request(true, 0, duration_ms);

        (
            StatusCode::OK,
            Json(TranscribeResponse {
                text: String::new(),
                language: "en".to_string(),
                duration_seconds: 0.0,
                latency_ms: duration_ms,
            }),
        )
            .into_response()
    }

    // Handler: Method not allowed (EH04)
    async fn method_not_allowed(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
        state.metrics.record_client_error();
        (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(ErrorResponse::new(
                "method_not_allowed",
                "Method not allowed for this endpoint",
            )),
        )
    }

    // Handler: 404 for unknown endpoints (EH03)
    async fn fallback_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
        state.metrics.record_client_error();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new("not_found", "Endpoint not found")),
        )
    }

    Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/predict", post(predict_handler))
        .route("/generate", post(generate_handler))
        .route("/transcribe", post(transcribe_handler))
        // EH04: Method not allowed for wrong HTTP methods
        .route("/predict", get(method_not_allowed))
        .route("/generate", get(method_not_allowed))
        .route("/transcribe", get(method_not_allowed))
        .layer(middleware::from_fn(size_limit_middleware))
        .fallback(fallback_handler)
        .with_state(state)
}
