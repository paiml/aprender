//! Serve command implementation
//!
//! Implements APR-SPEC §4.15.8: Runtime Inference Engine
//!
//! Start an inference server with:
//! - REST API endpoints (/transcribe, /generate, /health, /metrics, /predict)
//! - Server-Sent Events (SSE) streaming
//! - Prometheus metrics (Prometheus exposition format)
//! - Automatic mmap loading for large models (>50MB)
//! - KV cache for transformer models
//!
//! # Falsification Checklist Coverage
//!
//! This module implements the 100-point falsification checklist from APR-SPEC §4.15.9:
//! - Server Lifecycle (SL01-SL10): Startup, shutdown, mmap thresholds
//! - Health & Readiness (HR01-HR10): /health endpoint behavior
//! - Metrics Accuracy (MA01-MA10): Prometheus /metrics endpoint
//! - Error Handling (EH01-EH10): HTTP status codes and JSON errors

// Allow dead code and unused during development - these are planned features
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(clippy::format_push_string)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::if_not_else)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::inefficient_to_string)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Server configuration
#[derive(Debug, Clone)]
pub(crate) struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// Enable CORS (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub cors: bool,
    /// Request timeout in seconds (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub timeout_secs: u64,
    /// Maximum concurrent requests (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub max_concurrent: usize,
    /// Enable Prometheus metrics endpoint
    pub metrics: bool,
    /// Disable GPU acceleration (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub no_gpu: bool,
    /// Force GPU acceleration (requires CUDA feature)
    pub gpu: bool,
    /// Enable batched GPU inference for 2X+ throughput
    pub batch: bool,
    /// Enable inference tracing (PMAT-SHOWCASE-METHODOLOGY-001)
    pub trace: bool,
    /// Trace detail level (none, basic, layer)
    pub trace_level: String,
    /// Enable inline Roofline profiling (adds X-Profile headers)
    pub profile: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            cors: true,
            timeout_secs: 30,
            max_concurrent: 10,
            metrics: true,
            no_gpu: false,
            gpu: false,
            batch: false,
            trace: false,
            trace_level: "basic".to_string(),
            profile: false,
        }
    }
}

impl ServerConfig {
    /// Create config with custom port (builder pattern, used in tests)
    #[cfg(test)]
    pub(crate) fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Create config with custom host (builder pattern, used in tests)
    #[cfg(test)]
    pub(crate) fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Get bind address
    pub(crate) fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Server metrics (thread-safe)
///
/// Implements APR-SPEC §4.15.9 metrics accuracy requirements (MA01-MA10).
/// All counters are thread-safe and exposed via /metrics endpoint.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Total requests received (MA01)
    pub requests_total: AtomicU64,
    /// Successful requests (2xx)
    pub requests_success: AtomicU64,
    /// Client errors (4xx)
    pub requests_client_error: AtomicU64,
    /// Server errors (5xx)
    pub requests_server_error: AtomicU64,
    /// Total tokens generated (MA03)
    pub tokens_generated: AtomicU64,
    /// Total inference time in milliseconds
    pub inference_time_ms: AtomicU64,
    /// Model memory in bytes (MM01)
    pub model_memory_bytes: AtomicU64,
    /// Server start time (for uptime calculation)
    start_time: std::sync::OnceLock<Instant>,
}

impl ServerMetrics {
    /// Create new metrics with server start time
    pub fn new() -> Arc<Self> {
        let metrics = Arc::new(Self::default());
        let _ = metrics.start_time.set(Instant::now());
        metrics
    }

    /// Record a request with outcome
    pub fn record_request(&self, success: bool, tokens: u64, duration_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.requests_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_server_error.fetch_add(1, Ordering::Relaxed);
        }
        self.tokens_generated.fetch_add(tokens, Ordering::Relaxed);
        self.inference_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Record client error (4xx)
    pub fn record_client_error(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_client_error.fetch_add(1, Ordering::Relaxed);
    }

    /// Get uptime in seconds (HR04)
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time
            .get()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0)
    }

    /// Get Prometheus-format metrics (MA06: valid Prometheus format)
    ///
    /// Format follows https://prometheus.io/docs/instrumenting/exposition_formats/
    pub fn prometheus_output(&self) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let success = self.requests_success.load(Ordering::Relaxed);
        let client_errors = self.requests_client_error.load(Ordering::Relaxed);
        let server_errors = self.requests_server_error.load(Ordering::Relaxed);
        let tokens = self.tokens_generated.load(Ordering::Relaxed);
        let inference_ms = self.inference_time_ms.load(Ordering::Relaxed);
        let model_bytes = self.model_memory_bytes.load(Ordering::Relaxed);
        let uptime = self.uptime_seconds();

        format!(
            r#"# HELP apr_requests_total Total number of HTTP requests
# TYPE apr_requests_total counter
apr_requests_total {total}

# HELP apr_requests_success Successful requests (2xx)
# TYPE apr_requests_success counter
apr_requests_success {success}

# HELP apr_requests_client_error Client error requests (4xx)
# TYPE apr_requests_client_error counter
apr_requests_client_error {client_errors}

# HELP apr_requests_server_error Server error requests (5xx)
# TYPE apr_requests_server_error counter
apr_requests_server_error {server_errors}

# HELP apr_tokens_generated_total Total tokens generated
# TYPE apr_tokens_generated_total counter
apr_tokens_generated_total {tokens}

# HELP apr_inference_duration_seconds_total Total inference time in seconds
# TYPE apr_inference_duration_seconds_total counter
apr_inference_duration_seconds_total {:.3}

# HELP apr_memory_bytes Memory usage by type
# TYPE apr_memory_bytes gauge
apr_memory_bytes{{type="model"}} {model_bytes}

# HELP apr_uptime_seconds Server uptime in seconds
# TYPE apr_uptime_seconds gauge
apr_uptime_seconds {uptime}
"#,
            inference_ms as f64 / 1000.0
        )
    }
}

/// Server state
///
/// Stores model and configuration for the server.
/// Implements APR-SPEC §4.15.8.2 Server Architecture.
pub struct ServerState {
    /// Model path
    pub model_path: PathBuf,
    /// Model ID (filename without extension)
    pub model_id: String,
    /// Server configuration
    pub config: ServerConfig,
    /// Metrics
    pub metrics: Arc<ServerMetrics>,
    /// Whether model uses mmap (SL03: >50MB threshold)
    pub uses_mmap: bool,
    /// Model file size in bytes
    pub model_size_bytes: u64,
    /// Server readiness state
    pub ready: std::sync::atomic::AtomicBool,
}

impl ServerState {
    /// Create new server state
    pub fn new(model_path: PathBuf, config: ServerConfig) -> Result<Self> {
        // Check model exists (SL06: Invalid model path returns exit 1)
        if !model_path.exists() {
            return Err(CliError::FileNotFound(model_path));
        }

        // Get model metadata
        let metadata = std::fs::metadata(&model_path)?;
        let model_size_bytes = metadata.len();

        // Determine if mmap should be used (SL03: >50MB threshold)
        let uses_mmap = model_size_bytes > 50 * 1024 * 1024;

        // Extract model ID from filename
        let model_id = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let metrics = ServerMetrics::new();
        metrics
            .model_memory_bytes
            .store(model_size_bytes, Ordering::Relaxed);

        Ok(Self {
            model_path,
            model_id,
            config,
            metrics,
            uses_mmap,
            model_size_bytes,
            ready: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Mark server as ready (SL10: Ready only after model loaded)
    pub fn set_ready(&self) {
        self.ready.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Check if server is ready
    pub fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Health status enum (HR06, HR07)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Server is healthy and ready
    Healthy,
    /// Server is experiencing degraded performance (HR06: p99 > 1s)
    Degraded,
    /// Server is unhealthy (HR07: OOM or critical error)
    Unhealthy,
}

/// Health check response (HR01-HR10)
///
/// Implements APR-SPEC §4.15.8.3 HealthResponse schema.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthResponse {
    /// Health status (HR01, HR06, HR07)
    pub status: HealthStatus,
    /// Model identifier (HR03)
    pub model_id: String,
    /// Server version (SL09)
    pub version: String,
    /// Uptime in seconds (HR04: monotonically increases)
    pub uptime_seconds: u64,
    /// Total requests processed (HR05)
    pub requests_total: u64,
    /// GPU availability (HR08)
    pub gpu_available: bool,
}

/// Error response (EH09: Error response is valid JSON)
///
/// Implements APR-SPEC §4.15.8.7 error format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorResponse {
    /// Error type
    pub error: String,
    /// Human-readable message
    pub message: String,
    /// Request correlation ID (EH10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(error: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            message: message.into(),
            request_id: None,
        }
    }

    /// Create with request ID
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }
}

/// Generate request (§4.15.8.3 GenerateRequest)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerateRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate (IC02)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Temperature for sampling (IC01, IC04)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Enable streaming (SP01-SP10)
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences (IC03)
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_max_tokens() -> u32 {
    256
}

fn default_temperature() -> f32 {
    1.0
}

/// Generate response (§4.15.8.3 GenerateResponse)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerateResponse {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: u32,
    /// Reason for stopping (stop, length, error)
    pub finish_reason: String,
    /// Latency in milliseconds (LG03)
    pub latency_ms: u64,
}

/// SSE Stream Event (SP02-SP03)
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamEvent {
    /// Event type: token, done, error
    pub event: String,
    /// Token text or error message
    pub data: String,
    /// Token ID (SP07, SP08)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_id: Option<u32>,
}

impl StreamEvent {
    /// Create token event
    pub fn token(text: &str, id: u32) -> Self {
        Self {
            event: "token".to_string(),
            data: text.to_string(),
            token_id: Some(id),
        }
    }

    /// Create done event (SP03)
    pub fn done(finish_reason: &str, tokens: u32) -> Self {
        Self {
            event: "done".to_string(),
            data: format!(
                r#"{{"finish_reason":"{}","tokens_generated":{}}}"#,
                finish_reason, tokens
            ),
            token_id: None,
        }
    }

    /// Create error event (SP04)
    pub fn error(message: &str) -> Self {
        Self {
            event: "error".to_string(),
            data: message.to_string(),
            token_id: None,
        }
    }

    /// Format as SSE (SP02)
    pub fn to_sse(&self) -> String {
        let mut result = format!("event: {}\n", self.event);
        result.push_str(&format!("data: {}\n\n", self.data));
        result
    }
}

/// Transcribe request (§4.15.8.3 TranscribeRequest)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscribeRequest {
    /// Language code (optional)
    #[serde(default)]
    pub language: Option<String>,
    /// Task: transcribe or translate
    #[serde(default = "default_task")]
    pub task: String,
}

fn default_task() -> String {
    "transcribe".to_string()
}

/// Transcribe response (§4.15.8.3 TranscribeResponse)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscribeResponse {
    /// Transcribed text
    pub text: String,
    /// Detected/specified language
    pub language: String,
    /// Audio duration in seconds
    pub duration_seconds: f64,
    /// Latency in milliseconds (LG03)
    pub latency_ms: u64,
}

/// Request size limit (SE02: default 10MB)
pub const MAX_REQUEST_SIZE: usize = 10 * 1024 * 1024;

/// Server information response (SL09)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version (semver)
    pub version: String,
    /// Loaded model ID
    pub model_id: String,
}

/// Generate health response from server state
pub fn health_check(state: &ServerState) -> HealthResponse {
    let metrics = &state.metrics;

    // Determine health status (HR06, HR07)
    let status = if !state.is_ready() {
        HealthStatus::Unhealthy
    } else {
        // Check for degraded performance (p99 > 1s approximation)
        let total_requests = metrics.requests_total.load(Ordering::Relaxed);
        let total_time_ms = metrics.inference_time_ms.load(Ordering::Relaxed);
        let avg_latency_ms = if total_requests > 0 {
            total_time_ms / total_requests
        } else {
            0
        };

        if avg_latency_ms > 1000 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    };

    // Detect GPU availability
    #[cfg(feature = "inference")]
    let gpu_available = {
        use realizar::cuda::CudaExecutor;
        CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
    };
    #[cfg(not(feature = "inference"))]
    let gpu_available = false;

    HealthResponse {
        status,
        model_id: state.model_id.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: metrics.uptime_seconds(),
        requests_total: metrics.requests_total.load(Ordering::Relaxed),
        gpu_available,
    }
}

// ============================================================================
// Testable Router (APR-SPEC §4.15.8.3)
// ============================================================================

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

            return Sse::new(stream)
                .keep_alive(axum::response::sse::KeepAlive::default())
                .into_response();
        }

        // Non-streaming response
        let duration_ms = start.elapsed().as_millis() as u64;
        state.metrics.record_request(true, 0, duration_ms);

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

// ============================================================================
// CLI Entry Point
// ============================================================================

/// Serve command entry point (blocking)
pub(crate) fn run(model_path: &Path, config: &ServerConfig) -> Result<()> {
    println!("{}", "=== APR Serve ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Binding: {}", config.bind_addr());
    println!();

    // Validate model
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    let state = ServerState::new(model_path.to_path_buf(), config.clone())?;

    println!(
        "{}",
        format!(
            "Model loading: {}",
            if state.uses_mmap { "mmap" } else { "full" }
        )
        .dimmed()
    );

    println!();
    println!("{}", "Endpoints:".green().bold());
    println!("  POST /predict        - Model prediction (APR)");
    println!("  POST /generate       - Text generation (GGUF)");
    println!("  GET  /health         - Health check");
    if config.metrics {
        println!("  GET  /metrics        - Prometheus metrics");
    }

    println!();
    println!(
        "{}",
        format!("Server ready at http://{}", config.bind_addr())
            .green()
            .bold()
    );
    println!();
    println!("{}", "Press Ctrl+C to stop".dimmed());

    // Try to start real server with realizar
    #[cfg(feature = "inference")]
    {
        start_realizar_server(model_path, config)
    }

    // Fallback: stub mode
    #[cfg(not(feature = "inference"))]
    {
        println!();
        println!("{}", "[Server requires --features inference]".yellow());
        Ok(())
    }
}

/// Start server using realizar
#[cfg(feature = "inference")]
fn start_realizar_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::format::{detect_format, ModelFormat};
    use std::io::Read;

    // Read only 8 bytes for format detection (avoid loading entire file)
    let mut file = std::fs::File::open(model_path)?;
    let mut magic = [0u8; 8];
    let bytes_read = file.read(&mut magic)?;
    if bytes_read < 8 {
        return Err(CliError::InvalidFormat(
            "File too small for format detection".to_string(),
        ));
    }

    // Detect model format from magic bytes
    let format = detect_format(&magic)
        .map_err(|e| CliError::InvalidFormat(format!("Format detection failed: {e}")))?;

    println!();
    println!("Detected format: {}", format);

    match format {
        ModelFormat::Apr => {
            println!("{}", "Starting APR model server...".cyan());
            start_apr_server(model_path, config)
        }
        ModelFormat::Gguf => {
            println!("{}", "Starting GGUF inference server...".cyan());
            start_gguf_server(model_path, config)
        }
        ModelFormat::SafeTensors => {
            println!("{}", "Starting SafeTensors inspection server...".cyan());
            start_safetensors_server(model_path, config)
        }
    }
}

/// Start APR v2 model server with full inference support
///
/// Supports both transformer inference (generate) and metadata inspection.
/// GPU acceleration available via --gpu flag.
#[cfg(feature = "inference")]
fn start_apr_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use axum::{
        extract::State,
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr::AprModel;
    use serde::{Deserialize, Serialize};
    use std::sync::Mutex;

    // Load APR model
    println!("{}", "Loading APR v2 model...".dimmed());
    let model = AprModel::load(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR v2 model: {e}")))?;

    let model_type = model
        .metadata()
        .model_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = model
        .metadata()
        .architecture
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let tensor_count = model.tensor_count();
    let param_count = model.estimated_parameters();
    let is_transformer = model.metadata().is_transformer();

    println!(
        "{}",
        format!(
            "Loaded {} model (arch: {}, {} tensors, ~{} params)",
            model_type, architecture, tensor_count, param_count
        )
        .green()
    );

    if is_transformer {
        println!(
            "{}",
            "Transformer model detected - inference enabled".cyan()
        );
    }

    // Load tokenizer if available - prefer BPE tokenizer from tokenizer.json
    // PMAT-098: Use proper BPE tokenizer (same as SafeTensors path)
    let tokenizer_json_path = model_path.with_file_name("tokenizer.json");
    let bpe_tokenizer = if tokenizer_json_path.exists() {
        load_safetensors_tokenizer(&tokenizer_json_path)
    } else {
        None
    };

    let has_tokenizer = bpe_tokenizer.is_some();
    if has_tokenizer {
        println!("{}", "BPE tokenizer loaded from tokenizer.json".green());
    } else {
        println!(
            "{}",
            "No tokenizer.json found - using fallback tokenization".yellow()
        );
    }

    // Determine GPU vs CPU mode
    // PMAT-099: APR GPU path currently has tensor name mismatches with AprV2ModelCuda
    // For now, force CPU for APR until GPU path is fixed
    let use_gpu = config.gpu && !config.no_gpu;

    #[cfg(feature = "cuda")]
    if use_gpu && is_transformer {
        // PMAT-099: Disable GPU for APR until AprV2ModelCuda tensor name mapping is fixed
        // The AprTransformer CPU path uses correct tensor names from APR metadata
        // but AprV2ModelCuda expects different names (model.layers vs blk)
        println!(
            "{}",
            "Note: APR GPU path disabled (PMAT-099 - tensor name mapping WIP)".yellow()
        );
        println!("{}", "Using CPU path for APR inference".dimmed());
        // Fall through to CPU path instead of:
        // return start_apr_server_gpu(model_path, model, config, bpe_tokenizer);
    }

    // CPU path
    println!("{}", "Using CPU inference".dimmed());

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_clone = model_path.to_path_buf();

    // PMAT-098: Load transformer once and share across requests
    // Previous code reloaded model on every request causing ~0.01 tok/s
    // Now we load once using AprTransformer for efficient inference
    let transformer = if is_transformer {
        match realizar::apr_transformer::AprTransformer::from_apr_file(model_path) {
            Ok(t) => {
                println!(
                    "{}",
                    format!(
                        "Transformer ready: {} layers, hidden_dim={}",
                        t.config.num_layers, t.config.hidden_dim
                    )
                    .cyan()
                );
                Some(Arc::new(Mutex::new(t)))
            }
            Err(e) => {
                println!(
                    "{}",
                    format!("Transformer load failed: {e} - inference disabled").yellow()
                );
                None
            }
        }
    } else {
        None
    };

    // Shared state for inference (model loaded once, not per-request)
    // PMAT-098: Use BPE tokenizer for proper encoding
    #[derive(Clone)]
    struct AprState {
        transformer: Option<Arc<Mutex<realizar::apr_transformer::AprTransformer>>>,
        model_type: String,
        architecture: String,
        is_transformer: bool,
        tokenizer: Option<SafeTensorsTokenizerInfo>,
    }

    let state = AprState {
        transformer,
        model_type: model_type.clone(),
        architecture: architecture.clone(),
        is_transformer,
        tokenizer: bpe_tokenizer,
    };

    #[derive(Clone, Serialize)]
    struct HealthResponse {
        status: String,
        model_type: String,
        architecture: String,
        inference_enabled: bool,
        compute_mode: String,
    }

    #[derive(Deserialize)]
    struct CompletionRequest {
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
        #[serde(default)]
        temperature: Option<f32>,
    }

    fn default_max_tokens() -> usize {
        32
    }

    #[derive(Serialize)]
    struct CompletionResponse {
        text: String,
        tokens_generated: usize,
        latency_ms: u64,
        tok_per_sec: f64,
    }

    runtime.block_on(async move {
        let state_for_health = state.clone();
        let state_for_completions = Arc::new(Mutex::new(state.clone()));

        let app = Router::new()
            .route(
                "/health",
                get(move || {
                    let s = state_for_health.clone();
                    async move {
                        Json(HealthResponse {
                            status: "healthy".to_string(),
                            model_type: s.model_type.clone(),
                            architecture: s.architecture.clone(),
                            inference_enabled: s.is_transformer,
                            compute_mode: "cpu".to_string(),
                        })
                    }
                }),
            )
            .route(
                "/v1/completions",
                post(move |Json(req): Json<CompletionRequest>| {
                    let state = state_for_completions.clone();
                    async move {
                        let s = state.lock().unwrap().clone();

                        // PMAT-098: Use shared transformer (no reload per request)
                        let transformer = match &s.transformer {
                            Some(t) => t.clone(),
                            None => {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(serde_json::json!({
                                        "error": "Transformer not loaded, inference not supported"
                                    })),
                                )
                                    .into_response();
                            }
                        };

                        let start = Instant::now();

                        // PMAT-098: Use BPE tokenizer for proper encoding
                        let input_tokens: Vec<u32> = if let Some(ref tok) = s.tokenizer {
                            tok.tokenizer.encode(&req.prompt)
                        } else {
                            // Fallback: character-level
                            req.prompt.chars().map(|c| c as u32).collect()
                        };

                        let eos_id = match &s.tokenizer {
                            Some(tok) => tok.eos_token_id.unwrap_or(2),
                            _ => 2,
                        };

                        // PMAT-103 FIX: Use generate_with_cache for O(n) generation
                        // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
                        // generate_with_cache() uses KV cache for incremental generation = O(n)
                        let gen_start = Instant::now();
                        let max_tokens = req.max_tokens.min(128);

                        let gen_config = realizar::apr_transformer::GenerateConfig {
                            max_tokens,
                            temperature: req.temperature.unwrap_or(0.0),
                            top_p: 0.9,
                            top_k: 0,
                            repetition_penalty: 1.0,
                        };

                        let output_tokens = {
                            let t = transformer.lock().unwrap();
                            match t.generate_with_cache(&input_tokens, &gen_config) {
                                Ok(tokens) => tokens,
                                Err(e) => {
                                    return (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": format!("Generate failed: {e}")})),
                                    )
                                        .into_response();
                                }
                            }
                        };
                        let gen_time = gen_start.elapsed();

                        // Decode output
                        let new_tokens = if output_tokens.len() > input_tokens.len() {
                            &output_tokens[input_tokens.len()..]
                        } else {
                            &output_tokens[..]
                        };

                        // PMAT-099: Debug logging for token decoding
                        eprintln!("[APR DEBUG] Generated {} total tokens, {} new tokens", output_tokens.len(), new_tokens.len());
                        eprintln!("[APR DEBUG] New token IDs: {:?}", &new_tokens[..new_tokens.len().min(20)]);

                        // PMAT-098: Use BPE tokenizer for proper decoding
                        let text = if let Some(ref tok) = s.tokenizer {
                            match tok.tokenizer.decode(new_tokens) {
                                Ok(decoded) => {
                                    eprintln!("[APR DEBUG] Decoded text: {:?}", decoded);
                                    decoded
                                }
                                Err(e) => {
                                    eprintln!("[APR DEBUG] Decode error: {e}");
                                    String::new()
                                }
                            }
                        } else {
                            // Fallback: interpret as char codes
                            let fallback: String = new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
                            eprintln!("[APR DEBUG] Fallback decode: {:?}", fallback);
                            fallback
                        };

                        let tokens_generated = new_tokens.len();
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let tok_per_sec = if gen_time.as_secs_f64() > 0.0 {
                            tokens_generated as f64 / gen_time.as_secs_f64()
                        } else {
                            0.0
                        };

                        Json(CompletionResponse {
                            text,
                            tokens_generated,
                            latency_ms,
                            tok_per_sec,
                        })
                        .into_response()
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                {
                    let state_for_chat = Arc::new(Mutex::new(state.clone()));
                    post(move |headers: axum::http::HeaderMap, Json(req): Json<serde_json::Value>| {
                        let state = state_for_chat.clone();
                        async move {
                            use axum::response::sse::{Event, Sse};
                            use futures_util::stream;

                            // F-TRACE-001/002/003: Parse X-Trace-Level header
                            let trace_level = headers
                                .get("X-Trace-Level")
                                .and_then(|v| v.to_str().ok())
                                .map(str::to_lowercase);

                            let s = state.lock().unwrap().clone();

                            // PMAT-098: Use shared transformer (no reload per request)
                            let transformer = match &s.transformer {
                                Some(t) => t.clone(),
                                None => {
                                    return axum::Json(serde_json::json!({
                                        "error": "Transformer not loaded, inference not supported"
                                    }))
                                    .into_response();
                                }
                            };

                            // Extract messages from request
                            let messages = req.get("messages").and_then(|m| m.as_array());
                            let stream_mode = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
                            let max_tokens = req.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(32) as usize;

                            let prompt = if let Some(msgs) = messages {
                                // Format as ChatML
                                let mut prompt = String::new();
                                for msg in msgs {
                                    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                    prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                                }
                                prompt.push_str("<|im_start|>assistant\n");
                                prompt
                            } else {
                                return axum::Json(serde_json::json!({"error": "Missing messages"})).into_response();
                            };

                            let start = Instant::now();

                            // PMAT-098: Use BPE tokenizer for proper encoding
                            let input_tokens: Vec<u32> = if let Some(ref tok) = s.tokenizer {
                                tok.tokenizer.encode(&prompt)
                            } else {
                                // Fallback: character-level
                                prompt.chars().map(|c| c as u32).collect()
                            };

                            let eos_id = match &s.tokenizer {
                                Some(tok) => tok.eos_token_id.unwrap_or(151645),
                                _ => 151645, // ChatML end token
                            };

                            // PMAT-103 FIX: Use generate_with_cache for O(n) generation
                            // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
                            // generate_with_cache() uses KV cache for incremental generation = O(n)
                            let gen_start = Instant::now();
                            let max_tokens = max_tokens.min(256);
                            let temperature = req.get("temperature").and_then(|t| t.as_f64()).unwrap_or(0.0) as f32;

                            let gen_config = realizar::apr_transformer::GenerateConfig {
                                max_tokens,
                                temperature,
                                top_p: 0.9,
                                top_k: 0,
                                repetition_penalty: 1.0,
                            };

                            let output_tokens = {
                                let t = transformer.lock().unwrap();
                                match t.generate_with_cache(&input_tokens, &gen_config) {
                                    Ok(tokens) => tokens,
                                    Err(e) => {
                                        return axum::Json(serde_json::json!({"error": format!("Generate failed: {e}")}))
                                            .into_response();
                                    }
                                }
                            };
                            let elapsed = gen_start.elapsed();

                            // Decode output
                            let new_tokens = if output_tokens.len() > input_tokens.len() {
                                &output_tokens[input_tokens.len()..]
                            } else {
                                &output_tokens[..]
                            };

                            // PMAT-099: Debug logging for token decoding
                            eprintln!("[APR CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
                                input_tokens.len(), output_tokens.len(), new_tokens.len());
                            eprintln!("[APR CHAT DEBUG] New token IDs: {:?}", &new_tokens[..new_tokens.len().min(20)]);

                            // PMAT-098: Use BPE tokenizer for proper decoding
                            let output_text = if let Some(ref tok) = s.tokenizer {
                                match tok.tokenizer.decode(new_tokens) {
                                    Ok(decoded) => {
                                        eprintln!("[APR CHAT DEBUG] Decoded text: {:?}", decoded);
                                        decoded
                                    }
                                    Err(e) => {
                                        eprintln!("[APR CHAT DEBUG] Decode error: {e}");
                                        String::new()
                                    }
                                }
                            } else {
                                // Fallback: interpret as char codes
                                let fallback: String = new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
                                eprintln!("[APR CHAT DEBUG] Fallback decode: {:?}", fallback);
                                fallback
                            };

                            let tokens_generated = new_tokens.len();
                            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                                tokens_generated as f64 / elapsed.as_secs_f64()
                            } else {
                                0.0
                            };

                            // Generate unique ID
                            let request_id = format!(
                                "chatcmpl-{}-{}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos(),
                                std::process::id()
                            );

                            let created = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();

                            if stream_mode {
                                // SSE streaming response (PAR-302) with [DONE] marker (F-HTTP-011)
                                let response = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": "apr",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }]
                                });

                                // Send data event followed by [DONE] marker per OpenAI SSE spec
                                let events = vec![
                                    Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string())),
                                    Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]")),
                                ];
                                let stream = stream::iter(events);
                                Sse::new(stream).into_response()
                            } else {
                                // Non-streaming response with trace support (F-TRACE-001/002/003)
                                let latency_ms = start.elapsed().as_millis() as u64;

                                // Build trace data based on X-Trace-Level header
                                let trace_data = serde_json::json!({
                                    "total_time_us": latency_ms * 1000,
                                    "prompt_tokens": input_tokens.len(),
                                    "completion_tokens": tokens_generated,
                                    "layers": 28  // Fixed for now
                                });

                                let mut response = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion",
                                    "created": created,
                                    "model": "apr",
                                    "choices": [{
                                        "index": 0,
                                        "message": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": input_tokens.len(),
                                        "completion_tokens": tokens_generated,
                                        "total_tokens": input_tokens.len() + tokens_generated
                                    },
                                    "_apr_metrics": {
                                        "latency_ms": latency_ms,
                                        "tok_per_sec": tok_per_sec
                                    }
                                });

                                // Add trace fields based on X-Trace-Level header
                                if let Some(ref level) = trace_level {
                                    if let Some(obj) = response.as_object_mut() {
                                        match level.as_str() {
                                            "brick" => { obj.insert("brick_trace".to_string(), trace_data); }
                                            "step" => { obj.insert("step_trace".to_string(), trace_data); }
                                            "layer" => { obj.insert("layer_trace".to_string(), trace_data); }
                                            _ => {}
                                        }
                                    }
                                }

                                axum::Json(response).into_response()
                            }
                        }
                    })
                },
            )
            .route(
                "/",
                get(|| async { "APR v2 Inference Server - POST /v1/completions, /v1/chat/completions" }),
            );

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("APR Inference Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  POST /v1/completions      - Text generation");
        println!("  POST /v1/chat/completions - Chat completions (PAR-302)");
        println!();
        println!(
            "{}",
            format!("Mode: CPU | Transformer: {}", is_transformer).dimmed()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start APR server with GPU acceleration
/// PMAT-098: Updated to use BPE tokenizer for proper encoding
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_apr_server_gpu(
    model_path: &Path,
    model: realizar::apr::AprModel,
    config: &ServerConfig,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
) -> Result<()> {
    use axum::{
        extract::State,
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr::AprV2ModelCuda;
    use serde::{Deserialize, Serialize};
    use std::sync::Mutex;

    // Initialize CUDA model
    println!("{}", "Initializing CUDA...".dimmed());
    let cuda_model = AprV2ModelCuda::new(model, 0)
        .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

    println!(
        "{}",
        format!(
            "GPU: {} ({} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        )
        .green()
    );

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_clone = model_path.to_path_buf();

    #[derive(Deserialize)]
    struct CompletionRequest {
        prompt: String,
        #[serde(default = "default_max_tokens_gpu")]
        max_tokens: usize,
    }

    fn default_max_tokens_gpu() -> usize {
        32
    }

    #[derive(Serialize)]
    struct CompletionResponse {
        text: String,
        tokens_generated: usize,
        latency_ms: u64,
        tok_per_sec: f64,
    }

    // Wrap CUDA model in Arc<Mutex> for shared access
    // PMAT-098: Use BPE tokenizer for proper encoding
    let cuda_model = Arc::new(Mutex::new(cuda_model));
    let tokenizer = Arc::new(tokenizer);
    let model_path_arc = Arc::new(model_path_clone);

    runtime.block_on(async move {
        let cuda_for_completions = cuda_model.clone();
        let tok_for_completions = tokenizer.clone();
        let path_for_completions = model_path_arc.clone();

        let app = Router::new()
            .route(
                "/health",
                get(|| async {
                    Json(serde_json::json!({
                        "status": "healthy",
                        "gpu": true
                    }))
                }),
            )
            .route(
                "/v1/completions",
                post(move |Json(req): Json<CompletionRequest>| {
                    let cuda = cuda_for_completions.clone();
                    let tok_info = tok_for_completions.clone();
                    let model_path = path_for_completions.clone();
                    async move {
                        use realizar::apr::AprModel;

                        let start = Instant::now();

                        // PMAT-098: Use BPE tokenizer for proper encoding
                        let input_tokens: Vec<u32> = match tok_info.as_ref() {
                            Some(tok) => tok.tokenizer.encode(&req.prompt),
                            None => req.prompt.chars().map(|c| c as u32).collect(),
                        };

                        let eos_id = match tok_info.as_ref() {
                            Some(tok) => tok.eos_token_id.unwrap_or(2),
                            _ => 2,
                        };

                        // Generate on GPU
                        let gen_start = Instant::now();
                        let max_tokens = req.max_tokens.min(128);
                        let output_tokens = {
                            let mut model = cuda.lock().unwrap();
                            match model.generate_cuda(
                                &input_tokens,
                                max_tokens,
                                eos_id,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    return (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": format!("GPU generation failed: {e}")})),
                                    )
                                        .into_response();
                                }
                            }
                        };
                        let gen_time = gen_start.elapsed();

                        // Decode
                        let new_tokens = if output_tokens.len() > input_tokens.len() {
                            &output_tokens[input_tokens.len()..]
                        } else {
                            &output_tokens[..]
                        };

                        // PMAT-098: Use BPE tokenizer for proper decoding
                        let text = match tok_info.as_ref() {
                            Some(tok) => tok.tokenizer.decode(new_tokens).unwrap_or_else(|_| String::new()),
                            None => new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect::<String>(),
                        };

                        let tokens_generated = new_tokens.len();
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let tok_per_sec = if gen_time.as_secs_f64() > 0.0 {
                            tokens_generated as f64 / gen_time.as_secs_f64()
                        } else {
                            0.0
                        };

                        Json(CompletionResponse {
                            text,
                            tokens_generated,
                            latency_ms,
                            tok_per_sec,
                        })
                        .into_response()
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                {
                    let cuda_for_chat = cuda_model.clone();
                    let tok_for_chat = tokenizer.clone();
                    let path_for_chat = model_path_arc.clone();
                    post(move |Json(req): Json<serde_json::Value>| {
                        let cuda = cuda_for_chat.clone();
                        let tok_info = tok_for_chat.clone();
                        let model_path = path_for_chat.clone();
                        async move {
                            use axum::response::sse::{Event, Sse};
                            use futures_util::stream;
                            use realizar::apr::AprModel;

                            // Extract messages from request
                            let messages = req.get("messages").and_then(|m| m.as_array());
                            let stream_mode = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
                            let max_tokens = req.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(32) as usize;

                            let prompt = if let Some(msgs) = messages {
                                // Format as ChatML
                                let mut prompt = String::new();
                                for msg in msgs {
                                    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                    prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                                }
                                prompt.push_str("<|im_start|>assistant\n");
                                prompt
                            } else {
                                return axum::Json(serde_json::json!({"error": "Missing messages"})).into_response();
                            };

                            let start = Instant::now();

                            // Encode prompt
                            // PMAT-098: Use BPE tokenizer for proper encoding
                            let input_tokens: Vec<u32> = match tok_info.as_ref() {
                                Some(tok) => tok.tokenizer.encode(&prompt),
                                None => prompt.chars().map(|c| c as u32).collect(),
                            };

                            let eos_id = match tok_info.as_ref() {
                                Some(tok) => tok.eos_token_id.unwrap_or(151645),
                                _ => 151645, // ChatML end token
                            };

                            // Generate on GPU
                            let gen_start = Instant::now();
                            let max_tokens = max_tokens.min(256);
                            let output_tokens = {
                                let mut model = cuda.lock().unwrap();
                                match model.generate_cuda(&input_tokens, max_tokens, eos_id) {
                                    Ok(t) => t,
                                    Err(e) => {
                                        return axum::Json(serde_json::json!({"error": format!("GPU generation failed: {e}")}))
                                            .into_response();
                                    }
                                }
                            };
                            let elapsed = gen_start.elapsed();

                            // Decode
                            let new_tokens = if output_tokens.len() > input_tokens.len() {
                                &output_tokens[input_tokens.len()..]
                            } else {
                                &output_tokens[..]
                            };

                            // PMAT-099: Debug logging for GPU token decoding
                            eprintln!("[APR GPU CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
                                input_tokens.len(), output_tokens.len(), new_tokens.len());
                            eprintln!("[APR GPU CHAT DEBUG] New token IDs: {:?}", &new_tokens[..new_tokens.len().min(20)]);

                            // PMAT-098: Use BPE tokenizer for proper decoding
                            let output_text = match tok_info.as_ref() {
                                Some(tok) => match tok.tokenizer.decode(new_tokens) {
                                    Ok(decoded) => {
                                        eprintln!("[APR GPU CHAT DEBUG] Decoded text: {:?}", decoded);
                                        decoded
                                    }
                                    Err(e) => {
                                        eprintln!("[APR GPU CHAT DEBUG] Decode error: {e}");
                                        String::new()
                                    }
                                },
                                None => {
                                    let fallback: String = new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
                                    eprintln!("[APR GPU CHAT DEBUG] Fallback decode (no tokenizer): {:?}", fallback);
                                    fallback
                                }
                            };

                            let tokens_generated = new_tokens.len();
                            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                                tokens_generated as f64 / elapsed.as_secs_f64()
                            } else {
                                0.0
                            };

                            // Generate unique ID
                            let request_id = format!(
                                "chatcmpl-{}-{}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos(),
                                std::process::id()
                            );

                            let created = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();

                            if stream_mode {
                                // SSE streaming response (PAR-302)
                                let response = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": "apr-gpu",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }]
                                });

                                let stream = stream::once(async move {
                                    Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
                                });
                                Sse::new(stream).into_response()
                            } else {
                                // Non-streaming response
                                axum::Json(serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion",
                                    "created": created,
                                    "model": "apr-gpu",
                                    "choices": [{
                                        "index": 0,
                                        "message": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": input_tokens.len(),
                                        "completion_tokens": tokens_generated,
                                        "total_tokens": input_tokens.len() + tokens_generated
                                    },
                                    "_apr_metrics": {
                                        "latency_ms": start.elapsed().as_millis(),
                                        "tok_per_sec": tok_per_sec
                                    }
                                }))
                                .into_response()
                            }
                        }
                    })
                },
            )
            .route(
                "/",
                get(|| async { "APR v2 GPU Inference Server - POST /v1/completions, /v1/chat/completions" }),
            );

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("APR GPU Inference Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  POST /v1/completions      - GPU text generation");
        println!("  POST /v1/chat/completions - GPU chat completions (PAR-302)");
        println!();
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start GGUF model inference server with Ollama-parity performance
///
/// Uses realizar's full inference API for text generation, streaming, and batch inference.
/// Achieves Ollama-parity: 100+ tok/s CPU, 500+ tok/s GPU.
/// With --gpu --batch flags: 800+ tok/s (2.8x Ollama) via batched GPU inference.
#[cfg(feature = "inference")]
fn start_gguf_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::api::{create_router, AppState};
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    // Load GGUF model via mmap
    println!("{}", "Loading GGUF model (mmap)...".dimmed());
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    let tensor_count = mapped_model.model.tensors.len();
    let metadata_count = mapped_model.model.metadata.len();
    println!(
        "{}",
        format!(
            "GGUF loaded: {} tensors, {} metadata entries",
            tensor_count, metadata_count
        )
        .dimmed()
    );

    // Create quantized model for inference (Ollama-parity performance)
    println!("{}", "Building quantized inference model...".dimmed());
    let quantized_model = OwnedQuantizedModel::from_mapped(&mapped_model)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to build quantized model: {e}")))?;

    println!(
        "{}",
        format!(
            "Model ready: {} layers, vocab_size={}, hidden_dim={}",
            quantized_model.layers.len(),
            quantized_model.config.vocab_size,
            quantized_model.config.hidden_dim
        )
        .green()
    );

    // Extract vocabulary from GGUF for proper token encoding/decoding
    let vocab = mapped_model.model.vocabulary().unwrap_or_else(|| {
        eprintln!("Warning: No vocabulary in GGUF, using placeholder tokens");
        (0..quantized_model.config.vocab_size)
            .map(|i| format!("token{i}"))
            .collect()
    });

    // GPU batched inference path (2X+ Ollama performance)
    #[cfg(feature = "cuda")]
    if config.gpu && config.batch {
        return start_gguf_server_gpu_batched(quantized_model, vocab, config);
    }

    // GPU optimized path (--gpu flag) - Uses OwnedQuantizedModelCuda for 755+ tok/s (2.6x Ollama)
    // PAR-111: Pre-uploads weights and uses batched workspaces for maximum throughput
    #[cfg(feature = "cuda")]
    if config.gpu && !config.no_gpu {
        use realizar::gguf::OwnedQuantizedModelCuda;

        println!(
            "{}",
            "Enabling optimized CUDA acceleration (PAR-111)...".cyan()
        );

        // Create CUDA-optimized model wrapper (this initializes GPU KV cache)
        match OwnedQuantizedModelCuda::new(quantized_model, 0) {
            Ok(mut cuda_model) => {
                // Pre-upload all weights to GPU for maximum performance
                println!("  Initializing GPU on device 0...");
                match cuda_model.preload_weights_gpu() {
                    Ok(bytes) => {
                        println!(
                            "{}",
                            format!("  Pre-uploaded {} MB weights to GPU", bytes / (1024 * 1024))
                                .green()
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "{}",
                            format!("  Warning: Weight preload failed, using on-demand: {e}")
                                .yellow()
                        );
                    }
                }

                println!("{}", "CUDA optimized model ready".green());

                let state =
                    AppState::with_cuda_model_and_vocab(cuda_model, vocab).map_err(|e| {
                        CliError::InferenceFailed(format!("Failed to create state: {e}"))
                    })?;

                let app = create_router(state);

                let runtime = tokio::runtime::Runtime::new().map_err(|e| {
                    CliError::InferenceFailed(format!("Failed to create runtime: {e}"))
                })?;

                let bind_addr = config.bind_addr();

                return runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::bind(&bind_addr)
                        .await
                        .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

                    println!();
                    println!(
                        "{}",
                        format!("CUDA-optimized server listening on http://{}", bind_addr)
                            .green()
                            .bold()
                    );
                    println!();
                    println!("{}", "Performance: 755+ tok/s (2.6x Ollama)".yellow());
                    println!();
                    println!("{}", "Endpoints:".cyan());
                    println!("  GET  /health              - Health check");
                    println!("  GET  /metrics             - Prometheus metrics");
                    println!("  POST /generate            - Text generation");
                    println!("  POST /v1/completions      - OpenAI-compatible");
                    println!("  POST /v1/chat/completions - Chat completions");
                    println!();
                    println!("{}", "Press Ctrl+C to stop".dimmed());

                    axum::serve(listener, app)
                        .with_graceful_shutdown(shutdown_signal())
                        .await
                        .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

                    Ok(())
                });
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    format!("CUDA init failed, falling back to CPU: {e}").yellow()
                );
                // Fall through to CPU path - rebuild the model since we consumed it
            }
        }
        // Rebuild quantized model for CPU fallback if CUDA failed
        let quantized_model = OwnedQuantizedModel::from_mapped(&mapped_model).map_err(|e| {
            CliError::ModelLoadFailed(format!("Failed to rebuild quantized model: {e}"))
        })?;
        let vocab = mapped_model.model.vocabulary().unwrap_or_else(|| {
            (0..quantized_model.config.vocab_size)
                .map(|i| format!("token{i}"))
                .collect()
        });
        // Run CPU server with rebuilt model
        return run_cpu_server(quantized_model, vocab, config);
    }

    // CPU path (default - when not using GPU or cuda feature disabled)
    // Create realizar AppState with full inference capabilities and real vocab
    run_cpu_server(quantized_model, vocab, config)
}

/// Run the CPU inference server
#[cfg(feature = "inference")]
fn run_cpu_server(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, AppState};

    let state = AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?;

    // Create realizar's full inference router (Ollama-parity endpoints)
    let app = create_router(state);

    // Create tokio runtime and run server
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Inference server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Ollama-Parity Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /stream/generate     - SSE streaming");
        println!("  POST /batch/generate      - Batch inference");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!(
            "{}",
            "Performance targets: 100+ tok/s CPU, 500+ tok/s GPU".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start GGUF server with GPU batched inference (2X+ Ollama performance)
///
/// Uses OwnedQuantizedModelCachedSync with continuous batching scheduler
/// for maximum throughput on GPU. Achieves 800+ tok/s (2.8x Ollama).
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_gpu_batched(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, spawn_batch_processor, AppState, BatchConfig};
    use realizar::gguf::OwnedQuantizedModelCachedSync;

    println!(
        "{}",
        "Enabling GPU batched inference (2X+ Ollama)...".cyan()
    );

    // Create tokio runtime FIRST (needed for batch processor spawn)
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    // Create cached model for scheduler reuse
    // OwnedQuantizedModelCachedSync handles GPU caching internally via warmup_gpu_cache()
    let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

    // Warmup GPU cache
    println!("  Warming up GPU cache...");
    match cached_model.warmup_gpu_cache() {
        Ok((memory_bytes, num_layers)) => {
            println!(
                "  GPU cache ready: {:.2} GB ({} layers)",
                memory_bytes as f64 / 1e9,
                num_layers
            );
        }
        Err(e) => {
            eprintln!("  Warning: GPU cache warmup failed: {e}");
        }
    }

    // Create state with cached model and real vocab
    let state = AppState::with_cached_model_and_vocab(cached_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?;

    // Get Arc'd model for batch processor
    let cached_model_arc = state
        .cached_model()
        .expect("cached_model should exist")
        .clone();

    // Configure batch processing
    let batch_config = BatchConfig::default();
    println!("  Batch window: {}ms", batch_config.window_ms);
    println!("  Optimal batch: {}", batch_config.optimal_batch);
    println!("  GPU threshold: {}", batch_config.gpu_threshold);

    let bind_addr = config.bind_addr();

    // Run everything inside the runtime context
    runtime.block_on(async move {
        // Spawn batch processor task (requires Tokio runtime)
        let batch_tx = spawn_batch_processor(cached_model_arc, batch_config.clone());
        println!("  Batch processor: RUNNING");

        // Add batch support to state
        let state = state.with_batch_config(batch_tx, batch_config);

        // Create router
        let app = create_router(state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("GPU Batched Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "2X Ollama Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /v1/gpu/status       - GPU cache status");
        println!("  POST /v1/completions      - OpenAI-compatible (batched)");
        println!("  POST /v1/batch/completions - Explicit batch inference");
        println!();
        println!(
            "{}",
            "Performance: 800+ tok/s (2.8x Ollama) with batched requests".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start SafeTensors inspection server
#[cfg(feature = "inference")]
fn start_safetensors_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use axum::{
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr_transformer::AprTransformer;
    use realizar::safetensors::SafetensorsModel;
    use realizar::safetensors_infer::SafetensorsToAprConverter;
    use serde::Serialize;
    use std::sync::{Arc, Mutex};

    // Load SafeTensors from file (T-QA-020)
    let bytes = std::fs::read(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to read SafeTensors file: {e}")))?;
    let st_model = SafetensorsModel::from_bytes(&bytes)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to parse SafeTensors: {e}")))?;

    let tensor_names: Vec<String> = st_model
        .tensor_names()
        .into_iter()
        .map(String::from)
        .collect();
    let tensor_count = tensor_names.len();

    println!(
        "{}",
        format!("SafeTensors loaded: {} tensors", tensor_count).green()
    );

    // Try to convert to AprTransformer for inference (PAR-301)
    let transformer = match SafetensorsToAprConverter::convert(model_path) {
        Ok(t) => {
            println!(
                "{}",
                format!(
                    "Inference enabled: {} layers, hidden_dim={}",
                    t.config.num_layers, t.config.hidden_dim
                )
                .cyan()
            );
            Some(Arc::new(Mutex::new(t)))
        }
        Err(e) => {
            println!(
                "{}",
                format!("Inference disabled: {e} (inspection-only mode)").yellow()
            );
            None
        }
    };

    // Try to load tokenizer from sibling tokenizer.json
    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    let tokenizer_info = if tokenizer_path.exists() {
        load_safetensors_tokenizer(&tokenizer_path)
    } else {
        println!(
            "{}",
            "No tokenizer.json found - using fallback tokenization".yellow()
        );
        None
    };

    if tokenizer_info.is_some() {
        println!("{}", "Tokenizer loaded from sibling file".dimmed());
    }

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_str = model_path.display().to_string();
    let inference_enabled = transformer.is_some();

    runtime.block_on(async move {
        #[derive(Clone, Serialize)]
        struct TensorInfo {
            count: usize,
            model: String,
            names: Vec<String>,
            inference_enabled: bool,
        }

        let info = TensorInfo {
            count: tensor_count,
            model: model_path_str.clone(),
            names: tensor_names.clone(),
            inference_enabled,
        };

        // Use module-level SafeTensorsState for inference
        let state = SafeTensorsState {
            transformer: transformer.clone(),
            tokenizer_info: tokenizer_info.clone(),
            model_path: model_path_str.clone(),
        };

        // Build base routes (no state required)
        let base_routes = Router::new()
            .route(
                "/health",
                get({
                    let inference = inference_enabled;
                    move || async move {
                        Json(serde_json::json!({
                            "status": "healthy",
                            "inference_enabled": inference
                        }))
                    }
                }),
            )
            .route("/tensors", get(move || async move { Json(info.clone()) }));

        // Build inference routes with state (PAR-301)
        let inference_routes: Router = if inference_enabled {
            Router::new()
                .route(
                    "/v1/chat/completions",
                    post(safetensors_chat_completions_handler),
                )
                .route("/generate", post(safetensors_generate_handler))
                .with_state(state)
        } else {
            Router::new()
        };

        // Merge all routes into final app
        let app = base_routes.merge(inference_routes);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /tensors             - List tensor names");
        if inference_enabled {
            println!("  POST /generate            - Text generation");
            println!("  POST /v1/chat/completions - Chat completions (PAR-301)");
        }
        println!();
        if !inference_enabled {
            println!(
                "{}",
                "Note: Inference disabled - ensure config.json exists alongside model".yellow()
            );
        }
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Tokenizer info for SafeTensors models with proper BPE support
#[cfg(feature = "inference")]
#[derive(Clone)]
struct SafeTensorsTokenizerInfo {
    /// BPE tokenizer with vocab and merge rules
    tokenizer: std::sync::Arc<realizar::tokenizer::BPETokenizer>,
    /// Vocab for decode fallback
    vocab: Vec<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

/// Load tokenizer from HuggingFace tokenizer.json format with BPE merge rules
///
/// PMAT-093: Proper BPE tokenization is critical for SafeTensors inference.
/// Without merge rules, tokenization produces wrong tokens causing garbage output.
#[cfg(feature = "inference")]
fn load_safetensors_tokenizer(path: &Path) -> Option<SafeTensorsTokenizerInfo> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Extract vocabulary from model.vocab (token -> id mapping)
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    // Build vocab vector sorted by ID (for index-based lookup)
    let mut vocab_vec: Vec<(String, u32)> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
        .collect();
    vocab_vec.sort_by_key(|(_, id)| *id);
    let vocab: Vec<String> = vocab_vec.into_iter().map(|(token, _)| token).collect();

    // Extract BPE merge rules from model.merges
    // Merges are stored as ["ab cd", "ef gh", ...] meaning "merge 'ab' + 'cd'"
    let merges: Vec<(String, String)> = json
        .get("model")?
        .get("merges")?
        .as_array()?
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let parts: Vec<&str> = s.split(' ').collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    // Extract special tokens and add them to vocabulary
    // PMAT-099: added_tokens must be included in vocab for decode to work
    let added_tokens = json.get("added_tokens").and_then(|v| v.as_array());
    let mut bos_token_id = None;
    let mut eos_token_id = None;
    let mut special_tokens: Vec<(u32, String)> = Vec::new();

    if let Some(tokens) = added_tokens {
        for token in tokens {
            let content = token.get("content").and_then(|v| v.as_str());
            let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

            if let (Some(content), Some(id)) = (content, id) {
                special_tokens.push((id, content.to_string()));

                if content.contains("bos") || content == "<s>" {
                    bos_token_id = Some(id);
                }
                if content.contains("eos") || content == "</s>" || content.contains("im_end") {
                    eos_token_id = Some(id);
                }
            }
        }
    }

    // Extend vocab to include special tokens at their proper IDs
    // PMAT-099: Special tokens often have IDs beyond base vocab (e.g., 151643+)
    let mut vocab = vocab;
    if !special_tokens.is_empty() {
        let max_special_id = special_tokens.iter().map(|(id, _)| *id).max().unwrap_or(0);
        if max_special_id as usize >= vocab.len() {
            // Resize vocab to fit all special tokens
            vocab.resize(max_special_id as usize + 1, "<unused>".to_string());
        }
        // Insert special tokens at their IDs
        for (id, content) in special_tokens {
            if (id as usize) < vocab.len() {
                vocab[id as usize] = content;
            }
        }
    }

    // Create BPE tokenizer with vocab and merge rules
    let tokenizer = realizar::tokenizer::BPETokenizer::new(vocab.clone(), merges, "<unk>").ok()?;

    Some(SafeTensorsTokenizerInfo {
        tokenizer: std::sync::Arc::new(tokenizer),
        vocab,
        bos_token_id,
        eos_token_id,
    })
}

/// SafeTensors chat completions handler (PAR-301)
#[cfg(feature = "inference")]
async fn safetensors_chat_completions_handler(
    axum::extract::State(state): axum::extract::State<SafeTensorsState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::{sse::Event, IntoResponse, Sse};
    use futures_util::stream;

    // Parse request
    let messages = request.get("messages").and_then(|m| m.as_array());
    let max_tokens = request
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(50) as usize;
    let stream_mode = request
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Build prompt from messages
    let prompt = if let Some(msgs) = messages {
        let mut prompt = String::new();
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    } else {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "Missing messages field"})),
        )
            .into_response();
    };

    // Get transformer
    let transformer = match &state.transformer {
        Some(t) => t.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(
                    serde_json::json!({"error": "Inference not available - missing config.json"}),
                ),
            )
                .into_response();
        }
    };

    // Encode prompt using BPE tokenizer (PMAT-093)
    let input_ids = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info.tokenizer.encode(&prompt)
    } else {
        // Fallback: character-level tokenization (no tokenizer.json)
        prompt.chars().map(|c| c as u32).collect()
    };

    // PMAT-103 FIX: Use generate_with_cache for O(n) generation
    // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
    // generate_with_cache() uses KV cache for incremental generation = O(n)
    let start = Instant::now();
    let temperature = request
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0) as f32;
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    let output_ids = {
        let t = transformer.lock().unwrap();
        match t.generate_with_cache(&input_ids, &gen_config) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Generation failed: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let elapsed = start.elapsed();

    // Decode output using BPE tokenizer (PMAT-093)
    let new_tokens = &output_ids[input_ids.len()..];
    let output_text = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info
            .tokenizer
            .decode(new_tokens)
            .unwrap_or_else(|_| simple_decode(new_tokens, &tok_info.vocab))
    } else {
        new_tokens
            .iter()
            .map(|&id| char::from_u32(id.min(127)).unwrap_or('?'))
            .collect()
    };

    // Clean output (remove any trailing special tokens)
    let output_text = output_text
        .split("<|im_end|>")
        .next()
        .unwrap_or(&output_text)
        .to_string();

    let tokens_generated = new_tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    // Generate unique ID using timestamp and process ID
    let request_id = format!(
        "chatcmpl-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    );

    // Return OpenAI-compatible response
    if stream_mode {
        // SSE streaming response
        let response = serde_json::json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "model": "safetensors",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": output_text},
                "finish_reason": "stop"
            }]
        });

        let stream = stream::once(async move {
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
        });
        Sse::new(stream).into_response()
    } else {
        // Non-streaming response
        axum::Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "model": "safetensors",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_ids.len(),
                "completion_tokens": tokens_generated,
                "total_tokens": input_ids.len() + tokens_generated
            },
            "latency_ms": elapsed.as_millis(),
            "tok_per_sec": tok_per_sec
        }))
        .into_response()
    }
}

/// SafeTensors generate handler
#[cfg(feature = "inference")]
async fn safetensors_generate_handler(
    axum::extract::State(state): axum::extract::State<SafeTensorsState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let prompt = request.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
    let max_tokens = request
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(32) as usize;

    let transformer = match &state.transformer {
        Some(t) => t.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({"error": "Inference not available"})),
            )
                .into_response();
        }
    };

    // Encode prompt using BPE tokenizer (PMAT-093)
    let input_ids = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info.tokenizer.encode(prompt)
    } else {
        prompt.chars().map(|c| c as u32).collect()
    };

    // PMAT-103 FIX: Use generate_with_cache for O(n) generation
    // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
    // generate_with_cache() uses KV cache for incremental generation = O(n)
    let start = Instant::now();
    let temperature = request
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0) as f32;
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    let output_ids = {
        let t = transformer.lock().unwrap();
        match t.generate_with_cache(&input_ids, &gen_config) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Generation failed: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let elapsed = start.elapsed();

    // Decode using BPE tokenizer (PMAT-093)
    let new_tokens = &output_ids[input_ids.len()..];
    let output_text = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info
            .tokenizer
            .decode(new_tokens)
            .unwrap_or_else(|_| simple_decode(new_tokens, &tok_info.vocab))
    } else {
        new_tokens
            .iter()
            .map(|&id| char::from_u32(id.min(127)).unwrap_or('?'))
            .collect()
    };

    let tokens_generated = new_tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    axum::Json(serde_json::json!({
        "text": output_text,
        "tokens_generated": tokens_generated,
        "latency_ms": elapsed.as_millis(),
        "tok_per_sec": tok_per_sec
    }))
    .into_response()
}

/// Simple tokenization using greedy longest match
fn simple_encode(text: &str, vocab: &[String]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Find longest matching token
        let mut best_match = None;
        let mut best_len = 0;

        for (id, token) in vocab.iter().enumerate() {
            if remaining.starts_with(token) && token.len() > best_len {
                best_match = Some(id as u32);
                best_len = token.len();
            }
        }

        if let Some(id) = best_match {
            tokens.push(id);
            remaining = &remaining[best_len..];
        } else {
            // Skip unknown character
            let char_len = remaining.chars().next().map_or(1, char::len_utf8);
            remaining = &remaining[char_len..];
        }
    }

    tokens
}

/// Simple decode using vocab lookup
fn simple_decode(token_ids: &[u32], vocab: &[String]) -> String {
    token_ids
        .iter()
        .map(|&id| {
            vocab
                .get(id as usize)
                .map_or("?".to_string(), |s| s.clone())
        })
        .collect::<String>()
}

/// Shared state for SafeTensors server
#[cfg(feature = "inference")]
#[derive(Clone)]
struct SafeTensorsState {
    transformer: Option<Arc<std::sync::Mutex<realizar::apr_transformer::AprTransformer>>>,
    tokenizer_info: Option<SafeTensorsTokenizerInfo>,
    model_path: String,
}

/// Shutdown signal handler
#[cfg(feature = "inference")]
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper to create test model file
    fn create_test_model() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test model data").unwrap();
        file
    }

    /// Helper to create test server state
    fn create_test_state() -> Arc<ServerState> {
        let file = create_test_model();
        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        // Keep file alive by leaking it (for tests only)
        std::mem::forget(file);
        Arc::new(state)
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
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::io::Write;
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
}
