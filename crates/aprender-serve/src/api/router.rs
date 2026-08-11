
/// APR explanation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainResponse {
    /// Request ID for audit trail
    pub request_id: String,
    /// Model ID used
    pub model: String,
    /// Prediction (same as /v1/predict)
    pub prediction: serde_json::Value,
    /// Confidence score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// SHAP explanation
    pub explanation: ShapExplanation,
    /// Human-readable summary
    pub summary: String,
    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Audit record retrieval response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResponse {
    /// The audit record
    pub record: AuditRecord,
}

/// Router configuration options (GH-148: wire openai_api flag)
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Enable OpenAI-compatible API at /v1/* (default: true)
    pub openai_api: bool,
    /// Send permissive CORS headers (default: true).
    ///
    /// `apr serve run --no-cors` sets this to `false`, which removes the
    /// `CorsLayer` entirely so no `access-control-*` header is emitted.
    pub cors: bool,
    /// Expose the Prometheus/dispatch metrics endpoints (default: true).
    ///
    /// `apr serve run --no-metrics` sets this to `false`, which unregisters
    /// `/metrics`, `/metrics/dispatch` and `/metrics/dispatch/reset` so they
    /// return the 404 fallback instead of serving telemetry.
    pub metrics: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            openai_api: true,
            cors: true,
            metrics: true,
        }
    }
}

/// Routes mounted unconditionally by [`create_router_with_config`], as (method, path).
///
/// aprender#2376(12): the 404 body told clients "See /health for available
/// endpoints", and `/health` returns five status fields and no route list — so
/// following the instruction in the error message yielded nothing. The 404 now
/// serves this list itself, and `test_advertised_routes_are_all_mounted` probes
/// every entry so the list cannot drift into a second false advertisement.
const NATIVE_ROUTES: &[(&str, &str)] = &[
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
    ("GET", "/ready"),
    ("GET", "/metrics"),
    ("GET", "/metrics/dispatch"),
    ("POST", "/metrics/dispatch/reset"),
    ("GET", "/models"),
    ("POST", "/tokenize"),
    ("POST", "/generate"),
    ("POST", "/batch/tokenize"),
    ("POST", "/batch/generate"),
    ("POST", "/stream/generate"),
    ("POST", "/realize/generate"),
    ("POST", "/realize/batch"),
    ("POST", "/realize/embed"),
    ("GET", "/realize/model"),
    ("POST", "/realize/reload"),
];

/// Routes mounted only when `RouterConfig::openai_api` is set (the default).
const OPENAI_ROUTES: &[(&str, &str)] = &[
    ("GET", "/v1/models"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/chat/completions/stream"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/predict"),
    ("POST", "/v1/explain"),
    ("GET", "/v1/audit/:request_id"),
    ("POST", "/v1/gpu/warmup"),
    ("GET", "/v1/gpu/status"),
    ("POST", "/v1/batch/completions"),
    ("GET", "/v1/metrics"),
    ("POST", "/api/chat"),
    ("POST", "/api/generate"),
    ("POST", "/api/embeddings"),
];

/// Routes mounted only in CUDA builds (realizr#191).
#[cfg(feature = "cuda")]
const CUDA_ROUTES: &[(&str, &str)] = &[("POST", "/v1/logprobs"), ("POST", "/v1/perplexity")];

/// The routes this router mounts, as `"METHOD /path"` strings for the 404 body.
fn route_index(openai_api: bool) -> Vec<String> {
    let fmt = |(method, path): &(&str, &str)| format!("{method} {path}");
    let mut routes: Vec<String> = NATIVE_ROUTES.iter().map(fmt).collect();
    if openai_api {
        routes.extend(OPENAI_ROUTES.iter().map(fmt));
    }
    #[cfg(feature = "cuda")]
    routes.extend(CUDA_ROUTES.iter().map(fmt));
    routes
}

/// Create the API router with default options (OpenAI API enabled)
///
/// # Arguments
///
/// * `state` - Application state with model and tokenizer
pub fn create_router(state: AppState) -> Router {
    create_router_with_config(state, RouterConfig::default())
}

/// Create the API router with explicit configuration (GH-148)
///
/// # Arguments
///
/// * `state` - Application state with model and tokenizer
/// * `config` - Router configuration (controls which route groups are enabled)
pub fn create_router_with_config(state: AppState, config: RouterConfig) -> Router {
    // aprender#2376(8): `GET /` and `GET /ready` are registered by the two OTHER
    // routers in this repo (apr-cli `commands/serve/routes.rs`, `serve_run_model.rs`)
    // and 404'd here, so which of three route surfaces you got depended on the
    // format of the file you passed to `apr serve run`. `/` now answers with the
    // route table this router actually mounted — the one thing a client needs to
    // discover the surface it landed on — and `/ready` is the conventional
    // readiness path, an alias of `/health/ready`.
    let index_routes = route_index(config.openai_api);
    let mut router = Router::new()
        .route(
            "/",
            get(move || {
                let routes = index_routes.clone();
                async move {
                    Json(serde_json::json!({
                        "service": "apr serve",
                        "version": env!("CARGO_PKG_VERSION"),
                        "routes": routes,
                    }))
                }
            }),
        )
        // Health and metrics (CRUX-C-34: /health, /health/live, /health/ready)
        .route("/health", get(health_handler))
        .route("/health/live", get(health_live_handler))
        .route("/health/ready", get(health_ready_handler))
        // `/ready` is the conventional readiness path and an alias of
        // `/health/ready`. It is listed in this router's advertised route table
        // (NATIVE_ROUTES above), so leaving it unmounted would advertise a route
        // that 404s — the index would lie about the surface it is serving.
        // Verified by test_root_and_ready_are_mounted, whose failure body lists
        // "GET /ready" among the available routes while 404ing on it.
        .route("/ready", get(health_ready_handler))
        // Native Realizar API (legacy paths)
        .route("/models", get(models_handler))
        .route("/tokenize", post(tokenize_handler))
        .route("/generate", post(generate_handler))
        .route("/batch/tokenize", post(batch_tokenize_handler))
        .route("/batch/generate", post(batch_generate_handler))
        .route("/stream/generate", post(stream_generate_handler))
        // Native Realizar API (spec §5.2 /realize/* paths)
        .route("/realize/generate", post(stream_generate_handler))
        .route("/realize/batch", post(batch_generate_handler))
        .route("/realize/embed", post(realize_embed_handler))
        .route("/realize/model", get(realize_model_handler))
        .route("/realize/reload", post(realize_reload_handler));

    // Metrics endpoints conditionally enabled: `apr serve run --no-metrics`
    // must actually withhold telemetry, not just hide the banner line.
    if config.metrics {
        router = router
            .route("/metrics", get(metrics_handler))
            .route("/metrics/dispatch", get(dispatch_metrics_handler))
            .route("/metrics/dispatch/reset", post(dispatch_reset_handler));
    }

    // GH-148: OpenAI-compatible API conditionally enabled
    if config.openai_api {
        router = router
            // OpenAI-compatible API (v1) - spec §5.1
            .route("/v1/models", get(openai_models_handler))
            .route("/v1/completions", post(openai_completions_handler))
            .route(
                "/v1/chat/completions",
                post(openai_chat_completions_handler),
            )
            .route(
                "/v1/chat/completions/stream",
                post(openai_chat_completions_stream_handler),
            )
            .route("/v1/embeddings", post(openai_embeddings_handler))
            // APR-specific API (spec §15.1)
            .route("/v1/predict", post(apr_predict_handler))
            .route("/v1/explain", post(apr_explain_handler))
            .route("/v1/audit/:request_id", get(apr_audit_handler))
            // GPU batch inference API (PARITY-022)
            .route("/v1/gpu/warmup", post(gpu_warmup_handler))
            .route("/v1/gpu/status", get(gpu_status_handler))
            .route("/v1/batch/completions", post(gpu_batch_completions_handler))
            // TUI monitoring API (PARITY-107)
            .route("/v1/metrics", get(server_metrics_handler))
            // PMAT-923: Ollama-native HTTP API (/api/* prefix) — makes `apr serve`
            // a drop-in Ollama HTTP replacement. Both delegate to the OpenAI chat
            // generation path. Discharges OBLIG-OLLAMA-API-CHAT-GENERATE-ROUTED.
            .route("/api/chat", post(ollama_chat_handler))
            .route("/api/generate", post(ollama_generate_handler))
            // Model discovery. Ollama clients call /api/tags BEFORE issuing any
            // chat request and /api/show to probe capabilities; without them the
            // "drop-in Ollama replacement" claim above is unreachable in practice.
            .route("/api/tags", get(ollama_tags_handler))
            .route("/api/show", post(ollama_show_handler))
            .route("/api/version", get(ollama_version_handler))
            // aprender#2396(2): every Ollama embedding client posts here; the route
            // did not exist, so they got the 404 fallback.
            .route("/api/embeddings", post(ollama_embeddings_handler));
    }

    // realizr#191: Logprobs + perplexity endpoints (CUDA only, F-QUALITY-01)
    #[cfg(feature = "cuda")]
    {
        router = router
            .route("/v1/logprobs", post(logprobs_handler))
            .route("/v1/perplexity", post(perplexity_handler));
    }

    // GH-672: Return JSON error body for unmatched routes (not empty 404)
    // aprender#2376(12): serve the route list here instead of pointing clients at
    // /health, which does not have one.
    let routes = route_index(config.openai_api);
    router = router.fallback(move || {
        let routes = routes.clone();
        async move {
            (
                axum::http::StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": "not_found",
                    "message": "Route not found. Available routes are listed in `routes`.",
                    "routes": routes,
                })),
            )
        }
    });

    // GH-649: Sanitize axum deserialization errors to avoid leaking internals to clients.
    // Axum returns 422 with raw serde error details by default; replace with a generic message.
    router = router.layer(axum::middleware::from_fn(sanitize_json_rejection));

    // GH-671: CORS support — allow cross-origin requests from browser-based clients.
    // Conditional: `apr serve run --no-cors` must emit no `access-control-*` header.
    if config.cors {
        router = router.layer(tower_http::cors::CorsLayer::permissive());
    }

    router.with_state(state)
}

/// The client-safe replacement body for an error response that is not already JSON.
///
/// Returns `None` for statuses we have no better wording for than the status line
/// itself would give — those still get an envelope, just a generic message.
fn sanitized_error_message(status: StatusCode) -> String {
    match status {
        // Axum `JsonSyntaxError`: the default body is
        // "Failed to parse the request body as JSON: key must be a string at line 1
        // column 2" — a serde parser position, which is exactly what the GH-649
        // sanitizer was added to stop leaking.
        StatusCode::BAD_REQUEST => {
            "Invalid request body. Expected a JSON object matching this endpoint's schema."
                .to_string()
        },
        StatusCode::UNSUPPORTED_MEDIA_TYPE => {
            "Expected request with Content-Type: application/json.".to_string()
        },
        StatusCode::PAYLOAD_TOO_LARGE => "Request body is too large.".to_string(),
        StatusCode::METHOD_NOT_ALLOWED => {
            "Method not allowed for this route. See the `allow` header.".to_string()
        },
        StatusCode::UNPROCESSABLE_ENTITY => {
            "Invalid request body. Check that the JSON structure matches the expected schema."
                .to_string()
        },
        other => format!(
            "Request failed with status {} {}.",
            other.as_u16(),
            other.canonical_reason().unwrap_or("Error")
        ),
    }
}

/// GH-649 + aprender#2376(7): give every failure the same `{"error": "..."}`
/// envelope, and never let a parser's internals reach a client.
///
/// The original sanitizer intercepted `422` only, so axum's own `400`
/// (`JsonSyntaxError`) and `415` (`MissingJsonContentType`) rejections sailed
/// through as `text/plain` — the 400 still quoting the serde error position that
/// the 422 branch existed to hide. A client could not parse failures uniformly:
/// most were JSON, two were bare text.
///
/// Responses that already carry `content-type: application/json` are passed
/// through untouched, so every handler-authored message survives verbatim. The
/// original headers are preserved as well — notably `allow` on a 405, which a
/// rebuilt response would have dropped.
async fn sanitize_json_rejection(
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let response = next.run(request).await;

    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return response;
    }

    let already_json = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("application/json"));
    if already_json {
        return response;
    }

    let body = serde_json::to_vec(&ErrorResponse {
        error: sanitized_error_message(status),
    })
    .unwrap_or_else(|_| br#"{"error":"Request failed."}"#.to_vec());

    // Keep the original head (status, `allow`, `retry-after`, …) and swap only the
    // representation — rebuilding the response from scratch would silently drop
    // headers a client depends on.
    let (mut parts, _discarded) = response.into_parts();
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    parts.headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    axum::response::Response::from_parts(parts, axum::body::Body::from(body))
}

/// Process-wide server start instant.
///
/// Initialised lazily on the first `/health*` hit. `Instant` is
/// monotonic in `std` — see `std::time::Instant` docs — which
/// discharges FALSIFY-CRUX-C-34-003 (monotonic `uptime_sec`).
fn server_uptime_sec() -> f64 {
    static SERVER_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    SERVER_START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// Test-only hook: force the health handler to report `status = "loading"`.
///
/// Wired by `APR_TEST_FORCE_LOADING=1`. Present in all builds so
/// FALSIFY-CRUX-C-34-005 can drive the loading/503 branch without a
/// separate test-only feature.
fn force_loading() -> bool {
    std::env::var("APR_TEST_FORCE_LOADING").is_ok_and(|v| v == "1")
}

/// Build a `HealthResponse` consistent with the CRUX-C-34 contract.
///
/// Caller picks the HTTP status via `health_status_code(&response)`.
fn build_health_response(state: &AppState) -> HealthResponse {
    // BUG-HEALTH-001: all GPU dispatch paths must register as "gpu".
    let mut compute_mode = "cpu";
    #[cfg(feature = "gpu")]
    if state.has_gpu_model() || state.has_cached_model() {
        compute_mode = "gpu";
    }
    #[cfg(feature = "cuda")]
    if state.has_cuda_model() {
        compute_mode = "gpu";
    }

    let model_loaded = state.model_loaded();
    // Contract §health_response_schema:
    //   status == "ok"       ⇒ ready to serve (HTTP 200)
    //   status == "loading"  ⇒ model not yet resident (HTTP 503)
    //   status == "degraded" ⇒ reserved for partial failure modes
    let status = if force_loading() || !model_loaded {
        "loading"
    } else {
        "ok"
    };

    HealthResponse {
        status: status.to_string(),
        version: crate::VERSION.to_string(),
        compute_mode: compute_mode.to_string(),
        model_loaded,
        uptime_sec: server_uptime_sec(),
    }
}

/// HTTP status derived from the body's `status` field.
///
/// Contract §health_response_schema: 200 iff `status == "ok"`; 503 for
/// every non-`ok` status (loading, degraded).
fn health_status_code(body: &HealthResponse) -> StatusCode {
    if body.status == "ok" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// `GET /health` — vLLM / llama.cpp-parity liveness probe.
///
/// Discharges FALSIFY-CRUX-C-34-001/002/003.
async fn health_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health");
    }
    let body = build_health_response(&state);
    let code = health_status_code(&body);
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health -> {} status={}", code, body.status);
    }
    (code, Json(body))
}

/// `GET /health/live` — k8s liveness probe.
///
/// Always returns 200 once the HTTP port is bound (CRUX-C-34
/// §liveness_vs_readiness). Body mirrors `/health` for debuggability.
async fn health_live_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health/live");
    }
    (StatusCode::OK, Json(build_health_response(&state)))
}

/// `GET /health/ready` — k8s readiness probe.
///
/// 200 iff `status == "ok"` AND `model_loaded == true`; 503 otherwise.
/// Discharges FALSIFY-CRUX-C-34-004.
async fn health_ready_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health/ready");
    }
    let body = build_health_response(&state);
    let code = if body.status == "ok" && body.model_loaded {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body))
}

/// Metrics handler - returns Prometheus-formatted metrics
async fn metrics_handler(State(state): State<AppState>) -> String {
    state.metrics.to_prometheus()
}

/// Response for dispatch metrics endpoint (IMP-127)
#[derive(Debug, Clone, serde::Serialize)]
pub struct DispatchMetricsResponse {
    /// Number of CPU dispatch decisions
    pub cpu_dispatches: usize,
    /// Number of GPU dispatch decisions
    pub gpu_dispatches: usize,
    /// Total dispatch decisions
    pub total_dispatches: usize,
    /// Ratio of GPU dispatches (0.0 to 1.0)
    pub gpu_ratio: f64,
    /// CPU latency p50 (median) in microseconds (IMP-131)
    pub cpu_latency_p50_us: f64,
    /// CPU latency p95 in microseconds (IMP-131)
    pub cpu_latency_p95_us: f64,
    /// CPU latency p99 in microseconds (IMP-131)
    pub cpu_latency_p99_us: f64,
    /// GPU latency p50 (median) in microseconds (IMP-131)
    pub gpu_latency_p50_us: f64,
    /// GPU latency p95 in microseconds (IMP-131)
    pub gpu_latency_p95_us: f64,
    /// GPU latency p99 in microseconds (IMP-131)
    pub gpu_latency_p99_us: f64,
    /// CPU latency mean in microseconds (IMP-133)
    pub cpu_latency_mean_us: f64,
    /// GPU latency mean in microseconds (IMP-133)
    pub gpu_latency_mean_us: f64,
    /// CPU latency minimum in microseconds (IMP-134)
    pub cpu_latency_min_us: u64,
    /// CPU latency maximum in microseconds (IMP-134)
    pub cpu_latency_max_us: u64,
    /// GPU latency minimum in microseconds (IMP-134)
    pub gpu_latency_min_us: u64,
    /// GPU latency maximum in microseconds (IMP-134)
    pub gpu_latency_max_us: u64,
    /// CPU latency variance in microseconds squared (IMP-135)
    pub cpu_latency_variance_us: f64,
    /// CPU latency standard deviation in microseconds (IMP-135)
    pub cpu_latency_stddev_us: f64,
    /// GPU latency variance in microseconds squared (IMP-135)
    pub gpu_latency_variance_us: f64,
    /// GPU latency standard deviation in microseconds (IMP-135)
    pub gpu_latency_stddev_us: f64,
    /// Human-readable bucket boundary ranges (IMP-136)
    pub bucket_boundaries_us: Vec<String>,
    /// CPU latency histogram bucket counts (IMP-136)
    pub cpu_latency_bucket_counts: Vec<usize>,
    /// GPU latency histogram bucket counts (IMP-136)
    pub gpu_latency_bucket_counts: Vec<usize>,
    /// Throughput in requests per second (IMP-140)
    pub throughput_rps: f64,
    /// Elapsed time in seconds since start/reset (IMP-140)
    pub elapsed_seconds: f64,
}

/// Server metrics response for TUI monitoring (PARITY-107)
/// Used by realizar-monitor to display real-time server status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerMetricsResponse {
    /// Current throughput in tokens per second
    pub throughput_tok_per_sec: f64,
    /// P50 (median) latency in milliseconds
    pub latency_p50_ms: f64,
    /// P95 latency in milliseconds
    pub latency_p95_ms: f64,
    /// P99 latency in milliseconds
    pub latency_p99_ms: f64,
    /// GPU memory currently used in bytes
    pub gpu_memory_used_bytes: u64,
    /// Total GPU memory available in bytes
    pub gpu_memory_total_bytes: u64,
    /// GPU utilization as percentage (0-100)
    pub gpu_utilization_percent: u32,
    /// Whether CUDA path is active
    pub cuda_path_active: bool,
    /// Current batch size
    pub batch_size: usize,
    /// Current queue depth
    pub queue_depth: usize,
    /// Total tokens generated since start
    pub total_tokens: u64,
    /// Total requests processed since start
    pub total_requests: u64,
    /// Server uptime in seconds
    pub uptime_secs: u64,
    /// Model name being served
    pub model_name: String,
}

/// Query parameters for dispatch metrics endpoint (IMP-128)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DispatchMetricsQuery {
    /// Output format: "json" (default) or "prometheus"
    #[serde(default)]
    pub format: Option<String>,
}

/// Response for dispatch metrics reset endpoint (IMP-138)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DispatchResetResponse {
    /// Whether the reset was successful
    pub success: bool,
    /// Human-readable message
    pub message: String,
}

/// Dispatch metrics reset handler - resets all dispatch statistics (IMP-138)
/// POST /v1/dispatch/reset
#[cfg(feature = "gpu")]
async fn dispatch_reset_handler(State(state): State<AppState>) -> axum::response::Response {
    use axum::response::IntoResponse;

    if let Some(metrics) = state.dispatch_metrics() {
        metrics.reset();
        Json(DispatchResetResponse {
            success: true,
            message: "Metrics reset successfully".to_string(),
        })
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Dispatch metrics not available. No GPU model configured.".to_string(),
            }),
        )
            .into_response()
    }
}

/// Dispatch metrics reset handler stub for non-GPU builds (IMP-138)
#[cfg(not(feature = "gpu"))]
async fn dispatch_reset_handler(State(_state): State<AppState>) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "Dispatch metrics not available. GPU feature not enabled.".to_string(),
        }),
    )
        .into_response()
}

/// Server metrics handler for TUI monitoring (PARITY-107)
/// GET /v1/metrics - Returns JSON metrics for realizar-monitor
#[cfg(feature = "gpu")]
async fn server_metrics_handler(State(state): State<AppState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();

    // Get latency percentiles from dispatch metrics (in microseconds, convert to ms)
    let (latency_p50_ms, latency_p95_ms, latency_p99_ms, gpu_dispatches, cuda_path_active) =
        if let Some(dispatch) = state.dispatch_metrics() {
            // Use GPU latency if available, otherwise CPU latency
            let gpu_p50 = dispatch.gpu_latency_p50_us();
            let gpu_p95 = dispatch.gpu_latency_p95_us();
            let gpu_p99 = dispatch.gpu_latency_p99_us();
            let gpu_count = dispatch.gpu_dispatches();

            if gpu_count > 0 {
                (
                    gpu_p50 / 1000.0,
                    gpu_p95 / 1000.0,
                    gpu_p99 / 1000.0,
                    gpu_count,
                    true,
                )
            } else {
                let cpu_p50 = dispatch.cpu_latency_p50_us();
                let cpu_p95 = dispatch.cpu_latency_p95_us();
                let cpu_p99 = dispatch.cpu_latency_p99_us();
                (
                    cpu_p50 / 1000.0,
                    cpu_p95 / 1000.0,
                    cpu_p99 / 1000.0,
                    0,
                    false,
                )
            }
        } else {
            (0.0, 0.0, 0.0, 0, false)
        };

    // Get GPU memory from cached model
    let (gpu_memory_used_bytes, gpu_memory_total_bytes): (u64, u64) =
        if let Some(model) = state.cached_model() {
            let used = model.gpu_cache_memory() as u64;
            // RTX 4090 has 24GB VRAM
            let total = 24 * 1024 * 1024 * 1024u64;
            (used, total)
        } else {
            (0, 0)
        };

    // Estimate GPU utilization from dispatch ratio
    let gpu_utilization_percent = if let Some(dispatch) = state.dispatch_metrics() {
        let total = dispatch.total_dispatches();
        if total > 0 {
            ((gpu_dispatches as f64 / total as f64) * 100.0) as u32
        } else {
            0
        }
    } else {
        0
    };

    // Get batch configuration
    let (batch_size, queue_depth) = if let Some(config) = state.batch_config() {
        (config.optimal_batch, config.queue_size)
    } else {
        (1, 0)
    };

    // Model name from cached model or default
    let model_name = if state.cached_model().is_some() {
        "phi-2-q4_k_m".to_string()
    } else {
        "N/A".to_string()
    };

    Json(ServerMetricsResponse {
        throughput_tok_per_sec: snapshot.tokens_per_sec,
        latency_p50_ms,
        latency_p95_ms,
        latency_p99_ms,
        gpu_memory_used_bytes,
        gpu_memory_total_bytes,
        gpu_utilization_percent,
        cuda_path_active,
        batch_size,
        queue_depth,
        total_tokens: snapshot.total_tokens as u64,
        total_requests: snapshot.total_requests as u64,
        uptime_secs: snapshot.uptime_secs,
        model_name,
    })
}

/// Server metrics handler stub for non-GPU builds (PARITY-107)
#[cfg(not(feature = "gpu"))]
async fn server_metrics_handler(State(state): State<AppState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();

    Json(ServerMetricsResponse {
        throughput_tok_per_sec: snapshot.tokens_per_sec,
        latency_p50_ms: snapshot.avg_latency_ms,
        latency_p95_ms: snapshot.avg_latency_ms * 1.5,
        latency_p99_ms: snapshot.avg_latency_ms * 2.0,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 0,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 1,
        queue_depth: 0,
        total_tokens: snapshot.total_tokens as u64,
        total_requests: snapshot.total_requests as u64,
        uptime_secs: snapshot.uptime_secs,
        model_name: "N/A".to_string(),
    })
}
