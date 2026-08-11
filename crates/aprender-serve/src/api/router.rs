
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

/// One route: the method and path a client calls, and the handler that answers it.
///
/// aprender#2376(8): the mounted surface and the advertised surface used to be two
/// hand-maintained lists that had already drifted apart — `--no-metrics` unmounted
/// `/metrics`, `/metrics/dispatch` and `/metrics/dispatch/reset` while the 404 body
/// kept advertising all three, and `/api/tags`, `/api/show` and `/api/version` were
/// mounted but advertised nowhere. Binding the advertised string to the handler in
/// one tuple makes both halves of that drift unrepresentable: a row cannot enter the
/// list without a handler to mount, and a handler cannot be mounted without its row
/// entering the list.
type Route = (
    &'static str,
    &'static str,
    axum::routing::MethodRouter<AppState>,
);

/// Every route this configuration serves, in the order clients see them advertised.
///
/// This is the single source of truth: [`create_router_with_config`] mounts exactly
/// these and the 404 body advertises exactly these.
fn route_table(config: &RouterConfig) -> Vec<Route> {
    let mut routes: Vec<Route> = vec![
        // Health (CRUX-C-34: /health, /health/live, /health/ready)
        ("GET", "/health", get(health_handler)),
        ("GET", "/health/live", get(health_live_handler)),
        ("GET", "/health/ready", get(health_ready_handler)),
        // Native Realizar API (legacy paths)
        ("GET", "/models", get(models_handler)),
        ("POST", "/tokenize", post(tokenize_handler)),
        ("POST", "/generate", post(generate_handler)),
        ("POST", "/batch/tokenize", post(batch_tokenize_handler)),
        ("POST", "/batch/generate", post(batch_generate_handler)),
        ("POST", "/stream/generate", post(stream_generate_handler)),
        // Native Realizar API (spec §5.2 /realize/* paths)
        ("POST", "/realize/generate", post(stream_generate_handler)),
        ("POST", "/realize/batch", post(batch_generate_handler)),
        ("POST", "/realize/embed", post(realize_embed_handler)),
        ("GET", "/realize/model", get(realize_model_handler)),
        ("POST", "/realize/reload", post(realize_reload_handler)),
    ];

    // Metrics endpoints conditionally enabled: `apr serve run --no-metrics`
    // must actually withhold telemetry, not just hide the banner line.
    if config.metrics {
        routes.extend([
            ("GET", "/metrics", get(metrics_handler)),
            ("GET", "/metrics/dispatch", get(dispatch_metrics_handler)),
            (
                "POST",
                "/metrics/dispatch/reset",
                post(dispatch_reset_handler),
            ),
        ]);
    }

    // GH-148: OpenAI-compatible API conditionally enabled
    if config.openai_api {
        routes.extend([
            // OpenAI-compatible API (v1) - spec §5.1
            ("GET", "/v1/models", get(openai_models_handler)),
            ("POST", "/v1/completions", post(openai_completions_handler)),
            (
                "POST",
                "/v1/chat/completions",
                post(openai_chat_completions_handler),
            ),
            (
                "POST",
                "/v1/chat/completions/stream",
                post(openai_chat_completions_stream_handler),
            ),
            ("POST", "/v1/embeddings", post(openai_embeddings_handler)),
            // APR-specific API (spec §15.1)
            ("POST", "/v1/predict", post(apr_predict_handler)),
            ("POST", "/v1/explain", post(apr_explain_handler)),
            ("GET", "/v1/audit/:request_id", get(apr_audit_handler)),
            // GPU batch inference API (PARITY-022)
            ("POST", "/v1/gpu/warmup", post(gpu_warmup_handler)),
            ("GET", "/v1/gpu/status", get(gpu_status_handler)),
            (
                "POST",
                "/v1/batch/completions",
                post(gpu_batch_completions_handler),
            ),
            // TUI monitoring API (PARITY-107)
            ("GET", "/v1/metrics", get(server_metrics_handler)),
            // PMAT-923: Ollama-native HTTP API (/api/* prefix) — makes `apr serve`
            // a drop-in Ollama HTTP replacement. Both delegate to the OpenAI chat
            // generation path. Discharges OBLIG-OLLAMA-API-CHAT-GENERATE-ROUTED.
            ("POST", "/api/chat", post(ollama_chat_handler)),
            ("POST", "/api/generate", post(ollama_generate_handler)),
            // Model discovery. Ollama clients call /api/tags BEFORE issuing any
            // chat request and /api/show to probe capabilities; without them the
            // "drop-in Ollama replacement" claim above is unreachable in practice.
            ("GET", "/api/tags", get(ollama_tags_handler)),
            ("POST", "/api/show", post(ollama_show_handler)),
            ("GET", "/api/version", get(ollama_version_handler)),
        ]);
    }

    // realizr#191: Logprobs + perplexity endpoints (CUDA only, F-QUALITY-01)
    #[cfg(feature = "cuda")]
    routes.extend([
        ("POST", "/v1/logprobs", post(logprobs_handler)),
        ("POST", "/v1/perplexity", post(perplexity_handler)),
    ]);

    routes
}

/// The routes a server built with `config` serves, as `"METHOD /path"` strings.
///
/// Callers that advertise the surface — the 404 body, the CLI startup banner —
/// MUST read it from here rather than restating it, so that what is printed is
/// what is mounted.
pub fn advertised_routes(config: &RouterConfig) -> Vec<String> {
    route_table(config)
        .iter()
        .map(|(method, path, _)| format!("{method} {path}"))
        .collect()
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
    // One table, two consumers: the mount loop below and the 404 body. Neither can
    // name a route the other does not (aprender#2376(8)).
    let table = route_table(&config);
    let routes: Vec<String> = table
        .iter()
        .map(|(method, path, _)| format!("{method} {path}"))
        .collect();

    let mut router = Router::new();
    for (_, path, handler) in table {
        router = router.route(path, handler);
    }

    // GH-672: Return JSON error body for unmatched routes (not empty 404)
    // aprender#2376(12): serve the route list here instead of pointing clients at
    // /health, which does not have one.
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

    // GH-649 / aprender#2376(7): every error leaving this server is a JSON envelope
    // that says nothing about our internals.
    router = router.layer(axum::middleware::from_fn(envelope_error_body));

    // GH-671: CORS support — allow cross-origin requests from browser-based clients.
    // Conditional: `apr serve run --no-cors` must emit no `access-control-*` header.
    if config.cors {
        router = router.layer(tower_http::cors::CorsLayer::permissive());
    }

    router.with_state(state)
}

/// What a client is told about a rejection axum produced before any handler ran.
///
/// Deliberately says nothing an attacker or a confused user could not have derived
/// from the request they just sent. Axum's own rejection text does the opposite: it
/// echoes the serde parser's cursor ("key must be a string at line 1 column 2") and,
/// for typed rejections, the field names of our internal request structs.
fn rejection_message(status: StatusCode) -> String {
    match status {
        StatusCode::BAD_REQUEST => {
            "Malformed request body: expected a JSON object.".to_string()
        }
        StatusCode::UNSUPPORTED_MEDIA_TYPE => {
            "Unsupported Content-Type: this endpoint expects `application/json`.".to_string()
        }
        StatusCode::UNPROCESSABLE_ENTITY => {
            "Invalid request body. Check that the JSON structure matches the expected schema."
                .to_string()
        }
        StatusCode::PAYLOAD_TOO_LARGE => "Request body too large.".to_string(),
        other => format!(
            "Request rejected: {}.",
            other.canonical_reason().unwrap_or("error")
        ),
    }
}

/// GH-649 / aprender#2376(7): the outermost layer over every route and the 404
/// fallback, so that no error response can leave this server with a body that is
/// not a JSON envelope.
///
/// GH-649 sanitised only 422. Axum rejects a malformed body with 400 and a missing
/// `Content-Type` with 415, and both bypassed that check — 0.63.0 answered
/// `POST /generate` with `{not json` in `text/plain` carrying the serde parser
/// position, on every route that takes a body. Keying on the *shape* of the
/// response (non-JSON content type on an error status) rather than on an
/// enumeration of statuses is what makes the leak unrepresentable: a rejection
/// variant added by a future axum release is covered the day it appears.
///
/// A handler's own JSON error — `{"error":"temperature must be >= 0, got -1"}` —
/// is already an envelope and passes through untouched, so this does not cost the
/// caller a real diagnostic.
async fn envelope_error_body(
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let response = next.run(request).await;
    let status = response.status();

    if !status.is_client_error() && !status.is_server_error() {
        return response;
    }

    let is_json = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"));
    if is_json {
        return response;
    }

    // The original body is dropped, never inspected: it is precisely the text we
    // must not forward.
    (
        status,
        Json(ErrorResponse {
            error: rejection_message(status),
        }),
    )
        .into_response()
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
