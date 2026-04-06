
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
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self { openai_api: true }
    }
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
    let mut router = Router::new()
        // Health and metrics
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/metrics/dispatch", get(dispatch_metrics_handler))
        .route("/metrics/dispatch/reset", post(dispatch_reset_handler))
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
            .route("/v1/metrics", get(server_metrics_handler));
    }

    // realizr#191: Logprobs + perplexity endpoints (CUDA only, F-QUALITY-01)
    #[cfg(feature = "cuda")]
    {
        router = router
            .route("/v1/logprobs", post(logprobs_handler))
            .route("/v1/perplexity", post(perplexity_handler));
    }

    // GH-672: Return JSON error body for unmatched routes (not empty 404)
    router = router.fallback(|| async {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "not_found",
                "message": "Route not found. See /health for available endpoints."
            })),
        )
    });

    // GH-649: Sanitize axum deserialization errors to avoid leaking internals to clients.
    // Axum returns 422 with raw serde error details by default; replace with a generic message.
    router = router.layer(axum::middleware::from_fn(sanitize_json_rejection));

    // GH-671: CORS support — allow cross-origin requests from browser-based clients
    let cors = tower_http::cors::CorsLayer::permissive();
    router.layer(cors).with_state(state)
}

/// GH-649: Middleware that intercepts axum's 422 JSON rejection responses and replaces
/// the body with a generic error message, preventing internal serde error details from
/// leaking to API clients.
async fn sanitize_json_rejection(
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let response = next.run(request).await;

    // Axum returns 422 Unprocessable Entity for JSON deserialization failures.
    // Replace the body to avoid leaking serde error internals.
    if response.status() == StatusCode::UNPROCESSABLE_ENTITY {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "Invalid request body. Check that the JSON structure matches the expected schema.".to_string(),
            }),
        )
            .into_response();
    }

    response
}

/// Health check handler
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    // GH-152: Verbose request logging
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health");
    }

    // Determine compute mode based on what's available
    // BUG-HEALTH-001: Must check all GPU dispatch paths.
    // - has_gpu_model(): legacy wgpu path
    // - cached_model: batched GPU inference
    // - has_cuda_model(): PAR-111 CUDA path (stores in AppState.cuda_model)
    let mut compute_mode = "cpu";

    #[cfg(feature = "gpu")]
    if state.has_gpu_model() || state.cached_model.is_some() {
        compute_mode = "gpu";
    }

    #[cfg(feature = "cuda")]
    if state.has_cuda_model() {
        compute_mode = "gpu";
    }

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: crate::VERSION.to_string(),
        compute_mode: compute_mode.to_string(),
    };

    // GH-152: Verbose response logging
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /health -> status={}", response.status);
    }

    Json(response)
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
