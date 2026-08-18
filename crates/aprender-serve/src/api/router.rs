
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

/// One row of the route table: the method and path used to *advertise* the
/// route, and the handler used to *mount* it.
///
/// aprender#2376(12): the 404 body told clients "See /health for available
/// endpoints", and `/health` returns five status fields and no route list — so
/// following the instruction in the error message yielded nothing. The 404 now
/// serves this list itself.
///
/// The list and the mount used to be two copies — `const`s of `(method, path)`
/// beside a chain of `.route()` calls. `unadvertised_routes_do_not_answer` was
/// written to catch a route that is mounted but advertised to nobody, and could
/// not: it builds its candidate universe by unioning the ADVERTISED lists, so a
/// route in no list never enters the universe and is never probed. Its doc says
/// the universe is "every route any configuration mounts"; the code says
/// advertises. Those differed by exactly three routes — `/api/tags`,
/// `/api/show` and `/api/version` were mounted, named in no const, and so absent
/// from the 404 body AND from the `apr serve` startup banner, which prints
/// `advertised_routes`. An Ollama client calls `/api/tags` before it will chat.
///
/// Deriving both from this one table is what makes the class impossible rather
/// than merely tested: there is no second copy to disagree with, and the
/// guard's universe is now the table itself.
type Route = (
    &'static str,
    &'static str,
    axum::routing::MethodRouter<AppState>,
);

/// Routes mounted unconditionally. `GET /` is not here — it serves the index
/// built *from* this table, so it cannot be constructed until the table is.
fn native_routes() -> Vec<Route> {
    vec![
        ("GET", "/health", get(health_handler)),
        ("GET", "/health/live", get(health_live_handler)),
        ("GET", "/health/ready", get(health_ready_handler)),
        // `/ready` is the conventional readiness path and an alias of `/health/ready`.
        ("GET", "/ready", get(health_ready_handler)),
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
    ]
}

/// Routes mounted only when `RouterConfig::metrics` is set (the default).
///
/// `apr serve run --no-metrics` must actually withhold telemetry, not just hide
/// the banner line — and must not go on advertising it either.
fn metrics_routes() -> Vec<Route> {
    vec![
        ("GET", "/metrics", get(metrics_handler)),
        ("GET", "/metrics/dispatch", get(dispatch_metrics_handler)),
        (
            "POST",
            "/metrics/dispatch/reset",
            post(dispatch_reset_handler),
        ),
    ]
}

/// Routes mounted only when `RouterConfig::openai_api` is set (the default).
fn openai_routes() -> Vec<Route> {
    vec![
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
        // PMAT-923: Ollama-native HTTP API (/api/* prefix) — makes `apr serve` a
        // drop-in Ollama HTTP replacement. Both delegate to the OpenAI chat
        // generation path. Discharges OBLIG-OLLAMA-API-CHAT-GENERATE-ROUTED.
        ("POST", "/api/chat", post(ollama_chat_handler)),
        ("POST", "/api/generate", post(ollama_generate_handler)),
        // Model discovery. Ollama clients call /api/tags BEFORE issuing any chat
        // request and /api/show to probe capabilities; without them the "drop-in
        // Ollama replacement" claim above is unreachable in practice.
        ("GET", "/api/tags", get(ollama_tags_handler)),
        ("POST", "/api/show", post(ollama_show_handler)),
        ("GET", "/api/version", get(ollama_version_handler)),
        // aprender#2396(2): every Ollama embedding client posts here; the route
        // did not exist, so they got the 404 fallback.
        ("POST", "/api/embeddings", post(ollama_embeddings_handler)),
    ]
}

/// Routes mounted only in CUDA builds (realizr#191, F-QUALITY-01).
#[cfg(feature = "cuda")]
fn cuda_routes() -> Vec<Route> {
    vec![
        ("POST", "/v1/logprobs", post(logprobs_handler)),
        ("POST", "/v1/perplexity", post(perplexity_handler)),
    ]
}

/// Every route this configuration mounts, in advertised order.
fn route_table(config: &RouterConfig) -> Vec<Route> {
    let mut table = native_routes();
    if config.metrics {
        table.extend(metrics_routes());
    }
    if config.openai_api {
        table.extend(openai_routes());
    }
    #[cfg(feature = "cuda")]
    table.extend(cuda_routes());
    table
}

/// The routes a server built with `config` serves, as `"METHOD /path"` strings.
///
/// Callers that advertise the surface — the 404 body, the CLI startup banner —
/// MUST read it from here rather than restating it, so that what is printed is
/// what is mounted. `apr serve run` printed a hand-written list of 11 of the 31
/// mounted routes, from a point in the program before the model format was even
/// known; it named `/generate` for `.apr` models, where the route is not mounted,
/// and `/v1/predict` for GGUF models, where it can only answer 503.
///
/// Derived from `route_table`, the same table `create_router_with_config` mounts,
/// so advertising a route and mounting it are one act.
pub fn advertised_routes(config: &RouterConfig) -> Vec<String> {
    route_index_of(&route_table(config))
}

fn route_index_of(table: &[Route]) -> Vec<String> {
    std::iter::once("GET /".to_string())
        .chain(
            table
                .iter()
                .map(|(method, path, _)| format!("{method} {path}")),
        )
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
    // aprender#2376(8): `GET /` and `GET /ready` are registered by the two OTHER
    // routers in this repo (apr-cli `commands/serve/routes.rs`, `serve_run_model.rs`)
    // and 404'd here, so which of three route surfaces you got depended on the
    // format of the file you passed to `apr serve run`. `/` now answers with the
    // route table this router actually mounted — the one thing a client needs to
    // discover the surface it landed on — and `/ready` is the conventional
    // readiness path, an alias of `/health/ready`.
    let table = route_table(&config);
    let index_routes = route_index_of(&table);

    // `GET /` answers with the route table this router actually mounted — the one
    // thing a client needs to discover the surface it landed on. It is mounted
    // separately because its body is the index derived from `table`.
    let root_index = index_routes.clone();
    let mut router = Router::new().route(
        "/",
        get(move || {
            let routes = root_index.clone();
            async move {
                Json(serde_json::json!({
                    "service": "apr serve",
                    "version": env!("CARGO_PKG_VERSION"),
                    "routes": routes,
                }))
            }
        }),
    );

    // Mount from the same table the index was built from. Advertising a route and
    // mounting it are now one act, so they cannot disagree.
    for (_, path, handler) in table {
        router = router.route(path, handler);
    }

    // GH-672: Return JSON error body for unmatched routes (not empty 404)
    // aprender#2376(12): serve the route list here instead of pointing clients at
    // /health, which does not have one.
    router = router.fallback(move || {
        let routes = index_routes.clone();
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

    // aprender#2376(3): mint a per-request CancelToken, publish it to the handlers
    // via request extensions, and cancel it when axum drops this request because
    // the client went away. Applied to the WHOLE router, not just the generate
    // routes, so a route added later cannot silently opt out of cancellation.
    router = router.layer(axum::middleware::from_fn(cancel_on_disconnect));

    // GH-649: Sanitize axum deserialization errors to avoid leaking internals to clients.
    // Axum returns 422 with raw serde error details by default; replace with a generic message.
    router = router.layer(axum::middleware::from_fn(sanitize_json_rejection));

    // GH-671: CORS support — allow cross-origin requests from browser-based clients.
    // Conditional: `apr serve run --no-cors` must emit no `access-control-*` header.
    if config.cors {
        router = router.layer(tower_http::cors::CorsLayer::permissive());
    }

    // #2506 (SURF-7/R14): contain handler panics.
    //
    // OUTERMOST, deliberately: a panic in any layer below -- including the
    // sanitizer and the cancel middleware -- must still become a response.
    // Mounted after `cors` for the same reason, so a panic cannot escape by
    // being raised in a layer that was added later.
    //
    // Before this, `catch_unwind` and `CatchPanicLayer` existed nowhere in this
    // crate and `tower-http`'s `catch-panic` feature was not enabled, so a
    // panicking handler unwound out of the service: no status, no body, nothing
    // a client could act on. That is also the one error shape that escaped
    // `route_surface_2376`'s "every error is actionable JSON" invariant --
    // it never became a response at all.
    router = router.layer(tower_http::catch_panic::CatchPanicLayer::custom(
        panic_to_json_500,
    ));

    router.with_state(state)
}

/// Turn a caught panic into the same JSON envelope every other error uses.
///
/// The panic payload is deliberately NOT forwarded: it is a Rust-internals
/// detail, and #2376 finding 7 already bans bodies naming things a client
/// cannot act on. It goes to the host's stderr instead, which is where this
/// server's telemetry belongs.
pub(crate) fn panic_to_json_500(err: Box<dyn std::any::Any + Send + 'static>) -> axum::response::Response {
    use axum::response::IntoResponse as _;
    let detail = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&'static str>().copied())
        .unwrap_or("<non-string panic payload>");
    eprintln!("apr serve: handler panicked: {detail}");

    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": "internal_error",
            "message": "The server hit an internal error handling this request. \
                        This is a bug; the request was not completed.",
        })),
    )
        .into_response()
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

    // The body is read ONLY to look for a marker a handler deliberately planted
    // (`client_visible_reason`). Axum's own rejection text carries no marker and
    // is therefore still discarded unread-for-forwarding — the property this
    // middleware exists to guarantee. A validator that wants to speak to the
    // client, e.g. the `n` refusal in #2375(9), opts in explicitly.
    let (parts_for_body, body_in) = response.into_parts();
    let raw = axum::body::to_bytes(body_in, usize::MAX)
        .await
        .unwrap_or_default();
    let message = client_visible_reason(&String::from_utf8_lossy(&raw))
        .unwrap_or_else(|| sanitized_error_message(status));
    let response = axum::response::Response::from_parts(parts_for_body, axum::body::Body::empty());

    let body = serde_json::to_vec(&ErrorResponse { error: message })
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

/// Marker a validator puts in front of a message that is MEANT for the client.
///
/// GH-649 sanitizes every 422 body so raw serde internals (Rust type paths,
/// field offsets) never reach an API client. That is right for accidental
/// errors and wrong for deliberate ones: aprender#2375(9) rejects `n > 1` at
/// deserialization, and without this marker the caller was told only "Invalid
/// request body" — refused, but with no way to learn which field was refused.
/// A validator prefixes its message with this marker to opt into being shown.
pub(crate) const CLIENT_VISIBLE_MARKER: &str = "[request] ";

/// Recover an authored, client-visible reason out of an axum rejection body.
///
/// Returns `None` for anything unmarked, so unauthored serde text stays hidden.
fn client_visible_reason(rejection_body: &str) -> Option<String> {
    let reason = rejection_body.split(CLIENT_VISIBLE_MARKER).nth(1)?;
    let reason = reason.split('\n').next().unwrap_or(reason);
    // serde_json appends " at line L column C" to a custom error; that offset is
    // about our parse position, not about the caller's mistake.
    let reason = reason
        .split(" at line ")
        .next()
        .unwrap_or(reason)
        .trim_end_matches(['"', ' ', '.']);
    (!reason.is_empty()).then(|| reason.to_string())
}

#[cfg(test)]
mod client_visible_reason_tests {
    use super::client_visible_reason;

    #[test]
    fn unmarked_serde_text_stays_hidden() {
        // GH-649: raw axum/serde text must never reach the client.
        assert_eq!(
            client_visible_reason(
                "Failed to deserialize the JSON body into the target type: \
                 missing field `messages` at line 1 column 42"
            ),
            None
        );
    }

    #[test]
    fn marked_reason_is_extracted_without_the_serde_frame() {
        let extracted = client_visible_reason(
            "Failed to deserialize the JSON body into the target type: n: \
             [request] n must be 1: this server returns exactly one choice per request",
        )
        .expect("marked reason is surfaced");
        assert!(extracted.starts_with("n must be 1"));
        assert!(
            !extracted.contains("deserialize"),
            "the serde frame must be stripped: {extracted}"
        );
    }

    #[test]
    fn serde_position_suffix_is_stripped() {
        // Observed on the live server: serde_json appends its parse position to
        // a custom error, which means nothing to the caller.
        let extracted = client_visible_reason(
            "Failed to deserialize the JSON body into the target type: n: \
             [request] n must be 1: send 3 requests instead at line 1 column 84",
        )
        .expect("marked reason is surfaced");
        assert_eq!(extracted, "n must be 1: send 3 requests instead");
    }
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

/// The name this server answers to for the model it is serving.
///
/// aprender#2375(7): `/v1/metrics` reported `model_name` as the literal
/// `"phi-2-q4_k_m"` whenever a cached GPU model was resident and `"N/A"`
/// otherwise — neither derived from the model actually loaded, so a monitor
/// watching a fleet labelled every server with the same wrong name or with no
/// name at all. Derived here from what the loader measured, falling back to the
/// id `/v1/models` advertises, and only reporting `"N/A"` when nothing is
/// resident to name.
fn served_model_name(state: &AppState) -> String {
    if let Some(stem) = state
        .model_source()
        .and_then(crate::api::ModelSourceInfo::path)
        .and_then(|p| {
            std::path::Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .filter(|s| !s.is_empty())
    {
        return stem;
    }
    if let Some(id) = state.default_model_id.clone() {
        return id;
    }
    if state.model_loaded() {
        // The id `GET /v1/models` lists in single-model mode, so a client can
        // send this straight back as `"model"`.
        return "default".to_string();
    }
    "N/A".to_string()
}

/// Request-latency percentiles for `/v1/metrics`, in milliseconds.
///
/// Shared by both feature variants of [`server_metrics_handler`] so they cannot
/// disagree about the same server: the GPU build reported a hardcoded
/// `(0.0, 0.0, 0.0)` whenever no GPU dispatch metrics existed — which is every
/// CPU deployment — while the non-GPU build reported `avg`, `avg * 1.5` and
/// `avg * 2.0`, two of which are not measurements at all.
///
/// Kernel-dispatch percentiles still win when GPU work was actually dispatched;
/// otherwise these are the collector's measured request latencies, and
/// `(0.0, 0.0, 0.0)` now means only "no request has completed yet".
fn measured_latency_percentiles(state: &AppState) -> (f64, f64, f64) {
    #[cfg(feature = "gpu")]
    if let Some(dispatch) = state.dispatch_metrics() {
        if dispatch.gpu_dispatches() > 0 {
            return (
                dispatch.gpu_latency_p50_us() / 1000.0,
                dispatch.gpu_latency_p95_us() / 1000.0,
                dispatch.gpu_latency_p99_us() / 1000.0,
            );
        }
        if dispatch.cpu_dispatches() > 0 {
            return (
                dispatch.cpu_latency_p50_us() / 1000.0,
                dispatch.cpu_latency_p95_us() / 1000.0,
                dispatch.cpu_latency_p99_us() / 1000.0,
            );
        }
    }
    state
        .metrics
        .latency_percentiles()
        .map_or((0.0, 0.0, 0.0), |p| (p.p50_ms, p.p95_ms, p.p99_ms))
}

/// Server metrics handler for TUI monitoring (PARITY-107)
/// GET /v1/metrics - Returns JSON metrics for realizar-monitor
#[cfg(feature = "gpu")]
async fn server_metrics_handler(State(state): State<AppState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();

    let (latency_p50_ms, latency_p95_ms, latency_p99_ms) = measured_latency_percentiles(&state);
    let (gpu_dispatches, cuda_path_active) = state
        .dispatch_metrics()
        .map_or((0, false), |dispatch| {
            let gpu = dispatch.gpu_dispatches();
            (gpu, gpu > 0)
        });

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

    let model_name = served_model_name(&state);

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

/// Server metrics handler for non-GPU builds (PARITY-107).
///
/// Reports the SAME measured percentiles as the GPU variant — this build used
/// to serve `p95 = avg * 1.5` and `p99 = avg * 2.0`, which describe no request
/// that ever happened.
#[cfg(not(feature = "gpu"))]
async fn server_metrics_handler(State(state): State<AppState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();
    let (latency_p50_ms, latency_p95_ms, latency_p99_ms) = measured_latency_percentiles(&state);

    Json(ServerMetricsResponse {
        throughput_tok_per_sec: snapshot.tokens_per_sec,
        latency_p50_ms,
        latency_p95_ms,
        latency_p99_ms,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 0,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 1,
        queue_depth: 0,
        total_tokens: snapshot.total_tokens as u64,
        total_requests: snapshot.total_requests as u64,
        uptime_secs: snapshot.uptime_secs,
        model_name: served_model_name(&state),
    })
}
