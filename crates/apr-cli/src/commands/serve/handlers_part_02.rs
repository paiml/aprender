
/// Handle POST /v1/completions for APR CPU inference.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)]
fn handle_apr_cpu_completion(
    state: &std::sync::Mutex<AprServerState>,
    req: &AprCompletionRequest,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let s = match state.lock() {
        Ok(guard) => guard.clone(),
        Err(_poisoned) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "Server state corrupted (lock poisoned). Please restart the server."
                })),
            )
                .into_response();
        }
    };

    let max_tokens = req.max_tokens.min(128);
    match run_apr_cpu_inference(&s, &req.prompt, max_tokens, req.temperature.unwrap_or(0.0)) {
        Ok(out) => Json(AprCompletionResponse {
            text: out.text,
            tokens_generated: out.tokens_generated,
            latency_ms: out.gen_duration.as_millis() as u64,
            tok_per_sec: compute_tok_per_sec(out.tokens_generated, out.gen_duration),
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

/// GH-283: Validate the "model" field in the request matches the loaded model.
///
/// Returns an error response if the model name doesn't match. Accepts "apr" as
/// a wildcard for backward compatibility. If no "model" field is present in the
/// request, validation passes (OpenAI spec allows omitting it).
#[cfg(feature = "inference")]
pub(crate) fn validate_request_model(
    req: &serde_json::Value,
    loaded_model: &str,
) -> Option<axum::response::Response> {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let requested = match req.get("model").and_then(serde_json::Value::as_str) {
        Some(m) => m,
        None => return None, // No model field — accept
    };

    // Accept "apr" as wildcard for backward compatibility
    if requested == "apr" || requested == loaded_model {
        return None;
    }

    Some(
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "The model '{}' does not exist. This server is serving '{}'.",
                        requested, loaded_model
                    ),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response(),
    )
}

/// Handle POST /v1/chat/completions for APR CPU inference (PAR-302).
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)]
fn handle_apr_cpu_chat_completion(
    state: &std::sync::Mutex<AprServerState>,
    headers: &axum::http::HeaderMap,
    req: &serde_json::Value,
) -> axum::response::Response {
    use axum::{
        response::{
            sse::{Event, Sse},
            IntoResponse,
        },
        Json,
    };
    use futures_util::stream;

    let trace_level = headers
        .get("X-Trace-Level")
        .and_then(|v| v.to_str().ok())
        .map(str::to_lowercase);

    let s = match state.lock() {
        Ok(guard) => guard.clone(),
        Err(_poisoned) => {
            return Json(serde_json::json!({
                "error": "Server state corrupted (lock poisoned). Please restart the server."
            }))
            .into_response();
        }
    };

    // GH-283: Validate model name before processing
    if let Some(err_response) = validate_request_model(req, &s.model_name) {
        return err_response;
    }

    let messages = req.get("messages").and_then(|m| m.as_array());
    let stream_mode = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32) as usize;
    let temperature = req
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0) as f32;

    let Some(msgs) = messages else {
        return Json(serde_json::json!({"error": "Missing messages"})).into_response();
    };

    let prompt = format_chatml(msgs);
    let start = Instant::now();

    let out = match run_apr_cpu_inference(&s, &prompt, max_tokens.min(256), temperature) {
        Ok(out) => out,
        Err(e) => return Json(serde_json::json!({"error": e})).into_response(),
    };

    let tok_per_sec = compute_tok_per_sec(out.tokens_generated, out.gen_duration);
    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if stream_mode {
        let response = serde_json::json!({
            "id": request_id, "object": "chat.completion.chunk", "created": created, "model": &s.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}]
        });
        let events = vec![
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string())),
            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]")),
        ];
        return Sse::new(stream::iter(events)).into_response();
    }

    let latency_ms = start.elapsed().as_millis() as u64;
    let mut response = serde_json::json!({
        "id": request_id, "object": "chat.completion", "created": created, "model": &s.model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": out.input_token_count, "completion_tokens": out.tokens_generated, "total_tokens": out.input_token_count + out.tokens_generated},
        "_apr_metrics": {"latency_ms": latency_ms, "tok_per_sec": tok_per_sec}
    });

    if let Some(ref level) = trace_level {
        let trace_data = serde_json::json!({
            "total_time_us": latency_ms * 1000, "prompt_tokens": out.input_token_count,
            "completion_tokens": out.tokens_generated, "layers": 28
        });
        if let Some(obj) = response.as_object_mut() {
            let key = match level.as_str() {
                "brick" => Some("brick_trace"),
                "step" => Some("step_trace"),
                "layer" => Some("layer_trace"),
                _ => None,
            };
            if let Some(key) = key {
                obj.insert(key.to_string(), trace_data);
            }
        }
    }

    Json(response).into_response()
}

/// Print APR CPU server startup banner.
fn print_apr_cpu_banner(bind_addr: &str, is_transformer: bool) {
    println!();
    println!(
        "{}",
        format!("APR Inference Server listening on http://{bind_addr}")
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
        format!("Mode: CPU | Transformer: {is_transformer}").dimmed()
    );
    println!("{}", "Press Ctrl+C to stop".dimmed());
}

// ============================================================================
// APR GPU server handler
// ============================================================================

/// Start APR server with GPU acceleration
/// PMAT-098: Updated to use BPE tokenizer for proper encoding
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap internally
/// Encode a prompt using BPE tokenizer or char fallback (PMAT-098).
fn encode_prompt(tok: Option<&SafeTensorsTokenizerInfo>, prompt: &str) -> Vec<u32> {
    match tok {
        Some(tok) => tok.tokenizer.encode(prompt),
        None => prompt.chars().map(|c| c as u32).collect(),
    }
}

/// Get EOS token ID from tokenizer info or use provided default.
fn eos_token_id(tok: Option<&SafeTensorsTokenizerInfo>, default: u32) -> u32 {
    tok.and_then(|t| t.eos_token_id).unwrap_or(default)
}

/// Run GPU generation with lock poisoning handling (PMAT-189).
#[cfg(feature = "inference")]
fn run_gpu_generation(
    cuda: &std::sync::Mutex<realizar::apr::AprV2ModelCuda>,
    input_tokens: &[u32],
    max_tokens: usize,
    eos_id: u32,
) -> std::result::Result<Vec<u32>, String> {
    use realizar::apr::AprModel;
    let mut model = cuda.lock().map_err(|_| {
        "GPU model state corrupted (lock poisoned). Please restart the server.".to_string()
    })?;
    model
        .generate_cuda(input_tokens, max_tokens, eos_id)
        .map_err(|e| format!("GPU generation failed: {e}"))
}

/// Decode output tokens using BPE tokenizer or char fallback (PMAT-098).
fn decode_tokens(tok: Option<&SafeTensorsTokenizerInfo>, tokens: &[u32]) -> String {
    match tok {
        Some(tok) => tok.tokenizer.decode(tokens).unwrap_or_default(),
        None => tokens.iter().filter_map(|&t| char::from_u32(t)).collect(),
    }
}

/// Slice new tokens from output (tokens after input prefix).
fn extract_new_tokens(output: &[u32], input_len: usize) -> &[u32] {
    if output.len() > input_len {
        &output[input_len..]
    } else {
        output
    }
}

/// Compute tokens/second from count and duration.
fn compute_tok_per_sec(count: usize, elapsed: std::time::Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        count as f64 / secs
    } else {
        0.0
    }
}

/// Generate unique request ID for OpenAI-compatible responses.
fn generate_request_id() -> String {
    format!(
        "chatcmpl-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    )
}

/// Format chat messages as ChatML prompt.
fn format_chatml(messages: &[serde_json::Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        write!(prompt, "<|im_start|>{role}\n{content}<|im_end|>\n")
            .expect("write to String cannot fail");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[cfg(feature = "inference")]
#[derive(serde::Deserialize)]
struct GpuCompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens_gpu")]
    max_tokens: usize,
}

#[cfg(feature = "inference")]
fn default_max_tokens_gpu() -> usize {
    32
}

#[cfg(feature = "inference")]
#[derive(serde::Serialize)]
struct GpuCompletionResponse {
    text: String,
    tokens_generated: usize,
    latency_ms: u64,
    tok_per_sec: f64,
}

#[cfg(feature = "inference")]
fn start_apr_server_gpu(
    model_path: &Path,
    model: realizar::apr::AprModel,
    config: &ServerConfig,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
) -> Result<()> {
    use realizar::apr::AprV2ModelCuda;
    use std::sync::Mutex;

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

    // GH-261: Load CPU transformer as per-request fallback
    let cpu_state = load_apr_model_state(model_path, config)?;
    println!(
        "{}",
        "CPU fallback: enabled (AprTransformer ready)".cyan()
    );

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let cuda_model = Arc::new(Mutex::new(cuda_model));
    let tokenizer = Arc::new(tokenizer);
    let cpu_state = Arc::new(Mutex::new(cpu_state));

    runtime.block_on(async move {
        let app = build_gpu_router(cuda_model, tokenizer, cpu_state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        print_gpu_server_banner(&bind_addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Build the axum Router for GPU inference endpoints.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn build_gpu_router(
    cuda_model: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tokenizer: Arc<Option<SafeTensorsTokenizerInfo>>,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
) -> axum::Router {
    use axum::{
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };

    let cuda_for_completions = cuda_model.clone();
    let tok_for_completions = tokenizer.clone();
    let cpu_for_completions = cpu_state.clone();
    let cuda_for_chat = cuda_model;
    let tok_for_chat = tokenizer;
    let cpu_for_chat = cpu_state;

    Router::new()
        .route(
            "/health",
            get(|| async {
                Json(serde_json::json!({"status": "healthy", "gpu": true, "gpu_fallback": true}))
            }),
        )
        .route(
            "/v1/completions",
            post(move |Json(req): Json<GpuCompletionRequest>| {
                let cuda = cuda_for_completions.clone();
                let tok_info = tok_for_completions.clone();
                let cpu = cpu_for_completions.clone();
                async move {
                    handle_gpu_completion(&cuda, tok_info.as_ref().as_ref(), &req, &cpu)
                        .into_response()
                }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |Json(req): Json<serde_json::Value>| {
                let cuda = cuda_for_chat.clone();
                let tok_info = tok_for_chat.clone();
                let cpu = cpu_for_chat.clone();
                async move {
                    handle_gpu_chat_completion(&cuda, tok_info.as_ref().as_ref(), &req, &cpu)
                        .into_response()
                }
            }),
        )
        .route(
            "/",
            get(|| async {
                "APR v2 GPU Inference Server - POST /v1/completions, /v1/chat/completions"
            }),
        )
}
