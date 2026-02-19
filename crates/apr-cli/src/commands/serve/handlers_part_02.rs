
/// Handle POST /v1/completions for APR CPU inference.
///
/// GH-284: Now async with `spawn_blocking` to avoid blocking the tokio runtime.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)]
async fn handle_apr_cpu_completion(
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

    let max_tokens = req.max_tokens.min(4096);
    let prompt = req.prompt.clone();
    let temperature = req.temperature.unwrap_or(0.0);

    // GH-284: Run inference off the async runtime to avoid blocking
    let result = tokio::task::spawn_blocking(move || {
        run_apr_cpu_inference(&s, &prompt, max_tokens, temperature)
    })
    .await;

    match result {
        Ok(Ok(out)) => Json(AprCompletionResponse {
            text: out.text,
            tokens_generated: out.tokens_generated,
            latency_ms: out.gen_duration.as_millis() as u64,
            tok_per_sec: compute_tok_per_sec(out.tokens_generated, out.gen_duration),
        })
        .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Task failed: {e}")})),
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
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
pub(crate) fn validate_request_model(
    req: &serde_json::Value,
    loaded_model: &str,
) -> Option<axum::response::Response> {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let Some(requested) = req.get("model").and_then(serde_json::Value::as_str) else {
        return None; // No model field — accept
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

/// Decode a single token ID to text using BPE tokenizer or char fallback.
#[cfg(feature = "inference")]
fn decode_single_token(tokenizer: Option<&SafeTensorsTokenizerInfo>, token_id: u32) -> String {
    match tokenizer {
        Some(tok) => tok.tokenizer.decode(&[token_id]).unwrap_or_default(),
        None => char::from_u32(token_id)
            .map_or(String::new(), |c| c.to_string()),
    }
}

/// Handle POST /v1/chat/completions for APR CPU inference (PAR-302).
///
/// GH-284: True per-token SSE streaming via `spawn_blocking` + mpsc channel.
/// Non-streaming path also uses `spawn_blocking` to avoid blocking the runtime.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)]
async fn handle_apr_cpu_chat_completion(
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

    // ========================================================================
    // GH-284: True SSE streaming path
    // ========================================================================
    if stream_mode {
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<u32, String>>(16);
        let s_for_gen = s.clone();
        let prompt_for_gen = prompt;
        let max_tokens_clamped = max_tokens.min(4096);

        // Run generation in a blocking thread so the tokio runtime stays free
        tokio::task::spawn_blocking(move || {
            let Some(transformer) = s_for_gen.transformer.as_ref() else {
                let _ = tx.blocking_send(Err("Transformer not loaded".to_string()));
                return;
            };

            let input_tokens: Vec<u32> = match &s_for_gen.tokenizer {
                Some(tok) => tok.tokenizer.encode(&prompt_for_gen),
                None => prompt_for_gen.chars().map(|c| c as u32).collect(),
            };

            let gen_config = realizar::apr_transformer::GenerateConfig {
                max_tokens: max_tokens_clamped,
                temperature,
                top_p: 0.9,
                top_k: 0,
                repetition_penalty: 1.0,
                trace: false,
            };

            let Ok(t) = transformer.lock() else {
                let _ = tx.blocking_send(Err("Lock poisoned".to_string()));
                return;
            };

            // GH-284: Stream each token via the callback → mpsc channel
            let _ = t.generate_with_cache_streaming(&input_tokens, &gen_config, |token_id| {
                // blocking_send returns Err when receiver is dropped (client disconnected)
                tx.blocking_send(Ok(token_id)).is_ok()
            });
            // tx is dropped here → channel closes → SSE stream ends
        });

        // Build async SSE stream from the channel receiver
        let tokenizer = s.tokenizer.clone();
        let request_id = generate_request_id();
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let model_name = s.model_name.clone();

        // State for unfold: Option<Receiver> — None means we've sent [DONE]
        let stream = futures_util::stream::unfold(
            (Some(rx), tokenizer, request_id, created, model_name),
            |(maybe_rx, tokenizer, request_id, created, model_name)| async move {
                let mut rx = maybe_rx?;
                match rx.recv().await {
                    Some(Ok(token_id)) => {
                        let text = decode_single_token(tokenizer.as_ref(), token_id);
                        let chunk = serde_json::json!({
                            "id": &request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        let event = Event::default().data(chunk.to_string());
                        Some((
                            Ok::<_, std::convert::Infallible>(event),
                            (Some(rx), tokenizer, request_id, created, model_name),
                        ))
                    }
                    Some(Err(_)) | None => {
                        // Generation complete — yield [DONE] and end stream
                        let event = Event::default().data("[DONE]");
                        Some((
                            Ok::<_, std::convert::Infallible>(event),
                            (None, tokenizer, request_id, created, model_name),
                        ))
                    }
                }
            },
        );

        return Sse::new(stream).into_response();
    }

    // ========================================================================
    // GH-284: Non-streaming path — still use spawn_blocking
    // ========================================================================
    let start = Instant::now();
    let s_for_blocking = s.clone();
    let prompt_owned = prompt;
    let max_t = max_tokens.min(4096);

    let result = tokio::task::spawn_blocking(move || {
        run_apr_cpu_inference(&s_for_blocking, &prompt_owned, max_t, temperature)
    })
    .await;

    let out = match result {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Json(serde_json::json!({"error": e})).into_response(),
        Err(e) => {
            return Json(serde_json::json!({"error": format!("Task failed: {e}")}))
                .into_response()
        }
    };

    let tok_per_sec = compute_tok_per_sec(out.tokens_generated, out.gen_duration);
    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

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

include!("handlers_part_02_include_01.rs");
