
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

/// Spawn a blocking task to stream tokens via an mpsc channel.
///
/// GH-284: Runs transformer generation in a dedicated thread, sending each
/// token through the channel. The channel closes when generation completes.
#[cfg(feature = "inference")]
fn spawn_cpu_streaming_task(
    s: AprServerState,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    tx: tokio::sync::mpsc::Sender<std::result::Result<u32, String>>,
) {
    tokio::task::spawn_blocking(move || {
        let Some(transformer) = s.transformer.as_ref() else {
            if tx.blocking_send(Err("Transformer not loaded".to_string())).is_err() {
                eprintln!("Warning: failed to send error to client (channel closed)");
            }
            return;
        };

        let input_tokens: Vec<u32> = match &s.tokenizer {
            Some(tok) => tok.tokenizer.encode(&prompt),
            None => prompt.chars().map(|c| c as u32).collect(),
        };

        let gen_config = realizar::apr_transformer::GenerateConfig {
            max_tokens,
            temperature,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            trace: false,
            stop_tokens: vec![],
        };

        let Ok(t) = transformer.lock() else {
            if tx.blocking_send(Err("Lock poisoned".to_string())).is_err() {
                eprintln!("Warning: failed to send error to client (channel closed)");
            }
            return;
        };

        // GH-326: Log generation errors instead of silently discarding
        if let Err(e) = t.generate_with_cache_streaming(&input_tokens, &gen_config, |token_id| {
            tx.blocking_send(Ok(token_id)).is_ok()
        }) {
            eprintln!("Warning: streaming generation failed: {e}");
        }
    });
}

/// PMAT-928: spawn the per-token generation task, sending each token's DECODED
/// TEXT through `tx`. Reuses the exact incremental stream the OpenAI SSE path
/// uses (`generate_with_cache_streaming`); only the channel payload differs
/// (already-decoded `String` instead of a raw `u32`) so the Ollama NDJSON
/// reshaper stays decoupled from the tokenizer.
///
/// The test seam path (`demo_scripted_tokens`) feeds a deterministic token
/// sequence through the SAME channel + reshape pipeline, so a falsifier can
/// observe true multi-chunk NDJSON without a real transformer.
#[cfg(feature = "inference")]
fn spawn_cpu_token_text_stream(
    s: AprServerState,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    tx: tokio::sync::mpsc::Sender<std::result::Result<String, String>>,
) {
    // Test seam: replay a scripted token sequence through the channel.
    if let Some(scripted) = s.demo_scripted_tokens.clone() {
        tokio::task::spawn_blocking(move || {
            for tok in scripted {
                if tx.blocking_send(Ok(tok)).is_err() {
                    break; // client disconnected
                }
            }
            // Dropping tx closes the channel → terminal `done:true` chunk.
        });
        return;
    }

    let tokenizer = s.tokenizer.clone();
    tokio::task::spawn_blocking(move || {
        let Some(transformer) = s.transformer.as_ref() else {
            if tx
                .blocking_send(Err("Transformer not loaded".to_string()))
                .is_err()
            {
                eprintln!("Warning: failed to send error to client (channel closed)");
            }
            return;
        };

        let input_tokens: Vec<u32> = match &s.tokenizer {
            Some(tok) => tok.tokenizer.encode(&prompt),
            None => prompt.chars().map(|c| c as u32).collect(),
        };

        let gen_config = realizar::apr_transformer::GenerateConfig {
            max_tokens,
            temperature,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            trace: false,
            stop_tokens: vec![],
        };

        let Ok(t) = transformer.lock() else {
            if tx.blocking_send(Err("Lock poisoned".to_string())).is_err() {
                eprintln!("Warning: failed to send error to client (channel closed)");
            }
            return;
        };

        if let Err(e) = t.generate_with_cache_streaming(&input_tokens, &gen_config, |token_id| {
            let text = decode_single_token(tokenizer.as_ref(), token_id);
            tx.blocking_send(Ok(text)).is_ok()
        }) {
            eprintln!("Warning: streaming generation failed: {e}");
        }
    });
}

/// Build an SSE stream from a token receiver channel.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
fn build_cpu_sse_stream(
    rx: tokio::sync::mpsc::Receiver<std::result::Result<u32, String>>,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
    model_name: String,
) -> axum::response::Response {
    use axum::response::{sse::{Event, Sse}, IntoResponse};

    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

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
                    let event = Event::default().data("[DONE]");
                    Some((
                        Ok::<_, std::convert::Infallible>(event),
                        (None, tokenizer, request_id, created, model_name),
                    ))
                }
            }
        },
    );

    Sse::new(stream).into_response()
}

/// Insert trace data into a chat completion response based on trace level.
#[cfg(feature = "inference")]
fn insert_trace_data(response: &mut serde_json::Value, trace_level: Option<&str>, trace_data: serde_json::Value) {
    let Some(level) = trace_level else { return };
    let key = match level {
        "brick" => "brick_trace",
        "step" => "step_trace",
        "layer" => "layer_trace",
        _ => return,
    };
    if let Some(obj) = response.as_object_mut() {
        obj.insert(key.to_string(), trace_data);
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
    use axum::{response::IntoResponse, Json};

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

    if let Some(err_response) = validate_request_model(req, &s.model_name) {
        return err_response;
    }

    let messages = req.get("messages").and_then(|m| m.as_array());
    let stream_mode = req.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(serde_json::Value::as_u64).unwrap_or(32) as usize;
    let temperature = req.get("temperature").and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32;

    let Some(msgs) = messages else {
        return Json(serde_json::json!({"error": "Missing messages"})).into_response();
    };

    let prompt = format_chatml(msgs);

    // GH-284: True SSE streaming path
    if stream_mode {
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<u32, String>>(16);
        spawn_cpu_streaming_task(s.clone(), prompt, max_tokens.min(4096), temperature, tx);
        return build_cpu_sse_stream(rx, s.tokenizer.clone(), s.model_name.clone());
    }

    // GH-284: Non-streaming path
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
        Err(e) => return Json(serde_json::json!({"error": format!("Task failed: {e}")})).into_response(),
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

    let trace_data = serde_json::json!({
        "total_time_us": latency_ms * 1000, "prompt_tokens": out.input_token_count,
        "completion_tokens": out.tokens_generated, "layers": 28
    });
    insert_trace_data(&mut response, trace_level.as_deref(), trace_data);

    Json(response).into_response()
}

/// PMAT-928: count prompt tokens the way generation will, for the streamed
/// final object's `prompt_eval_count`. Mirrors the encode in
/// `spawn_cpu_token_text_stream` / `run_apr_cpu_inference`.
#[cfg(feature = "inference")]
fn count_prompt_tokens(s: &AprServerState, prompt: &str) -> usize {
    if let Some(ref tok) = s.embedded_tokenizer {
        tok.encode(prompt).len()
    } else if let Some(ref tok) = s.tokenizer {
        tok.tokenizer.encode(prompt).len()
    } else {
        prompt.chars().count()
    }
}

/// PMAT-928: handle Ollama `/api/chat` on the APR-CPU router.
///
/// `stream != false` (Ollama default) ⇒ NDJSON streaming body reusing the SAME
/// per-token stream `/v1/chat/completions` uses; `stream:false` ⇒ the existing
/// coalesced single-object body via the shared chat handler.
#[cfg(feature = "inference")]
async fn handle_apr_cpu_ollama_chat(
    state: &std::sync::Mutex<AprServerState>,
    req: &super::ollama::OllamaChatRequest,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let model = super::ollama::model_label(&req.model);

    if req.stream {
        let s = match state.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => return ollama_stream_error(super::ollama::OllamaStreamKind::Chat, model),
        };
        let msgs: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        let prompt = format_chatml(&msgs);
        let (max_tokens, temperature) = ollama_sampling(&req.options);
        let prompt_eval_count = count_prompt_tokens(&s, &prompt);
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, String>>(16);
        spawn_cpu_token_text_stream(s, prompt, max_tokens, temperature, tx);
        return super::ollama::ollama_ndjson_stream(
            super::ollama::OllamaStreamKind::Chat,
            model,
            prompt_eval_count,
            rx,
        );
    }

    // Coalesced path: reuse the existing shared chat backend + reshape.
    let openai_body = super::ollama::ollama_chat_to_openai(req);
    let inner =
        handle_apr_cpu_chat_completion(state, &axum::http::HeaderMap::new(), &openai_body).await;
    super::ollama::reshape_openai_to_ollama_chat(model, inner)
        .await
        .into_response()
}

/// PMAT-928: handle Ollama `/api/generate` on the APR-CPU router (flat
/// `response` field; same streaming/coalescing rules as chat).
#[cfg(feature = "inference")]
async fn handle_apr_cpu_ollama_generate(
    state: &std::sync::Mutex<AprServerState>,
    req: &super::ollama::OllamaGenerateRequest,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let model = super::ollama::model_label(&req.model);

    if req.stream {
        let s = match state.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => return ollama_stream_error(super::ollama::OllamaStreamKind::Generate, model),
        };
        let mut msgs: Vec<serde_json::Value> = Vec::new();
        if let Some(system) = req.system.as_ref().filter(|sys| !sys.is_empty()) {
            msgs.push(serde_json::json!({"role": "system", "content": system}));
        }
        msgs.push(serde_json::json!({"role": "user", "content": req.prompt}));
        let prompt = format_chatml(&msgs);
        let (max_tokens, temperature) = ollama_sampling(&req.options);
        let prompt_eval_count = count_prompt_tokens(&s, &prompt);
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, String>>(16);
        spawn_cpu_token_text_stream(s, prompt, max_tokens, temperature, tx);
        return super::ollama::ollama_ndjson_stream(
            super::ollama::OllamaStreamKind::Generate,
            model,
            prompt_eval_count,
            rx,
        );
    }

    let openai_body = super::ollama::ollama_generate_to_openai(req);
    let inner =
        handle_apr_cpu_chat_completion(state, &axum::http::HeaderMap::new(), &openai_body).await;
    super::ollama::reshape_openai_to_ollama_generate(model, inner)
        .await
        .into_response()
}

/// PMAT-928: translate an Ollama `options` block to `(max_tokens, temperature)`.
#[cfg(feature = "inference")]
fn ollama_sampling(options: &Option<super::ollama::OllamaOptions>) -> (usize, f32) {
    let mut max_tokens = 32usize;
    let mut temperature = 0.0f32;
    if let Some(opts) = options {
        if let Some(n) = opts.num_predict {
            max_tokens = n as usize;
        }
        if let Some(t) = opts.temperature {
            temperature = t;
        }
    }
    (max_tokens.min(4096), temperature)
}

/// PMAT-928: a degenerate NDJSON stream that emits only the terminal
/// `done:true` object, used when state can't be acquired. Keeps the wire shape
/// NDJSON (never an error JSON object lacking `done`).
#[cfg(feature = "inference")]
fn ollama_stream_error(
    kind: super::ollama::OllamaStreamKind,
    model: String,
) -> axum::response::Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, String>>(1);
    // Send an error to immediately terminate with the `done:true` object.
    let _ = tx.try_send(Err("server state unavailable".to_string()));
    drop(tx);
    super::ollama::ollama_ndjson_stream(kind, model, 0, rx)
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
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
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

// ============================================================================
// PMAT-125 B3: unit tests for APR-CPU completion pure helpers
//
// These functions are `include!`d into the `serve::handlers` module, so this
// test module exercises them at that scope (CPU-only; no transformer/GPU).
// ============================================================================
#[cfg(all(test, feature = "inference"))]
mod apr_cpu_completion_tests {
    use super::*;

    // ---- validate_request_model ---------------------------------------

    #[test]
    fn validate_request_model_accepts_missing_field() {
        let req = serde_json::json!({"messages": []});
        assert!(validate_request_model(&req, "qwen2.5-1.5b").is_none());
    }

    #[test]
    fn validate_request_model_accepts_exact_match() {
        let req = serde_json::json!({"model": "qwen2.5-1.5b"});
        assert!(validate_request_model(&req, "qwen2.5-1.5b").is_none());
    }

    #[test]
    fn validate_request_model_accepts_apr_wildcard() {
        let req = serde_json::json!({"model": "apr"});
        assert!(validate_request_model(&req, "anything").is_none());
    }

    #[test]
    fn validate_request_model_rejects_mismatch() {
        let req = serde_json::json!({"model": "gpt-4"});
        let resp = validate_request_model(&req, "qwen2.5-1.5b");
        assert!(
            resp.is_some(),
            "mismatched model name must produce a 404 response"
        );
        assert_eq!(
            resp.expect("response").status(),
            axum::http::StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn validate_request_model_non_string_model_field_accepted() {
        // as_str() yields None for a numeric model field → accept.
        let req = serde_json::json!({"model": 123});
        assert!(validate_request_model(&req, "loaded").is_none());
    }

    // ---- insert_trace_data --------------------------------------------

    #[test]
    fn insert_trace_data_none_level_is_noop() {
        let mut resp = serde_json::json!({"a": 1});
        insert_trace_data(&mut resp, None, serde_json::json!({"x": 1}));
        assert!(resp.get("brick_trace").is_none());
        assert!(resp.get("step_trace").is_none());
        assert!(resp.get("layer_trace").is_none());
    }

    #[test]
    fn insert_trace_data_unknown_level_is_noop() {
        let mut resp = serde_json::json!({"a": 1});
        insert_trace_data(&mut resp, Some("garbage"), serde_json::json!({"x": 1}));
        assert_eq!(resp.as_object().expect("obj").len(), 1);
    }

    #[test]
    fn insert_trace_data_maps_levels_to_keys() {
        for (level, key) in [
            ("brick", "brick_trace"),
            ("step", "step_trace"),
            ("layer", "layer_trace"),
        ] {
            let mut resp = serde_json::json!({});
            insert_trace_data(&mut resp, Some(level), serde_json::json!({"layers": 28}));
            assert_eq!(resp[key]["layers"], 28, "level {level} → key {key}");
        }
    }

    // ---- ollama_sampling ----------------------------------------------

    #[test]
    fn ollama_sampling_defaults_when_none() {
        let (max_tokens, temp) = ollama_sampling(&None);
        assert_eq!(max_tokens, 32);
        assert!((temp).abs() < f32::EPSILON);
    }

    #[test]
    fn ollama_sampling_reads_options() {
        let opts = super::super::ollama::OllamaOptions {
            temperature: Some(0.7),
            num_predict: Some(128),
            ..Default::default()
        };
        let (max_tokens, temp) = ollama_sampling(&Some(opts));
        assert_eq!(max_tokens, 128);
        assert!((temp - 0.7).abs() < 1e-6);
    }

    #[test]
    fn ollama_sampling_clamps_max_tokens_to_4096() {
        let opts = super::super::ollama::OllamaOptions {
            num_predict: Some(100_000),
            ..Default::default()
        };
        let (max_tokens, _) = ollama_sampling(&Some(opts));
        assert_eq!(max_tokens, 4096, "max_tokens is capped at 4096");
    }

    // ---- eos_token_id -------------------------------------------------

    #[test]
    fn eos_token_id_uses_default_when_no_tokenizer() {
        assert_eq!(eos_token_id(None, 2), 2);
    }

    // ---- extract_new_tokens -------------------------------------------

    #[test]
    fn extract_new_tokens_slices_after_prompt() {
        let output = [1u32, 2, 3, 4, 5];
        assert_eq!(extract_new_tokens(&output, 2), &[3, 4, 5]);
    }

    #[test]
    fn extract_new_tokens_returns_all_when_not_longer() {
        let output = [1u32, 2, 3];
        // input_len >= output.len() → return whole slice (defensive).
        assert_eq!(extract_new_tokens(&output, 3), &[1, 2, 3]);
        assert_eq!(extract_new_tokens(&output, 10), &[1, 2, 3]);
    }

    #[test]
    fn extract_new_tokens_empty_output() {
        let output: [u32; 0] = [];
        assert!(extract_new_tokens(&output, 0).is_empty());
    }

    // ---- compute_tok_per_sec ------------------------------------------

    #[test]
    fn compute_tok_per_sec_positive_duration() {
        let tps = compute_tok_per_sec(100, std::time::Duration::from_secs(2));
        assert!((tps - 50.0).abs() < 1e-6);
    }

    #[test]
    fn compute_tok_per_sec_zero_duration_is_zero() {
        let tps = compute_tok_per_sec(100, std::time::Duration::ZERO);
        assert!((tps).abs() < f64::EPSILON, "no divide-by-zero blowup");
    }

    #[test]
    fn compute_tok_per_sec_zero_tokens() {
        let tps = compute_tok_per_sec(0, std::time::Duration::from_secs(1));
        assert!((tps).abs() < f64::EPSILON);
    }

    // ---- generate_request_id ------------------------------------------

    #[test]
    fn generate_request_id_has_chatcmpl_prefix_and_pid() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"), "got {id}");
        // Format is chatcmpl-<nanos>-<pid>; three dash-separated parts.
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 3, "id should be chatcmpl-<nanos>-<pid>: {id}");
        assert!(parts[1].chars().all(|c| c.is_ascii_digit()));
        assert!(parts[2].chars().all(|c| c.is_ascii_digit()));
    }

    // ---- format_chatml ------------------------------------------------

    #[test]
    fn format_chatml_wraps_messages_and_appends_assistant() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "be brief"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let prompt = format_chatml(&msgs);
        assert!(prompt.contains("<|im_start|>system\nbe brief<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_chatml_defaults_role_and_content() {
        // Missing role defaults to "user"; missing content defaults to "".
        let msgs = vec![serde_json::json!({})];
        let prompt = format_chatml(&msgs);
        assert!(prompt.contains("<|im_start|>user\n<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_chatml_empty_messages_just_assistant_header() {
        let prompt = format_chatml(&[]);
        assert_eq!(prompt, "<|im_start|>assistant\n");
    }

    // ---- default_max_tokens_gpu ---------------------------------------

    #[test]
    fn default_max_tokens_gpu_is_32() {
        assert_eq!(default_max_tokens_gpu(), 32);
    }
}

include!("handlers_include_01.rs");
