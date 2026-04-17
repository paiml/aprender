
/// #169: SafeTensors CUDA backend — format parity with GGUF GPU path.
#[cfg(feature = "cuda")]
fn try_safetensors_cuda_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
) -> Option<Response> {
    let model_lock = state.safetensors_cuda_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };

    let prompt = crate::api::realize_handlers::format_chat_messages(&request.messages, Some(&request.model));
    let input_ids = tokenizer.encode(&prompt);
    let max_tokens = request.max_tokens.unwrap_or(256).min(4096) as usize;

    // Qwen2 EOS: 151645 (<|endoftext|>)
    let eos_id = 151645u32;

    let mut model = match model_lock.lock() {
        Ok(m) => m,
        Err(e) => {
            return Some(
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Model lock failed: {e}")})),
                )
                    .into_response(),
            );
        }
    };

    let output_ids = match model.generate(&input_ids, max_tokens, eos_id) {
        Ok(ids) => ids,
        Err(e) => {
            let msg = format!("SafeTensors CUDA generation failed: {e}");
            return Some(
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": msg})),
                )
                    .into_response(),
            );
        }
    };

    let output_text = tokenizer.decode(&output_ids).unwrap_or_else(|_| String::from("[decode error]"));
    let completion_tokens = output_ids.len();
    let prompt_tokens = input_ids.len();

    let finish_reason = if completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    };

    let body = format!(
        r#"{{"id":"{}","object":"chat.completion","model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":"{}"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        request_id,
        request.model,
        serde_json::to_string(&output_text).unwrap_or_default(),
        finish_reason,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    );

    Some((
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    ).into_response())
}

/// CUDA-optimized backend (true streaming support).
#[cfg(feature = "cuda")]
async fn try_cuda_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let ttft_trace = std::env::var("TTFT_TRACE").is_ok();
    let t0 = if ttft_trace { Some(std::time::Instant::now()) } else { None };

    let cuda_model_lock = state.cuda_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    // GH-319: Use actual model architecture for chat template detection
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    if let Some(t) = t0 {
        eprintln!("[TTFT] {:>20}: {:>7.2}ms ({}tok)", "tokenize", t.elapsed().as_secs_f64() * 1000.0, prompt_ids.len());
    }
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, eos_token_id) = chat_gen_params(request, &tokenizer, state.model_eos_token_id());

    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        stop_tokens: vec![eos_token_id],
        trace: state.should_trace(trace_level),
        ..Default::default()
    };

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(16);

        // PMAT-044: Use batch scheduler if available (continuous batching)
        if let Some(batch_tx) = state.cuda_batch_tx() {
            let batch_req = super::cuda_batch_scheduler::CudaBatchRequest {
                prompt_ids,
                config: q_config,
                token_tx: tx,
                non_streaming: false,
                enqueue_time: std::time::Instant::now(),
            };
            if let Err(e) = batch_tx.try_send(batch_req) {
                return Some(fail_response(
                    state,
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Batch queue full: {e}"),
                ));
            }
        } else {
            // Fallback: direct RwLock path (serialized)
            let cuda_model_clone = cuda_model_lock.clone();
            let prompt_ids_clone = prompt_ids.clone();
            let q_config_clone = q_config.clone();

            tokio::task::spawn_blocking(move || {
                let mut cuda_model = cuda_model_clone.write().expect("operation failed");
                let result = cuda_model.generate_gpu_resident_streaming(
                    &prompt_ids_clone,
                    &q_config_clone,
                    |token_id| tx.blocking_send(Ok(token_id)).is_ok(),
                );
                if let Err(e) = result {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            });
        }

        return Some(true_streaming_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            false,
        ));
    }

    // Non-streaming CUDA — route through batch scheduler when available (realizr#211)
    let (token_ids, completion_tokens, response_text) = if let Some(batch_tx) = state.cuda_batch_tx() {
        // Use batch scheduler: submit request and collect all tokens
        // realizr#212: capacity 512 for bulk-send after non-streaming generation
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(512);
        let batch_req = super::cuda_batch_scheduler::CudaBatchRequest {
            prompt_ids,
            config: q_config,
            token_tx: tx,
            non_streaming: true, // realizr#212: scheduler accumulates + bulk-sends
            enqueue_time: std::time::Instant::now(),
        };
        if let Err(e) = batch_tx.try_send(batch_req) {
            return Some(fail_response(
                state,
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Batch queue full: {e}"),
            ));
        }
        // Collect all tokens via async receive (realizr#211)
        let mut tokens = Vec::new();
        while let Some(result) = rx.recv().await {
            match result {
                Ok(token_id) => tokens.push(token_id),
                Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
            }
        }
        let n = tokens.len();
        let text = tokenizer.decode(&tokens).unwrap_or_else(|_| String::new());
        (tokens, n, clean_chat_output(&text))
    } else {
        // Fallback: direct RwLock path (serialized, no batch scheduler)
        let mut cuda_model = cuda_model_lock.write().expect("operation failed");
        let generated = match cuda_model.generate_gpu_resident(&prompt_ids, &q_config) {
            Ok(g) => g,
            Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
        };
        let tokens: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
        let n = tokens.len();
        let text = tokenizer.decode(&tokens).unwrap_or_else(|_| String::new());
        (tokens, n, clean_chat_output(&text))
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        latency,
    ))
}

/// Quantized model (GGUF serve mode) backend with true streaming.
fn try_quantized_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let quantized_model = state.quantized_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    // GH-319: Use actual model architecture for chat template detection
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, eos_token_id) = chat_gen_params(request, &tokenizer, state.model_eos_token_id());

    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        stop_tokens: vec![eos_token_id],
        trace: state.should_trace(trace_level),
        ..Default::default()
    };

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(16);
        let quantized_model_clone = quantized_model.clone();
        let prompt_ids_clone = prompt_ids.clone();
        let q_config_clone = q_config.clone();

        tokio::task::spawn_blocking(move || {
            let result = quantized_model_clone.generate_with_cache_streaming(
                &prompt_ids_clone,
                &q_config_clone,
                |token_id| tx.blocking_send(Ok(token_id)).is_ok(),
            );
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e.to_string()));
            }
        });

        return Some(true_streaming_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            true,
        ));
    }

    // Non-streaming quantized
    let generated = match quantized_model.generate_with_cache(&prompt_ids, &q_config) {
        Ok(g) => g,
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let text = match tokenizer.decode(&token_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        latency,
    ))
}

/// Convert usize token IDs to u32, returning error string on overflow
fn convert_token_ids(ids: &[usize]) -> Result<Vec<u32>, String> {
    ids.iter()
        .map(|&id| u32::try_from(id).map_err(|_| format!("Token ID {id} exceeds u32 range")))
        .collect()
}

/// Build generation config from request parameters
fn build_gen_config(request: &ChatCompletionRequest) -> GenerationConfig {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p };
    }
    config
}

/// Registry-based model fallback (no specialized backend).
fn registry_fallback(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
) -> Response {
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = match state.get_model(model_id) {
        Ok((m, t)) => (m, t),
        Err(e) => return fail_response(state, StatusCode::NOT_FOUND, e),
    };

    let prompt_text = format_chat_messages(&request.messages, Some(&request.model));
    let prompt_ids = tokenizer.encode(&prompt_text);
    if prompt_ids.is_empty() {
        return fail_response(state, StatusCode::BAD_REQUEST, "Messages cannot be empty");
    }

    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();
    let config = build_gen_config(request);

    let generated = match model.generate(&prompt, &config) {
        Ok(g) => g,
        Err(e) => return fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let token_ids: Vec<u32> = match convert_token_ids(&generated) {
        Ok(ids) => ids,
        Err(e) => return fail_response(state, StatusCode::BAD_REQUEST, e),
    };

    let generated_ids: Vec<u32> = token_ids[prompt.len()..].to_vec();
    let completion_tokens = generated_ids.len();

    if request.stream {
        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return pregenerated_sse_response(
            generated_ids,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            false,
        );
    }

    let response_text = match tokenizer.decode(&generated_ids) {
        Ok(t) => t,
        Err(e) => return fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let duration = start.elapsed();
    state.metrics.record_success(completion_tokens, duration);

    let max_tokens = request.max_tokens.unwrap_or(256);
    build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        None,
        duration,
    )
}

// ============================================================================
// Handlers
// ============================================================================

/// OpenAI-compatible models listing handler
///
/// Returns available models in OpenAI API format (GET /v1/models).
pub async fn openai_models_handler(State(state): State<AppState>) -> Json<OpenAIModelsResponse> {
    let models = if let Some(registry) = &state.registry {
        registry
            .list()
            .into_iter()
            .map(|m| OpenAIModel {
                id: m.id,
                object: "model".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0),
                owned_by: "realizar".to_string(),
            })
            .collect()
    } else {
        // Single model mode
        vec![OpenAIModel {
            id: "default".to_string(),
            object: "model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            owned_by: "realizar".to_string(),
        }]
    };

    Json(OpenAIModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

/// ALB-110: APR Q4K GPU chat backend via dedicated inference thread.
///
/// Mirrors `try_apr_q4k_completions` from gpu_completions_handler.rs but for
/// chat format. Tokenizes chat messages, sends to Q4K scheduler, returns
/// chat-formatted response.
#[cfg(feature = "cuda")]
async fn try_apr_q4k_chat_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::api::apr_q4k_scheduler::AprQ4kRequest;

    let q4k_tx = state.apr_q4k_tx()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, _eos_single) =
        chat_gen_params(request, &tokenizer, state.model_eos_token_id());
    let eos_ids = state.model_eos_ids();

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    if q4k_tx
        .send(AprQ4kRequest {
            prompt_ids,
            max_tokens,
            temperature,
            eos_ids,
            response_tx,
        })
        .await
        .is_err()
    {
        return Some(fail_response(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "Q4K thread unavailable",
        ));
    }

    let result = match response_rx.await {
        Ok(r) => r,
        Err(_) => {
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                "Q4K thread dropped response",
            ))
        }
    };

    let resp = match result {
        Ok(r) => r,
        Err(e) => {
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Q4K generation failed: {e}"),
            ))
        }
    };

    let text = match tokenizer.decode(&resp.output_tokens) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => {
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                e,
            ))
        }
    };
    let completion_tokens = resp.tokens_generated;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        start.elapsed(),
    ))
}

/// OpenAI-compatible /v1/chat/completions endpoint (supports streaming)
pub async fn openai_chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let start = Instant::now();
    // GH-152: Verbose request logging
    if state.is_verbose() {
        let msg_count = request.messages.len();
        let last_msg = request
            .messages
            .last()
            .map(|m| m.content.chars().take(50).collect::<String>())
            .unwrap_or_default();
        eprintln!(
            "[VERBOSE] POST /v1/chat/completions model={} messages={} last={:?}",
            request.model, msg_count, last_msg
        );
    }

    let trace_level = headers
        .get("X-Trace-Level")
        .and_then(|v| v.to_str().ok())
        .map(str::to_lowercase);

    let request_id = format!(
        "chatcmpl-q4k-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    #[cfg(feature = "gpu")]
    if let Some(r) = try_gpu_backend(&state, &request, &request_id, trace_level.as_deref(), start) {
        return r;
    }

    #[cfg(feature = "gpu")]
    if let Some(r) =
        try_cached_backend(&state, &request, &request_id, trace_level.as_deref(), start)
    {
        return r;
    }

    #[cfg(feature = "cuda")]
    if let Some(r) = try_cuda_backend(&state, &request, &request_id, trace_level.as_deref(), start).await
    {
        return r;
    }

    // ALB-110: APR Q4K GPU backend via dedicated inference thread
    #[cfg(feature = "cuda")]
    if let Some(r) = try_apr_q4k_chat_backend(&state, &request, &request_id, trace_level.as_deref(), start).await {
        return r;
    }

    // #169: SafeTensors CUDA backend (format parity)
    #[cfg(feature = "cuda")]
    if let Some(r) = try_safetensors_cuda_backend(&state, &request, &request_id, start) {
        return r;
    }

    if let Some(r) =
        try_quantized_backend(&state, &request, &request_id, trace_level.as_deref(), start)
    {
        return r;
    }

    registry_fallback(&state, &request, &request_id, start)
}
