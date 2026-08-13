
/// #169: SafeTensors CUDA backend — format parity with GGUF GPU path.
#[cfg(feature = "cuda")]
fn try_safetensors_cuda_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
    cancel: &CancelToken,
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

    // PMAT-756: this backend builds the chat JSON inline (not via build_chat_response) and
    // previously ignored request.stop entirely — the returned message kept the stop string
    // and ran to max_tokens. Reuse the shared finalize_chat_text helper so it matches every
    // other non-streaming chat backend (stop truncation + "stop" finish_reason precedence).
    let (output_text, finish_reason) =
        finalize_chat_text(output_text, request.stop.as_deref(), completion_tokens, max_tokens);

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
    cancel: &CancelToken,
) -> Option<Response> {
    // PMAT-821: config now built by chat_quantized_config (in openai_handlers.rs).
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
    // PMAT-821: build the config via chat_quantized_config so ALL request sampling
    // params (top_p/repeat_penalty/repeat_last_n/seed) reach the sampler — the prior
    // inline builder dropped them, leaving the chat endpoint on neutral defaults.
    let q_config = chat_quantized_config(
        request,
        &tokenizer,
        state.model_eos_token_id(),
        state.should_trace(trace_level),
        cancel,
    );
    let max_tokens = q_config.max_tokens;

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
            let sink_metrics = state.metrics.clone();

            tokio::task::spawn_blocking(move || {
                let mut cuda_model = cuda_model_clone.write().expect("operation failed");
                let result = cuda_model.generate_gpu_resident_streaming(
                    &prompt_ids_clone,
                    &q_config_clone,
                    // Stops when the client goes away — see `streaming_token_sink`.
                    crate::api::openai_handlers::streaming_token_sink(tx.clone(), sink_metrics),
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
            max_tokens,
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
        request.stop.as_deref(),
        trace_level,
        latency,
        request.tools.as_deref(),
        request_tool_choice(request),
    ))
}

/// Quantized model (GGUF serve mode) backend with true streaming.
fn try_quantized_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
    cancel: &CancelToken,
) -> Option<Response> {
    // PMAT-821: config now built by chat_quantized_config (in openai_handlers.rs).
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
    // PMAT-821: build the config via chat_quantized_config so ALL request sampling
    // params (top_p/repeat_penalty/repeat_last_n/seed) reach the sampler — the prior
    // inline builder dropped them, leaving the chat endpoint on neutral defaults.
    let q_config = chat_quantized_config(
        request,
        &tokenizer,
        state.model_eos_token_id(),
        state.should_trace(trace_level),
        cancel,
    );
    let max_tokens = q_config.max_tokens;

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(16);
        let quantized_model_clone = quantized_model.clone();
        let prompt_ids_clone = prompt_ids.clone();
        let q_config_clone = q_config.clone();
        let sink_metrics = state.metrics.clone();

        tokio::task::spawn_blocking(move || {
            let result = quantized_model_clone.generate_with_cache_streaming(
                &prompt_ids_clone,
                &q_config_clone,
                // Stops when the client goes away — see `streaming_token_sink`.
                crate::api::openai_handlers::streaming_token_sink(tx.clone(), sink_metrics),
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
            max_tokens,
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
        request.stop.as_deref(),
        trace_level,
        latency,
        request.tools.as_deref(),
        request_tool_choice(request),
    ))
}

/// Convert usize token IDs to u32, returning error string on overflow
fn convert_token_ids(ids: &[usize]) -> Result<Vec<u32>, String> {
    ids.iter()
        .map(|&id| u32::try_from(id).map_err(|_| format!("Token ID {id} exceeds u32 range")))
        .collect()
}

/// Build generation config from request parameters.
///
/// #2375: `temperature: 0` — the OpenAI-canonical deterministic request —
/// reached `apply_temperature` unchanged here and made this backend answer
/// HTTP 500 ("Temperature must be a positive finite number") for every dense
/// model. The resolution now lives in ONE place, shared with `/v1/completions`.
fn build_gen_config(request: &ChatCompletionRequest) -> GenerationConfig {
    crate::api::realize_handlers::resolve_dense_generation_config(
        request.temperature.unwrap_or(0.7),
        request.top_p,
        request.max_tokens.unwrap_or(256),
    )
}

/// Registry-based model fallback (no specialized backend).
fn registry_fallback(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
    cancel: &CancelToken,
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
    let config = build_gen_config(request).with_cancel(cancel.clone());

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
            request.stop.as_deref(),
            request.max_tokens.unwrap_or(256),
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
        request.stop.as_deref(),
        None,
        duration,
        request.tools.as_deref(),
        request_tool_choice(request),
    )
}

// ============================================================================
// Handlers
// ============================================================================

/// Process-wide model-load timestamp (Unix seconds).
///
/// CRUX-C-33 §created_timestamp_domain: `created` must represent model load
/// time — it MUST be stable across requests (not `SystemTime::now()` at each
/// call). First access latches the current wall clock; subsequent accesses
/// return the latched value. Discharges FALSIFY-CRUX-C-33-004.
fn model_loaded_at_unix_secs() -> i64 {
    static LOADED_AT: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    *LOADED_AT.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(1)
    })
}

/// OpenAI-compatible models listing handler
///
/// Returns available models in OpenAI API format (GET /v1/models).
/// Contract: `contracts/crux-C-33-v1.yaml` — envelope `{object:"list",data:[...]}`;
/// per-model `{id, object:"model", created>0, owned_by}`; `created` stable
/// across requests (model-load time, not request time).
pub async fn openai_models_handler(State(state): State<AppState>) -> Json<OpenAIModelsResponse> {
    let created = model_loaded_at_unix_secs();
    let models = if let Some(registry) = &state.registry {
        registry
            .list()
            .into_iter()
            .map(|m| OpenAIModel {
                id: m.id,
                object: "model".to_string(),
                created,
                owned_by: "realizar".to_string(),
            })
            .collect()
    } else {
        // Single model mode
        vec![OpenAIModel {
            id: "default".to_string(),
            object: "model".to_string(),
            created,
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
    // aprender#2376(3): this backend hands the work to the Q4K scheduler thread
    // rather than decoding here, so there is no in-handler loop to poll.
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
        request.stop.as_deref(),
        trace_level,
        start.elapsed(),
        request.tools.as_deref(),
        request_tool_choice(request),
    ))
}

/// OpenAI-compatible /v1/chat/completions endpoint (supports streaming)
pub async fn openai_chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(cancel): Extension<CancelToken>,
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

    if let Some(r) = try_qwen3_moe_backend(&state, &request, &request_id, start, &cancel) {
        return r;
    }

    #[cfg(feature = "gpu")]
    if let Some(r) = try_gpu_backend(
        &state,
        &request,
        &request_id,
        trace_level.as_deref(),
        start,
        &cancel,
    ) {
        return r;
    }

    #[cfg(feature = "gpu")]
    if let Some(r) = try_cached_backend(
        &state,
        &request,
        &request_id,
        trace_level.as_deref(),
        start,
        &cancel,
    ) {
        return r;
    }

    #[cfg(feature = "cuda")]
    if let Some(r) =
        try_cuda_backend(&state, &request, &request_id, trace_level.as_deref(), start, &cancel).await
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
    if let Some(r) = try_safetensors_cuda_backend(&state, &request, &request_id, start, &cancel) {
        return r;
    }

    if let Some(r) = try_quantized_backend(
        &state,
        &request,
        &request_id,
        trace_level.as_deref(),
        start,
        &cancel,
    ) {
        return r;
    }

    registry_fallback(&state, &request, &request_id, start, &cancel)
}

/// aprender#1789 Option B: qwen3_moe MoE-aware dispatch for /v1/chat/completions.
///
/// Detects qwen3_moe architecture + dispatches inference through
/// `run_qwen3_moe_generate` (the same path used by the `apr run` CLI),
/// which correctly indexes per-expert FFN tensors from the mmap.
///
/// For non-qwen3_moe archs returns `None` — handler falls through to the
/// dense backend chain (CUDA / cached / quantized / registry-fallback).
///
/// For qwen3_moe archs where AppState was constructed WITHOUT
/// `with_mapped_gguf_model` (no retained mmap), returns NOT_IMPLEMENTED
/// with the same actionable error class Option A surfaced. The
/// defensive guard from aprender#1790's `validate_matmul_weight_shape`
/// will NOT fire because we never reach the dense FFN matmul.
///
/// Discharges FALSIFY-QWEN3_MOE_SERVE_DISPATCH_V1_001 + V1_003 in
/// `contracts/qwen3-moe-serve-dispatch-v1.yaml`.
fn try_qwen3_moe_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
    cancel: &CancelToken,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let raw_arch = state.model_architecture()?;
    if !is_qwen3_moe_arch(&raw_arch) {
        return None;
    }

    let mapped = match state.mapped_gguf_model() {
        Some(m) => m,
        None => {
            eprintln!(
                "[WARN] aprender#1789: qwen3_moe arch detected at \
                 /v1/chat/completions (raw_arch={raw_arch}, canonical=qwen3_moe) \
                 but AppState has no retained MappedGGUFModel. This means the \
                 CLI server-command load path didn't call \
                 .with_mapped_gguf_model(). Returning NOT_IMPLEMENTED. \
                 See contracts/qwen3-moe-serve-dispatch-v1.yaml + \
                 https://github.com/paiml/aprender/issues/1789"
            );
            return Some(fail_response(
                state,
                StatusCode::NOT_IMPLEMENTED,
                "qwen3_moe arch detected but mapped GGUF not retained in AppState. \
                 See aprender#1789 + contracts/qwen3-moe-serve-dispatch-v1.yaml.",
            ));
        }
    };
    let quantized = match state.quantized_model() {
        Some(q) => q.clone(),
        None => {
            return Some(fail_response(
                state,
                StatusCode::NOT_IMPLEMENTED,
                "qwen3_moe arch detected but no OwnedQuantizedModel in AppState. \
                 See aprender#1789.",
            ));
        }
    };
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };

    let input_ids = match tokenize_chat_prompt(
        &tokenizer,
        &request.messages,
        Some(&request.model),
        state,
    ) {
        Ok(ids) => ids,
        Err(r) => return Some(r),
    };
    let prompt_token_count = input_ids.len();

    let max_tokens = request.max_tokens.unwrap_or(256).min(4096) as usize;
    // 3-knob toolkit (qwen3-moe-sampling-v1 + qwen3-moe-repetition-penalty-v1):
    // thread top_k/top_p/repeat_penalty/repeat_last_n/seed from the HTTP
    // request through to QuantizedGenerateConfig. Defaults match the dense
    // path's chat-completion behavior (greedy when unspecified).
    //
    // EOS stop-token: mirror the dense path's chat_gen_params fallback chain.
    // Generation halts on natural turn-end (model EOS or ChatML boundary).
    // Without this, qwen3_moe burns the full max_tokens budget per turn,
    // allowing self-prompted "Human:" runaway text — the root cause of
    // paiml/claude-code-parity-apr M287's verbosity pattern.
    //
    // Fallback order (matches chat_gen_params at openai_handlers.rs:97):
    //   1. state.model_eos_token_id() — from GGUF metadata
    //   2. tokenizer "<|im_end|>" — ChatML standard (Qwen, OpenHermes, Yi)
    //   3. tokenizer "<|endoftext|>" — GPT-style alternative
    //   4. None → empty stop_tokens (no behavior change from pre-fix)
    let defaults = QuantizedGenerateConfig::default();
    let eos_id = state.model_eos_token_id().or_else(|| {
        tokenizer
            .get_token_id("<|im_end|>")
            .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
    });
    let stop_tokens: Vec<u32> = eos_id.into_iter().collect();
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_k: request.top_k.unwrap_or(defaults.top_k),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        repeat_penalty: request.repeat_penalty.unwrap_or(defaults.repeat_penalty),
        repeat_last_n: request.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        seed: request.seed.unwrap_or(defaults.seed),
        stop_tokens,
        cancel: cancel.clone(),
        ..defaults
    };

    // qwen3-moe-streaming-sse-v1: per-token SSE when stream=true.
    // Dispatches to the callback variant + builds an SSE response from
    // a tokio mpsc channel. Non-streaming path falls through below.
    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(64);
        let mapped_clone = mapped.clone();
        let quantized_clone = quantized.clone();
        let input_ids_clone = input_ids.clone();
        let gen_config_clone = gen_config.clone();
        let sink_metrics = state.metrics.clone();

        tokio::task::spawn_blocking(move || {
            let result = crate::infer::qwen3_moe_generate::run_qwen3_moe_generate_streaming(
                &mapped_clone,
                &quantized_clone,
                &input_ids_clone,
                &gen_config_clone,
                // Stops when the client goes away — see `streaming_token_sink`.
                crate::api::openai_handlers::streaming_token_sink(tx.clone(), sink_metrics),
            );
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e.to_string()));
            }
        });

        return Some(crate::api::openai_handlers::true_streaming_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            max_tokens,
        ));
    }

    let tokens = match crate::infer::qwen3_moe_generate::run_qwen3_moe_generate(
        &mapped,
        &quantized,
        &input_ids,
        &gen_config,
    ) {
        Ok(t) => t,
        Err(e) => {
            state.metrics.record_failure();
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("qwen3_moe generation failed: {e}"),
            ));
        }
    };

    let generated_ids: Vec<u32> = tokens[input_ids.len()..].to_vec();
    let completion_tokens = generated_ids.len();

    // Apply clean_chat_output to strip self-emitted "Human:" / "User:" /
    // "<|im_end|>" / etc. prefixes from response text. Mirrors the dense
    // path at line 295 (PMAT-088). Without this, the M287 "Human: I need..."
    // runaway leaks into the chat response even after EOS detection.
    let response_text = match tokenizer.decode(&generated_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => {
            state.metrics.record_failure();
            return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e));
        }
    };

    let duration = start.elapsed();
    state.metrics.record_success(completion_tokens, duration);

    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_token_count,
        completion_tokens,
        max_tokens,
        request.stop.as_deref(),
        None,
        duration,
        request.tools.as_deref(),
        request_tool_choice(request),
    ))
}

/// Predicate: does this raw architecture string canonicalize to qwen3_moe?
///
/// Extracted for unit testing the dispatch classification independently of
/// the full handler. See contracts/qwen3-moe-serve-dispatch-v1.yaml.
fn is_qwen3_moe_arch(raw_arch: &str) -> bool {
    crate::tensor_names::normalize_architecture(raw_arch) == "qwen3_moe"
}

#[cfg(test)]
mod qwen3_moe_dispatch_guard_tests {
    use super::is_qwen3_moe_arch;

    #[test]
    fn canonical_qwen3_moe_matches() {
        assert!(is_qwen3_moe_arch("qwen3_moe"));
    }

    #[test]
    fn huggingface_class_names_canonicalize() {
        assert!(is_qwen3_moe_arch("Qwen3MoeForCausalLM"));
        assert!(is_qwen3_moe_arch("Qwen3MoEForCausalLM"));
        assert!(is_qwen3_moe_arch("Qwen3CoderForCausalLM"));
        assert!(is_qwen3_moe_arch("Qwen3_5MoeForCausalLM"));
        assert!(is_qwen3_moe_arch("Qwen3_5MoeForConditionalGeneration"));
    }

    #[test]
    fn lowercase_underscore_variants_match() {
        assert!(is_qwen3_moe_arch("qwen3moe"));
    }

    #[test]
    fn dense_archs_do_not_match() {
        // FALSIFY-QWEN3_MOE_SERVE_DISPATCH_V1_002 negative cases: the guard
        // MUST NOT fire for dense architectures, otherwise it regresses
        // every existing chat-completions request.
        assert!(!is_qwen3_moe_arch("qwen2"));
        assert!(!is_qwen3_moe_arch("qwen3"));
        assert!(!is_qwen3_moe_arch("llama"));
        assert!(!is_qwen3_moe_arch("mistral"));
        assert!(!is_qwen3_moe_arch("phi"));
        assert!(!is_qwen3_moe_arch("gemma"));
    }

    #[test]
    fn unknown_arch_does_not_match() {
        // normalize_architecture defaults unknowns to "llama", which is not
        // qwen3_moe — so the guard should NOT fire for unknown archs.
        assert!(!is_qwen3_moe_arch("some-future-arch-3000"));
        assert!(!is_qwen3_moe_arch(""));
    }
}
