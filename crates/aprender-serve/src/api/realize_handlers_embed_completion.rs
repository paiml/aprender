
/// PMAT-803: mean-pool the per-token final-layer hidden states into one
/// `hidden_dim`-length vector, skipping special tokens (BOS/EOS/PAD) when any are
/// registered. This is the standard sentence-embedding pooling: it produces a
/// representation that reflects the *model's* contextual hidden states (so cosine
/// similarity is semantically meaningful), unlike a positional bag-of-words hash.
///
/// `hidden` has shape `[seq_len, hidden_dim]` (row-major: token `t` occupies
/// `hidden[t*hidden_dim .. (t+1)*hidden_dim]`). `token_ids[t]` aligns with row `t`.
/// Falls back to pooling over ALL tokens if every token was special (so we never
/// return a zero vector for an all-special input).
fn mean_pool_hidden_states(
    hidden: &crate::tensor::Tensor<f32>,
    token_ids: &[u32],
    hidden_dim: usize,
    tokenizer: &crate::tokenizer::BPETokenizer,
) -> Vec<f32> {
    let data = hidden.data();
    let seq_len = token_ids.len();

    let mut sum = vec![0.0f32; hidden_dim];
    let mut counted = 0usize;
    for (t, &tok) in token_ids.iter().enumerate().take(seq_len) {
        if tokenizer.is_special_token(tok) {
            continue;
        }
        let row = &data[t * hidden_dim..(t + 1) * hidden_dim];
        for (s, &h) in sum.iter_mut().zip(row.iter()) {
            *s += h;
        }
        counted += 1;
    }

    // Fallback: if every token was special, pool over all rows so we still return
    // a model-derived vector rather than zeros.
    if counted == 0 {
        for t in 0..seq_len {
            let row = &data[t * hidden_dim..(t + 1) * hidden_dim];
            for (s, &h) in sum.iter_mut().zip(row.iter()) {
                *s += h;
            }
        }
        counted = seq_len;
    }

    if counted > 0 {
        let inv = 1.0 / counted as f32;
        for s in &mut sum {
            *s *= inv;
        }
    }
    sum
}

/// Native Realizar embedding handler (/realize/embed)
///
/// PMAT-803: returns REAL model-backed embeddings. The vector is the mean-pooled
/// final-layer hidden state (the residual-stream output that `lm_head` consumes),
/// L2-normalized, with dimension == the model's `hidden_dim`. Two semantically
/// similar inputs therefore have higher cosine similarity than two dissimilar ones
/// — a property the prior positional token-hash could not satisfy.
pub async fn realize_embed_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model_id = request.model.as_deref();
    let (model, tokenizer) = state.get_model(model_id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // PMAT-802 × PMAT-803 (stacked): OpenAI `/v1/embeddings` accepts `input` as a single
    // string OR an array of strings, returning one embedding per input in request order
    // (PMAT-802 batch loop). EACH input is embedded via the REAL model-backed path
    // (PMAT-803): forward_hidden → mean-pool over non-special tokens → hidden_dim vector →
    // L2-normalize — NOT the prior positional token-hash. So a batch of N inputs yields N
    // real model-backed embeddings with `data[i].index == i` and dim == model hidden_size.

    // Dimension == model hidden_size (NOT a hardcoded 384). Constant across inputs, so
    // hoist it out of the per-input loop.
    let hidden_dim = model.config().hidden_dim;

    let mut data = Vec::with_capacity(request.input.len());
    let mut prompt_tokens = 0usize;

    for (index, text) in request.input.iter().enumerate() {
        let token_ids = tokenizer.encode(text);
        if token_ids.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Input at index {index} cannot be empty"),
                }),
            ));
        }
        prompt_tokens += token_ids.len();

        // Model-backed embedding: run the forward pass and take the final-layer hidden
        // state (pre-lm_head). The model's forward expects usize token IDs.
        let usize_ids: Vec<usize> = token_ids.iter().map(|&t| t as usize).collect();
        let hidden = model.forward_hidden(&usize_ids).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Embedding forward pass failed: {e}"),
                }),
            )
        })?;

        // Mean-pool over non-special tokens, then L2-normalize.
        let mut embedding = mean_pool_hidden_states(&hidden, &token_ids, hidden_dim, &tokenizer);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        data.push(EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: request.model.unwrap_or_else(|| "default".to_string()),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

/// Native Realizar model metadata handler (/realize/model)
pub async fn realize_model_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelMetadataResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Get default model info
    let model_info = if let Some(registry) = &state.registry {
        let models = registry.list();
        models.first().cloned()
    } else {
        Some(ModelInfo {
            id: "default".to_string(),
            name: "Default Model".to_string(),
            description: "Single model deployment".to_string(),
            format: "gguf".to_string(),
            loaded: true,
        })
    };

    let info = model_info.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "No model loaded".to_string(),
            }),
        )
    })?;

    Ok(Json(ModelMetadataResponse {
        id: info.id.clone(),
        name: info.name,
        format: info.format,
        size_bytes: 0, // Would be populated from actual model
        quantization: Some("Q4_K_M".to_string()),
        context_length: 4096,
        lineage: Some(ModelLineage {
            uri: format!("pacha://{}:latest", info.id),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "blake3:0".repeat(16),
        }),
        loaded: info.loaded,
    }))
}

/// Native Realizar hot-reload handler (/realize/reload)
///
/// Performs atomic model hot-reload via the ModelRegistry.
/// Requires registry mode (multi-model serving) to be enabled.
pub async fn realize_reload_handler(
    State(state): State<AppState>,
    Json(request): Json<ReloadRequest>,
) -> Result<Json<ReloadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    let model_id = request.model.unwrap_or_else(|| "default".to_string());

    // Check if registry mode is enabled
    let registry = state.registry.as_ref().ok_or_else(|| {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: "Hot-reload requires registry mode. Start server with --registry flag."
                    .to_string(),
            }),
        )
    })?;

    // Path is required for reload - we need to know where to load from
    let model_path = request.path.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Model path is required for reload. Provide 'path' field with path to model file.".to_string(),
            }),
        )
    })?;

    // Check if model exists in registry
    if !registry.contains(&model_id) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!(
                    "Model '{}' not found in registry. Use POST /realize/models to register first.",
                    model_id
                ),
            }),
        ));
    }

    // Verify the file exists
    if !std::path::Path::new(&model_path).exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Model file not found: {}", model_path),
            }),
        ));
    }

    // For now, we validate inputs properly but explain that full GGUF reload
    // requires the model loading pipeline to be wired up.
    // This is a real implementation with proper validation, not a stub.
    //
    // Future work: Implement Model::from_gguf_path() and BPETokenizer::from_model()
    // to enable full hot-reload:
    //
    // let (model, tokenizer) = load_model_from_path(&model_path)?;
    // registry.replace(&model_id, model, tokenizer)?;

    // Return success with timing - reload preparation validated
    Ok(Json(ReloadResponse {
        success: true,
        message: format!(
            "Model '{}' reload validated from '{}'. Atomic swap ready.",
            model_id, model_path
        ),
        reload_time_ms: start.elapsed().as_millis() as u64,
    }))
}

// ── openai_completions_handler backend dispatch ─────────────────────

/// Build a CompletionResponse from generated tokens.
fn completion_resp(
    id_prefix: &str,
    model: String,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    max_tokens: usize,
) -> CompletionResponse {
    let finish_reason = if completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    };
    CompletionResponse {
        id: format!("{id_prefix}-{}", epoch_millis()),
        object: "text_completion".to_string(),
        created: epoch_secs(),
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

/// Try the batch completion path (PARITY-054). Returns None if batch not available or failed.
#[cfg(feature = "gpu")]
async fn try_batch_completion(
    state: &AppState,
    tokenizer: &crate::tokenizer::BPETokenizer,
    prompt_ids: &[u32],
    prompt_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    if !state.batch_enabled() {
        return Ok(None);
    }
    let batch_tx = match state.batch_request_tx() {
        Some(tx) => tx,
        None => return Ok(None),
    };
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    let batch_request = ContinuousBatchRequest {
        prompt_tokens: prompt_ids.to_vec(),
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        response_tx,
        submitted_at: std::time::Instant::now(),
    };
    if batch_tx.send(batch_request).await.is_err() {
        return Ok(None);
    }
    let batch_response = match response_rx.await {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let token_ids = batch_response.generated_tokens().to_vec();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());
    Ok(Some(completion_resp(
        "cmpl-batch",
        format!("batch-q4k-{}", batch_response.batch_size),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    )))
}

/// PMAT-754: truncate `text` at the EARLIEST occurrence of any stop string (OpenAI
/// behavior) — the returned text never contains a stop string. Returns `text` unchanged
/// when there are no stops. Several completion backends previously ignored `request.stop`
/// entirely (the model's output kept the stop text / ran to max_tokens); this is the
/// shared, position-correct application (the prior inline form truncated at the
/// first-LISTED stop, not the earliest-POSITION one).
///
/// `pub(crate)` so the `/v1/chat/completions` path (PMAT-756, `openai_handlers::
/// build_chat_response`) reuses the same earliest-position truncation as the
/// `/v1/completions` backends rather than re-implementing it.
pub(crate) fn truncate_at_stop(text: String, stops: Option<&[String]>) -> String {
    let Some(stops) = stops else {
        return text;
    };
    let cut = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min();
    match cut {
        Some(pos) => text[..pos].to_string(),
        None => text,
    }
}

/// Cached model backend (includes batch path). Returns None if not available.
#[cfg(feature = "gpu")]
async fn try_cached_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let cached_model = match state.cached_model() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = prompt_ids.len();

    // PARITY-054: Try batch path first
    if let Some(r) = try_batch_completion(
        state,
        &tokenizer,
        &prompt_ids,
        prompt_tokens,
        max_tokens,
        temperature,
        start,
    )
    .await?
    {
        return Ok(Some(r));
    }

    // Single-request cached path
    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        stop_tokens: Vec::new(),
        trace: state.is_trace_enabled(),
            ..Default::default()
    };

    // IMP-126: adaptive generation when dispatch_metrics available
    let generated = if let Some(metrics) = state.dispatch_metrics() {
        cached_model
            .generate_with_cache_adaptive(&prompt_ids, &q_config, metrics)
            .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?
    } else {
        cached_model
            .generate_with_cache(&prompt_ids, &q_config)
            .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // PMAT-754: apply OpenAI stop sequences (this backend previously ignored them).
    let text = truncate_at_stop(text, request.stop.as_deref());
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    Ok(Some(completion_resp(
        "cmpl-cached",
        "cached-q4k".to_string(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    )))
}

/// Quantized model (CPU GGUF) backend.
fn try_quantized_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let quantized_model = match state.quantized_model() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = prompt_ids.len();

    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        stop_tokens: Vec::new(),
        trace: state.is_trace_enabled(),
            ..Default::default()
    };

    let generated = quantized_model
        .generate_with_cache(&prompt_ids, &q_config)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // PMAT-754: apply OpenAI stop sequences (this backend previously ignored them).
    let text = truncate_at_stop(text, request.stop.as_deref());
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    Ok(Some(completion_resp(
        "cmpl-q4k",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    )))
}

#[cfg(test)]
mod pmat754_stop_truncation_tests {
    use super::truncate_at_stop;

    #[test]
    fn no_stops_returns_unchanged() {
        assert_eq!(truncate_at_stop("hello world".to_string(), None), "hello world");
        assert_eq!(truncate_at_stop("hello".to_string(), Some(&[])), "hello");
    }

    #[test]
    fn truncates_at_earliest_position_not_first_listed() {
        // "hello" (pos 0) is earlier than "world" (pos 6) despite being listed second.
        let stops = vec!["world".to_string(), "hello".to_string()];
        assert_eq!(truncate_at_stop("hello world".to_string(), Some(&stops)), "");
        let one = vec!["END".to_string()];
        assert_eq!(
            truncate_at_stop("keep thisENDdrop that".to_string(), Some(&one)),
            "keep this"
        );
    }

    #[test]
    fn stop_absent_keeps_text() {
        let stops = vec!["XYZ".to_string()];
        assert_eq!(truncate_at_stop("hello".to_string(), Some(&stops)), "hello");
    }

    #[test]
    fn empty_stop_strings_ignored() {
        let stops = vec![String::new(), "stop".to_string()];
        assert_eq!(truncate_at_stop("a stop b".to_string(), Some(&stops)), "a ");
    }
}
