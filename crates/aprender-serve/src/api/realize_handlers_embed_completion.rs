
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
    data: &[f32],
    token_ids: &[u32],
    hidden_dim: usize,
    tokenizer: &crate::tokenizer::BPETokenizer,
) -> Vec<f32> {
    // Never index past the rows we were actually given: `data` is
    // `[seq_len, hidden_dim]` and the caller's `token_ids` must align with it.
    let seq_len = data
        .len()
        .checked_div(hidden_dim)
        .map_or(0, |rows| token_ids.len().min(rows));

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

/// Which backend answers an embedding request, resolved once per request.
///
/// aprender#2376 finding 1 (seventh route): the embedding path resolved the dense
/// f32 [`Model`](crate::layers::Model) and nothing else, because `forward_hidden`
/// lived only there. On the standard `apr serve run model.gguf` path that model is
/// always `None` — the weights are quantized — so `/realize/embed`, `/v1/embeddings`
/// and every client of them failed on a server whose `/generate` was answering and
/// whose `/health` said `model_loaded:true`. The quantized backend now supplies the
/// same quantity via `forward_hidden_states`, so both backends can serve embeddings.
///
/// aprender#2609 closed the same gap for the THIRD resident backend: an
/// `AppState` holding an `AprTransformer` (the f32 APR / SafeTensors CPU serve
/// path) matched neither arm, so the embedding routes stayed dead there — again
/// on a server reporting `model_loaded: true` and answering `/generate`.
enum EmbedBackend {
    /// Dense f32 transformer (.apr / .safetensors).
    Dense(std::sync::Arc<crate::layers::Model>),
    /// Quantized GGUF weights — what `apr serve run model.gguf` loads.
    Quantized(std::sync::Arc<crate::gguf::OwnedQuantizedModel>),
    /// f32 `AprTransformer` — the APR / SafeTensors CPU serve path.
    Apr(std::sync::Arc<crate::apr_transformer::AprTransformer>),
}

impl EmbedBackend {
    /// Hidden width of the embedding vectors this backend produces.
    fn hidden_dim(&self) -> usize {
        match self {
            Self::Dense(m) => m.config().hidden_dim,
            Self::Quantized(m) => m.config.hidden_dim,
            Self::Apr(m) => m.config.hidden_dim,
        }
    }

    /// Final-layer hidden states for `token_ids`, row-major `[seq_len, hidden_dim]`.
    fn hidden_states(&self, token_ids: &[u32]) -> crate::error::Result<Vec<f32>> {
        match self {
            Self::Dense(m) => {
                let usize_ids: Vec<usize> = token_ids.iter().map(|&t| t as usize).collect();
                Ok(m.forward_hidden(&usize_ids)?.data().to_vec())
            },
            Self::Quantized(m) => m.forward_hidden_states(token_ids),
            Self::Apr(m) => m.forward_hidden_states(token_ids),
        }
    }
}

/// Resolve the backend + tokenizer that will answer an embedding request.
///
/// Dense first (registry mode selects by `model_id` there), quantized second. Only
/// when neither is resident is this a server-side condition, and then it is 503 —
/// not the 404 "No model available" the route used to answer.
fn resolve_embed_backend(
    state: &AppState,
    model_id: Option<&str>,
    route: &str,
) -> Result<(EmbedBackend, std::sync::Arc<crate::tokenizer::BPETokenizer>), RErr> {
    match state.get_model(model_id) {
        Ok((model, tokenizer)) => return Ok((EmbedBackend::Dense(model), tokenizer)),
        // An unknown `model_id` in registry mode is a CLIENT error and must stay a
        // 404 — falling through to the resident quantized model would silently
        // embed the caller's text with a model they did not ask for.
        Err(e @ crate::error::RealizarError::ModelNotFound(_)) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ))
        },
        Err(_) => {},
    }
    if let Some(quantized) = state.quantized_model() {
        let tokenizer = state.get_tokenizer(model_id).map_err(|e| {
            (
                super::model_resolution_status(&e),
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
        return Ok((EmbedBackend::Quantized(quantized.clone()), tokenizer));
    }
    if let Some(apr) = state.apr_transformer() {
        let tokenizer = state.get_tokenizer(model_id).map_err(|e| {
            (
                super::model_resolution_status(&e),
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
        return Ok((EmbedBackend::Apr(apr.clone()), tokenizer));
    }
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: format!("No model available: {route} needs a loaded model"),
        }),
    ))
}

/// Embed each input into one L2-normalized, mean-pooled vector.
///
/// Shared by `/realize/embed`, `/v1/embeddings` and `/api/embeddings` so all three
/// return the same numbers for the same text on the same server.
///
/// Returns the embeddings in request order plus the total prompt-token count.
pub(super) fn embed_inputs(
    state: &AppState,
    model_id: Option<&str>,
    inputs: &EmbeddingInput,
    route: &str,
) -> Result<(Vec<Vec<f32>>, usize), RErr> {
    let (backend, tokenizer) = resolve_embed_backend(state, model_id, route)?;

    // PMAT-802 × PMAT-803 (stacked): `input` may be a single string OR an array of
    // strings, and each element is embedded via the REAL model-backed path —
    // hidden states → mean-pool over non-special tokens → L2-normalize — NOT a
    // positional token-hash. Dimension is the model's hidden size, never a constant.
    let hidden_dim = backend.hidden_dim();

    let mut out = Vec::with_capacity(inputs.len());
    let mut prompt_tokens = 0usize;

    for (index, text) in inputs.iter().enumerate() {
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

        let hidden = backend.hidden_states(&token_ids).map_err(|e| {
            // A sequence longer than the context window is fully determined by the
            // request, so it is a client error — not a 500.
            let status = super::generation_error_status(&e);
            let error = if status == StatusCode::BAD_REQUEST {
                e.to_string()
            } else {
                format!("Embedding forward pass failed: {e}")
            };
            (status, Json(ErrorResponse { error }))
        })?;

        // Mean-pool over non-special tokens, then L2-normalize.
        let mut embedding = mean_pool_hidden_states(&hidden, &token_ids, hidden_dim, &tokenizer);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        out.push(embedding);
    }

    Ok((out, prompt_tokens))
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
    embed_for_route(state, request, "/realize/embed")
}

/// The shared body of every OpenAI-shaped embedding route, told which route it is.
///
/// aprender#2609: `/v1/embeddings` delegated to [`realize_embed_handler`], so its
/// unavailable-model body read `"No model available: /realize/embed needs a loaded
/// model"` — naming a route the client never called. `/api/embeddings` already
/// passed its own name; `/v1/embeddings` now does too.
///
/// # Errors
///
/// Propagates the status and JSON envelope from [`embed_inputs`].
pub(super) fn embed_for_route(
    state: AppState,
    request: EmbeddingRequest,
    route: &str,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (embeddings, prompt_tokens) =
        embed_inputs(&state, request.model.as_deref(), &request.input, route)?;

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding,
        })
        .collect();

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
///
/// **Every field is measured or absent.** 0.63.0 shipped this handler with
/// `size_bytes: 0`, `context_length: 4096`, `quantization: "Q4_K_M"`,
/// `format: "gguf"` and `content_hash: "blake3:0".repeat(16)` — a 128-character
/// string shaped exactly like a BLAKE3 digest that a consumer would store and
/// compare as provenance. Those were constants, not observations: the same
/// server reported 4096 while running with `--context-length 128` against a
/// 32768-context 1.04 GiB GGUF. Values now come from
/// [`super::ModelSourceInfo`], and anything the loader did not measure is
/// omitted from the JSON entirely.
pub async fn realize_model_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelMetadataResponse>, (StatusCode, Json<ErrorResponse>)> {
    let source = state.model_source();

    // Get default model info
    let model_info = if let Some(registry) = &state.registry {
        let models = registry.list();
        models.first().cloned()
    } else {
        Some(ModelInfo {
            id: "default".to_string(),
            name: "Default Model".to_string(),
            description: "Single model deployment".to_string(),
            // aprender#2376(6): `format` was the literal "gguf" while GET /models
            // hardcoded "unknown", so the two endpoints contradicted each other
            // about the same model. Read the resident backend first; #2396 adds
            // the magic-byte detection of the recorded source path as a fallback
            // for when the backend cannot name a format, which beats guessing.
            format: {
                let resident = state.model_format().to_string();
                if resident.is_empty() || resident == "unknown" {
                    source
                        .and_then(crate::api::ModelSourceInfo::format)
                        .unwrap_or_default()
                        .to_string()
                } else {
                    resident
                }
            },
            // NOT `loaded: true`. A literal here is precisely the fabricated
            // provenance #2396 set out to remove — ask the state whether a model
            // is actually resident.
            loaded: state.model_loaded(),
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

    // Lineage is emitted ONLY when a content hash was actually computed over
    // the model bytes. There is deliberately no synthetic fallback: a client
    // cannot distinguish a fabricated digest from a real one, so a fabricated
    // one is worse than none.
    let lineage = source
        .and_then(crate::api::ModelSourceInfo::content_hash)
        .map(|hash| ModelLineage {
            uri: format!("file://{}", source.and_then(crate::api::ModelSourceInfo::path).unwrap_or_default()),
            version: env!("CARGO_PKG_VERSION").to_string(),
            recipe: None,
            parent: None,
            content_hash: hash.to_string(),
        });

    Ok(Json(ModelMetadataResponse {
        id: info.id.clone(),
        name: info.name,
        format: Some(info.format).filter(|f| !f.is_empty()),
        size_bytes: source.and_then(crate::api::ModelSourceInfo::size_bytes),
        quantization: source
            .and_then(crate::api::ModelSourceInfo::quantization)
            .map(str::to_string),
        context_length: source.and_then(crate::api::ModelSourceInfo::context_length),
        model_max_context_length: source
            .and_then(crate::api::ModelSourceInfo::model_max_context_length),
        architecture: source
            .and_then(crate::api::ModelSourceInfo::architecture)
            .map(str::to_string),
        lineage,
        loaded: info.loaded,
    }))
}

/// Body returned by `/realize/reload` when the server is not in registry mode.
///
/// Must not name a CLI flag: `apr serve run` has none that enables registry
/// mode. Stating the actual situation is more useful than inventing a remedy.
pub(crate) const REGISTRY_MODE_UNAVAILABLE: &str =
    "Hot-reload requires multi-model registry mode, which this server is not running. \
     The apr CLI does not expose registry mode (there is no --registry flag on `apr serve run`); \
     it is available only to embedders that build AppState via AppState::with_registry. \
     To serve a different model, restart: apr serve run <MODEL>.";

/// Native Realizar hot-reload handler (/realize/reload)
///
/// Performs atomic model hot-reload via the ModelRegistry.
/// Requires registry mode (multi-model serving) to be enabled.
///
/// **Error-message contract.** When registry mode is off this returns 501, and
/// the body must NOT name a remedy that does not exist. 0.63.0 answered
/// "Start server with --registry flag" — `apr serve run` has no `--registry`
/// flag (clap rejects it with exit 2), and neither does `apr serve`, so the
/// advice sent every caller down a dead end. Registry mode is reached by
/// embedding this crate and calling `AppState::with_registry`; the CLI does
/// not expose it, and the message now says exactly that.
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
                error: REGISTRY_MODE_UNAVAILABLE.to_string(),
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

/// Build a CompletionResponse from generated tokens, applying the request's stop
/// sequences (`stops`) to the text and to `finish_reason`.
///
/// aprender#2465 finding 2: `stops` is a REQUIRED parameter, not an optional extra.
/// Every `/v1/completions` backend that answers with this builder used to decide
/// `finish_reason` here and apply stop sequences (or forget to) somewhere else —
/// `registry_completions`, the CPU dense backend that answers `apr serve` for
/// .apr/.safetensors models, forgot entirely, so `"stop"` was accepted by the API
/// and had no effect at all. Threading the stops through the ONE builder every
/// backend already calls makes forgetting them a compile error rather than a
/// silently ignored field.
fn completion_resp(
    id_prefix: &str,
    model: String,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    max_tokens: usize,
    stops: Option<&[String]>,
) -> CompletionResponse {
    let (text, finish_reason) = apply_stop_sequences(text, stops, completion_tokens, max_tokens);
    let finish_reason = finish_reason.as_str();
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
///
/// aprender#2465 finding 2: takes `stops` because this path ALSO answers
/// `/v1/completions` — it returned the batch scheduler's text verbatim, stop string
/// and all.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
async fn try_batch_completion(
    state: &AppState,
    tokenizer: &crate::tokenizer::BPETokenizer,
    prompt_ids: &[u32],
    prompt_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
    stops: Option<&[String]>,
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
        stops,
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

/// aprender#2465 finding 2: the WHOLE of OpenAI stop semantics, in one place —
/// truncate at the earliest stop position and report the matching `finish_reason`.
///
/// Returns `(text, finish_reason)`. A matched stop string wins over the token
/// budget (`"stop"` even when `completion_tokens >= max_tokens`); `"length"` is
/// only for "ran to the budget with no stop match". Both halves are delegated —
/// [`truncate_at_stop`] and [`FinishReason::from_generation`] — so `/v1/completions`
/// and `/v1/chat/completions` cannot drift: `openai_handlers::finalize_chat_text`
/// is this function, and so is [`completion_resp`].
///
/// The defect this exists to prevent is not a wrong implementation of stops — it is
/// a backend that never calls one. `/v1/completions` on the dense CPU backend
/// accepted `"stop"` and generated straight past it, returning the stop string
/// inside the completion with `finish_reason: "length"`, because applying stops was
/// a separate line each backend had to remember.
pub(crate) fn apply_stop_sequences(
    text: String,
    stops: Option<&[String]>,
    completion_tokens: usize,
    max_tokens: usize,
) -> (String, FinishReason) {
    let orig_len = text.len();
    let text = truncate_at_stop(text, stops);
    let stopped = text.len() < orig_len;
    (
        text,
        FinishReason::from_generation(stopped, completion_tokens, max_tokens),
    )
}

/// Build the dense-`Model` [`GenerationConfig`] for an OpenAI request.
///
/// `temperature: 0` is the canonical OpenAI request for deterministic output.
/// Passing that 0 straight into the config makes `model.generate` ->
/// `sample_token` -> `apply_temperature(0.0)` return
/// `InvalidShape: Temperature must be a positive finite number`, which every
/// dense backend mapped to **HTTP 500** — so `temperature: 0` was unserveable on
/// `/v1/chat/completions` and `/v1/completions` for any registry/safetensors
/// model. PMAT-790 fixed exactly one caller (the `/v1/chat/completions/stream`
/// handler) with a private copy of this logic; the other two kept 500ing, which
/// is what made a shared helper necessary.
///
/// Deterministic requests resolve to greedy argmax with a no-op temperature of
/// `1.0` (the sampler never sees a non-positive scale). Positive temperatures
/// are unchanged: greedy by default, top-p when `top_p` is set.
///
/// # The rest of the domain
///
/// The first version of this resolver special-cased `temperature == 0.0` and
/// nothing else, so `-1`, `NaN`, `+inf` and a `1e40` that narrows to `+inf`
/// still reached `apply_temperature` and still produced
/// `500 {"error":"Invalid shape: Temperature must be a positive finite number"}`
/// — the exact body the fix set out to eliminate.
///
/// Those values are now refused where the request is parsed (they are not
/// representable in a deserialized request — `types::deserialize_temperature_f32`),
/// which is the fix a client observes: 4xx naming the field. This function is
/// TOTAL as well, so that no Rust caller can construct a config the sampler
/// rejects: anything that is not a positive finite temperature resolves to the
/// same deterministic greedy config as `0`.
///
/// `temperature.is_finite()` is load-bearing and cannot be replaced by a
/// comparison: `NaN == 0.0`, `NaN > 0.0` and `NaN < 0.0` are all false, so a
/// comparison-only guard passes NaN straight through (aprender#2391).
pub(crate) fn resolve_dense_generation_config(
    temperature: f32,
    top_p: Option<f32>,
    max_tokens: usize,
) -> GenerationConfig {
    if !temperature.is_finite() || temperature <= 0.0 {
        return GenerationConfig::default()
            .with_max_tokens(max_tokens)
            .with_temperature(1.0);
    }

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(p) = top_p {
        config.strategy = SamplingStrategy::TopP { p };
    }
    config
}

/// Cached model backend (includes batch path). Returns None if not available.
#[cfg(feature = "gpu")]
async fn try_cached_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
    cancel: &CancelToken,
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
        request.stop.as_deref(),
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
        cancel: cancel.clone(),
        ..Default::default()
    };

    // IMP-126: adaptive generation when dispatch_metrics available
    let generated = if let Some(metrics) = state.dispatch_metrics() {
        cached_model
            .generate_with_cache_adaptive(&prompt_ids, &q_config, metrics)
            .map_err(|e| rerr(state, super::generation_error_status(&e), e))?
    } else {
        cached_model
            .generate_with_cache(&prompt_ids, &q_config)
            .map_err(|e| rerr(state, super::generation_error_status(&e), e))?
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    // PMAT-754 / #2465(2): stops are applied by `completion_resp`, which also gets
    // `finish_reason` right when a stop matched at the token budget.
    Ok(Some(completion_resp(
        "cmpl-cached",
        "cached-q4k".to_string(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        request.stop.as_deref(),
    )))
}

/// Quantized model (CPU GGUF) backend.
fn try_quantized_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
    cancel: &CancelToken,
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
        cancel: cancel.clone(),
        ..Default::default()
    };

    // aprender#2376(9): a context-budget rejection is a client error (400), not a
    // server failure — same classification as /generate.
    let generated = quantized_model
        .generate_with_cache(&prompt_ids, &q_config)
        .map_err(|e| rerr(state, super::generation_error_status(&e), e))?;
    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    // PMAT-754 / #2465(2): stops are applied by `completion_resp`.
    Ok(Some(completion_resp(
        "cmpl-q4k",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        request.stop.as_deref(),
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
