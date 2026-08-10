
/// Dense f32 `Model` backend for `POST /stream/generate` (registry / safetensors).
///
/// Unchanged behaviour, lifted out of the handler so the quantized backend can be
/// tried first. Only the status for an unresolvable model moved: a server with no
/// usable model is 503, not 404 (see `model_resolution_status`).
fn dense_stream_tokens(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<(Vec<u32>, usize, std::sync::Arc<BPETokenizer>), ApiErr> {
    let (model, tokenizer) = state
        .get_model(request.model_id.as_deref())
        .map_err(|e| api_err(super::model_resolution_status(&e), e))?;

    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();
    let prompt_len = prompt.len();

    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        other => {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Invalid strategy: {other}"),
            ))
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);
    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    let generated = model
        .generate(&prompt, &config)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                api_err(
                    StatusCode::BAD_REQUEST,
                    format!("Token ID {id} exceeds u32 range"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((token_ids, prompt_len, tokenizer))
}

/// Quantized (GGUF / APR Q4_K) backend for `POST /stream/generate` and
/// `/realize/generate`.
///
/// aprender#2376(1, 10): this handler resolved the dense f32 `Model` via
/// `get_model()`, which is `None` on every `apr serve run model.gguf`, so a route
/// the startup banner advertises as "SSE streaming" answered 404
/// `"No model available"` while `/health` reported `model_loaded:true` and
/// `/generate` on the same process returned tokens.
///
/// Returns `Ok(None)` when no quantized model is resident, so the dense path below
/// is unchanged.
fn try_quantized_stream_tokens(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<(Vec<u32>, usize, std::sync::Arc<BPETokenizer>)>, ApiErr> {
    let quantized_model = match state.quantized_model() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_len = prompt_ids.len();

    let sampling = resolve_quantized_sampling(
        &request.strategy,
        request.top_k,
        request.top_p,
        request.temperature,
    )?;
    let q_config = quantized_config(
        state,
        &tokenizer,
        request.max_tokens,
        request.temperature,
        &sampling,
        request.seed,
    );

    let generated = quantized_model
        .generate_with_cache(&prompt_ids, &q_config)
        .map_err(|e| generation_err(&e))?;

    Ok(Some((generated, prompt_len, tokenizer)))
}

/// Stream generate handler — generates tokens one by one via Server-Sent Events.
///
/// Tries the quantized backend first (the `apr serve run model.gguf` path), then
/// the dense f32 `Model`.
pub async fn stream_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    // NOTE: Streaming via CUDA model uses /v1/chat/completions endpoint with stream=true
    // This handler uses the CPU model path; for GPU streaming use OpenAI-compatible endpoint

    let (token_ids, prompt_len, tokenizer_clone) =
        if let Some(resolved) = try_quantized_stream_tokens(&state, &request)? {
            resolved
        } else {
            dense_stream_tokens(&state, &request)?
        };

    // Create stream that emits tokens one by one
    let stream = async_stream::stream! {
        // Skip prompt tokens, only stream generated tokens. A backend that stops
        // on the first sampled token returns the prompt alone, so clamp rather
        // than slice past the end.
        let generated_start = prompt_len.min(token_ids.len());
        for &token_id in &token_ids[generated_start..] {
            // Decode single token
            let text = match tokenizer_clone.decode(&[token_id]) {
                Ok(t) => t,
                Err(_) => String::from("<error>"),
            };

            let event = StreamTokenEvent { token_id, text };
            // Serialization of simple struct should not fail, but handle gracefully
            let data = serde_json::to_string(&event)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());

            yield Ok::<_, Infallible>(Event::default().event("token").data(data));
        }

        // Send done event
        let done_event = StreamDoneEvent {
            num_generated: token_ids.len().saturating_sub(prompt_len),
        };
        // Serialization of simple struct should not fail, but handle gracefully
        let data = serde_json::to_string(&done_event)
            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());
        yield Ok(Event::default().event("done").data(data));
    };

    Ok(Sse::new(stream))
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
#[path = "gpu_handlers_tests.rs"]
mod gpu_handlers_tests;
