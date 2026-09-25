
/// Dense f32 `Model` backend for `POST /stream/generate` (registry / safetensors).
///
/// Unchanged behaviour, lifted out of the handler so the quantized backend can be
/// tried first. Only the status for an unresolvable model moved: a server with no
/// usable model is 503, not 404 (see `model_resolution_status`).
fn dense_stream_tokens(
    state: &AppState,
    request: &GenerateRequest,
    cancel: &CancelToken,
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
        .with_temperature(request.temperature)
        .with_cancel(cancel.clone());
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
    cancel: &CancelToken,
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
        cancel,
    );

    let generated = quantized_model
        .generate_with_cache(&prompt_ids, &q_config)
        .map_err(|e| generation_err(&e))?;

    Ok(Some((generated, prompt_len, tokenizer)))
}

/// `AprTransformer` (f32 APR / SafeTensors CPU) backend for `POST /stream/generate`
/// and `/realize/generate`.
///
/// aprender#2609: `/generate` and `/batch/generate` grew this backend
/// (`try_apr_generate`, `try_apr_batch_generate`) and `/stream/generate` did not,
/// so on an `AprTransformer` server the SSE route the startup banner advertises
/// answered `"No model available"` while `/generate` on the SAME process returned
/// real tokens and `/health` reported `model_loaded: true`. The backend chain here
/// is now the same one `/generate` walks: quantized, then APR, then dense.
///
/// Returns `Ok(None)` when no `AprTransformer` is resident, so the dense path is
/// unchanged.
fn try_apr_stream_tokens(
    state: &AppState,
    request: &GenerateRequest,
    cancel: &CancelToken,
) -> Result<Option<(Vec<u32>, usize, std::sync::Arc<BPETokenizer>)>, ApiErr> {
    use crate::apr_transformer::GenerateConfig;

    let apr_transformer = match state.apr_transformer() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_len = prompt_ids.len();

    let gen_config = GenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        cancel: cancel.clone(),
        ..Default::default()
    };

    let generated = apr_transformer
        .generate_with_cache(&prompt_ids, &gen_config)
        .map_err(|e| {
            api_err(
                super::generation_error_status(&e),
                format!("APR generation failed: {e}"),
            )
        })?;

    Ok(Some((generated, prompt_len, tokenizer)))
}

/// Cut a pregenerated token stream at the first special-token marker the
/// completion spells out (aprender#4344), keeping one event per token.
///
/// Every token before the marker is kept as is. The token the marker starts in
/// is kept with its text cut at the marker, or dropped if nothing precedes the
/// marker in it. Everything after is dropped. `/stream/generate` and
/// `/realize/generate` streamed `"<answer>7</answer><|im_end|>"` before this.
fn cut_pieces_at_marker(pieces: Vec<(u32, String)>) -> Vec<(u32, String)> {
    let full: String = pieces.iter().map(|(_, t)| t.as_str()).collect();
    let Some(cut) = crate::api::realize_handlers::first_special_marker(&full) else {
        return pieces;
    };
    let mut out = Vec::new();
    let mut offset = 0usize;
    for (id, mut text) in pieces {
        if offset >= cut {
            break;
        }
        let end = offset + text.len();
        if end > cut {
            text.truncate(cut - offset);
            if !text.is_empty() {
                out.push((id, text));
            }
            break;
        }
        offset = end;
        out.push((id, text));
    }
    out
}

/// Stream generate handler — generates tokens one by one via Server-Sent Events.
///
/// Tries the quantized backend first (the `apr serve run model.gguf` path), then
/// the `AprTransformer` (f32 APR / SafeTensors), then the dense f32 `Model`.
///
/// aprender#2376(3): the whole sequence is generated *before* the SSE stream is
/// built, so this handler's synchronous decode is exactly the work an abandoned
/// request used to keep doing. `cancel` (minted per request by
/// `cancel_on_disconnect`) is installed on the config so the loop stops at its
/// next token boundary once the client goes away.
pub async fn stream_generate_handler(
    State(state): State<AppState>,
    Extension(cancel): Extension<CancelToken>,
    Json(request): Json<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    // NOTE: Streaming via CUDA model uses /v1/chat/completions endpoint with stream=true
    // This handler uses the CPU model path; for GPU streaming use OpenAI-compatible endpoint

    let (token_ids, prompt_len, tokenizer_clone) =
        if let Some(resolved) = try_quantized_stream_tokens(&state, &request, &cancel)? {
            resolved
        } else if let Some(resolved) = try_apr_stream_tokens(&state, &request, &cancel)? {
            resolved
        } else {
            dense_stream_tokens(&state, &request, &cancel)?
        };

    // Create stream that emits tokens one by one
    let stream = async_stream::stream! {
        // Skip prompt tokens, only stream generated tokens. A backend that stops
        // on the first sampled token returns the prompt alone, so clamp rather
        // than slice past the end.
        let generated_start = prompt_len.min(token_ids.len());
        let pieces: Vec<(u32, String)> = token_ids[generated_start..]
            .iter()
            .map(|&id| (id, tokenizer_clone.decode(&[id]).unwrap_or_else(|_| String::from("<error>"))))
            .collect();
        for (token_id, text) in cut_pieces_at_marker(pieces) {
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
