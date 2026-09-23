// The Qwen3.5 arm of the realizar RAW routes: `/generate`, `/batch/generate`,
// `/stream/generate`, `/realize/generate`, `/realize/batch`.
//
// WHY THIS EXISTS. On every Qwen3.5 model those five routes answered
// `503 {"error":"Model registry error: No model available"}` while `/v1/completions`
// and every chat route answered 200 on the same server. Measured by aprender-83's
// CRUX serve sweep (lambda, Qwen3.5-2B-Q4_K_M, `apr serve --gpu`, 2026-09-23): 5 of
// the 11 generation routes `GET /` advertises. It is #3874's defect on a sibling
// chain: the raw handlers try cuda -> quantized -> apr -> the dense registry, and
// none of them reads `state.qwen35_session()`, so the chain ends at the registry.
//
// Like `try_qwen35_completions`, it is PURELY ADDITIVE: it sits ahead of the
// registry fallback, a state with no Qwen3.5 session returns `Ok(None)`, and every
// path it intercepts answered 503 before. The prompt is RAW — encoded by the GGUF's
// own tokenizer with no chat template, because these routes take a rendered prompt.

/// `(prompt ids ++ generated ids, prompt length)` for one raw prompt through the
/// Qwen3.5 session, or `None` when this state serves no Qwen3.5 hybrid.
fn qwen35_raw_tokens(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    cancel: &CancelToken,
) -> Result<Option<(Vec<u32>, usize)>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let Some(session) = state.qwen35_session() else {
        return Ok(None);
    };
    let Some(mapped) = state.mapped_gguf_model() else {
        return Err(api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the Qwen3.5 session has no retained GGUF to tokenize with (#3571)",
        ));
    };
    let input_ids = mapped.model.encode(prompt).unwrap_or_default();
    if input_ids.is_empty() {
        return Err(api_err(StatusCode::BAD_REQUEST, "Prompt cannot be empty"));
    }
    let prompt_len = input_ids.len();
    // Refused whole rather than truncated (#3571): a shortened prompt answers a
    // question the caller did not ask.
    if prompt_len >= session.context_length {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            format!(
                "the prompt is {prompt_len} tokens and this model declares a context of {}: \
                 it was refused whole rather than truncated (#3571)",
                session.context_length
            ),
        ));
    }
    let stop_tokens: Vec<u32> = state.model_eos_token_id().into_iter().collect();
    let gen_config = QuantizedGenerateConfig {
        max_tokens: max_tokens.min(session.context_length - prompt_len),
        temperature,
        top_k: crate::infer::sampling_top_k(temperature, None),
        stop_tokens: stop_tokens.clone(),
        trace: state.is_trace_enabled(),
        cancel: cancel.clone(),
        ..Default::default()
    };
    let mut s = session.session.lock().map_err(|_| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the Qwen3.5 session is unusable: an earlier request panicked while holding it (#3571)",
        )
    })?;
    let turn = s.generate(&input_ids, &gen_config, &mut |_| true);
    session
        .on_gpu
        .store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
    let mut tokens = turn
        .map_err(|e| {
            state.metrics.record_failure();
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Qwen3.5 generation failed: {e}"),
            )
        })?
        .tokens;
    if tokens.len() > prompt_len && tokens.last().is_some_and(|t| stop_tokens.contains(t)) {
        tokens.pop();
    }
    Ok(Some((tokens, prompt_len)))
}

/// `/generate` on a Qwen3.5 server.
fn try_qwen35_generate(
    state: &AppState,
    request: &GenerateRequest,
    cancel: &CancelToken,
) -> Result<Option<GenerateResponse>, ApiErr> {
    let Some((tokens, prompt_len)) = qwen35_raw_tokens(
        state,
        &request.prompt,
        request.max_tokens,
        request.temperature,
        cancel,
    )?
    else {
        return Ok(None);
    };
    let Some(mapped) = state.mapped_gguf_model() else {
        return Ok(None);
    };
    let generated = tokens[prompt_len.min(tokens.len())..].to_vec();
    Ok(Some(GenerateResponse {
        text: mapped.model.decode(&generated),
        num_generated: generated.len(),
        token_ids: tokens,
    }))
}

/// `/batch/generate` and `/realize/batch` on a Qwen3.5 server: one prompt at a
/// time, because the hybrid is a single-stream model.
fn try_qwen35_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
    cancel: &CancelToken,
) -> Result<Option<Vec<GenerateResponse>>, ApiErr> {
    if state.qwen35_session().is_none() {
        return Ok(None);
    }
    let mut results = Vec::with_capacity(request.prompts.len());
    for prompt in &request.prompts {
        let one = GenerateRequest {
            prompt: prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            strategy: request.strategy.clone(),
            top_k: request.top_k,
            top_p: request.top_p,
            seed: request.seed,
            model_id: None,
        };
        match try_qwen35_generate(state, &one, cancel)? {
            Some(r) => results.push(r),
            None => return Ok(None),
        }
    }
    Ok(Some(results))
}

/// `/stream/generate` and `/realize/generate` on a Qwen3.5 server: the ids, and
/// the server's tokenizer to decode the events with.
fn try_qwen35_stream_tokens(
    state: &AppState,
    request: &GenerateRequest,
    cancel: &CancelToken,
) -> Result<Option<(Vec<u32>, usize, std::sync::Arc<BPETokenizer>)>, ApiErr> {
    let Some((tokens, prompt_len)) = qwen35_raw_tokens(
        state,
        &request.prompt,
        request.max_tokens,
        request.temperature,
        cancel,
    )?
    else {
        return Ok(None);
    };
    let tokenizer = state.get_tokenizer(None).map_err(|e| {
        api_err(super::model_resolution_status(&e), e.to_string())
    })?;
    Ok(Some((tokens, prompt_len, tokenizer)))
}

/// A state that serves no Qwen3.5 hybrid is not this arm's: every entry declines,
/// so the raw chain reaches the registry exactly as before.
#[cfg(test)]
mod qwen35_raw_generate_tests {
    use super::*;

    #[test]
    fn a_state_without_a_hybrid_leaves_the_raw_chain_unchanged() {
        let state = AppState::demo().expect("demo state");
        assert!(state.qwen35_session().is_none(), "the fixture must hold no hybrid");
        let cancel = CancelToken::new();
        let one: GenerateRequest =
            serde_json::from_value(serde_json::json!({"prompt": "hi", "max_tokens": 4}))
                .expect("request");
        let many: BatchGenerateRequest =
            serde_json::from_value(serde_json::json!({"prompts": ["hi"], "max_tokens": 4}))
                .expect("request");
        assert!(matches!(try_qwen35_generate(&state, &one, &cancel), Ok(None)));
        assert!(matches!(try_qwen35_batch_generate(&state, &many, &cancel), Ok(None)));
        assert!(matches!(try_qwen35_stream_tokens(&state, &one, &cancel), Ok(None)));
    }
}
