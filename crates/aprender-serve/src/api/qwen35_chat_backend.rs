// #3571: /v1/chat/completions for the Qwen3.5 hybrid, served from the resident
// `Qwen35Session` the server built once at startup.

/// Spawn the blocking generate that feeds the SSE stream.
///
/// Extracted from `try_qwen35_backend` (#3844). This was the function's deepest
/// nesting -- `if` -> `spawn_blocking` closure -> `match lock()` -> `Ok` arm ->
/// `generate` callback closure -> `if let Err` -- and cognitive complexity counts
/// NESTING, which is why flattening the nine flat `unwrap_or` arms alone did not move
/// the number. Behaviour is unchanged: same lock, same stop-token-ends-the-turn
/// callback, same `on_gpu` store, same error forwarded down the channel.
fn spawn_streaming_generate(
    session: Arc<crate::api::Qwen35Served>,
    input_ids: Vec<u32>,
    gen_config: crate::gguf::QuantizedGenerateConfig,
    stop_tokens: Vec<u32>,
    tx: tokio::sync::mpsc::Sender<Result<u32, String>>,
    sink_metrics: Arc<crate::metrics::MetricsCollector>,
) {
    tokio::task::spawn_blocking(move || {
        let mut sink = crate::api::openai_handlers::streaming_token_sink(tx.clone(), sink_metrics);
        let result = match session.session.lock() {
            Ok(mut s) => {
                let r = s.generate(&input_ids, &gen_config, &mut |tok| {
                    // The stop token ends the turn; it is not content.
                    stop_tokens.contains(&tok) || sink(tok)
                });
                session
                    .on_gpu
                    .store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
                r.map(|_| ()).map_err(|e| e.to_string())
            },
            Err(_) => Err(POISONED.to_string()),
        };
        if let Err(e) = result {
            let _ = tx.blocking_send(Err(e));
        }
    });
}

/// Build the generate config from a chat request, with the CONTEXT-BOUNDED budget.
///
/// Extracted from `try_qwen35_backend` (#3844): nine `unwrap_or(defaults.*)` arms are
/// nine branches inside a function whose job is dispatch, which is what put it over the
/// complexity ratchet's cognitive ceiling of 25. `budget` arrives already bounded, so
/// what is decoded and what `finish_reason` is judged against remain the same count.
fn gen_config_from_request(
    request: &ChatCompletionRequest,
    budget: usize,
    stop_tokens: Vec<u32>,
    cancel: CancelToken,
) -> crate::gguf::QuantizedGenerateConfig {
    use crate::gguf::QuantizedGenerateConfig;
    let defaults = QuantizedGenerateConfig::default();
    QuantizedGenerateConfig {
        max_tokens: budget,
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_k: request.top_k.unwrap_or(defaults.top_k),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        repeat_penalty: request.repeat_penalty.unwrap_or(defaults.repeat_penalty),
        repeat_last_n: request.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        seed: request.seed.unwrap_or(defaults.seed),
        stop_tokens,
        cancel,
        ..defaults
    }
}

/// The Qwen3.5 hybrid's prompt for a chat request: the model's own template in the
/// request's thinking mode (#3723), encoded by the model.
///
/// The chat path and `POST /v1/chat/prompt-ids` both call this, so the ids that
/// endpoint reports are the ids this path prefills (PRM-S1 v2, #4354).
fn qwen35_prompt(
    model: &crate::gguf::GGUFModel,
    request: &ChatCompletionRequest,
    architecture: Option<&str>,
) -> Result<(String, Vec<u32>), String> {
    let text = crate::api::realize_handlers::format_chat_messages_official_thinking(
        Some(model),
        &request.messages,
        architecture,
        request.thinking(),
    )
    .map_err(|e| e.to_string())?;
    let ids = model.encode(&text).unwrap_or_default();
    Ok((text, ids))
}

/// `POST /v1/chat/prompt-ids`: the rendered prompt and its ids for a chat request
/// body, without generating (PRM-S1 v2, #4354).
///
/// A cross-engine replay must show both engines prefill the same ids before it
/// compares their speed. llama.cpp answers that with `/apply-template` + `/tokenize`;
/// apr's `/tokenize` is not the chat path's encoder, so this answers from the chat
/// path itself. Only the Qwen3.5 hybrid path is covered; any other backend is 501,
/// never ids some other path might have built.
pub async fn chat_prompt_ids_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let (Some(_), Some(mapped)) = (state.qwen35_session(), state.mapped_gguf_model()) else {
        return fail_response(
            &state,
            StatusCode::NOT_IMPLEMENTED,
            "prompt ids are reported only for the Qwen3.5 hybrid chat path (#4354)",
        );
    };
    let architecture = state.model_architecture();
    match qwen35_prompt(&mapped.model, &request, architecture.as_deref()) {
        Ok((prompt, ids)) => Json(serde_json::json!({
            "path": "qwen35",
            "prompt": prompt,
            "num_tokens": ids.len(),
            "prompt_ids": ids,
        }))
        .into_response(),
        Err(e) => fail_response(&state, StatusCode::BAD_REQUEST, e),
    }
}

/// The Qwen3.5 arm of the chat backend chain (#3571).
///
/// `None` when this state serves no hybrid, so the chain falls through
/// unchanged. Otherwise the request is answered here and nowhere else: the
/// state holds no dense model for the rest of the chain to decode through (see
/// [`AppState::with_qwen35_session`]).
///
/// The prompt is the chat template for the MODEL's architecture — not the
/// client's `model` string — encoded with the GGUF's own tokenizer, so a
/// request hands the model the tokens `apr chat` would. The session serves one
/// request at a time (a single-stream model, like the dense CUDA server), and a
/// request whose prompt extends the previous one's tokens reuses its state.
///
/// A prompt the declared context cannot hold is refused with 400 and both
/// numbers — never truncated. A reply the context cut short reports
/// `finish_reason: "length"`, because its budget is what the context left.
async fn try_qwen35_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
    cancel: &CancelToken,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let session = state.qwen35_session()?;
    let Some(mapped) = state.mapped_gguf_model() else {
        return Some(fail_response(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "the Qwen3.5 session has no retained GGUF to tokenize with (#3571)",
        ));
    };
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };

    let architecture = state.model_architecture();
    let input_ids = match qwen35_prompt(&mapped.model, request, architecture.as_deref()) {
        Ok((_, ids)) => ids,
        Err(e) => return Some(fail_response(state, StatusCode::BAD_REQUEST, e)),
    };
    if input_ids.is_empty() {
        return Some(fail_response(
            state,
            StatusCode::BAD_REQUEST,
            "Messages cannot be empty",
        ));
    }
    let prompt_token_count = input_ids.len();

    let context_length = session.context_length;
    if prompt_token_count >= context_length {
        return Some(fail_response(
            state,
            StatusCode::BAD_REQUEST,
            format!(
                "the prompt is {prompt_token_count} tokens and this model declares a context of \
                 {context_length}: it was refused whole rather than truncated (#3571)"
            ),
        ));
    }
    let max_tokens = request.max_tokens.unwrap_or(256);
    // What the context leaves — the budget the session will actually decode.
    let budget = max_tokens.min(context_length - prompt_token_count);

    let stop_tokens = stop_tokens_unless_ignore_eos(request, state.model_eos_token_id());
    // The context-bounded budget, not the request's number: what is decoded and what
    // `finish_reason` is judged against are the same count.
    let mut gen_config =
        gen_config_from_request(request, budget, stop_tokens.clone(), cancel.clone());
    // #4026: validated by the handler before any arm ran.
    let top_logprobs = crate::api::chat_logprobs::requested_top_logprobs(request)
        .ok()
        .flatten();
    if let Some(n) = top_logprobs {
        // At least one alternative is recorded: `logprobs_top_k` 0 records nothing.
        gen_config.logprobs_top_k = n.max(1);
    }

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(64);
        let sink_metrics = state.metrics.clone();
        spawn_streaming_generate(
            session,
            input_ids,
            gen_config,
            stop_tokens,
            tx,
            sink_metrics,
        );
        return Some(crate::api::openai_handlers::true_streaming_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            budget,
            prompt_token_count,
            None,
        ));
    }

    let decode_mapped = mapped.clone();
    let turn = tokio::task::spawn_blocking(move || match session.session.lock() {
        Ok(mut s) => {
            let r = s.generate(&input_ids, &gen_config, &mut |_| true);
            session.on_gpu.store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
            r.map_err(|e| e.to_string())
        },
        Err(_) => Err(POISONED.to_string()),
    })
    .await;
    let turn = match turn {
        Ok(Ok(turn)) => turn,
        Ok(Err(e)) => {
            state.metrics.record_failure();
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Qwen3.5 generation failed: {e}"),
            ));
        },
        Err(e) => {
            state.metrics.record_failure();
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Qwen3.5 generation task failed: {e}"),
            ));
        },
    };

    let mut generated_ids = turn.tokens[prompt_token_count..].to_vec();
    if generated_ids.last().is_some_and(|t| stop_tokens.contains(t)) {
        generated_ids.pop();
    }
    let completion_tokens = generated_ids.len();
    let response_text = clean_chat_output(&decode_mapped.model.decode(&generated_ids));

    let duration = start.elapsed();
    state.metrics.record_success(completion_tokens, duration);
    if let Some(n) = top_logprobs {
        // One record per generated id; the popped stop token's record goes too.
        let steps = &turn.steps[..completion_tokens.min(turn.steps.len())];
        let decode = |id: u32| decode_mapped.model.decode(&[id]);
        let mut body = serde_json::to_value(chat_response_body(
            request_id.to_string(),
            request.model.clone(),
            response_text,
            prompt_token_count,
            completion_tokens,
            budget,
            request.stop.as_deref(),
            None,
            duration,
            request.tools.as_deref(),
            request_tool_choice(request),
            None,
            None,
        ))
        .unwrap_or_default();
        body["choices"][0]["logprobs"] =
            crate::api::chat_logprobs::chat_logprobs_json(steps, n, &decode);
        return Some(Json(body).into_response());
    }
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_token_count,
        completion_tokens,
        budget,
        request.stop.as_deref(),
        None,
        duration,
        request.tools.as_deref(),
        request_tool_choice(request),
        None,
        None,
    ))
}

const POISONED: &str =
    "the Qwen3.5 session is unusable: an earlier request panicked while holding it (#3571)";

#[cfg(test)]
#[path = "qwen35_chat_backend_tests.rs"]
mod qwen35_serve_tests;
