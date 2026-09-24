// The Qwen3.5 arm of the `/v1/completions` chain (#3874).
//
// WHY THIS EXISTS. `/v1/completions` returned **503 for the entire Qwen3.5
// architecture**, on both backends, while `/api/chat` and `/v1/chat/completions`
// returned 200 for the same models. Measured by aprender-d8 on the 0.69.1 receipt:
// 503 on qwen35-0.8b, -2b, -4b and -9b, cpu and cuda alike; 200 on qwen2-1.5b.
//
// The cause was not a bug in any arm — it was a MISSING arm. `completions_inner`
// tries cached → quantized → gpu → apr_q4k → cuda_gguf → an inline cuda block →
// `registry_completions`, and **none of them reads `state.qwen35_session()`**.
// `try_qwen35_backend` (the chat equivalent) is called from exactly one place,
// `cuda_chat_backend.rs`. So on a Qwen3.5 model every completions arm declined and
// the chain terminated at "no model available".
//
// `qwen35_chat_backend.rs`'s own doc comment stated the consequence a year before
// anyone hit it — *"the state holds no dense model for the rest of the chain to
// decode through"* — accurately, next to the code, and nobody read it as a warning
// about a sibling endpoint.
//
// THIS ARM IS SIMPLER THAN THE CHAT ARM, deliberately. `Qwen35Session::generate`
// takes `&[u32]` token ids and the chat template is applied by the CALLER, not the
// session. So `/v1/completions` encodes `request.prompt` directly and never calls
// `format_chat_messages` — applying a chat template to a raw completion prompt
// would be a defect in its own right, not a convenience.
//
// IT IS PURELY ADDITIVE. It sits ahead of `registry_completions`, and every path it
// intercepts returns 503 today. It cannot regress anything that works: a state with
// no Qwen3.5 session returns `Ok(None)` and the chain falls through unchanged.

// Included into `gpu_completions_handler.rs`, which is itself included into
// `realize_handlers.rs` — the repo idiom for this chain. No `use` block: the arm
// shares the host's scope, exactly as its sibling arms do.

/// `None` when this state serves no Qwen3.5 hybrid — "not mine", never "handled
/// badly" — so a non-hybrid state falls through to the rest of the chain untouched.
pub(crate) async fn try_qwen35_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
    cancel: &CancelToken,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::gguf::QuantizedGenerateConfig;

    // The guard. Absence of a session is not an error here: it means another arm
    // owns this request.
    let Some(session) = state.qwen35_session() else {
        return Ok(None);
    };
    let Some(mapped) = state.mapped_gguf_model() else {
        return Err(rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "the Qwen3.5 session has no retained GGUF to tokenize with (#3571)",
        ));
    };

    // The RAW prompt, through the GGUF's own tokenizer. No chat template: this is
    // `/v1/completions`, and templating a completion prompt would change what the
    // caller asked for.
    let input_ids = mapped.model.encode(&request.prompt).unwrap_or_default();
    if input_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = input_ids.len();

    // Refused whole rather than truncated, with both numbers — the chat arm's rule
    // (#3571), and for the same reason: a silently shortened prompt answers a
    // question the caller did not ask.
    let context_length = session.context_length;
    if prompt_tokens >= context_length {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            format!(
                "the prompt is {prompt_tokens} tokens and this model declares a context of \
                 {context_length}: it was refused whole rather than truncated (#3571)"
            ),
        ));
    }
    // What the context leaves. Decoded budget and the count `finish_reason` is
    // judged against are the same number.
    let budget = max_tokens.min(context_length - prompt_tokens);

    let stop_tokens: Vec<u32> = state.model_eos_token_id().into_iter().collect();
    let gen_config = QuantizedGenerateConfig {
        max_tokens: budget,
        temperature,
        top_k: crate::infer::sampling_top_k(temperature, None),
        stop_tokens: stop_tokens.clone(),
        trace: state.is_trace_enabled(),
        cancel: cancel.clone(),
        ..Default::default()
    };

    let decode_mapped = mapped.clone();
    let turn = tokio::task::spawn_blocking(move || match session.session.lock() {
        Ok(mut s) => {
            let r = s.generate(&input_ids, &gen_config, &mut |_| true);
            session
                .on_gpu
                .store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
            r.map_err(|e| e.to_string())
        },
        Err(_) => Err(POISONED.to_string()),
    })
    .await;

    let turn = match turn {
        Ok(Ok(turn)) => turn,
        Ok(Err(e)) => {
            state.metrics.record_failure();
            return Err(rerr(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Qwen3.5 generation failed: {e}"),
            ));
        },
        Err(e) => {
            state.metrics.record_failure();
            return Err(rerr(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Qwen3.5 generation task failed: {e}"),
            ));
        },
    };

    let mut generated_ids = turn.tokens[prompt_tokens..].to_vec();
    if generated_ids.last().is_some_and(|t| stop_tokens.contains(t)) {
        generated_ids.pop();
    }
    let completion_tokens = generated_ids.len();
    let text = decode_mapped.model.decode(&generated_ids);

    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    // Stops are applied by `completion_resp`, as every other arm does.
    Ok(Some(completion_resp(
        "cmpl-qwen35",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        budget,
        request.stop.as_deref(),
        // MEASURED, not assumed: the blocking closure stored `s.on_gpu()` after the
        // generation, so this reads what the session actually did (#3894).
        state
            .qwen35_session()
            .map(|s| s.on_gpu.load(std::sync::atomic::Ordering::Relaxed)),
    )))
}

const POISONED: &str =
    "the Qwen3.5 session is unusable: an earlier request panicked while holding it (#3571)";
