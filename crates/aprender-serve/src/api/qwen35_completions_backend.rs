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

/// One `/v1/completions` request, ready for the Qwen3.5 session: shared by the
/// buffered arm and the live stream (#4272), so the two cannot disagree on the
/// prompt, the budget or the stops.
struct Qwen35CompletionPlan {
    session: std::sync::Arc<crate::api::Qwen35Served>,
    mapped: std::sync::Arc<crate::gguf::MappedGGUFModel>,
    input_ids: Vec<u32>,
    prompt_tokens: usize,
    budget: usize,
    stop_tokens: Vec<u32>,
    gen_config: crate::gguf::QuantizedGenerateConfig,
}

/// `None` when this state serves no Qwen3.5 hybrid.
fn qwen35_completion_plan(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    cancel: &CancelToken,
) -> Result<Option<Qwen35CompletionPlan>, RErr> {
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
    Ok(Some(Qwen35CompletionPlan {
        session,
        mapped,
        input_ids,
        prompt_tokens,
        budget,
        stop_tokens,
        gen_config,
    }))
}

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
    let Some(Qwen35CompletionPlan {
        session,
        mapped,
        input_ids,
        prompt_tokens,
        budget,
        stop_tokens,
        gen_config,
    }) = qwen35_completion_plan(state, request, max_tokens, temperature, cancel)?
    else {
        return Ok(None);
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
    let mut response = completion_resp(
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
    );
    // SRV-TIM-001: the session measured the split; the counts are the response's.
    response.timings =
        super::PhaseTimings::from_turn(&turn).to_timings(prompt_tokens, completion_tokens);
    Ok(Some(response))
}

const POISONED: &str =
    "the Qwen3.5 session is unusable: an earlier request panicked while holding it (#3571)";

/// What the decode thread sends the live stream (#4272).
enum Qwen35StreamMsg {
    /// Text that is final: it can no longer be cut by a stop sequence.
    Delta(String),
    /// The turn ended.
    Done {
        finish_reason: String,
        completion_tokens: usize,
        /// SRV-TIM-001: the turn's measured split, with the counts above.
        timings: Option<super::Timings>,
    },
    /// The turn failed after the response had started.
    Failed(String),
}

/// The text of `generated` that is safe to send now, given `emitted` bytes
/// already sent: the decoded text, less a tail that could still grow into a
/// stop sequence, less an incomplete UTF-8 character. `None`: nothing new, or
/// the decode of the prefix does not extend what was sent (it is then sent at
/// the end, when the whole text is known).
pub(crate) fn qwen35_stream_delta(decoded: &str, emitted: usize, stops: &[String]) -> Option<String> {
    if decoded.ends_with('\u{FFFD}') || !decoded.is_char_boundary(emitted.min(decoded.len())) {
        return None;
    }
    let hold = stops.iter().map(|s| s.len().saturating_sub(1)).max().unwrap_or(0);
    let mut safe = decoded.len().saturating_sub(hold);
    while !decoded.is_char_boundary(safe) {
        safe -= 1;
    }
    (safe > emitted).then(|| decoded[emitted..safe].to_string())
}

/// `stream: true` on a Qwen3.5 session, LIVE (#4272): every chosen token goes
/// through the session's `on_token`, and the text it makes final is sent as a
/// chunk at once — the first chunk arrives after the prefill and one token, not
/// after the whole completion. The terminal chunk carries `finish_reason` and
/// `usage`. A client that goes away ends the turn at the next token.
///
/// `None` when this state serves no Qwen3.5 hybrid (the buffered chain then
/// answers, as before).
pub(crate) fn try_qwen35_completions_stream(
    state: &AppState,
    request: &CompletionRequest,
    cancel: &CancelToken,
) -> Result<Option<axum::response::Response>, RErr> {
    use axum::response::sse::{Event, Sse};
    use axum::response::IntoResponse;

    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7) as f32;
    let Some(plan) = qwen35_completion_plan(state, request, max_tokens, temperature, cancel)?
    else {
        return Ok(None);
    };
    let start = std::time::Instant::now();
    let stops: Vec<String> = request.stop.clone().unwrap_or_default();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Qwen35StreamMsg>();

    let Qwen35CompletionPlan {
        session,
        mapped,
        input_ids,
        prompt_tokens,
        budget,
        stop_tokens,
        gen_config,
    } = plan;
    tokio::task::spawn_blocking(move || {
        let Ok(mut s) = session.session.lock() else {
            let _ = tx.send(Qwen35StreamMsg::Failed(POISONED.to_string()));
            return;
        };
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let result = s.generate(&input_ids, &gen_config, &mut |token| {
            if stop_tokens.contains(&token) {
                return true; // the session ends the turn on it
            }
            generated.push(token);
            let decoded = mapped.model.decode(&generated);
            if stops.iter().any(|stop| decoded.contains(stop.as_str())) {
                return false; // apply_stop_sequences cuts it below
            }
            if let Some(delta) = qwen35_stream_delta(&decoded, emitted, &stops) {
                emitted += delta.len();
                if tx.send(Qwen35StreamMsg::Delta(delta)).is_err() {
                    return false; // the client went away
                }
            }
            true
        });
        session
            .on_gpu
            .store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
        drop(s);
        let turn = match result {
            Ok(turn) => turn,
            Err(e) => {
                let _ = tx.send(Qwen35StreamMsg::Failed(format!("Qwen3.5 generation failed: {e}")));
                return;
            },
        };
        let completion_tokens = generated.len();
        let (text, finish) = apply_stop_sequences(
            mapped.model.decode(&generated),
            Some(stops.as_slice()),
            completion_tokens,
            budget,
        );
        if text.len() > emitted && text.is_char_boundary(emitted) {
            let _ = tx.send(Qwen35StreamMsg::Delta(text[emitted..].to_string()));
        }
        let _ = tx.send(Qwen35StreamMsg::Done {
            finish_reason: finish.as_str().to_string(),
            completion_tokens,
            timings: super::PhaseTimings::from_turn(&turn)
                .to_timings(prompt_tokens, completion_tokens),
        });
    });

    let id = format!("cmpl-qwen35-{}", epoch_millis());
    let created = epoch_secs();
    let model = request.model.clone();
    let metrics = state.metrics.clone();
    let (log_id, log_model) = (id.clone(), request.model.clone());
    let chunk = move |text: String,
                      finish_reason: Option<String>,
                      usage: Option<Usage>,
                      timings: Option<super::Timings>| {
        CompletionChunk {
            id: id.clone(),
            object: "text_completion".to_string(),
            created,
            model: model.clone(),
            choices: vec![CompletionChunkChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason,
            }],
            usage,
            timings,
        }
    };
    let events = async_stream::stream! {
        while let Some(msg) = rx.recv().await {
            let data = match msg {
                Qwen35StreamMsg::Delta(text) => serde_json::to_string(&chunk(text, None, None, None)),
                Qwen35StreamMsg::Done { finish_reason, completion_tokens, timings } => {
                    metrics.record_success(completion_tokens, start.elapsed());
                    crate::api::request_log::emit(&crate::api::request_log::RequestRecord::new(
                        &log_id,
                        &log_model,
                        "completions",
                        Some(true),
                        true,
                        prompt_tokens,
                        completion_tokens,
                        timings.as_ref(),
                        start.elapsed(),
                        &finish_reason,
                    ));
                    serde_json::to_string(&chunk(
                        String::new(),
                        Some(finish_reason),
                        Some(Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                        }),
                        timings,
                    ))
                },
                Qwen35StreamMsg::Failed(e) => {
                    metrics.record_failure();
                    serde_json::to_string(&serde_json::json!({ "error": e }))
                },
            };
            if let Ok(data) = data {
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(data));
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Ok(Some(Sse::new(events).into_response()))
}
