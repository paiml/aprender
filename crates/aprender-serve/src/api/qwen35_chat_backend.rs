// #3571: /v1/chat/completions for the Qwen3.5 hybrid, served from the resident
// `Qwen35Session` the server built once at startup.

/// One generate on the resident session: the turn, with the stop token that
/// ends it kept from `on_token` -- it is not content, and with `top_n` (#4971)
/// neither is its logprobs record. A poisoned lock is refused, and `on_gpu`
/// records the route the turn ended on. Returned with the turn: the instant its
/// decode ended (#4954).
fn qwen35_generate(
    served: &crate::api::Qwen35Served,
    prompt: &[u32],
    config: &crate::gguf::QuantizedGenerateConfig,
    top_n: Option<usize>,
    on_token: &mut dyn FnMut(u32, Option<crate::gguf::logprobs::StepLogprobs>) -> bool,
) -> Result<(crate::session::Turn, Instant), String> {
    let mut s = served.session.lock().map_err(|_| POISONED.to_string())?;
    let turn = s.generate_with_logprobs(prompt, config, top_n, &mut |token, record| {
        config.stop_tokens.contains(&token) || on_token(token, record)
    });
    let decode_ended = Instant::now();
    served
        .on_gpu
        .store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
    turn.map(|turn| (turn, decode_ended))
        .map_err(|e| e.to_string())
}

/// The non-streaming Qwen3.5 reply (#4971, APR-OBS-001 OBS-09).
struct Qwen35Reply {
    /// The reply's tokens, without the stop token that ended it.
    tokens: Vec<u32>,
    /// With `top_n`, one logprobs record per reply token; else empty.
    steps: Vec<crate::gguf::logprobs::StepLogprobs>,
    /// What THIS turn ran on, measured by the session at its end (#3719).
    used_gpu: bool,
    /// What a step|layer trace measured over the turn.
    measured: TracedTurn,
    /// #4954: the turn's phase split, which the reply's `timings` report.
    phases: crate::api::PhaseTimings,
}

/// The non-streaming Qwen3.5 turn on the resident session: with `top_n`
/// (#4971) one logprobs record per reply token, the stop token's dropped with
/// it; with `tracer` (APR-OBS-001 OBS-09) the turn's events, and with
/// `time_layers` its per-layer time. A poisoned lock is refused, and `on_gpu`
/// records the route the turn ended on.
fn qwen35_turn(
    served: &crate::api::Qwen35Served,
    prompt: &[u32],
    config: &crate::gguf::QuantizedGenerateConfig,
    top_n: Option<usize>,
    mut tracer: Option<crate::inference_trace::InferenceTracer>,
    time_layers: bool,
) -> Result<Qwen35Reply, String> {
    let mut s = served.session.lock().map_err(|_| POISONED.to_string())?;
    let trace = tracer.as_mut().map(|tracer| crate::session::TurnTrace {
        tracer,
        layers: time_layers,
    });
    let mut steps = Vec::new();
    let turn = s.generate_traced_with_logprobs(
        prompt,
        config,
        top_n,
        &mut |token, record| {
            if !config.stop_tokens.contains(&token) {
                steps.extend(record);
            }
            true
        },
        trace,
    );
    let decode_ended = Instant::now();
    let layers = if time_layers {
        s.take_layer_timings()
    } else {
        None
    };
    let measured = TracedTurn {
        tracer,
        layers,
        num_layers: s.num_layers(),
    };
    served
        .on_gpu
        .store(s.on_gpu(), std::sync::atomic::Ordering::Relaxed);
    let turn = turn.map_err(|e| e.to_string())?;
    let mut reply = turn.tokens[prompt.len()..].to_vec();
    if reply.last().is_some_and(|t| config.stop_tokens.contains(t)) {
        reply.pop();
    }
    if top_n.is_some() && steps.len() != reply.len() {
        return Err(format!(
            "the reply's {} tokens came with {} logprobs records (#4971)",
            reply.len(),
            steps.len()
        ));
    }
    Ok(Qwen35Reply {
        tokens: reply,
        steps,
        used_gpu: turn.used_gpu,
        measured,
        phases: crate::api::PhaseTimings::from_marks(
            turn.prefill_started,
            turn.prefill_ended,
            decode_ended,
        ),
    })
}

/// Spawn the blocking generate that feeds the SSE stream, and with `top_n`
/// (#4971) the [`StreamLogprobs`](super::chat_logprobs::StreamLogprobs) its
/// records go to, each ahead of its token.
///
/// Extracted from `try_qwen35_backend` (#3844). This was the function's deepest
/// nesting -- `if` -> `spawn_blocking` closure -> `match lock()` -> `Ok` arm ->
/// `generate` callback closure -> `if let Err` -- and cognitive complexity counts
/// NESTING, which is why flattening the nine flat `unwrap_or` arms alone did not move
/// the number. The lock, the stop-token-ends-the-turn callback and the `on_gpu`
/// store are [`qwen35_generate`]'s; an error is forwarded down the channel.
///
/// #4954: a turn that completes sends its phase split down the returned
/// receiver, so the stream's terminal chunk carries `timings` (prompt_ms,
/// predicted_ms and the load and first-token edges). A failed turn drops the
/// sender: no timings.
fn spawn_streaming_generate(
    session: Arc<crate::api::Qwen35Served>,
    input_ids: Vec<u32>,
    gen_config: crate::gguf::QuantizedGenerateConfig,
    sink_metrics: Arc<crate::metrics::MetricsCollector>,
    top_n: Option<usize>,
) -> (
    tokio::sync::mpsc::Receiver<Result<u32, String>>,
    Option<super::chat_logprobs::StreamLogprobs>,
    tokio::sync::oneshot::Receiver<crate::api::PhaseTimings>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(64);
    let (timing_tx, timing_rx) = tokio::sync::oneshot::channel::<crate::api::PhaseTimings>();
    let (records, logprobs) = top_n
        .map(|_| super::chat_logprobs::StreamLogprobs::channel())
        .unzip();
    tokio::task::spawn_blocking(move || {
        let mut sink = crate::api::openai_handlers::streaming_token_sink(tx.clone(), sink_metrics);
        let result = qwen35_generate(
            &session,
            &input_ids,
            &gen_config,
            top_n,
            &mut |t, record| {
                // A record that cannot be sent means the response is gone, and
                // the sink sees that too and records the abandonment.
                if let (Some(records), Some(record)) = (&records, record) {
                    let _ = records.send(record);
                }
                sink(t)
            },
        );
        match result {
            Ok((turn, decode_ended)) => {
                let _ = timing_tx.send(crate::api::PhaseTimings::from_marks(
                    turn.prefill_started,
                    turn.prefill_ended,
                    decode_ended,
                ));
            },
            Err(e) => {
                let _ = tx.blocking_send(Err(e));
            },
        }
    });
    (rx, logprobs, timing_rx)
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
/// request's thinking mode (#3723) and with the request's tools (#4650), encoded by
/// the model.
///
/// The chat path and `POST /v1/chat/prompt-ids` both call this, so the ids that
/// endpoint reports are the ids this path prefills (PRM-S1 v2, #4354).
fn qwen35_prompt(
    model: &crate::gguf::GGUFModel,
    request: &ChatCompletionRequest,
    architecture: Option<&str>,
) -> Result<(String, Vec<u32>), String> {
    // #4650: the template's `# Tools` block is the only place the model learns a tool
    // exists and the `<tool_call>` format to call it in.
    let text = crate::api::realize_handlers::format_chat_messages_official_thinking_tools(
        Some(model),
        &request.messages,
        architecture,
        request.thinking(),
        request.tools.as_deref(),
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
    trace_level: Option<&str>,
    start: Instant,
    cancel: &CancelToken,
) -> Option<Response> {
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
    // #3723 thinking mode + template, via the same helper `POST /v1/chat/prompt-ids`
    // reports from (PRM-S1 v2), so the reported ids are the prefilled ids.
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

    // aprender#4345: the EOS and every EOG marker the vocabulary has.
    let stop_tokens = stop_tokens_unless_ignore_eos(
        request,
        crate::api::realize_handlers::completion_stop_tokens(
            &tokenizer,
            state.model_eos_token_id(),
        ),
    );
    // The context-bounded budget, not the request's number: what is decoded and what
    // `finish_reason` is judged against are the same count.
    let gen_config = gen_config_from_request(request, budget, stop_tokens, cancel.clone());
    // #4971: the session records each step's logprobs when they were asked for.
    let top_n = request.logprobs_top_n();

    // #5293: with a batch scheduler running, concurrent requests decode as one
    // batch, logprobs and all. One that asks for a trace stays on the session,
    // which traces it.
    #[cfg(feature = "cuda")]
    if let Some(batch_tx) = state.cuda_batch_tx().filter(|_| trace_level.is_none()) {
        let turn = Qwen35BatchTurn {
            input_ids,
            gen_config,
            top_n,
            budget,
            mapped,
            tokenizer,
            session,
        };
        let batch_tx = batch_tx.clone();
        return Some(qwen35_batched(state, request, request_id, start, batch_tx, turn).await);
    }

    if request.stream {
        let (rx, logprobs, timing_rx) =
            spawn_streaming_generate(session, input_ids, gen_config, state.metrics.clone(), top_n);
        return Some(crate::api::openai_handlers::live_sse_response(
            rx,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            budget,
            prompt_token_count,
            Some(timing_rx),
            request.stop.as_deref(),
            crate::api::stream_tool_calls::StreamTools::from_request(request),
            logprobs,
        ));
    }

    let decode_mapped = mapped.clone();
    // APR-OBS-001 OBS-09: a step|layer header runs the tracer over this turn.
    let tracer = crate::api::serve_trace::tracer_for(trace_level);
    let time_layers = trace_level == Some("layer");
    let turn = tokio::task::spawn_blocking(move || {
        qwen35_turn(
            &session,
            &input_ids,
            &gen_config,
            top_n,
            tracer,
            time_layers,
        )
    })
    .await;
    let reply = match turn {
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

    let completion_tokens = reply.tokens.len();
    let response_text = clean_chat_output(&decode_mapped.model.decode(&reply.tokens));
    let logprobs = top_n.map(|_| super::ChatLogprobs::from_steps(&tokenizer, &reply.steps));
    let measured = reply.measured;

    // #4954: the same phase split the stream reports; no stream, so no
    // first-content edge.
    let timings = reply
        .phases
        .to_timings_at(prompt_token_count, completion_tokens, start, None);
    let duration = start.elapsed();
    state.metrics.record_success(completion_tokens, duration);
    let traces = crate::api::serve_trace::traces_for(&crate::api::serve_trace::ServeTrace {
        level: trace_level,
        events: measured.tracer.as_ref().map_or(&[], |t| t.events()),
        layers: measured.layers.as_deref(),
        wall_us: u64::try_from(duration.as_micros()).unwrap_or(u64::MAX),
        prompt_tokens: prompt_token_count,
        completion_tokens,
        num_layers: measured.num_layers,
    });
    Some(build_chat_response_traced(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_token_count,
        completion_tokens,
        budget,
        request.stop.as_deref(),
        traces,
        request.tools.as_deref(),
        request_tool_choice(request),
        timings,
        // #3719: what THIS turn ran on, measured by the session at its end. A forced-GPU
        // turn that fell back to the CPU still answers 200; this is how the client
        // (`apr code`'s document) can tell.
        Some(reply.used_gpu),
        logprobs,
    ))
}

/// #5293: a Qwen3.5 chat turn bound for the batch scheduler.
#[cfg(feature = "cuda")]
struct Qwen35BatchTurn {
    input_ids: Vec<u32>,
    gen_config: crate::gguf::QuantizedGenerateConfig,
    /// #4971: the `top_logprobs` asked for, `None` when logprobs were not.
    top_n: Option<usize>,
    /// The context-bounded budget, as on the session path.
    budget: usize,
    mapped: Arc<crate::gguf::MappedGGUFModel>,
    tokenizer: Arc<BPETokenizer>,
    session: Arc<crate::api::Qwen35Served>,
}

/// #5293: `turn` through the batch scheduler (`try_qwen35_backend`'s batched
/// arm). A streamed reply goes out as its tokens arrive; otherwise every token
/// is collected first, through a channel with room for the whole budget, so
/// none is ever dropped. A full queue is the 503 the dense path gives.
#[cfg(feature = "cuda")]
async fn qwen35_batched(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    start: Instant,
    batch_tx: tokio::sync::mpsc::Sender<super::cuda_batch_scheduler::CudaBatchRequest>,
    turn: Qwen35BatchTurn,
) -> Response {
    let prompt_token_count = turn.input_ids.len();
    let capacity = if request.stream { 64 } else { turn.budget + 1 };
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(capacity);
    let (timing_tx, timing_rx) = tokio::sync::oneshot::channel::<crate::api::PhaseTimings>();
    // #4971: the engine's end of the logprobs records; a stream reads them as
    // its tokens go out, a reply collects them with its tokens.
    let (sink, stream_logprobs, mut collected) = match (turn.top_n, request.stream) {
        (None, _) => (None, None, None),
        (Some(top_n), true) => {
            let (records, logprobs) = super::chat_logprobs::StreamLogprobs::channel();
            let sink = super::cuda_batch_scheduler::RecordSink { top_n, records };
            (Some(sink), Some(logprobs), None)
        },
        (Some(top_n), false) => {
            let (records, collected) = tokio::sync::mpsc::unbounded_channel();
            let sink = super::cuda_batch_scheduler::RecordSink { top_n, records };
            (Some(sink), None, Some(collected))
        },
    };
    let batch_req = super::cuda_batch_scheduler::CudaBatchRequest {
        prompt_ids: turn.input_ids,
        config: turn.gen_config,
        token_tx: tx,
        non_streaming: false,
        enqueue_time: std::time::Instant::now(),
        timing_tx: Some(timing_tx),
        logprobs: sink,
    };
    if let Err(e) = batch_tx.try_send(batch_req) {
        state.record_admission_rejected();
        return fail_response(
            state,
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Batch queue full: {e}"),
        );
    }
    if request.stream {
        return crate::api::openai_handlers::live_sse_response(
            rx,
            turn.tokenizer,
            request_id.to_string(),
            request.model.clone(),
            state.metrics.clone(),
            start,
            turn.budget,
            prompt_token_count,
            Some(timing_rx),
            request.stop.as_deref(),
            crate::api::stream_tool_calls::StreamTools::from_request(request),
            stream_logprobs,
        );
    }
    let mut tokens = Vec::new();
    while let Some(result) = rx.recv().await {
        match result {
            Ok(token) => tokens.push(token),
            Err(e) => {
                state.metrics.record_failure();
                return fail_response(
                    state,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Qwen3.5 generation failed: {e}"),
                );
            },
        }
    }
    let completion_tokens = tokens.len();
    let response_text = clean_chat_output(&turn.mapped.model.decode(&tokens));
    // Every record went ahead of its token, so all are here once the tokens are.
    let logprobs = collected.as_mut().map(|records| {
        let mut steps = Vec::new();
        while let Ok(step) = records.try_recv() {
            steps.push(step);
        }
        super::ChatLogprobs::from_steps(&turn.tokenizer, &steps)
    });
    // A batched turn sends no phase split: there is none per request.
    let timings = timing_rx.await.ok().and_then(|phases| {
        phases.to_timings_at(prompt_token_count, completion_tokens, start, None)
    });
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());
    build_chat_response_traced(
        request_id.to_string(),
        request.model.clone(),
        response_text,
        prompt_token_count,
        completion_tokens,
        turn.budget,
        request.stop.as_deref(),
        (None, None, None),
        request.tools.as_deref(),
        request_tool_choice(request),
        timings,
        Some(
            turn.session
                .on_gpu
                .load(std::sync::atomic::Ordering::Relaxed),
        ),
        logprobs,
    )
}

/// What a traced Qwen3.5 turn measured, carried out of the blocking task.
struct TracedTurn {
    tracer: Option<crate::inference_trace::InferenceTracer>,
    layers: Option<Vec<crate::session::LayerTiming>>,
    num_layers: usize,
}

const POISONED: &str =
    "the Qwen3.5 session is unusable: an earlier request panicked while holding it (#3571)";

#[cfg(test)]
#[path = "qwen35_chat_backend_tests.rs"]
mod qwen35_serve_tests;
