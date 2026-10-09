// #3568 PR 4: OpenAI's `response_format` on /v1/chat/completions, through the one engine.
//
// A request whose `response_format` is `json_schema` or `json_object` is answered here or
// refused here, by name. It takes the engine (`Session::generate_constrained`) on exactly the
// backend the unconstrained request would take: the Qwen3.5 hybrid's resident session, the
// dense CUDA model, or the dense CPU model. Every other backend in the chain applies no
// constraint, so a constrained request there is refused (`SchemaUnsupportedPath`) before any
// model work: never answered unconstrained, never moved to another backend (a quiet switch is
// the fallback #3711 forbids). The finished text is then read again by the second reader
// (`jsonschema`, which shares nothing with the engine and fetches nothing) before it ships.

use crate::constrain::OUTSIDE_3568;

/// The status and the `type` a constraint's refusal answers with, keyed on the variant, never
/// on its text: the request's fault is 4xx, the server's is 5xx.
fn constraint_status(e: &crate::constrain::ConstraintError) -> (StatusCode, &'static str) {
    use crate::constrain::ConstraintError as E;
    match e {
        E::SchemaInvalid(_) => (StatusCode::BAD_REQUEST, "schema_invalid"),
        E::SchemaUnsupported(_) => (StatusCode::BAD_REQUEST, "schema_unsupported"),
        E::UnsupportedPath { .. } => (StatusCode::BAD_REQUEST, "schema_unsupported_path"),
        E::WithThinking(_) => (StatusCode::BAD_REQUEST, "schema_with_thinking"),
        // The budget the request set ran out: no partial document is returned
        E::Truncated { .. } => (StatusCode::UNPROCESSABLE_ENTITY, "constraint_truncated"),
        E::Violation(_) => (StatusCode::INTERNAL_SERVER_ERROR, "schema_violation"),
        E::DeadEnd { .. } => (StatusCode::INTERNAL_SERVER_ERROR, "constraint_dead_end"),
        E::Rejected { .. } => (StatusCode::INTERNAL_SERVER_ERROR, "constraint_rejected"),
        E::Vocab(_) => (StatusCode::INTERNAL_SERVER_ERROR, "constraint_vocab"),
        E::NotCompiled => (
            StatusCode::NOT_IMPLEMENTED,
            "structured_output_not_compiled",
        ),
    }
}

/// A constraint's refusal as the response: its status, its one line, its `type`.
fn constraint_refusal(state: &AppState, e: &crate::constrain::ConstraintError) -> Response {
    state.metrics.record_failure();
    let (status, kind) = constraint_status(e);
    (
        status,
        Json(serde_json::json!({ "error": e.to_string(), "type": kind })),
    )
        .into_response()
}

/// `SchemaUnsupportedPath` for `path`.
fn unsupported_path(path: &str, removed_by: &str) -> crate::constrain::ConstraintError {
    crate::constrain::ConstraintError::UnsupportedPath {
        path: path.to_string(),
        removed_by: removed_by.to_string(),
    }
}

/// The parts of a chat request the engine cannot honour under a constraint, refused by name
/// before any model work: answering them unconstrained and dropping them are both lies.
fn refuse_unconstrainable(
    request: &ChatCompletionRequest,
    traced: bool,
) -> Result<(), crate::constrain::ConstraintError> {
    if request.thinking() == Some(true) {
        return Err(crate::constrain::ConstraintError::WithThinking(
            "the request turns thinking on, so the model would reason before it answers; a \
             constraint applies to no-think output only (#3735 constrains after </think>)"
                .to_string(),
        ));
    }
    let refused = [
        (
            request.stream,
            "serve-stream",
            "not scheduled: a streamed reply leaves before the second reader can read it \
             whole; send stream: false",
        ),
        (
            request.tools.as_ref().is_some_and(|t| !t.is_empty()),
            "serve-tools",
            "not scheduled: a tool call is not the schema's document; send tools or \
             response_format, not both",
        ),
        (
            request.stop.as_ref().is_some_and(|s| !s.is_empty()),
            "serve-stop",
            "not scheduled: a stop string could cut the document the schema needs whole",
        ),
        (
            request.ignore_eos == Some(true),
            "serve-ignore-eos",
            "not scheduled: the document ends at end-of-sequence, which ignore_eos removes",
        ),
        (
            traced,
            "serve-trace",
            "not scheduled: a trace runs the instrumented loop, which applies no constraint; \
             without X-Trace-Level, on a server not started with tracing on, the engine runs \
             and applies it",
        ),
    ];
    match refused.iter().find(|(asked, ..)| *asked) {
        Some((_, path, removed_by)) => Err(unsupported_path(path, removed_by)),
        None => Ok(()),
    }
}

/// The constraint a request asks for, checked before any model work: `None` when it asks for
/// none (no `response_format`, or `text`), so the chain answers it unchanged.
fn requested_constraint(
    request: &ChatCompletionRequest,
    traced: bool,
) -> Result<Option<crate::constrain::ConstraintRequest>, crate::constrain::ConstraintError> {
    let Some(format) = request.response_format.as_ref() else {
        return Ok(None);
    };
    let Some(constraint) = format.constraint()? else {
        return Ok(None);
    };
    refuse_unconstrainable(request, traced)?;
    // The second reader compiles the schema now: one it cannot read is `SchemaInvalid`, and a
    // `$ref` to anything outside it `SchemaUnsupported`, before the model is touched
    if let crate::constrain::ConstraintRequest::JsonSchema(schema) = &constraint {
        crate::constrain::check_schema(schema)?;
    }
    Ok(Some(constraint))
}

/// A backend that runs the engine, and so applies a constraint.
enum ConstrainedRoute {
    /// The Qwen3.5 hybrid's resident session (#3571).
    Qwen35(Arc<crate::api::Qwen35Served>),
    /// The dense CUDA model, borrowed under its write lock for the turn (#4280).
    #[cfg(feature = "cuda")]
    Cuda(Arc<std::sync::RwLock<crate::gguf::OwnedQuantizedModelCuda>>),
    /// The dense CPU model.
    Cpu(Arc<crate::gguf::OwnedQuantizedModel>),
}

/// The backend the unconstrained chain would answer this request on, when that backend runs
/// the engine; else that backend's refusal. The predicates are the chain's own, in its order
/// (`openai_chat_completions_handler`, then `cpu_chat_backends`), and read only what the server
/// holds: the same request is refused or answered the same way on every call.
fn constrained_route(
    state: &AppState,
) -> Result<ConstrainedRoute, crate::constrain::ConstraintError> {
    if let Some(session) = state.qwen35_session() {
        return Ok(ConstrainedRoute::Qwen35(session));
    }
    if state
        .model_architecture()
        .is_some_and(|arch| is_qwen3_moe_arch(&arch))
    {
        return Err(unsupported_path("serve-qwen3-moe", OUTSIDE_3568));
    }
    if let Some(route) = accelerated_route(state) {
        return route;
    }
    if let Some(model) = state.quantized_model() {
        return Ok(ConstrainedRoute::Cpu(Arc::clone(model)));
    }
    if state.apr_transformer().is_some() {
        return Err(unsupported_path("serve-apr", OUTSIDE_3568));
    }
    Err(unsupported_path("serve-registry", OUTSIDE_3568))
}

/// The accelerated backends the chain tries between the MoE refusal and the CPU model, in the
/// chain's order; `None` when none of them holds a model.
#[cfg_attr(not(any(feature = "gpu", feature = "cuda")), allow(unused_variables))]
fn accelerated_route(
    state: &AppState,
) -> Option<Result<ConstrainedRoute, crate::constrain::ConstraintError>> {
    #[cfg(feature = "gpu")]
    if state.gpu_model().is_some() {
        return Some(Err(unsupported_path("serve-wgpu", OUTSIDE_3568)));
    }
    #[cfg(feature = "gpu")]
    if state.cached_model().is_some() {
        return Some(Err(unsupported_path("serve-wgpu-cached", OUTSIDE_3568)));
    }
    #[cfg(feature = "cuda")]
    if let Some(model) = state.cuda_model() {
        return Some(Ok(ConstrainedRoute::Cuda(Arc::clone(model))));
    }
    #[cfg(feature = "cuda")]
    if state.apr_q4k_tx().is_some() {
        return Some(Err(unsupported_path("serve-apr-q4k-cuda", OUTSIDE_3568)));
    }
    #[cfg(feature = "cuda")]
    if state.safetensors_cuda_model().is_some() {
        return Some(Err(unsupported_path(
            "serve-safetensors-cuda",
            OUTSIDE_3568,
        )));
    }
    None
}

/// The prompt the unconstrained route builds for this request, as tokens: the model's own
/// template, the route's own tokenizer. Refused with `SchemaWithThinking` when the rendered
/// template leaves a `<think>` block open, so the model would reason before it answers.
#[allow(clippy::result_large_err)]
fn constrained_prompt(
    state: &AppState,
    route: &ConstrainedRoute,
    mapped: &crate::gguf::MappedGGUFModel,
    request: &ChatCompletionRequest,
) -> Result<Vec<u32>, Response> {
    let arch = state.model_architecture();
    let rendered = match route {
        ConstrainedRoute::Qwen35(_) => {
            crate::api::realize_handlers::format_chat_messages_official_thinking_tools(
                Some(&mapped.model),
                &request.messages,
                arch.as_deref(),
                request.thinking(),
                request.tools.as_deref(),
            )
        },
        _ => format_chat_messages_for_state_thinking_tools(
            state,
            &request.messages,
            arch.as_deref(),
            request.thinking(),
            request.tools.as_deref(),
        ),
    };
    let text = rendered.map_err(|e| fail_response(state, StatusCode::BAD_REQUEST, e))?;
    if crate::infer::prompt_opens_thinking(&text) {
        return Err(constraint_refusal(
            state,
            &crate::constrain::ConstraintError::WithThinking(
                "the model's chat template leaves a <think> block open, so the model would \
                 reason before it answers; a constraint applies to no-think output only (#3735 \
                 constrains after </think>)"
                    .to_string(),
            ),
        ));
    }
    let ids = match route {
        ConstrainedRoute::Qwen35(_) => mapped.model.encode(&text).unwrap_or_default(),
        _ => require_tokenizer(state)?.encode(&text),
    };
    if ids.is_empty() {
        return Err(fail_response(
            state,
            StatusCode::BAD_REQUEST,
            "Messages cannot be empty",
        ));
    }
    Ok(ids)
}

/// The generate config the unconstrained route builds, with its own context admission (a
/// prompt that cannot fit is the request's fault: 400, before the model is touched). The stop
/// set also holds the end-of-sequence id the constraint was compiled with, so a complete
/// document always ends the turn.
#[allow(clippy::result_large_err)]
fn constrained_config(
    state: &AppState,
    route: &ConstrainedRoute,
    mapped: &crate::gguf::MappedGGUFModel,
    request: &ChatCompletionRequest,
    prompt: Vec<u32>,
    cancel: &CancelToken,
) -> Result<(Vec<u32>, crate::gguf::QuantizedGenerateConfig), Response> {
    let mut config = match route {
        ConstrainedRoute::Qwen35(served) => {
            let context_length = served.context_length;
            if prompt.len() >= context_length {
                return Err(fail_response(
                    state,
                    StatusCode::BAD_REQUEST,
                    format!(
                        "the prompt is {} tokens and this model declares a context of \
                         {context_length}: it was refused whole rather than truncated (#3571)",
                        prompt.len()
                    ),
                ));
            }
            let budget = request
                .max_tokens
                .unwrap_or(256)
                .min(context_length - prompt.len());
            let stop = stop_tokens_unless_ignore_eos(request, state.model_eos_token_id());
            gen_config_from_request(request, budget, stop, cancel.clone())
        },
        #[cfg(feature = "cuda")]
        ConstrainedRoute::Cuda(_) => {
            // D5: a turn the device's KV cache cannot hold is refused whole, as unconstrained
            fit_serving_context(state, Ok(prompt.clone()))?;
            let tokenizer = require_tokenizer(state)?;
            chat_quantized_config(
                request,
                &tokenizer,
                state.model_eos_token_id(),
                false,
                cancel,
            )
        },
        ConstrainedRoute::Cpu(model) => {
            let tokenizer = require_tokenizer(state)?;
            let mut config = chat_quantized_config(
                request,
                &tokenizer,
                state.model_eos_token_id(),
                false,
                cancel,
            );
            config.max_tokens = model
                .effective_max_tokens(prompt.len(), config.max_tokens)
                .map_err(|e| fail_response(state, crate::api::generation_error_status(&e), e))?;
            config
        },
    };
    if let Some(eos) = mapped.model.eos_token_id() {
        if !config.stop_tokens.contains(&eos) {
            config.stop_tokens.push(eos);
        }
    }
    Ok((prompt, config))
}

/// One constrained turn on `route`, on the calling (blocking) thread.
fn constrained_turn(
    route: ConstrainedRoute,
    constraint: &mut dyn crate::constrain::TokenConstraint,
    prompt: &[u32],
    config: &crate::gguf::QuantizedGenerateConfig,
) -> crate::error::Result<(crate::session::Turn, crate::session::ConstrainedStop)> {
    use crate::error::RealizarError;
    match route {
        ConstrainedRoute::Qwen35(served) => {
            let mut session = served
                .session
                .lock()
                .map_err(|_| RealizarError::InferenceError(POISONED.to_string()))?;
            let turn = session.generate_constrained(prompt, config, constraint);
            served
                .on_gpu
                .store(session.on_gpu(), std::sync::atomic::Ordering::Relaxed);
            turn
        },
        #[cfg(feature = "cuda")]
        ConstrainedRoute::Cuda(model) => {
            let mut model = model.write().map_err(|_| {
                RealizarError::InferenceError(
                    "the CUDA model is unusable: an earlier request panicked while holding it"
                        .to_string(),
                )
            })?;
            let mut session = crate::session::Session::new(
                crate::gguf::dense_session_borrowed::BorrowedCudaForward::new(&mut model),
            );
            session.generate_constrained(prompt, config, constraint)
        },
        ConstrainedRoute::Cpu(model) => {
            dense_cpu_session(&model).generate_constrained(prompt, config, constraint)
        },
    }
}

/// The verdict on a finished constrained reply: `Ok` when it ships. A reply the budget cut is
/// `Truncated` (no partial document is returned); an empty one, or one the second reader
/// refuses, is `SchemaViolation`.
fn constrained_verdict(
    constraint: &crate::constrain::ConstraintRequest,
    stop: crate::session::ConstrainedStop,
    text: &str,
    budget: usize,
) -> Result<(), crate::constrain::ConstraintError> {
    use crate::constrain::{ConstraintError, ConstraintRequest};
    use crate::session::ConstrainedStop;
    match stop {
        ConstrainedStop::Length => Err(ConstraintError::Truncated { max_tokens: budget }),
        ConstrainedStop::Complete if text.trim().is_empty() => Err(ConstraintError::Violation(
            "the output is empty".to_string(),
        )),
        ConstrainedStop::Complete => match constraint {
            ConstraintRequest::JsonSchema(schema) => crate::constrain::second_reader(schema, text),
            ConstraintRequest::Lark(_) => Ok(()),
        },
    }
}

/// The response for a reply that stands. Built here, not by `build_chat_response`: that one
/// reads a reply as long as the budget as cut ("length"), and a constrained reply that is
/// whole exactly at the budget is complete ("stop"). The text is the decoded tokens verbatim:
/// the chat-output cleaner would edit the document the second reader just passed.
fn constrained_response(
    request_id: &str,
    request: &ChatCompletionRequest,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    used_gpu: bool,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: request_id.to_string(),
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
                ..Default::default()
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
        timings: None,
        used_gpu: Some(used_gpu),
    }
}

/// Answer a request that asks for a constraint (#3568 PR 4), or `None` when it asks for none.
async fn try_constrained_chat(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
    cancel: &CancelToken,
) -> Option<Response> {
    let constraint = match requested_constraint(request, state.should_trace(trace_level)) {
        Ok(Some(c)) => c,
        Ok(None) => return None,
        Err(e) => return Some(constraint_refusal(state, &e)),
    };
    let answered = constrained_chat(state, request, constraint, request_id, start, cancel).await;
    Some(answered.unwrap_or_else(|refusal| refusal))
}

/// The constrained turn on the route the chain would take, read again before it ships.
async fn constrained_chat(
    state: &AppState,
    request: &ChatCompletionRequest,
    constraint: crate::constrain::ConstraintRequest,
    request_id: &str,
    start: Instant,
    cancel: &CancelToken,
) -> Result<Response, Response> {
    use crate::error::RealizarError;
    let route = constrained_route(state).map_err(|e| constraint_refusal(state, &e))?;
    let mapped = state.mapped_gguf_model().ok_or_else(|| {
        let e = unsupported_path(
            "serve-without-gguf",
            "not scheduled: a constraint's vocabulary is read from the GGUF the server \
             retains, and this server retains none",
        );
        constraint_refusal(state, &e)
    })?;
    let prompt = constrained_prompt(state, &route, &mapped, request)?;
    let (prompt, config) = constrained_config(state, &route, &mapped, request, prompt, cancel)?;
    let prompt_tokens = prompt.len();
    let job_mapped = Arc::clone(&mapped);
    let job_constraint = constraint.clone();
    let joined = tokio::task::spawn_blocking(move || {
        // Compiled against the model's vocabulary here, off the async runtime: the first
        // request indexes the whole vocabulary, once, for the life of the server
        let env = job_mapped
            .constraint_env()
            .map_err(RealizarError::Constraint)?;
        let mut compiled = job_constraint
            .compile(&env)
            .map_err(RealizarError::Constraint)?;
        constrained_turn(route, compiled.as_mut(), &prompt, &config)
    })
    .await;
    let (turn, stop) = match joined {
        Ok(Ok(done)) => done,
        Ok(Err(RealizarError::Constraint(e))) => return Err(constraint_refusal(state, &e)),
        Ok(Err(e)) => {
            return Err(fail_response(
                state,
                crate::api::generation_error_status(&e),
                e,
            ))
        },
        Err(e) => {
            return Err(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the constrained generation task failed: {e}"),
            ))
        },
    };
    let reply = &turn.tokens[prompt_tokens..];
    let text = mapped.model.decode(reply);
    constrained_verdict(&constraint, stop, &text, reply.len())
        .map_err(|e| constraint_refusal(state, &e))?;
    state.metrics.record_success(reply.len(), start.elapsed());
    let response = constrained_response(
        request_id,
        request,
        text,
        prompt_tokens,
        reply.len(),
        turn.used_gpu,
    );
    Ok(Json(response).into_response())
}

#[cfg(test)]
#[path = "constrained_chat_backend_tests.rs"]
mod constrained_chat_tests;
