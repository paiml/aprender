//! Banco HTTP endpoint handlers.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::Stream;

use super::state::BancoState;
use super::types::{
    BancoChatChunk, BancoChatRequest, BancoChatResponse, ChatChoice, ChatChunkChoice, ChatDelta,
    ErrorResponse, Usage,
};

// Re-export conversation and prompt handlers from split modules
pub use super::handlers_conversations::{
    create_conversation_handler, delete_conversation_handler, export_conversations_handler,
    get_conversation_handler, import_conversations_handler, list_conversations_handler,
    rename_conversation_handler, search_conversations_handler,
};
pub use super::handlers_prompts::{
    delete_prompt_handler, get_prompt_handler, list_prompts_handler, save_prompt_handler,
};
use crate::serve::router::RoutingDecision;
use crate::serve::templates::{ChatMessage, Role};

// ============================================================================
// BANCO-HDL-001: Health
// ============================================================================

pub async fn health_handler(State(state): State<BancoState>) -> Json<super::types::HealthResponse> {
    Json(state.health_status())
}

/// GET /health/ready — readiness probe (k8s compatible).
/// Returns 200 when model is loaded and system is ready to serve.
pub async fn readiness_handler(State(state): State<BancoState>) -> StatusCode {
    if state.model.is_loaded() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// GET /health/live — liveness probe (k8s compatible).
/// Returns 200 if the server process is alive.
pub async fn liveness_handler() -> StatusCode {
    StatusCode::OK
}

// ============================================================================
// BANCO-HDL-002: Models
// ============================================================================

pub async fn models_handler(State(state): State<BancoState>) -> Json<super::types::ModelsResponse> {
    Json(state.list_models())
}

// ============================================================================
// BANCO-HDL-003: System
// ============================================================================

pub async fn system_handler(State(state): State<BancoState>) -> Json<super::types::SystemResponse> {
    Json(state.system_info())
}

// ============================================================================
// BANCO-HDL-004: Chat Completions
// ============================================================================

pub async fn chat_completions_handler(
    State(state): State<BancoState>,
    Json(request): Json<BancoChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Expand @preset: references in message content
    let mut request = request;
    for msg in &mut request.messages {
        msg.content = state.prompts.expand(&msg.content);
    }

    // Attachments: inject file content as system context
    if !request.attachments.is_empty() {
        let mut attachment_context = String::new();
        for att in &request.attachments {
            attachment_context
                .push_str(&format!("[Attached file: {}]\n{}\n\n", att.name, att.content));
        }
        let att_msg = ChatMessage::system(format!(
            "The user has attached the following files:\n\n{attachment_context}"
        ));
        request.messages.insert(0, att_msg);
    }

    // RAG: retrieve relevant chunks and prepend as context
    if request.rag {
        let query = request.messages.last().map(|m| m.content.as_str()).unwrap_or("");
        let top_k = request.rag_config.as_ref().map(|c| c.top_k).unwrap_or(5);
        let min_score = request.rag_config.as_ref().map(|c| c.min_score).unwrap_or(0.1);
        let results = state.rag.search(query, top_k, min_score);
        if !results.is_empty() {
            let context: String = results
                .iter()
                .map(|r| format!("[Source: {} chunk {}]\n{}", r.file, r.chunk, r.text))
                .collect::<Vec<_>>()
                .join("\n\n");
            let rag_msg = ChatMessage::system(format!(
                "Use the following context to answer the user's question:\n\n{context}"
            ));
            request.messages.insert(0, rag_msg);
        }
    }

    // Validate messages are not empty
    if request.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("messages must not be empty", "invalid_request", 400)),
        ));
    }

    // Check context window
    if !state.context_manager.fits(&request.messages) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "messages exceed context window",
                "context_length_exceeded",
                400,
            )),
        ));
    }

    // Check circuit breaker (Phase 1: estimate tiny cost)
    if let Err(e) = state.circuit_breaker.check(0.001) {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse::new(e.to_string(), "rate_limit", 429)),
        ));
    }

    // Save user messages to conversation if conversation_id provided
    if let Some(ref conv_id) = request.conversation_id {
        for msg in &request.messages {
            let _ = state.conversations.append(conv_id, msg.clone());
        }
    }

    // Route the request
    let decision = state.router.route();

    if request.stream {
        Ok(stream_response(state, request, decision).into_response())
    } else {
        Ok(sync_response(state, request, decision).into_response())
    }
}

// ============================================================================
// BANCO-HDL-005: Non-streaming response (echo/dry-run)
// ============================================================================

fn sync_response(
    state: BancoState,
    request: BancoChatRequest,
    decision: RoutingDecision,
) -> Json<BancoChatResponse> {
    let model_name = request
        .model
        .clone()
        .or_else(|| state.model.info().map(|m| m.model_id))
        .unwrap_or_else(|| "banco-echo".to_string());

    // Apply template engine to format the prompt (used by inference path)
    let _formatted = state.template_engine.apply(&request.messages);

    // Determine response mode
    #[cfg(feature = "realizar")]
    let inference_result = super::handlers_inference::try_inference(&state, &request);
    #[cfg(not(feature = "realizar"))]
    let inference_result: Option<(String, String, u32)> = None;

    let (content, finish_reason, actual_completion_tokens) = if let Some(result) = inference_result
    {
        result
    } else if state.model.is_loaded() {
        let model_info = state
            .model
            .info()
            .map(|m| {
                format!(
                    "{}({} layers, {}d)",
                    m.architecture.as_deref().unwrap_or("?"),
                    m.num_layers.unwrap_or(0),
                    m.hidden_dim.unwrap_or(0)
                )
            })
            .unwrap_or_default();
        (
            format!(
                "[banco model-loaded] {model_info} | route={decision:?} | prompt_len={} | inference=pending",
                request.messages.len()
            ),
            "model_loaded".to_string(),
            0u32,
        )
    } else {
        let last_msg = request.messages.last().map(|m| m.content.as_str()).unwrap_or("");
        let echo = if last_msg.len() > 200 { &last_msg[..200] } else { last_msg };
        (
            format!(
                "No model loaded. To enable inference, load a GGUF model:\n\
                 curl -X POST http://localhost:8090/api/v1/models/load -d '{{\"model\": \"./model.gguf\"}}'\n\n\
                 Your message ({} chars, {} tokens estimated): {echo}",
                last_msg.len(),
                request.messages.len(),
            ),
            "dry_run".to_string(),
            0u32,
        )
    };

    let prompt_tokens = state.context_manager.estimate_tokens(&request.messages) as u32;
    let completion_tokens = if actual_completion_tokens > 0 {
        actual_completion_tokens
    } else {
        (content.len() / 4) as u32
    };

    // Save assistant response to conversation
    if let Some(ref conv_id) = request.conversation_id {
        let _ = state.conversations.append(conv_id, ChatMessage::assistant(&content));
    }

    Json(BancoChatResponse {
        id: format!("banco-{}", now_epoch()),
        object: "chat.completion".to_string(),
        created: now_epoch(),
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::assistant(content),
            finish_reason,
        }],
        usage: {
            let total = prompt_tokens + completion_tokens;
            let window = state.context_manager.available_tokens() as u32;
            let pct = if window > 0 { (total as f32 / window as f32) * 100.0 } else { 0.0 };
            Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: total,
                context_window: Some(window),
                context_used_pct: Some(pct),
            }
        },
    })
}

// ============================================================================
// BANCO-HDL-006: SSE streaming response
// ============================================================================

fn stream_response(
    state: BancoState,
    request: BancoChatRequest,
    _decision: RoutingDecision,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let model_name = request.model.clone().unwrap_or_else(|| "banco-echo".to_string());
    let id = format!("banco-{}", now_epoch());
    let created = now_epoch();

    // Try to generate real tokens via inference
    #[cfg(feature = "realizar")]
    let real_tokens = super::handlers_inference::try_stream_inference(&state, &request);
    #[cfg(not(feature = "realizar"))]
    let real_tokens: Option<Vec<(String, Option<String>)>> = {
        let _ = &state;
        None
    };

    let tokens_with_finish: Vec<(String, Option<String>)> = if let Some(toks) = real_tokens {
        toks
    } else {
        // Helpful dry-run message
        vec![
            ("No model loaded. ".to_string(), None),
            ("Load a GGUF model via ".to_string(), None),
            ("POST /api/v1/models/load ".to_string(), None),
            ("to enable real inference.".to_string(), None),
            (String::new(), Some("dry_run".to_string())),
        ]
    };

    let stream = async_stream::stream! {
        // Role chunk
        let role_chunk = BancoChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_name.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: Some(Role::Assistant), content: None },
                finish_reason: None,
            }],
        };
        if let Ok(data) = serde_json::to_string(&role_chunk) {
            yield Ok(Event::default().data(data));
        }

        // Content + final chunks
        for (text, finish) in &tokens_with_finish {
            let chunk = BancoChatChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: if text.is_empty() { None } else { Some(text.clone()) },
                    },
                    finish_reason: finish.clone(),
                }],
            };
            if let Ok(data) = serde_json::to_string(&chunk) {
                yield Ok(Event::default().data(data));
            }
        }

        // [DONE] sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

// Re-export token handlers from split module
pub use super::handlers_tokens::{
    detokenize_handler, embeddings_handler, get_parameters_handler, tokenize_handler,
    update_parameters_handler,
};

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
