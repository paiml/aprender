//! Anthropic Messages API handler (aprender#7484, G-01): `POST /v1/messages`.
//!
//! The request is validated here and translated into a [`ChatCompletionRequest`]; generation
//! then runs through [`openai_chat_completions_handler`], the same backend chain as
//! `/v1/chat/completions`. The reply is reshaped into the Messages wire format. Refusals use
//! the Anthropic error envelope `{"type":"error","error":{"type":...,"message":...}}`.
//!
//! The non-streaming body's key set is vendored, with its source URL and date, in
//! `anthropic_message_schema.json` and checked by `tests/messages_wire_7484.rs`.
//! Streaming events: `message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`.
#![allow(unreachable_pub)] // re-exported as pub from api/mod.rs

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::Deserialize;

use super::ollama_handlers::content_fragments;
use super::{
    openai_chat_completions_handler, AppState, ChatChoice, ChatCompletionRequest, ChatMessage,
    ChoiceCount, Usage,
};

/// `POST /v1/messages` request body (the fields G-01 exercises).
#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    pub messages: Vec<AnthropicMessage>,
    /// A string, or an array of text blocks.
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: bool,
    /// Same domain guard as the chat request (aprender#2375). `to_chat_request` builds that
    /// struct in Rust, so its own guard never sees a Messages request.
    #[serde(
        default,
        deserialize_with = "crate::api::types::deserialize_temperature_f32"
    )]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// The next four are parsed only so they can be refused by name.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
}

impl AnthropicMessagesRequest {
    /// The request-level refusals, in order. `Ok` carries the model name.
    fn checked_model(&self) -> Result<String, String> {
        if let Some(field) = self.unsupported_field() {
            return Err(format!("{field}: not supported by this server"));
        }
        if self.top_p.is_some_and(|p| !(p > 0.0 && p <= 1.0)) {
            return Err("top_p: must be greater than 0 and at most 1".to_string());
        }
        if self.max_tokens.unwrap_or(0) == 0 {
            return Err("max_tokens: a positive integer is required".to_string());
        }
        if self.messages.is_empty() {
            return Err("messages: at least one message is required".to_string());
        }
        self.model
            .clone()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| "model: a model name is required".to_string())
    }

    /// The first field that asks for behaviour this route does not have. A 200 that ignored
    /// it would answer a different request from the one sent.
    fn unsupported_field(&self) -> Option<&'static str> {
        if self.stop_sequences.as_ref().is_some_and(|s| !s.is_empty()) {
            Some("stop_sequences")
        } else if self.tools.as_ref().is_some_and(|t| !t.is_empty()) {
            Some("tools")
        } else if self.tool_choice.is_some() {
            Some("tool_choice")
        } else if self
            .thinking
            .as_ref()
            .is_some_and(|t| t.get("type").and_then(|v| v.as_str()) != Some("disabled"))
        {
            Some("thinking")
        } else {
            None
        }
    }
}

/// The parts of the chat reply this route reads. Narrower than `ChatCompletionResponse`:
/// the SafeTensors CUDA backend writes its 200 body without `created`.
#[derive(Debug, Deserialize)]
struct ChatReply {
    id: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

/// The Messages error `type` for a status returned by the chat path.
fn error_kind_of(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::NOT_FOUND => "not_found_error",
        _ => "api_error",
    }
}

/// `/v1/messages` on an embedding model: the kind refusal in the Messages envelope.
pub async fn anthropic_embedding_kind_refusal(State(state): State<AppState>) -> Response {
    anthropic_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        &super::embedding_kind_reason(&state),
    )
}

/// One conversation turn. `content` is a string, or an array of text blocks.
#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// Flatten string or text-block content into plain text. Any other block is refused by name,
/// never dropped: a silently dropped image would answer a question about something not sent.
fn text_of(content: &serde_json::Value, field: &str) -> Result<String, String> {
    match content {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(blocks) => {
            let mut out = String::new();
            for block in blocks {
                match (
                    block.get("type").and_then(|t| t.as_str()),
                    block.get("text").and_then(|t| t.as_str()),
                ) {
                    (Some("text"), Some(text)) => out.push_str(text),
                    _ => return Err(format!("{field}: only text content blocks are supported")),
                }
            }
            Ok(out)
        },
        _ => Err(format!(
            "{field}: content must be a string or an array of text blocks"
        )),
    }
}

/// Translate a validated Messages request into the internal chat request.
fn to_chat_request(
    request: &AnthropicMessagesRequest,
    model: &str,
) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    if let Some(system) = &request.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: text_of(system, "system")?,
            ..Default::default()
        });
    }
    for message in &request.messages {
        if message.role != "user" && message.role != "assistant" {
            return Err(format!(
                "messages: role must be \"user\" or \"assistant\", not \"{}\"",
                message.role
            ));
        }
        messages.push(ChatMessage {
            role: message.role.clone(),
            content: text_of(&message.content, "messages")?,
            ..Default::default()
        });
    }
    Ok(ChatCompletionRequest {
        model: model.to_string(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        n: ChoiceCount::ONE,
        // The internal chat path is always driven non-streaming; `stream` is honoured on the wire.
        stream: false,
        ..Default::default()
    })
}

/// The Anthropic error envelope.
fn anthropic_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {"type": kind, "message": message},
        })),
    )
        .into_response()
}

/// Map an OpenAI finish reason to a Messages `stop_reason`.
fn stop_reason_of(finish_reason: &str) -> &'static str {
    if finish_reason == "length" {
        "max_tokens"
    } else {
        "end_turn"
    }
}

fn push_event(out: &mut String, name: &str, data: serde_json::Value) {
    out.push_str("event: ");
    out.push_str(name);
    out.push_str("\ndata: ");
    out.push_str(&data.to_string());
    out.push_str("\n\n");
}

/// The Messages event sequence for a completed generation, as `text/event-stream`.
fn sse_response(
    id: &str,
    model: &str,
    content: &str,
    input_tokens: usize,
    output_tokens: usize,
    stop_reason: &str,
) -> Response {
    let mut out = String::new();
    push_event(
        &mut out,
        "message_start",
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }),
    );
    push_event(
        &mut out,
        "content_block_start",
        serde_json::json!({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
    );
    for fragment in content_fragments(content) {
        push_event(
            &mut out,
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": fragment},
            }),
        );
    }
    push_event(
        &mut out,
        "content_block_stop",
        serde_json::json!({"type": "content_block_stop", "index": 0}),
    );
    push_event(
        &mut out,
        "message_delta",
        serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens},
        }),
    );
    push_event(
        &mut out,
        "message_stop",
        serde_json::json!({"type": "message_stop"}),
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], out).into_response()
}

/// `POST /v1/messages` — Anthropic Messages API (G-01, aprender#7484).
pub async fn anthropic_messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(cancel): Extension<crate::generate::CancelToken>,
    body: Bytes,
) -> Response {
    let request: AnthropicMessagesRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            // aprender#2376(7): serde's text names our parse position. Only a reason a
            // validator marked as client-visible is forwarded.
            let message = super::client_visible_reason(&e.to_string())
                .unwrap_or_else(|| "request body is not a valid Messages request".to_string());
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &message);
        },
    };
    let model = match request.checked_model() {
        Ok(model) => model,
        Err(message) => {
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &message)
        },
    };
    let chat_request = match to_chat_request(&request, &model) {
        Ok(chat_request) => chat_request,
        Err(message) => {
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &message)
        },
    };

    let inner = openai_chat_completions_handler(
        State(state),
        headers,
        Extension(cancel),
        Json(chat_request),
    )
    .await;
    let status = inner.status();
    let bytes = axum::body::to_bytes(inner.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    if !status.is_success() {
        let message = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str().map(str::to_string)))
            .unwrap_or_else(|| "generation unavailable".to_string());
        return anthropic_error(status, error_kind_of(status), &message);
    }
    let Ok(generated) = serde_json::from_slice::<ChatReply>(&bytes) else {
        return anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "unreadable generation result",
        );
    };
    let (content, stop_reason) = generated
        .choices
        .first()
        .map(|c| (c.message.content.clone(), stop_reason_of(&c.finish_reason)))
        .unwrap_or_else(|| (String::new(), "end_turn"));
    let input_tokens = generated.usage.prompt_tokens;
    let output_tokens = generated.usage.completion_tokens;
    let id = format!("msg_{}", generated.id);

    if request.stream {
        return sse_response(
            &id,
            &model,
            &content,
            input_tokens,
            output_tokens,
            stop_reason,
        );
    }
    Json(serde_json::json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> AnthropicMessagesRequest {
        serde_json::from_str(json).expect("a Messages request")
    }

    #[test]
    fn system_and_text_blocks_flatten_into_the_chat_messages() {
        let request = parse(
            r#"{"model":"m","max_tokens":8,"system":"Be brief.",
                "messages":[{"role":"user","content":[{"type":"text","text":"Hi "},{"type":"text","text":"there"}]}]}"#,
        );
        let chat = to_chat_request(&request, "m").expect("translates");
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].content, "Be brief.");
        assert_eq!(chat.messages[1].role, "user");
        assert_eq!(chat.messages[1].content, "Hi there");
        assert_eq!(chat.max_tokens, Some(8));
        assert!(!chat.stream, "the internal path is always non-streaming");
    }

    #[test]
    fn non_text_blocks_are_refused_by_name_not_dropped() {
        let request = parse(
            r#"{"model":"m","max_tokens":8,
                "messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}"#,
        );
        let err = to_chat_request(&request, "m").expect_err("an image is refused");
        assert!(err.contains("text content blocks"), "{err}");
    }

    #[test]
    fn unknown_roles_are_refused() {
        let request =
            parse(r#"{"model":"m","max_tokens":8,"messages":[{"role":"system","content":"x"}]}"#);
        let err = to_chat_request(&request, "m").expect_err("system is not a message role");
        assert!(err.contains("role must be"), "{err}");
    }

    #[test]
    fn a_chat_reply_without_created_is_readable() {
        // The shape `try_safetensors_cuda_backend` writes.
        let reply: ChatReply = serde_json::from_str(
            r#"{"id":"x","object":"chat.completion","model":"m",
                "choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
        )
        .expect("readable");
        assert_eq!(reply.choices[0].message.content, "4");
    }

    #[test]
    fn a_missing_model_is_not_found_not_api_error() {
        assert_eq!(error_kind_of(StatusCode::NOT_FOUND), "not_found_error");
        assert_eq!(
            error_kind_of(StatusCode::BAD_REQUEST),
            "invalid_request_error"
        );
        assert_eq!(error_kind_of(StatusCode::SERVICE_UNAVAILABLE), "api_error");
    }

    #[test]
    fn length_finish_is_max_tokens_and_anything_else_is_end_turn() {
        assert_eq!(stop_reason_of("length"), "max_tokens");
        assert_eq!(stop_reason_of("stop"), "end_turn");
        assert_eq!(stop_reason_of(""), "end_turn");
    }
}
