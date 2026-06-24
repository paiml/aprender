//! Ollama-compatible API handlers (PMAT-923).
//!
//! Makes `apr serve` a drop-in replacement for an Ollama HTTP server by exposing
//! Ollama's native `/api/chat` and `/api/generate` endpoints on the realizar
//! router. Both delegate to the existing OpenAI `/v1/chat/completions` generation
//! path ([`openai_chat_completions_handler`]) and re-shape the result into
//! Ollama's wire schema, so a single generation path serves both protocols.
//!
//! Ollama response schema (non-streaming):
//! ```json
//! {"model":"...","created_at":"...","message":{"role":"assistant","content":"..."},
//!  "done":true,"prompt_eval_count":N,"eval_count":M}
//! ```
//! `/api/generate` differs only in carrying a flat `response` string instead of a
//! nested `message` object.
//!
//! Discharges OBLIG-OLLAMA-API-CHAT-GENERATE-ROUTED in
//! `contracts/apr-serve-openai-compat-v1.yaml`.
#![allow(unreachable_pub)] // re-exported as pub from api/mod.rs

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::{
    openai_chat_completions_handler, AppState, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage,
};

// ============================================================================
// Ollama wire types
// ============================================================================

/// Ollama `/api/chat` request.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatRequest {
    /// Model tag (Ollama-style). Optional — defaults to the loaded model.
    #[serde(default)]
    pub model: Option<String>,
    /// Conversation messages.
    pub messages: Vec<OllamaMessage>,
    /// Stream tokens (currently coalesced into a single final message).
    #[serde(default)]
    pub stream: bool,
    /// Optional Ollama `options` block (temperature, num_predict, top_k, top_p, seed).
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama message (`role` + `content`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    /// "system" | "user" | "assistant".
    pub role: String,
    /// Message text.
    pub content: String,
}

/// Ollama `options` block (subset that maps onto our sampling config).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaOptions {
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Random seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Ollama's name for max tokens.
    #[serde(default)]
    pub num_predict: Option<usize>,
}

/// Ollama `/api/chat` response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatResponse {
    /// Model tag echoed back.
    pub model: String,
    /// RFC-3339-style creation timestamp.
    pub created_at: String,
    /// The assistant turn.
    pub message: OllamaMessage,
    /// Terminal flag — always true for the coalesced response.
    pub done: bool,
    /// Prompt token count.
    pub prompt_eval_count: usize,
    /// Generated token count.
    pub eval_count: usize,
}

/// Ollama `/api/generate` request (single prompt, non-chat).
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateRequest {
    /// Model tag. Optional — defaults to the loaded model.
    #[serde(default)]
    pub model: Option<String>,
    /// The prompt to complete.
    pub prompt: String,
    /// Optional system preamble.
    #[serde(default)]
    pub system: Option<String>,
    /// Stream tokens (currently coalesced into a single final response).
    #[serde(default)]
    pub stream: bool,
    /// Optional Ollama `options` block.
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama `/api/generate` response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateResponse {
    /// Model tag echoed back.
    pub model: String,
    /// RFC-3339-style creation timestamp.
    pub created_at: String,
    /// The generated text (flat, not nested in a message object).
    pub response: String,
    /// Terminal flag — always true for the coalesced response.
    pub done: bool,
    /// Prompt token count.
    pub prompt_eval_count: usize,
    /// Generated token count.
    pub eval_count: usize,
}

// ============================================================================
// Conversion helpers (pure — unit-tested)
// ============================================================================

/// RFC-3339-style timestamp for `created_at` (Ollama wire format).
fn created_at_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Clients only require a string field, not strict parsing.
    format!("{secs}.000000000Z")
}

/// Default model label when the request omits `model`.
fn model_label(model: &Option<String>) -> String {
    model
        .clone()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "apr".to_string())
}

/// Build a [`ChatCompletionRequest`] from Ollama messages + options.
///
/// This is the single translation point Ollama→internal; the resulting request
/// runs through the SAME backend chain as `/v1/chat/completions`.
fn to_chat_request(
    model: &str,
    messages: Vec<OllamaMessage>,
    options: &Option<OllamaOptions>,
) -> ChatCompletionRequest {
    let opts = options.clone().unwrap_or_default();
    ChatCompletionRequest {
        model: model.to_string(),
        messages: messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
                ..Default::default()
            })
            .collect(),
        max_tokens: opts.num_predict,
        temperature: opts.temperature,
        top_p: opts.top_p,
        top_k: opts.top_k,
        seed: opts.seed,
        n: 1,
        // We always coalesce into one final Ollama message, so drive the
        // underlying chat path non-streaming regardless of the client flag.
        stream: false,
        ..Default::default()
    }
}

/// Extract `(content, prompt_tokens, completion_tokens)` from the OpenAI chat
/// response, or a fallback `(error_text, 0, 0)` when generation failed.
///
/// Crucially this ALWAYS yields an Ollama-shaped body — even on a backend error
/// or a missing model — so a wired route is observably distinct from the axum
/// `not_found` fallback (which has no `done` field).
fn chat_response_to_parts(status: StatusCode, body: &[u8]) -> (String, usize, usize) {
    if status.is_success() {
        if let Ok(resp) = serde_json::from_slice::<ChatCompletionResponse>(body) {
            let content = resp
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();
            return (
                content,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            );
        }
    }
    // Surface the upstream error message as assistant content so the Ollama
    // client still receives a well-formed, terminal (`done:true`) turn.
    let msg = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str().map(str::to_string)))
        .unwrap_or_else(|| "generation unavailable".to_string());
    (msg, 0, 0)
}

/// Read an axum [`Response`] into `(status, body bytes)`.
async fn split_response(resp: Response) -> (StatusCode, axum::body::Bytes) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    (status, bytes)
}

// ============================================================================
// Handlers
// ============================================================================

/// `POST /api/chat` — Ollama chat endpoint.
///
/// Delegates generation to [`openai_chat_completions_handler`] and re-shapes the
/// result into Ollama's `{message:{role,content}, done}` schema.
pub async fn ollama_chat_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OllamaChatRequest>,
) -> Response {
    let model = model_label(&request.model);
    let chat_req = to_chat_request(&model, request.messages, &request.options);

    let inner = openai_chat_completions_handler(State(state), headers, Json(chat_req)).await;
    let (status, body) = split_response(inner).await;
    let (content, prompt_tokens, eval_count) = chat_response_to_parts(status, &body);

    Json(OllamaChatResponse {
        model,
        created_at: created_at_now(),
        message: OllamaMessage {
            role: "assistant".to_string(),
            content,
        },
        done: true,
        prompt_eval_count: prompt_tokens,
        eval_count,
    })
    .into_response()
}

/// `POST /api/generate` — Ollama single-prompt generate endpoint.
///
/// Folds `system` + `prompt` into a chat request and reuses the same generation
/// path, then emits Ollama's flat `{response, done}` schema.
pub async fn ollama_generate_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OllamaGenerateRequest>,
) -> Response {
    let model = model_label(&request.model);

    let mut messages = Vec::new();
    if let Some(system) = request.system.filter(|s| !s.is_empty()) {
        messages.push(OllamaMessage {
            role: "system".to_string(),
            content: system,
        });
    }
    messages.push(OllamaMessage {
        role: "user".to_string(),
        content: request.prompt,
    });

    let chat_req = to_chat_request(&model, messages, &request.options);
    let inner = openai_chat_completions_handler(State(state), headers, Json(chat_req)).await;
    let (status, body) = split_response(inner).await;
    let (content, prompt_tokens, eval_count) = chat_response_to_parts(status, &body);

    Json(OllamaGenerateResponse {
        model,
        created_at: created_at_now(),
        response: content,
        done: true,
        prompt_eval_count: prompt_tokens,
        eval_count,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn model_label_defaults_to_apr_when_absent() {
        assert_eq!(model_label(&None), "apr");
        assert_eq!(model_label(&Some(String::new())), "apr");
        assert_eq!(model_label(&Some("qwen".to_string())), "qwen");
    }

    #[test]
    fn to_chat_request_maps_messages_and_options() {
        let msgs = vec![
            OllamaMessage {
                role: "system".to_string(),
                content: "be brief".to_string(),
            },
            OllamaMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
        ];
        let opts = Some(OllamaOptions {
            temperature: Some(0.5),
            top_k: Some(10),
            num_predict: Some(32),
            ..Default::default()
        });
        let req = to_chat_request("m", msgs, &opts);
        assert_eq!(req.model, "m");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].content, "hi");
        assert_eq!(req.max_tokens, Some(32));
        assert_eq!(req.top_k, Some(10));
        assert!(!req.stream, "Ollama path always drives chat non-streaming");
    }

    #[test]
    fn chat_response_to_parts_extracts_content_on_success() {
        let body = br#"{
            "id":"x","object":"chat.completion","created":0,"model":"m",
            "choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
        }"#;
        let (content, p, c) = chat_response_to_parts(StatusCode::OK, body);
        assert_eq!(content, "4");
        assert_eq!(p, 3);
        assert_eq!(c, 1);
    }

    #[test]
    fn chat_response_to_parts_surfaces_error_as_content() {
        // On a backend error (e.g. no model), the Ollama body must still be
        // well-formed: the error text becomes the assistant content, tokens 0.
        let body = br#"{"error":"model not found"}"#;
        let (content, p, c) = chat_response_to_parts(StatusCode::NOT_FOUND, body);
        assert_eq!(content, "model not found");
        assert_eq!(p, 0);
        assert_eq!(c, 0);
    }

    #[test]
    fn ollama_chat_response_serializes_with_ollama_fields() {
        let resp = OllamaChatResponse {
            model: "apr".to_string(),
            created_at: created_at_now(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: "hello".to_string(),
            },
            done: true,
            prompt_eval_count: 1,
            eval_count: 2,
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["message"]["role"], "assistant");
        assert_eq!(json["message"]["content"], "hello");
        assert_eq!(json["done"], true);
    }

    #[test]
    fn ollama_generate_response_serializes_flat_response_field() {
        let resp = OllamaGenerateResponse {
            model: "apr".to_string(),
            created_at: created_at_now(),
            response: "hi".to_string(),
            done: true,
            prompt_eval_count: 0,
            eval_count: 1,
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["response"], "hi");
        assert_eq!(json["done"], true);
        assert!(json.get("message").is_none(), "generate uses flat response");
    }
}
