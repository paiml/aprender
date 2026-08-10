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

/// Ollama `/api/embeddings` request.
///
/// Ollama names the text field `prompt` (not `input`), and the response is a
/// single flat vector — this is the shape `OllamaEmbeddings` and Open WebUI send.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaEmbeddingsRequest {
    /// Model tag. Optional — defaults to the loaded model.
    #[serde(default)]
    pub model: Option<String>,
    /// Text to embed.
    pub prompt: String,
}

/// Ollama `/api/embeddings` response: one flat vector, no envelope.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaEmbeddingsResponse {
    /// The embedding vector (length == the model's hidden size).
    pub embedding: Vec<f32>,
}

// ============================================================================
// Conversion helpers (pure — unit-tested)
// ============================================================================

/// RFC 3339 timestamp for `created_at` (Ollama wire format).
///
/// Ollama's own client declares `CreatedAt` as a Go `time.Time`, so the value
/// goes through `time.Time.UnmarshalJSON`, which accepts RFC 3339 and nothing
/// else. A bare epoch with a `Z` glued on (`"1786293998.000000000Z"`) fails
/// that parse and `encoding/json` then discards the WHOLE response — message,
/// `done`, counts — not merely the timestamp. Emit UTC with nanosecond
/// precision, the shape real ollama puts on the wire.
fn created_at_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
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

/// `POST /api/embeddings` — Ollama embedding endpoint.
///
/// aprender#2396 finding 2: the startup banner advertises "Ollama-Parity
/// Endpoints" and every Ollama embedding client (Open WebUI's knowledge base,
/// LangChain `OllamaEmbeddings`, LlamaIndex) posts here — but the route was not
/// mounted at all, so they got the router's 404 and no embeddings.
///
/// Ollama's wire shape is a single `prompt` in and a single flat `embedding` out;
/// the numbers come from the SAME [`embed_inputs`](super::realize_handlers::embed_inputs)
/// path as `/realize/embed` and `/v1/embeddings`, so the three routes cannot
/// disagree about the same text on the same server.
pub async fn ollama_embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<OllamaEmbeddingsRequest>,
) -> Result<Json<OllamaEmbeddingsResponse>, (StatusCode, Json<super::ErrorResponse>)> {
    let input = super::EmbeddingInput::Single(request.prompt);
    let (embeddings, _prompt_tokens) = super::realize_handlers::embed_inputs(
        &state,
        request.model.as_deref(),
        &input,
        "/api/embeddings",
    )?;

    Ok(Json(OllamaEmbeddingsResponse {
        // Exactly one input went in, so exactly one vector comes back.
        embedding: embeddings.into_iter().next().unwrap_or_default(),
    }))
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

    // ------------------------------------------------------------------
    // `created_at` must be RFC 3339 (PMAT-923 follow-up).
    //
    // Ollama's Go client decodes `created_at` into a `time.Time`. A value it
    // cannot parse makes `encoding/json` abandon the WHOLE object, so the
    // client sees an empty message and `done:false` — not just a bad clock.
    // ------------------------------------------------------------------

    /// The oracle used below has to discriminate: the shape apr 0.63.0 emitted
    /// (a bare epoch with a `Z` glued on) MUST fail it, otherwise the
    /// assertions that follow would pass on the defect too.
    #[test]
    fn rfc3339_oracle_rejects_the_bare_epoch_shape() {
        chrono::DateTime::parse_from_rfc3339("1786293998.000000000Z")
            .expect_err("a bare epoch string is not RFC 3339 — oracle is not discriminating");
    }

    #[test]
    fn created_at_now_is_rfc3339_utc_at_the_current_instant() {
        let s = created_at_now();
        let parsed = chrono::DateTime::parse_from_rfc3339(&s)
            .unwrap_or_else(|e| panic!("created_at {s:?} must parse as RFC 3339: {e}"));

        // Parseable is not enough: it must denote *now*, so a decoding client
        // gets a usable instant rather than the zero time.
        let now = chrono::Utc::now().timestamp();
        let skew = (parsed.timestamp() - now).abs();
        assert!(skew <= 60, "created_at {s:?} is {skew}s away from now");
        assert_eq!(parsed.offset().local_minus_utc(), 0, "must be UTC: {s:?}");
    }

    /// The value that actually reaches the wire for `/api/chat`.
    #[test]
    fn chat_response_created_at_is_client_decodable() {
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
        let created_at = json["created_at"].as_str().expect("created_at is a string");
        chrono::DateTime::parse_from_rfc3339(created_at).unwrap_or_else(|e| {
            panic!("/api/chat created_at {created_at:?} must parse as RFC 3339: {e}")
        });
    }

    /// The value that actually reaches the wire for `/api/generate`.
    #[test]
    fn generate_response_created_at_is_client_decodable() {
        let resp = OllamaGenerateResponse {
            model: "apr".to_string(),
            created_at: created_at_now(),
            response: "hi".to_string(),
            done: true,
            prompt_eval_count: 0,
            eval_count: 1,
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        let created_at = json["created_at"].as_str().expect("created_at is a string");
        chrono::DateTime::parse_from_rfc3339(created_at).unwrap_or_else(|e| {
            panic!("/api/generate created_at {created_at:?} must parse as RFC 3339: {e}")
        });
    }
}
