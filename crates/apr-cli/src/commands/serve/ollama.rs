//! Ollama-compatible HTTP shims for `apr serve` (PMAT-923).
//!
//! `apr serve <model>` builds its OWN bespoke axum routers (one per backend:
//! APR-CPU, GPU-fallback, WGPU, SafeTensors) — it does NOT mount realizar's
//! `create_router`. Those routers historically exposed only the OpenAI
//! `/v1/chat/completions` endpoint, so an Ollama HTTP client POSTing to
//! `/api/chat` or `/api/generate` hit the axum 404 fallback and `apr serve`
//! was NOT a drop-in Ollama replacement.
//!
//! This module supplies the reusable translation layer that each of those
//! routers wires in next to its `/v1/chat/completions` route:
//!
//! 1. [`ollama_chat_to_openai`] / [`ollama_generate_to_openai`] turn an Ollama
//!    request into the SAME OpenAI-chat JSON the router's existing chat handler
//!    already consumes (`messages`, `max_tokens`, `temperature`, ...).
//! 2. The router invokes ITS OWN chat handler (same generation backend as
//!    `/v1/chat/completions`) on that JSON, yielding an OpenAI-shaped
//!    [`Response`].
//! 3. [`reshape_openai_to_ollama_chat`] / [`reshape_openai_to_ollama_generate`]
//!    re-shape that response into Ollama's wire schema.
//!
//! Streaming is scoped HONESTLY: the Ollama endpoints always return a single
//! coalesced (`done:true`) body today; NDJSON `stream:true` is a documented
//! follow-up. The handlers always emit a terminal Ollama-shaped body (even on a
//! backend error), so a wired route is observably distinct from the axum 404
//! fallback (which has no `done` field).
//!
//! Discharges OBLIG-OLLAMA-API-ROUTED-ON-APR-SERVE in
//! `contracts/apr-serve-openai-compat-v1.yaml`.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Ollama wire types
// ============================================================================

/// Ollama `/api/chat` request.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OllamaChatRequest {
    /// Model tag. Optional — defaults to the loaded model.
    #[serde(default)]
    pub model: Option<String>,
    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<OllamaMessage>,
    /// Stream tokens (currently coalesced into a single final message).
    #[serde(default)]
    pub stream: bool,
    /// Optional Ollama `options` block (temperature, num_predict, top_k, ...).
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama message (`role` + `content`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OllamaMessage {
    /// "system" | "user" | "assistant".
    pub role: String,
    /// Message text.
    pub content: String,
}

/// Ollama `options` block (subset that maps onto our sampling config).
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OllamaOptions {
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Ollama's name for max tokens.
    #[serde(default)]
    pub num_predict: Option<u32>,
}

/// Ollama `/api/generate` request (single prompt, non-chat).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OllamaGenerateRequest {
    /// Model tag. Optional — defaults to the loaded model.
    #[serde(default)]
    pub model: Option<String>,
    /// The prompt to complete.
    #[serde(default)]
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

/// Ollama `/api/chat` response.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct OllamaChatResponse {
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

/// Ollama `/api/generate` response.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct OllamaGenerateResponse {
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
pub(crate) fn model_label(model: &Option<String>) -> String {
    model
        .clone()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "apr".to_string())
}

/// Translate an Ollama `options` block onto the OpenAI sampling fields the
/// existing `/v1/chat/completions` handlers already read off the JSON body.
fn apply_options(
    body: &mut serde_json::Map<String, serde_json::Value>,
    options: &Option<OllamaOptions>,
) {
    let Some(opts) = options else { return };
    if let Some(t) = opts.temperature {
        body.insert("temperature".to_string(), serde_json::json!(t));
    }
    if let Some(p) = opts.top_p {
        body.insert("top_p".to_string(), serde_json::json!(p));
    }
    if let Some(k) = opts.top_k {
        body.insert("top_k".to_string(), serde_json::json!(k));
    }
    if let Some(n) = opts.num_predict {
        body.insert("max_tokens".to_string(), serde_json::json!(n));
    }
}

/// Build the OpenAI-chat JSON body from an Ollama `/api/chat` request.
///
/// This is the single Ollama→internal translation point; the resulting body is
/// fed to the SAME chat handler the router uses for `/v1/chat/completions`, so
/// generation goes through one backend path for both protocols. `stream` is
/// forced off — we always coalesce into one final Ollama message.
pub(crate) fn ollama_chat_to_openai(req: &OllamaChatRequest) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::json!(model_label(&req.model)),
    );
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();
    body.insert("messages".to_string(), serde_json::json!(messages));
    body.insert("stream".to_string(), serde_json::json!(false));
    apply_options(&mut body, &req.options);
    serde_json::Value::Object(body)
}

/// Build the OpenAI-chat JSON body from an Ollama `/api/generate` request,
/// folding `system` + `prompt` into chat messages.
pub(crate) fn ollama_generate_to_openai(req: &OllamaGenerateRequest) -> serde_json::Value {
    let mut messages = Vec::new();
    if let Some(system) = req.system.as_ref().filter(|s| !s.is_empty()) {
        messages.push(serde_json::json!({"role": "system", "content": system}));
    }
    messages.push(serde_json::json!({"role": "user", "content": req.prompt}));

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::json!(model_label(&req.model)),
    );
    body.insert("messages".to_string(), serde_json::json!(messages));
    body.insert("stream".to_string(), serde_json::json!(false));
    apply_options(&mut body, &req.options);
    serde_json::Value::Object(body)
}

/// Extract `(content, prompt_tokens, completion_tokens)` from the OpenAI-chat
/// response JSON, or a fallback `(error_text, 0, 0)` when generation failed.
///
/// Crucially this ALWAYS yields parts for an Ollama-shaped body — even on a
/// backend error or a missing model — so a wired route is observably distinct
/// from the axum 404 fallback (which has no `done` field).
fn openai_response_to_parts(status: StatusCode, body: &[u8]) -> (String, usize, usize) {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) {
        if status.is_success() {
            if let Some(content) = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                let prompt_tokens = v
                    .get("usage")
                    .and_then(|u| u.get("prompt_tokens"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as usize;
                let completion_tokens = v
                    .get("usage")
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as usize;
                return (content.to_string(), prompt_tokens, completion_tokens);
            }
        }
        // Surface the upstream error message as assistant content so the Ollama
        // client still receives a well-formed, terminal (`done:true`) turn.
        if let Some(err) = v.get("error") {
            let msg = err
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| err.to_string());
            return (msg, 0, 0);
        }
    }
    ("generation unavailable".to_string(), 0, 0)
}

/// Read an axum [`Response`] into `(status, body bytes)`.
async fn split_response(resp: Response) -> (StatusCode, axum::body::Bytes) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    (status, bytes)
}

/// Re-shape an OpenAI-chat [`Response`] into an Ollama `/api/chat` body.
pub(crate) async fn reshape_openai_to_ollama_chat(model: String, inner: Response) -> Response {
    let (status, body) = split_response(inner).await;
    let (content, prompt_tokens, eval_count) = openai_response_to_parts(status, &body);
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

/// Re-shape an OpenAI-chat [`Response`] into an Ollama `/api/generate` body
/// (flat `response` field, no nested `message`).
pub(crate) async fn reshape_openai_to_ollama_generate(model: String, inner: Response) -> Response {
    let (status, body) = split_response(inner).await;
    let (content, prompt_tokens, eval_count) = openai_response_to_parts(status, &body);
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

/// `GET /api/tags` — Ollama model-list endpoint.
///
/// `apr serve` serves a single model, so we report exactly that model. Clients
/// (e.g. the Ollama CLI, OpenWebUI) hit `/api/tags` to enumerate models before
/// chatting; returning a one-entry list keeps them from erroring on startup.
pub(crate) fn ollama_tags_body(model: &str) -> serde_json::Value {
    serde_json::json!({
        "models": [{
            "name": model,
            "model": model,
            "modified_at": created_at_now(),
            "size": 0,
            "digest": "",
            "details": {"family": "apr", "format": "apr"}
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_label_defaults_to_apr_when_absent() {
        assert_eq!(model_label(&None), "apr");
        assert_eq!(model_label(&Some(String::new())), "apr");
        assert_eq!(model_label(&Some("qwen".to_string())), "qwen");
    }

    #[test]
    fn ollama_chat_to_openai_maps_messages_and_options() {
        let req = OllamaChatRequest {
            model: Some("m".to_string()),
            messages: vec![
                OllamaMessage {
                    role: "system".to_string(),
                    content: "be brief".to_string(),
                },
                OllamaMessage {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                },
            ],
            stream: true,
            options: Some(OllamaOptions {
                temperature: Some(0.5),
                top_k: Some(10),
                num_predict: Some(32),
                ..Default::default()
            }),
        };
        let body = ollama_chat_to_openai(&req);
        assert_eq!(body["model"], "m");
        assert_eq!(body["messages"].as_array().expect("messages").len(), 2);
        assert_eq!(body["messages"][1]["content"], "hi");
        assert_eq!(body["max_tokens"], 32);
        assert_eq!(body["top_k"], 10);
        // Always coalesce — drive the underlying chat path non-streaming.
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn ollama_generate_to_openai_folds_system_and_prompt() {
        let req = OllamaGenerateRequest {
            model: None,
            prompt: "2+2?".to_string(),
            system: Some("answer with a number".to_string()),
            stream: false,
            options: None,
        };
        let body = ollama_generate_to_openai(&req);
        assert_eq!(body["model"], "apr");
        let msgs = body["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "2+2?");
    }

    #[test]
    fn openai_response_to_parts_extracts_content_on_success() {
        let body = br#"{
            "id":"x","object":"chat.completion","created":0,"model":"m",
            "choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
        }"#;
        let (content, p, c) = openai_response_to_parts(StatusCode::OK, body);
        assert_eq!(content, "4");
        assert_eq!(p, 3);
        assert_eq!(c, 1);
    }

    #[test]
    fn openai_response_to_parts_surfaces_error_as_content() {
        let body = br#"{"error":"model not found"}"#;
        let (content, p, c) = openai_response_to_parts(StatusCode::NOT_FOUND, body);
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

    #[test]
    fn ollama_tags_body_lists_the_served_model() {
        let body = ollama_tags_body("qwen");
        let models = body["models"].as_array().expect("models");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "qwen");
    }
}
