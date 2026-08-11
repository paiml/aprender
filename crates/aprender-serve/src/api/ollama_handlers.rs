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
    ChatMessage, ModelSourceInfo,
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

/// One newline-delimited object of an Ollama `/api/chat` stream.
///
/// Non-terminal chunks carry a content fragment and `done:false`; the terminal
/// chunk carries empty content, `done:true`, `done_reason` and the token
/// counts. This is exactly the framing `ollama serve` puts on the wire, and
/// the framing every Ollama client's read loop expects.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatChunk {
    /// Model tag echoed back.
    pub model: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// Fragment of the assistant turn (empty on the terminal chunk).
    pub message: OllamaMessage,
    /// Terminal flag — false on every chunk but the last.
    pub done: bool,
    /// Why generation ended (terminal chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Prompt token count (terminal chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<usize>,
    /// Generated token count (terminal chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<usize>,
}

/// One newline-delimited object of an Ollama `/api/generate` stream.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateChunk {
    /// Model tag echoed back.
    pub model: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// Fragment of the completion (empty on the terminal chunk).
    pub response: String,
    /// Terminal flag — false on every chunk but the last.
    pub done: bool,
    /// Why generation ended (terminal chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Prompt token count (terminal chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<usize>,
    /// Generated token count (terminal chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<usize>,
}

// ============================================================================
// Ollama discovery wire types (`/api/tags`, `/api/show`, `/api/version`)
// ============================================================================

/// `details` block shared by `/api/tags` and `/api/show`.
///
/// Every field is `Option` and skipped when absent: a server that did not
/// measure a model's family or quantization must say nothing about it rather
/// than emit a plausible constant.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OllamaModelDetails {
    /// Container format (`gguf`, `apr`, `safetensors`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Model family (architecture, e.g. `qwen2`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Families list — Ollama clients read this as well as `family`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub families: Option<Vec<String>>,
    /// Human-readable parameter count (e.g. `1.5B`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_size: Option<String>,
    /// Quantization of the loaded weights (e.g. `Q4_K`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_level: Option<String>,
}

/// One entry of `GET /api/tags`.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaTag {
    /// `name:tag` the client should send back as `model`.
    pub name: String,
    /// Same value as `name` (Ollama emits both; clients read either).
    pub model: String,
    /// File mtime, RFC 3339. Omitted when unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    /// File size in bytes. Omitted when unknown — never 0 as a stand-in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Content digest. Omitted unless one was actually computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    /// Measured detail block.
    pub details: OllamaModelDetails,
}

/// `GET /api/tags` response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaTagsResponse {
    /// Models this server can serve.
    pub models: Vec<OllamaTag>,
}

/// `POST /api/show` request.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaShowRequest {
    /// Model tag (Ollama's newer field).
    #[serde(default)]
    pub model: Option<String>,
    /// Model tag (Ollama's older field). Accepted for compatibility.
    #[serde(default)]
    pub name: Option<String>,
}

/// `POST /api/show` response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaShowResponse {
    /// Measured detail block.
    pub details: OllamaModelDetails,
    /// Measured key/value metadata. Only keys this server actually measured.
    pub model_info: serde_json::Map<String, serde_json::Value>,
    /// What this model can do.
    pub capabilities: Vec<String>,
}

/// `GET /api/version` response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaVersionResponse {
    /// Server version.
    pub version: String,
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
        // The internal chat path has no token callback, so it is always driven
        // non-streaming. `stream:true` on the Ollama request is honoured at the
        // WIRE level instead — see `ndjson_response` — never discarded.
        stream: false,
        ..Default::default()
    }
}

/// Split generated text into the fragments an Ollama stream carries.
///
/// Invariant (asserted by the falsifiers): `content_fragments(s).concat() == s`
/// for every `s`. A client that concatenates the `content` of every chunk must
/// reconstruct the response byte-for-byte — that is the whole contract of the
/// streaming protocol, and dropping or duplicating a character breaks it.
///
/// Empty input yields no fragments: the response is then just the terminal
/// `done:true` object, which is what ollama does for an empty generation.
fn content_fragments(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    content
        .split_inclusive(char::is_whitespace)
        .map(str::to_string)
        .collect()
}

/// Serialize objects to newline-delimited JSON and send them as a streamed
/// body (`application/x-ndjson`, chunked — no `content-length`).
///
/// The shipped 0.63.0 returned ONE buffered JSON object with a `content-length`
/// no matter what `stream` said; every Ollama-compatible UI showed a frozen
/// cursor because its read loop waits for objects terminated by `done:true`.
fn ndjson_response<T: Serialize>(objects: &[T]) -> Response {
    let mut lines: Vec<axum::body::Bytes> = Vec::with_capacity(objects.len());
    for obj in objects {
        match serde_json::to_string(obj) {
            Ok(mut s) => {
                s.push('\n');
                lines.push(axum::body::Bytes::from(s));
            },
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            },
        }
    }

    let stream = tokio_stream::iter(
        lines
            .into_iter()
            .map(Ok::<axum::body::Bytes, std::convert::Infallible>),
    );

    match Response::builder()
        .status(StatusCode::OK)
        .header(
            axum::http::header::CONTENT_TYPE,
            "application/x-ndjson; charset=utf-8",
        )
        .body(axum::body::Body::from_stream(stream))
    {
        Ok(resp) => resp,
        // Unreachable: the only header is a constant. Fall back rather than panic.
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// Build the full `/api/chat` NDJSON stream for a completed generation.
fn chat_stream_objects(
    model: &str,
    content: &str,
    prompt_eval_count: usize,
    eval_count: usize,
) -> Vec<OllamaChatChunk> {
    let created_at = created_at_now();
    let mut out: Vec<OllamaChatChunk> = content_fragments(content)
        .into_iter()
        .map(|fragment| OllamaChatChunk {
            model: model.to_string(),
            created_at: created_at.clone(),
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: fragment,
            },
            done: false,
            done_reason: None,
            prompt_eval_count: None,
            eval_count: None,
        })
        .collect();
    out.push(OllamaChatChunk {
        model: model.to_string(),
        created_at,
        message: OllamaMessage {
            role: "assistant".to_string(),
            content: String::new(),
        },
        done: true,
        done_reason: Some("stop".to_string()),
        prompt_eval_count: Some(prompt_eval_count),
        eval_count: Some(eval_count),
    });
    out
}

/// Build the full `/api/generate` NDJSON stream for a completed generation.
fn generate_stream_objects(
    model: &str,
    content: &str,
    prompt_eval_count: usize,
    eval_count: usize,
) -> Vec<OllamaGenerateChunk> {
    let created_at = created_at_now();
    let mut out: Vec<OllamaGenerateChunk> = content_fragments(content)
        .into_iter()
        .map(|fragment| OllamaGenerateChunk {
            model: model.to_string(),
            created_at: created_at.clone(),
            response: fragment,
            done: false,
            done_reason: None,
            prompt_eval_count: None,
            eval_count: None,
        })
        .collect();
    out.push(OllamaGenerateChunk {
        model: model.to_string(),
        created_at,
        response: String::new(),
        done: true,
        done_reason: Some("stop".to_string()),
        prompt_eval_count: Some(prompt_eval_count),
        eval_count: Some(eval_count),
    });
    out
}

/// Render a parameter count the way Ollama does (`1.5B`, `370M`).
fn parameter_size_label(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.0}M", count as f64 / 1e6)
    } else {
        format!("{count}")
    }
}

/// The `name:tag` this server answers to, derived from the model file name.
///
/// Falls back to `apr` when no path was recorded. This is an addressable
/// identifier the client sends back as `model`, not a provenance claim.
fn tag_name(source: Option<&ModelSourceInfo>) -> String {
    let stem = source
        .and_then(ModelSourceInfo::path)
        .and_then(|p| {
            std::path::Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "apr".to_string());
    format!("{stem}:latest")
}

/// Build the `details` block from measured facts only.
fn details_from_source(source: Option<&ModelSourceInfo>) -> OllamaModelDetails {
    let Some(src) = source else {
        return OllamaModelDetails::default();
    };
    OllamaModelDetails {
        format: src.format().map(str::to_string),
        family: src.architecture().map(str::to_string),
        families: src.architecture().map(|a| vec![a.to_string()]),
        parameter_size: src.parameter_count().map(parameter_size_label),
        quantization_level: src.quantization().map(str::to_string),
    }
}

/// File mtime as RFC 3339, when the file is readable.
fn modified_at(source: Option<&ModelSourceInfo>) -> Option<String> {
    let path = source.and_then(ModelSourceInfo::path)?;
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(
        chrono::DateTime::<chrono::Utc>::from(modified)
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    )
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
    let stream = request.stream;
    let chat_req = to_chat_request(&model, request.messages, &request.options);

    let inner = openai_chat_completions_handler(State(state), headers, Json(chat_req)).await;
    let (status, body) = split_response(inner).await;
    let (content, prompt_tokens, eval_count) = chat_response_to_parts(status, &body);

    if stream {
        return ndjson_response(&chat_stream_objects(
            &model,
            &content,
            prompt_tokens,
            eval_count,
        ));
    }

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
    let stream = request.stream;

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

    if stream {
        return ndjson_response(&generate_stream_objects(
            &model,
            &content,
            prompt_tokens,
            eval_count,
        ));
    }

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

/// `GET /api/tags` — Ollama model enumeration.
///
/// Every Ollama-compatible client (the `ollama` CLI, Open WebUI, LangChain
/// `ChatOllama`, LlamaIndex) calls this BEFORE it will issue a single chat
/// request. 0.63.0 returned 404 while the startup banner advertised
/// "Ollama-Parity Endpoints", so none of those clients could reach the server
/// at all.
pub async fn ollama_tags_handler(State(state): State<AppState>) -> Json<OllamaTagsResponse> {
    let source = state.model_source();
    let name = tag_name(source);
    Json(OllamaTagsResponse {
        models: vec![OllamaTag {
            model: name.clone(),
            name,
            modified_at: modified_at(source),
            size: source.and_then(ModelSourceInfo::size_bytes),
            // Never invent a digest — omitted until one is computed.
            digest: None,
            details: details_from_source(source),
        }],
    })
}

/// `POST /api/show` — Ollama capability/metadata probe.
///
/// Reports only what the loader measured. Unknown keys are absent from
/// `model_info` rather than present with a plausible constant.
pub async fn ollama_show_handler(
    State(state): State<AppState>,
    Json(_request): Json<OllamaShowRequest>,
) -> Json<OllamaShowResponse> {
    let source = state.model_source();
    let mut model_info = serde_json::Map::new();
    if let Some(src) = source {
        if let Some(arch) = src.architecture() {
            model_info.insert(
                "general.architecture".to_string(),
                serde_json::Value::String(arch.to_string()),
            );
        }
        if let Some(q) = src.quantization() {
            model_info.insert(
                "general.quantization".to_string(),
                serde_json::Value::String(q.to_string()),
            );
        }
        if let Some(size) = src.size_bytes() {
            model_info.insert("general.size_bytes".to_string(), serde_json::json!(size));
        }
        // The context this server will actually serve (the KV-cache bound) is a
        // different fact from the model's advertised maximum. Report both, and
        // label them so neither can be mistaken for the other.
        if let Some(ctx) = src.context_length() {
            model_info.insert(
                "apr.configured_context_length".to_string(),
                serde_json::json!(ctx),
            );
        }
        if let Some(ctx) = src.model_max_context_length() {
            model_info.insert("general.context_length".to_string(), serde_json::json!(ctx));
        }
    }

    Json(OllamaShowResponse {
        details: details_from_source(source),
        model_info,
        capabilities: vec!["completion".to_string()],
    })
}

/// `GET /api/version` — server version.
///
/// Reports THIS server's version. It is not pretending to be some ollama
/// release; a client that version-gates gets a real, comparable semver.
pub async fn ollama_version_handler() -> Json<OllamaVersionResponse> {
    Json(OllamaVersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
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
        // The INTERNAL chat path has no token callback, so it is always driven
        // non-streaming. This says nothing about the wire: `stream:true` is
        // honoured by re-framing the finished generation as NDJSON (see the
        // `stream_*` falsifiers below). Asserting the internal flag is NOT a
        // licence to discard the client's flag.
        assert!(!req.stream, "internal chat path is driven non-streaming");
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

// ============================================================================
// Streaming + discovery falsifiers (dogfood 0.63.0, #2396)
// ============================================================================
//
// 0.63.0 parsed `stream` on both Ollama endpoints and then threw it away: the
// wire always carried ONE buffered JSON object with a `content-length`, so
// every ollama-compatible UI (ollama CLI, Open WebUI, continue.dev) sat on a
// frozen cursor for the whole generation. And `/api/tags`, `/api/show` and
// `/api/version` 404'd while the startup banner advertised "Ollama-Parity
// Endpoints" — clients call /api/tags BEFORE they will issue any chat request,
// so the server was unreachable to them.
#[cfg(test)]
mod stream_and_discovery_tests {
    use super::*;

    /// Concatenating the fragments must reproduce the input exactly. A client
    /// assembles the reply by joining every chunk's `content`; a dropped space
    /// or a duplicated word is a silently corrupted answer.
    #[test]
    fn fragments_reassemble_to_the_original_text() {
        for text in [
            "The capital of France is Paris.",
            "one",
            "  leading and trailing  ",
            "multi\nline\ttext with  double  spaces",
            "unicode: héllo wörld 日本語 🎉",
        ] {
            let joined: String = content_fragments(text).concat();
            assert_eq!(joined, text, "fragments must reassemble {text:?} exactly");
        }
    }

    /// Empty generation carries no content chunks — only the terminal object.
    #[test]
    fn empty_content_yields_no_fragments() {
        assert!(content_fragments("").is_empty());
    }

    /// A stream is a SEQUENCE terminated by `done:true`, and only the last
    /// object may be terminal. One object with `done:true` is what 0.63.0 sent
    /// and is indistinguishable from a non-streaming reply.
    #[test]
    fn chat_stream_is_a_sequence_terminated_by_done_true() {
        let objs = chat_stream_objects("apr", "The capital of France is Paris.", 21, 7);
        assert!(
            objs.len() > 2,
            "a 6-word answer must arrive as several chunks, got {}",
            objs.len()
        );
        let (last, rest) = objs.split_last().expect("non-empty");
        assert!(last.done, "final object must be done:true");
        assert_eq!(last.done_reason.as_deref(), Some("stop"));
        assert_eq!(last.prompt_eval_count, Some(21));
        assert_eq!(last.eval_count, Some(7));
        assert!(
            last.message.content.is_empty(),
            "ollama's terminal chat object carries no content"
        );
        for chunk in rest {
            assert!(!chunk.done, "only the last object may be done:true");
            assert!(chunk.done_reason.is_none());
            assert!(chunk.prompt_eval_count.is_none());
            assert!(chunk.eval_count.is_none());
            assert_eq!(chunk.message.role, "assistant");
        }
        let assembled: String = rest.iter().map(|c| c.message.content.as_str()).collect();
        assert_eq!(assembled, "The capital of France is Paris.");
    }

    /// Same contract on `/api/generate`, whose chunks carry a flat `response`.
    #[test]
    fn generate_stream_is_a_sequence_terminated_by_done_true() {
        let objs = generate_stream_objects("apr", "1 2 3 4 5 6 7 8", 16, 16);
        let (last, rest) = objs.split_last().expect("non-empty");
        assert!(last.done);
        assert_eq!(last.done_reason.as_deref(), Some("stop"));
        assert_eq!(last.eval_count, Some(16));
        assert!(last.response.is_empty());
        assert!(!rest.is_empty(), "must emit incremental chunks");
        assert!(rest.iter().all(|c| !c.done));
        let assembled: String = rest.iter().map(|c| c.response.as_str()).collect();
        assert_eq!(assembled, "1 2 3 4 5 6 7 8");
    }

    /// Even an empty generation must terminate the stream, or a client's read
    /// loop hangs waiting for `done:true`.
    #[test]
    fn empty_generation_still_terminates_the_stream() {
        let objs = chat_stream_objects("apr", "", 3, 0);
        assert_eq!(objs.len(), 1);
        assert!(objs[0].done);
    }

    /// Each object must be ONE line of valid JSON: NDJSON framing is the
    /// protocol, and an embedded newline splits one object into two garbage
    /// halves in the client's line reader.
    #[test]
    fn each_object_serializes_to_exactly_one_json_line() {
        for chunk in chat_stream_objects("apr", "a b\nc", 1, 3) {
            let line = serde_json::to_string(&chunk).expect("serialize");
            assert!(
                !line.contains('\n'),
                "raw newline would break NDJSON framing: {line}"
            );
            serde_json::from_str::<serde_json::Value>(&line).expect("each line is valid JSON");
        }
    }

    /// Non-terminal chunks must OMIT the counts rather than send zeros: a
    /// client that reads the first present `eval_count` would otherwise be
    /// told the model generated nothing.
    #[test]
    fn non_terminal_chunks_omit_counts_and_done_reason() {
        let objs = chat_stream_objects("apr", "two words", 5, 2);
        let first = serde_json::to_value(&objs[0]).expect("serialize");
        assert!(first.get("eval_count").is_none());
        assert!(first.get("prompt_eval_count").is_none());
        assert!(first.get("done_reason").is_none());
        assert_eq!(first["done"], false);
    }

    /// `created_at` must be RFC 3339 on streamed chunks too — the Go client
    /// decodes every chunk into the same `api.ChatResponse`.
    #[test]
    fn stream_chunk_created_at_is_rfc3339() {
        for chunk in chat_stream_objects("apr", "hi there", 1, 2) {
            let json = serde_json::to_value(&chunk).expect("serialize");
            let created_at = json["created_at"].as_str().expect("string");
            chrono::DateTime::parse_from_rfc3339(created_at)
                .unwrap_or_else(|e| panic!("chunk created_at {created_at:?} not RFC 3339: {e}"));
        }
    }

    /// A model without a recorded source must not acquire a family, a
    /// quantization or a size out of thin air.
    #[test]
    fn details_without_a_source_claim_nothing() {
        let details = details_from_source(None);
        let json = serde_json::to_value(details).expect("serialize");
        assert_eq!(
            json.as_object().map(serde_json::Map::len),
            Some(0),
            "unmeasured details must be absent, got {json}"
        );
    }

    /// With a measured source, `details` reports what was measured — and still
    /// omits what was not (no `parameter_size` here).
    #[test]
    fn details_report_measured_values_only() {
        let src = ModelSourceInfo::default()
            .with_quantization("Q4_K")
            .with_architecture("qwen2");
        let json = serde_json::to_value(details_from_source(Some(&src))).expect("serialize");
        assert_eq!(json["family"], "qwen2");
        assert_eq!(json["families"][0], "qwen2");
        assert_eq!(json["quantization_level"], "Q4_K");
        assert!(json.get("parameter_size").is_none());
        assert!(json.get("format").is_none());
    }

    /// The tag must be derived from the served file, so `ollama list` shows
    /// the model the operator actually launched.
    #[test]
    fn tag_name_comes_from_the_served_file() {
        let src = ModelSourceInfo::default();
        assert_eq!(tag_name(Some(&src)), "apr:latest");
        assert_eq!(tag_name(None), "apr:latest");

        let dir = std::env::temp_dir().join(format!("apr-tag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("qwen2.5-coder-1.5b.gguf");
        std::fs::write(&path, b"GGUF\0\0\0\0").expect("write");
        let src = ModelSourceInfo::from_path(&path);
        assert_eq!(tag_name(Some(&src)), "qwen2.5-coder-1.5b:latest");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parameter_size_labels_match_ollama_style() {
        assert_eq!(parameter_size_label(1_500_000_000), "1.5B");
        assert_eq!(parameter_size_label(370_000_000), "370M");
        assert_eq!(parameter_size_label(512), "512");
    }
}
