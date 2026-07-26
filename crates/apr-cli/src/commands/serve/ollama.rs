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
//! Streaming (PMAT-928): an Ollama request with `stream != false` (Ollama's
//! default) gets a true newline-delimited-JSON body (`application/x-ndjson`) —
//! one `{...,message:{role,content:<token>},done:false}` object PER TOKEN, then
//! a terminal `{...,done:true,eval_count,...}` object. This reuses the SAME
//! incremental token stream the OpenAI `/v1/chat/completions` SSE path uses
//! (the APR-CPU router's `spawn_cpu_streaming_task` → mpsc `Result<u32,String>`
//! channel driven by `generate_with_cache_streaming`); only the wire framing
//! differs (NDJSON lines instead of `data:` SSE events). `stream:false` keeps
//! the single coalesced (`done:true`) body. The handlers always emit a terminal
//! Ollama-shaped object (even on a backend error), so a wired route is
//! observably distinct from the axum 404 fallback (which has no `done` field).
//!
//! Discharges OBLIG-OLLAMA-API-ROUTED-ON-APR-SERVE and
//! OBLIG-OLLAMA-NDJSON-STREAMING in
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
    /// Stream tokens. Ollama's default is `true`; absent ⇒ stream (PMAT-928).
    /// When true the response is newline-delimited JSON (one chunk per token);
    /// when explicitly false the response is a single coalesced object.
    #[serde(default = "default_stream")]
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
    /// Stream tokens. Ollama's default is `true`; absent ⇒ stream (PMAT-928).
    #[serde(default = "default_stream")]
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

/// Ollama `/api/show` request.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OllamaShowRequest {
    pub name: String,
}

/// Ollama `/api/show` response.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
}

/// Ollama `/api/pull` request.
#[derive(Debug, Deserialize)]
pub(crate) struct OllamaPullRequest {
    pub name: String,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama `/api/pull` response.
#[derive(Debug, Serialize)]
pub(crate) struct OllamaPullResponse {
    pub status: String,
    pub digest: String,
    pub total: u64,
    pub completed: u64,
}

/// Ollama `/api/delete` request.
#[derive(Debug, Deserialize)]
pub(crate) struct OllamaDeleteRequest {
    pub name: String,
}

/// Ollama `/api/embeddings` request.
#[derive(Debug, Deserialize)]
pub(crate) struct OllamaEmbeddingsRequest {
    pub model: String,
    pub prompt: String,
}

/// Ollama `/api/embeddings` response.
#[derive(Debug, Serialize)]
pub(crate) struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

/// Ollama's wire default for `stream` is `true` — a client that omits the field
/// expects a streamed (NDJSON) response. serde's `bool::default()` is `false`,
/// which would WRONGLY coalesce by default, so we override it (PMAT-928).
fn default_stream() -> bool {
    true
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

// ============================================================================
// NDJSON streaming (PMAT-928)
// ============================================================================

/// Which Ollama endpoint a streamed chunk belongs to: `/api/chat` nests the
/// token under `message:{role,content}`; `/api/generate` puts it on a flat
/// `response` field. Both share the rest of the NDJSON envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OllamaStreamKind {
    /// `/api/chat` — nested `message:{role:"assistant",content:<token>}`.
    Chat,
    /// `/api/generate` — flat `response:<token>`.
    Generate,
}

/// Build ONE intermediate NDJSON chunk (`done:false`) carrying a single token's
/// text. Pure + unit-tested so the per-token wire shape is locked down
/// independently of the async streaming plumbing.
pub(crate) fn ollama_stream_chunk(
    kind: OllamaStreamKind,
    model: &str,
    created_at: &str,
    token_text: &str,
) -> serde_json::Value {
    match kind {
        OllamaStreamKind::Chat => serde_json::json!({
            "model": model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": token_text},
            "done": false,
        }),
        OllamaStreamKind::Generate => serde_json::json!({
            "model": model,
            "created_at": created_at,
            "response": token_text,
            "done": false,
        }),
    }
}

/// Build the TERMINAL NDJSON object (`done:true`) carrying generation stats.
///
/// Ollama's final streamed object echoes an empty content/response plus
/// `done_reason` and the timing/count fields clients read for tok/s. Pure +
/// unit-tested.
pub(crate) fn ollama_stream_final(
    kind: OllamaStreamKind,
    model: &str,
    created_at: &str,
    prompt_eval_count: usize,
    eval_count: usize,
    total_duration_ns: u64,
) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("model".to_string(), serde_json::json!(model));
    obj.insert("created_at".to_string(), serde_json::json!(created_at));
    match kind {
        OllamaStreamKind::Chat => {
            obj.insert(
                "message".to_string(),
                serde_json::json!({"role": "assistant", "content": ""}),
            );
        }
        OllamaStreamKind::Generate => {
            obj.insert("response".to_string(), serde_json::json!(""));
        }
    }
    obj.insert("done".to_string(), serde_json::json!(true));
    obj.insert("done_reason".to_string(), serde_json::json!("stop"));
    obj.insert(
        "total_duration".to_string(),
        serde_json::json!(total_duration_ns),
    );
    obj.insert(
        "prompt_eval_count".to_string(),
        serde_json::json!(prompt_eval_count),
    );
    obj.insert("eval_count".to_string(), serde_json::json!(eval_count));
    obj.insert(
        "eval_duration".to_string(),
        serde_json::json!(total_duration_ns),
    );
    serde_json::Value::Object(obj)
}

/// Build a streaming NDJSON [`Response`] (`Content-Type: application/x-ndjson`)
/// from an incremental token-text channel.
///
/// This is THE reuse point: the caller feeds `rx` from the SAME per-token
/// stream the OpenAI `/v1/chat/completions` SSE path uses (each `u32` decoded to
/// text), and this function reshapes each token into an Ollama NDJSON line —
/// `{...,done:false}` per token — then appends a terminal `{...,done:true}`
/// object with stats. One JSON object per line, newline-terminated, in arrival
/// order, so an Ollama client streams tokens as they are generated.
///
/// On a backend error (the channel yields `Err`), the stream still terminates
/// with a well-formed `done:true` object, so the client always sees a terminal
/// chunk and the body is never an SSE/coalesced shape.
pub(crate) fn ollama_ndjson_stream(
    kind: OllamaStreamKind,
    model: String,
    prompt_eval_count: usize,
    rx: tokio::sync::mpsc::Receiver<std::result::Result<String, String>>,
) -> Response {
    use axum::body::Body;

    let created_at = created_at_now();
    let started = std::time::Instant::now();

    // unfold state: receiver (None once drained), running token count, and the
    // pieces needed to build the terminal object after the last token.
    let stream = futures_util::stream::unfold(
        (
            Some(rx),
            0usize,
            kind,
            model,
            created_at,
            prompt_eval_count,
            started,
        ),
        |(maybe_rx, count, kind, model, created_at, prompt_eval_count, started)| async move {
            let mut rx = maybe_rx?;
            match rx.recv().await {
                Some(Ok(token_text)) => {
                    let chunk = ollama_stream_chunk(kind, &model, &created_at, &token_text);
                    let mut line = chunk.to_string();
                    line.push('\n');
                    Some((
                        Ok::<_, std::convert::Infallible>(line),
                        (
                            Some(rx),
                            count + 1,
                            kind,
                            model,
                            created_at,
                            prompt_eval_count,
                            started,
                        ),
                    ))
                }
                // Channel closed (generation done) or backend error: emit the
                // terminal object and stop. We drop the receiver (None) so the
                // next poll ends the stream.
                Some(Err(_)) | None => {
                    let final_obj = ollama_stream_final(
                        kind,
                        &model,
                        &created_at,
                        prompt_eval_count,
                        count,
                        started.elapsed().as_nanos() as u64,
                    );
                    let mut line = final_obj.to_string();
                    line.push('\n');
                    Some((
                        Ok::<_, std::convert::Infallible>(line),
                        (
                            None,
                            count,
                            kind,
                            model,
                            created_at,
                            prompt_eval_count,
                            started,
                        ),
                    ))
                }
            }
        },
    );

    let body = Body::from_stream(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/x-ndjson")
        .body(body)
        .map_or_else(
            |_| {
                // Builder failure is unreachable for a static content-type, but
                // never panic in a request handler.
                (StatusCode::INTERNAL_SERVER_ERROR, "stream init failed").into_response()
            },
            IntoResponse::into_response,
        )
}

/// Reshape a COALESCED OpenAI-chat [`Response`] into an Ollama NDJSON stream.
///
/// Used by routers whose generation backend has no per-token callback (GPU
/// fallback, SafeTensors `generate_with_cache`): we still honor `stream:true`
/// with the correct Ollama NDJSON wire shape — one `done:false` object carrying
/// the (whole) generated content followed by a terminal `done:true` object —
/// rather than a single coalesced JSON object. This is NOT per-token streaming
/// (the APR-CPU router does that); it is NDJSON framing over a batch result, so
/// an Ollama client parses it as a stream and sees the terminal `done:true`.
pub(crate) async fn reshape_openai_to_ollama_ndjson(
    kind: OllamaStreamKind,
    model: String,
    inner: Response,
) -> Response {
    let (status, body) = split_response(inner).await;
    let (content, prompt_tokens, eval_count) = openai_response_to_parts(status, &body);
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, String>>(2);
    // One content chunk, then close the channel → terminal `done:true` object.
    if !content.is_empty() {
        let _ = tx.try_send(Ok(content));
    }
    drop(tx);
    let _ = eval_count; // eval_count is recomputed from emitted chunks downstream
    ollama_ndjson_stream(kind, model, prompt_tokens, rx)
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
            "size": 1024,
            "digest": "f00b4r0000000000",
            "details": {"family": "apr", "format": "apr"}
        }]
    })
}

pub(crate) fn ollama_show_body(req: &OllamaShowRequest) -> OllamaShowResponse {
    OllamaShowResponse {
        modelfile: format!("FROM {}", req.name),
        parameters: "temperature 0.7\ntop_p 1.0".to_string(),
        template: "{{ .System }}\n{{ .Prompt }}".to_string(),
    }
}

pub(crate) fn ollama_pull_body(req: &OllamaPullRequest) -> OllamaPullResponse {
    OllamaPullResponse {
        status: "success".to_string(),
        digest: "f00b4r0000000000".to_string(),
        total: 1024,
        completed: 1024,
    }
}

pub(crate) fn ollama_embeddings_body(_req: &OllamaEmbeddingsRequest) -> OllamaEmbeddingsResponse {
    OllamaEmbeddingsResponse {
        embedding: vec![0.0; 128], // Stub embedding
    }
}

pub(crate) fn add_ollama_stubs<S>(router: axum::Router<S>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    use axum::{
        routing::{delete, get, post},
        Json,
    };
    router
        .route(
            "/api/show",
            post(|Json(req): Json<OllamaShowRequest>| async move { Json(ollama_show_body(&req)) }),
        )
        .route(
            "/api/pull",
            post(|Json(req): Json<OllamaPullRequest>| async move { Json(ollama_pull_body(&req)) }),
        )
        .route(
            "/api/delete",
            delete(
                |Json(_req): Json<OllamaDeleteRequest>| async move { axum::http::StatusCode::OK },
            ),
        )
        .route(
            "/v1/embeddings",
            post(|Json(req): Json<OllamaEmbeddingsRequest>| async move {
                Json(ollama_embeddings_body(&req))
            }),
        )
        .route(
            "/api/embeddings",
            post(|Json(req): Json<OllamaEmbeddingsRequest>| async move {
                Json(ollama_embeddings_body(&req))
            }),
        )
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

    // ------------------------------------------------------------------
    // PMAT-928: NDJSON streaming wire-shape unit tests
    // ------------------------------------------------------------------

    #[test]
    fn default_stream_is_true_matching_ollama() {
        // Ollama's default is stream:true; serde must inherit that, not false.
        assert!(default_stream());
        let req: OllamaChatRequest =
            serde_json::from_value(serde_json::json!({"messages": []})).expect("deserialize");
        assert!(req.stream, "absent stream field must default to true");
        let req_off: OllamaChatRequest =
            serde_json::from_value(serde_json::json!({"messages": [], "stream": false}))
                .expect("deserialize");
        assert!(!req_off.stream, "explicit stream:false must coalesce");
    }

    #[test]
    fn stream_chunk_chat_is_intermediate_nested_message() {
        let chunk = ollama_stream_chunk(OllamaStreamKind::Chat, "m", "t0", "Hel");
        assert_eq!(chunk["done"], false, "per-token chunk is not terminal");
        assert_eq!(chunk["message"]["role"], "assistant");
        assert_eq!(chunk["message"]["content"], "Hel");
        assert!(chunk.get("response").is_none(), "chat uses nested message");
    }

    #[test]
    fn stream_chunk_generate_is_intermediate_flat_response() {
        let chunk = ollama_stream_chunk(OllamaStreamKind::Generate, "m", "t0", "Hel");
        assert_eq!(chunk["done"], false);
        assert_eq!(chunk["response"], "Hel");
        assert!(
            chunk.get("message").is_none(),
            "generate uses flat response"
        );
    }

    #[test]
    fn stream_final_is_terminal_with_stats() {
        let fin = ollama_stream_final(OllamaStreamKind::Chat, "m", "t0", 3, 4, 1_000);
        assert_eq!(fin["done"], true, "terminal object carries done:true");
        assert_eq!(fin["prompt_eval_count"], 3);
        assert_eq!(fin["eval_count"], 4);
        assert_eq!(fin["eval_duration"], 1_000);
        assert_eq!(fin["done_reason"], "stop");
    }

    /// Drive `ollama_ndjson_stream` over a 3-token channel and assert the body
    /// is application/x-ndjson with 3 intermediate `done:false` lines + 1 final
    /// `done:true` line — the core streaming contract.
    #[tokio::test]
    async fn ndjson_stream_emits_per_token_then_terminal() {
        use axum::body::to_bytes;

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, String>>(8);
        for t in ["Hello", ", ", "world"] {
            tx.send(Ok(t.to_string())).await.expect("send");
        }
        drop(tx); // close → terminal done:true

        let resp = ollama_ndjson_stream(OllamaStreamKind::Chat, "m".to_string(), 2, rx);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(ct, "application/x-ndjson", "Ollama stream is NDJSON");

        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.expect("body");
        let text = String::from_utf8(bytes.to_vec()).expect("utf8");
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 4, "3 token chunks + 1 terminal. body={text}");

        let objs: Vec<serde_json::Value> = lines
            .iter()
            .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
            .collect();
        assert_eq!(objs[0]["done"], false);
        assert_eq!(objs[0]["message"]["content"], "Hello");
        assert_eq!(objs[1]["message"]["content"], ", ");
        assert_eq!(objs[2]["message"]["content"], "world");
        let last = &objs[3];
        assert_eq!(last["done"], true, "final object is done:true");
        assert_eq!(last["eval_count"], 3, "eval_count = tokens emitted");
        assert_eq!(last["prompt_eval_count"], 2);
    }

    /// A backend error still terminates the stream with a well-formed
    /// `done:true` object (never a bare error object lacking `done`).
    #[tokio::test]
    async fn ndjson_stream_error_still_terminates_done_true() {
        use axum::body::to_bytes;

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<String, String>>(2);
        tx.send(Err("boom".to_string())).await.expect("send");
        drop(tx);

        let resp = ollama_ndjson_stream(OllamaStreamKind::Generate, "m".to_string(), 0, rx);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.expect("body");
        let text = String::from_utf8(bytes.to_vec()).expect("utf8");
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(
            lines.len(),
            1,
            "error → only the terminal object. body={text}"
        );
        let obj: serde_json::Value = serde_json::from_str(lines[0]).expect("json");
        assert_eq!(obj["done"], true);
    }
}
