//! OpenAI-compatible HTTP client for LLM inference endpoints.
//!
//! Supports chat completions against realizar, ollama, llama.cpp,
//! and any server exposing the OpenAI `/v1/chat/completions` API.

use serde::{Deserialize, Serialize};
use std::time::Duration;
#[cfg(feature = "llm")]
use std::time::Instant;

use crate::perf_gate::drain::StreamMode;

/// SSE streaming chunk from an OpenAI-compatible chat completion endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamDelta {
    /// Content fragment (may be empty or absent).
    pub content: Option<String>,
}

/// A single choice in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    /// The delta content for this choice.
    pub delta: StreamDelta,
    /// Finish reason (present on final chunk).
    pub finish_reason: Option<String>,
}

/// PP-LLAMA-001 v3.0 §3 — the phase timings the SERVER measured.
///
/// The key names are llama.cpp's (`timings{prompt_n, prompt_ms,
/// prompt_per_second, predicted_n, predicted_ms, predicted_per_second}`), and
/// `apr serve` emits the same block under the same names, so ONE parser serves
/// both lanes — which is the whole point of PP-25's one-client rule.
///
/// Every field is optional and unknown keys are ignored: llama.cpp carries
/// `prompt_per_token_ms` and friends that this harness does not read, and a
/// stricter parser would drop the entire block over a key it did not need.
/// `prefill` is `Σ prompt_tokens / Σ prompt_ms`, and PP-13 makes a
/// harness-computed substitute schema-fatal — so an absent block stays absent.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
pub struct ServerTimings {
    /// Prompt tokens processed in the prefill phase.
    #[serde(default)]
    pub prompt_n: Option<u64>,
    /// Wall-clock milliseconds the server spent in prefill.
    #[serde(default)]
    pub prompt_ms: Option<f64>,
    /// The server's own `prompt_n / prompt_ms`, in tokens per second.
    #[serde(default)]
    pub prompt_per_second: Option<f64>,
    /// Tokens produced in the decode phase.
    #[serde(default)]
    pub predicted_n: Option<u64>,
    /// Wall-clock milliseconds the server spent decoding.
    #[serde(default)]
    pub predicted_ms: Option<f64>,
    /// The server's own `predicted_n / predicted_ms`, in tokens per second.
    #[serde(default)]
    pub predicted_per_second: Option<f64>,
}

/// A streaming chunk response from the chat completion endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    /// Generated choices. `default` because OpenAI's `include_usage` terminal
    /// frame may carry `usage` and nothing else; refusing to parse that frame
    /// would lose exactly the token counts PP-27 requires.
    #[serde(default)]
    pub choices: Vec<StreamChoice>,
    /// Token usage. PP-27 requires it on the TERMINAL chunk; a stream that
    /// never carries one is refused by [`LlmClient::chat_completion_stream`].
    pub usage: Option<Usage>,
    /// PP-27 — how the server says this stream is produced, declared on the
    /// FIRST chunk. Absent on a server that does not declare it, which is
    /// recorded as `stream_mode: null` rather than assumed to be `live`.
    #[serde(default)]
    pub stream_mode: Option<StreamMode>,
    /// §3 — the server's phase timings, on the terminal chunk.
    #[serde(default)]
    pub timings: Option<ServerTimings>,
}

/// Result of a streaming chat completion with per-token timestamps.
#[derive(Debug, Clone)]
pub struct StreamedChatResponse {
    /// Concatenated response text.
    pub content: String,
    /// Total request duration.
    pub latency: Duration,
    /// Time to first token (first SSE data event with non-empty content).
    pub ttft: Duration,
    /// Timestamps of each token arrival relative to request start.
    pub token_timestamps: Vec<Duration>,
    /// Token usage, as the SERVER reported it on the terminal chunk.
    ///
    /// **Not an `Option`.** PP-27: "`usage` on the terminal chunk on both
    /// lanes; chunk-count fallback is a hard refusal." A chunk count is not a
    /// token count — an SSE frame carries whatever the server chose to flush —
    /// and while this field was optional the band producer silently substituted
    /// `token_timestamps.len()` for it. Making the field total means the
    /// substitution has nowhere to happen: a stream without terminal `usage`
    /// yields [`LlmClientError::StreamNoUsage`] and no response at all.
    pub usage: Usage,
    /// PP-27 — what the server declared on the first chunk, or `None` when it
    /// declared nothing.
    pub stream_mode: Option<StreamMode>,
    /// §3 — the server's phase timings, when it reported them.
    pub timings: Option<ServerTimings>,
    /// Why generation stopped (e.g., "stop", "length").
    pub finish_reason: Option<String>,
}

/// Chat message role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: Role,
    /// The content of the message.
    pub content: String,
}

/// Parameters for a chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    /// Model identifier (may be ignored by some backends).
    pub model: String,
    /// The messages for the chat completion.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature (0.0 = deterministic).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Sampling seed, sent as OpenAI-compatible `seed`.
    ///
    /// APR-PERF-GATE-001 v2.2 §4.4.4 requires the confidence interval to be
    /// "reproducible from retained samples". A run whose sampler was seeded
    /// from entropy is not reproducible, so a benchmark that cannot put the
    /// seed on the wire cannot satisfy §4.4.4 no matter what it retains. This
    /// field carried nothing before PERF-039: the struct had model, messages,
    /// temperature, max_tokens and stream, and `seed` reached the server on no
    /// request the harness could construct.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Suppress end-of-sequence stopping (`ignore_eos`).
    ///
    /// §4.3.1 pins W1 at `max_tokens = 128` with ignore-EOS. Without it the
    /// tokens generated per request are whatever the model decides to stop
    /// after, so the work per band is not pinned and an Arm A ratchet floor
    /// committed over it would drift with the model's stopping behaviour
    /// rather than with the server's throughput — a floor that moves for a
    /// reason the gate is not measuring.
    ///
    /// **Not an OpenAI standard field.** vLLM, SGLang and llama.cpp's server
    /// all accept it as an extension; a server that does not will, by serde's
    /// default, IGNORE it rather than reject it. See
    /// `realizar::api::ChatCompletionRequest::ignore_eos` for the receiving
    /// side and for which backends honour it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    /// OpenAI `stream_options`. Sent **only** on a streamed request.
    ///
    /// PP-27 makes a stream without terminal `usage` a hard refusal, and this
    /// client never asked for the frame: OpenAI, vLLM, SGLang and llama-server
    /// all emit `usage` on the final SSE chunk **only when
    /// `stream_options.include_usage` is set**. So the refusal fired on servers
    /// that would happily have answered, and the operator's only remedy was to
    /// stop streaming — which PP-27 also refuses. [`LlmClient::wire_request`]
    /// sets it on every streamed request.
    ///
    /// It must be absent when `stream` is not `true`: OpenAI rejects
    /// `stream_options` on a non-streaming request outright, so a field left
    /// set on the blocking path would turn a working lane into a 400.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

/// OpenAI `stream_options` — the ask that makes a server emit terminal usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StreamOptions {
    /// Emit a final chunk carrying `usage`.
    pub include_usage: bool,
}

impl StreamOptions {
    /// The only shape this harness sends: ask for the usage frame.
    #[must_use]
    pub const fn include_usage() -> Self {
        Self {
            include_usage: true,
        }
    }
}

impl ChatRequest {
    /// A request carrying only the two fields that have no default: which model
    /// and what to say. Every optional field is `None`.
    ///
    /// # Why this exists
    ///
    /// Every other constructor in this file builds on it with struct-update
    /// syntax (`ChatRequest { temperature, ..ChatRequest::new(..) }`). Before
    /// #2746, `chat_completion_stream` rebuilt the request field by field, so
    /// when `seed` and `ignore_eos` were added to this struct the streaming
    /// path silently kept sending `None` for both — the sampler was pinned on
    /// the non-streaming lane and unpinned on the streaming one, which is
    /// exactly the PP-28 defect. A struct update cannot drop a field: adding
    /// one to this type either lands here or fails to compile.
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            stream: None,
            seed: None,
            ignore_eos: None,
            stream_options: None,
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
}

/// A single completion choice.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponseChoice {
    /// Index of this choice.
    pub index: u32,
    /// The generated message.
    pub message: ChatMessage,
    /// Why generation stopped.
    pub finish_reason: Option<String>,
}

/// Response from a chat completion endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponse {
    /// Unique identifier for this completion.
    pub id: String,
    /// Object type (always "chat.completion").
    pub object: String,
    /// Unix timestamp of creation.
    pub created: u64,
    /// Model used.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatResponseChoice>,
    /// Token usage statistics.
    pub usage: Option<Usage>,
    /// Brick-level trace data (when X-Trace-Level: brick header is sent).
    #[serde(default)]
    pub brick_trace: Option<BrickTrace>,
}

/// Brick-level trace data from BrickProfiler (GH-114).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BrickTrace {
    /// Trace level (e.g., "brick").
    pub level: String,
    /// Number of operations traced.
    pub operations: usize,
    /// Total time in microseconds.
    pub total_time_us: u64,
    /// Per-operation timing breakdown.
    pub breakdown: Vec<BrickTraceOp>,
}

/// Individual traced operation from BrickProfiler.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BrickTraceOp {
    /// Operation name (e.g., "attention_qkv", "mlp_gate_up").
    pub name: String,
    /// Time in microseconds.
    pub time_us: u64,
    /// Additional details.
    #[serde(default)]
    pub details: Option<String>,
}

/// A chat response with timing metadata.
#[derive(Debug, Clone)]
pub struct TimedChatResponse {
    /// The API response.
    pub response: ChatResponse,
    /// Total request duration (time to last byte).
    pub latency: Duration,
    /// Time to first byte (approximation for non-streaming).
    pub ttfb: Duration,
    /// Brick trace data extracted from response (when trace_level was set).
    pub brick_trace: Option<BrickTrace>,
}

/// Errors from the LLM client.
#[cfg(feature = "llm")]
#[derive(Debug, thiserror::Error)]
pub enum LlmClientError {
    /// HTTP request failed.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// Server returned an error status.
    #[error("API error {status}: {body}")]
    ApiError {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
    /// Health check failed.
    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),
    /// Health check timed out waiting for server readiness.
    #[error("Health check timed out after {0:?}")]
    HealthCheckTimeout(Duration),
    /// PP-27 — the stream ended without a terminal `usage` block.
    ///
    /// A hard refusal, never a fallback. Counting SSE frames instead answers a
    /// different question: a frame carries whatever the server chose to flush,
    /// so `frames` and `completion_tokens` differ by an amount that depends on
    /// the server's buffering. Every metric derived from the substitute — `agg`,
    /// `dec`, `short_of_n_predict` — would then be wrong by that amount and
    /// look exactly like a measurement.
    #[error(
        "PP-27: the SSE stream from {url} ended with no terminal `usage` block after {frames} \
         content frames. Token counts are the server's (§3); a frame count is not a token count, \
         so this request is refused rather than counted from chunks. This client DOES ask for the \
         frame — every streamed request carries `stream_options: {{\"include_usage\": true}}` — so \
         either the server ignores it (use one that emits terminal usage: vLLM, SGLang, \
         llama-server, `apr serve`) or, for the legacy non-receipt `apr test llm bench`, pass \
         --no-stream and take the blocking path, whose usage comes back in the response body."
    )]
    StreamNoUsage {
        /// The endpoint that produced the stream.
        url: String,
        /// Content-bearing frames seen before the stream ended.
        frames: usize,
    },
    /// PP-27 — the stream carried no content-bearing chunk, so there is no
    /// first-token instant.
    ///
    /// Reported rather than defaulted: `ttft = e2e` is what the old
    /// `ttft.unwrap_or(latency)` produced, and a TTFT equal to the whole
    /// request is indistinguishable from a replayed stream.
    #[error(
        "PP-27: the SSE stream from {url} carried no content-bearing chunk, so there is no \
         first-token instant. Reporting ttft == e2e here would be indistinguishable from a \
         replayed stream"
    )]
    StreamNoContent {
        /// The endpoint that produced the stream.
        url: String,
    },
    /// A metadata GET answered with a non-2xx status that is not 404.
    ///
    /// 404 means the route is ABSENT — an older build — and is reported as
    /// `Ok(None)` so the caller can record a declared input. Anything else
    /// means the route exists and the server failed answering it, which is a
    /// different fact and must not be laundered into "this build is old".
    #[error(
        "GET {url} answered {status}, which is not 404: the route EXISTS and the server failed \
         answering it. 404 (route absent, older build) is the only status this client reads as \
         `not routed`. Body: {body}"
    )]
    RouteFailed {
        /// The endpoint that answered.
        url: String,
        /// The status it answered with.
        status: u16,
        /// Its body, verbatim.
        body: String,
    },
    /// A response body that was expected to be JSON did not parse as JSON.
    #[error("{url} returned a body that is not JSON: {source}")]
    NotJson {
        /// The endpoint that answered.
        url: String,
        /// The parse failure.
        source: serde_json::Error,
    },
}

/// OpenAI-compatible HTTP client for LLM inference.
#[cfg(feature = "llm")]
#[derive(Debug, Clone)]
pub struct LlmClient {
    base_url: String,
    client: reqwest::Client,
    model: String,
}

#[cfg(feature = "llm")]
impl LlmClient {
    /// Create a new client pointing at the given base URL.
    ///
    /// # Arguments
    /// * `base_url` - Base URL of the API server (e.g., `http://localhost:8081`)
    /// * `model` - Model name to include in requests
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client,
            model: model.into(),
        }
    }

    /// Create a client with a custom reqwest client (for custom timeouts, etc.).
    pub fn with_client(
        base_url: impl Into<String>,
        model: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client,
            model: model.into(),
        }
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// PP-28 — the request exactly as it goes on the wire.
    ///
    /// One place, so every transport puts the SAME body on the wire: the
    /// caller's request with the client's model substituted when the request
    /// named none, and `stream` forced when the transport requires a setting.
    /// Built with struct-update syntax, so a field added to [`ChatRequest`]
    /// travels on every path by construction rather than by review — the exact
    /// discipline whose absence dropped `seed` and `ignore_eos` from the
    /// streaming lane while the blocking lane carried both.
    #[must_use]
    pub fn wire_request(&self, request: &ChatRequest, stream: Option<bool>) -> ChatRequest {
        let stream = stream.or(request.stream);
        ChatRequest {
            model: if request.model.is_empty() {
                self.model.clone()
            } else {
                request.model.clone()
            },
            stream,
            // PP-27: a streamed request ALWAYS asks for the terminal usage
            // frame, and a non-streamed one never carries the field (OpenAI
            // rejects `stream_options` without `stream: true`). Derived from
            // the resolved `stream` here, once, so no caller can send a stream
            // that forgot to ask and then be refused for the server's silence.
            stream_options: if stream == Some(true) {
                Some(StreamOptions::include_usage())
            } else {
                None
            },
            ..request.clone()
        }
    }

    /// `GET {base_url}{path}`, returning the body verbatim as JSON, or `None`
    /// when the server answered a non-success status.
    ///
    /// PP-2 stores `GET /v1/effective-config` **verbatim**, and §5.3 stores the
    /// comparator's `GET /props` verbatim per band. Both go through this one
    /// client, which is also what PP-25 asserts: one client binary drives both
    /// lanes, and its digest is in the receipt.
    ///
    /// **404 alone** is `Ok(None)` — an older build simply does not route the
    /// path, and the caller records that as a declared input rather than a
    /// failure. Every other non-2xx is an `Err` naming the status: a 500 from
    /// `/v1/effective-config` means the route EXISTS and the server failed
    /// answering it, and reading that as "this build is older than the route"
    /// silently converts a broken server into a declared-provenance receipt —
    /// the receipt then says the compute class and feature set were the
    /// operator's declarations, and no one learns the server was on fire. A
    /// transport error is `Err` for the same reason: unreachable is not absent.
    ///
    /// # Errors
    /// On any transport failure, on any non-2xx status other than 404, or when
    /// a 2xx body does not parse as JSON.
    #[cfg(feature = "llm")]
    pub async fn get_json(&self, path: &str) -> Result<Option<serde_json::Value>, LlmClientError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmClientError::RouteFailed {
                url,
                status: status.as_u16(),
                body,
            });
        }
        let body = resp.text().await?;
        serde_json::from_str(&body)
            .map(Some)
            .map_err(|source| LlmClientError::NotJson { url, source })
    }

    /// Send a chat completion request and return the response with timing.
    pub async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<TimedChatResponse, LlmClientError> {
        // PP-28: struct update over the single base constructor. This path has
        // no parameter for `seed` or `ignore_eos`, so it cannot pin the sampler
        // and does not pretend to — but when a field is added to `ChatRequest`,
        // it arrives here through `new` instead of being silently dropped. The
        // band producer does not use this method; it builds the pinned request
        // from the corpus and sends it through `send` / `chat_completion_stream`.
        let request = ChatRequest {
            temperature,
            max_tokens,
            stream: Some(false),
            ..ChatRequest::new(self.model.clone(), messages)
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let start = Instant::now();

        let resp = self.client.post(&url).json(&request).send().await?;
        let ttfb = start.elapsed();

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmClientError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let response: ChatResponse = resp.json().await?;
        let latency = start.elapsed();
        let brick_trace = response.brick_trace.clone();

        Ok(TimedChatResponse {
            response,
            latency,
            ttfb,
            brick_trace,
        })
    }

    /// Send a raw `ChatRequest` and return the timed response.
    pub async fn send(&self, request: &ChatRequest) -> Result<TimedChatResponse, LlmClientError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let start = Instant::now();

        let req = self.wire_request(request, None);

        let resp = self.client.post(&url).json(&req).send().await?;
        let ttfb = start.elapsed();

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmClientError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let response: ChatResponse = resp.json().await?;
        let latency = start.elapsed();
        let brick_trace = response.brick_trace.clone();

        Ok(TimedChatResponse {
            response,
            latency,
            ttfb,
            brick_trace,
        })
    }

    /// Send a raw `ChatRequest` with X-Trace-Level header.
    pub async fn send_with_trace(
        &self,
        request: &ChatRequest,
        trace_level: &str,
    ) -> Result<TimedChatResponse, LlmClientError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let start = Instant::now();

        let req = self.wire_request(request, None);

        let resp = self
            .client
            .post(&url)
            .header("X-Trace-Level", trace_level)
            .json(&req)
            .send()
            .await?;
        let ttfb = start.elapsed();

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmClientError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let response: ChatResponse = resp.json().await?;
        let latency = start.elapsed();
        let brick_trace = response.brick_trace.clone();

        Ok(TimedChatResponse {
            response,
            latency,
            ttfb,
            brick_trace,
        })
    }

    /// Check if the server is reachable by hitting common health endpoints.
    pub async fn health_check(&self) -> Result<bool, LlmClientError> {
        // Try /health, /v1/models, then root
        for path in &["/health", "/v1/models", "/"] {
            let url = format!("{}{path}", self.base_url);
            if let Ok(resp) = self.client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(true);
                }
            }
        }
        Err(LlmClientError::HealthCheckFailed(format!(
            "No health endpoint responded at {}",
            self.base_url
        )))
    }

    /// Send a streaming chat completion request and collect per-token timestamps.
    ///
    /// Sends `stream: true` and parses SSE `data: {...}` events. Records
    /// the arrival time of each content-bearing chunk for TPOT computation.
    pub async fn chat_completion_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<StreamedChatResponse, LlmClientError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        // PP-28: a struct update over the caller's request, so `temperature`,
        // `max_tokens`, `seed` and `ignore_eos` all reach the wire. The
        // field-by-field copy this replaces listed `seed: None, ignore_eos:
        // None` and therefore un-pinned the sampler on every streamed request
        // while the blocking path carried both.
        let stream_request = self.wire_request(request, Some(true));

        let start = Instant::now();
        let resp = self.client.post(&url).json(&stream_request).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmClientError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let mut content = String::new();
        let mut token_timestamps = Vec::new();
        let mut ttft = None;
        let mut final_usage = None;
        let mut finish_reason = None;
        let mut stream_mode = None;
        let mut timings = None;

        // Read the response incrementally via chunk() for real per-token timestamps.
        // Each chunk() call returns data as it arrives from the server, so timestamps
        // reflect actual token delivery times rather than full-response download time.
        let mut resp = resp;
        let mut buffer = String::new();
        let mut done = false;

        while !done {
            match resp.chunk().await? {
                Some(chunk_bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));
                }
                None => {
                    done = true;
                }
            }

            // Process complete lines from buffer
            while let Some(newline_pos) = buffer.find('\n') {
                let line: String = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line == "data: [DONE]" {
                    done = true;
                    break;
                }
                if let Some(json_str) = line.strip_prefix("data: ") {
                    if let Ok(sse_chunk) = serde_json::from_str::<StreamChunk>(json_str) {
                        if let Some(choice) = sse_chunk.choices.first() {
                            if let Some(ref c) = choice.delta.content {
                                if !c.is_empty() {
                                    let now = start.elapsed();
                                    if ttft.is_none() {
                                        ttft = Some(now);
                                    }
                                    token_timestamps.push(now);
                                    content.push_str(c);
                                }
                            }
                            if choice.finish_reason.is_some() {
                                finish_reason = choice.finish_reason.clone();
                            }
                        }
                        // PP-27: the server declares the mechanism on the
                        // FIRST chunk. Later chunks do not carry it, and a
                        // later one that did must not overwrite the first --
                        // the declaration is a property of the stream.
                        if stream_mode.is_none() {
                            stream_mode = sse_chunk.stream_mode;
                        }
                        if sse_chunk.usage.is_some() {
                            final_usage = sse_chunk.usage;
                        }
                        if sse_chunk.timings.is_some() {
                            timings = sse_chunk.timings;
                        }
                    }
                }
            }
        }

        let latency = start.elapsed();

        // PP-27, both refusals. Neither has a fallback, because both fallbacks
        // produce a number with the shape of a measurement: a chunk count that
        // is not a token count, and a `ttft` equal to `e2e` that is exactly
        // what a REPLAYED stream looks like.
        let usage = final_usage.ok_or_else(|| LlmClientError::StreamNoUsage {
            url: url.clone(),
            frames: token_timestamps.len(),
        })?;
        let ttft = ttft.ok_or(LlmClientError::StreamNoContent { url })?;

        Ok(StreamedChatResponse {
            content,
            latency,
            ttft,
            token_timestamps,
            usage,
            stream_mode,
            timings,
            finish_reason,
        })
    }

    /// Poll the server until it becomes ready or the timeout expires.
    ///
    /// Returns the time elapsed until the server was ready.
    pub async fn wait_ready(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<Duration, LlmClientError> {
        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(LlmClientError::HealthCheckTimeout(timeout));
            }
            if self.health_check().await.is_ok() {
                return Ok(start.elapsed());
            }
            tokio::time::sleep(poll_interval).await;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[cfg(feature = "llm")]
    #[test]
    fn test_client_creation() {
        let client = LlmClient::new("http://localhost:8081", "qwen-coder");
        assert_eq!(client.base_url(), "http://localhost:8081");
        assert_eq!(client.model(), "qwen-coder");
    }

    #[cfg(feature = "llm")]
    #[test]
    fn test_client_strips_trailing_slash() {
        let client = LlmClient::new("http://localhost:8081/", "model");
        assert_eq!(client.base_url(), "http://localhost:8081");
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: Role::User,
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
    }

    #[test]
    fn test_chat_request_serialization() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "Hi".to_string(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(32),
            stream: None,
            seed: None,
            ignore_eos: None,
            stream_options: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"temperature\":0.0"));
        assert!(json.contains("\"max_tokens\":32"));
        // stream is None, should be omitted
        assert!(!json.contains("stream"));
    }

    #[test]
    fn test_chat_request_omits_none_fields() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: None,
            seed: None,
            ignore_eos: None,
            stream_options: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("stream"));
    }

    #[test]
    fn test_chat_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "qwen-coder",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "chatcmpl-123");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, "Hello!");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_apr_response_deserialization() {
        let json = r#"{"_apr_metrics":{"latency_ms":1978,"tok_per_sec":4.14},"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hello","role":"assistant"}}],"created":1772386202,"id":"chatcmpl-123","model":"test","object":"chat.completion","usage":{"completion_tokens":8,"prompt_tokens":9,"total_tokens":17}}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content, "hello");
    }

    #[test]
    fn test_gguf_response_with_name_null() {
        let json = r#"{"id":"chatcmpl-q4k-123","object":"chat.completion","created":1772385841,"model":"qwen","choices":[{"index":0,"message":{"role":"assistant","content":"4","name":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":1,"total_tokens":25}}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content, "4");
    }

    #[test]
    fn test_chat_response_without_usage() {
        let json = r#"{
            "id": "abc",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": []
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage.is_none());
        assert!(resp.choices.is_empty());
    }

    #[test]
    fn test_role_serialization_roundtrip() {
        for (role, expected) in [
            (Role::System, "\"system\""),
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, expected);
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(back, role);
        }
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[cfg(feature = "llm")]
    #[test]
    fn test_client_with_custom_client() {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let client = LlmClient::with_client("http://example.com", "model", http);
        assert_eq!(client.base_url(), "http://example.com");
    }

    #[cfg(feature = "llm")]
    #[test]
    fn test_health_check_timeout_error_display() {
        let err = LlmClientError::HealthCheckTimeout(Duration::from_secs(30));
        let msg = err.to_string();
        assert!(msg.contains("30"));
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        // GH-24: Parse SSE streaming chunk from OpenAI-compatible API
        let json = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_stream_chunk_final_with_usage() {
        // GH-24: Final streaming chunk with usage stats
        let json = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(chunk.choices[0].delta.content.is_none());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_stream_chunk_empty_content() {
        // GH-24: Chunk with empty content (role-only delta)
        let json = r#"{"choices":[{"delta":{"content":""},"finish_reason":null}]}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some(""));
    }

    /// PP-27 — a streamed request ALWAYS asks for the terminal usage frame,
    /// and a blocking one never carries the field.
    ///
    /// `StreamNoUsage` is a hard refusal and the client never asked: OpenAI,
    /// vLLM, SGLang and llama-server all emit `usage` on the final SSE chunk
    /// **only** when `stream_options.include_usage` is set. So the refusal
    /// fired on servers that would have answered, and the operator's only
    /// remedy — stop streaming — is the other thing PP-27 refuses.
    ///
    /// The negative half matters just as much: OpenAI rejects `stream_options`
    /// on a non-streaming request, so leaving it set on the blocking path would
    /// turn a working lane into a 400.
    #[cfg(feature = "llm")]
    #[test]
    fn a_streamed_wire_request_asks_for_the_usage_frame_and_a_blocking_one_does_not() {
        let client = LlmClient::new("http://127.0.0.1:8081", "qwen");
        let base = ChatRequest::new("qwen", vec![]);

        let streamed = client.wire_request(&base, Some(true));
        assert_eq!(
            streamed.stream_options,
            Some(StreamOptions::include_usage()),
            "a streamed request must ask for terminal usage"
        );
        let json = serde_json::to_string(&streamed).expect("serialises");
        assert!(
            json.contains(r#""stream_options":{"include_usage":true}"#),
            "and it must be on the WIRE: {json}"
        );

        for blocking in [
            client.wire_request(&base, Some(false)),
            client.wire_request(&base, None),
        ] {
            assert_eq!(
                blocking.stream_options, None,
                "a non-streamed request must not carry stream_options (OpenAI 400s on it)"
            );
            let json = serde_json::to_string(&blocking).expect("serialises");
            assert!(!json.contains("stream_options"), "{json}");
        }

        // The caller's own `stream: Some(true)` reaches the same conclusion:
        // the ask is derived from the RESOLVED stream flag, not from the
        // argument, so no path can stream without asking.
        let self_declared = ChatRequest {
            stream: Some(true),
            ..ChatRequest::new("qwen", vec![])
        };
        assert_eq!(
            client.wire_request(&self_declared, None).stream_options,
            Some(StreamOptions::include_usage())
        );
    }

    #[test]
    fn test_chat_request_with_stream_true() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: Some(true),
            seed: None,
            ignore_eos: None,
            stream_options: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"stream\":true"));
    }

    // ======================================================================
    // PP-27 / PP-28 — what actually goes on the wire, and what a stream must
    // carry before its timings may be believed.
    //
    // Until #2746 there was NO test that invoked `chat_completion_stream` and
    // NO test that asserted what the streaming request body contains. The two
    // gaps compounded: the streaming path rebuilt the request field by field
    // and therefore dropped `seed` and `ignore_eos`, and nothing noticed
    // because the only coverage was struct-level serde on a hand-built value.
    //
    //  case                                   | must  | why
    //  ---------------------------------------|-------|--------------------
    //  seed/ignore_eos on the stream body     | OK    | PP-28 sampler pin
    //  seed/ignore_eos on the blocking body   | OK    | same pin, both lanes
    //  terminal `usage` absent    [MUST-FIRE] | ERR   | PP-27 no fallback
    //  no content chunk           [MUST-FIRE] | ERR   | PP-27 no ttft==e2e
    //  first-chunk stream_mode + timings      | OK    | PP-27 / §3 capture
    // ======================================================================

    /// One prompt with the §5.1 sampler pinned on it, as the W1 corpus builds.
    #[cfg(feature = "llm")]
    fn pinned_request() -> ChatRequest {
        ChatRequest {
            temperature: Some(0.0),
            max_tokens: Some(128),
            stream: Some(false),
            seed: Some(0),
            ignore_eos: Some(true),
            ..ChatRequest::new(
                String::new(),
                vec![ChatMessage {
                    role: Role::User,
                    content: "// w1-0000".to_string(),
                }],
            )
        }
    }

    /// PP-28 — the STREAM request keeps the sampler pin.
    ///
    /// RED before the struct-update fix: the old `chat_completion_stream`
    /// listed `seed: None, ignore_eos: None` explicitly, so this body reached
    /// the server with the sampler unpinned on every streamed request.
    #[cfg(feature = "llm")]
    #[test]
    fn stream_request_keeps_seed_and_ignore_eos() {
        let client = LlmClient::new("http://127.0.0.1:1", "qwen-coder");
        let wire = client.wire_request(&pinned_request(), Some(true));
        let json = serde_json::to_string(&wire).unwrap();
        assert!(json.contains("\"seed\":0"), "{json}");
        assert!(json.contains("\"ignore_eos\":true"), "{json}");
        assert!(json.contains("\"temperature\":0.0"), "{json}");
        assert!(json.contains("\"max_tokens\":128"), "{json}");
        assert!(json.contains("\"stream\":true"), "{json}");
        // The empty model is substituted with the client's, so a corpus record
        // that names no model is still served by the model under measurement.
        assert!(json.contains("\"model\":\"qwen-coder\""), "{json}");
    }

    /// PP-28 — the BLOCKING request keeps the same pin, and does NOT acquire a
    /// `stream: true` it was not asked for.
    #[cfg(feature = "llm")]
    #[test]
    fn blocking_request_keeps_seed_and_ignore_eos() {
        let client = LlmClient::new("http://127.0.0.1:1", "qwen-coder");
        let wire = client.wire_request(&pinned_request(), None);
        let json = serde_json::to_string(&wire).unwrap();
        assert!(json.contains("\"seed\":0"), "{json}");
        assert!(json.contains("\"ignore_eos\":true"), "{json}");
        assert!(json.contains("\"stream\":false"), "{json}");
    }

    /// DISCRIMINATION for the model substitution: a request that NAMES a model
    /// keeps it. Without this, `wire_request` could always write the client's
    /// model and both assertions above would still pass.
    #[cfg(feature = "llm")]
    #[test]
    fn a_request_that_names_a_model_keeps_it() {
        let client = LlmClient::new("http://127.0.0.1:1", "client-model");
        let named = ChatRequest {
            ..ChatRequest::new("request-model", vec![])
        };
        assert_eq!(client.wire_request(&named, None).model, "request-model");
        let anonymous = ChatRequest::new(String::new(), vec![]);
        assert_eq!(client.wire_request(&anonymous, None).model, "client-model");
    }

    // --- loopback SSE probe -------------------------------------------------

    /// What one SSE probe should emit. Each field is a separate defect shape:
    /// a stream with no `usage`, a stream with no content, a stream that
    /// declares nothing.
    #[cfg(feature = "llm")]
    #[derive(Clone, Copy)]
    struct SseScript {
        content_chunks: usize,
        declare_mode: Option<&'static str>,
        terminal_usage: bool,
        timings: bool,
    }

    #[cfg(feature = "llm")]
    impl SseScript {
        /// The PP-27 conformant shape: declared live, content, terminal usage
        /// and server timings.
        fn conformant() -> Self {
            Self {
                content_chunks: 4,
                declare_mode: Some("live"),
                terminal_usage: true,
                timings: true,
            }
        }
    }

    /// Serve one scripted SSE response and hand the request body back.
    #[cfg(feature = "llm")]
    async fn serve_scripted_sse(
        mut sock: tokio::net::TcpStream,
        script: SseScript,
        bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut buf = vec![0_u8; 8192];
        let mut seen = Vec::new();
        loop {
            let Ok(n) = sock.read(&mut buf).await else {
                return;
            };
            if n == 0 {
                return;
            }
            seen.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&seen).to_string();
            let Some(head_end) = text.find("\r\n\r\n") else {
                continue;
            };
            let len: usize = text
                .to_lowercase()
                .split("content-length:")
                .nth(1)
                .and_then(|t| t.split("\r\n").next())
                .and_then(|t| t.trim().parse().ok())
                .unwrap_or(0);
            if seen.len() >= head_end + 4 + len {
                let body = String::from_utf8_lossy(&seen[head_end + 4..]).to_string();
                bodies.lock().unwrap_or_else(|e| e.into_inner()).push(body);
                break;
            }
        }

        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                    Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if sock.write_all(head.as_bytes()).await.is_err() {
            return;
        }
        let mode = script
            .declare_mode
            .map_or(String::new(), |m| format!(",\"stream_mode\":\"{m}\""));
        let first = format!(
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}}}}]{mode}}}\n\n"
        );
        let _ = sock.write_all(first.as_bytes()).await;
        for i in 0..script.content_chunks {
            tokio::time::sleep(Duration::from_millis(5)).await;
            let chunk = format!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"t{i} \"}}}}]}}\n\n"
            );
            if sock.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
        }
        let timings = if script.timings {
            ",\"timings\":{\"prompt_n\":512,\"prompt_ms\":40.0,\"prompt_per_second\":12800.0,\
             \"predicted_n\":4,\"predicted_ms\":20.0,\"predicted_per_second\":200.0,\
             \"clock\":\"server std::time::Instant (CLOCK_MONOTONIC)\"}"
        } else {
            ""
        };
        if script.terminal_usage {
            let terminal = format!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"length\"}}],\
                 \"usage\":{{\"prompt_tokens\":512,\"completion_tokens\":128,\"total_tokens\":640}}\
                 {timings}}}\n\n"
            );
            let _ = sock.write_all(terminal.as_bytes()).await;
        } else {
            let terminal = "data: {\"choices\":[{\"index\":0,\"delta\":{},\
                            \"finish_reason\":\"length\"}]}\n\n";
            let _ = sock.write_all(terminal.as_bytes()).await;
        }
        let _ = sock.write_all(b"data: [DONE]\n\n").await;
        let _ = sock.flush().await;
        let _ = sock.shutdown().await;
    }

    /// A loopback endpoint serving `script`, plus the bodies it received.
    #[cfg(feature = "llm")]
    async fn spawn_scripted_sse(
        script: SseScript,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&bodies);
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(serve_scripted_sse(
                    sock,
                    script,
                    std::sync::Arc::clone(&sink),
                ));
            }
        });
        (format!("http://{addr}"), bodies)
    }

    /// A loopback endpoint that answers every GET with one fixed status.
    #[cfg(feature = "llm")]
    async fn spawn_status_endpoint(status_line: &'static str, body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0_u8; 2048];
                    let _ = sock.read(&mut buf).await;
                    let response = format!(
                        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(response.as_bytes()).await;
                    let _ = sock.flush().await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        format!("http://{addr}")
    }

    /// PP-2 MUST-FIRE: only **404** means "this build does not route the path".
    ///
    /// Every non-2xx was `Ok(None)`, so a 500 from `/v1/effective-config` —
    /// the route EXISTS and the server failed answering it — read as an older
    /// build. The producer then recorded compute class, feature set and subject
    /// identity as the operator's DECLARATIONS and wrote a receipt that looks
    /// exactly like an honest one taken against a server with no endpoint. The
    /// fact that the server was failing appears nowhere.
    #[cfg(feature = "llm")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn only_a_404_reads_as_a_route_this_build_does_not_have() {
        // MUST-NOT-FIRE: 404 is the absent route, reported as absent.
        let absent = spawn_status_endpoint("404 Not Found", "not found").await;
        let none = LlmClient::new(&absent, "qwen")
            .get_json("/v1/effective-config")
            .await
            .expect("404 is a fact, not a failure");
        assert!(none.is_none());

        // MUST-FIRE: anything else names the status and does not pretend.
        for (status_line, code) in [
            ("500 Internal Server Error", 500_u16),
            ("503 Service Unavailable", 503),
            ("403 Forbidden", 403),
        ] {
            let broken = spawn_status_endpoint(status_line, "{\"error\":\"boom\"}").await;
            let err = LlmClient::new(&broken, "qwen")
                .get_json("/v1/effective-config")
                .await
                .expect_err("a non-404 non-2xx must not read as an absent route");
            match err {
                LlmClientError::RouteFailed { status, .. } => assert_eq!(status, code),
                other => panic!("expected RouteFailed, got {other:?}"),
            }
        }

        // And a 200 still returns the body verbatim.
        let ok = spawn_status_endpoint("200 OK", "{\"compute_class\":\"cuda\"}").await;
        let body = LlmClient::new(&ok, "qwen")
            .get_json("/v1/effective-config")
            .await
            .expect("2xx")
            .expect("a body");
        assert_eq!(body["compute_class"], "cuda");
    }

    /// THE END-TO-END PROOF for PP-28: the bytes the server received carry the
    /// sampler pin. A struct-level assertion cannot show this — the defect was
    /// between the struct and the socket.
    #[cfg(feature = "llm")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_stream_path_puts_seed_and_ignore_eos_on_the_wire() {
        let (url, bodies) = spawn_scripted_sse(SseScript::conformant()).await;
        let client = LlmClient::new(&url, "qwen-coder");
        client
            .chat_completion_stream(&pinned_request())
            .await
            .expect("a conformant stream is accepted");

        let received = bodies.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let body = received.first().expect("the probe recorded one body");
        assert!(
            body.contains("\"seed\":0"),
            "seed missing from wire: {body}"
        );
        assert!(
            body.contains("\"ignore_eos\":true"),
            "ignore_eos missing from wire: {body}"
        );
        assert!(body.contains("\"stream\":true"), "{body}");
        assert!(body.contains("\"max_tokens\":128"), "{body}");
        // PP-27, on the socket: the request ASKS for the terminal usage frame.
        // Every OpenAI-compatible server emits `usage` on the final chunk only
        // when asked, so without this the `StreamNoUsage` refusal below fires
        // on servers that would have answered.
        assert!(
            body.contains("\"stream_options\":{\"include_usage\":true}"),
            "stream_options missing from wire: {body}"
        );
    }

    /// MUST-FIRE for PP-27: a stream whose terminal chunk carries no `usage` is
    /// refused. The alternative -- counting the four content frames as four
    /// tokens -- is a number with the shape of a measurement.
    #[cfg(feature = "llm")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stream_without_terminal_usage_is_refused() {
        let script = SseScript {
            terminal_usage: false,
            ..SseScript::conformant()
        };
        let (url, _) = spawn_scripted_sse(script).await;
        let client = LlmClient::new(&url, "qwen-coder");
        let err = client
            .chat_completion_stream(&pinned_request())
            .await
            .expect_err("no terminal usage must refuse");
        assert!(
            matches!(err, LlmClientError::StreamNoUsage { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("PP-27"), "{err}");

        // REVERT -> GREEN. The same probe WITH the usage block is accepted, so
        // the refusal is about the missing block and not about the fixture.
        let (ok_url, _) = spawn_scripted_sse(SseScript::conformant()).await;
        let ok = LlmClient::new(&ok_url, "qwen-coder");
        let r = ok
            .chat_completion_stream(&pinned_request())
            .await
            .expect("terminal usage present");
        assert_eq!(r.usage.completion_tokens, 128);
        assert_eq!(r.usage.prompt_tokens, 512);
        // And the SERVER's count, not the frame count: the probe emitted four
        // content frames while declaring 128 completion tokens.
        assert_eq!(r.token_timestamps.len(), 4);
    }

    /// MUST-FIRE for PP-27's other half: a stream with no content-bearing chunk
    /// has no first-token instant, and `ttft = e2e` would be indistinguishable
    /// from a replay.
    #[cfg(feature = "llm")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stream_with_no_content_chunk_is_refused() {
        let script = SseScript {
            content_chunks: 0,
            ..SseScript::conformant()
        };
        let (url, _) = spawn_scripted_sse(script).await;
        let client = LlmClient::new(&url, "qwen-coder");
        let err = client
            .chat_completion_stream(&pinned_request())
            .await
            .expect_err("no content chunk must refuse");
        assert!(
            matches!(err, LlmClientError::StreamNoContent { .. }),
            "{err:?}"
        );
    }

    /// PP-27 / §3 — the server's declaration and its phase timings are captured
    /// rather than discarded, and an undeclared stream stays undeclared.
    #[cfg(feature = "llm")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_mode_and_timings_are_captured() {
        let (url, _) = spawn_scripted_sse(SseScript::conformant()).await;
        let live = LlmClient::new(&url, "qwen-coder")
            .chat_completion_stream(&pinned_request())
            .await
            .expect("conformant stream");
        assert_eq!(live.stream_mode, Some(StreamMode::Live));
        let t = live.timings.expect("timings on the terminal chunk");
        assert_eq!(t.prompt_n, Some(512));
        assert_eq!(t.prompt_ms, Some(40.0));
        assert_eq!(t.predicted_n, Some(4));

        // A server that says `replayed` is recorded as replayed, not corrected.
        let (replay_url, _) = spawn_scripted_sse(SseScript {
            declare_mode: Some("replayed"),
            ..SseScript::conformant()
        })
        .await;
        let replayed = LlmClient::new(&replay_url, "qwen-coder")
            .chat_completion_stream(&pinned_request())
            .await
            .expect("a replayed stream still parses");
        assert_eq!(replayed.stream_mode, Some(StreamMode::Replayed));

        // DISCRIMINATION: a server that declares nothing is `None`, never
        // `live`. Defaulting to live is how a replay buys a latency number.
        let (silent_url, _) = spawn_scripted_sse(SseScript {
            declare_mode: None,
            timings: false,
            ..SseScript::conformant()
        })
        .await;
        let silent = LlmClient::new(&silent_url, "qwen-coder")
            .chat_completion_stream(&pinned_request())
            .await
            .expect("an undeclared stream still parses");
        assert_eq!(silent.stream_mode, None);
        assert_eq!(silent.timings, None);
    }

    /// The terminal frame OpenAI emits under `stream_options.include_usage`
    /// carries `usage` and an EMPTY `choices` array. Refusing to parse it would
    /// lose exactly the counts PP-27 requires.
    #[test]
    fn a_usage_only_terminal_frame_parses() {
        let json = r#"{"choices":[],"usage":{"prompt_tokens":512,"completion_tokens":128,"total_tokens":640}}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices.is_empty());
        assert_eq!(chunk.usage.unwrap().completion_tokens, 128);
    }

    /// A `timings` block with keys this harness does not read (llama.cpp emits
    /// several) must still parse: dropping the block over an unread key would
    /// take `prefill` with it.
    #[test]
    fn unknown_timing_keys_do_not_discard_the_block() {
        let json = r#"{"choices":[],"timings":{"prompt_n":512,"prompt_ms":40.0,
            "prompt_per_token_ms":0.078,"predicted_n":128,"predicted_ms":600.0,
            "predicted_per_token_ms":4.7,"cache_n":0}}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        let t = chunk.timings.expect("block survives unknown keys");
        assert_eq!(t.prompt_ms, Some(40.0));
        assert_eq!(t.predicted_n, Some(128));
        assert_eq!(t.prompt_per_second, None, "absent stays absent, never 0.0");
    }
}
