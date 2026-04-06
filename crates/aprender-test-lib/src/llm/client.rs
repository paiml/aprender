//! OpenAI-compatible HTTP client for LLM inference endpoints.
//!
//! Supports chat completions against realizar, ollama, llama.cpp,
//! and any server exposing the OpenAI `/v1/chat/completions` API.

use serde::{Deserialize, Serialize};
use std::time::Duration;
#[cfg(feature = "llm")]
use std::time::Instant;

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

/// A streaming chunk response from the chat completion endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    /// Generated choices (typically 1).
    pub choices: Vec<StreamChoice>,
    /// Token usage (only present on final chunk for some backends).
    pub usage: Option<Usage>,
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
    /// Token usage (if reported by server).
    pub usage: Option<Usage>,
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

    /// Send a chat completion request and return the response with timing.
    pub async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<TimedChatResponse, LlmClientError> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            temperature,
            max_tokens,
            stream: Some(false),
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

        // Use the client's model name if the request's model is empty
        let actual_request;
        let req = if request.model.is_empty() {
            actual_request = ChatRequest {
                model: self.model.clone(),
                ..request.clone()
            };
            &actual_request
        } else {
            request
        };

        let resp = self.client.post(&url).json(req).send().await?;
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

        let actual_request;
        let req = if request.model.is_empty() {
            actual_request = ChatRequest {
                model: self.model.clone(),
                ..request.clone()
            };
            &actual_request
        } else {
            request
        };

        let resp = self
            .client
            .post(&url)
            .header("X-Trace-Level", trace_level)
            .json(req)
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

        // Force streaming on
        let stream_request = ChatRequest {
            model: if request.model.is_empty() {
                self.model.clone()
            } else {
                request.model.clone()
            },
            messages: request.messages.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(true),
        };

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
                        if sse_chunk.usage.is_some() {
                            final_usage = sse_chunk.usage;
                        }
                    }
                }
            }
        }

        let latency = start.elapsed();

        Ok(StreamedChatResponse {
            content,
            latency,
            ttft: ttft.unwrap_or(latency),
            token_timestamps,
            usage: final_usage,
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

    #[test]
    fn test_chat_request_with_stream_true() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: Some(true),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"stream\":true"));
    }
}
