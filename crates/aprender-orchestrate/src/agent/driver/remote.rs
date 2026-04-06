//! Remote HTTP driver for external LLM APIs.
//!
//! Supports Anthropic Messages API and OpenAI-compatible Chat
//! Completions API. Privacy tier is `Standard` — blocked under
//! Sovereign manifests (Poka-Yoke).
//!
//! Phase 2: Used by `RoutingDriver` as fallback when local
//! inference fails or is unavailable.
//!
//! SSE streaming parsers live in `remote_stream.rs` (QA-002).

#[path = "remote_stream.rs"]
mod remote_stream;

use async_trait::async_trait;

use crate::agent::driver::{
    CompletionRequest, CompletionResponse, LlmDriver, Message, StreamEvent, ToolCall,
};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// API provider format for request/response translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    /// Anthropic Messages API (`tool_use` blocks).
    Anthropic,
    /// OpenAI-compatible Chat Completions API.
    OpenAi,
}

/// Remote HTTP LLM driver configuration.
#[derive(Debug, Clone)]
pub struct RemoteDriverConfig {
    /// API base URL (e.g., `https://api.anthropic.com`).
    pub base_url: String,
    /// API key for authentication.
    pub api_key: String,
    /// Model identifier (e.g., "claude-sonnet-4-20250514").
    pub model: String,
    /// API provider format.
    pub provider: ApiProvider,
    /// Context window size in tokens.
    pub context_window: usize,
}

/// Remote HTTP driver for external LLM APIs.
///
/// Sends requests to Anthropic/OpenAI-compatible endpoints.
/// Always operates at `PrivacyTier::Standard` — data leaves
/// the machine, so this driver is blocked under Sovereign
/// privacy manifests.
pub struct RemoteDriver {
    config: RemoteDriverConfig,
}

impl RemoteDriver {
    /// Create a new remote driver.
    pub fn new(config: RemoteDriverConfig) -> Self {
        Self { config }
    }

    /// Build URL and body for the configured provider.
    fn build_request(&self, request: &CompletionRequest) -> (String, serde_json::Value) {
        match self.config.provider {
            ApiProvider::Anthropic => {
                let url = format!("{}/v1/messages", self.config.base_url);
                (url, self.build_anthropic_body(request))
            }
            ApiProvider::OpenAi => {
                let url = format!("{}/v1/chat/completions", self.config.base_url);
                (url, self.build_openai_body(request))
            }
        }
    }

    /// Send an HTTP request with provider-specific auth headers.
    ///
    /// Returns the response or maps HTTP errors to `AgentError`.
    async fn send_http(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, AgentError> {
        let client = reqwest::Client::new();
        let mut req = client.post(url);
        req = match self.config.provider {
            ApiProvider::Anthropic => req
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json"),
            ApiProvider::OpenAi => req
                .header("authorization", format!("Bearer {}", self.config.api_key))
                .header("content-type", "application/json"),
        };

        let response = req.json(body).send().await.map_err(|e| {
            AgentError::Driver(DriverError::Network(format!("HTTP request failed: {e}")))
        })?;

        let status = response.status().as_u16();
        if status == 429 {
            return Err(AgentError::Driver(DriverError::RateLimited { retry_after_ms: 1000 }));
        }
        if status == 529 || status == 503 {
            return Err(AgentError::Driver(DriverError::Overloaded { retry_after_ms: 2000 }));
        }
        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Driver(DriverError::Network(format!("HTTP {status}: {text}"))));
        }

        Ok(response)
    }

    /// Build request body for Anthropic Messages API.
    fn build_anthropic_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User(text) => Some(serde_json::json!({
                    "role": "user",
                    "content": text
                })),
                Message::Assistant(text) => Some(serde_json::json!({
                    "role": "assistant",
                    "content": text
                })),
                Message::AssistantToolUse(call) => Some(serde_json::json!({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.input
                    }]
                })),
                Message::ToolResult(result) => Some(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": result.tool_use_id,
                        "content": result.content,
                        "is_error": result.is_error
                    }]
                })),
                Message::System(_) => None,
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        });

        if let Some(ref system) = request.system {
            body["system"] = serde_json::json!(system);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        body
    }

    /// Build request body for `OpenAI` Chat Completions API.
    fn build_openai_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        for m in &request.messages {
            match m {
                Message::System(text) => {
                    messages.push(serde_json::json!({
                        "role": "system",
                        "content": text
                    }));
                }
                Message::User(text) => {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": text
                    }));
                }
                Message::Assistant(text) => {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": text
                    }));
                }
                Message::AssistantToolUse(call) => {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.input.to_string()
                            }
                        }]
                    }));
                }
                Message::ToolResult(result) => {
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": result.tool_use_id,
                        "content": result.content
                    }));
                }
            }
        }

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        });

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        body
    }
}

#[async_trait]
impl LlmDriver for RemoteDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        let (url, body) = self.build_request(&request);
        let response = self.send_http(&url, &body).await?;

        let resp_body: serde_json::Value = response.json().await.map_err(|e| {
            AgentError::Driver(DriverError::InferenceFailed(format!("JSON parse error: {e}")))
        })?;

        Ok(match self.config.provider {
            ApiProvider::Anthropic => remote_stream::parse_anthropic_response(&resp_body),
            ApiProvider::OpenAi => remote_stream::parse_openai_response(&resp_body),
        })
    }

    /// Streaming completion via SSE.
    ///
    /// Anthropic: `"stream": true` + Server-Sent Events.
    /// OpenAI: `"stream": true` + `data:` prefixed chunks.
    /// Emits `TextDelta` events for each token, `ContentComplete` at end.
    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, AgentError> {
        use futures_util::StreamExt;

        let (url, mut body) = self.build_request(&request);
        body["stream"] = serde_json::json!(true);

        let response = self.send_http(&url, &body).await?;

        let mut full_text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop_reason = StopReason::EndTurn;
        let mut current_tool: Option<(String, String, String)> = None; // (id, name, json_accum)

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| {
                AgentError::Driver(DriverError::Network(format!("stream error: {e}")))
            })?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                let data = if let Some(stripped) = line.strip_prefix("data: ") {
                    stripped
                } else {
                    continue;
                };

                if data == "[DONE]" {
                    break;
                }

                let Ok(event): Result<serde_json::Value, _> = serde_json::from_str(data) else {
                    continue;
                };

                match self.config.provider {
                    ApiProvider::Anthropic => {
                        remote_stream::process_anthropic_event(
                            &event,
                            &mut full_text,
                            &mut tool_calls,
                            &mut usage,
                            &mut stop_reason,
                            &mut current_tool,
                            &tx,
                        )
                        .await;
                    }
                    ApiProvider::OpenAi => {
                        remote_stream::process_openai_event(
                            &event,
                            &mut full_text,
                            &mut tool_calls,
                            &mut usage,
                            &mut stop_reason,
                            &tx,
                        )
                        .await;
                    }
                }
            }
        }

        let _ = tx
            .send(StreamEvent::ContentComplete {
                stop_reason: stop_reason.clone(),
                usage: usage.clone(),
            })
            .await;

        Ok(CompletionResponse { text: full_text, stop_reason, tool_calls, usage })
    }

    fn context_window(&self) -> usize {
        self.config.context_window
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Standard // data leaves the machine
    }

    /// Estimate cost using conservative per-token pricing.
    ///
    /// Uses approximate rates: $3/$15 per 1M tokens (input/output)
    /// for Anthropic Sonnet-class models. Overestimates for cheaper
    /// models — the budget is a guardrail, not an invoice.
    #[allow(clippy::cast_precision_loss)] // token counts fit in f64 mantissa
    fn estimate_cost(&self, usage: &TokenUsage) -> f64 {
        let input_cost = usage.input_tokens as f64 * 3.0 / 1_000_000.0;
        let output_cost = usage.output_tokens as f64 * 15.0 / 1_000_000.0;
        input_cost + output_cost
    }
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "remote_tests_body.rs"]
mod tests_body;
