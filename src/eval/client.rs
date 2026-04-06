//! Minimal Anthropic API client for eval operations
//!
//! No external anthropic crate — just reqwest + serde.
//! Supports only the Messages API endpoint.

use serde::{Deserialize, Serialize};

/// Minimal Anthropic Messages API client
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

/// Token usage from an API call
#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Input tokens consumed
    pub input_tokens: u32,
    /// Output tokens generated
    pub output_tokens: u32,
}

/// Result of a single API call
#[derive(Debug)]
pub struct CompletionResult {
    /// The text response
    pub text: String,
    /// Token usage
    pub usage: Usage,
}

impl AnthropicClient {
    /// Create a new client from an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api_key: api_key.into(), client: reqwest::Client::new() }
    }

    /// Create a new client from the `ANTHROPIC_API_KEY` environment variable
    pub fn from_env() -> Result<Self, String> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;
        Ok(Self::new(key))
    }

    /// Send a messages request and return the text response
    pub async fn complete(
        &self,
        model: &str,
        system: Option<&str>,
        user_message: &str,
        max_tokens: u32,
    ) -> Result<CompletionResult, String> {
        let request = MessagesRequest {
            model: model.to_string(),
            max_tokens,
            system: system.map(String::from),
            messages: vec![Message { role: "user".to_string(), content: user_message.to_string() }],
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_else(|_| "unable to read body".to_string());
            return Err(format!("API error {status}: {body}"));
        }

        let resp: MessagesResponse =
            response.json().await.map_err(|e| format!("Failed to parse response: {e}"))?;

        let text = resp.content.first().map(|b| b.text.clone()).unwrap_or_default();

        Ok(CompletionResult { text, usage: resp.usage })
    }
}
