//! Chat completion types (request, response, streaming chunks, parameters).

use crate::serve::templates::{ChatMessage, Role};
use serde::{Deserialize, Serialize};

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BancoChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Structured output format (Phase 2b: json_schema, json_object, regex).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Enable RAG: retrieve relevant chunks from indexed documents before generating.
    #[serde(default)]
    pub rag: bool,
    /// RAG configuration overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rag_config: Option<RagConfig>,
    /// File attachments — documents injected as context before generation.
    #[serde(default)]
    pub attachments: Vec<ChatAttachment>,
    /// Tools available for this request (OpenAI-compatible function calling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,
    /// Tool choice: "auto", "none", or specific tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

/// File attachment in a chat request — text extracted and injected as context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatAttachment {
    /// Filename (used for format detection).
    pub name: String,
    /// Text content of the file (pre-extracted by client or Banco).
    pub content: String,
    /// Optional content type override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
}

/// Tool specification for function calling (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool type — currently always "function".
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
    /// Function definition.
    pub function: ToolFunction,
}

/// Function definition within a tool spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

fn default_tool_type() -> String {
    "function".to_string()
}

/// RAG configuration for chat requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default = "default_rag_top_k")]
    pub top_k: usize,
    #[serde(default = "default_rag_min_score")]
    pub min_score: f64,
}

fn default_rag_top_k() -> usize {
    5
}
fn default_rag_min_score() -> f64 {
    0.1
}

/// Structured output format specification (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Any valid JSON object.
    JsonObject,
    /// JSON conforming to a specific schema.
    JsonSchema {
        /// Schema name.
        #[serde(default)]
        name: Option<String>,
        /// JSON schema definition.
        schema: serde_json::Value,
    },
    /// Output matching a regex pattern.
    Regex {
        /// Regex pattern.
        pattern: String,
    },
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BancoChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage statistics with context window info.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_used_pct: Option<f32>,
}

/// A single Server-Sent Event chunk for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BancoChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// A single choice within a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Delta content within a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Server-wide default inference parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repeat_penalty: default_repeat_penalty(),
            max_tokens: default_max_tokens(),
        }
    }
}

pub(crate) fn default_max_tokens() -> u32 {
    256
}
pub(crate) fn default_temperature() -> f32 {
    0.7
}
pub(crate) fn default_top_p() -> f32 {
    1.0
}
fn default_top_k() -> u32 {
    40
}
fn default_repeat_penalty() -> f32 {
    1.1
}
