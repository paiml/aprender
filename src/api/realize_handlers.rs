//! Native Realize API handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains context window management and native Realize API endpoints.
#![allow(unreachable_pub)] // Items re-exported as pub from api/mod.rs

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use super::ContinuousBatchRequest;
use super::{AppState, ChatMessage, ErrorResponse, Usage};
use crate::generate::{GenerationConfig, SamplingStrategy};
use crate::registry::ModelInfo;

// ============================================================================
// Shared helpers
// ============================================================================

/// Shorthand error type for realize handlers.
type RErr = (StatusCode, Json<ErrorResponse>);

/// Build an error response, recording a failure metric.
fn rerr(state: &AppState, status: StatusCode, msg: impl std::fmt::Display) -> RErr {
    state.metrics.record_failure();
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
}

/// Current unix epoch seconds.
fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Current unix epoch millis (for response IDs).
fn epoch_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

// ============================================================================
// Context Window Management (per spec §5.2)
// ============================================================================

/// Configuration for context window management
#[derive(Debug, Clone)]
pub struct ContextWindowConfig {
    /// Maximum context window size in tokens
    pub max_tokens: usize,
    /// Reserved tokens for generation output
    pub reserved_output_tokens: usize,
    /// Whether to preserve system messages during truncation
    pub preserve_system: bool,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            reserved_output_tokens: 256,
            preserve_system: true,
        }
    }
}

impl ContextWindowConfig {
    /// Create new context window config
    #[must_use]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }

    /// Set reserved output tokens
    #[must_use]
    pub fn with_reserved_output(mut self, tokens: usize) -> Self {
        self.reserved_output_tokens = tokens;
        self
    }

    /// Calculate available tokens for prompt
    pub fn available_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserved_output_tokens)
    }
}

/// Context window manager for truncating chat messages
pub struct ContextWindowManager {
    config: ContextWindowConfig,
}

impl ContextWindowManager {
    /// Create new context window manager
    #[must_use]
    pub fn new(config: ContextWindowConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    #[must_use]
    pub fn default_manager() -> Self {
        Self::new(ContextWindowConfig::default())
    }

    /// Estimate token count for a message (rough approximation: ~4 chars per token)
    fn estimate_tokens(text: &str) -> usize {
        // Add overhead for role prefix and formatting
        const ROLE_OVERHEAD: usize = 10;
        text.len().div_ceil(4) + ROLE_OVERHEAD
    }

    /// Truncate messages to fit within context window
    ///
    /// Returns truncated messages and whether truncation occurred
    pub fn truncate_messages(&self, messages: &[ChatMessage]) -> (Vec<ChatMessage>, bool) {
        let available = self.config.available_tokens();

        // Calculate total tokens
        let total_tokens: usize = messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum();

        if total_tokens <= available {
            return (messages.to_vec(), false);
        }

        // Need to truncate - preserve system message if configured
        let mut result = Vec::new();
        let mut used_tokens = 0;

        // First pass: collect system messages if preserving
        let (system_msgs, other_msgs): (Vec<_>, Vec<_>) = messages
            .iter()
            .partition(|m| m.role == "system" && self.config.preserve_system);

        // Add system messages first
        for msg in &system_msgs {
            let tokens = Self::estimate_tokens(&msg.content);
            if used_tokens + tokens <= available {
                result.push((*msg).clone());
                used_tokens += tokens;
            }
        }

        // Add other messages from most recent, then reverse
        let mut temp_msgs: Vec<ChatMessage> = Vec::new();
        for msg in other_msgs.iter().rev() {
            let tokens = Self::estimate_tokens(&msg.content);
            if used_tokens + tokens <= available {
                temp_msgs.push((*msg).clone());
                used_tokens += tokens;
            } else {
                // No more room
                break;
            }
        }

        // Reverse to maintain chronological order
        temp_msgs.reverse();
        result.extend(temp_msgs);

        (result, true)
    }

    /// Check if messages need truncation
    pub fn needs_truncation(&self, messages: &[ChatMessage]) -> bool {
        let available = self.config.available_tokens();
        let total_tokens: usize = messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum();
        total_tokens > available
    }

    /// Get token estimate for messages
    pub fn estimate_total_tokens(&self, messages: &[ChatMessage]) -> usize {
        messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum()
    }
}

/// Format chat messages into a single prompt string using model-specific templates
///
/// Uses the chat_template module to format messages according to the model's
/// expected format (ChatML, LLaMA2, Mistral, Phi, Alpaca, or Raw fallback).
pub fn format_chat_messages(messages: &[ChatMessage], model_name: Option<&str>) -> String {
    use crate::chat_template::{self, ChatMessage as TemplateMessage};

    // Convert API ChatMessage to template ChatMessage
    let template_messages: Vec<TemplateMessage> = messages
        .iter()
        .map(|m| TemplateMessage::new(&m.role, &m.content))
        .collect();

    // Use model-aware template formatting
    chat_template::format_messages(&template_messages, model_name).unwrap_or_else(|_| {
        // Fallback to simple concatenation if template fails
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&msg.content);
            prompt.push('\n');
        }
        prompt
    })
}

/// Clean chat output to prevent prompt injection (PMAT-088)
///
/// Stops output at the first stop sequence to prevent the model from
/// generating additional conversation turns or injected content.
pub fn clean_chat_output(text: &str) -> String {
    // List of stop sequences that indicate end of assistant response
    const STOP_SEQUENCES: &[&str] = &[
        "<|im_end|>",    // ChatML (Qwen, OpenHermes, Yi)
        "<|endoftext|>", // GPT-style
        "<|end|>",       // Alternative
        "</s>",          // LLaMA style
        "\nHuman:",      // Anthropic/Claude style
        "\nUser:",       // Alternative user turn
        "\n\nHuman:",    // With extra newline
        "\n\nUser:",     // With extra newline
        "<|im_start|>",  // Start of new turn in ChatML
    ];

    let mut result = text.to_string();

    // Find the earliest stop sequence and truncate there
    let mut earliest_pos = result.len();
    for stop in STOP_SEQUENCES {
        if let Some(pos) = result.find(stop) {
            if pos < earliest_pos {
                earliest_pos = pos;
            }
        }
    }

    result.truncate(earliest_pos);
    result.trim().to_string()
}

// ============================================================================
// Native Realizar API Handlers (spec §5.2)
// ============================================================================

/// Request for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Text to embed
    pub input: String,
    /// Model ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Response for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Embedding object
    pub object: String,
    /// Embedding data
    pub data: Vec<EmbeddingData>,
    /// Model used
    pub model: String,
    /// Usage statistics
    pub usage: EmbeddingUsage,
}

/// Embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Object type
    pub object: String,
    /// Index
    pub index: usize,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// Embedding usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// Model metadata response (for /realize/model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataResponse {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: String,
    /// Model format (GGUF, APR, SafeTensors)
    pub format: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Quantization type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Context window size
    pub context_length: usize,
    /// Model lineage from Pacha
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage: Option<ModelLineage>,
    /// Whether model is loaded
    pub loaded: bool,
}

/// Model lineage information from Pacha registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLineage {
    /// Pacha URI
    pub uri: String,
    /// Version
    pub version: String,
    /// Training recipe (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// Parent model (if derived)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// BLAKE3 content hash
    pub content_hash: String,
}

/// Reload request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadRequest {
    /// Model ID to reload (optional, reloads current if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Path to model file to reload from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Reload response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReloadResponse {
    /// Success status
    pub success: bool,
    /// Message
    pub message: String,
    /// Reload time in ms
    pub reload_time_ms: u64,
}

/// OpenAI-compatible completions request (non-chat)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model ID
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// OpenAI-compatible completions response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Completion choices
    pub choices: Vec<CompletionChoice>,
    /// Usage statistics
    pub usage: Usage,
}

/// Completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Generated text
    pub text: String,
    /// Choice index
    pub index: usize,
    /// Log probabilities (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// Finish reason
    pub finish_reason: String,
}

include!("realize_handlers_embed_completion.rs");
include!("gpu_completions_handler.rs");
include!("realize_handlers_model_lineage.rs");
