//! Banco API types — re-exports from domain-specific modules.

#[path = "types_chat.rs"]
mod types_chat;
#[path = "types_data.rs"]
mod types_data;

// Re-export all types for backward compatibility
pub use types_chat::*;
pub use types_data::*;

use serde::{Deserialize, Serialize};

// ============================================================================
// BANCO-TYP-004: Health / Models / System
// ============================================================================

/// Health endpoint response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub circuit_breaker_state: String,
    pub uptime_secs: u64,
}

/// A single model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub local: bool,
}

/// Models list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// System information response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResponse {
    pub privacy_tier: String,
    pub backends: Vec<String>,
    pub gpu_available: bool,
    pub version: String,
    pub telemetry: bool,
    pub model_loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Hint for next action (empty when fully operational)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// Tokenizer mode: "bpe" or "greedy". Null when no model loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    /// Operational stats
    pub endpoints: u32,
    pub files: usize,
    pub conversations: usize,
    pub rag_indexed: bool,
    pub rag_chunks: usize,
    pub training_runs: usize,
    pub audit_entries: usize,
}

// ============================================================================
// BANCO-TYP-005: Error Response
// ============================================================================

/// OpenAI-compatible error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail within an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub code: u16,
}

impl ErrorResponse {
    #[must_use]
    pub fn new(message: impl Into<String>, type_: impl Into<String>, code: u16) -> Self {
        Self { error: ErrorDetail { message: message.into(), type_: type_.into(), code } }
    }
}

// ============================================================================
// BANCO-TYP-008: Conversations
// ============================================================================

/// Create conversation request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
}

/// Conversation list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationsListResponse {
    pub conversations: Vec<super::conversations::ConversationMeta>,
}

/// Single conversation response (with messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationResponse {
    #[serde(flatten)]
    pub conversation: super::conversations::Conversation,
}

/// Conversation created response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationCreatedResponse {
    pub id: String,
    pub title: String,
}

// ============================================================================
// BANCO-TYP-010: Model Management
// ============================================================================

/// Model load request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadRequest {
    pub model: String,
    #[serde(default = "default_slot")]
    pub slot: String,
}

fn default_slot() -> String {
    "primary".to_string()
}

/// Model status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusResponse {
    pub loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<super::model_slot::ModelSlotInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime_secs: Option<u64>,
    /// Tokenizer mode: "bpe" (proper merge rules) or "greedy" (approximate fallback).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
}

// ============================================================================
// BANCO-TYP-009: Prompt Presets
// ============================================================================

/// Create/update prompt preset request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavePromptRequest {
    pub name: String,
    pub content: String,
}

/// Prompt presets list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsListResponse {
    pub presets: Vec<super::prompts::PromptPreset>,
}
