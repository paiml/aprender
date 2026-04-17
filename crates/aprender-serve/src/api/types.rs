//! API Request/Response Types (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.
//! Contains all basic API data structures.

use crate::registry::ModelInfo;
use serde::{Deserialize, Serialize};

/// Health check response
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    pub version: String,
    /// Compute mode: "cpu" or "gpu"
    pub compute_mode: String,
}

/// Tokenize request
#[derive(Serialize, Deserialize)]
pub struct TokenizeRequest {
    /// Text to tokenize
    pub text: String,
    /// Model ID (optional, uses default if not specified)
    pub model_id: Option<String>,
}

/// Tokenize response
#[derive(Serialize, Deserialize)]
pub struct TokenizeResponse {
    /// Token IDs
    pub token_ids: Vec<u32>,
    /// Number of tokens
    pub num_tokens: usize,
}

/// Generate request
#[derive(Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Input prompt (token IDs or text)
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Sampling strategy: "greedy", "`top_k`", or "`top_p`"
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Top-k value (if strategy is "`top_k`")
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Top-p value (if strategy is "`top_p`")
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Model ID (optional, uses default if not specified)
    pub model_id: Option<String>,
}

/// Default max tokens for generation requests.
pub fn default_max_tokens() -> usize {
    50
}
pub(crate) fn default_temperature() -> f32 {
    1.0
}
pub(crate) fn default_strategy() -> String {
    "greedy".to_string()
}
/// Default top-k value for sampling.
pub fn default_top_k() -> usize {
    50
}
pub(crate) fn default_top_p() -> f32 {
    0.9
}

/// Generate response
#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Decoded text
    pub text: String,
    /// Number of generated tokens
    pub num_generated: usize,
}

/// Error response
#[derive(Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
}

/// Batch tokenize request
#[derive(Serialize, Deserialize)]
pub struct BatchTokenizeRequest {
    /// Texts to tokenize
    pub texts: Vec<String>,
}

/// Batch tokenize response
#[derive(Serialize, Deserialize)]
pub struct BatchTokenizeResponse {
    /// Results for each text in the same order
    pub results: Vec<TokenizeResponse>,
}

/// Batch generate request
#[derive(Serialize, Deserialize)]
pub struct BatchGenerateRequest {
    /// Input prompts
    pub prompts: Vec<String>,
    /// Maximum tokens to generate (shared across all prompts)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature (shared)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Sampling strategy (shared)
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Top-k value (shared)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Top-p value (shared)
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Batch generate response
#[derive(Serialize, Deserialize)]
pub struct BatchGenerateResponse {
    /// Results for each prompt in the same order
    pub results: Vec<GenerateResponse>,
}

/// Stream token event (SSE)
#[derive(Serialize, Deserialize)]
pub struct StreamTokenEvent {
    /// Token ID
    pub token_id: u32,
    /// Decoded text for this token
    pub text: String,
}

/// Stream done event (SSE)
#[derive(Serialize, Deserialize)]
pub struct StreamDoneEvent {
    /// Total number of tokens generated
    pub num_generated: usize,
}

/// Models list response
#[derive(Serialize, Deserialize)]
pub struct ModelsResponse {
    /// List of available models
    pub models: Vec<ModelInfo>,
}
