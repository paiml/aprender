//! API Request/Response Types (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.
//! Contains all basic API data structures.

use crate::registry::ModelInfo;
use serde::{Deserialize, Serialize};

/// Health check response.
///
/// Schema is defined by `contracts/crux-C-34-v1.yaml` (CRUX-C-34,
/// competitor parity: vLLM `/health`, llama.cpp server `/health`).
///
/// * `status ∈ {"ok", "loading", "degraded"}`
/// * HTTP 200 iff `status == "ok"`; 503 for `loading` / `degraded`.
/// * `model_loaded` gates `/health/ready` (k8s readiness probe).
/// * `uptime_sec > 0` and strictly monotonic across sequential GETs.
///
/// `version` and `compute_mode` are aprender extensions (not forbidden
/// by the contract) and remain for operator diagnostics.
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status: `"ok"`, `"loading"`, or `"degraded"`.
    pub status: String,
    /// Service version
    pub version: String,
    /// Compute mode: "cpu" or "gpu"
    pub compute_mode: String,
    /// Whether a model is resident and ready for inference.
    pub model_loaded: bool,
    /// Seconds since the server process first bound a router.
    pub uptime_sec: f64,
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

/// Why a completion stopped, in OpenAI's vocabulary.
///
/// Dogfood 0.63.0 (#2375 finding 6): the STREAMING chat path emitted the string
/// literal `"stop"` in its terminal chunk no matter what happened, while the
/// non-streaming path on the identical request correctly reported `"length"`
/// when the generation hit `max_tokens`. A client that streams therefore cannot
/// tell a truncated answer from a finished one, and every "continue from where
/// you stopped" flow silently breaks.
///
/// The type exists so that literal cannot come back: the terminal-chunk
/// constructor takes a `FinishReason`, which is only obtainable from
/// [`FinishReason::from_generation`] (or an explicit, named variant). There is
/// no `&str` parameter left to hardcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted a stop token or a stop string matched.
    Stop,
    /// Generation was cut off at the `max_tokens` budget.
    Length,
}

impl FinishReason {
    /// Decide the reason from what the generation actually did.
    ///
    /// Mirrors `finalize_chat_text` / `completion_finish_reason`: a matched stop
    /// string wins over the budget, and a model that terminated early is
    /// `Stop`. Only "ran to the budget with no stop match" is `Length`.
    #[must_use]
    pub fn from_generation(stopped: bool, completion_tokens: usize, max_tokens: usize) -> Self {
        if !stopped && completion_tokens >= max_tokens {
            Self::Length
        } else {
            Self::Stop
        }
    }

    /// The wire string OpenAI clients match on.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The OpenAI `n` field: how many completions the client asked for.
///
/// Dogfood 0.63.0 (#2375 finding 9): `n` was declared as a plain `usize` on the
/// request structs and read by no handler, so `n: 3` returned one choice with
/// HTTP 200 and no warning — a caller doing best-of-n sampling silently got
/// one sample and could not detect it.
///
/// This server generates exactly one choice per request. Rather than accept a
/// number it will not honour, deserialization REJECTS anything but 1, so the
/// value can never again reach a handler that ignores it: `n > 1` is not
/// representable in a deserialized request at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ChoiceCount(usize);

impl ChoiceCount {
    /// The only supported value.
    pub const ONE: Self = Self(1);

    /// The requested number of choices (always 1 — see the type docs).
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

impl Default for ChoiceCount {
    fn default() -> Self {
        Self::ONE
    }
}

impl<'de> Deserialize<'de> for ChoiceCount {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let requested = usize::deserialize(deserializer)?;
        if requested == 1 {
            Ok(Self::ONE)
        } else {
            Err(serde::de::Error::custom(format!(
                "{}n must be 1: this server returns exactly one choice per request, \
                 so n={requested} cannot be honoured (send {requested} requests instead)",
                crate::api::CLIENT_VISIBLE_MARKER
            )))
        }
    }
}
