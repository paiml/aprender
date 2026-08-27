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
#[derive(Serialize, Deserialize, Default)]
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
    /// PERF-006 andon — the dispatch path this process takes, in the receipt's
    /// own vocabulary (`cpu` / `cuda` / `metal` / `wgpu` / `unknown`).
    ///
    /// Distinct from `compute_mode`, which is the older two-valued
    /// `cpu`/`gpu` operator field and stays for compatibility. This one is
    /// produced by [`crate::andon::compute_class`], the same function the
    /// serve banner and `apr bench --json`'s `provenance.compute_class` read,
    /// so the three cannot disagree.
    ///
    /// `#[serde(default)]` is for READING a body emitted by a server older
    /// than PERF-006, where the field does not exist; empty then means "the
    /// server that produced this body did not report a class", never "cpu".
    /// It has no effect on what this server emits — `build_health_response`
    /// always fills it from the shared function.
    #[serde(default)]
    pub compute_class: String,
    /// PERF-006 andon — how many generations this server runs AT ONCE.
    ///
    /// `1` means serialized: a request that arrives while another is
    /// generating waits for it (`contracts/batch-admission-v1.yaml`). Reported
    /// on that path too, which is the whole point — a field that only appears
    /// when batching is active reports success and is silent on the failure it
    /// exists to expose.
    ///
    /// `#[serde(default)]` for the same backward-read reason as
    /// `compute_class`: `0` on a PARSED body means the emitting server predates
    /// the field. A body this server emits is never `0` — the floor is 1,
    /// asserted by `andon_health_tests`.
    #[serde(default)]
    pub max_in_flight: usize,
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
    /// Sampling temperature.
    ///
    /// Rejected at deserialization when outside `[0, ∞)` finite (aprender#2375),
    /// so it cannot reach a backend that does not validate it: only the
    /// QUANTIZED path ran `resolve_quantized_sampling`, and on a dense server
    /// `/generate`, `/stream/generate` and `/batch/generate` all answered
    /// `500 "Temperature must be a positive finite number"`.
    #[serde(
        default = "default_temperature",
        deserialize_with = "deserialize_temperature_f32_required"
    )]
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
    /// Sampling temperature (shared).
    ///
    /// Rejected at deserialization when outside `[0, ∞)` finite (aprender#2375).
    #[serde(
        default = "default_temperature",
        deserialize_with = "deserialize_temperature_f32_required"
    )]
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

// ---------------------------------------------------------------------------
// Temperature: the servable domain, enforced where the request is parsed
// ---------------------------------------------------------------------------

/// Is this temperature one the samplers can honour?
///
/// The servable domain is `[0, ∞)` **finite**. `0` is the OpenAI-canonical
/// deterministic request and resolves to greedy; every positive finite value
/// scales the logits.
///
/// Everything else is unservable, and each mode fails differently, which is why
/// none of them may reach a sampler:
///
/// * **negative** — `logit / -t` inverts the distribution, so the dense path
///   answered HTTP 500 (`apply_temperature`) and the quantized path served the
///   model's LEAST likely tokens with a 200.
/// * **NaN** — every comparison against NaN is false, so a `t < 0.0` guard does
///   not catch it (aprender#2391 is an entire issue about exactly that). It
///   poisons every logit and the cumulative draw never fires.
/// * **±∞** — `t.is_nan() || t < 0.0` misses `+inf` as well; the dense sampler
///   rejects it (500) and the quantized one flattens every logit to 0.
///
/// `is_finite()` is what carries the NaN and ±∞ cases: a comparison-only guard
/// such as `t < 0.0` is FALSE for NaN and lets it straight through.
#[must_use]
pub(crate) fn temperature_is_servable(temperature: f64) -> bool {
    temperature.is_finite() && temperature >= 0.0
}

/// The client-visible refusal for an unservable temperature.
fn temperature_rejection<E: serde::de::Error>(temperature: f64) -> E {
    E::custom(format!(
        "{}temperature must be a finite number >= 0 (0 means deterministic/greedy), \
         got {temperature}",
        crate::api::CLIENT_VISIBLE_MARKER
    ))
}

/// `deserialize_with` for an `Option<f32>` temperature field.
///
/// aprender#2375: `temperature: 0` was fixed by hand in the handlers and the
/// REST of the domain was left reaching `apply_temperature`, which answers
/// `500 {"error":"Invalid shape: Temperature must be a positive finite number"}`
/// — the very body that fix set out to eliminate. Refusing here makes an
/// unservable temperature unrepresentable in a deserialized request, exactly as
/// [`ChoiceCount`] does for `n > 1`, so no handler and no backend can be the one
/// that forgot.
///
/// The `f64 -> f32` narrowing is part of the check: `1e40` is a perfectly finite
/// `f64` that becomes `+inf` as an `f32`, and that infinity is what would reach
/// the sampler.
///
/// # Errors
///
/// Rejects a temperature outside the servable domain — see
/// [`temperature_is_servable`].
pub(crate) fn deserialize_temperature_f32<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(raw) = Option::<f64>::deserialize(deserializer)? else {
        return Ok(None);
    };
    let narrowed = f64::from(raw as f32);
    if !temperature_is_servable(raw) || !temperature_is_servable(narrowed) {
        return Err(temperature_rejection(raw));
    }
    Ok(Some(raw as f32))
}

/// `deserialize_with` for an `Option<f64>` temperature field.
///
/// Same rule as [`deserialize_temperature_f32`]; the value still narrows to
/// `f32` before it reaches a sampler, so the narrowing is checked here too.
///
/// # Errors
///
/// Rejects a temperature outside the servable domain.
pub(crate) fn deserialize_temperature_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(deserialize_temperature_f32(deserializer)?.map(f64::from))
}

/// `deserialize_with` for a non-`Option` `f32` temperature field.
///
/// A MISSING field never reaches here — serde's `default` attribute answers it —
/// so this sees only values the client actually sent. An explicit `null` is
/// refused rather than silently becoming a temperature the client did not
/// choose, which is what `f32::deserialize` did before this guard existed.
///
/// # Errors
///
/// Rejects `null` and any temperature outside the servable domain.
pub(crate) fn deserialize_temperature_f32_required<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_temperature_f32(deserializer)?.ok_or_else(|| {
        serde::de::Error::custom(format!(
            "{}temperature must be a finite number >= 0, not null",
            crate::api::CLIENT_VISIBLE_MARKER
        ))
    })
}
