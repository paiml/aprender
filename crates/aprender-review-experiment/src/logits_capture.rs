//! PRM-C11 `sparse-logits-capture-v1`: the per-step top-k an `apr run --json`
//! records (`--logprobs K`, #4026) as a `sparse-logits-v1` blob (contract
//! `sparse-logits-capture-v1`, FALSIFY-SLC-001..003). Local rows only.
//!
//! - **Input**: the run's `logprobs` object, `{top_k, prompt_token_ids,
//!   steps[{step, chosen, top[{token_id, logit, logprob}]}]}`, read strictly.
//! - **logsumexp_full** is not recorded by the run; every entry carries it as
//!   `logit − logprob`. The entries of one step must agree on it within
//!   [`LSE_TOLERANCE`], or they are not one softmax and the step is refused.
//! - **topk_mass** is `Σ exp(logprob)`, so the residual mass `1 − M` the
//!   tail-bucket correction needs survives the round trip.
//! - The blob is written by [`crate::sparse_logits::encode`]; `logits_sha` is
//!   the sha256 of those bytes.

use serde::{Deserialize, Serialize};

use crate::sparse_logits::{Header, TokenLogits};

pub const SCHEME: &str = "sparse-logits-capture-v1";

/// How far two entries of one step may disagree on `logit − logprob`. The
/// run computes both in f32; anything larger is two distributions.
pub const LSE_TOLERANCE: f64 = 1e-3;

/// One of a step's most likely tokens, as #4026 serialises it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Top {
    pub token_id: u32,
    pub logit: f32,
    pub logprob: f32,
}

/// One generated token's distribution, as #4026 serialises it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub step: usize,
    pub chosen: u32,
    pub top: Vec<Top>,
}

/// The `logprobs` object of `apr run --json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunLogprobs {
    pub top_k: usize,
    pub prompt_token_ids: Vec<u32>,
    pub steps: Vec<Step>,
}

/// What the run itself does not record, for the blob header.
#[derive(Debug, Clone, PartialEq)]
pub struct Meta {
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub apr_tag: String,
    /// The backend that served the run (`served_by`).
    pub backend: String,
    pub temperature_of_record: f32,
}

/// The header and tokens of one run. A run that recorded nothing, a step
/// out of order, a short or unordered top list, a repeated id, or entries
/// that are not one softmax is an error.
pub fn capture(lp: &RunLogprobs, meta: &Meta) -> Result<(Header, Vec<TokenLogits>), String> {
    let _ = (lp, meta);
    Err("unimplemented".into())
}

/// The `sparse-logits-v1` blob of a run's `logprobs` value and its
/// `logits_sha`. A `null` (the run had no `--logprobs`) is an error.
pub fn capture_blob(
    logprobs: &serde_json::Value,
    meta: &Meta,
) -> Result<(Vec<u8>, String), String> {
    let _ = (logprobs, meta);
    Err("unimplemented".into())
}

#[cfg(test)]
#[path = "logits_capture_tests.rs"]
mod tests;
