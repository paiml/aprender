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

use std::collections::HashSet;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::prereg::sha256_hex;
use crate::sparse_logits::{self, Header, Mode, TokenLogits};

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
    let k = u8::try_from(lp.top_k)
        .ok()
        .filter(|&k| k > 0)
        .ok_or_else(|| format!("top_k {} is not in 1..=255", lp.top_k))?;
    if lp.steps.is_empty() {
        return Err("no steps: the run recorded nothing".into());
    }
    let tokens = lp
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| token(i, s, lp.top_k))
        .collect::<Result<Vec<_>, _>>()?;
    let header = Header {
        schema: sparse_logits::SCHEME.into(),
        mode: Mode::Topk,
        model_sha256: meta.model_sha256.clone(),
        tokenizer_sha256: meta.tokenizer_sha256.clone(),
        apr_tag: meta.apr_tag.clone(),
        backend: meta.backend.clone(),
        temperature_of_record: meta.temperature_of_record,
        k,
        n_tokens: u32::try_from(tokens.len()).map_err(|_| "too many steps".to_string())?,
    };
    Ok((header, tokens))
}

fn token(i: usize, s: &Step, k: usize) -> Result<TokenLogits, String> {
    if s.step != i {
        return Err(format!("step {i}: recorded as step {}", s.step));
    }
    if s.top.len() != k {
        return Err(format!("step {i}: {} entries, top_k = {k}", s.top.len()));
    }
    let lse = f64::from(s.top[0].logit) - f64::from(s.top[0].logprob);
    let mut seen = HashSet::new();
    for (j, t) in s.top.iter().enumerate() {
        if !seen.insert(t.token_id) {
            return Err(format!("step {i}: token {} twice", t.token_id));
        }
        if j > 0 && t.logprob > s.top[j - 1].logprob {
            return Err(format!("step {i}: entry {j} outranks entry {}", j - 1));
        }
        let l = f64::from(t.logit) - f64::from(t.logprob);
        if !l.is_finite() || (l - lse).abs() > LSE_TOLERANCE {
            return Err(format!(
                "step {i}: entry {j} has logsumexp {l}, entry 0 has {lse}: not one softmax"
            ));
        }
    }
    let mass: f64 = s.top.iter().map(|t| f64::from(t.logprob).exp()).sum();
    Ok(TokenLogits {
        token_id_sampled: s.chosen,
        ids: s.top.iter().map(|t| t.token_id).collect(),
        logprobs: s.top.iter().map(|t| f16::from_f32(t.logprob)).collect(),
        logsumexp_full: lse as f32,
        topk_mass: f16::from_f64(mass.min(1.0)),
    })
}

/// The `sparse-logits-v1` blob of a run's `logprobs` value and its
/// `logits_sha`. A `null` (the run had no `--logprobs`) is an error.
pub fn capture_blob(
    logprobs: &serde_json::Value,
    meta: &Meta,
) -> Result<(Vec<u8>, String), String> {
    if logprobs.is_null() {
        return Err("logprobs is null: the run had no --logprobs".into());
    }
    let lp = RunLogprobs::deserialize(logprobs).map_err(|e| format!("logprobs: {e}"))?;
    let (h, tokens) = capture(&lp, meta)?;
    let blob = sparse_logits::encode(&h, &tokens)?;
    let sha = sha256_hex(&blob);
    Ok((blob, sha))
}

#[cfg(test)]
#[path = "logits_capture_tests.rs"]
mod tests;
