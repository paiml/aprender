//! Perplexity command — CRUX-E-02.
//!
//! Reads a JSON array of per-token natural-log probabilities and computes
//! `PPL = exp(-mean(log p))` via the pure classifier in
//! `aprender::metrics::perplexity`. Mirrors the llama.cpp `llama-perplexity`
//! convention without requiring live model compute.
//!
//! Contract: `contracts/crux-E-02-v1.yaml`.
//!
//! FALSIFY-CRUX-E-02-CLI: this path is the CLI binding for the classifier.
//! The live-inference path (perplexity over a held-out corpus using a real
//! GGUF/APR model) is PARTIAL_ALGORITHM_LEVEL-discharged under
//! `BLOCKER-UPSTREAM-MISSING` until a stable log-probs extraction path lands.

use std::path::Path;

use aprender::metrics::perplexity::{compute_perplexity, PerplexityOutcome};

use crate::error::CliError;

pub(crate) fn run(log_probs_file: &Path, json: bool) -> Result<(), CliError> {
    let bytes = std::fs::read(log_probs_file).map_err(|e| {
        CliError::Io(std::io::Error::new(
            e.kind(),
            format!("reading {}: {e}", log_probs_file.display()),
        ))
    })?;

    let log_probs: Vec<f64> = serde_json::from_slice(&bytes).map_err(|e| {
        CliError::InvalidFormat(format!(
            "{}: expected JSON array of f64 natural-log probabilities: {e}",
            log_probs_file.display()
        ))
    })?;

    match compute_perplexity(&log_probs) {
        PerplexityOutcome::Ok {
            ppl,
            mean_nll,
            num_tokens,
        } => {
            if json {
                let out = serde_json::json!({
                    "ppl": ppl,
                    "mean_nll": mean_nll,
                    "num_tokens": num_tokens,
                    "log_probs_path": log_probs_file.display().to_string(),
                });
                println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
            } else {
                println!(
                    "PPL = {ppl:.4}  (mean NLL = {mean_nll:.4} nats, num_tokens = {num_tokens})"
                );
            }
            Ok(())
        }
        PerplexityOutcome::EmptyLogProbs => Err(CliError::ValidationFailed(
            "log-probs file is an empty array; at least one token required".into(),
        )),
        PerplexityOutcome::NonFiniteLogProb => Err(CliError::ValidationFailed(
            "log-probs contains a NaN or ±∞ value; log-probabilities must be finite and \
             non-positive"
                .into(),
        )),
        PerplexityOutcome::PositiveLogProb(v) => Err(CliError::ValidationFailed(format!(
            "log-prob {v} is strictly positive; log-probabilities must be ≤ 0 \
             (probability ≤ 1)"
        ))),
    }
}
