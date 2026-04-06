//! realizr#191: Per-token log probability types for perplexity measurement.
//!
//! Supports F-QUALITY-01: comparing realizr vs llama.cpp perplexity
//! on WikiText-2 with Q4_K_M.

/// Per-token log probability for OpenAI API compatibility.
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    /// Token ID
    pub token_id: u32,
    /// Log probability of the chosen token: ln(softmax(logits)[token_id])
    pub logprob: f32,
}

/// Generation result with optional logprobs.
#[derive(Debug)]
pub struct GenerateResult {
    /// Generated token IDs (including prompt)
    pub tokens: Vec<u32>,
    /// Per-token logprobs (empty if logprobs not requested)
    pub logprobs: Vec<TokenLogprob>,
}

/// Compute log probability of a token from raw logits.
///
/// Returns ln(softmax(logits)[token_id]) using the log-sum-exp trick
/// for numerical stability. Used for perplexity measurement (F-QUALITY-01).
pub fn logprob_of(logits: &[f32], token_id: u32) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>()
        .ln();
    logits[token_id as usize] - max_logit - log_sum_exp
}
