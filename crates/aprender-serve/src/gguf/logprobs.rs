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

/// One of the most likely next tokens at a step (#4026).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TopLogprob {
    /// Token ID
    pub token_id: u32,
    /// The raw logit the forward produced for it
    pub logit: f32,
    /// ln(softmax(logits)[token_id])
    pub logprob: f32,
}

/// The distribution one generated step was chosen from (#4026): the `K` most
/// likely tokens, before any repetition penalty or sampling, and the token the
/// engine then chose. `step` 0 is the token after the prompt.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StepLogprobs {
    /// 0-based index of the generated token this step produced
    pub step: usize,
    /// The token the engine chose at this step
    pub chosen: u32,
    /// The `K` most likely tokens, most likely first
    pub top: Vec<TopLogprob>,
    /// ln(softmax(logits)[chosen]) on the same forward logits as `top`, so it
    /// is defined when a sampled `chosen` is outside the top `K` (#4026, PRM C11)
    pub chosen_logprob: f32,
    /// Full-vocab log-sum-exp of those logits at T = 1.0: `top` plus this gives
    /// the retained mass and the residual tail `sparse-logits-v1` stores
    pub logsumexp_full: f32,
}

/// Full-vocab log-sum-exp of `logits`, NaNs skipped, accumulated in f64 so a
/// 150k-entry vocab does not lose the tail to f32 rounding (#4026).
#[must_use]
pub fn logsumexp_full(logits: &[f32]) -> f32 {
    let max = logits
        .iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = logits
        .iter()
        .filter(|x| !x.is_nan())
        .map(|&x| f64::from(x - max).exp())
        .sum();
    (f64::from(max) + sum.ln()) as f32
}

/// The `k` most likely tokens in `logits`, most likely first (#4026).
///
/// Ties go to the lower token id, as [`crate::gguf::ops::argmax`] breaks them,
/// so with no penalty the first entry IS the greedy choice. NaN logits are
/// never ranked. `logprob` is `logit` minus [`logsumexp_full`].
#[must_use]
pub fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<TopLogprob> {
    if k == 0 {
        return Vec::new();
    }
    // The step record's `logsumexp_full`, so Σexp(top) and the tail a reader
    // derives from the two agree exactly (PRM C11).
    let lse = logsumexp_full(logits);
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .map(|(i, &x)| (i as u32, x))
        .collect();
    let k = k.min(ranked.len());
    let by_rank = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if k < ranked.len() {
        ranked.select_nth_unstable_by(k, by_rank);
        ranked.truncate(k);
    }
    ranked.sort_by(by_rank);
    ranked
        .into_iter()
        .map(|(token_id, logit)| TopLogprob {
            token_id,
            logit,
            logprob: logit - lse,
        })
        .collect()
}

#[cfg(test)]
mod tests_4026 {
    use super::*;

    fn ids(top: &[TopLogprob]) -> Vec<u32> {
        top.iter().map(|t| t.token_id).collect()
    }

    #[test]
    fn top_k_logprobs_case_table() {
        let logits = [0.5, 2.0, -1.0, 2.0, 1.0];
        // (k, expected ids): ties go to the lower id, as ops::argmax.
        let table: &[(usize, &[u32])] = &[
            (0, &[]),
            (1, &[1]),
            (2, &[1, 3]),
            (3, &[1, 3, 4]),
            (5, &[1, 3, 4, 0, 2]),
            (99, &[1, 3, 4, 0, 2]),
        ];
        for &(k, want) in table {
            assert_eq!(ids(&top_k_logprobs(&logits, k)), want, "k={k}");
        }
        assert_eq!(
            top_k_logprobs(&logits, 1)[0].token_id,
            crate::gguf::ops::argmax(&logits),
            "top-1 is the greedy choice"
        );
    }

    #[test]
    fn top_k_logprobs_agree_with_logprob_of_and_sum_to_one() {
        let logits = [0.5_f32, 2.0, -1.0, 2.0, 1.0];
        let all = top_k_logprobs(&logits, logits.len());
        for t in &all {
            assert!((t.logprob - logprob_of(&logits, t.token_id)).abs() < 1e-6);
            assert_eq!(t.logit, logits[t.token_id as usize]);
        }
        let total: f32 = all.iter().map(|t| t.logprob.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "sum={total}");
    }

    #[test]
    fn top_k_logprobs_never_rank_nan() {
        let logits = [f32::NAN, 1.0, f32::NAN, 0.0];
        let top = top_k_logprobs(&logits, 4);
        assert_eq!(ids(&top), vec![1, 3]);
        assert!(top.iter().all(|t| t.logprob.is_finite()));
    }

    #[test]
    fn top_k_logprobs_survive_huge_logits() {
        let top = top_k_logprobs(&[1.0e30, 0.0], 2);
        assert_eq!(top[0].logprob, 0.0);
        assert!(top[1].logprob.is_finite() && top[1].logprob <= -1.0e29);
    }
}
