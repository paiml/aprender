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
/// likely tokens, from the logits the choice read (after any repetition
/// penalty, before temperature; #4971), and the token the engine then chose.
/// `step` 0 is the token after the prompt.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StepLogprobs {
    /// 0-based index of the generated token this step produced
    pub step: usize,
    /// The token the engine chose at this step
    pub chosen: u32,
    /// ln(softmax(logits)[chosen]), recorded whether or not `chosen` is in `top`
    pub chosen_logprob: f32,
    /// The `K` most likely tokens, most likely first
    pub top: Vec<TopLogprob>,
}

impl StepLogprobs {
    /// The record of one step (#4971): the chosen token's own logprob and the
    /// `n` best tokens of `logits`. The chosen token need not be among them (a
    /// sampled step, or `n` 0), so its logprob is computed on its own.
    #[must_use]
    pub fn of(step: usize, chosen: u32, logits: &[f32], n: usize) -> Self {
        let (max_logit, log_sum_exp) = max_and_log_sum_exp(logits);
        let chosen_logprob = logits
            .get(chosen as usize)
            .map_or(f32::NEG_INFINITY, |&x| x - max_logit - log_sum_exp);
        Self {
            step,
            chosen,
            chosen_logprob,
            top: top_k_logprobs(logits, n),
        }
    }

    /// `top[0].logit - top[1].logit`: how close the step came to choosing
    /// another token (#4971, the V3 near-tie measure). `None` below two entries.
    #[must_use]
    pub fn top2_margin(&self) -> Option<f32> {
        match self.top.as_slice() {
            [first, second, ..] => Some(first.logit - second.logit),
            _ => None,
        }
    }
}

/// The largest non-NaN logit and the log-sum-exp of the rest relative to it,
/// so `logit - max - lse` is a logprob. NaN logits take no part.
fn max_and_log_sum_exp(logits: &[f32]) -> (f32, f32) {
    let max_logit = logits
        .iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits
        .iter()
        .filter(|x| !x.is_nan())
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>()
        .ln();
    (max_logit, log_sum_exp)
}

/// The `k` most likely tokens in `logits`, most likely first (#4026).
///
/// Ties go to the lower token id, as [`crate::gguf::ops::argmax`] breaks them,
/// so with no penalty the first entry IS the greedy choice. NaN logits are
/// never ranked. `logprob` uses the same log-sum-exp as [`logprob_of`].
#[must_use]
pub fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<TopLogprob> {
    if k == 0 {
        return Vec::new();
    }
    let (max_logit, log_sum_exp) = max_and_log_sum_exp(logits);
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
            logprob: logit - max_logit - log_sum_exp,
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

#[cfg(test)]
mod tests_4971 {
    use super::*;

    #[test]
    fn step_records_the_chosen_logprob_even_outside_the_top_n() {
        let logits = [0.5_f32, 2.0, -1.0, 2.0, 1.0];
        // Token 2 is the least likely: a sampled step can still choose it.
        let s = StepLogprobs::of(7, 2, &logits, 2);
        assert_eq!(s.step, 7);
        assert_eq!(s.chosen, 2);
        assert_eq!(s.top.iter().map(|t| t.token_id).collect::<Vec<_>>(), [1, 3]);
        assert!((s.chosen_logprob - logprob_of(&logits, 2)).abs() < 1e-6);
        // With n 0 there is no top, and the chosen logprob is still there.
        let s0 = StepLogprobs::of(0, 1, &logits, 0);
        assert!(s0.top.is_empty());
        assert!((s0.chosen_logprob - logprob_of(&logits, 1)).abs() < 1e-6);
    }

    #[test]
    fn chosen_logprob_matches_its_top_entry_and_ignores_nan() {
        let logits = [f32::NAN, 1.0, 3.0, 0.0];
        let s = StepLogprobs::of(0, 2, &logits, 3);
        assert_eq!(s.top[0].token_id, 2);
        assert_eq!(s.chosen_logprob, s.top[0].logprob);
        assert!(s.chosen_logprob.is_finite());
        // An id past the vocabulary has no probability, never a panic.
        assert_eq!(
            StepLogprobs::of(0, 99, &logits, 1).chosen_logprob,
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn top2_margin_case_table() {
        // (logits, n, expected margin): the V3 near-tie measure is top[0] - top[1].
        let table: &[(&[f32], usize, Option<f32>)] = &[
            (&[1.0, 1.01, -3.0], 2, Some(0.01)),
            (&[5.0, 1.0, 0.0], 2, Some(4.0)),
            (&[2.0, 2.0], 2, Some(0.0)),
            (&[5.0, 1.0, 0.0], 1, None),
            (&[5.0, 1.0, 0.0], 0, None),
            (&[5.0], 2, None),
        ];
        for &(logits, n, want) in table {
            let got = StepLogprobs::of(0, 0, logits, n).top2_margin();
            match (got, want) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-5, "{logits:?} n={n}: {g}"),
                (g, w) => assert_eq!(g, w, "{logits:?} n={n}"),
            }
        }
        let near_tie = StepLogprobs::of(0, 1, &[1.0, 1.01, -3.0], 2);
        assert!(near_tie.top2_margin().is_some_and(|m| m < 0.05));
        let clear = StepLogprobs::of(0, 0, &[5.0, 1.0, 0.0], 2);
        assert!(clear.top2_margin().is_some_and(|m| m > 1.0));
    }
}
