//! #4971 V3-d: the witness from OpenAI `logprobs`, with the top-2 margin.
//!
//! [`BatchInvarianceWitness::compare_batch`] compares token ids, and the
//! OpenAI wire carries none: each generated token arrives as one
//! `logprobs.content[]` entry with its text, its bytes and the step's best
//! tokens. This module is the bridge. [`token_ids`] numbers the entries by text
//! and bytes, so the comparison runs unchanged on them, and [`top2_margin`]
//! reads the gap between a step's two best tokens.
//!
//! Two logprobs of one step share one log-sum-exp, so their difference IS the
//! logit difference, `ln p0 - ln p1 = l0 - l1`, whenever the logprobs are taken
//! over the unscaled logits. The witness decodes at temperature 0, where they
//! are. The margin needs no logit on the wire.
//!
//! Only the `m = 1` stream's margin is recorded (master §12 row 22). It says how
//! close the reference's own choice was at the token where the batch parted
//! from it; below [`super::v3_shape::NEAR_TIE_EPS`] the V3 shape check reads
//! the divergence as a near-tie flip, not a defect.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::witness::BatchInvarianceWitness;

/// One generated token, as OpenAI's `logprobs.content[]` carries it.
///
/// `logprob` is optional because a server serialising `-inf` through
/// `serde_json` writes `null`, and `bytes` because OpenAI allows `null` there.
/// Unknown keys (llama.cpp sends `id`) are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The chosen token's text.
    pub token: String,
    /// `ln P(token)` at this step.
    #[serde(default)]
    pub logprob: Option<f64>,
    /// The token's own bytes, which can be part of one character.
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
    /// The step's best tokens, best first.
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprob>,
}

/// One of a step's best tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopLogprob {
    /// The token's text.
    pub token: String,
    /// `ln P(token)` at this step.
    #[serde(default)]
    pub logprob: Option<f64>,
    /// The token's own bytes.
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
}

/// The source string of a witness built from logprob entries.
pub const LOGPROBS_SOURCE: &str =
    "client-side comparison of OpenAI logprob entries across the batch's slots (m=1 top-2 margin recorded)";

impl TokenLogprob {
    /// What makes two entries the same token: the text and the bytes. Text
    /// alone is lossy for a token that is part of one character.
    fn identity(&self) -> (&str, &[u8]) {
        (
            self.token.as_str(),
            self.bytes.as_deref().unwrap_or_default(),
        )
    }
}

/// The gap between a step's two best tokens, `ln p_best - ln p_second`.
///
/// `None` when the step lists fewer than two finite logprobs: an absent margin
/// is never `0.0`, which would read as the closest near-tie there is. The two
/// largest are taken whatever order the server listed them in, so the margin is
/// never negative.
#[must_use]
pub fn top2_margin(entry: &TokenLogprob) -> Option<f64> {
    let mut finite = entry
        .top_logprobs
        .iter()
        .filter_map(|top| top.logprob)
        .filter(|lp| lp.is_finite());
    let (mut best, mut second) = (finite.next()?, finite.next()?);
    if second > best {
        std::mem::swap(&mut best, &mut second);
    }
    for lp in finite {
        if lp > best {
            second = best;
            best = lp;
        } else if lp > second {
            second = lp;
        }
    }
    Some(best - second)
}

/// Number every stream's tokens so they compare as ids. Two entries share an id
/// exactly when they share text and bytes, across all the streams given.
#[must_use]
pub fn token_ids(streams: &[&[TokenLogprob]]) -> Vec<Vec<u32>> {
    let mut ids: HashMap<(&str, &[u8]), u32> = HashMap::new();
    streams
        .iter()
        .map(|stream| {
            stream
                .iter()
                .map(|entry| {
                    let next = u32::try_from(ids.len()).unwrap_or(u32::MAX);
                    *ids.entry(entry.identity()).or_insert(next)
                })
                .collect()
        })
        .collect()
}

impl BatchInvarianceWitness {
    /// PP-26 v3.1 over logprob streams, as [`Self::compare_batch`] decides it,
    /// plus the `m = 1` stream's top-2 margin at `divergence_at`.
    ///
    /// The margin stays `None` when the streams never part, or when the `m = 1`
    /// entry at the divergence lists fewer than two finite logprobs.
    #[must_use]
    pub fn compare_logprobs(
        m1: &[TokenLogprob],
        slots: &[&[TokenLogprob]],
        declared_min: u32,
        max_constant_run: u32,
    ) -> Self {
        let streams: Vec<&[TokenLogprob]> =
            std::iter::once(m1).chain(slots.iter().copied()).collect();
        let ids = token_ids(&streams);
        let (m1_ids, slot_ids) = ids
            .split_first()
            .map_or((&[][..], &[][..]), |(m1, rest)| (m1.as_slice(), rest));
        let slot_refs: Vec<&[u32]> = slot_ids.iter().map(Vec::as_slice).collect();
        let mut witness = Self::compare_batch(m1_ids, &slot_refs, declared_min, max_constant_run);
        witness.top2_margin_at_divergence = witness
            .divergence_at
            .and_then(|at| m1.get(usize::try_from(at).ok()?))
            .and_then(top2_margin);
        witness.source = LOGPROBS_SOURCE.to_string();
        witness
    }
}

#[cfg(test)]
mod tests {
    use super::super::v3_shape::NEAR_TIE_EPS;
    use super::super::witness::{BatchInvariance, DEFAULT_MAX_CONSTANT_RUN};
    use super::*;

    /// `ln softmax(logits)`, as a server computes logprobs from the logits.
    fn log_softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
        logits.iter().map(|l| l - lse).collect()
    }

    fn top(token: &str, logprob: f64) -> TopLogprob {
        TopLogprob {
            token: token.to_string(),
            logprob: Some(logprob),
            bytes: Some(token.as_bytes().to_vec()),
        }
    }

    /// One step that chose `token`, whose step logits put `token` first by
    /// `gap` over `runner_up` and far ahead of a third token.
    fn step(token: &str, runner_up: &str, gap: f64) -> TokenLogprob {
        let lps = log_softmax(&[10.0, 10.0 - gap, 0.0]);
        TokenLogprob {
            token: token.to_string(),
            logprob: Some(lps[0]),
            bytes: Some(token.as_bytes().to_vec()),
            top_logprobs: vec![top(token, lps[0]), top(runner_up, lps[1]), top("~", lps[2])],
        }
    }

    /// A 128-token `m = 1` stream, decisive at every step except `tie_at`,
    /// where its top two are `tie_gap` apart.
    fn reference(tie_at: usize, tie_gap: f64) -> Vec<TokenLogprob> {
        (0..128)
            .map(|i| {
                let gap = if i == tie_at { tie_gap } else { 4.0 };
                step(&format!("t{i}"), &format!("u{i}"), gap)
            })
            .collect()
    }

    /// The batch's stream: the reference until `at`, the runner-up there, and
    /// its own tokens after.
    fn parted_at(m1: &[TokenLogprob], at: usize) -> Vec<TokenLogprob> {
        let mut out = m1.to_vec();
        out[at] = step(&format!("u{at}"), &format!("t{at}"), 0.5);
        for (i, entry) in out.iter_mut().enumerate().skip(at + 1) {
            *entry = step(&format!("b{i}"), &format!("c{i}"), 4.0);
        }
        out
    }

    /// The claim the module rests on: a logprob gap equals the logit gap.
    #[test]
    fn the_logprob_gap_is_the_logit_gap() {
        for gap in [0.0, 0.01, 0.049, 0.5, 3.25] {
            let margin = top2_margin(&step("a", "b", gap)).expect("two finite logprobs");
            assert!((margin - gap).abs() < 1e-9, "gap {gap} read as {margin}");
        }
    }

    /// The planted near-tie: the batch takes the runner-up where the `m = 1`
    /// choice led by 0.01. The witness records the divergence and the margin,
    /// and the margin sits inside the near-tie band.
    #[test]
    fn a_near_tie_flip_records_its_margin() {
        let m1 = reference(3, 0.01);
        let batch = parted_at(&m1, 3);
        let w = BatchInvarianceWitness::compare_logprobs(
            &m1,
            &[&batch, &batch],
            64,
            DEFAULT_MAX_CONSTANT_RUN,
        );
        assert_eq!(w.divergence_at, Some(3));
        assert_eq!(w.batch_invariance, BatchInvariance::Pass, "slots agree");
        let margin = w.top2_margin_at_divergence.expect("margin recorded");
        assert!((margin - 0.01).abs() < 1e-9, "{margin}");
        assert!(margin < NEAR_TIE_EPS);
        assert_eq!(w.source, LOGPROBS_SOURCE);
    }

    /// The must-fire twin: the same flip where the `m = 1` choice led by 2.5
    /// records 2.5, which no near-tie band admits.
    #[test]
    fn a_decisive_flip_records_a_margin_outside_the_band() {
        let m1 = reference(3, 2.5);
        let batch = parted_at(&m1, 3);
        let w = BatchInvarianceWitness::compare_logprobs(&m1, &[&batch], 64, 16);
        let margin = w.top2_margin_at_divergence.expect("margin recorded");
        assert!((margin - 2.5).abs() < 1e-9, "{margin}");
        assert!(margin >= NEAR_TIE_EPS);
    }

    /// The margin is the `m = 1` stream's, not the batch's: the batch's own
    /// step at the divergence leads by 0.5, and 0.5 is not what is recorded.
    #[test]
    fn the_margin_is_read_from_the_m1_stream() {
        let m1 = reference(5, 0.02);
        let batch = parted_at(&m1, 5);
        let w = BatchInvarianceWitness::compare_logprobs(&m1, &[&batch], 64, 16);
        let margin = w.top2_margin_at_divergence.expect("margin recorded");
        assert!((margin - 0.02).abs() < 1e-9, "{margin}");
    }

    /// No divergence, no margin: identical streams record neither.
    #[test]
    fn identical_streams_record_no_divergence_and_no_margin() {
        let m1 = reference(3, 0.01);
        let w = BatchInvarianceWitness::compare_logprobs(&m1, &[&m1, &m1], 64, 16);
        assert_eq!(w.divergence_at, None);
        assert_eq!(w.top2_margin_at_divergence, None);
        assert_eq!(w.batch_invariance, BatchInvariance::Pass);
    }

    /// MUST-FIRE: an `m = 1` step with fewer than two finite logprobs has no
    /// margin. Reading it as `0.0` would admit the divergence as the closest
    /// near-tie there is.
    #[test]
    fn a_step_without_two_finite_logprobs_has_no_margin_never_zero() {
        let mut lone = step("a", "b", 0.01);
        lone.top_logprobs.truncate(1);
        assert_eq!(top2_margin(&lone), None);
        lone.top_logprobs.clear();
        assert_eq!(top2_margin(&lone), None);
        let mut nulled = step("a", "b", 0.01);
        nulled.top_logprobs[1].logprob = None;
        nulled.top_logprobs[2].logprob = None;
        assert_eq!(top2_margin(&nulled), None);
        let mut nan = step("a", "b", 0.01);
        nan.top_logprobs[1].logprob = Some(f64::NAN);
        nan.top_logprobs[2].logprob = Some(f64::NEG_INFINITY);
        assert_eq!(top2_margin(&nan), None);

        let mut m1 = reference(3, 0.01);
        m1[3].top_logprobs.truncate(1);
        let batch = parted_at(&m1, 3);
        let w = BatchInvarianceWitness::compare_logprobs(&m1, &[&batch], 64, 16);
        assert_eq!(w.divergence_at, Some(3));
        assert_eq!(w.top2_margin_at_divergence, None);
    }

    /// The best two are found in any listed order, and the gap is never negative.
    #[test]
    fn the_margin_does_not_depend_on_the_listed_order() {
        let mut entry = step("a", "b", 0.3);
        entry.top_logprobs.reverse();
        let margin = top2_margin(&entry).expect("two finite logprobs");
        assert!((margin - 0.3).abs() < 1e-9, "{margin}");
        entry.top_logprobs.swap(0, 1);
        assert!((top2_margin(&entry).expect("still two") - 0.3).abs() < 1e-9);
    }

    /// Ids follow text AND bytes: one text with other bytes is another token,
    /// and the same token gets one id across streams.
    #[test]
    fn token_ids_follow_text_and_bytes_across_streams() {
        let a = step("a", "b", 1.0);
        let mut a_other_bytes = a.clone();
        a_other_bytes.bytes = Some(vec![0xE2]);
        let mut a_no_bytes = a.clone();
        a_no_bytes.bytes = None;
        let first = [a.clone(), a_other_bytes];
        let second = [a, a_no_bytes];
        let ids = token_ids(&[&first, &second]);
        assert_eq!(ids[0][0], ids[1][0], "same token, same id across streams");
        assert_ne!(ids[0][0], ids[0][1], "other bytes, other token");
        assert_ne!(
            ids[0][0], ids[1][1],
            "absent bytes differ from present ones"
        );
        assert_ne!(ids[0][1], ids[1][1]);
    }

    /// The wire shapes the witness reads: realizar's (bytes as numbers), OpenAI's
    /// (`bytes: null`), a `-inf` serialised as `null`, and llama.cpp's extra `id`.
    #[test]
    fn the_served_wire_shapes_parse() {
        let realizar = r#"{"token":"Hi","logprob":-0.1,"bytes":[72,105],
            "top_logprobs":[{"token":"Hi","logprob":-0.1,"bytes":[72,105]},
                            {"token":"Hey","logprob":-2.4,"bytes":[72,101,121]}]}"#;
        let e: TokenLogprob = serde_json::from_str(realizar).expect("realizar shape");
        assert_eq!(e.bytes.as_deref(), Some(&b"Hi"[..]));
        assert!((top2_margin(&e).expect("two") - 2.3).abs() < 1e-9);

        let openai = r#"{"token":"Hi","logprob":-0.1,"bytes":null,"top_logprobs":[]}"#;
        let e: TokenLogprob = serde_json::from_str(openai).expect("openai shape");
        assert_eq!(e.bytes, None);
        assert_eq!(top2_margin(&e), None);

        let neg_inf = r#"{"token":"Hi","logprob":null,"bytes":[72,105],
            "top_logprobs":[{"token":"Hi","logprob":-0.1,"bytes":[72,105]},
                            {"token":"x","logprob":null,"bytes":[120]}]}"#;
        let e: TokenLogprob = serde_json::from_str(neg_inf).expect("null logprob");
        assert_eq!(e.logprob, None);
        assert_eq!(top2_margin(&e), None);

        let llama = r#"{"id":13048,"token":"Hi","logprob":-0.1,"bytes":[72,105],
            "top_logprobs":[{"id":13048,"token":"Hi","logprob":-0.1,"bytes":[72,105]},
                            {"id":17,"token":"Hey","logprob":-0.15,"bytes":[72,101,121]}]}"#;
        let e: TokenLogprob = serde_json::from_str(llama).expect("llama.cpp shape");
        assert!((top2_margin(&e).expect("two") - 0.05).abs() < 1e-9);
    }
}
