//! FALSIFY-CB-008 (`contracts/continuous-batching-v1.yaml`), as a checker that runs everywhere.
//!
//! The rule is *"No frozen slots — all M slots produce distinct tokens per decode step (not
//! constant)"*, and aprender#2753 is that rule failing in production: every slot served from a
//! batch emitted **one token id to the `max_tokens` cap** and never reached a stop token, while
//! the request that happened to take the m=1 fast path in the same second, on the same binary
//! and the same prompt, returned coherent English with `finish_reason=stop`.
//!
//! ```text
//!   slot  path            output                     finish   tokens
//!   A     m=1 fast path   coherent English           stop     319
//!   B,C,D from the batch  `!!!!…` (token id 0)       length   400
//! ```
//!
//! The contract forbade that before it shipped and nothing checked it. CB-008's `test:` field
//! named a `BATCHED_DECODE_TRACE` log — a variable that did not exist — so the evidence it
//! described could never have been read, and the check that DOES exist today
//! (`aprender-gpu`'s `cb008_online_softmax_rescale`) asserts about the PTX of one kernel that
//! turned out to be one of several mechanisms. Neither states the rule.
//!
//! This module states it, as a pure function over a slot's generated token stream, so that:
//!
//! 1. the property is decided by ONE piece of code, used by the GPU falsifier
//!    (`tests/falsify_cb008_no_frozen_slots_2753.rs`) and testable without a GPU;
//! 2. its **can-it-fire** controls run in the ordinary workspace `--lib` line — the checker is
//!    proven to reject the exact recorded signatures (id 0 to the cap, id 151662 to the cap)
//!    on every PR, on a host with no CUDA at all. A checker whose discrimination is only ever
//!    exercised on a nightly GPU lane is a checker nobody checks, which is the failure this
//!    file exists to end.
//!
//! WHAT IS ASSERTED, and why it is the temporal reading. For M identical greedy prompts the
//! slots *should* agree with each other, so "distinct **across** slots" would be a wrong
//! requirement — it would go red on a correct decoder. The defect was never cross-slot: it was
//! a slot whose OWN stream stopped moving. So the property is per slot, over time: a stream
//! that is one token repeated is frozen, and a stream that stalls on one token for a long run
//! is frozen even if it moved before or after.
//!
//! Two thresholds, both far from BOTH regimes so the verdict is never a coin flip:
//! the recorded defect produced 1–3 distinct ids across 120–400 tokens with a run of ~115,
//! while ordinary generated text produces a near-unique id per position with runs of 1–2.

/// Fewest distinct token ids a healthy generated stream of at least [`MIN_GENERATED`] tokens
/// must contain. The recorded #2753 streams had 1–3 across 120–400 tokens.
pub const MIN_DISTINCT: usize = 6;

/// Longest run of one repeated token id a healthy stream may contain. The recorded #2753
/// streams ran the same id for ~115 of 120 steps. Real text does repeat — `" "`, `"\n"`,
/// a list marker — so this is deliberately loose; it is not a repetition-quality gate.
pub const MAX_RUN: usize = 8;

/// Fewest generated tokens for a verdict to mean anything. Below this, "the stream varies" is
/// a statement about a handful of tokens, so the checker refuses rather than passing.
pub const MIN_GENERATED: usize = 24;

/// Longest run of a single repeated token id.
#[must_use]
pub fn longest_run(tokens: &[u32]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    let mut prev: Option<u32> = None;
    for &t in tokens {
        if prev == Some(t) {
            run += 1;
        } else {
            run = 1;
            prev = Some(t);
        }
        best = best.max(run);
    }
    best
}

/// Number of distinct token ids in the stream.
#[must_use]
pub fn distinct_count(tokens: &[u32]) -> usize {
    let mut seen: Vec<u32> = tokens.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

/// CB-008 for one slot's GENERATED tokens (the suffix after the prompt).
///
/// `Err` carries the whole verdict, ready to panic with. **A stream too short to judge is an
/// `Err` too, not an `Ok`** — this repo has shipped four gates that could not fail, and
/// "nothing to check" reading as "checked and fine" is how three of them got there.
///
/// # Errors
/// Returns the verdict text when the slot is frozen, or when the stream is too short to decide.
pub fn frozen_slot_verdict(slot: usize, tokens: &[u32]) -> Result<(), String> {
    if tokens.len() < MIN_GENERATED {
        return Err(format!(
            "CB-008 UNMEASURABLE: slot {slot} generated {} token(s), fewer than the \
             {MIN_GENERATED} this verdict needs. Reported as a failure, not a pass: a slot \
             that produced almost nothing cannot show whether it FREEZES, and a silent pass \
             here is the shape aprender#2753 hid behind for four releases.",
            tokens.len()
        ));
    }
    let distinct = distinct_count(tokens);
    let run = longest_run(tokens);
    if distinct >= MIN_DISTINCT && run <= MAX_RUN {
        return Ok(());
    }
    let head: Vec<u32> = tokens.iter().copied().take(12).collect();
    Err(format!(
        "CB-008 RED (aprender#2753): slot {slot} is FROZEN. {len} generated tokens with \
         {distinct} distinct id(s) (floor {MIN_DISTINCT}) and a longest single-id run of \
         {run} (ceiling {MAX_RUN}). head={head:?}. The contract's rule is \"No frozen slots — \
         all M slots produce distinct tokens per decode step (not constant)\"; a batched slot \
         emitting one id to the max_tokens cap never reaches a stop token, which is why every \
         such request also returns finish_reason=length.",
        len = tokens.len(),
    ))
}

/// CB-008 across a whole batch. Every slot is judged; the verdict names all failures, because
/// "which slots froze" is the first question asked of a red run and re-running to find out
/// costs a GPU minute.
///
/// # Errors
/// Returns the combined verdict when any slot is frozen or too short to judge.
pub fn batch_frozen_verdict(slots: &[Vec<u32>]) -> Result<(), String> {
    if slots.is_empty() {
        return Err("CB-008 UNMEASURABLE: the batch produced no slots at all.".to_string());
    }
    let mut bad: Vec<String> = Vec::new();
    for (i, tokens) in slots.iter().enumerate() {
        if let Err(v) = frozen_slot_verdict(i, tokens) {
            bad.push(v);
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of {} slot(s) failed CB-008:\n{}",
            bad.len(),
            slots.len(),
            bad.join("\n")
        ))
    }
}

#[cfg(test)]
mod cb008_checker_controls {
    use super::*;

    /// A varied stream, the shape ordinary generated text has.
    fn healthy(len: usize) -> Vec<u32> {
        (0..len).map(|i| 1000 + (i as u32 * 7) % 313).collect()
    }

    #[test]
    fn a_healthy_stream_is_accepted() {
        frozen_slot_verdict(0, &healthy(48)).expect("a varied stream must pass CB-008");
    }

    /// The signature aprender#2753 opened with: token id 0 repeated to the 400-token cap.
    #[test]
    fn token_id_zero_to_the_cap_is_rejected() {
        let v = frozen_slot_verdict(2, &vec![0u32; 400]).expect_err(
            "400 copies of token id 0 is the exact output #2753 reported; the checker MUST \
             reject it or it cannot see the defect it was written for",
        );
        assert!(v.contains("FROZEN"), "{v}");
        assert!(v.contains("1 distinct"), "{v}");
    }

    /// The signature measured on this branch before the fix: id 151662 for 115 of 120 steps.
    #[test]
    fn the_measured_151662_stream_is_rejected() {
        let mut s = vec![151_662u32; 116];
        s.extend_from_slice(&[151_661, 151_661, 27_224, 151_661]);
        let v = frozen_slot_verdict(1, &s).expect_err(
            "this is the byte-for-byte token stream BATCHED_DECODE_TRACE printed at m=3 on \
             origin/main; a checker that accepts it is theatre",
        );
        assert!(v.contains("FROZEN"), "{v}");
    }

    /// Variety alone must not buy a pass: a stream that stalls for a long run is still frozen.
    /// Without this, a decoder that emits 30 distinct ids and then wedges reads as healthy.
    #[test]
    fn a_long_stall_is_rejected_even_with_variety() {
        let mut s = healthy(30);
        s.extend(std::iter::repeat_n(4242u32, 60));
        let v = frozen_slot_verdict(0, &s)
            .expect_err("a 60-token stall must be rejected even though 31 ids appear");
        assert!(v.contains("longest single-id run of 60"), "{v}");
    }

    /// A stream too short to judge is an error, never a pass.
    #[test]
    fn a_stream_too_short_to_judge_is_not_a_pass() {
        let v = frozen_slot_verdict(0, &[7, 8, 9])
            .expect_err("3 tokens cannot show whether a slot freezes");
        assert!(v.contains("UNMEASURABLE"), "{v}");
        assert!(
            !v.contains("RED"),
            "an unmeasurable run must not claim a code defect: {v}"
        );
    }

    /// The boundary is exercised in both directions, so a threshold typo cannot pass silently.
    #[test]
    fn the_thresholds_are_the_ones_documented() {
        let just_enough: Vec<u32> = (0..MIN_GENERATED as u32).map(|i| i % 6).collect();
        assert_eq!(distinct_count(&just_enough), MIN_DISTINCT);
        frozen_slot_verdict(0, &just_enough).expect("exactly MIN_DISTINCT ids must pass");

        let one_short: Vec<u32> = (0..MIN_GENERATED as u32).map(|i| i % 5).collect();
        assert_eq!(distinct_count(&one_short), MIN_DISTINCT - 1);
        frozen_slot_verdict(0, &one_short).expect_err("one below MIN_DISTINCT must fail");

        let mut at_ceiling = healthy(48);
        at_ceiling.splice(0..0, std::iter::repeat_n(99u32, MAX_RUN));
        assert_eq!(longest_run(&at_ceiling), MAX_RUN);
        frozen_slot_verdict(0, &at_ceiling).expect("a run of exactly MAX_RUN must pass");
    }

    /// One frozen slot beside healthy ones must fail the batch and NAME it. The #2753 table
    /// has exactly this shape: slot A fine, slots B/C/D frozen.
    #[test]
    fn one_frozen_slot_fails_the_batch_and_is_named() {
        let slots = vec![healthy(48), vec![151_662u32; 48], healthy(48)];
        let v = batch_frozen_verdict(&slots).expect_err("a frozen slot must fail the batch");
        assert!(v.contains("1 of 3 slot(s)"), "{v}");
        assert!(
            v.contains("slot 1"),
            "the verdict must name which slot froze: {v}"
        );
    }

    #[test]
    fn an_all_healthy_batch_passes() {
        batch_frozen_verdict(&[healthy(48), healthy(48), healthy(48)])
            .expect("three varied slots must pass");
    }

    /// An empty batch is refused rather than passing vacuously — the same class as the
    /// short-stream control, at the batch level.
    #[test]
    fn an_empty_batch_is_refused() {
        let v = batch_frozen_verdict(&[]).expect_err("no slots is not a pass");
        assert!(v.contains("UNMEASURABLE"), "{v}");
    }

    #[test]
    fn longest_run_and_distinct_count_are_what_they_say() {
        assert_eq!(longest_run(&[]), 0);
        assert_eq!(longest_run(&[5]), 1);
        assert_eq!(longest_run(&[1, 1, 2, 1, 1, 1]), 3);
        assert_eq!(distinct_count(&[]), 0);
        assert_eq!(distinct_count(&[4, 4, 4]), 1);
        assert_eq!(distinct_count(&[4, 5, 4, 6]), 3);
    }
}
