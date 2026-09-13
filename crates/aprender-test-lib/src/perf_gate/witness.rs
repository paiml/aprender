//! PP-LLAMA-001 v3.0 PP-26 — the batch-invariance witness.
//!
//! # The defect this exists to make unrepresentable
//!
//! §7.0 puts **L0 Correctness** below every speed layer: "were the tokens
//! right?" is asked before "how fast?". Issue #2753 is the shape it catches — a
//! batched decode that emitted the *same token id* for every position, at full
//! speed. Every throughput number that run produced was arithmetically correct
//! and completely meaningless, and nothing in the receipt could say so.
//!
//! P-4: a band whose correctness witness is absent or failing is
//! `INVALID-CORRECTNESS`. Its throughput is not reported, not gated, and never
//! a baseline. That is stronger than "flagged": the numbers are not written.
//!
//! # What is compared (PP-26 v3.1)
//!
//! The witness prompt is decoded at `temperature 0` alone (`m = 1`) and inside
//! a batch of `m = c` identical prompts. Two rules decide the verdict:
//!
//! * **(a) intra-batch invariance** — every slot of the batch agrees with slot 0
//!   for at least `declared_min` tokens (`witness.min_agree_tokens`);
//! * **(b) no frozen slot** — no slot repeats one token id `max_constant_run`
//!   times in a row (`witness.max_constant_run`; #2753's signature ran for 116
//!   steps).
//!
//! The `m = 1` stream's agreement with the batch is **recorded** in
//! `divergence_at`, not gated. The single-stream and batched engines select
//! different kernels and part at the first near-tie; measured on lambda, each
//! kernel family is batch-size invariant to the end while the families differ
//! from each other (`evidence/perf041/lambda/`). Without a top-2 margin on the
//! wire a near-tie flip and a wrong-KV divergence are indistinguishable, so that
//! comparison stays a report until master §12 row 22 lands.
//!
//! Three verdicts, not two. "The slots agree but every stream stopped at 32
//! tokens" is **not** a pass — it is [`BatchInvariance::Unmeasurable`], because
//! the evidence does not reach the declared point. Collapsing it into `Pass` is
//! how a short run buys a correctness verdict it never earned.

use serde::{Deserialize, Serialize};

/// PP-26's verdict. The wire tokens are `"PASS"`, `"FAIL"`, `"UNMEASURABLE"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum BatchInvariance {
    /// Every slot agreed with slot 0 for at least `declared_min` tokens and no
    /// slot froze.
    Pass,
    /// A slot parted from slot 0 before `declared_min`, or a slot repeated one
    /// token id `max_constant_run` times.
    Fail,
    /// No slot stream reached `declared_min`, so nothing was proven either way.
    Unmeasurable,
}

/// `witness.max_constant_run` when the caller declares none: #2753 repeated
/// one id for 116 steps, and coherent text does not repeat one token this long.
pub const DEFAULT_MAX_CONSTANT_RUN: u32 = 16;

impl BatchInvariance {
    /// The wire token, matching the receipt schema.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Unmeasurable => "UNMEASURABLE",
        }
    }
}

/// PP-26 — one band's correctness witness, as it appears in the receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchInvarianceWitness {
    /// The verdict.
    pub batch_invariance: BatchInvariance,
    /// Index of the first token at which the `m = 1` stream and the batch
    /// differ, when they do before the end of the shorter one; `None` when they
    /// never diverged. Recorded, not gated (PP-26 v3.1).
    pub divergence_at: Option<u32>,
    /// PP-26 (a): the shortest agreement between any slot and slot 0. `None`
    /// on a witness written before v3.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_agree_to: Option<u32>,
    /// PP-26 (b): the longest run of one repeated token id in any slot. `None`
    /// on a witness written before v3.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_constant_run: Option<u32>,
    /// Tokens that had to agree, from `perf-matrix.yaml`.
    pub declared_min: u32,
    /// The batch size the batched arm actually formed. `0` when the caller has
    /// not declared it — see [`BatchInvarianceWitness::formed_at`].
    pub m_formed: u32,
    /// Where the comparison came from, e.g. `scripts/perf041_batched_parity_probe.py`.
    pub source: String,
}

impl BatchInvarianceWitness {
    /// Compare an `m = 1` token stream against ONE slot of the batch, with the
    /// default `max_constant_run`. The single-slot case of [`Self::compare_batch`]:
    /// rule (a) is vacuous with one slot, so the verdict is rule (b) plus
    /// reachability of `declared_min`, and the `m = 1` agreement is recorded.
    ///
    /// The `m_formed` and `source` fields are left undeclared (`0`, and a
    /// self-describing string); a caller that knows the batch size records it
    /// with [`Self::formed_at`]. Keeping them out of this signature is
    /// deliberate: they are provenance about *how the probe ran*, and inventing
    /// them from the token streams would be exactly the harness-inferred field
    /// PP-13 refuses.
    #[must_use]
    pub fn compare(m1_tokens: &[u32], batched: &[u32], declared_min: u32) -> Self {
        Self::compare_batch(
            m1_tokens,
            &[batched],
            declared_min,
            DEFAULT_MAX_CONSTANT_RUN,
        )
    }

    /// PP-26 v3.1 over every slot of the batch: (a) each slot agrees with slot 0
    /// for `declared_min` tokens, (b) no slot repeats one id `max_constant_run`
    /// times. `divergence_at` records where the `m = 1` stream and slot 0 part.
    #[must_use]
    pub fn compare_batch(
        m1_tokens: &[u32],
        slots: &[&[u32]],
        declared_min: u32,
        max_constant_run: u32,
    ) -> Self {
        let source = "client-side token comparison across the batch's slots (m=1 recorded)";
        let Some(first) = slots.first() else {
            return Self {
                batch_invariance: BatchInvariance::Unmeasurable,
                divergence_at: None,
                intra_agree_to: None,
                max_constant_run: None,
                declared_min,
                m_formed: 0,
                source: source.to_string(),
            };
        };
        let divergence_at = Self::first_difference(m1_tokens, first);
        let intra = slots
            .iter()
            .map(|slot| Self::agreement(first, slot))
            .min()
            .unwrap_or(0);
        let run = slots
            .iter()
            .map(|slot| Self::longest_constant_run(slot))
            .max()
            .unwrap_or(0);
        let shortest = slots.iter().map(|slot| slot.len()).min().unwrap_or(0);
        let shortest_u32 = u32::try_from(shortest).unwrap_or(u32::MAX);
        let verdict = if slots.iter().any(|slot| slot.is_empty()) || shortest_u32 < declared_min {
            if run >= max_constant_run && max_constant_run > 0 {
                BatchInvariance::Fail
            } else {
                BatchInvariance::Unmeasurable
            }
        } else if (max_constant_run > 0 && run >= max_constant_run) || intra < declared_min {
            BatchInvariance::Fail
        } else {
            BatchInvariance::Pass
        };
        Self {
            batch_invariance: verdict,
            divergence_at,
            intra_agree_to: Some(intra),
            max_constant_run: Some(run),
            declared_min,
            m_formed: 0,
            source: source.to_string(),
        }
    }

    /// Tokens on which two streams agree from the start.
    fn agreement(a: &[u32], b: &[u32]) -> u32 {
        let agree = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
        u32::try_from(agree).unwrap_or(u32::MAX)
    }

    /// `Some(index)` of the first differing token when the streams part before
    /// the end of the shorter one; `None` when they never do (or either is empty).
    fn first_difference(a: &[u32], b: &[u32]) -> Option<u32> {
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let agree = Self::agreement(a, b);
        let shortest = u32::try_from(a.len().min(b.len())).unwrap_or(u32::MAX);
        (agree < shortest).then_some(agree)
    }

    /// Longest run of one repeated token id.
    fn longest_constant_run(tokens: &[u32]) -> u32 {
        let mut best = 0u32;
        let mut run = 0u32;
        let mut prev: Option<u32> = None;
        for &t in tokens {
            run = if prev == Some(t) { run + 1 } else { 1 };
            prev = Some(t);
            best = best.max(run);
        }
        best
    }

    /// Record the batch size the batched arm formed, and where the comparison
    /// came from.
    #[must_use]
    pub fn formed_at(mut self, m_formed: u32, source: impl Into<String>) -> Self {
        self.m_formed = m_formed;
        self.source = source.into();
        self
    }

    /// True only for [`BatchInvariance::Pass`]. Named so the band-status rule
    /// reads as the spec does: absent **or** failing is `INVALID-CORRECTNESS`,
    /// and `Unmeasurable` is on the failing side of that line.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.batch_invariance == BatchInvariance::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #2753's shape: an `m = 3` batch that emitted one token id forever. It
    /// diverges from the real answer at the second position, long before the
    /// declared 64, so the band can never report throughput.
    #[test]
    fn a_constant_token_batch_is_invalid_correctness() {
        let m1: Vec<u32> = (0..128).map(|i| 1000 + i).collect();
        let batched: Vec<u32> = vec![474; 128];
        let w = BatchInvarianceWitness::compare(&m1, &batched, 64)
            .formed_at(3, "scripts/perf041_batched_parity_probe.py");
        assert_eq!(w.batch_invariance, BatchInvariance::Fail);
        assert_eq!(w.divergence_at, Some(0));
        assert_eq!(w.m_formed, 3);
        assert!(!w.passed());
    }

    /// The must-not-fire: `m = 1` and `m = 4` identical for 128 tokens.
    #[test]
    fn identical_128_token_prefixes_pass() {
        let tokens: Vec<u32> = (0..128).map(|i| 2000 + i).collect();
        let w = BatchInvarianceWitness::compare(&tokens, &tokens, 64).formed_at(4, "perf041");
        assert_eq!(w.batch_invariance, BatchInvariance::Pass);
        assert_eq!(w.divergence_at, None);
        assert!(w.passed());
    }

    /// Agreement past the declared point is a PASS even if the tails part —
    /// fp non-determinism after the declared point is what `declared_min`
    /// exists to tolerate. The divergence index is still recorded.
    #[test]
    fn divergence_after_the_declared_point_still_passes_and_is_recorded() {
        let m1: Vec<u32> = (0..128).map(|i| 3000 + i).collect();
        let mut batched = m1.clone();
        batched[100] = 9;
        let w = BatchInvarianceWitness::compare(&m1, &batched, 64);
        assert_eq!(w.batch_invariance, BatchInvariance::Pass);
        assert_eq!(w.divergence_at, Some(100));
    }

    /// And the boundary is exact: two slots agreeing on exactly `declared_min`
    /// tokens pass, one fewer fails (PP-26 v3.1 (a)).
    #[test]
    fn the_declared_minimum_is_an_inclusive_boundary() {
        let m1: Vec<u32> = (0..128).map(|i| 4000 + i).collect();
        let mut at_min = m1.clone();
        at_min[64] = 7;
        let mut below = m1.clone();
        below[63] = 7;
        let w = BatchInvarianceWitness::compare_batch(&m1, &[&m1, &at_min], 64, 16);
        assert_eq!(w.batch_invariance, BatchInvariance::Pass);
        assert_eq!(w.intra_agree_to, Some(64));
        let w = BatchInvarianceWitness::compare_batch(&m1, &[&m1, &below], 64, 16);
        assert_eq!(w.batch_invariance, BatchInvariance::Fail);
        assert_eq!(w.intra_agree_to, Some(63));
    }

    /// The lambda measurement (evidence/perf041/lambda/): every slot of the
    /// batch agrees with every other to the end, and all of them part from the
    /// `m = 1` stream at the third token. That is a PASS with the `m = 1`
    /// agreement RECORDED — the number is kept, the verdict is not taken from it.
    #[test]
    fn a_kernel_family_flip_passes_and_records_the_m1_agreement() {
        let m1: Vec<u32> = (0..128).map(|i| 6000 + i).collect();
        let mut flipped = m1.clone();
        for t in flipped.iter_mut().skip(3) {
            *t += 500;
        }
        let w = BatchInvarianceWitness::compare_batch(
            &m1,
            &[&flipped, &flipped, &flipped, &flipped],
            64,
            16,
        )
        .formed_at(4, "perf041");
        assert_eq!(w.batch_invariance, BatchInvariance::Pass);
        assert_eq!(w.divergence_at, Some(3), "the m=1 agreement is recorded");
        assert_eq!(w.intra_agree_to, Some(128));
        assert_eq!(w.max_constant_run, Some(1));
        assert!(w.passed());
    }

    /// A frozen slot is a defect however short the streams are: the #2753
    /// signature inside a 20-token run must not hide behind `Unmeasurable`.
    #[test]
    fn a_frozen_slot_fails_even_below_the_declared_length() {
        let m1: Vec<u32> = (0..20).map(|i| 7000 + i).collect();
        let frozen = vec![474_u32; 20];
        let w = BatchInvarianceWitness::compare_batch(&m1, &[&m1, &frozen], 64, 16);
        assert_eq!(w.batch_invariance, BatchInvariance::Fail);
        assert_eq!(w.max_constant_run, Some(20));
    }

    /// A witness written before v3.1 carries neither new field and must still
    /// read; the fields are absent from the wire when `None`, not `null`.
    #[test]
    fn a_v3_0_witness_still_deserialises_and_none_fields_stay_off_the_wire() {
        let old = r#"{"batch_invariance":"PASS","divergence_at":null,"declared_min":64,"m_formed":4,"source":"perf041"}"#;
        let w: BatchInvarianceWitness = serde_json::from_str(old).expect("v3.0 shape reads");
        assert_eq!(w.intra_agree_to, None);
        let j = serde_json::to_string(&w).expect("serialises");
        assert!(!j.contains("intra_agree_to"), "{j}");
        let new =
            BatchInvarianceWitness::compare_batch(&[1, 2, 3], &[&[1, 2, 3], &[1, 2, 3]], 2, 16);
        let j = serde_json::to_string(&new).expect("serialises");
        assert!(j.contains("\"intra_agree_to\":3"), "{j}");
    }

    /// Two identical but SHORT streams prove nothing. Reading that as a pass is
    /// how a 32-token smoke run buys a correctness verdict it never earned.
    #[test]
    fn agreement_short_of_the_declared_point_is_unmeasurable_not_a_pass() {
        let short: Vec<u32> = (0..32).map(|i| 5000 + i).collect();
        let w = BatchInvarianceWitness::compare(&short, &short, 64);
        assert_eq!(w.batch_invariance, BatchInvariance::Unmeasurable);
        assert_eq!(w.divergence_at, None);
        assert!(!w.passed(), "Unmeasurable is on the failing side of P-4");
    }

    #[test]
    fn an_empty_stream_is_unmeasurable() {
        assert_eq!(
            BatchInvarianceWitness::compare(&[], &[1, 2, 3], 64).batch_invariance,
            BatchInvariance::Unmeasurable
        );
        assert_eq!(
            BatchInvarianceWitness::compare(&[1, 2, 3], &[], 64).batch_invariance,
            BatchInvariance::Unmeasurable
        );
    }

    #[test]
    fn the_verdict_wire_tokens_are_the_schema_spelling() {
        assert_eq!(BatchInvariance::Pass.wire_token(), "PASS");
        assert_eq!(BatchInvariance::Fail.wire_token(), "FAIL");
        assert_eq!(BatchInvariance::Unmeasurable.wire_token(), "UNMEASURABLE");
        let j = serde_json::to_string(&BatchInvariance::Fail).expect("serialises");
        assert_eq!(j, "\"FAIL\"", "serde must spell it as wire_token does");
    }
}
