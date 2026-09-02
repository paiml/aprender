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
//! # What is compared
//!
//! The witness prompt is decoded twice at `temperature 0`: once alone (`m = 1`)
//! and once inside a batch of `m = c` sequences. The two token-id sequences must
//! agree for at least `declared_min` tokens (`witness.min_agree_tokens` in
//! `perf-matrix.yaml`, 64 `[U]` until fp-nondeterminism is measured).
//!
//! Three verdicts, not two. "The streams agree but both stopped at 32 tokens"
//! is **not** a pass — it is [`BatchInvariance::Unmeasurable`], because the
//! evidence does not reach the declared point. Collapsing it into `Pass` is how
//! a short run buys a correctness verdict it never earned.

use serde::{Deserialize, Serialize};

/// PP-26's verdict. The wire tokens are `"PASS"`, `"FAIL"`, `"UNMEASURABLE"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum BatchInvariance {
    /// The two streams agreed for at least `declared_min` tokens.
    Pass,
    /// The two streams diverged before `declared_min`.
    Fail,
    /// Neither stream reached `declared_min`, so nothing was proven either way.
    Unmeasurable,
}

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
    /// Index of the first differing token, when the streams diverged before the
    /// end of the shorter one. `None` when they never diverged.
    pub divergence_at: Option<u32>,
    /// Tokens that had to agree, from `perf-matrix.yaml`.
    pub declared_min: u32,
    /// The batch size the batched arm actually formed. `0` when the caller has
    /// not declared it — see [`BatchInvarianceWitness::formed_at`].
    pub m_formed: u32,
    /// Where the comparison came from, e.g. `scripts/perf041_batched_parity_probe.py`.
    pub source: String,
}

impl BatchInvarianceWitness {
    /// Compare an `m = 1` token stream against the same prompt's stream from
    /// inside a batch.
    ///
    /// The `m_formed` and `source` fields are left undeclared (`0`, and a
    /// self-describing string); a caller that knows the batch size records it
    /// with [`Self::formed_at`]. Keeping them out of this signature is
    /// deliberate: they are provenance about *how the probe ran*, and inventing
    /// them from the token streams would be exactly the harness-inferred field
    /// PP-13 refuses.
    #[must_use]
    pub fn compare(m1_tokens: &[u32], batched: &[u32], declared_min: u32) -> Self {
        let (verdict, divergence_at) = Self::verdict(m1_tokens, batched, declared_min);
        Self {
            batch_invariance: verdict,
            divergence_at,
            declared_min,
            m_formed: 0,
            source: "client-side token comparison of m=1 against the batched arm".to_string(),
        }
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

    fn verdict(m1: &[u32], batched: &[u32], declared_min: u32) -> (BatchInvariance, Option<u32>) {
        if m1.is_empty() || batched.is_empty() {
            return (BatchInvariance::Unmeasurable, None);
        }
        let shortest = m1.len().min(batched.len());
        let agree = m1
            .iter()
            .zip(batched.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let diverged = agree < shortest;
        let agree_u32 = u32::try_from(agree).unwrap_or(u32::MAX);
        if agree_u32 >= declared_min {
            // Enough agreement to decide, whether or not they later parted.
            return (
                BatchInvariance::Pass,
                if diverged { Some(agree_u32) } else { None },
            );
        }
        if diverged {
            (BatchInvariance::Fail, Some(agree_u32))
        } else {
            // Identical as far as either stream goes, but that is short of the
            // declared point. Nothing was proven.
            (BatchInvariance::Unmeasurable, None)
        }
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

    /// And the boundary is exact: agreeing on exactly `declared_min` tokens
    /// passes, one fewer fails.
    #[test]
    fn the_declared_minimum_is_an_inclusive_boundary() {
        let m1: Vec<u32> = (0..128).map(|i| 4000 + i).collect();
        let mut at_min = m1.clone();
        at_min[64] = 7;
        let mut below = m1.clone();
        below[63] = 7;
        assert_eq!(
            BatchInvarianceWitness::compare(&m1, &at_min, 64).batch_invariance,
            BatchInvariance::Pass
        );
        assert_eq!(
            BatchInvarianceWitness::compare(&m1, &below, 64).batch_invariance,
            BatchInvariance::Fail
        );
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
