// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE-001)
// algorithm-level PARTIAL discharge for INV-BPE-006.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 tokenizer pipeline (§26.3), AC-SHIP2-003.
//
// ## What INV-BPE-006 says
//
//   description: Determinism: tokenizer.encode(text) is a pure
//                function of text across process boundaries. Same
//                text → same IDs on every OS, every hardware, every
//                run.
//   falsifier:   Tokenize a fixed 1000-string corpus twice in
//                separate processes; assert ID sequences are
//                bit-identical.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two token-ID sequences from two independent
// processes encoding the same input text, Pass iff:
//
//   run_a.len() == run_b.len() AND
//   run_a == run_b (element-wise equality across all token IDs)
//
// AND both sequences are non-empty (caller error: tokenizing empty
// input is degenerate). Determinism is binary — no near-equality
// band; a single ID drift signals a non-deterministic tokenizer
// (HashMap iteration order, OS-dependent code path, race condition).

/// Binary verdict for `INV-BPE-006`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv006Verdict {
    /// Both ID sequences are non-empty AND element-wise equal.
    Pass,
    /// One or more of:
    /// - Either sequence is empty (caller error — degenerate input).
    /// - Sequences differ in length (token-count mismatch — major
    ///   non-determinism).
    /// - Sequences differ at any element (single-token drift —
    ///   subtle non-determinism).
    Fail,
}

/// Pure verdict function for `INV-BPE-006`.
///
/// Inputs:
/// - `run_a`: token-ID sequence from `tokenizer.encode(text)` in
///   process A.
/// - `run_b`: token-ID sequence from `tokenizer.encode(text)` in
///   process B (independent process, same text).
///
/// Pass iff:
/// 1. `!run_a.is_empty()`,
/// 2. `!run_b.is_empty()`,
/// 3. `run_a == run_b` (element-wise).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Same tokenizer, same text, two independent processes — `Pass`:
/// ```
/// use aprender::format::bpe_inv_006::{
///     verdict_from_token_id_pair, BpeInv006Verdict,
/// };
/// let run_a = vec![1_u32, 42, 17, 99];
/// let run_b = vec![1_u32, 42, 17, 99];
/// let v = verdict_from_token_id_pair(&run_a, &run_b);
/// assert_eq!(v, BpeInv006Verdict::Pass);
/// ```
///
/// Single-token drift (e.g., HashMap iteration order changed) —
/// `Fail`:
/// ```
/// use aprender::format::bpe_inv_006::{
///     verdict_from_token_id_pair, BpeInv006Verdict,
/// };
/// let run_a = vec![1_u32, 42, 17, 99];
/// let run_b = vec![1_u32, 42, 17, 100];
/// let v = verdict_from_token_id_pair(&run_a, &run_b);
/// assert_eq!(v, BpeInv006Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_token_id_pair(run_a: &[u32], run_b: &[u32]) -> BpeInv006Verdict {
    if run_a.is_empty() || run_b.is_empty() {
        return BpeInv006Verdict::Fail;
    }
    if run_a == run_b {
        BpeInv006Verdict::Pass
    } else {
        BpeInv006Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — identical sequences.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_short_identical_sequences() {
        let run_a = vec![1_u32, 42, 17, 99];
        let run_b = vec![1_u32, 42, 17, 99];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Pass);
    }

    #[test]
    fn pass_single_token_identical() {
        let run_a = vec![42_u32];
        let run_b = vec![42_u32];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Pass);
    }

    #[test]
    fn pass_realistic_long_sequence() {
        // 1000-token sequence (the contract's fixed-corpus size).
        let seq: Vec<u32> = (0..1000_u32).collect();
        let v = verdict_from_token_id_pair(&seq, &seq);
        assert_eq!(v, BpeInv006Verdict::Pass);
    }

    #[test]
    fn pass_with_repeated_token_ids() {
        // Common token (e.g., space) repeated.
        let run_a = vec![1_u32, 1, 1, 1, 1];
        let run_b = vec![1_u32, 1, 1, 1, 1];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — single-element drift.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_first_token_differs() {
        let run_a = vec![1_u32, 42, 17, 99];
        let run_b = vec![2_u32, 42, 17, 99];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail, "first-token drift must Fail");
    }

    #[test]
    fn fail_last_token_differs() {
        let run_a = vec![1_u32, 42, 17, 99];
        let run_b = vec![1_u32, 42, 17, 100];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    #[test]
    fn fail_middle_token_differs() {
        let run_a = vec![1_u32, 42, 17, 99];
        let run_b = vec![1_u32, 41, 17, 99];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    #[test]
    fn fail_off_by_one_id() {
        // Smallest possible drift: one ID off by one. Catches a
        // regression that uses a non-deterministic merge-rank
        // tiebreaker.
        let run_a = vec![100_u32, 200, 300];
        let run_b = vec![100_u32, 201, 300];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — length mismatch.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_run_a_longer() {
        let run_a = vec![1_u32, 2, 3, 4];
        let run_b = vec![1_u32, 2, 3];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    #[test]
    fn fail_run_b_longer() {
        let run_a = vec![1_u32, 2, 3];
        let run_b = vec![1_u32, 2, 3, 4];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    #[test]
    fn fail_run_a_double_length() {
        let run_a = vec![1_u32, 2, 3, 4, 1, 2, 3, 4];
        let run_b = vec![1_u32, 2, 3, 4];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — caller errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_both_empty() {
        let v = verdict_from_token_id_pair(&[], &[]);
        assert_eq!(
            v,
            BpeInv006Verdict::Fail,
            "empty sequences must Fail (degenerate input)"
        );
    }

    #[test]
    fn fail_run_a_empty() {
        let v = verdict_from_token_id_pair(&[], &[1_u32, 2, 3]);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    #[test]
    fn fail_run_b_empty() {
        let v = verdict_from_token_id_pair(&[1_u32, 2, 3], &[]);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — completely different sequences.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_completely_different() {
        let run_a = vec![1_u32, 2, 3, 4];
        let run_b = vec![5_u32, 6, 7, 8];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    #[test]
    fn fail_reversed_sequence() {
        let run_a = vec![1_u32, 2, 3, 4];
        let run_b = vec![4_u32, 3, 2, 1];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry — verdict is symmetric in (a, b).
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric_pass() {
        let run_a = vec![1_u32, 2, 3];
        let run_b = vec![1_u32, 2, 3];
        let ab = verdict_from_token_id_pair(&run_a, &run_b);
        let ba = verdict_from_token_id_pair(&run_b, &run_a);
        assert_eq!(ab, ba);
        assert_eq!(ab, BpeInv006Verdict::Pass);
    }

    #[test]
    fn verdict_is_symmetric_fail() {
        let run_a = vec![1_u32, 2, 3];
        let run_b = vec![1_u32, 2, 4];
        let ab = verdict_from_token_id_pair(&run_a, &run_b);
        let ba = verdict_from_token_id_pair(&run_b, &run_a);
        assert_eq!(ab, ba);
        assert_eq!(ab, BpeInv006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Sweep — drift at every position in a 1000-token sequence.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_at_every_drift_position() {
        // Catches a regression that compares only first-N or last-N
        // tokens. Drift at any of 1000 positions must Fail.
        let baseline: Vec<u32> = (0..1000_u32).collect();
        for pos in [0_usize, 1, 100, 500, 999] {
            let mut drift = baseline.clone();
            drift[pos] = drift[pos].wrapping_add(1);
            let v = verdict_from_token_id_pair(&baseline, &drift);
            assert_eq!(
                v,
                BpeInv006Verdict::Fail,
                "drift at position {pos} must Fail"
            );
        }
    }

    #[test]
    fn pass_max_u32_token_id() {
        // Edge: an unusually large token ID (e.g., 152_063 in
        // Qwen2.5-Coder vocab) must compare correctly.
        let run_a = vec![152_063_u32];
        let run_b = vec![152_063_u32];
        let v = verdict_from_token_id_pair(&run_a, &run_b);
        assert_eq!(v, BpeInv006Verdict::Pass);
    }
}
