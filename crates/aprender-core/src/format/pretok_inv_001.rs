// SHIP-TWO-001 MODEL-2 — `pretokenize-bin-v1` (C-PRETOK-BIN) algorithm-level
// PARTIAL discharge for INV-PRETOK-001.
//
// Contract: `contracts/pretokenize-bin-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` MODEL-2
// corpus pipeline (§26.2), §35 distill needs the binary shard format.
//
// ## What INV-PRETOK-001 says
//
//   description: Every token id written to any shard is in [0, vocab_size).
//                No token id may equal or exceed the paired tokenizer's
//                vocab_size (otherwise the embedding lookup overflows).
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a per-shard maximum token id and the paired
// tokenizer's `vocab_size`, Pass iff every shard's max < vocab_size AND
// at least one shard exists AND vocab_size is at least 1.
//
// Future implementations (the actual `apr tokenize encode-corpus` shard
// writer) cannot:
// - Emit a shard with `max >= vocab_size` (would overflow embedding lookup
//   and crash the trainer's `nn::Embedding` — caught here)
// - Emit zero shards (caller bug — Fail)
// - Be paired with a `vocab_size = 0` tokenizer (no tokens valid — Fail)

/// Binary verdict for `INV-PRETOK-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokInv001Verdict {
    /// Every shard's maximum token id is strictly less than `vocab_size`,
    /// at least one shard was emitted, and `vocab_size > 0`.
    /// The pretokenizer's output is safe for `nn::Embedding(vocab_size, _)`
    /// lookup downstream.
    Pass,
    /// One or more of:
    /// - `vocab_size == 0` (no tokens valid — caller error).
    /// - Empty shard list (no work done — caller error).
    /// - Some shard's max token id is `>= vocab_size`
    ///   (would crash the trainer's embedding lookup).
    Fail,
}

/// Pure verdict function for `INV-PRETOK-001`.
///
/// Inputs:
/// - `per_shard_max_token_ids`: max token id observed per output shard
///   (e.g., from `apr tokenize encode-corpus` post-encode summary).
/// - `vocab_size`: tokenizer's vocab size (e.g., 50257 for GPT-2 BPE,
///   151_936 for Qwen2.5-Coder).
///
/// Pass iff:
/// 1. `vocab_size > 0`,
/// 2. `per_shard_max_token_ids` is non-empty,
/// 3. every entry is `< vocab_size`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Typical Qwen2.5-Coder 151,936-vocab corpus — `Pass`:
/// ```
/// use aprender::format::pretok_inv_001::{
///     verdict_from_per_shard_max_token_ids, PretokInv001Verdict,
/// };
/// let max_ids = vec![151_500_u32, 151_900, 0, 42];
/// assert_eq!(
///     verdict_from_per_shard_max_token_ids(&max_ids, 151_936),
///     PretokInv001Verdict::Pass,
/// );
/// ```
///
/// Embedding overflow — `Fail`:
/// ```
/// use aprender::format::pretok_inv_001::{
///     verdict_from_per_shard_max_token_ids, PretokInv001Verdict,
/// };
/// let max_ids = vec![151_936_u32]; // == vocab_size; would overflow nn::Embedding
/// assert_eq!(
///     verdict_from_per_shard_max_token_ids(&max_ids, 151_936),
///     PretokInv001Verdict::Fail,
/// );
/// ```
#[must_use]
pub fn verdict_from_per_shard_max_token_ids(
    per_shard_max_token_ids: &[u32],
    vocab_size: u32,
) -> PretokInv001Verdict {
    if vocab_size == 0 {
        return PretokInv001Verdict::Fail;
    }
    if per_shard_max_token_ids.is_empty() {
        return PretokInv001Verdict::Fail;
    }
    for &max_id in per_shard_max_token_ids {
        if max_id >= vocab_size {
            return PretokInv001Verdict::Fail;
        }
    }
    PretokInv001Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Common Qwen2.5-Coder vocab size used in MODEL-1 work.
    const QWEN_VOCAB: u32 = 151_936;

    /// Common 370M sovereign model vocab (GPT-2 BPE).
    const SMALL_VOCAB: u32 = 50_257;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — well-formed inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_typical_qwen_corpus() {
        let max_ids = vec![QWEN_VOCAB - 1, QWEN_VOCAB / 2, 1, 0];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Pass
        );
    }

    #[test]
    fn pass_typical_small_corpus() {
        let max_ids = vec![50_256_u32, 1234, 0, 50_000];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, SMALL_VOCAB),
            PretokInv001Verdict::Pass
        );
    }

    #[test]
    fn pass_single_shard() {
        let max_ids = vec![100_u32];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, 200),
            PretokInv001Verdict::Pass
        );
    }

    #[test]
    fn pass_max_id_just_below_vocab() {
        // max == vocab - 1 is the highest valid id (strict `<`).
        let max_ids = vec![QWEN_VOCAB - 1];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Pass
        );
    }

    #[test]
    fn pass_all_zeros() {
        // Degenerate but valid: every shard contains only token id 0
        // (e.g., a corpus where every text gets the BOS token only).
        let max_ids = vec![0_u32; 100];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — token id ≥ vocab_size (embedding overflow).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_max_id_equals_vocab_size() {
        // == vocab_size is OOB — embedding indexed by [0, vocab_size).
        let max_ids = vec![QWEN_VOCAB];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Fail
        );
    }

    #[test]
    fn fail_max_id_above_vocab() {
        let max_ids = vec![QWEN_VOCAB + 100];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Fail
        );
    }

    #[test]
    fn fail_one_bad_among_many_good() {
        let mut max_ids = vec![QWEN_VOCAB - 1; 100];
        max_ids[42] = QWEN_VOCAB; // single shard violates
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Fail
        );
    }

    #[test]
    fn fail_max_id_at_u32_max() {
        // Worst-case attacker / mojibake corruption: token id = u32::MAX.
        let max_ids = vec![u32::MAX];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — caller errors (vocab_size = 0, empty input).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_vocab_size_zero() {
        let max_ids = vec![0_u32];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, 0),
            PretokInv001Verdict::Fail,
            "vocab_size == 0 has no valid tokens; conservative Fail"
        );
    }

    #[test]
    fn fail_empty_shard_list() {
        let max_ids: Vec<u32> = vec![];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Fail,
            "empty shard list implies pretokenize did nothing"
        );
    }

    #[test]
    fn fail_both_empty_and_zero_vocab() {
        let max_ids: Vec<u32> = vec![];
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, 0),
            PretokInv001Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Boundary sweep at vocab_size threshold.
    // -------------------------------------------------------------------------
    #[test]
    fn boundary_sweep_around_vocab_size() {
        let v: u32 = 100;
        let probes: Vec<(u32, PretokInv001Verdict)> = vec![
            (0, PretokInv001Verdict::Pass),
            (1, PretokInv001Verdict::Pass),
            (50, PretokInv001Verdict::Pass),
            (98, PretokInv001Verdict::Pass),
            (99, PretokInv001Verdict::Pass), // == v - 1, the max valid
            (100, PretokInv001Verdict::Fail), // == v, OOB
            (101, PretokInv001Verdict::Fail),
            (200, PretokInv001Verdict::Fail),
        ];
        for (max_id, expected) in probes {
            let max_ids = vec![max_id];
            assert_eq!(
                verdict_from_per_shard_max_token_ids(&max_ids, v),
                expected,
                "max_id={max_id} vocab={v} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 5: Vocab-size sweep — a single fixed max_id at varying vocab.
    // -------------------------------------------------------------------------
    #[test]
    fn vocab_size_sweep_at_fixed_max_id() {
        let max_id: u32 = 100;
        let probes: Vec<(u32, PretokInv001Verdict)> = vec![
            (0, PretokInv001Verdict::Fail),   // vocab=0 → Fail
            (50, PretokInv001Verdict::Fail),  // 100 >= 50 → Fail
            (99, PretokInv001Verdict::Fail),  // 100 >= 99 → Fail
            (100, PretokInv001Verdict::Fail), // 100 >= 100 → Fail (boundary)
            (101, PretokInv001Verdict::Pass), // 100 < 101 → Pass
            (1_000, PretokInv001Verdict::Pass),
            (u32::MAX, PretokInv001Verdict::Pass),
        ];
        for (vocab, expected) in probes {
            let max_ids = vec![max_id];
            assert_eq!(
                verdict_from_per_shard_max_token_ids(&max_ids, vocab),
                expected,
                "max_id={max_id} vocab={vocab} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Multi-shard mutation — bad shard at any position fails.
    // -------------------------------------------------------------------------
    #[test]
    fn single_bad_shard_at_each_position_fails() {
        for bad_idx in [0_usize, 1, 5, 9] {
            let mut max_ids = vec![QWEN_VOCAB - 1; 10];
            max_ids[bad_idx] = QWEN_VOCAB;
            assert_eq!(
                verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
                PretokInv001Verdict::Fail,
                "bad shard at index {bad_idx} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic-scale stress — 10K shards.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_scale_10k_shards_pass() {
        let max_ids: Vec<u32> = (0..10_000).map(|i| i % QWEN_VOCAB).collect();
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Pass
        );
    }

    #[test]
    fn realistic_scale_10k_shards_one_bad_fails() {
        let mut max_ids: Vec<u32> = (0..10_000).map(|i| i % QWEN_VOCAB).collect();
        max_ids[5_000] = QWEN_VOCAB + 1;
        assert_eq!(
            verdict_from_per_shard_max_token_ids(&max_ids, QWEN_VOCAB),
            PretokInv001Verdict::Fail
        );
    }
}
