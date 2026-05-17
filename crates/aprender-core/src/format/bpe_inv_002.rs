// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE-001)
// algorithm-level PARTIAL discharge for INV-BPE-002.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 tokenizer pipeline (§26.3), AC-SHIP2-003.
//
// ## What INV-BPE-002 says
//
//   description: All four required special tokens (BOS, EOS, PAD,
//                UNK) are present in the vocabulary with distinct
//                IDs in [0, vocab_size).
//   falsifier:   For each of <|bos|>, <|eos|>, <|pad|>, <|unk|>:
//                tokenizer.token_to_id() returns Some(i) with
//                i ∈ [0, vocab_size). No two IDs may collide.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the four resolved special-token IDs and the
// declared `vocab_size`, Pass iff:
//
//   vocab_size > 0 AND
//   (bos_id, eos_id, pad_id, unk_id) all in [0, vocab_size) AND
//   all four IDs are pairwise distinct
//
// The contract falsifier admits no near-miss: a single missing
// special token, an out-of-range ID, OR any pair colliding (e.g.,
// BOS == EOS) trips the gate. Tokenizers that "save bytes" by
// reusing UNK as PAD are the most common regression mode caught
// here — pairwise distinct catches all `C(4,2) = 6` collision
// patterns.

/// Number of required special tokens (BOS, EOS, PAD, UNK).
///
/// Per contract `INV-BPE-002`: pinning the count at 4 catches a
/// regression that adds a 5th required token (without bumping the
/// contract) OR drops one of the four below. The count is enforced
/// implicitly by the verdict's 4-tuple input shape.
pub const AC_BPE_INV_002_REQUIRED_SPECIAL_TOKENS: u32 = 4;

/// Binary verdict for `INV-BPE-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv002Verdict {
    /// All four required special tokens have IDs in `[0, vocab_size)`
    /// AND are pairwise distinct.
    Pass,
    /// One or more of:
    /// - `vocab_size == 0` (caller error — empty vocabulary).
    /// - One of `(bos_id, eos_id, pad_id, unk_id)` is `>= vocab_size`
    ///   (out-of-range).
    /// - Any pair of the four IDs collide (e.g., BOS == EOS, PAD == UNK).
    Fail,
}

/// Pure verdict function for `INV-BPE-002`.
///
/// Inputs:
/// - `bos_id`: token ID returned by `tokenizer.token_to_id("<|bos|>")`.
/// - `eos_id`: token ID returned by `tokenizer.token_to_id("<|eos|>")`.
/// - `pad_id`: token ID returned by `tokenizer.token_to_id("<|pad|>")`.
/// - `unk_id`: token ID returned by `tokenizer.token_to_id("<|unk|>")`.
/// - `vocab_size`: declared size of the BPE vocabulary.
///
/// Pass iff:
/// 1. `vocab_size > 0`,
/// 2. `bos_id < vocab_size`,
/// 3. `eos_id < vocab_size`,
/// 4. `pad_id < vocab_size`,
/// 5. `unk_id < vocab_size`,
/// 6. `bos_id`, `eos_id`, `pad_id`, `unk_id` are pairwise distinct.
///
/// Otherwise `Fail`.
///
/// Note: callers must resolve "missing" tokens to a sentinel that
/// fails the range check — typically `u32::MAX`. If the underlying
/// tokenizer's `token_to_id` returns `None`, the caller should pass
/// `u32::MAX` so the verdict catches it via the out-of-range
/// predicate.
///
/// # Examples
///
/// Typical 50K vocab with 4 distinct special-token IDs at the top —
/// `Pass`:
/// ```
/// use aprender::format::bpe_inv_002::{
///     verdict_from_special_token_ids, BpeInv002Verdict,
/// };
/// let v = verdict_from_special_token_ids(50_000, 50_001, 50_002, 50_003, 50_257);
/// assert_eq!(v, BpeInv002Verdict::Pass);
/// ```
///
/// PAD reused as UNK (collision) — `Fail`:
/// ```
/// use aprender::format::bpe_inv_002::{
///     verdict_from_special_token_ids, BpeInv002Verdict,
/// };
/// let v = verdict_from_special_token_ids(50_000, 50_001, 50_002, 50_002, 50_257);
/// assert_eq!(v, BpeInv002Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_special_token_ids(
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
    unk_id: u32,
    vocab_size: u32,
) -> BpeInv002Verdict {
    if vocab_size == 0 {
        return BpeInv002Verdict::Fail;
    }
    if bos_id >= vocab_size || eos_id >= vocab_size || pad_id >= vocab_size || unk_id >= vocab_size
    {
        return BpeInv002Verdict::Fail;
    }
    // Pairwise distinct: 6 unordered pairs over 4 IDs.
    if bos_id == eos_id
        || bos_id == pad_id
        || bos_id == unk_id
        || eos_id == pad_id
        || eos_id == unk_id
        || pad_id == unk_id
    {
        return BpeInv002Verdict::Fail;
    }
    BpeInv002Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_special_token_count_is_four() {
        assert_eq!(AC_BPE_INV_002_REQUIRED_SPECIAL_TOKENS, 4);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — typical layouts.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_typical_50k_vocab_top_special_ids() {
        let v = verdict_from_special_token_ids(50_000, 50_001, 50_002, 50_003, 50_257);
        assert_eq!(v, BpeInv002Verdict::Pass);
    }

    #[test]
    fn pass_special_ids_at_low_end() {
        // GPT-style sometimes places special tokens at low IDs.
        let v = verdict_from_special_token_ids(0, 1, 2, 3, 50_257);
        assert_eq!(v, BpeInv002Verdict::Pass);
    }

    #[test]
    fn pass_realistic_albor_llama_370m_vocab() {
        // MODEL-2 albor-llama uses Llama-style 32K vocab with special
        // tokens scattered.
        let v = verdict_from_special_token_ids(1, 2, 0, 3, 32_000);
        assert_eq!(v, BpeInv002Verdict::Pass);
    }

    #[test]
    fn pass_minimal_vocab_size_4() {
        // Smallest valid vocab: 4 distinct IDs in [0, 4).
        let v = verdict_from_special_token_ids(0, 1, 2, 3, 4);
        assert_eq!(v, BpeInv002Verdict::Pass);
    }

    #[test]
    fn pass_at_vocab_minus_one_boundary() {
        // Highest valid ID: vocab_size - 1.
        let v = verdict_from_special_token_ids(99, 98, 97, 96, 100);
        assert_eq!(v, BpeInv002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — out-of-range IDs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_bos_at_vocab_size() {
        // ID == vocab_size is out-of-range (range is exclusive).
        let v = verdict_from_special_token_ids(100, 1, 2, 3, 100);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_eos_above_vocab_size() {
        let v = verdict_from_special_token_ids(0, 200, 1, 2, 100);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_missing_token_sentinel_u32_max() {
        // Convention: caller passes u32::MAX for "missing" token.
        let v = verdict_from_special_token_ids(0, 1, 2, u32::MAX, 100);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_all_four_at_u32_max() {
        let v = verdict_from_special_token_ids(u32::MAX, u32::MAX, u32::MAX, u32::MAX, 100);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — pairwise collisions (all 6 unordered pairs).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_bos_eos_collision() {
        let v = verdict_from_special_token_ids(50_000, 50_000, 50_001, 50_002, 50_257);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_bos_pad_collision() {
        let v = verdict_from_special_token_ids(50_000, 50_001, 50_000, 50_002, 50_257);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_bos_unk_collision() {
        let v = verdict_from_special_token_ids(50_000, 50_001, 50_002, 50_000, 50_257);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_eos_pad_collision() {
        let v = verdict_from_special_token_ids(50_000, 50_001, 50_001, 50_002, 50_257);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_eos_unk_collision() {
        let v = verdict_from_special_token_ids(50_000, 50_001, 50_002, 50_001, 50_257);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    #[test]
    fn fail_pad_unk_collision() {
        // Most common regression: bytes-saving tokenizer reuses UNK
        // as PAD. Catches it explicitly.
        let v = verdict_from_special_token_ids(50_000, 50_001, 50_002, 50_002, 50_257);
        assert_eq!(
            v,
            BpeInv002Verdict::Fail,
            "PAD == UNK is the canonical regression — must Fail"
        );
    }

    #[test]
    fn fail_all_four_collide() {
        let v = verdict_from_special_token_ids(0, 0, 0, 0, 100);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — caller errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_vocab_size() {
        let v = verdict_from_special_token_ids(0, 1, 2, 3, 0);
        assert_eq!(
            v,
            BpeInv002Verdict::Fail,
            "vocab_size == 0 is degenerate and must Fail"
        );
    }

    #[test]
    fn fail_vocab_size_3_cannot_fit_4_distinct() {
        // 4 distinct IDs in [0, 3) is impossible by pigeonhole.
        let v = verdict_from_special_token_ids(0, 1, 2, 0, 3);
        assert_eq!(v, BpeInv002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry — verdict invariant under permuting the 4 IDs.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_invariant_under_permutation() {
        // (10, 20, 30, 40) and (40, 30, 20, 10) are both 4 distinct
        // IDs in [0, 100) — both Pass regardless of slot assignment.
        let v1 = verdict_from_special_token_ids(10, 20, 30, 40, 100);
        let v2 = verdict_from_special_token_ids(40, 30, 20, 10, 100);
        let v3 = verdict_from_special_token_ids(20, 40, 10, 30, 100);
        assert_eq!(v1, BpeInv002Verdict::Pass);
        assert_eq!(v2, BpeInv002Verdict::Pass);
        assert_eq!(v3, BpeInv002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — Qwen2.5-Coder-7B-Instruct vocab layout.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_realistic_qwen2_5_coder_layout() {
        // Qwen2.5-Coder-7B uses 152_064 vocab with special tokens at
        // the top. Plausible IDs:
        let v = verdict_from_special_token_ids(151_643, 151_645, 151_644, 151_646, 152_064);
        assert_eq!(v, BpeInv002Verdict::Pass);
    }
}
