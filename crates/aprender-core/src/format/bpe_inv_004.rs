// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE-001)
// algorithm-level PARTIAL discharge for INV-BPE-004.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 tokenizer pipeline (§26.3), AC-SHIP2-003.
//
// ## What INV-BPE-004 says
//
//   description: Merge rules count equals
//                (vocab_size − |special_tokens| − 256 byte-level
//                fallback) ± 4. The ±4 slack covers byte-level
//                fallback edge cases where a byte ID is added
//                directly without a merge.
//   falsifier:   Load merges.txt (or equivalent from
//                tokenizer.json), count lines. Assert
//                |merges| ∈ [49993, 50001] (for the canonical 50_257
//                / 4 / 256 / ±4 example). Outside range fails.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`merge_count`, `vocab_size`,
// `special_token_count`, `byte_fallback_count`), compute the
// expected merge count
//
//   expected = vocab_size − special_token_count − byte_fallback_count
//
// and Pass iff:
//
//   |merge_count − expected| <= AC_BPE_INV_004_SLACK (4)
//
// AND the inputs are well-formed:
// - vocab_size > special_token_count + byte_fallback_count (no
//   negative expected; the sum mustn't exhaust the vocab)
// - byte_fallback_count <= 256 (a u9 invariant — there are exactly
//   256 byte values; a count above 256 indicates corruption)
// - the underlying subtraction is `checked_sub` to prevent silent
//   wrapping at degenerate inputs

/// Slack tolerance band on either side of the merge-count
/// expectation.
///
/// Per contract `INV-BPE-004`: ±4 covers byte-level fallback edge
/// cases (e.g., a byte id that is added directly without a merge,
/// or 1-2 reserved bytes that the trainer skips). Drift to ±0 would
/// over-tighten and reject valid byte-fallback variants; drift to
/// ±10 would let a 6-merge regression slip through unnoticed.
pub const AC_BPE_INV_004_SLACK: u64 = 4;

/// Maximum number of byte-level fallback IDs.
///
/// There are exactly 256 byte values (0..256). The fallback set
/// cannot exceed this; a value above 256 indicates corruption in
/// the tokenizer config.
pub const AC_BPE_INV_004_MAX_BYTE_FALLBACK: u64 = 256;

/// Binary verdict for `INV-BPE-004`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv004Verdict {
    /// Merge count is within ±4 of `vocab_size − specials − bytes`,
    /// and all inputs are well-formed.
    Pass,
    /// One or more of:
    /// - `byte_fallback_count > 256` (corruption — there are only 256 bytes).
    /// - `vocab_size <= special_token_count + byte_fallback_count`
    ///   (caller error — no room for any merges).
    /// - `merge_count` differs from the expected value by more than ±4.
    /// - Subtraction underflow (degenerate caller input).
    Fail,
}

/// Pure verdict function for `INV-BPE-004`.
///
/// Inputs:
/// - `merge_count`: actual merge-rule line count from `merges.txt`.
/// - `vocab_size`: declared total vocabulary size.
/// - `special_token_count`: number of special tokens (e.g., 4 for
///   BOS/EOS/PAD/UNK; can be larger for chat-template tokens).
/// - `byte_fallback_count`: number of byte-level fallback IDs
///   (typically 256).
///
/// Pass iff:
/// 1. `byte_fallback_count <= 256`,
/// 2. `vocab_size > special_token_count + byte_fallback_count`
///    (computed via `checked_add` + `checked_sub`),
/// 3. `|merge_count − expected| <= 4`
///    where `expected = vocab_size − special_token_count − byte_fallback_count`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// GPT-2 canonical: 50_257 vocab − 1 special − 256 bytes = 50_000
/// expected merges. Actual 50_000 → `Pass`:
/// ```
/// use aprender::format::bpe_inv_004::{
///     verdict_from_merge_rule_count, BpeInv004Verdict,
/// };
/// let v = verdict_from_merge_rule_count(50_000, 50_257, 1, 256);
/// assert_eq!(v, BpeInv004Verdict::Pass);
/// ```
///
/// Same expectation, actual 50_004 (within ±4 slack) → `Pass`:
/// ```
/// use aprender::format::bpe_inv_004::{
///     verdict_from_merge_rule_count, BpeInv004Verdict,
/// };
/// let v = verdict_from_merge_rule_count(50_004, 50_257, 1, 256);
/// assert_eq!(v, BpeInv004Verdict::Pass);
/// ```
///
/// Drift of 5 merges → `Fail`:
/// ```
/// use aprender::format::bpe_inv_004::{
///     verdict_from_merge_rule_count, BpeInv004Verdict,
/// };
/// let v = verdict_from_merge_rule_count(50_005, 50_257, 1, 256);
/// assert_eq!(v, BpeInv004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_merge_rule_count(
    merge_count: u64,
    vocab_size: u64,
    special_token_count: u64,
    byte_fallback_count: u64,
) -> BpeInv004Verdict {
    if byte_fallback_count > AC_BPE_INV_004_MAX_BYTE_FALLBACK {
        return BpeInv004Verdict::Fail;
    }
    let reserved = match special_token_count.checked_add(byte_fallback_count) {
        Some(v) => v,
        None => return BpeInv004Verdict::Fail,
    };
    let expected = match vocab_size.checked_sub(reserved) {
        Some(v) if v > 0 => v,
        _ => return BpeInv004Verdict::Fail,
    };
    let abs_diff = merge_count.abs_diff(expected);
    if abs_diff <= AC_BPE_INV_004_SLACK {
        BpeInv004Verdict::Pass
    } else {
        BpeInv004Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — slack and byte cap.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_slack_is_four() {
        assert_eq!(AC_BPE_INV_004_SLACK, 4);
    }

    #[test]
    fn provenance_max_byte_fallback_is_256() {
        assert_eq!(AC_BPE_INV_004_MAX_BYTE_FALLBACK, 256);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — exact and within-slack matches.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_gpt2_canonical_exact() {
        // 50_257 − 1 − 256 = 50_000 expected.
        let v = verdict_from_merge_rule_count(50_000, 50_257, 1, 256);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }

    #[test]
    fn pass_gpt2_at_plus_4_slack() {
        let v = verdict_from_merge_rule_count(50_004, 50_257, 1, 256);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }

    #[test]
    fn pass_gpt2_at_minus_4_slack() {
        // 50_257 − 1 − 256 = 50_000; merge=49_996 is at -4 slack.
        let v = verdict_from_merge_rule_count(49_996, 50_257, 1, 256);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }

    #[test]
    fn pass_qwen_4_specials() {
        // Qwen-style: 152_064 vocab − 22 specials − 256 bytes = 151_786 expected.
        let v = verdict_from_merge_rule_count(151_786, 152_064, 22, 256);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }

    #[test]
    fn pass_albor_llama_370m_4_specials() {
        // MODEL-2 albor-llama: 32_000 − 4 − 256 = 31_740 expected.
        let v = verdict_from_merge_rule_count(31_740, 32_000, 4, 256);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — outside ±4 slack.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_gpt2_at_plus_5() {
        let v = verdict_from_merge_rule_count(50_005, 50_257, 1, 256);
        assert_eq!(
            v,
            BpeInv004Verdict::Fail,
            "+5 merges exceeds ±4 slack"
        );
    }

    #[test]
    fn fail_gpt2_at_minus_5() {
        let v = verdict_from_merge_rule_count(49_995, 50_257, 1, 256);
        assert_eq!(v, BpeInv004Verdict::Fail);
    }

    #[test]
    fn fail_gpt2_at_plus_100() {
        let v = verdict_from_merge_rule_count(50_100, 50_257, 1, 256);
        assert_eq!(v, BpeInv004Verdict::Fail);
    }

    #[test]
    fn fail_zero_merges_when_thousands_expected() {
        // Catastrophic: merges.txt empty.
        let v = verdict_from_merge_rule_count(0, 50_257, 1, 256);
        assert_eq!(v, BpeInv004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — input domain violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_byte_fallback_above_256() {
        // Only 256 byte values exist; 257 is corruption.
        let v = verdict_from_merge_rule_count(50_000, 50_257, 1, 257);
        assert_eq!(
            v,
            BpeInv004Verdict::Fail,
            "byte_fallback > 256 must Fail (impossible by definition)"
        );
    }

    #[test]
    fn fail_byte_fallback_far_above_256() {
        let v = verdict_from_merge_rule_count(50_000, 50_257, 1, 1_000_000);
        assert_eq!(v, BpeInv004Verdict::Fail);
    }

    #[test]
    fn fail_vocab_too_small_for_reserved() {
        // vocab=200, specials=4, bytes=256 → reserved=260 > 200 → Fail.
        let v = verdict_from_merge_rule_count(0, 200, 4, 256);
        assert_eq!(
            v,
            BpeInv004Verdict::Fail,
            "vocab < specials + bytes must Fail (no room for merges)"
        );
    }

    #[test]
    fn fail_vocab_exactly_equals_reserved() {
        // vocab=260, specials=4, bytes=256 → expected=0 (no merges
        // at all). The verdict refuses this as caller error: a BPE
        // tokenizer with zero merges is degenerate.
        let v = verdict_from_merge_rule_count(0, 260, 4, 256);
        assert_eq!(v, BpeInv004Verdict::Fail);
    }

    #[test]
    fn fail_specials_plus_bytes_overflow() {
        // u64 overflow on (specials + bytes).
        let huge = u64::MAX / 2 + 1;
        let v = verdict_from_merge_rule_count(50_000, u64::MAX, huge, 256);
        // huge + 256 < huge + huge; still a single overflow path
        // exists if specials = u64::MAX. Test the unambiguous case:
        let v2 = verdict_from_merge_rule_count(50_000, u64::MAX, u64::MAX, 256);
        assert_eq!(
            v,
            BpeInv004Verdict::Fail,
            "any caller-error condition must Fail"
        );
        assert_eq!(v2, BpeInv004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — ±4 at fixed expectation.
    // -------------------------------------------------------------------------
    #[test]
    fn merge_count_sweep_around_50_000() {
        let probes: Vec<(u64, BpeInv004Verdict)> = vec![
            (49_990, BpeInv004Verdict::Fail),
            (49_995, BpeInv004Verdict::Fail), // -5
            (49_996, BpeInv004Verdict::Pass), // -4 (inclusive)
            (49_997, BpeInv004Verdict::Pass),
            (49_999, BpeInv004Verdict::Pass),
            (50_000, BpeInv004Verdict::Pass), // exact
            (50_001, BpeInv004Verdict::Pass),
            (50_003, BpeInv004Verdict::Pass),
            (50_004, BpeInv004Verdict::Pass), // +4 (inclusive)
            (50_005, BpeInv004Verdict::Fail), // +5
            (50_010, BpeInv004Verdict::Fail),
        ];
        for (merge, expected) in probes {
            let v = verdict_from_merge_rule_count(merge, 50_257, 1, 256);
            assert_eq!(v, expected, "merge={merge} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry — ±k symmetric around exact.
    // -------------------------------------------------------------------------
    #[test]
    fn slack_is_symmetric() {
        // For any k in 0..=4, both expected+k and expected-k must Pass.
        let expected = 50_000_u64;
        for k in 0..=4_u64 {
            let v_high = verdict_from_merge_rule_count(expected + k, 50_257, 1, 256);
            let v_low = verdict_from_merge_rule_count(expected - k, 50_257, 1, 256);
            assert_eq!(v_high, BpeInv004Verdict::Pass, "+{k} slack");
            assert_eq!(v_low, BpeInv004Verdict::Pass, "-{k} slack");
        }
        // ±5 must Fail on both sides.
        for k in 5..=8_u64 {
            let v_high = verdict_from_merge_rule_count(expected + k, 50_257, 1, 256);
            let v_low = verdict_from_merge_rule_count(expected - k, 50_257, 1, 256);
            assert_eq!(v_high, BpeInv004Verdict::Fail, "+{k} drift");
            assert_eq!(v_low, BpeInv004Verdict::Fail, "-{k} drift");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — multiple model families with non-256-byte fallbacks.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_with_zero_byte_fallback() {
        // Pure word-piece tokenizer: no byte-level fallback (0).
        // 1000 vocab − 4 specials − 0 bytes = 996 expected.
        let v = verdict_from_merge_rule_count(996, 1_000, 4, 0);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }

    #[test]
    fn pass_at_exactly_256_byte_fallback_boundary() {
        // byte_fallback = 256 is the inclusive cap.
        let v = verdict_from_merge_rule_count(31_740, 32_000, 4, 256);
        assert_eq!(v, BpeInv004Verdict::Pass);
    }
}
