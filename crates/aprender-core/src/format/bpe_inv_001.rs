// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE) algorithm-level
// PARTIAL discharge for INV-BPE-001.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// AC-SHIP2-002 (tokenizer trained, vocab bound + paired). 8th PROPOSED
// contract surface bound at the algorithm level.
//
// ## What INV-BPE-001 says
//
//   description: vocab_size ∈ [32000, 65536] and matches the paired
//                model's embedding row count (see llama-370m-sovereign-v1
//                INV-ARCH-370M-006). Default: exactly 50257 (GPT-2
//                canonical — 50_000 BPE merges + 256 byte-level fallback
//                tokens + 1 sentinel, with our 4 special tokens
//                allocated from the non-mergeable slots).
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the tokenizer's actual vocab_size and the paired
// model's embedding row count, Pass iff:
//
//   1. tokenizer_vocab ∈ [32000, 65536] inclusive,
//   2. paired_model_vocab ∈ [32000, 65536] inclusive,
//   3. tokenizer_vocab == paired_model_vocab.
//
// Pinning the [32000, 65536] bounds means a future drift to a 25K vocab
// (would cripple Python source representation) or a 100K+ vocab (would
// blow up the embedding matrix) trips the gate. Pinning equality between
// tokenizer and paired model means a future contract bump that updates
// only one of the two artifacts (drift class — `feedback_monorepo
// _single_source_of_truth.md`) is also caught.

/// Lower bound for vocab_size (inclusive). Per contract §INV-BPE-001:
/// 32K is the minimum size that gives reasonable Python BPE coverage
/// without inflating the embedding matrix beyond the 370M model's
/// parameter budget. Sub-32K would force excessive byte fallback.
pub const AC_BPE_INV_001_MIN_VOCAB: u32 = 32_000;

/// Upper bound for vocab_size (inclusive). Per contract §INV-BPE-001:
/// 64K is the maximum size before the embedding matrix dominates the
/// 370M parameter budget (50_257 * 1024 = 51M params already at GPT-2
/// canonical; 65_536 doubles the embed-cost share to ~67M).
pub const AC_BPE_INV_001_MAX_VOCAB: u32 = 65_536;

/// Default GPT-2 canonical vocab. Per contract §INV-BPE-001:
/// `50_000` BPE merges + `256` byte-level fallback + `1` sentinel.
pub const AC_BPE_INV_001_DEFAULT_VOCAB: u32 = 50_257;

/// Binary verdict for `INV-BPE-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv001Verdict {
    /// Both tokenizer and paired model report vocab in [32000, 65536]
    /// AND the two values are byte-equal.
    Pass,
    /// One or more of:
    /// - tokenizer_vocab outside [32000, 65536].
    /// - paired_model_vocab outside [32000, 65536].
    /// - tokenizer_vocab != paired_model_vocab (drift between
    ///   tokenizer and embedding rows).
    Fail,
}

/// Pure verdict function for `INV-BPE-001`.
///
/// Inputs:
/// - `tokenizer_vocab`: vocab size from the trained tokenizer
///   (e.g., `apr tokenize --info tokenizer.json | jq '.vocab_size'`).
/// - `paired_model_vocab`: embedding row count of the paired model
///   (e.g., `Llama370MConfig::VOCAB_SIZE`).
///
/// Pass iff both are in `[AC_BPE_INV_001_MIN_VOCAB, AC_BPE_INV_001_MAX_VOCAB]`
/// inclusive AND `tokenizer_vocab == paired_model_vocab`.
///
/// # Examples
///
/// GPT-2 canonical 50,257 paired correctly — `Pass`:
/// ```
/// use aprender::format::bpe_inv_001::{
///     verdict_from_vocab_size_pair, BpeInv001Verdict,
/// };
/// let v = verdict_from_vocab_size_pair(50_257, 50_257);
/// assert_eq!(v, BpeInv001Verdict::Pass);
/// ```
///
/// Tokenizer / model drift — `Fail`:
/// ```
/// use aprender::format::bpe_inv_001::{
///     verdict_from_vocab_size_pair, BpeInv001Verdict,
/// };
/// // Tokenizer trained at 50_257, model embed sized at 32_768 — mismatch.
/// let v = verdict_from_vocab_size_pair(50_257, 32_768);
/// assert_eq!(v, BpeInv001Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_vocab_size_pair(
    tokenizer_vocab: u32,
    paired_model_vocab: u32,
) -> BpeInv001Verdict {
    if tokenizer_vocab < AC_BPE_INV_001_MIN_VOCAB || tokenizer_vocab > AC_BPE_INV_001_MAX_VOCAB {
        return BpeInv001Verdict::Fail;
    }
    if paired_model_vocab < AC_BPE_INV_001_MIN_VOCAB
        || paired_model_vocab > AC_BPE_INV_001_MAX_VOCAB
    {
        return BpeInv001Verdict::Fail;
    }
    if tokenizer_vocab != paired_model_vocab {
        return BpeInv001Verdict::Fail;
    }
    BpeInv001Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — bounds match contract.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_vocab_is_32_000() {
        assert_eq!(AC_BPE_INV_001_MIN_VOCAB, 32_000);
    }

    #[test]
    fn provenance_max_vocab_is_65_536() {
        assert_eq!(AC_BPE_INV_001_MAX_VOCAB, 65_536);
    }

    #[test]
    fn provenance_default_vocab_is_gpt2_canonical() {
        assert_eq!(AC_BPE_INV_001_DEFAULT_VOCAB, 50_257);
    }

    #[test]
    fn provenance_default_within_bounds() {
        assert!(AC_BPE_INV_001_DEFAULT_VOCAB >= AC_BPE_INV_001_MIN_VOCAB);
        assert!(AC_BPE_INV_001_DEFAULT_VOCAB <= AC_BPE_INV_001_MAX_VOCAB);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — vocabs in range AND paired correctly.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_gpt2_canonical_50_257() {
        let v = verdict_from_vocab_size_pair(50_257, 50_257);
        assert_eq!(v, BpeInv001Verdict::Pass);
    }

    #[test]
    fn pass_at_lower_boundary() {
        let v = verdict_from_vocab_size_pair(32_000, 32_000);
        assert_eq!(
            v,
            BpeInv001Verdict::Pass,
            "exact 32_000 must Pass (inclusive)"
        );
    }

    #[test]
    fn pass_at_upper_boundary() {
        let v = verdict_from_vocab_size_pair(65_536, 65_536);
        assert_eq!(
            v,
            BpeInv001Verdict::Pass,
            "exact 65_536 must Pass (inclusive)"
        );
    }

    #[test]
    fn pass_qwen_typical_32_768() {
        let v = verdict_from_vocab_size_pair(32_768, 32_768);
        assert_eq!(v, BpeInv001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — drift between tokenizer and model.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_drift_50_257_vs_32_768() {
        // Both within bounds, but unequal — the canonical drift class.
        let v = verdict_from_vocab_size_pair(50_257, 32_768);
        assert_eq!(v, BpeInv001Verdict::Fail);
    }

    #[test]
    fn fail_off_by_one_high() {
        let v = verdict_from_vocab_size_pair(50_257, 50_258);
        assert_eq!(v, BpeInv001Verdict::Fail, "1-token drift must Fail");
    }

    #[test]
    fn fail_off_by_one_low() {
        let v = verdict_from_vocab_size_pair(50_257, 50_256);
        assert_eq!(v, BpeInv001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — tokenizer below lower bound.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_tokenizer_below_min() {
        let v = verdict_from_vocab_size_pair(31_999, 31_999);
        assert_eq!(
            v,
            BpeInv001Verdict::Fail,
            "31_999 below 32_000 minimum must Fail"
        );
    }

    #[test]
    fn fail_tokenizer_at_zero() {
        let v = verdict_from_vocab_size_pair(0, 0);
        assert_eq!(v, BpeInv001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — model above upper bound.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_model_above_max() {
        let v = verdict_from_vocab_size_pair(65_537, 65_537);
        assert_eq!(
            v,
            BpeInv001Verdict::Fail,
            "65_537 above 65_536 maximum must Fail"
        );
    }

    #[test]
    fn fail_qwen_full_vocab_151_936() {
        // Qwen2.5-Coder-7B has vocab 151_936, far above this contract's
        // [32K, 64K] window — that's a different contract surface.
        let v = verdict_from_vocab_size_pair(151_936, 151_936);
        assert_eq!(
            v,
            BpeInv001Verdict::Fail,
            "Qwen vocab is out-of-scope for this contract; Fail"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Asymmetry probe — only one of the pair out-of-bounds.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_only_tokenizer_below() {
        let v = verdict_from_vocab_size_pair(31_999, 50_257);
        assert_eq!(v, BpeInv001Verdict::Fail);
    }

    #[test]
    fn fail_only_model_above() {
        let v = verdict_from_vocab_size_pair(50_257, 65_537);
        assert_eq!(v, BpeInv001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Boundary sweep at fixed paired-model-vocab.
    // -------------------------------------------------------------------------
    #[test]
    fn boundary_sweep_around_default() {
        let model = AC_BPE_INV_001_DEFAULT_VOCAB; // 50_257
        let probes: Vec<(u32, BpeInv001Verdict)> = vec![
            (0, BpeInv001Verdict::Fail),
            (31_999, BpeInv001Verdict::Fail),
            (32_000, BpeInv001Verdict::Fail), // in range but != model
            (50_256, BpeInv001Verdict::Fail), // 1 below model
            (50_257, BpeInv001Verdict::Pass), // exact match
            (50_258, BpeInv001Verdict::Fail), // 1 above model
            (65_536, BpeInv001Verdict::Fail), // in range but != model
            (65_537, BpeInv001Verdict::Fail), // out of range
            (u32::MAX, BpeInv001Verdict::Fail),
        ];
        for (tok_vocab, expected) in probes {
            let v = verdict_from_vocab_size_pair(tok_vocab, model);
            assert_eq!(
                v, expected,
                "tokenizer={tok_vocab} model={model} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Const evaluability — verdict is `pub const fn`.
    // -------------------------------------------------------------------------
    #[test]
    fn const_eval_works_in_static_context() {
        const PASS: BpeInv001Verdict = verdict_from_vocab_size_pair(50_257, 50_257);
        const FAIL_DRIFT: BpeInv001Verdict = verdict_from_vocab_size_pair(50_257, 32_768);
        const FAIL_OOR: BpeInv001Verdict = verdict_from_vocab_size_pair(0, 0);
        assert_eq!(PASS, BpeInv001Verdict::Pass);
        assert_eq!(FAIL_DRIFT, BpeInv001Verdict::Fail);
        assert_eq!(FAIL_OOR, BpeInv001Verdict::Fail);
    }
}
