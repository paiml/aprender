// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE-001)
// algorithm-level PARTIAL discharge for INV-BPE-007.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 tokenizer pipeline (§26.3), AC-SHIP2-003.
//
// ## What INV-BPE-007 says
//
//   description: Byte coverage: every byte 0x00..=0xFF is
//                representable via byte-level fallback. No input
//                produces the <|unk|> token on valid UTF-8 — <|unk|>
//                is a sentinel that MUST NOT appear in tokenized
//                output of well-formed input.
//   falsifier:   Tokenize a corpus spanning every UTF-8 codepoint
//                class (ASCII, Latin-1, CJK, emoji, mathematical
//                symbols). Assert no token ID equals UNK.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a scan that produces
// (`docs_scanned`, `unk_token_count`, `expected_codepoint_classes_covered`),
// Pass iff:
//
//   docs_scanned > 0 AND
//   expected_codepoint_classes_covered >= AC_BPE_INV_007_REQUIRED_CLASSES (5) AND
//   unk_token_count == 0 AND
//   unk_token_count <= docs_scanned * <reasonable token cap>
//
// 5 codepoint classes match the contract: ASCII, Latin-1, CJK,
// emoji, mathematical symbols. Below 5, the scan didn't span the
// contract's mandated coverage. Zero-tolerance on UNK appearances
// — the contract calls UNK "a sentinel that MUST NOT appear", and
// even one occurrence on valid UTF-8 indicates broken byte-level
// fallback (BPE merge table missing a byte ID, or a codepoint
// class the tokenizer can't represent).

/// Required minimum number of UTF-8 codepoint classes the scan must
/// cover.
///
/// Per contract `INV-BPE-007`: "ASCII, Latin-1, CJK, emoji,
/// mathematical symbols" = 5 classes. A scan that covers fewer
/// classes lacks evidence that byte-level fallback works for the
/// uncovered ranges. Drift to 1-2 classes would let CJK or emoji
/// regressions slip; drift to 10 over-tightens unnecessarily.
pub const AC_BPE_INV_007_REQUIRED_CLASSES: u64 = 5;

/// Binary verdict for `INV-BPE-007`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv007Verdict {
    /// Scan visited at least one document, covered ≥5 codepoint
    /// classes, AND zero UNK tokens were observed.
    Pass,
    /// One or more of:
    /// - `docs_scanned == 0` (caller error — vacuous Pass refused).
    /// - `expected_codepoint_classes_covered < 5` (insufficient
    ///   coverage of the contract-mandated classes).
    /// - `unk_token_count > 0` (one UNK is enough — broken
    ///   byte-level fallback).
    Fail,
}

/// Pure verdict function for `INV-BPE-007`.
///
/// Inputs:
/// - `docs_scanned`: number of documents the byte-coverage scan
///   evaluated.
/// - `expected_codepoint_classes_covered`: number of distinct
///   UTF-8 codepoint classes (ASCII, Latin-1, CJK, emoji, math, …)
///   the scan corpus actually spans.
/// - `unk_token_count`: count of UNK token IDs observed in the
///   scan output.
///
/// Pass iff:
/// 1. `docs_scanned > 0`,
/// 2. `expected_codepoint_classes_covered >= 5`,
/// 3. `unk_token_count == 0`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 100 docs spanning all 5 contract classes, zero UNKs — `Pass`:
/// ```
/// use aprender::format::bpe_inv_007::{
///     verdict_from_unk_count_scan, BpeInv007Verdict,
/// };
/// let v = verdict_from_unk_count_scan(100, 5, 0);
/// assert_eq!(v, BpeInv007Verdict::Pass);
/// ```
///
/// One UNK in 100 docs — `Fail` (one is enough):
/// ```
/// use aprender::format::bpe_inv_007::{
///     verdict_from_unk_count_scan, BpeInv007Verdict,
/// };
/// let v = verdict_from_unk_count_scan(100, 5, 1);
/// assert_eq!(v, BpeInv007Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_unk_count_scan(
    docs_scanned: u64,
    expected_codepoint_classes_covered: u64,
    unk_token_count: u64,
) -> BpeInv007Verdict {
    if docs_scanned == 0 {
        return BpeInv007Verdict::Fail;
    }
    if expected_codepoint_classes_covered < AC_BPE_INV_007_REQUIRED_CLASSES {
        return BpeInv007Verdict::Fail;
    }
    if unk_token_count == 0 {
        BpeInv007Verdict::Pass
    } else {
        BpeInv007Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 5 required codepoint classes.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_classes_is_five() {
        assert_eq!(AC_BPE_INV_007_REQUIRED_CLASSES, 5);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clean tokenizer, full class coverage.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_minimum_5_classes_zero_unk() {
        // Exactly the contract minimum: ASCII + Latin-1 + CJK +
        // emoji + math = 5.
        let v = verdict_from_unk_count_scan(100, 5, 0);
        assert_eq!(v, BpeInv007Verdict::Pass);
    }

    #[test]
    fn pass_above_minimum_classes_zero_unk() {
        // 8 classes (e.g., +Cyrillic +Arabic +Devanagari).
        let v = verdict_from_unk_count_scan(100, 8, 0);
        assert_eq!(v, BpeInv007Verdict::Pass);
    }

    #[test]
    fn pass_single_doc_full_coverage() {
        // One doc that happens to span all 5 classes.
        let v = verdict_from_unk_count_scan(1, 5, 0);
        assert_eq!(v, BpeInv007Verdict::Pass);
    }

    #[test]
    fn pass_realistic_csn_python_size() {
        // CSN-Python codebase: ~455k docs, mostly ASCII but with
        // CJK comments, emoji in docstrings, math symbols — 5+.
        let v = verdict_from_unk_count_scan(455_000, 5, 0);
        assert_eq!(v, BpeInv007Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — UNK appearances (zero-tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_unk_in_100_docs() {
        let v = verdict_from_unk_count_scan(100, 5, 1);
        assert_eq!(
            v,
            BpeInv007Verdict::Fail,
            "one UNK on valid UTF-8 must Fail (broken byte fallback)"
        );
    }

    #[test]
    fn fail_handful_of_unks() {
        let v = verdict_from_unk_count_scan(100, 5, 7);
        assert_eq!(v, BpeInv007Verdict::Fail);
    }

    #[test]
    fn fail_one_unk_in_million_docs() {
        // Even at huge scale, one UNK fails. UNK is a sentinel — by
        // contract, it is never valid output for valid UTF-8.
        let v = verdict_from_unk_count_scan(1_000_000, 5, 1);
        assert_eq!(v, BpeInv007Verdict::Fail);
    }

    #[test]
    fn fail_many_unks() {
        let v = verdict_from_unk_count_scan(100, 5, 1_000);
        assert_eq!(v, BpeInv007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — insufficient class coverage.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_classes_covered() {
        let v = verdict_from_unk_count_scan(100, 0, 0);
        assert_eq!(v, BpeInv007Verdict::Fail);
    }

    #[test]
    fn fail_one_class_covered() {
        // Pure-ASCII corpus: only 1 class. Cannot validate
        // byte-level fallback for non-ASCII.
        let v = verdict_from_unk_count_scan(100, 1, 0);
        assert_eq!(v, BpeInv007Verdict::Fail);
    }

    #[test]
    fn fail_just_below_minimum_classes() {
        // 4 classes < contract floor of 5.
        let v = verdict_from_unk_count_scan(100, 4, 0);
        assert_eq!(
            v,
            BpeInv007Verdict::Fail,
            "4 classes lacks contract-mandated coverage"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — caller / counter errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_docs_scanned() {
        let v = verdict_from_unk_count_scan(0, 5, 0);
        assert_eq!(
            v,
            BpeInv007Verdict::Fail,
            "zero scanned docs must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_docs_with_classes_and_unks() {
        let v = verdict_from_unk_count_scan(0, 5, 7);
        assert_eq!(v, BpeInv007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep on class count.
    // -------------------------------------------------------------------------
    #[test]
    fn class_count_sweep_at_zero_unks() {
        let probes: Vec<(u64, BpeInv007Verdict)> = vec![
            (0, BpeInv007Verdict::Fail),
            (1, BpeInv007Verdict::Fail),
            (3, BpeInv007Verdict::Fail),
            (4, BpeInv007Verdict::Fail),
            (5, BpeInv007Verdict::Pass), // contract floor inclusive
            (6, BpeInv007Verdict::Pass),
            (10, BpeInv007Verdict::Pass),
            (100, BpeInv007Verdict::Pass),
        ];
        for (classes, expected) in probes {
            let v = verdict_from_unk_count_scan(100, classes, 0);
            assert_eq!(
                v, expected,
                "classes={classes} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — zero-tolerance UNK property at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_unk_count_is_exactly_zero_at_full_coverage() {
        for docs in [1_u64, 100, 10_000, 1_000_000] {
            let v_pass = verdict_from_unk_count_scan(docs, 5, 0);
            assert_eq!(v_pass, BpeInv007Verdict::Pass, "docs={docs}");

            let v_fail = verdict_from_unk_count_scan(docs, 5, 1);
            assert_eq!(
                v_fail,
                BpeInv007Verdict::Fail,
                "docs={docs} with one UNK"
            );
        }
    }
}
