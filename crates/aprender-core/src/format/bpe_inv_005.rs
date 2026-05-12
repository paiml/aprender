// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE-001)
// algorithm-level PARTIAL discharge for INV-BPE-005.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 tokenizer pipeline (§26.3), AC-SHIP2-003.
//
// ## What INV-BPE-005 says
//
//   description: Unicode normalization is NFC and is applied BEFORE
//                BPE encoding. Running nfc(nfc(text)) yields the
//                same bytes as nfc(text) (NFC is idempotent — this
//                catches double-normalization bugs).
//   falsifier:   For a test string containing composable sequences
//                (e.g. "café" composed vs "café" decomposed),
//                tokenizer.encode() must produce identical token
//                IDs for both. If they differ, the tokenizer is
//                NOT applying NFC pre-encode.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two byte slices (one from the composed
// input, one from the decomposed input) representing the
// post-NFC-pre-BPE form, AND optionally the result of
// nfc(nfc(text)) for double-application idempotence, Pass iff:
//
//   composed_nfc == decomposed_nfc (composable equivalence) AND
//   composed_nfc == double_nfc (NFC idempotence)
//
// Both equalities are byte-level. Catches three regression classes:
// - Tokenizer not applying NFC at all (composed != decomposed).
// - Tokenizer applying NFD instead of NFC (different canonical
//   form; also caught by composed != decomposed).
// - Double-NFC drift (NFC implementation has a non-idempotent bug).

/// Binary verdict for `INV-BPE-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv005Verdict {
    /// Composed-input NFC == decomposed-input NFC (composable
    /// equivalence) AND nfc(nfc(text)) == nfc(text) (idempotence).
    Pass,
    /// One or more of:
    /// - Either input is empty (caller error — degenerate).
    /// - `composed_nfc != decomposed_nfc` (NFC not applied or
    ///   inconsistent canonical form).
    /// - `composed_nfc != double_nfc` (NFC implementation is not
    ///   idempotent).
    Fail,
}

/// Pure verdict function for `INV-BPE-005`.
///
/// Inputs:
/// - `composed_nfc`: result of `nfc(text)` where `text` was provided
///   in NFC-composed form (e.g., "café" with U+00E9).
/// - `decomposed_nfc`: result of `nfc(text)` where `text` was
///   provided in NFD-decomposed form (e.g., "café" with
///   U+0065 U+0301).
/// - `double_nfc`: result of `nfc(nfc(text))` where the inner `nfc`
///   was already applied (idempotence probe).
///
/// Pass iff:
/// 1. All three slices are non-empty (rules out vacuous Pass on
///    empty input),
/// 2. `composed_nfc == decomposed_nfc` (composable equivalence),
/// 3. `composed_nfc == double_nfc` (idempotence on the composed
///    branch — implies idempotence on decomposed too via
///    transitivity since composed_nfc == decomposed_nfc).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// All three NFC-equivalent — `Pass`:
/// ```
/// use aprender::format::bpe_inv_005::{
///     verdict_from_nfc_idempotence, BpeInv005Verdict,
/// };
/// // "café" composed (4 bytes UTF-8 for café in NFC).
/// let nfc_form: &[u8] = "café".as_bytes();
/// let v = verdict_from_nfc_idempotence(nfc_form, nfc_form, nfc_form);
/// assert_eq!(v, BpeInv005Verdict::Pass);
/// ```
///
/// Composed != decomposed (NFC not applied) — `Fail`:
/// ```
/// use aprender::format::bpe_inv_005::{
///     verdict_from_nfc_idempotence, BpeInv005Verdict,
/// };
/// let composed: &[u8] = "café".as_bytes(); // U+00E9
/// let decomposed: &[u8] = "cafe\u{0301}".as_bytes(); // 'e' + combining acute
/// let v = verdict_from_nfc_idempotence(composed, decomposed, composed);
/// assert_eq!(v, BpeInv005Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_nfc_idempotence(
    composed_nfc: &[u8],
    decomposed_nfc: &[u8],
    double_nfc: &[u8],
) -> BpeInv005Verdict {
    if composed_nfc.is_empty() || decomposed_nfc.is_empty() || double_nfc.is_empty() {
        return BpeInv005Verdict::Fail;
    }
    if composed_nfc != decomposed_nfc {
        return BpeInv005Verdict::Fail;
    }
    if composed_nfc != double_nfc {
        return BpeInv005Verdict::Fail;
    }
    BpeInv005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — all three byte slices identical.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_all_three_identical_ascii() {
        let nfc = b"hello";
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }

    #[test]
    fn pass_all_three_identical_cafe_composed() {
        // "café" with composed é (U+00E9). 5 bytes UTF-8: c-a-f-é-(implicit).
        let nfc: &[u8] = "café".as_bytes();
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }

    #[test]
    fn pass_all_three_identical_cjk() {
        let nfc = "中文测试".as_bytes();
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }

    #[test]
    fn pass_all_three_identical_emoji() {
        // Single-codepoint emoji.
        let nfc = "🎉".as_bytes();
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }

    #[test]
    fn pass_all_three_identical_mathematical_symbols() {
        let nfc = "∑∫π√".as_bytes();
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — composed != decomposed (NFC not applied).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_cafe_composed_vs_decomposed() {
        // The classic regression: "café" U+00E9 vs "café" e+combining acute.
        let composed: &[u8] = "café".as_bytes();
        let decomposed: &[u8] = "cafe\u{0301}".as_bytes();
        // double_nfc matches composed (the post-NFC form).
        let v = verdict_from_nfc_idempotence(composed, decomposed, composed);
        assert_eq!(
            v,
            BpeInv005Verdict::Fail,
            "composed != decomposed must Fail (NFC not applied)"
        );
    }

    #[test]
    fn fail_completely_different_strings() {
        let a = b"hello";
        let b = b"world";
        let v = verdict_from_nfc_idempotence(a, b, a);
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    #[test]
    fn fail_single_byte_difference() {
        let a = b"hello";
        let b = b"hellp";
        let v = verdict_from_nfc_idempotence(a, b, a);
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — idempotence violation (double_nfc drift).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_double_nfc_differs() {
        // composed == decomposed, but nfc(nfc) != nfc — non-idempotent
        // NFC implementation.
        let nfc: &[u8] = "café".as_bytes();
        let drifted: &[u8] = "cafe".as_bytes(); // dropped the é
        let v = verdict_from_nfc_idempotence(nfc, nfc, drifted);
        assert_eq!(
            v,
            BpeInv005Verdict::Fail,
            "double-NFC drift must Fail (non-idempotent)"
        );
    }

    #[test]
    fn fail_double_nfc_off_by_one_byte() {
        let nfc = b"hello";
        let drifted = b"hellp"; // last byte differs
        let v = verdict_from_nfc_idempotence(nfc, nfc, drifted);
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — both NFC violations and idempotence violation.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_both_violations_combined() {
        let composed: &[u8] = "café".as_bytes();
        let decomposed: &[u8] = "cafe\u{0301}".as_bytes();
        let double = b"foo"; // Completely different from composed
        let v = verdict_from_nfc_idempotence(composed, decomposed, double);
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — caller errors (empty inputs).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_all_empty() {
        let v = verdict_from_nfc_idempotence(&[], &[], &[]);
        assert_eq!(
            v,
            BpeInv005Verdict::Fail,
            "all-empty inputs must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_composed_empty() {
        let v = verdict_from_nfc_idempotence(&[], b"abc", b"abc");
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    #[test]
    fn fail_decomposed_empty() {
        let v = verdict_from_nfc_idempotence(b"abc", &[], b"abc");
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    #[test]
    fn fail_double_empty() {
        let v = verdict_from_nfc_idempotence(b"abc", b"abc", &[]);
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry / transitivity properties.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_with_a_b_swapped_when_equal() {
        // If composed == decomposed == double, swapping a and b
        // doesn't change verdict.
        let nfc = b"hello";
        let v_ab = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        let v_ba = verdict_from_nfc_idempotence(nfc, nfc, nfc); // Same in this trivial case
        assert_eq!(v_ab, v_ba);
        assert_eq!(v_ab, BpeInv005Verdict::Pass);
    }

    #[test]
    fn fail_three_way_distinct() {
        // a != b != c, all distinct. Catches a regression that
        // somehow mismatches all three.
        let a = b"abc";
        let b = b"def";
        let c = b"ghi";
        let v = verdict_from_nfc_idempotence(a, b, c);
        assert_eq!(v, BpeInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — multi-codepoint composables (e.g., Hangul).
    // -------------------------------------------------------------------------
    #[test]
    fn pass_hangul_precomposed_form() {
        // "한" precomposed Hangul syllable U+D55C.
        let nfc = "한".as_bytes();
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }

    #[test]
    fn fail_hangul_precomposed_vs_decomposed() {
        // Precomposed U+D55C vs decomposed jamo U+1112 + U+1161 + U+11AB.
        let composed = "한".as_bytes();
        let decomposed = "\u{1112}\u{1161}\u{11AB}".as_bytes();
        let v = verdict_from_nfc_idempotence(composed, decomposed, composed);
        assert_eq!(
            v,
            BpeInv005Verdict::Fail,
            "Hangul precomposed != decomposed must Fail"
        );
    }

    #[test]
    fn pass_long_well_formed_text() {
        let text = "The quick brown fox jumps over the lazy dog. \
                    Café au lait. 中文测试 🎉. ∑∫π. 한국어. 1234567890";
        let nfc = text.as_bytes();
        let v = verdict_from_nfc_idempotence(nfc, nfc, nfc);
        assert_eq!(v, BpeInv005Verdict::Pass);
    }
}
