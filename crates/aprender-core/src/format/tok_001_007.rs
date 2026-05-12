// SHIP-TWO-001 — `tokenizer-loading-v1` algorithm-level PARTIAL
// discharge for FALSIFY-TOK-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/tokenizer-loading-v1.yaml`.

use std::collections::HashMap;

// ===========================================================================
// Canonical Qwen2.5-Coder-7B special-token IDs and vocab size
// ===========================================================================

pub const AC_TOK_002_ENDOFTEXT_ID: u32 = 151_643;
pub const AC_TOK_002_IM_START_ID: u32 = 151_644;
pub const AC_TOK_002_IM_END_ID: u32 = 151_645;
pub const AC_TOK_003_VOCAB_SIZE: u64 = 151_936;
pub const AC_TOK_005_MAX_EMPTY_TOKENS: usize = 2;

// ===========================================================================
// TOK-001 — Roundtrip ASCII: encode then decode recovers original
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_ascii_roundtrip(original: &str, decoded: &str) -> Tok001Verdict {
    if original.is_empty() { return Tok001Verdict::Fail; }
    if !original.is_ascii() { return Tok001Verdict::Fail; }
    if original == decoded { Tok001Verdict::Pass } else { Tok001Verdict::Fail }
}

// ===========================================================================
// TOK-002 — Special token IDs match config
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_special_token_ids(
    endoftext_id: u32,
    im_start_id: u32,
    im_end_id: u32,
) -> Tok002Verdict {
    if endoftext_id != AC_TOK_002_ENDOFTEXT_ID { return Tok002Verdict::Fail; }
    if im_start_id != AC_TOK_002_IM_START_ID { return Tok002Verdict::Fail; }
    if im_end_id != AC_TOK_002_IM_END_ID { return Tok002Verdict::Fail; }
    Tok002Verdict::Pass
}

// ===========================================================================
// TOK-003 — Vocab size matches config
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_vocab_size(observed: u64) -> Tok003Verdict {
    if observed == AC_TOK_003_VOCAB_SIZE { Tok003Verdict::Pass } else { Tok003Verdict::Fail }
}

// ===========================================================================
// TOK-004 — Encoding determinism: same input → same output every time
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_encoding_determinism(repeats: &[Vec<u32>]) -> Tok004Verdict {
    if repeats.len() < 2 { return Tok004Verdict::Fail; }
    for w in repeats.windows(2) {
        if w[0] != w[1] { return Tok004Verdict::Fail; }
    }
    Tok004Verdict::Pass
}

// ===========================================================================
// TOK-005 — Empty input: no panic, |ids| <= 2
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_empty_input(empty_encoding_len: usize) -> Tok005Verdict {
    if empty_encoding_len <= AC_TOK_005_MAX_EMPTY_TOKENS { Tok005Verdict::Pass } else { Tok005Verdict::Fail }
}

// ===========================================================================
// TOK-006 — Byte coverage: all 256 bytes have encoder mappings
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_byte_coverage<S: std::hash::BuildHasher>(encoder: &HashMap<u8, u32, S>) -> Tok006Verdict {
    if encoder.len() < 256 { return Tok006Verdict::Fail; }
    for byte in 0_u8..=255 {
        if !encoder.contains_key(&byte) { return Tok006Verdict::Fail; }
    }
    Tok006Verdict::Pass
}

// ===========================================================================
// TOK-007 — Roundtrip UTF-8: works for non-ASCII multi-byte
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_utf8_roundtrip(original: &str, decoded: &str) -> Tok007Verdict {
    if original.is_empty() { return Tok007Verdict::Fail; }
    if original == decoded { Tok007Verdict::Pass } else { Tok007Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TOK-001
    #[test] fn tok001_pass_canonical() {
        let s = "echo $HOME && mkdir -p /tmp/test";
        assert_eq!(verdict_from_ascii_roundtrip(s, s), Tok001Verdict::Pass);
    }
    #[test] fn tok001_fail_drift() {
        assert_eq!(verdict_from_ascii_roundtrip("hello", "hellO"), Tok001Verdict::Fail);
    }
    #[test] fn tok001_fail_empty() {
        assert_eq!(verdict_from_ascii_roundtrip("", ""), Tok001Verdict::Fail);
    }
    #[test] fn tok001_fail_non_ascii() {
        assert_eq!(verdict_from_ascii_roundtrip("café", "café"), Tok001Verdict::Fail);
    }

    // TOK-002
    #[test] fn tok002_pass_canonical() {
        assert_eq!(
            verdict_from_special_token_ids(151_643, 151_644, 151_645),
            Tok002Verdict::Pass
        );
    }
    #[test] fn tok002_fail_endoftext_drift() {
        assert_eq!(
            verdict_from_special_token_ids(151_642, 151_644, 151_645),
            Tok002Verdict::Fail
        );
    }
    #[test] fn tok002_fail_swapped_im() {
        assert_eq!(
            verdict_from_special_token_ids(151_643, 151_645, 151_644),
            Tok002Verdict::Fail
        );
    }

    // TOK-003
    #[test] fn tok003_pass() { assert_eq!(verdict_from_vocab_size(151_936), Tok003Verdict::Pass); }
    #[test] fn tok003_fail_truncated() { assert_eq!(verdict_from_vocab_size(151_900), Tok003Verdict::Fail); }
    #[test] fn tok003_fail_inflated() { assert_eq!(verdict_from_vocab_size(152_000), Tok003Verdict::Fail); }

    // TOK-004
    #[test] fn tok004_pass_identical() {
        let r = vec![vec![1_u32, 2, 3]; 5];
        assert_eq!(verdict_from_encoding_determinism(&r), Tok004Verdict::Pass);
    }
    #[test] fn tok004_fail_drift() {
        let r = vec![vec![1_u32, 2, 3], vec![1, 2, 4]];
        assert_eq!(verdict_from_encoding_determinism(&r), Tok004Verdict::Fail);
    }
    #[test] fn tok004_fail_too_few() {
        let r = vec![vec![1_u32, 2, 3]];
        assert_eq!(verdict_from_encoding_determinism(&r), Tok004Verdict::Fail);
    }

    // TOK-005
    #[test] fn tok005_pass_zero() { assert_eq!(verdict_from_empty_input(0), Tok005Verdict::Pass); }
    #[test] fn tok005_pass_bos() { assert_eq!(verdict_from_empty_input(1), Tok005Verdict::Pass); }
    #[test] fn tok005_pass_bos_eos() { assert_eq!(verdict_from_empty_input(2), Tok005Verdict::Pass); }
    #[test] fn tok005_fail_too_many() { assert_eq!(verdict_from_empty_input(3), Tok005Verdict::Fail); }

    // TOK-006
    #[test] fn tok006_pass_full_coverage() {
        let mut enc = HashMap::new();
        for b in 0_u8..=255 { enc.insert(b, b as u32); }
        assert_eq!(verdict_from_byte_coverage(&enc), Tok006Verdict::Pass);
    }
    #[test] fn tok006_fail_missing_byte() {
        let mut enc = HashMap::new();
        for b in 0_u8..=254 { enc.insert(b, b as u32); }
        assert_eq!(verdict_from_byte_coverage(&enc), Tok006Verdict::Fail);
    }
    #[test] fn tok006_fail_empty() {
        let enc = HashMap::new();
        assert_eq!(verdict_from_byte_coverage(&enc), Tok006Verdict::Fail);
    }

    // TOK-007
    #[test] fn tok007_pass_utf8() {
        let s = "echo \"héllo wörld\" 🚀";
        assert_eq!(verdict_from_utf8_roundtrip(s, s), Tok007Verdict::Pass);
    }
    #[test] fn tok007_fail_drift() {
        assert_eq!(verdict_from_utf8_roundtrip("café", "cafe"), Tok007Verdict::Fail);
    }
    #[test] fn tok007_fail_empty() {
        assert_eq!(verdict_from_utf8_roundtrip("", ""), Tok007Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_constants() {
        assert_eq!(AC_TOK_002_ENDOFTEXT_ID, 151_643);
        assert_eq!(AC_TOK_002_IM_START_ID, 151_644);
        assert_eq!(AC_TOK_002_IM_END_ID, 151_645);
        assert_eq!(AC_TOK_003_VOCAB_SIZE, 151_936);
        assert_eq!(AC_TOK_005_MAX_EMPTY_TOKENS, 2);
    }
}
