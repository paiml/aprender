//! Additional tokenizer tests for edge cases and special token handling
//!
//! Part 2 of tokenizer tests - focuses on:
//! - BPE encoding edge cases (multibyte chars, byte tokens)
//! - GPT-2 char-to-byte mapping branches
//! - BPE merge behavior tests

use super::*;

// =============================================================================
// BPE Encode Edge Cases
// =============================================================================

#[test]
fn test_bpe_encode_multibyte_unicode() {
    // Test encoding text with multibyte Unicode characters
    let vocab = vec![
        "<unk>".to_string(),
        "<0xE4>".to_string(), // UTF-8 first byte
        "<0xB8>".to_string(),
        "<0xAD>".to_string(),
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // U+4E2D = E4 B8 AD in UTF-8 (CJK char)
    // Since it's not in vocab, it should fall back to byte tokens
    let encoded = tokenizer.encode("\u{4E2D}");
    // Should produce byte tokens for E4, B8, AD
    assert_eq!(encoded, vec![1, 2, 3]);
}

#[test]
fn test_bpe_encode_multibyte_to_unk() {
    // Test when byte tokens are also not in vocab
    let vocab = vec!["<unk>".to_string()];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // Any non-ASCII char should fall back to <unk> for each byte
    let encoded = tokenizer.encode("\u{4E2D}");
    assert_eq!(encoded.len(), 3); // 3 bytes in UTF-8
    assert!(encoded.iter().all(|&id| id == 0)); // All <unk>
}

#[test]
fn test_bpe_encode_emoji_fallback() {
    // Emoji are 4 bytes in UTF-8
    let vocab = vec!["<unk>".to_string()];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode("\u{1F600}"); // grinning face
    assert_eq!(encoded.len(), 4); // 4 bytes
    assert!(encoded.iter().all(|&id| id == 0)); // All <unk>
}

#[test]
fn test_bpe_encode_long_token_match() {
    // Test that longest match is found (up to 32 chars)
    let vocab = vec![
        "<unk>".to_string(),
        "a".to_string(),
        "ab".to_string(),
        "abc".to_string(),
        "abcd".to_string(),
        "abcde".to_string(),
        "abcdefghij".to_string(), // 10 chars
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // Should match longest token "abcdefghij"
    let encoded = tokenizer.encode("abcdefghij");
    assert_eq!(encoded, vec![6]);

    // "abcdefghijk" = "abcdefghij" + "k" (k is <unk>)
    let encoded2 = tokenizer.encode("abcdefghijk");
    assert_eq!(encoded2, vec![6, 0]);
}

#[test]
fn test_bpe_encode_newline_and_carriage_return() {
    // Test newline and carriage return encoding
    let vocab = vec![
        "<unk>".to_string(),
        "\u{010A}".to_string(), // Ċ = newline
        "\u{1E02}".to_string(), // Ḃ = carriage return
        "a".to_string(),
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode("a\na");
    assert_eq!(encoded, vec![3, 1, 3]); // a, Ċ, a

    let encoded2 = tokenizer.encode("a\ra");
    assert_eq!(encoded2, vec![3, 2, 3]); // a, Ḃ, a
}

#[test]
fn test_bpe_encode_mixed_special_whitespace() {
    // Test mixed spaces, newlines, and carriage returns
    let vocab = vec![
        "<unk>".to_string(),
        "\u{0120}".to_string(), // Ġ = space
        "\u{010A}".to_string(), // Ċ = newline
        "\u{1E02}".to_string(), // Ḃ = carriage return
        "x".to_string(),
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // "x x\n\rx" -> x, Ġ, x, Ċ, Ḃ, x
    let encoded = tokenizer.encode("x x\n\rx");
    assert_eq!(encoded, vec![4, 1, 4, 2, 3, 4]);
}

// =============================================================================
// BPE Decode Edge Cases
// =============================================================================

#[test]
fn test_bpe_decode_all_special_tokens() {
    // Test that all special token types are skipped
    let vocab = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<pad>".to_string(),
        "<|user|>".to_string(),
        "<|assistant|>".to_string(),
        "hello".to_string(),
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // All special tokens should be skipped, only "hello" remains
    let decoded = tokenizer.decode(&[0, 1, 2, 3, 4, 5, 6]).expect("test");
    assert_eq!(decoded, "hello");
}

#[test]
fn test_bpe_decode_byte_token_invalid_length() {
    // Byte tokens must be exactly 6 chars: <0xXX>
    let vocab = vec![
        "<unk>".to_string(),
        "<0xE6E>".to_string(), // 7 chars - invalid
        "<0xE>".to_string(),   // 5 chars - invalid
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // These should be treated as regular tokens, not byte tokens
    let decoded = tokenizer.decode(&[1, 2]).expect("test");
    assert!(decoded.contains("<0xE6E>"));
    assert!(decoded.contains("<0xE>"));
}

#[test]
fn test_bpe_decode_utf8_invalid_sequence() {
    // Test decoding byte tokens that form invalid UTF-8
    let vocab = vec![
        "<unk>".to_string(),
        "<0xFF>".to_string(), // Invalid UTF-8 byte
        "<0xFE>".to_string(), // Invalid UTF-8 byte
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // Should use lossy UTF-8 decoding (replacement chars)
    let decoded = tokenizer.decode(&[1, 2]).expect("test");
    assert!(decoded.contains('\u{FFFD}')); // Replacement character
}

// =============================================================================
// GPT-2 Char to Byte Mapping Edge Cases
// =============================================================================

#[test]
fn test_bpe_decode_gpt2_control_chars() {
    // GPT-2 maps bytes 0-32 via U+0100 offset
    // U+0100 = byte 0, U+0101 = byte 1, ..., U+0120 = byte 32
    let vocab = vec![
        "<unk>".to_string(),
        "\u{0100}".to_string(), // byte 0 (NUL)
        "\u{0109}".to_string(), // byte 9 (TAB)
        "\u{010D}".to_string(), // byte 13 (CR)
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let decoded = tokenizer.decode(&[1, 2, 3]).expect("test");
    // Control chars may be present or replaced
    assert!(!decoded.is_empty());
}

#[test]
fn test_bpe_decode_gpt2_del_and_extended() {
    // GPT-2 maps bytes 127-160 via U+0100 offset
    // U+017F = byte 127 (DEL), U+0180 = byte 128, etc.
    let vocab = vec![
        "<unk>".to_string(),
        "\u{017F}".to_string(), // byte 127 (DEL)
        "\u{0180}".to_string(), // byte 128
        "\u{01A0}".to_string(), // byte 160
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let decoded = tokenizer.decode(&[1, 2, 3]).expect("test");
    assert!(!decoded.is_empty());
}

#[test]
fn test_bpe_decode_gpt2_unmapped_range() {
    // Characters in U+0100-U+01FF range that don't map to special bytes
    // should return None from gpt2_char_to_byte
    let vocab = vec![
        "<unk>".to_string(),
        "\u{0150}".to_string(), // byte 80 - not in 0-32, 127-160, or 173
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // Should fall through to regular UTF-8 encoding
    let decoded = tokenizer.decode(&[1]).expect("test");
    assert!(!decoded.is_empty());
}

#[test]
fn test_bpe_decode_high_unicode_non_gpt2() {
    // Unicode chars outside the GPT-2 remapping range (>= U+0200)
    let vocab = vec![
        "<unk>".to_string(),
        "\u{0200}".to_string(), // Just outside GPT-2 range
        "\u{1000}".to_string(), // Myanmar character
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // Should be treated as regular UTF-8 characters
    let decoded = tokenizer.decode(&[1, 2]).expect("test");
    assert!(decoded.contains('\u{0200}'));
    assert!(decoded.contains('\u{1000}'));
}

// =============================================================================
// BPE Merge Behavior Tests (via public encode API)
// =============================================================================

#[test]
fn test_bpe_merge_no_applicable_pairs() {
    // Test when merge rules exist but don't apply to input
    let vocab = vec![
        "<unk>".to_string(),
        "x".to_string(),
        "y".to_string(),
        "z".to_string(),
        "ab".to_string(), // Merge result exists but won't be used
    ];
    let merges = vec![("a".to_string(), "b".to_string())];
    let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

    // "xyz" has no "a" or "b", so merge rule doesn't apply
    let encoded = tokenizer.encode("xyz");
    assert_eq!(encoded, vec![1, 2, 3]); // x, y, z
}

#[test]
fn test_bpe_merge_partial_sequence() {
    // Test when only part of input can be merged
    let vocab = vec![
        "<unk>".to_string(),
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "ab".to_string(),
    ];
    let merges = vec![("a".to_string(), "b".to_string())];
    let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

    // "abc" -> "ab" + "c"
    let encoded = tokenizer.encode("abc");
    assert_eq!(encoded, vec![4, 3]); // ab, c
}

#[test]
fn test_bpe_merge_multiple_pairs_in_sequence() {
    // Test multiple merge pairs in sequence
    let vocab = vec![
        "<unk>".to_string(),
        "a".to_string(),
        "b".to_string(),
        "ab".to_string(),
    ];
    let merges = vec![("a".to_string(), "b".to_string())];
    let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

    // "abab" -> "ab" + "ab"
    let encoded = tokenizer.encode("abab");
    assert_eq!(encoded, vec![3, 3]); // ab, ab
}

#[test]
fn test_bpe_merge_trailing_unmerged() {
    // Test when last token can't be merged
    let vocab = vec![
        "<unk>".to_string(),
        "a".to_string(),
        "b".to_string(),
        "ab".to_string(),
    ];
    let merges = vec![("a".to_string(), "b".to_string())];
    let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

    // "aba" -> "ab" + "a"
    let encoded = tokenizer.encode("aba");
    assert_eq!(encoded, vec![3, 1]); // ab, a
}

#[test]
fn test_bpe_merge_leading_unmerged() {
    // Test when first token can't be merged
    let vocab = vec![
        "<unk>".to_string(),
        "x".to_string(),
        "a".to_string(),
        "b".to_string(),
        "ab".to_string(),
    ];
    let merges = vec![("a".to_string(), "b".to_string())];
    let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

    // "xab" -> "x" + "ab"
    let encoded = tokenizer.encode("xab");
    assert_eq!(encoded, vec![1, 4]); // x, ab
}

// =============================================================================
// Additional BPE Coverage Tests
// =============================================================================

#[test]
fn test_bpe_encode_single_char_no_merges() {
    // Single character input should work
    let vocab = vec!["<unk>".to_string(), "a".to_string()];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode("a");
    assert_eq!(encoded, vec![1]);
}

#[test]
fn test_bpe_decode_printable_ascii() {
    // Test decode of printable ASCII (codes 33-126)
    let vocab = vec![
        "<unk>".to_string(),
        "!".to_string(), // ASCII 33
        "~".to_string(), // ASCII 126
        "A".to_string(), // ASCII 65
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let decoded = tokenizer.decode(&[1, 2, 3]).expect("test");
    assert_eq!(decoded, "!~A");
}

#[test]
fn test_bpe_gpt2_soft_hyphen() {
    // GPT-2 encodes soft hyphen (byte 173) as U+01AD
    let vocab = vec!["<unk>".to_string(), "\u{01AD}".to_string()];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let decoded = tokenizer.decode(&[1]).expect("test");
    // Should decode to byte 173 (soft hyphen) or similar
    assert!(!decoded.is_empty());
}
