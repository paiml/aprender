//! Additional tokenizer tests for SentencePiece, Vocabulary, and roundtrip
//!
//! Part 3 of tokenizer tests - focuses on:
//! - SentencePiece Viterbi edge cases
//! - Vocabulary edge cases
//! - BPE byte token edge cases
//! - Decode roundtrip edge cases

use super::*;

// =============================================================================
// SentencePiece Viterbi Edge Cases
// =============================================================================

#[test]
fn test_sentencepiece_viterbi_all_unknown() {
    // When all characters are unknown, viterbi should still complete
    let vocab = vec![("<unk>".to_string(), 0.0)];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    // "xyz" - all unknown
    let encoded = tokenizer.encode("xyz");
    assert_eq!(encoded.len(), 3); // One token per character
    assert!(encoded.iter().all(|&id| id == 0)); // All <unk>
}

#[test]
fn test_sentencepiece_viterbi_partial_match() {
    // Some chars match, some don't
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("a".to_string(), -1.0),
        ("c".to_string(), -1.0),
    ];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    // "abc" - a and c match, b doesn't
    let encoded = tokenizer.encode("abc");
    assert_eq!(encoded.len(), 3);
    assert_eq!(encoded[0], 1); // a
    assert_eq!(encoded[1], 0); // b -> <unk>
    assert_eq!(encoded[2], 2); // c
}

#[test]
fn test_sentencepiece_viterbi_long_unknown_sequence() {
    // Long sequence of unknown characters
    let vocab = vec![("<unk>".to_string(), 0.0), ("x".to_string(), -1.0)];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    // "xaaaax" - middle chars unknown
    let encoded = tokenizer.encode("xaaaax");
    assert_eq!(encoded.len(), 6);
    assert_eq!(encoded[0], 1); // x
    assert_eq!(encoded[5], 1); // x
    assert!(encoded[1..5].iter().all(|&id| id == 0)); // aaaa -> <unk>
}

#[test]
fn test_sentencepiece_score_ordering() {
    // Test that scores correctly influence segmentation
    // "hello" has lower score than "hel" + "lo", so subwords should be chosen
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("hello".to_string(), -10.0), // Very low score
        ("hel".to_string(), -0.5),    // Higher score
        ("lo".to_string(), -0.5),     // Higher score
    ];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    let encoded = tokenizer.encode("hello");
    // "hel" + "lo" = -1.0 > "hello" = -10.0
    assert_eq!(encoded, vec![2, 3]); // hel, lo
}

#[test]
fn test_sentencepiece_unicode_chars() {
    // Test with Unicode characters
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("\u{3053}".to_string(), -1.0),
        ("\u{3093}".to_string(), -1.0),
        ("\u{3053}\u{3093}".to_string(), -0.5),
    ];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    let encoded = tokenizer.encode("\u{3053}\u{3093}");
    // Combined token has better score (-0.5) than separate chars (-2.0)
    assert_eq!(encoded, vec![3]); // kon
}

// =============================================================================
// Vocabulary Edge Cases
// =============================================================================

#[test]
fn test_vocabulary_large_id() {
    // Test vocabulary with many tokens (ensure u32 conversion works)
    let tokens: Vec<String> = (0..1000).map(|i| format!("token_{i}")).collect();
    let vocab = Vocabulary::from_tokens(tokens).expect("test");

    assert_eq!(vocab.size(), 1000);
    assert_eq!(vocab.get_id("token_999"), Some(999));
    assert_eq!(vocab.get_token(999), Some("token_999"));
}

#[test]
fn test_tokenizer_multiple_unknowns() {
    // Test encoding multiple consecutive unknown words
    let tokens = vec!["<unk>".to_string(), "a".to_string()];
    let vocab = Vocabulary::from_tokens(tokens).expect("test");
    let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

    let encoded = tokenizer.encode("x y z a");
    assert_eq!(encoded, vec![0, 0, 0, 1]); // x, y, z -> <unk>, a
}

#[test]
fn test_tokenizer_whitespace_only() {
    // Test encoding whitespace-only string
    let tokens = vec!["<unk>".to_string()];
    let vocab = Vocabulary::from_tokens(tokens).expect("test");
    let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

    let encoded = tokenizer.encode("   ");
    assert!(encoded.is_empty()); // split_whitespace returns empty for whitespace
}

// =============================================================================
// BPE Byte Token Edge Cases
// =============================================================================

#[test]
fn test_bpe_encode_byte_token_in_vocab() {
    // When character is not in vocab but its byte token is
    let vocab = vec![
        "<unk>".to_string(),
        "<0xC3>".to_string(), // First byte of U+00E9 (e with acute)
        "<0xA9>".to_string(), // Second byte of U+00E9
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // "e" with acute accent (e) = C3 A9 in UTF-8
    let encoded = tokenizer.encode("\u{00E9}");
    assert_eq!(encoded, vec![1, 2]); // <0xC3>, <0xA9>
}

#[test]
fn test_bpe_encode_mixed_known_unknown_bytes() {
    // Some byte tokens in vocab, some not
    let vocab = vec![
        "<unk>".to_string(),
        "<0xC3>".to_string(), // First byte of e
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // e = C3 A9, only C3 is in vocab
    let encoded = tokenizer.encode("\u{00E9}");
    assert_eq!(encoded, vec![1, 0]); // <0xC3>, <unk>
}

// =============================================================================
// Decode Roundtrip Edge Cases
// =============================================================================

#[test]
fn test_sentencepiece_roundtrip_with_unknown() {
    // Roundtrip with unknown characters
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("hello".to_string(), -1.0),
        ("x".to_string(), -1.0),
    ];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    let encoded = tokenizer.encode("helloYworld");
    // hello, Y (unknown), w, o, r, l, d (all unknown)
    let decoded = tokenizer.decode(&encoded).expect("test");
    // Y, w, o, r, l, d become <unk> on encode, then <unk> on decode
    assert!(decoded.contains("hello"));
    assert!(decoded.contains("<unk>"));
}

#[test]
fn test_bpe_roundtrip_with_spaces() {
    // Roundtrip with GPT-2 space encoding
    let vocab = vec![
        "<unk>".to_string(),
        "\u{0120}hello".to_string(), // Ġhello
        "\u{0120}world".to_string(), // Ġworld
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode(" hello world");
    let decoded = tokenizer.decode(&encoded).expect("test");
    assert_eq!(decoded, " hello world");
}

#[test]
fn test_bpe_decode_consecutive_special_chars() {
    // Test decoding consecutive special whitespace characters
    let vocab = vec![
        "<unk>".to_string(),
        "\u{0120}".to_string(),         // Ġ = space
        "\u{0120}\u{0120}".to_string(), // ĠĠ = two spaces
        "\u{010A}".to_string(),         // Ċ = newline
        "\u{010A}\u{010A}".to_string(), // ĊĊ = two newlines
    ];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let decoded = tokenizer.decode(&[2, 4]).expect("test");
    assert_eq!(decoded, "  \n\n"); // two spaces + two newlines
}

// =============================================================================
// Additional Edge Cases for Coverage
// =============================================================================

#[test]
fn test_bpe_encode_only_spaces() {
    // Test encoding text that becomes only Ġ tokens
    let vocab = vec!["<unk>".to_string(), "\u{0120}".to_string()]; // Ġ
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode("   ");
    assert_eq!(encoded, vec![1, 1, 1]); // Three Ġ tokens
}

#[test]
fn test_bpe_encode_only_newlines() {
    // Test encoding text that becomes only Ċ tokens
    let vocab = vec!["<unk>".to_string(), "\u{010A}".to_string()]; // Ċ
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode("\n\n\n");
    assert_eq!(encoded, vec![1, 1, 1]); // Three Ċ tokens
}

#[test]
fn test_sentencepiece_equal_scores() {
    // Test when multiple segmentations have equal scores
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("a".to_string(), -1.0),
        ("b".to_string(), -1.0),
        ("ab".to_string(), -2.0), // Same total score as a + b
    ];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    // When scores are equal, the algorithm should still produce valid output
    let encoded = tokenizer.encode("ab");
    // Could be [3] (ab) or [1, 2] (a, b) - both valid
    assert!(!encoded.is_empty());
    let decoded = tokenizer.decode(&encoded).expect("test");
    assert_eq!(decoded, "ab");
}

#[test]
fn test_sentencepiece_negative_infinity_unreachable() {
    // Test that unreachable positions don't cause issues
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("ab".to_string(), -1.0), // Only matches "ab", not individual chars
    ];
    let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

    // "a" alone can't be matched by any token, should use unknown penalty
    let encoded = tokenizer.encode("a");
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0], 0); // <unk>
}

#[test]
fn test_bpe_very_long_token() {
    // Test with a token longer than typical
    let long_token = "a".repeat(30);
    let vocab = vec!["<unk>".to_string(), long_token.clone()];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    let encoded = tokenizer.encode(&long_token);
    assert_eq!(encoded, vec![1]); // Should match the full long token

    // Longer than 32 chars (max prefix check) should still work
    let very_long = "a".repeat(35);
    let encoded2 = tokenizer.encode(&very_long);
    // Should match 30-char token + 5 unknown 'a's
    assert_eq!(encoded2.len(), 6); // long_token + 5 <unk>
}

#[test]
fn test_vocabulary_special_chars_in_tokens() {
    // Test vocabulary with special characters
    let tokens = vec![
        "<unk>".to_string(),
        "hello\tworld".to_string(), // Tab in token
        "foo\nbar".to_string(),     // Newline in token
        "a b c".to_string(),        // Spaces in token
    ];
    let vocab = Vocabulary::from_tokens(tokens).expect("test");

    assert_eq!(vocab.get_id("hello\tworld"), Some(1));
    assert_eq!(vocab.get_id("foo\nbar"), Some(2));
    assert_eq!(vocab.get_id("a b c"), Some(3));
}

#[test]
fn test_bpe_decode_unk_token_skipped() {
    // Test that <unk> is skipped in decode
    let vocab = vec!["<unk>".to_string(), "hello".to_string()];
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    // <unk> (0) should be skipped in decode
    let decoded = tokenizer.decode(&[0, 1, 0]).expect("test");
    assert_eq!(decoded, "hello"); // Only "hello", no <unk>
}
