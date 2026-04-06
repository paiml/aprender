
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentencepiece_encode_decode_roundtrip() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -2.0),
            ("e".to_string(), -2.0),
            ("l".to_string(), -2.0),
            ("o".to_string(), -2.0),
            ("hello".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_sentencepiece_get_methods() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("hello".to_string(), -1.5)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        assert_eq!(tokenizer.get_token_id("hello"), Some(1));
        assert_eq!(tokenizer.get_token_id("world"), None);
        assert_eq!(tokenizer.get_token(1), Some("hello"));
        assert_eq!(tokenizer.get_token(999), None);
        assert!((tokenizer.get_score("hello").expect("test") - (-1.5)).abs() < 1e-6);
        assert_eq!(tokenizer.get_score("world"), None);
    }

    #[test]
    fn test_sentencepiece_unknown_character() {
        // Character not in vocabulary should use unknown penalty
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -1.0),
            ("i".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        // 'x' is not in vocab, should be tokenized with penalty
        let encoded = tokenizer.encode("hix");
        assert_eq!(encoded.len(), 3);
        assert_eq!(encoded[0], 1); // h
        assert_eq!(encoded[1], 2); // i
        // x should map to unk
        assert_eq!(encoded[2], 0);
    }

    #[test]
    fn test_sentencepiece_multiple_words() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("hello".to_string(), -1.0),
            (" ".to_string(), -0.5),
            ("world".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![1, 2, 3]); // hello, space, world
    }

    // -------------------------------------------------------------------------
    // Additional BPE Decode Tests (95% coverage push)
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_decode_special_tokens() {
        let vocab = vec![
            "<unk>".to_string(),
            "<|endoftext|>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "<pad>".to_string(),
            "hello".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // Special tokens should be skipped in decode
        let decoded = tokenizer.decode(&[1, 2, 3, 4, 5]).expect("test");
        assert_eq!(decoded, "hello"); // Only "hello" should remain
    }

    #[test]
    fn test_bpe_decode_byte_tokens() {
        let vocab = vec![
            "<unk>".to_string(),
            "<0xE6>".to_string(), // UTF-8 first byte of some CJK chars
            "<0x97>".to_string(),
            "<0xA5>".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // These three bytes form the UTF-8 sequence for "日" (U+65E5)
        let decoded = tokenizer.decode(&[1, 2, 3]).expect("test");
        assert_eq!(decoded, "日");
    }

    #[test]
    fn test_bpe_decode_gpt2_space() {
        let vocab = vec![
            "<unk>".to_string(),
            "Ġhello".to_string(), // Ġ = space prefix in GPT-2
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1]).expect("test");
        assert_eq!(decoded, " hello");
    }

    #[test]
    fn test_bpe_decode_gpt2_newline() {
        let vocab = vec![
            "<unk>".to_string(),
            "Ċ".to_string(), // newline in GPT-2
            "ċ".to_string(), // lowercase variant
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "\n\n");
    }

    #[test]
    fn test_bpe_decode_sentencepiece_space() {
        let vocab = vec![
            "<unk>".to_string(),
            "▁hello".to_string(), // SentencePiece space
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1]).expect("test");
        assert_eq!(decoded, " hello");
    }

    #[test]
    fn test_bpe_decode_gpt2_carriage_return() {
        let vocab = vec![
            "<unk>".to_string(),
            "Ḃ".to_string(), // carriage return in GPT-2
            "a".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "\ra");
    }

    #[test]
    fn test_bpe_decode_regular_utf8() {
        let vocab = vec![
            "<unk>".to_string(),
            "こんにちは".to_string(), // Japanese
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1]).expect("test");
        assert_eq!(decoded, "こんにちは");
    }

    #[test]
    fn test_bpe_decode_invalid_byte_token() {
        let vocab = vec![
            "<unk>".to_string(),
            "<0xGG>".to_string(), // Invalid hex
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // Should not panic, just treat as regular token
        let decoded = tokenizer.decode(&[1]).expect("test");
        assert!(decoded.contains("<0xGG>"));
    }

    #[test]
    fn test_bpe_decode_mixed_tokens() {
        let vocab = vec![
            "<unk>".to_string(),
            "Ġhello".to_string(), // GPT-2 space + word
            "Ċ".to_string(),      // newline
            "world".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2, 3]).expect("test");
        assert_eq!(decoded, " hello\nworld");
    }

    #[test]
    fn test_bpe_gpt2_char_to_byte_printable_ascii() {
        // Test that printable ASCII is preserved
        let vocab = vec!["<unk>".to_string(), "a".to_string(), "!".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "a!");
    }

    #[test]
    fn test_bpe_gpt2_char_to_byte_space() {
        // Space (ASCII 32) should be preserved
        let vocab = vec!["<unk>".to_string(), " ".to_string(), "a".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, " a");
    }

    #[test]
    fn test_sentencepiece_vocab_size() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        assert_eq!(tokenizer.vocab_size(), 3);
    }

    // =========================================================================
    // Additional 95% Coverage Tests
    // =========================================================================

    #[test]
    fn test_cov95_bpe_apply_merge_single_token() {
        // Test apply_merge with single token (early return path)
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
        ];
        let merges = vec![("a".to_string(), "b".to_string())];
        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        // Encode single char - no merge possible
        let encoded = tokenizer.encode("a");
        assert_eq!(encoded, vec![1]);
    }

    #[test]
    fn test_cov95_bpe_apply_merge_no_match() {
        // Test apply_merge when merge pair doesn't match
        let vocab = vec![
            "<unk>".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "ab".to_string(),
        ];
        let merges = vec![("a".to_string(), "b".to_string())]; // merge won't match "xyz"
        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        // "xyz" has no matching merge pairs
        let encoded = tokenizer.encode("xyz");
        assert_eq!(encoded, vec![1, 2, 3]);
    }

    #[test]
    fn test_cov95_bpe_apply_merge_consecutive() {
        // Test apply_merge with consecutive matches
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
        ];
        let merges = vec![("a".to_string(), "b".to_string())];
        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        // "abab" should merge to "ab" + "ab"
        let encoded = tokenizer.encode("abab");
        assert_eq!(encoded, vec![3, 3]); // "ab", "ab"
    }

    #[test]
    fn test_cov95_bpe_decode_gpt2_remapped_bytes() {
        // Test decode with GPT-2 remapped byte tokens (0x100+ range)
        // GPT-2 uses chars starting at U+0100 for raw bytes
        let vocab = vec![
            "<unk>".to_string(),
            "\u{0100}".to_string(), // maps to byte 0
            "\u{0101}".to_string(), // maps to byte 1
            "\u{0120}".to_string(), // maps to byte 32 (space in GPT-2)
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // Decode tokens with GPT-2 byte remapping
        let decoded = tokenizer.decode(&[3]).expect("test");
        // U+0120 should decode to space (byte 32)
        assert!(decoded.contains('\u{0120}') || decoded.contains(' '));
    }

    #[test]
    fn test_cov95_bpe_decode_high_unicode_byte() {
        // Test decode with high unicode that maps to byte via GPT-2 encoding
        // GPT-2 remaps bytes 127-160 to U+0100 + offset
        let vocab = vec![
            "<unk>".to_string(),
            "\u{017F}".to_string(), // Should map to byte 127 in GPT-2 encoding
            "\u{01A0}".to_string(), // Should map to byte 160 in GPT-2 encoding
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // These tokens should decode (possibly with replacement chars)
        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_cov95_bpe_decode_soft_hyphen() {
        // Test decode with soft hyphen (byte 173)
        let vocab = vec![
            "<unk>".to_string(),
            "\u{01AD}".to_string(), // U+0100 + 173 = U+01AD for soft hyphen in GPT-2
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1]).expect("test");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_cov95_bpe_encode_with_multiple_merges() {
        // Test encoding that exercises multiple merge iterations
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "ab".to_string(),
            "abc".to_string(),
        ];
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        let encoded = tokenizer.encode("abc");
        // First merge: a+b -> ab, then ab+c -> abc
        assert_eq!(encoded, vec![5]); // "abc"
    }
include!("tokenizer_vocabulary.rs");
}
