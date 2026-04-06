//! Property-based tests for tokenizer implementations
//!
//! These tests use proptest to verify tokenizer properties like roundtrip encoding.

use proptest::prelude::*;
use realizar::tokenizer::{BPETokenizer, SentencePieceTokenizer, Tokenizer, Vocabulary};

/// Strategy for generating simple ASCII strings for testing
fn ascii_string_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range('a', 'z'), 1..50)
        .prop_map(|chars| chars.into_iter().collect())
}

/// Strategy for generating strings from a known vocabulary
fn vocab_string_strategy() -> impl Strategy<Value = String> {
    // Use simple tokens that we know are in our test vocabulary
    prop::collection::vec(prop::sample::select(vec!["a", "b", "c", "ab", "bc"]), 1..10)
        .prop_map(|tokens| tokens.join(""))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Tokenizer encode produces non-empty output for non-empty input
    #[test]
    fn test_tokenizer_encode_nonempty(text in ascii_string_strategy()) {
        let tokens = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
            "f".to_string(),
            "g".to_string(),
            "h".to_string(),
            "i".to_string(),
            "j".to_string(),
            "k".to_string(),
            "l".to_string(),
            "m".to_string(),
            "n".to_string(),
            "o".to_string(),
            "p".to_string(),
            "q".to_string(),
            "r".to_string(),
            "s".to_string(),
            "t".to_string(),
            "u".to_string(),
            "v".to_string(),
            "w".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode(&text);
        prop_assert!(!encoded.is_empty(), "Encoded output should not be empty for non-empty input");
    }

    /// Basic tokenizer produces valid token IDs
    #[test]
    fn test_tokenizer_valid_ids(text in ascii_string_strategy()) {
        let tokens = vec![
            "<unk>".to_string(),
            "a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(),
            "e".to_string(), "f".to_string(), "g".to_string(), "h".to_string(),
            "i".to_string(), "j".to_string(), "k".to_string(), "l".to_string(),
            "m".to_string(), "n".to_string(), "o".to_string(), "p".to_string(),
            "q".to_string(), "r".to_string(), "s".to_string(), "t".to_string(),
            "u".to_string(), "v".to_string(), "w".to_string(), "x".to_string(),
            "y".to_string(), "z".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens.clone()).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode(&text);

        // All token IDs should be valid
        for &id in &encoded {
            prop_assert!((id as usize) < tokens.len(), "Token ID {} out of range", id);
        }
    }

    /// BPE tokenizer encode produces valid token IDs
    #[test]
    fn test_bpe_encode_valid_ids(text in vocab_string_strategy()) {
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "ab".to_string(),
            "bc".to_string(),
        ];
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let tokenizer = BPETokenizer::new(vocab.clone(), merges, "<unk>").expect("test");

        let encoded = tokenizer.encode(&text);

        // All token IDs should be valid
        for &id in &encoded {
            prop_assert!((id as usize) < vocab.len(), "Token ID {} out of range", id);
        }
    }

    /// BPE tokenizer roundtrip preserves text
    #[test]
    fn test_bpe_roundtrip(text in vocab_string_strategy()) {
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "ab".to_string(),
            "bc".to_string(),
        ];
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        let encoded = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&encoded).expect("test");

        prop_assert_eq!(decoded, text);
    }

    /// SentencePiece tokenizer encode produces valid token IDs
    #[test]
    fn test_sentencepiece_encode_valid_ids(text in vocab_string_strategy()) {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("c".to_string(), -1.0),
            ("ab".to_string(), -0.5),
            ("bc".to_string(), -0.5),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab.clone(), "<unk>").expect("test");

        let encoded = tokenizer.encode(&text);

        // All token IDs should be valid
        for &id in &encoded {
            prop_assert!((id as usize) < vocab.len(), "Token ID {} out of range", id);
        }
    }

    /// SentencePiece tokenizer roundtrip preserves text
    #[test]
    fn test_sentencepiece_roundtrip(text in vocab_string_strategy()) {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("c".to_string(), -1.0),
            ("ab".to_string(), -0.5),
            ("bc".to_string(), -0.5),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&encoded).expect("test");

        prop_assert_eq!(decoded, text);
    }

    /// Empty string encodes to empty token list
    #[test]
    fn test_empty_string_encoding(() in Just(())) {
        let tokens = vec!["<unk>".to_string(), "a".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        prop_assert!(encoded.is_empty());
    }

    /// Vocab size is consistent
    #[test]
    fn test_vocab_size_consistency(size in 2usize..100) {
        let tokens: Vec<String> = (0..size)
            .map(|i| if i == 0 { "<unk>".to_string() } else { format!("t{}", i) })
            .collect();
        let vocab = Vocabulary::from_tokens(tokens.clone()).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        prop_assert_eq!(tokenizer.vocab_size(), tokens.len());
    }
}
