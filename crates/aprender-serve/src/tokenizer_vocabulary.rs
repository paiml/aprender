
    #[test]
    fn test_vocabulary_from_tokens() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.get_id("<unk>"), Some(0));
        assert_eq!(vocab.get_id("hello"), Some(1));
        assert_eq!(vocab.get_id("world"), Some(2));
        assert_eq!(vocab.get_token(0), Some("<unk>"));
        assert_eq!(vocab.get_token(1), Some("hello"));
        assert_eq!(vocab.get_token(2), Some("world"));
    }

    #[test]
    fn test_vocabulary_empty_error() {
        let result = Vocabulary::from_tokens(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocabulary_duplicate_error() {
        let tokens = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(), // Duplicate
        ];
        let result = Vocabulary::from_tokens(tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocabulary_get_missing() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        assert_eq!(vocab.get_id("world"), None);
        assert_eq!(vocab.get_token(999), None);
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        // Encode known tokens
        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![1, 2]);

        // Decode back
        let decoded = tokenizer.decode(&encoded).expect("test");
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_tokenizer_unknown_token() {
        let tokens = vec!["<unk>".to_string(), "hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        // Unknown token should map to <unk> (ID 0)
        let encoded = tokenizer.encode("hello foo");
        assert_eq!(encoded, vec![1, 0]);
    }

    #[test]
    fn test_tokenizer_invalid_unk_token() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let result = Tokenizer::new(vocab, "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_decode_invalid_id() {
        let tokens = vec!["<unk>".to_string(), "hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let result = tokenizer.decode(&[1, 999]); // 999 is invalid
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_empty_string() {
        let tokens = vec!["<unk>".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        assert_eq!(encoded, Vec::<u32>::new());

        let decoded = tokenizer.decode(&[]).expect("test");
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        assert_eq!(tokenizer.vocab_size(), 3);
    }

    // BPE Tokenizer tests

    #[test]
    fn test_bpe_tokenizer_creation() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "hel".to_string(),
            "hello".to_string(),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "lo".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        assert_eq!(tokenizer.vocab_size(), 9);
    }

    #[test]
    fn test_bpe_tokenizer_empty_vocab_error() {
        let result = BPETokenizer::new(vec![], vec![], "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_tokenizer_invalid_unk_token_error() {
        let vocab = vec!["hello".to_string()];
        let result = BPETokenizer::new(vocab, vec![], "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_no_merges() {
        // Simple character-level tokenization without merges
        let vocab = vec!["<unk>".to_string(), "h".to_string(), "i".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let encoded = tokenizer.encode("hi");
        assert_eq!(encoded, vec![1, 2]); // h=1, i=2
    }

    #[test]
    fn test_bpe_encode_with_merges() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        let encoded = tokenizer.encode("hello");
        // h+e -> he, l+l -> ll, o stays
        // so: he, ll, o = [5, 6, 4]
        assert_eq!(encoded, vec![5, 6, 4]);
    }

    #[test]
    fn test_bpe_encode_unknown_char() {
        let vocab = vec!["<unk>".to_string(), "h".to_string(), "i".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // 'x' is not in vocab, should map to <unk>
        let encoded = tokenizer.encode("hix");
        assert_eq!(encoded, vec![1, 2, 0]);
    }

    #[test]
    fn test_bpe_encode_empty_string() {
        let vocab = vec!["<unk>".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_encode_multiple_words() {
        // BPE uses GPT-2 encoding: space -> Ġ (U+0120)
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "i".to_string(),
            "Ġ".to_string(),  // GPT-2 space encoding
            "Ġh".to_string(), // GPT-2 space + h
        ];
        let merges = vec![("Ġ".to_string(), "h".to_string())];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        // "hi hi" -> "hi" + " hi" (space becomes Ġ)
        // "hi" -> h, i
        // "Ġhi" -> "Ġ" + "h" -> "Ġh", then "i"
        let encoded = tokenizer.encode("hi hi");
        assert_eq!(encoded, vec![1, 2, 4, 2]); // h, i, "Ġh", i
    }

    #[test]
    fn test_bpe_decode() {
        let vocab = vec!["<unk>".to_string(), "hel".to_string(), "lo".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_decode_empty() {
        let vocab = vec!["<unk>".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[]).expect("test");
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_bpe_decode_invalid_id_error() {
        let vocab = vec!["<unk>".to_string(), "hi".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let result = tokenizer.decode(&[1, 999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "lo".to_string(),
            "hel".to_string(),
            "hello".to_string(),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("l".to_string(), "o".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "lo".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_get_token_methods() {
        let vocab = vec!["<unk>".to_string(), "hello".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        assert_eq!(tokenizer.get_token_id("hello"), Some(1));
        assert_eq!(tokenizer.get_token_id("world"), None);
        assert_eq!(tokenizer.get_token(1), Some("hello"));
        assert_eq!(tokenizer.get_token(999), None);
    }

    #[test]
    fn test_bpe_multiple_consecutive_merges() {
        // Test that multiple merges are applied correctly
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
            "abab".to_string(),
        ];
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "ab".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        let encoded = tokenizer.encode("abab");
        // First: a+b -> ab, a+b -> ab giving [ab, ab]
        // Then: ab+ab -> abab giving [abab]
        assert_eq!(encoded, vec![4]);
    }

    // SentencePiece Tokenizer tests

    #[test]
    fn test_sentencepiece_tokenizer_creation() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("hello".to_string(), -1.0),
            ("world".to_string(), -1.5),
        ];

        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");
        assert_eq!(tokenizer.vocab_size(), 3);
    }

    #[test]
    fn test_sentencepiece_empty_vocab_error() {
        let result = SentencePieceTokenizer::new(vec![], "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_sentencepiece_invalid_unk_token_error() {
        let vocab = vec![("hello".to_string(), -1.0)];
        let result = SentencePieceTokenizer::new(vocab, "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_sentencepiece_encode_empty() {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_sentencepiece_encode_single_token() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("hello".to_string(), -1.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        assert_eq!(encoded, vec![1]);
    }

    #[test]
    fn test_sentencepiece_encode_prefers_higher_score() {
        // "hello" as single token has score -1.0
        // "hel" + "lo" would have score -2.0 + -2.0 = -4.0
        // So "hello" should be preferred
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -5.0),
            ("e".to_string(), -5.0),
            ("l".to_string(), -5.0),
            ("o".to_string(), -5.0),
            ("hel".to_string(), -2.0),
            ("lo".to_string(), -2.0),
            ("hello".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        // Should prefer single "hello" token (score -1.0) over subwords
        assert_eq!(encoded, vec![7]);
    }

    #[test]
    fn test_sentencepiece_encode_subwords() {
        // Only subwords available, not full word
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -1.0),
            ("e".to_string(), -1.0),
            ("l".to_string(), -1.0),
            ("o".to_string(), -1.0),
            ("he".to_string(), -0.5),
            ("llo".to_string(), -0.5),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        // "he" (-0.5) + "llo" (-0.5) = -1.0 is better than "h" + "e" + "l" + "l" + "o" = -5.0
        assert_eq!(encoded, vec![5, 6]);
    }

    #[test]
    fn test_sentencepiece_decode() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("hel".to_string(), -1.0),
            ("lo".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_sentencepiece_decode_empty() {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let decoded = tokenizer.decode(&[]).expect("test");
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_sentencepiece_decode_invalid_id_error() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("hi".to_string(), -1.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let result = tokenizer.decode(&[1, 999]);
        assert!(result.is_err());
    }
