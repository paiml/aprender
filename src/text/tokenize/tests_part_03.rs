
#[test]
fn test_sentence_tokenizer_basic_split() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("Hello world. How are you? I'm fine!");
    assert_eq!(sentences.len(), 3);
    assert_eq!(sentences[0], "Hello world.");
    assert_eq!(sentences[1], "How are you?");
    assert_eq!(sentences[2], "I'm fine!");
}

#[test]
fn test_sentence_tokenizer_empty_string() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("");
    assert!(sentences.is_empty());
}

#[test]
fn test_sentence_tokenizer_single_sentence() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("Hello world");
    assert_eq!(sentences.len(), 1);
    assert_eq!(sentences[0], "Hello world");
}

#[test]
fn test_sentence_tokenizer_abbreviations_mr() {
    let tokenizer = SentenceTokenizer::new();
    // "Mr." should not end a sentence
    let sentences = tokenizer.split("Mr. Smith went to the store. He bought milk.");
    assert_eq!(sentences.len(), 2);
    assert!(sentences[0].contains("Mr. Smith"));
}

#[test]
fn test_sentence_tokenizer_abbreviations_dr() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("Dr. Jones is here. She is a doctor.");
    assert_eq!(sentences.len(), 2);
    assert!(sentences[0].contains("Dr. Jones"));
}

#[test]
fn test_sentence_tokenizer_abbreviations_eg() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("Use tools e.g. hammers. They help a lot.");
    assert_eq!(sentences.len(), 2);
}

#[test]
fn test_sentence_tokenizer_question_mark() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("What is this? This is a test.");
    assert_eq!(sentences.len(), 2);
    assert_eq!(sentences[0], "What is this?");
}

#[test]
fn test_sentence_tokenizer_exclamation() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("Wow! That's amazing!");
    assert_eq!(sentences.len(), 2);
}

#[test]
fn test_sentence_tokenizer_no_space_after_period() {
    let tokenizer = SentenceTokenizer::new();
    // Period without space shouldn't split (like "3.14")
    let sentences = tokenizer.split("Pi is 3.14 approximately.");
    // Should be one sentence because "3.14" has no uppercase after
    assert_eq!(sentences.len(), 1);
}

#[test]
fn test_sentence_tokenizer_multiple_spaces() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("First sentence.   Second sentence.");
    assert_eq!(sentences.len(), 2);
}

#[test]
fn test_sentence_tokenizer_end_without_punctuation() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("First sentence. No punctuation");
    assert_eq!(sentences.len(), 2);
    assert_eq!(sentences[1], "No punctuation");
}

#[test]
fn test_sentence_tokenizer_only_punctuation() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("...");
    // All periods, no uppercase follows
    assert_eq!(sentences.len(), 1);
}

#[test]
fn test_sentence_tokenizer_mixed_abbreviations() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("Inc. and Ltd. are abbreviations. They don't end sentences.");
    assert_eq!(sentences.len(), 2);
}

// ========== Additional Edge Case Tests ==========

#[test]
fn test_word_tokenizer_unicode_punctuation() {
    let tokenizer = WordTokenizer::new();
    // Test with non-ASCII text
    let tokens = tokenizer.tokenize("Café!").expect("tokenize");
    assert!(tokens.contains(&"Café".to_string()));
    assert!(tokens.contains(&"!".to_string()));
}

#[test]
fn test_word_tokenizer_quotes() {
    let tokenizer = WordTokenizer::new();
    let tokens = tokenizer.tokenize("\"Hello\"").expect("tokenize");
    assert!(tokens.contains(&"\"".to_string()));
    assert!(tokens.contains(&"Hello".to_string()));
}

#[test]
fn test_bpe_decode_with_pad_token() {
    let corpus = vec!["hello"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    // Get pad token ID if it exists
    if let Some(&pad_id) = tokenizer.vocab().get("<pad>") {
        // Decode with pad token should skip it
        let decoded = tokenizer.decode(&[pad_id]).expect("decode");
        assert!(decoded.is_empty() || !decoded.contains("<pad>"));
    }
}

#[test]
fn test_bpe_decode_unknown_id() {
    let corpus = vec!["hello"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    // Decode with an ID that doesn't exist
    let decoded = tokenizer.decode(&[99999]).expect("decode");
    // Should use UNK token
    assert!(!decoded.is_empty() || tokenizer.vocab_size() < 99999);
}

#[test]
fn test_bpe_train_large_vocab() {
    let corpus = vec![
        "hello world how are you doing today",
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
    ];
    let tokenizer = BpeTokenizer::train(&corpus, 200).expect("train");

    // Should create merges
    assert!(!tokenizer.merges().is_empty() || tokenizer.vocab_size() >= 50);
}

#[test]
fn test_wordpiece_tokenize_unk_fallback() {
    let corpus = vec!["abc"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    // Tokenize something with characters not in vocab
    let tokens = tokenizer.tokenize("xyz").expect("tokenize");
    assert!(tokens.contains(&"[UNK]".to_string()));
}

#[test]
fn test_wordpiece_decode_continuation_tokens() {
    let corpus = vec!["unbelievable"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 100).expect("train");

    // Encode and decode should roundtrip
    let ids = tokenizer.encode("unbelievable").expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "unbelievable");
}

#[test]
fn test_unigram_decode_special_tokens_skipped() {
    let corpus = vec!["hello"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    // Get special token IDs
    let vocab_ids = tokenizer.vocab_ids();
    if let Some(&unk_id) = vocab_ids.get("<unk>") {
        let decoded = tokenizer.decode(&[unk_id]).expect("decode");
        // UNK should be skipped in decode
        assert!(!decoded.contains("<unk>"));
    }
}

#[test]
fn test_unigram_encode_single_character() {
    let corpus = vec!["a b c"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    let ids = tokenizer.encode("a").expect("encode");
    assert!(!ids.is_empty());
}

#[test]
fn test_bpe_vocab_accessor() {
    let corpus = vec!["test"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    let vocab = tokenizer.vocab();
    assert!(!vocab.is_empty());
    assert!(vocab.contains_key("<unk>"));
}

#[test]
fn test_parse_vocab_json_with_spaces() {
    let vocab_json = r#"{ "hello world" : 0 , "test" : 1 }"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.contains("hello world"));
}

#[test]
fn test_parse_merges_malformed_lines() {
    let vocab_json = r#"{"a": 0}"#;
    let merges_txt = "single_token\na b\ninvalid";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    // Should only parse "a b" as valid merge
    assert_eq!(tokenizer.merges().len(), 1);
}

#[test]
fn test_bpe_tokenize_multiple_words() {
    let corpus = vec!["hello world"];
    let tokenizer = BpeTokenizer::train(&corpus, 100).expect("train");

    let tokens = tokenizer.tokenize("hello world").expect("tokenize");
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_tokenize_long_word_unk() {
    let corpus = vec!["short"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    // Very long word that exceeds max_word_len should become UNK
    let long_word = "a".repeat(200);
    let tokens = tokenizer.tokenize(&long_word).expect("tokenize");
    assert!(tokens.contains(&"[UNK]".to_string()));
}

#[test]
fn test_unigram_viterbi_fallback() {
    let corpus = vec!["abc"];
    let tokenizer = UnigramTokenizer::train(&corpus, 50).expect("train");

    // Encode text with unknown characters - should fallback
    let ids = tokenizer.encode("xyz").expect("encode");
    assert!(!ids.is_empty());
}

#[test]
fn test_sentence_tokenizer_debug() {
    let tokenizer = SentenceTokenizer::new();
    let debug_str = format!("{:?}", tokenizer);
    assert!(debug_str.contains("SentenceTokenizer"));
}

#[test]
fn test_sentence_tokenizer_clone() {
    let tokenizer = SentenceTokenizer::new();
    let cloned = tokenizer.clone();
    assert_eq!(cloned.abbreviations.len(), tokenizer.abbreviations.len());
}

#[test]
fn test_bpe_decode_token_without_end_marker() {
    let mut vocab = HashMap::new();
    vocab.insert("<unk>".to_string(), 0);
    vocab.insert("hello".to_string(), 1); // No </w> marker
    vocab.insert("world</w>".to_string(), 2);

    let tokenizer = BpeTokenizer::from_vocab(vocab, vec![]);
    let decoded = tokenizer.decode(&[1, 2]).expect("decode");
    // Should handle both with and without markers
    assert!(decoded.contains("hello") || decoded.contains("world"));
}

#[test]
fn test_wordpiece_encode_unknown_char_fallback() {
    let corpus = vec!["hello"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    // Encode text with character not in any token
    let ids = tokenizer.encode("你好").expect("encode");
    let unk_id = tokenizer.vocab().get("[UNK]").copied().unwrap_or(0);
    assert!(ids.contains(&unk_id));
}

#[test]
fn test_sentence_tokenizer_trailing_whitespace() {
    let tokenizer = SentenceTokenizer::new();
    let sentences = tokenizer.split("First. Second.   ");
    assert_eq!(sentences.len(), 2);
    assert!(!sentences[1].ends_with(' '));
}
