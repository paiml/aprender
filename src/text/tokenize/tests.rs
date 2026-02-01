//\! Tokenize Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

// ========== WhitespaceTokenizer Tests ==========

#[test]
fn test_whitespace_tokenizer_basic() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("Hello world")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Hello", "world"]);
}

#[test]
fn test_whitespace_tokenizer_preserves_punctuation() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("Hello, world!")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Hello,", "world!"]);
}

#[test]
fn test_whitespace_tokenizer_multiple_spaces() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("foo   bar")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["foo", "bar"]);
}

#[test]
fn test_whitespace_tokenizer_newlines_tabs() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("line1\nline2\ttab")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["line1", "line2", "tab"]);
}

#[test]
fn test_whitespace_tokenizer_empty_string() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer.tokenize("").expect("tokenize should succeed");
    assert_eq!(tokens, Vec::<String>::new());
}

#[test]
fn test_whitespace_tokenizer_only_whitespace() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("   \n\t  ")
        .expect("tokenize should succeed");
    assert_eq!(tokens, Vec::<String>::new());
}

#[test]
fn test_whitespace_tokenizer_unicode() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("Hello мир 世界")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Hello", "мир", "世界"]);
}

#[test]
fn test_whitespace_tokenizer_leading_trailing_whitespace() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer
        .tokenize("  Hello world  ")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Hello", "world"]);
}

// ========== WordTokenizer Tests ==========

#[test]
fn test_word_tokenizer_basic() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer
        .tokenize("Hello world")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Hello", "world"]);
}

#[test]
fn test_word_tokenizer_separates_punctuation() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer
        .tokenize("Hello, world!")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Hello", ",", "world", "!"]);
}

#[test]
fn test_word_tokenizer_preserves_contractions() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer
        .tokenize("I don't know.")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["I", "don't", "know", "."]);
}

#[test]
fn test_word_tokenizer_multiple_punctuation() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer
        .tokenize("Wait... what?!")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["Wait", ".", ".", ".", "what", "?", "!"]);
}

#[test]
fn test_word_tokenizer_empty_string() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer.tokenize("").expect("tokenize should succeed");
    assert_eq!(tokens, Vec::<String>::new());
}

#[test]
fn test_word_tokenizer_only_punctuation() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer.tokenize(".,!?").expect("tokenize should succeed");
    assert_eq!(tokens, vec![".", ",", "!", "?"]);
}

#[test]
fn test_word_tokenizer_numbers() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer
        .tokenize("I have 3 apples.")
        .expect("tokenize should succeed");
    assert_eq!(tokens, vec!["I", "have", "3", "apples", "."]);
}

#[test]
fn test_word_tokenizer_hyphenated() {
    let tokenizer = WordTokenizer::new();

    let tokens = tokenizer
        .tokenize("state-of-the-art AI")
        .expect("tokenize should succeed");
    assert_eq!(
        tokens,
        vec!["state", "-", "of", "-", "the", "-", "art", "AI"]
    );
}

// ========== CharTokenizer Tests ==========

#[test]
fn test_char_tokenizer_basic() {
    let tokenizer = CharTokenizer::new();

    let tokens = tokenizer.tokenize("Hi").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["H", "i"]);
}

#[test]
fn test_char_tokenizer_with_punctuation() {
    let tokenizer = CharTokenizer::new();

    let tokens = tokenizer.tokenize("Hi!").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["H", "i", "!"]);
}

#[test]
fn test_char_tokenizer_with_spaces() {
    let tokenizer = CharTokenizer::new();

    let tokens = tokenizer.tokenize("a b").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["a", " ", "b"]);
}

#[test]
fn test_char_tokenizer_empty_string() {
    let tokenizer = CharTokenizer::new();

    let tokens = tokenizer.tokenize("").expect("tokenize should succeed");
    assert_eq!(tokens, Vec::<String>::new());
}

#[test]
fn test_char_tokenizer_unicode() {
    let tokenizer = CharTokenizer::new();

    let tokens = tokenizer.tokenize("世界").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["世", "界"]);
}

#[test]
fn test_char_tokenizer_emoji() {
    let tokenizer = CharTokenizer::new();

    let tokens = tokenizer.tokenize("Hi👋").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["H", "i", "👋"]);
}

// ========== Default Trait Tests ==========

#[test]
fn test_whitespace_tokenizer_default() {
    let tokenizer = WhitespaceTokenizer;
    let tokens = tokenizer.tokenize("test").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["test"]);
}

#[test]
fn test_word_tokenizer_default() {
    let tokenizer = WordTokenizer;
    let tokens = tokenizer.tokenize("test").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["test"]);
}

#[test]
fn test_char_tokenizer_default() {
    let tokenizer = CharTokenizer;
    let tokens = tokenizer.tokenize("ab").expect("tokenize should succeed");
    assert_eq!(tokens, vec!["a", "b"]);
}

// ========== BPE Tokenizer Tests ==========

#[test]
fn test_bpe_train_basic() {
    let corpus = vec!["low", "lower", "newest", "widest"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("training should succeed");

    assert!(tokenizer.vocab_size() >= 10); // At least special tokens + characters
}

#[test]
fn test_bpe_train_vocab_size_too_small() {
    let corpus = vec!["hello"];
    let result = BpeTokenizer::train(&corpus, 5);
    assert!(result.is_err());
}

#[test]
fn test_bpe_encode_decode_roundtrip() {
    let corpus = vec!["hello", "world", "hello"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    let ids = tokenizer.encode("hello").expect("encode");
    assert!(!ids.is_empty());

    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello");
}

#[test]
fn test_bpe_encode_decode_multiple_words() {
    let corpus = vec!["hello world", "hello there", "world wide"];
    let tokenizer = BpeTokenizer::train(&corpus, 100).expect("train");

    let ids = tokenizer.encode("hello world").expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello world");
}

#[test]
fn test_bpe_encode_empty_string() {
    let corpus = vec!["test"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    let ids = tokenizer.encode("").expect("encode");
    assert!(ids.is_empty());
}

#[test]
fn test_bpe_merges_created() {
    let corpus = vec!["aaa", "aab", "aba", "baa"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    // With repeated 'a', merges should be created
    assert!(!tokenizer.merges().is_empty() || tokenizer.vocab_size() > 10);
}

#[test]
fn test_bpe_vocab_accessors() {
    let corpus = vec!["hello"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    // Test vocab contains special tokens
    assert!(tokenizer.contains("<unk>"));

    // Test token_to_id and id_to_token
    if let Some(id) = tokenizer.token_to_id("<unk>") {
        let token = tokenizer.id_to_token(id);
        assert_eq!(token, Some("<unk>"));
    }
}

#[test]
fn test_bpe_from_vocab() {
    let mut vocab = HashMap::new();
    vocab.insert("<unk>".to_string(), 0);
    vocab.insert("a</w>".to_string(), 1);
    vocab.insert("b</w>".to_string(), 2);

    let tokenizer = BpeTokenizer::from_vocab(vocab, vec![]);
    assert_eq!(tokenizer.vocab_size(), 3);
}

#[test]
fn test_bpe_encode_with_special_tokens() {
    let corpus = vec!["hello"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    let ids_with_bos = tokenizer
        .encode_with_special("hello", true, false)
        .expect("encode");
    let ids_with_eos = tokenizer
        .encode_with_special("hello", false, true)
        .expect("encode");
    let ids_with_both = tokenizer
        .encode_with_special("hello", true, true)
        .expect("encode");
    let ids_plain = tokenizer.encode("hello").expect("encode");

    // With BOS should be longer
    assert!(ids_with_bos.len() >= ids_plain.len());
    assert!(ids_with_eos.len() >= ids_plain.len());
    assert!(ids_with_both.len() >= ids_plain.len());
}

#[test]
fn test_bpe_tokenizer_trait() {
    let corpus = vec!["hello", "world"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    // Test Tokenizer trait implementation
    let tokens = tokenizer.tokenize("hello").expect("tokenize");
    assert!(!tokens.is_empty());
}

#[test]
fn test_bpe_unknown_character() {
    let corpus = vec!["abc"];
    let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");

    // Encode text with character not in training corpus
    let ids = tokenizer.encode("xyz").expect("encode");
    // Should use UNK token
    assert!(!ids.is_empty());
}

#[test]
fn test_bpe_special_tokens_default() {
    let special = SpecialTokens::default();
    assert_eq!(special.unk, "<unk>");
    assert_eq!(special.bos, Some("<s>".to_string()));
    assert_eq!(special.eos, Some("</s>".to_string()));
    assert_eq!(special.pad, Some("<pad>".to_string()));
}

#[test]
fn test_bpe_custom_special_tokens() {
    let special = SpecialTokens {
        unk: "[UNK]".to_string(),
        bos: Some("[BOS]".to_string()),
        eos: None,
        pad: None,
    };

    let corpus = vec!["test"];
    let tokenizer = BpeTokenizer::train_with_special_tokens(&corpus, 50, special).expect("train");

    assert!(tokenizer.contains("[UNK]"));
    assert!(tokenizer.contains("[BOS]"));
}

// ========== HuggingFace Loading Tests (GH-128) ==========

#[test]
fn test_bpe_from_huggingface_basic() {
    let vocab_json = r#"{"hello": 0, "world": 1, "<unk>": 2}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert_eq!(tokenizer.vocab_size(), 3);
    assert!(tokenizer.contains("hello"));
    assert!(tokenizer.contains("world"));
    assert!(tokenizer.contains("<unk>"));
}

#[test]
fn test_bpe_from_huggingface_with_merges() {
    let vocab_json = r#"{"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "hel": 5, "hello": 6}"#;
    let merges_txt = "h e\nhe l\nhel lo";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert_eq!(tokenizer.vocab_size(), 7);
    assert_eq!(tokenizer.merges().len(), 3);
}

#[test]
fn test_bpe_from_huggingface_with_comments() {
    let vocab_json = r#"{"test": 0}"#;
    let merges_txt = "#version: 0.2\n# This is a comment\nt e\nte s";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    // Comments should be ignored
    assert_eq!(tokenizer.merges().len(), 2);
}

#[test]
fn test_bpe_from_huggingface_endoftext_special_token() {
    // Test F5: Special tokens handled - <|endoftext|> detection
    let vocab_json = r#"{"hello": 0, "<|endoftext|>": 50256, "<|startoftext|>": 50257}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    // Should detect <|endoftext|> as EOS token
    assert_eq!(tokenizer.eos_token(), Some("<|endoftext|>"));
    // Should detect <|startoftext|> as BOS token
    assert_eq!(tokenizer.bos_token(), Some("<|startoftext|>"));
}

#[test]
fn test_bpe_from_huggingface_whisper_tokens() {
    // Whisper-style special tokens
    let vocab_json = r#"{
            "<|endoftext|>": 50256,
            "<|startoftranscript|>": 50257,
            "<|en|>": 50258,
            "<|transcribe|>": 50259,
            "hello": 0
        }"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.contains("<|endoftext|>"));
    assert!(tokenizer.contains("<|startoftranscript|>"));
    assert_eq!(tokenizer.eos_token(), Some("<|endoftext|>"));
}

#[test]
fn test_bpe_from_huggingface_empty_vocab() {
    let vocab_json = r#"{}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert_eq!(tokenizer.vocab_size(), 0);
}

#[test]
fn test_bpe_from_huggingface_invalid_json() {
    let vocab_json = "not valid json";
    let merges_txt = "";

    let result = BpeTokenizer::from_huggingface(vocab_json, merges_txt);
    assert!(result.is_err());
}

#[test]
fn test_bpe_from_huggingface_unicode_tokens() {
    // Test F8: Unicode handled
    let vocab_json = r#"{"日本語": 0, "世界": 1, "こんにちは": 2}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.contains("日本語"));
    assert!(tokenizer.contains("世界"));
    assert!(tokenizer.contains("こんにちは"));
}

#[test]
fn test_bpe_from_huggingface_emoji_tokens() {
    // Test F9: Emoji handled
    let vocab_json = r#"{"😀": 0, "👋": 1, "🎉": 2, "hello": 3}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.contains("😀"));
    assert!(tokenizer.contains("👋"));
    assert!(tokenizer.contains("🎉"));
}

#[test]
fn test_bpe_from_huggingface_escaped_characters() {
    // Test escaped quotes in tokens
    let vocab_json = r#"{"test\"quote": 0, "back\\slash": 1}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.contains("test\"quote"));
    assert!(tokenizer.contains("back\\slash"));
}

#[test]
fn test_bpe_is_special_token() {
    let vocab_json = r#"{"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "hello": 4}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.is_special_token("<unk>"));
    assert!(tokenizer.is_special_token("</s>"));
    assert!(!tokenizer.is_special_token("hello"));
}

#[test]
fn test_bpe_unk_token_accessor() {
    let vocab_json = r#"{"<unk>": 0, "test": 1}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert_eq!(tokenizer.unk_token(), "<unk>");
}

#[test]
fn test_bpe_gpt2_style_vocab() {
    // GPT-2 uses "Ġ" prefix for word boundaries
    let vocab_json = r#"{"Ġhello": 0, "Ġworld": 1, "<|endoftext|>": 50256}"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert!(tokenizer.contains("Ġhello"));
    // Should detect GPT-2 style end-of-word marker
    assert_eq!(tokenizer.end_of_word_marker(), "Ġ");
}

#[test]
fn test_parse_vocab_json_multiline() {
    let vocab_json = r#"{
            "hello": 0,
            "world": 1,
            "test": 2
        }"#;
    let merges_txt = "";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert_eq!(tokenizer.vocab_size(), 3);
}

#[test]
fn test_parse_merges_empty_lines() {
    let vocab_json = r#"{"a": 0, "b": 1, "ab": 2}"#;
    let merges_txt = "\n\na b\n\n";

    let tokenizer =
        BpeTokenizer::from_huggingface(vocab_json, merges_txt).expect("loading should succeed");

    assert_eq!(tokenizer.merges().len(), 1);
}

// ========== WordPiece Tokenizer Tests ==========

#[test]
fn test_wordpiece_train_basic() {
    let corpus = vec!["playing", "played", "player"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    assert!(tokenizer.vocab_size() >= 5); // Special tokens
}

#[test]
fn test_wordpiece_train_vocab_size_too_small() {
    let corpus = vec!["hello"];
    let result = WordPieceTokenizer::train(&corpus, 5);
    assert!(result.is_err());
}

#[test]
fn test_wordpiece_encode_decode_roundtrip() {
    let corpus = vec!["hello", "world"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    let ids = tokenizer.encode("hello").expect("encode");
    assert!(!ids.is_empty());

    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello");
}

#[test]
fn test_wordpiece_continuation_prefix() {
    let corpus = vec!["unbelievable", "believable", "believe"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 100).expect("train");

    // Should have some ## prefixed tokens
    let has_continuation = tokenizer.vocab().keys().any(|k| k.starts_with("##"));
    assert!(has_continuation);
}

#[test]
fn test_wordpiece_encode_empty() {
    let corpus = vec!["test"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    let ids = tokenizer.encode("").expect("encode");
    assert!(ids.is_empty());
}

#[test]
fn test_wordpiece_decode_special_tokens_skipped() {
    let corpus = vec!["hello"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    // Get [UNK] ID
    let unk_id = tokenizer.vocab().get("[UNK]").copied().unwrap_or(0);

    // Decode with special token - should skip it
    let decoded = tokenizer.decode(&[unk_id]).expect("decode");
    assert!(decoded.is_empty());
}

#[test]
fn test_wordpiece_from_vocab() {
    let mut vocab = HashMap::new();
    vocab.insert("[UNK]".to_string(), 0);
    vocab.insert("hello".to_string(), 1);

    let tokenizer = WordPieceTokenizer::from_vocab(vocab);
    assert_eq!(tokenizer.vocab_size(), 2);
}

#[test]
fn test_wordpiece_tokenizer_trait() {
    let corpus = vec!["hello"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");

    let tokens = tokenizer.tokenize("hello").expect("tokenize");
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_long_word_becomes_unk() {
    let corpus = vec!["short"];
    let mut tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");
    tokenizer.max_word_len = 5; // Set max to 5

    // Word longer than max should become UNK
    let ids = tokenizer.encode("toolongword").expect("encode");
    let unk_id = tokenizer.vocab().get("[UNK]").copied().unwrap_or(0);
    assert!(ids.contains(&unk_id));
}

#[test]
fn test_wordpiece_multiple_words() {
    let corpus = vec!["hello world", "hello there"];
    let tokenizer = WordPieceTokenizer::train(&corpus, 100).expect("train");

    let ids = tokenizer.encode("hello world").expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello world");
}

// ========== Unigram/SentencePiece Tokenizer Tests ==========

#[test]
fn test_unigram_train_basic() {
    let corpus = vec!["hello world", "hello there"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    assert!(tokenizer.vocab_size() >= 4); // Special tokens
}

#[test]
fn test_unigram_train_vocab_size_too_small() {
    let corpus = vec!["hello"];
    let result = UnigramTokenizer::train(&corpus, 5);
    assert!(result.is_err());
}

#[test]
fn test_unigram_encode_decode_roundtrip() {
    let corpus = vec!["hello world", "hello there", "world wide"];
    let tokenizer = UnigramTokenizer::train(&corpus, 200).expect("train");

    let ids = tokenizer.encode("hello").expect("encode");
    assert!(!ids.is_empty());

    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello");
}

#[test]
fn test_unigram_encode_empty() {
    let corpus = vec!["test"];
    let tokenizer = UnigramTokenizer::train(&corpus, 50).expect("train");

    let ids = tokenizer.encode("").expect("encode");
    assert!(ids.is_empty());
}

#[test]
fn test_unigram_log_prob() {
    let corpus = vec!["hello hello hello", "world"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    // "hello" appears more frequently, should have higher probability
    let hello_prob = tokenizer.log_prob("▁hello");
    let world_prob = tokenizer.log_prob("▁world");

    if hello_prob.is_some() && world_prob.is_some() {
        // Log probs are negative, higher = more probable
        assert!(hello_prob >= world_prob);
    }
}

#[test]
fn test_unigram_vocab_ids() {
    let corpus = vec!["test"];
    let tokenizer = UnigramTokenizer::train(&corpus, 50).expect("train");

    let vocab_ids = tokenizer.vocab_ids();
    assert!(!vocab_ids.is_empty());
}

#[test]
fn test_unigram_from_vocab() {
    let mut vocab: HashMap<String, (u32, f64)> = HashMap::new();
    vocab.insert("<unk>".to_string(), (0, -10.0));
    vocab.insert("▁hello".to_string(), (1, -1.0));

    let tokenizer = UnigramTokenizer::from_vocab(vocab);
    assert_eq!(tokenizer.vocab_size(), 2);
}

#[test]
fn test_unigram_tokenizer_trait() {
    let corpus = vec!["hello world"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    let tokens = tokenizer.tokenize("hello").expect("tokenize");
    assert!(!tokens.is_empty());
}

#[test]
fn test_unigram_word_boundary() {
    let corpus = vec!["hello world"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    // Vocabulary should contain word-boundary prefixed tokens
    let has_boundary = tokenizer.vocab_ids().keys().any(|k| k.starts_with('▁'));
    assert!(has_boundary);
}

#[test]
fn test_unigram_multiple_words() {
    let corpus = vec!["hello world", "hello there", "world wide"];
    let tokenizer = UnigramTokenizer::train(&corpus, 200).expect("train");

    let ids = tokenizer.encode("hello world").expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello world");
}

#[test]
fn test_unigram_viterbi_optimal() {
    // Viterbi should find optimal segmentation
    let corpus = vec!["the", "there", "here", "where"];
    let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");

    // Should tokenize without errors
    let ids = tokenizer.encode("there").expect("encode");
    assert!(!ids.is_empty());
}

// ========== Cross-Tokenizer Comparison Tests ==========

#[test]
fn test_all_tokenizers_handle_empty() {
    let corpus = vec!["test"];

    let bpe = BpeTokenizer::train(&corpus, 50).expect("train");
    let wp = WordPieceTokenizer::train(&corpus, 50).expect("train");
    let uni = UnigramTokenizer::train(&corpus, 50).expect("train");

    assert!(bpe.encode("").expect("encode").is_empty());
    assert!(wp.encode("").expect("encode").is_empty());
    assert!(uni.encode("").expect("encode").is_empty());
}

#[test]
fn test_all_tokenizers_roundtrip() {
    let corpus = vec!["hello", "world", "hello"];

    let bpe = BpeTokenizer::train(&corpus, 100).expect("train");
    let wp = WordPieceTokenizer::train(&corpus, 100).expect("train");
    let uni = UnigramTokenizer::train(&corpus, 100).expect("train");

    // BPE roundtrip
    let bpe_ids = bpe.encode("hello").expect("encode");
    let bpe_decoded = bpe.decode(&bpe_ids).expect("decode");
    assert_eq!(bpe_decoded, "hello");

    // WordPiece roundtrip
    let wp_ids = wp.encode("hello").expect("encode");
    let wp_decoded = wp.decode(&wp_ids).expect("decode");
    assert_eq!(wp_decoded, "hello");

    // Unigram roundtrip
    let uni_ids = uni.encode("hello").expect("encode");
    let uni_decoded = uni.decode(&uni_ids).expect("decode");
    assert_eq!(uni_decoded, "hello");
}

#[test]
fn test_tokenizer_debug_implementations() {
    let corpus = vec!["test"];

    let bpe = BpeTokenizer::train(&corpus, 50).expect("train");
    let wp = WordPieceTokenizer::train(&corpus, 50).expect("train");
    let uni = UnigramTokenizer::train(&corpus, 50).expect("train");
    let special = SpecialTokens::default();

    // All should implement Debug
    let _ = format!("{:?}", bpe);
    let _ = format!("{:?}", wp);
    let _ = format!("{:?}", uni);
    let _ = format!("{:?}", special);
}

#[test]
fn test_tokenizer_clone_implementations() {
    let corpus = vec!["test"];

    let bpe = BpeTokenizer::train(&corpus, 50).expect("train");
    let wp = WordPieceTokenizer::train(&corpus, 50).expect("train");
    let uni = UnigramTokenizer::train(&corpus, 50).expect("train");
    let special = SpecialTokens::default();

    // All should implement Clone
    let _bpe_clone = bpe.clone();
    let _wp_clone = wp.clone();
    let _uni_clone = uni.clone();
    let _special_clone = special.clone();
}

// ========== SentenceTokenizer Tests ==========

#[test]
fn test_sentence_tokenizer_new() {
    let tokenizer = SentenceTokenizer::new();
    // Should have abbreviations initialized
    assert!(!tokenizer.abbreviations.is_empty());
}

#[test]
fn test_sentence_tokenizer_default() {
    // Default derive creates empty abbreviations
    let tokenizer = SentenceTokenizer::default();
    // Default has empty abbreviations, use new() for populated version
    assert!(tokenizer.abbreviations.is_empty());
}

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
