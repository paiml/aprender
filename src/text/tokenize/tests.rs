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

include!("tests_part_02.rs");
include!("tests_part_03.rs");
