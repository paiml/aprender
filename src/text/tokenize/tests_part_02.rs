
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
