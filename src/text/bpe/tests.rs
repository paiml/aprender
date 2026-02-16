#[allow(clippy::wildcard_imports)]
pub(crate) use super::super::*;
#[test]
fn test_bpe_config_default() {
    let config = BpeConfig::default();
    assert_eq!(config.unk_token, "<unk>");
    assert_eq!(config.vocab_size, 50257);
    assert!(config.add_prefix_space);
}

#[test]
fn test_bpe_config_whisper() {
    let config = BpeConfig::whisper();
    assert_eq!(config.vocab_size, 51865);
    assert!(!config.add_prefix_space);
}

#[test]
fn test_bpe_config_llama() {
    let config = BpeConfig::llama();
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.bos_token, Some("<s>".to_string()));
}

#[test]
fn test_merge_rule() {
    let rule = MergeRule::new("hel", "lo");
    assert_eq!(rule.first, "hel");
    assert_eq!(rule.second, "lo");
    assert_eq!(rule.merged(), "hello");
}

#[test]
fn test_tokenizer_new() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    assert_eq!(tokenizer.vocab_size(), 0);
}

#[test]
fn test_tokenizer_gpt2_base() {
    let tokenizer = BpeTokenizer::gpt2_base();
    assert!(tokenizer.vocab_size() > 0);

    // Check special token
    assert!(tokenizer.is_special_token("<|endoftext|>"));
    assert_eq!(tokenizer.token_to_id("<|endoftext|>"), Some(50256));
}

#[test]
fn test_add_special_token() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_special_token("<test>", 100);

    assert!(tokenizer.is_special_token("<test>"));
    assert_eq!(tokenizer.token_to_id("<test>"), Some(100));
    assert_eq!(tokenizer.id_to_token(100), Some("<test>"));
}

#[test]
fn test_add_merge() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_merge("a", "b");
    tokenizer.add_merge("ab", "c");

    assert_eq!(tokenizer.merges.len(), 2);
    assert_eq!(
        tokenizer
            .merge_ranks
            .get(&("a".to_string(), "b".to_string())),
        Some(&0)
    );
    assert_eq!(
        tokenizer
            .merge_ranks
            .get(&("ab".to_string(), "c".to_string())),
        Some(&1)
    );
}

#[test]
fn test_encode_empty() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty());
}

#[test]
fn test_encode_basic() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let tokens = tokenizer.encode("Hello");
    // Should produce some tokens (byte-level encoding)
    assert!(!tokens.is_empty());
}

#[test]
fn test_decode_empty() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let text = tokenizer.decode(&[]);
    assert!(text.is_empty());
}

#[test]
fn test_encode_decode_roundtrip() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let original = "Hi";
    let tokens = tokenizer.encode(original);
    let decoded = tokenizer.decode(&tokens);

    // Should approximately match (may have prefix space)
    assert!(decoded.contains("Hi") || decoded.trim() == "Hi");
}

#[test]
fn test_encode_checked() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let result = tokenizer.encode_checked("test");
    assert!(result.is_ok());
    assert!(!result.expect("encode failed").is_empty());
}

#[test]
fn test_decode_checked() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let result = tokenizer.decode_checked(&[72]); // 'H' in byte encoding
    assert!(result.is_ok());
}

#[test]
fn test_bytes_to_unicode() {
    let (encoder, decoder) = bytes_to_unicode();

    // Check that encoder and decoder are inverses
    for b in 0..=255u8 {
        if let Some(&c) = encoder.get(&b) {
            assert_eq!(decoder.get(&c), Some(&b));
        }
    }

    // Check ASCII printable chars map to themselves
    assert_eq!(encoder.get(&b'A'), Some(&'A'));
    assert_eq!(encoder.get(&b'z'), Some(&'z'));
    assert_eq!(encoder.get(&b'0'), Some(&'0'));
}

#[test]
fn test_pre_tokenize() {
    let tokenizer = BpeTokenizer::new(BpeConfig {
        add_prefix_space: false,
        ..BpeConfig::default()
    });

    let words = tokenizer.pre_tokenize("Hello world");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0], "Hello");
    assert!(words[1].contains("world"));
}

#[test]
fn test_bpe_no_merges() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = tokenizer.bpe(&tokens);
    assert_eq!(result, tokens); // No merges, unchanged
}

#[test]
fn test_bpe_with_merge() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_merge("a", "b");

    let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = tokenizer.bpe(&tokens);
    assert_eq!(result, vec!["ab".to_string(), "c".to_string()]);
}

#[test]
fn test_bpe_multiple_merges() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_merge("a", "b");
    tokenizer.add_merge("ab", "c");

    let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = tokenizer.bpe(&tokens);
    assert_eq!(result, vec!["abc".to_string()]);
}

#[test]
fn test_load_from_json_empty() {
    let result = load_from_json("");
    assert!(result.is_err());
}

#[test]
fn test_load_from_json_basic() {
    // Valid HuggingFace tokenizer.json structure
    let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": ["he llo", "wor ld"]
            },
            "added_tokens": []
        }"#;
    let result = load_from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert_eq!(tokenizer.vocab_size(), 2);
    assert_eq!(tokenizer.merges.len(), 2);
}

#[test]
fn test_load_from_json_invalid() {
    // Invalid JSON (missing model field) should fail
    let result = load_from_json("{}");
    assert!(result.is_err());
}

#[test]
fn test_load_from_files_empty_vocab() {
    let result = load_from_files("", "");
    assert!(result.is_err());
}

#[test]
fn test_load_from_files_basic() {
    let vocab = "{}";
    let merges = "a b\nab c\n";
    let result = load_from_files(vocab, merges);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert_eq!(tokenizer.merges.len(), 2);
}

#[test]
fn test_load_from_files_skip_comments() {
    let vocab = "{}";
    let merges = "# comment\na b\n";
    let result = load_from_files(vocab, merges);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert_eq!(tokenizer.merges.len(), 1);
}

#[test]
fn test_encode_unicode() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let tokens = tokenizer.encode("日本語");
    // Should handle unicode without panicking
    assert!(!tokens.is_empty());
}

#[test]
fn test_encode_emoji() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let tokens = tokenizer.encode("Hello 😀");
    // Should handle emoji without panicking
    assert!(!tokens.is_empty());
}

#[test]
fn test_whitespace_handling() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let tokens1 = tokenizer.encode(" ");
    let tokens2 = tokenizer.encode("  ");

    // Whitespace should produce tokens
    assert!(!tokens1.is_empty());
    assert!(!tokens2.is_empty());
}

// ========================================================================
// Qwen2 BPE Tokenizer Tests
// ========================================================================

#[test]
fn test_qwen2_config() {
    let config = BpeConfig::qwen2();
    assert_eq!(config.vocab_size, 151936);
    assert_eq!(config.eos_token, Some("<|im_end|>".to_string()));
    assert_eq!(config.bos_token, Some("<|im_start|>".to_string()));
    assert!(!config.add_prefix_space);
}

#[test]
fn test_qwen2_tokenizer_new() {
    let tokenizer = Qwen2BpeTokenizer::new();
    assert_eq!(tokenizer.vocab_size(), 151936);
}

#[test]
fn test_qwen2_special_tokens() {
    let tokenizer = Qwen2BpeTokenizer::new();

    assert!(tokenizer.is_eos(Qwen2BpeTokenizer::IM_END_ID));
    assert!(tokenizer.is_eos(Qwen2BpeTokenizer::ENDOFTEXT_ID));
    assert!(!tokenizer.is_eos(0));

    assert!(tokenizer.is_bos(Qwen2BpeTokenizer::IM_START_ID));
    assert!(!tokenizer.is_bos(0));
}

#[test]
fn test_qwen2_format_chat() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let formatted = tokenizer.format_chat("user", "Hello!");

    assert!(formatted.starts_with("<|im_start|>user"));
    assert!(formatted.contains("Hello!"));
    assert!(formatted.ends_with("<|im_end|>\n"));
}

#[test]
fn test_qwen2_format_conversation() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let messages = vec![
        ("system", "You are a helpful assistant."),
        ("user", "Hello!"),
    ];
    let formatted = tokenizer.format_conversation(&messages);

    assert!(formatted.contains("<|im_start|>system"));
    assert!(formatted.contains("You are a helpful assistant."));
    assert!(formatted.contains("<|im_start|>user"));
    assert!(formatted.contains("Hello!"));
    assert!(formatted.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_qwen2_encode_decode() {
    let tokenizer = Qwen2BpeTokenizer::new();

    // Basic encode (without real vocab, just byte-level)
    let tokens = tokenizer.encode("Hi");
    assert!(!tokens.is_empty());

    // Decode back
    let decoded = tokenizer.decode(&tokens);
    assert!(decoded.contains("Hi") || decoded.trim() == "Hi");
}

#[test]
fn test_qwen2_token_ids() {
    let tokenizer = Qwen2BpeTokenizer::new();

    assert_eq!(tokenizer.im_start_id(), 151644);
    assert_eq!(tokenizer.im_end_id(), 151645);
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_bpe_config_gpt2() {
    let config = BpeConfig::gpt2();
    assert_eq!(config.vocab_size, 50257);
    assert_eq!(config.unk_token, "<unk>");
    assert!(config.add_prefix_space);
    assert_eq!(config.bos_token, Some("<|endoftext|>".to_string()));
    assert_eq!(config.eos_token, Some("<|endoftext|>".to_string()));
    assert!(config.pad_token.is_none());
}

#[test]
fn test_tokenizer_default() {
    let tokenizer = BpeTokenizer::default();
    assert_eq!(tokenizer.vocab_size(), 0);
    assert!(tokenizer.merges.is_empty());
}

#[test]
fn test_qwen2_tokenizer_default() {
    let tokenizer = Qwen2BpeTokenizer::default();
    assert_eq!(tokenizer.vocab_size(), 151936);
    assert!(tokenizer.is_eos(Qwen2BpeTokenizer::IM_END_ID));
}

#[test]
fn test_bpe_single_token() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    let tokens = vec!["abc".to_string()];
    let result = tokenizer.bpe(&tokens);
    assert_eq!(result, vec!["abc".to_string()]);
}

#[test]
fn test_split_on_special_tokens_empty_text() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_special_token("<test>", 100);
    let segments = tokenizer.split_on_special_tokens("");
    assert!(segments.is_empty() || segments == vec![String::new()]);
}

#[test]
fn test_split_on_special_tokens_no_match() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_special_token("<test>", 100);
    let segments = tokenizer.split_on_special_tokens("hello world");
    assert_eq!(segments, vec!["hello world".to_string()]);
}

#[test]
fn test_split_on_special_tokens_multiple() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_special_token("<a>", 100);
    tokenizer.add_special_token("<b>", 101);
    let segments = tokenizer.split_on_special_tokens("Hello<a>world<b>!");
    assert!(segments.contains(&"<a>".to_string()));
    assert!(segments.contains(&"<b>".to_string()));
}

#[test]
fn test_split_on_special_tokens_at_start() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_special_token("<s>", 100);
    let segments = tokenizer.split_on_special_tokens("<s>hello");
    assert_eq!(segments[0], "<s>");
    assert_eq!(segments[1], "hello");
}

#[test]
fn test_split_on_special_tokens_consecutive() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    tokenizer.add_special_token("<a>", 100);
    tokenizer.add_special_token("<b>", 101);
    let segments = tokenizer.split_on_special_tokens("<a><b>");
    assert!(segments.contains(&"<a>".to_string()));
    assert!(segments.contains(&"<b>".to_string()));
}

#[test]
fn test_encode_with_special_tokens() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let text = "<|endoftext|>Hello";
    let tokens = tokenizer.encode(text);
    // Should contain the special token ID
    assert!(tokens.contains(&50256));
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
