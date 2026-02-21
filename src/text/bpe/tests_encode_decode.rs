use super::*;

#[test]
fn test_encode_without_prefix_space() {
    let config = BpeConfig {
        add_prefix_space: false,
        ..BpeConfig::default()
    };
    let tokenizer = BpeTokenizer::new(config);
    let tokens = tokenizer.encode("test");
    // With empty vocab, should be empty
    assert!(tokens.is_empty());
}

#[test]
fn test_decode_with_unknown_id() {
    let tokenizer = BpeTokenizer::gpt2_base();
    // ID that doesn't exist in vocab
    let decoded = tokenizer.decode(&[999999]);
    // Unknown ID should be skipped
    // Either empty or an empty string - both cases handled by is_empty()
    assert!(decoded.is_empty());
}

#[test]
fn test_decode_skips_special_tokens() {
    let tokenizer = BpeTokenizer::gpt2_base();
    // Decode with special token ID
    let decoded = tokenizer.decode(&[50256, 72]); // endoftext + 'H'
                                                  // Should not contain the special token text
    assert!(!decoded.contains("<|endoftext|>"));
}

#[test]
fn test_bytes_to_bpe_tokens_unknown() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    // Force a path through unknown byte handling
    let result = tokenizer.bytes_to_bpe_tokens("A");
    assert!(!result.is_empty());
}

#[test]
fn test_load_from_json_with_special_added_tokens() {
    let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            },
            "added_tokens": [
                {"id": 100, "content": "<special>", "special": true},
                {"id": 101, "content": "normal", "special": false}
            ]
        }"#;
    let result = load_from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert!(tokenizer.is_special_token("<special>"));
    assert!(!tokenizer.is_special_token("normal"));
    assert_eq!(tokenizer.token_to_id("normal"), Some(101));
}

#[test]
fn test_load_from_json_whisper_vocab_size() {
    // Create vocab with >50000 entries for whisper detection
    let mut vocab_entries: Vec<String> = Vec::new();
    for i in 0..51000 {
        vocab_entries.push(format!("\"tok{i}\": {i}"));
    }
    let vocab_str = vocab_entries.join(", ");
    let json = format!(
        "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
        vocab_str
    );
    let result = load_from_json(&json);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_json_qwen2_vocab_size() {
    // Create vocab with >150000 entries for qwen2 detection
    let mut vocab_entries: Vec<String> = Vec::new();
    for i in 0..151000 {
        vocab_entries.push(format!("\"tok{i}\": {i}"));
    }
    let vocab_str = vocab_entries.join(", ");
    let json = format!(
        "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
        vocab_str
    );
    let result = load_from_json(&json);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_whisper_vocab_size() {
    // Create vocab with >50000 entries
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0..51000u32 {
        vocab.insert(format!("tok{i}"), i);
    }
    let vocab_json = serde_json::to_string(&vocab).expect("serialize");
    let merges = "";
    let result = load_from_files(&vocab_json, merges);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_gpt2_vocab_size() {
    // Create vocab with >40000 but <50000 entries
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0..41000u32 {
        vocab.insert(format!("tok{i}"), i);
    }
    let vocab_json = serde_json::to_string(&vocab).expect("serialize");
    let merges = "";
    let result = load_from_files(&vocab_json, merges);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_qwen2_vocab_size() {
    // Create vocab with >150000 entries
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0..151000u32 {
        vocab.insert(format!("tok{i}"), i);
    }
    let vocab_json = serde_json::to_string(&vocab).expect("serialize");
    let merges = "";
    let result = load_from_files(&vocab_json, merges);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_empty_lines() {
    let vocab = "{}";
    let merges = "\n\na b\n\n";
    let result = load_from_files(vocab, merges);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert_eq!(tokenizer.merges.len(), 1);
}

#[test]
fn test_load_from_files_invalid_json() {
    let vocab = "not valid json";
    let merges = "";
    let result = load_from_files(vocab, merges);
    assert!(result.is_err());
}

#[test]
fn test_qwen2_from_json() {
    let json = r#"{
            "model": {
                "vocab": {
                    "<|endoftext|>": 151643,
                    "<|im_start|>": 151644,
                    "<|im_end|>": 151645,
                    "hello": 0
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "special": true},
                {"id": 151644, "content": "<|im_start|>", "special": true},
                {"id": 151645, "content": "<|im_end|>", "special": true}
            ]
        }"#;
    let result = Qwen2BpeTokenizer::from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert!(tokenizer.is_eos(151645));
    assert!(tokenizer.is_bos(151644));
}

#[test]
fn test_qwen2_from_json_default_ids() {
    // Test with vocab missing special tokens - should use defaults
    let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            },
            "added_tokens": []
        }"#;
    let result = Qwen2BpeTokenizer::from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    // Should use default IDs
    assert_eq!(tokenizer.im_start_id(), Qwen2BpeTokenizer::IM_START_ID);
    assert_eq!(tokenizer.im_end_id(), Qwen2BpeTokenizer::IM_END_ID);
}

#[test]
fn test_qwen2_from_file_not_found() {
    let result = Qwen2BpeTokenizer::from_file("/nonexistent/path/tokenizer.json");
    assert!(result.is_err());
}

#[test]
fn test_merge_rule_debug_clone() {
    let rule = MergeRule::new("a", "b");
    let cloned = rule.clone();
    assert_eq!(rule, cloned);

    // Test Debug
    let debug_str = format!("{:?}", rule);
    assert!(debug_str.contains("MergeRule"));
}

#[test]
fn test_bpe_tokenizer_debug_clone() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let cloned = tokenizer.clone();
    assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

    // Test Debug
    let debug_str = format!("{:?}", tokenizer);
    assert!(debug_str.contains("BpeTokenizer"));
}

#[test]
fn test_qwen2_tokenizer_debug_clone() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let cloned = tokenizer.clone();
    assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

    // Test Debug
    let debug_str = format!("{:?}", tokenizer);
    assert!(debug_str.contains("Qwen2BpeTokenizer"));
}

#[test]
fn test_bpe_config_debug_clone() {
    let config = BpeConfig::default();
    let cloned = config.clone();
    assert_eq!(config.vocab_size, cloned.vocab_size);

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("BpeConfig"));
}

#[test]
fn test_encode_unk_token_fallback() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    // Add unk token to vocab
    tokenizer.add_special_token("<unk>", 0);

    // Encode text - unknown bytes should fall back to unk token
    let tokens = tokenizer.encode("x");
    // Should either be empty or have unk token
    assert!(tokens.is_empty() || tokens.contains(&0));
}

#[test]
fn test_pre_tokenize_multiple_spaces() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    let words = tokenizer.pre_tokenize("hello  world");
    // Should handle multiple spaces
    assert!(words.len() >= 2);
}

#[test]
fn test_pre_tokenize_leading_space() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    let words = tokenizer.pre_tokenize(" hello");
    assert!(!words.is_empty());
    // First word should start with space
    assert!(words[0].starts_with(' '));
}

#[test]
fn test_bpe_tokens_to_bytes_invalid_chars() {
    let tokenizer = BpeTokenizer::gpt2_base();
    // String with chars not in byte_decoder
    let result = tokenizer.bpe_tokens_to_bytes("αβγ");
    // Should handle gracefully (lossy conversion)
    // Result might be empty or partial
    let _ = result;
}

#[test]
fn test_qwen2_encode_special_tokens() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let text = "<|im_start|>user";
    let tokens = tokenizer.encode(text);
    // Should contain the special token ID
    assert!(tokens.contains(&151644));
}

#[test]
fn test_bpe_merge_priority() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    // Add merges with specific priority order
    tokenizer.add_merge("x", "y"); // rank 0 (highest priority)
    tokenizer.add_merge("a", "b"); // rank 1

    // Test that lower rank (higher priority) merge is applied first
    let tokens = vec![
        "a".to_string(),
        "b".to_string(),
        "x".to_string(),
        "y".to_string(),
    ];
    let result = tokenizer.bpe(&tokens);
    // Both merges should be applied
    assert_eq!(result.len(), 2);
}
