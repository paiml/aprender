
// ============================================================================
// 9. Config.json vocab_size padding
// ============================================================================

#[test]
fn test_config_json_vocab_size_pads_vocabulary() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"hello": 0, "world": 1}
        }
    });

    let config = serde_json::json!({
        "vocab_size": 10
    });

    let tok = parse_tokenizer_json(&json, Some(&config)).expect("should parse");
    assert_eq!(
        tok.vocabulary.len(),
        10,
        "vocabulary should be padded to config vocab_size"
    );
    assert_eq!(tok.vocabulary[0], "hello");
    assert_eq!(tok.vocabulary[1], "world");
    // Remaining slots padded with <unk>
    for i in 2..10 {
        assert_eq!(tok.vocabulary[i], "<unk>", "slot {} should be <unk>", i);
    }
}

#[test]
fn test_vocab_larger_than_config_vocab_size_uses_actual_size() {
    // If actual vocab is larger than config vocab_size, use actual size
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        }
    });

    let config = serde_json::json!({
        "vocab_size": 3
    });

    let tok = parse_tokenizer_json(&json, Some(&config)).expect("should parse");
    assert_eq!(
        tok.vocabulary.len(),
        5,
        "vocabulary should be max(config_vocab_size, max_id + 1)"
    );
}

// ============================================================================
// 10. Architecture and model_name fields are None (set externally)
// ============================================================================

#[test]
fn test_architecture_and_model_name_are_none() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"a": 0}
        }
    });

    let tok = parse_tokenizer_json(&json, None).expect("should parse");
    assert_eq!(
        tok.architecture, None,
        "architecture is set externally, not from tokenizer.json"
    );
    assert_eq!(
        tok.model_name, None,
        "model_name is set externally, not from tokenizer.json"
    );
}

// ============================================================================
// 11. Realistic end-to-end: Qwen2-style tokenizer
// ============================================================================

#[test]
fn test_realistic_qwen2_style_tokenizer() {
    let json = serde_json::json!({
        "model": {
            "vocab": {
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644,
                "<|im_end|>": 151645,
                "hello": 0,
                "world": 1,
                " ": 2
            },
            "merges": ["h e", "l l", "o w"],
            "type": "BPE"
        },
        "added_tokens": [
            {"content": "<|endoftext|>", "id": 151643, "special": true},
            {"content": "<|im_start|>", "id": 151644, "special": true},
            {"content": "<|im_end|>", "id": 151645, "special": true}
        ]
    });

    let config = serde_json::json!({
        "vocab_size": 151936,
        "bos_token_id": 151643,
        "eos_token_id": 151645
    });

    let tok = parse_tokenizer_json(&json, Some(&config)).expect("should parse Qwen2-style");
    assert_eq!(tok.vocabulary.len(), 151_936, "padded to config vocab_size");
    assert_eq!(tok.vocabulary[0], "hello");
    assert_eq!(tok.vocabulary[1], "world");
    assert_eq!(tok.vocabulary[2], " ");
    assert_eq!(tok.vocabulary[151_643], "<|endoftext|>");
    assert_eq!(tok.vocabulary[151_644], "<|im_start|>");
    assert_eq!(tok.vocabulary[151_645], "<|im_end|>");
    assert_eq!(tok.bos_token_id, Some(151_643));
    assert_eq!(tok.eos_token_id, Some(151_645));
    assert_eq!(tok.model_type.as_deref(), Some("BPE"));
    assert_eq!(tok.merges.len(), 3);
}

// ============================================================================
// 12. Realistic end-to-end: LLaMA-style tokenizer
// ============================================================================

#[test]
fn test_realistic_llama_style_tokenizer() {
    let json = serde_json::json!({
        "model": {
            "vocab": {"<unk>": 0, "the": 1, "of": 2},
            "merges": ["t h", "th e"],
            "type": "BPE"
        },
        "added_tokens": [
            {"content": "<s>", "id": 3},
            {"content": "</s>", "id": 4}
        ]
    });

    // No config.json provided - BOS/EOS should be inferred from added_tokens
    let tok = parse_tokenizer_json(&json, None).expect("should parse LLaMA-style");
    assert_eq!(tok.vocabulary.len(), 5);
    assert_eq!(tok.vocabulary[3], "<s>");
    assert_eq!(tok.vocabulary[4], "</s>");
    assert_eq!(
        tok.bos_token_id,
        Some(3),
        "<s> detected as BOS via fallback"
    );
    assert_eq!(
        tok.eos_token_id,
        Some(4),
        "</s> detected as EOS via fallback"
    );
    assert_eq!(tok.merges.len(), 2);
    assert_eq!(tok.model_type.as_deref(), Some("BPE"));
}
