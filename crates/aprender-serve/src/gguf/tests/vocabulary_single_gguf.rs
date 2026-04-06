
#[test]
fn test_vocabulary_single_token() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array("tokenizer.ggml.tokens", &["<unk>"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let vocab = model.vocabulary();
    assert!(vocab.is_some());
    assert_eq!(vocab.expect("vocab").len(), 1);
}

// ============================================================================
// GGUFConfig edge cases
// ============================================================================

#[test]
fn test_gguf_config_from_gguf_with_rope_freq_base() {
    use crate::gguf::test_factory::build_minimal_llama_gguf;
    let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = crate::gguf::GGUFConfig::from_gguf(&model);
    assert!(config.is_ok());
    let config = config.expect("config");
    assert_eq!(config.hidden_dim, 64);
    assert_eq!(config.num_heads, 4);
    assert!(config.rope_theta > 0.0);
}

// ============================================================================
// Decode mixed tokens (byte + regular + unknown)
// ============================================================================

#[test]
fn test_decode_mixed_byte_and_regular_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array(
            "tokenizer.ggml.tokens",
            &["Hello", "<0x20>", "world", "<0x21>"],
        )
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Hello<space>world!
    let decoded = model.decode(&[0, 1, 2, 3]);
    assert_eq!(decoded, "Hello world!");
}

#[test]
fn test_decode_invalid_byte_token_format() {
    // <0xZZ> is not valid hex, should NOT be treated as byte token
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array("tokenizer.ggml.tokens", &["<0xZZ>"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0]);
    // Should treat as regular string token, not byte
    assert!(decoded.contains("<0xZZ>"));
}
