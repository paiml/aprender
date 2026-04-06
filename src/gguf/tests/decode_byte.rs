
#[test]
fn test_decode_byte_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["<0x48>", "<0x69>", "!"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    let text = model.decode(&[0, 1, 2]);
    // <0x48> = 'H', <0x69> = 'i'
    assert!(text.contains('H'));
    assert!(text.contains('i'));
    assert!(text.contains('!'));
}

#[test]
fn test_decode_unknown_token_id() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    // Token ID 999 is out of vocabulary bounds
    let text = model.decode(&[999]);
    assert!(text.contains('�') || text.contains('?'));
}

#[test]
fn test_decode_empty_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    let text = model.decode(&[]);
    assert!(text.is_empty());
}

// ============================================================================
// encode: vocabulary-based
// ============================================================================

#[test]
fn test_encode_sentencepiece_basic() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["▁Hello", "▁world", "!"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    let tokens = model.encode("Hello world!");
    assert!(tokens.is_some());
}

#[test]
fn test_encode_no_vocabulary_returns_none() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert!(model.encode("test").is_none());
}

#[test]
fn test_encode_empty_text() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    let tokens = model.encode("");
    assert!(tokens.is_some());
    // Empty text produces empty or minimal tokens
}

// ============================================================================
// from_bytes: error paths
// ============================================================================

#[test]
fn test_from_bytes_empty() {
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_truncated_4_bytes() {
    let result = GGUFModel::from_bytes(&[0x47, 0x47, 0x55, 0x46]);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_truncated_8_bytes() {
    // Valid magic but truncated version
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_wrong_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("magic") || err.contains("Invalid") || err.contains("Magic"));
}

#[test]
fn test_from_bytes_wrong_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&99u32.to_le_bytes()); // version 99 (unsupported; v2 is valid per GH-310)
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("version") || err.contains("Unsupported"));
}

#[test]
fn test_from_bytes_truncated_16_bytes() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    // missing metadata count
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// rope_type: explicit scaling type metadata
// ============================================================================

#[test]
fn test_rope_type_scaling_none() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "none")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(0)); // NORM
}

#[test]
fn test_rope_type_scaling_linear() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "linear")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(0)); // NORM
}

#[test]
fn test_rope_type_scaling_yarn() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "yarn")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_scaling_neox() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_string("llama.rope.scaling.type", "neox")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

// ============================================================================
// vocabulary: edge cases
// ============================================================================

#[test]
fn test_vocabulary_returns_some_with_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string_array("tokenizer.ggml.tokens", &["a", "b", "c"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let vocab = model.vocabulary();
    assert!(vocab.is_some());
    assert_eq!(vocab.expect("test value should be present").len(), 3);
}

#[test]
fn test_vocabulary_returns_none_without_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert!(model.vocabulary().is_none());
}

// ============================================================================
// get_tensor_f32: unsupported qtype error path
// ============================================================================

#[test]
fn test_get_tensor_f32_unsupported_qtype() {
    // Build raw GGUF with unknown qtype (99)
    let mut data = Vec::new();
    // Header
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
    // Tensor info: name "bad.weight", 1 dim [4], qtype=99, offset=0
    let name = "bad.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&99u32.to_le_bytes()); // qtype = 99 (unsupported)
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
    // Align to 32 bytes
    let aligned = data.len().div_ceil(32) * 32;
    data.resize(aligned, 0);
    // Tensor data (some bytes)
    data.extend_from_slice(&[0u8; 64]);

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let result = model.get_tensor_f32("bad.weight", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unsupported") || err.contains("quantization"));
}

// ============================================================================
// Mixed quant types including new ones
// ============================================================================

#[test]
fn test_model_with_all_quant_types() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("f32.weight", &[32], &vec![1.0f32; 32])
        .add_q4_0_tensor("q4_0.weight", &[32], &create_q4_0_data(32))
        .add_q8_0_tensor("q8_0.weight", &[32], &create_q8_0_data(32))
        .add_q4_k_tensor("q4_k.weight", &[256], &create_q4_k_data(256))
        .add_q5_k_tensor("q5_k.weight", &[256], &create_q5_k_data(256))
        .add_q6_k_tensor("q6_k.weight", &[256], &create_q6_k_data(256))
        .add_q2_k_tensor("q2_k.weight", &[256], &create_q2_k_data(256))
        .add_f16_tensor("f16.weight", &[32], &create_f16_data(32))
        .add_q4_1_tensor("q4_1.weight", &[32], &create_q4_1_data(32))
        .add_q5_0_tensor("q5_0.weight", &[32], &create_q5_0_data(32))
        .add_q5_1_tensor("q5_1.weight", &[32], &create_q5_1_data(32))
        .build();

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.tensors.len(), 11);

    // Verify each tensor can be dequantized
    for tensor in &model.tensors {
        let result = model.get_tensor_f32(&tensor.name, &data);
        assert!(
            result.is_ok(),
            "Failed to dequantize {}: {:?}",
            tensor.name,
            result.err()
        );
    }
}
