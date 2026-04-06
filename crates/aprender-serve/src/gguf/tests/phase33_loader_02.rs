
#[test]
fn test_phase33_loader_get_tensor_unsupported_qtype() {
    // Use a fake quantization type that doesn't exist
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[32], 255, 0)); // 255 = invalid qtype

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);
    data.extend([0u8; 64]); // dummy data

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("255") || err.contains("Unsupported"),
        "Error: {}",
        err
    );
}

#[test]
fn test_phase33_loader_get_tensor_q2_k() {
    use crate::gguf::GGUF_TYPE_Q2_K;

    // Q2_K: 84 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q2_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q2_K super-block (84 bytes)
    data.extend([0u8; 84]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q2_K extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 256);
}

#[test]
fn test_phase33_loader_get_tensor_q4_k() {
    use crate::gguf::GGUF_TYPE_Q4_K;

    // Q4_K: 144 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q4_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q4_K super-block (144 bytes)
    data.extend([0u8; 144]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q4_K extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 256);
}

#[test]
fn test_phase33_loader_get_tensor_q5_k() {
    use crate::gguf::GGUF_TYPE_Q5_K;

    // Q5_K: 176 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q5_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q5_K super-block (176 bytes)
    data.extend([0u8; 176]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q5_K extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 256);
}

#[test]
fn test_phase33_loader_get_tensor_q6_k() {
    use crate::gguf::GGUF_TYPE_Q6_K;

    // Q6_K: 210 bytes per 256 elements (super-block)
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q6_K, 0));

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add Q6_K super-block (210 bytes)
    data.extend([0u8; 210]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let result = model.get_tensor_f32("test", &data);
    assert!(
        result.is_ok(),
        "Q6_K extraction should work: {:?}",
        result.err()
    );
    let tensor = result.expect("test value should be present");
    assert_eq!(tensor.len(), 256);
}

// =============================================================================
// RoPE Type Tests
// =============================================================================

#[test]
fn test_phase33_loader_rope_type_llama() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM style for LLaMA
}

#[test]
fn test_phase33_loader_rope_type_qwen2() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("qwen2");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX style for Qwen2
}

#[test]
fn test_phase33_loader_rope_type_phi3() {
    let mut data = build_gguf_header(0, 1);
    let arch_value = build_gguf_string("phi3");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX style for Phi3
}

#[test]
fn test_phase33_loader_rope_type_from_scaling() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let yarn_value = build_gguf_string("yarn");
    data.extend(build_gguf_metadata(
        "custom.rope.scaling.type",
        8,
        &yarn_value,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX from yarn scaling
}

#[test]
fn test_phase33_loader_rope_type_linear_scaling() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let linear_value = build_gguf_string("linear");
    data.extend(build_gguf_metadata(
        "custom.rope.scaling.type",
        8,
        &linear_value,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM from linear scaling
}

// =============================================================================
// Token ID Tests
// =============================================================================

#[test]
fn test_phase33_loader_bos_token_id() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.bos_token_id",
        4,
        &1u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.bos_token_id(), Some(1));
}

#[test]
fn test_phase33_loader_bos_token_id_missing() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.bos_token_id(), None);
}

#[test]
fn test_phase33_loader_eos_token_id() {
    let mut data = build_gguf_header(0, 1);
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.eos_token_id",
        4,
        &2u32.to_le_bytes(),
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.eos_token_id(), Some(2));
}

// =============================================================================
// Vocabulary Tests
// =============================================================================

#[test]
fn test_phase33_loader_vocabulary() {
    let mut data = build_gguf_header(0, 1);
    // Array of strings: element_type (8=string) + array_len (3) + strings
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes()); // element type: string
    array_bytes.extend_from_slice(&3u64.to_le_bytes()); // array length: 3
    array_bytes.extend(build_gguf_string("hello"));
    array_bytes.extend(build_gguf_string("world"));
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let vocab = model.vocabulary().expect("Should have vocabulary");
    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab[0], "hello");
    assert_eq!(vocab[1], "world");
    assert_eq!(vocab[2], "!");
}

#[test]
fn test_phase33_loader_vocabulary_missing() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.vocabulary().is_none());
}

// =============================================================================
// Decode Tests
// =============================================================================

#[test]
fn test_phase33_loader_decode_basic() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("hello"));
    array_bytes.extend(build_gguf_string("▁world")); // SentencePiece space
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[0, 1, 2]);
    assert_eq!(decoded, "hello world!");
}

#[test]
fn test_phase33_loader_decode_no_vocab() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[65, 66, 67]); // ASCII fallback
    assert_eq!(decoded, "ABC");
}

#[test]
fn test_phase33_loader_decode_byte_tokens() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("<0x48>")); // 'H'
    array_bytes.extend(build_gguf_string("<0x69>")); // 'i'
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[0, 1]);
    assert_eq!(decoded, "Hi");
}

// =============================================================================
// Encode Tests
// =============================================================================

#[test]
fn test_phase33_loader_encode_basic() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("▁hello")); // SentencePiece prefix
    array_bytes.extend(build_gguf_string("▁world"));
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tokens = model.encode("hello world!").expect("Should encode");
    // Encoding adds ▁ prefix to " hello" -> "▁hello" matches token 0
    assert!(!tokens.is_empty());
}

#[test]
fn test_phase33_loader_encode_no_vocab() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.encode("hello").is_none());
}
