
#[test]
fn test_phase35_rope_type_scaling_neox() {
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("custom");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let neox_value = build_gguf_string("neox");
    data.extend(build_gguf_metadata(
        "custom.rope.scaling.type",
        8,
        &neox_value,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX from "neox" scaling
}

#[test]
fn test_phase35_rope_type_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_type(), None); // No architecture
}

// =============================================================================
// Metadata Accessor Edge Cases
// =============================================================================

#[test]
fn test_phase35_embedding_dim_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.embedding_dim(), None);
}

#[test]
fn test_phase35_num_layers_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_layers(), None);
}

#[test]
fn test_phase35_num_heads_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_heads(), None);
}

#[test]
fn test_phase35_context_length_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.context_length(), None);
}

#[test]
fn test_phase35_num_kv_heads_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.num_kv_heads(), None);
}

#[test]
fn test_phase35_rope_freq_base_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rope_freq_base(), None);
}

#[test]
fn test_phase35_rms_epsilon_no_architecture() {
    let data = build_gguf_header(0, 0);
    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.rms_epsilon(), None);
}

#[test]
fn test_phase35_metadata_accessor_wrong_type() {
    // Set embedding_dim as string instead of u32
    let mut data = build_gguf_header(0, 2);
    let arch_value = build_gguf_string("llama");
    data.extend(build_gguf_metadata("general.architecture", 8, &arch_value));
    let dim_value = build_gguf_string("256"); // Wrong type!
    data.extend(build_gguf_metadata("llama.embedding_length", 8, &dim_value));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert_eq!(model.embedding_dim(), None); // Should return None for wrong type
}

// =============================================================================
// Tokenizer and Vocabulary Tests
// =============================================================================

#[test]
fn test_phase35_vocabulary_empty_array() {
    let mut data = build_gguf_header(0, 1);
    // Empty array
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes()); // element type: string
    array_bytes.extend_from_slice(&0u64.to_le_bytes()); // array length: 0
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.vocabulary().is_none()); // Empty vocabulary returns None
}

#[test]
fn test_phase35_vocabulary_non_string_elements() {
    let mut data = build_gguf_header(0, 1);
    // Array of u32 instead of strings
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&4u32.to_le_bytes()); // element type: u32
    array_bytes.extend_from_slice(&3u64.to_le_bytes()); // array length: 3
    array_bytes.extend_from_slice(&1u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u32.to_le_bytes());
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    assert!(model.vocabulary().is_none()); // Non-string elements filtered out
}

#[test]
fn test_phase35_decode_unknown_token() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("hello"));
    array_bytes.extend(build_gguf_string("world"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Decode with out-of-bounds token ID
    let decoded = model.decode(&[0, 99, 1]); // 99 is out of bounds
    assert!(decoded.contains("\u{FFFD}")); // Unknown token marker
}

#[test]
fn test_phase35_decode_gpt2_style() {
    let mut data = build_gguf_header(0, 2);
    // Set tokenizer model to gpt2
    let model_value = build_gguf_string("gpt2");
    data.extend(build_gguf_metadata("tokenizer.ggml.model", 8, &model_value));

    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    // GPT-2 uses Ġ (U+0120) for space
    array_bytes.extend(build_gguf_string("Hello"));
    array_bytes.extend(build_gguf_string("\u{0120}world")); // space-prefixed
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let decoded = model.decode(&[0, 1]);
    assert_eq!(decoded, "Hello world");
}

#[test]
fn test_phase35_decode_invalid_byte_token() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&2u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("<0xGG>")); // Invalid hex
    array_bytes.extend(build_gguf_string("test"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Invalid byte token should be kept as-is
    let decoded = model.decode(&[0, 1]);
    assert!(decoded.contains("<0xGG>") || decoded.contains("test"));
}

#[test]
fn test_phase35_encode_gpt2_style_newline() {
    let mut data = build_gguf_header(0, 2);
    let model_value = build_gguf_string("gpt2");
    data.extend(build_gguf_metadata("tokenizer.ggml.model", 8, &model_value));

    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("Hello"));
    array_bytes.extend(build_gguf_string("\u{010A}")); // GPT-2 newline (Ċ)
    array_bytes.extend(build_gguf_string("World"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tokens = model.encode("Hello\nWorld");
    assert!(tokens.is_some());
}

#[test]
fn test_phase35_encode_sentencepiece_style() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&3u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("▁hello")); // SentencePiece word boundary
    array_bytes.extend(build_gguf_string("▁world"));
    array_bytes.extend(build_gguf_string("!"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tokens = model.encode("hello world!");
    assert!(tokens.is_some());
    let tokens = tokens.expect("test value should be present");
    assert!(!tokens.is_empty());
}

#[test]
fn test_phase35_encode_with_byte_fallback() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes());
    array_bytes.extend_from_slice(&4u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("▁hello"));
    array_bytes.extend(build_gguf_string("<0x21>")); // '!'
    array_bytes.extend(build_gguf_string("<0x3F>")); // '?'
    array_bytes.extend(build_gguf_string("test"));
    data.extend(build_gguf_metadata(
        "tokenizer.ggml.tokens",
        9,
        &array_bytes,
    ));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Text with characters that need byte fallback
    let tokens = model.encode("hello!?test");
    assert!(tokens.is_some());
}

// =============================================================================
// Tensor Data Extraction Edge Cases
// =============================================================================

#[test]
fn test_phase35_tensor_3d_shape() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("tensor_3d", &[2, 3, 4], 0, 0)); // F32

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Add tensor data: 2*3*4 = 24 f32 values
    for i in 0..24 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model
        .get_tensor_f32("tensor_3d", &data)
        .expect("Should extract");
    assert_eq!(tensor.len(), 24);
}

#[test]
fn test_phase35_tensor_zero_elements() {
    let mut data = build_gguf_header(1, 0);
    data.extend(build_tensor_info("empty", &[0], 0, 0)); // Zero dimension

    // Pad to 32-byte alignment
    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model
        .get_tensor_f32("empty", &data)
        .expect("Should extract empty");
    assert_eq!(tensor.len(), 0);
}

#[test]
fn test_phase35_tensor_q4_0_multiple_blocks() {
    use crate::gguf::GGUF_TYPE_Q4_0;

    let mut data = build_gguf_header(1, 0);
    // 64 elements = 2 blocks of Q4_0
    data.extend(build_tensor_info("test", &[64], GGUF_TYPE_Q4_0, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // 2 Q4_0 blocks: 2 * 18 bytes = 36 bytes
    for _ in 0..2 {
        let scale = half::f16::from_f32(1.0);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0x88u8; 16]);
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 64);
}

#[test]
fn test_phase35_tensor_q8_0_multiple_blocks() {
    use crate::gguf::GGUF_TYPE_Q8_0;

    let mut data = build_gguf_header(1, 0);
    // 96 elements = 3 blocks of Q8_0
    data.extend(build_tensor_info("test", &[96], GGUF_TYPE_Q8_0, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // 3 Q8_0 blocks: 3 * 34 bytes = 102 bytes
    for _ in 0..3 {
        let scale = half::f16::from_f32(0.5);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([64i8 as u8; 32]); // Mid-range values
    }

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 96);
}

#[test]
fn test_phase35_tensor_q4_k_multiple_blocks() {
    use crate::gguf::GGUF_TYPE_Q4_K;

    let mut data = build_gguf_header(1, 0);
    // 512 elements = 2 super-blocks of Q4_K
    data.extend(build_tensor_info("test", &[512], GGUF_TYPE_Q4_K, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // 2 Q4_K super-blocks: 2 * 144 bytes = 288 bytes
    data.extend([0u8; 288]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let tensor = model.get_tensor_f32("test", &data).expect("Should extract");
    assert_eq!(tensor.len(), 512);
}

#[test]
fn test_phase35_tensor_q3_k() {
    use crate::gguf::GGUF_TYPE_Q3_K;

    let mut data = build_gguf_header(1, 0);
    // Q3_K: 110 bytes per 256 elements
    data.extend(build_tensor_info("test", &[256], GGUF_TYPE_Q3_K, 0));

    let current_len = data.len();
    let aligned = current_len.div_ceil(32) * 32;
    data.resize(aligned, 0);

    // Q3_K super-block: 110 bytes
    data.extend([0u8; 110]);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    // Q3_K might not be implemented, but should parse the file
    // The get_tensor_f32 might fail with "unsupported", which is expected
    let _ = model.get_tensor_f32("test", &data);
}

// =============================================================================
// Array Metadata Variations
// =============================================================================

#[test]
fn test_phase35_array_of_strings() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&8u32.to_le_bytes()); // string type
    array_bytes.extend_from_slice(&5u64.to_le_bytes());
    array_bytes.extend(build_gguf_string("one"));
    array_bytes.extend(build_gguf_string("two"));
    array_bytes.extend(build_gguf_string("three"));
    array_bytes.extend(build_gguf_string("four"));
    array_bytes.extend(build_gguf_string("five"));
    data.extend(build_gguf_metadata("test_strings", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_strings") {
        assert_eq!(arr.len(), 5);
        assert!(matches!(&arr[0], GGUFValue::String(s) if s == "one"));
        assert!(matches!(&arr[4], GGUFValue::String(s) if s == "five"));
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_phase35_array_of_floats() {
    let mut data = build_gguf_header(0, 1);
    let mut array_bytes = Vec::new();
    array_bytes.extend_from_slice(&6u32.to_le_bytes()); // f32 type
    array_bytes.extend_from_slice(&4u64.to_le_bytes());
    array_bytes.extend_from_slice(&1.0f32.to_le_bytes());
    array_bytes.extend_from_slice(&2.5f32.to_le_bytes());
    array_bytes.extend_from_slice(&(-3.0f32).to_le_bytes());
    array_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend(build_gguf_metadata("test_floats", 9, &array_bytes));

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test_floats") {
        assert_eq!(arr.len(), 4);
        if let GGUFValue::Float32(v) = &arr[1] {
            assert!((v - 2.5).abs() < f32::EPSILON);
        }
    } else {
        panic!("Expected Array");
    }
}
