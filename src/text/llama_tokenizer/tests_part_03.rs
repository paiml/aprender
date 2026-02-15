
// ========================================================================
// Additional GGUF Parsing Coverage Tests
// ========================================================================

#[test]
fn test_gguf_skip_value_u8_type() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&6u64.to_le_bytes()); // metadata_count

    // Tokens metadata
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores metadata
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Metadata with u8 type (type 0) - should be skipped
    let key3 = b"general.quantization_version";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&0u32.to_le_bytes()); // u8 type
    data.push(1u8); // value

    // BOS token ID
    let key4 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS token ID
    let key5 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // Unknown token ID
    let key6 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i8_type() {
    let data = create_gguf_with_extra_metadata(1, &[-1i8 as u8]); // i8 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_u16_type() {
    let data = create_gguf_with_extra_metadata(2, &100u16.to_le_bytes()); // u16 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i16_type() {
    let data = create_gguf_with_extra_metadata(3, &(-100i16).to_le_bytes()); // i16 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i32_type() {
    let data = create_gguf_with_extra_metadata(5, &(-1000i32).to_le_bytes()); // i32 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_f32_type() {
    let data = create_gguf_with_extra_metadata(6, &3.14f32.to_le_bytes()); // f32 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_bool_type() {
    let data = create_gguf_with_extra_metadata(7, &[1u8]); // bool type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_u64_type() {
    let data = create_gguf_with_extra_metadata(10, &10000u64.to_le_bytes()); // u64 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i64_type() {
    let data = create_gguf_with_extra_metadata(11, &(-10000i64).to_le_bytes()); // i64 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_f64_type() {
    let data = create_gguf_with_extra_metadata(12, &3.14159265f64.to_le_bytes()); // f64 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u8_elements() {
    let data = create_gguf_with_array_metadata(0, &[1u8, 2, 3, 4, 5]); // u8 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u16_elements() {
    let mut arr_data = Vec::new();
    for i in 0u16..5 {
        arr_data.extend_from_slice(&i.to_le_bytes());
    }
    let data = create_gguf_with_array_metadata(2, &arr_data); // u16 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u32_elements() {
    let mut arr_data = Vec::new();
    for i in 0u32..3 {
        arr_data.extend_from_slice(&i.to_le_bytes());
    }
    let data = create_gguf_with_array_metadata(4, &arr_data); // u32 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u64_elements() {
    let mut arr_data = Vec::new();
    for i in 0u64..2 {
        arr_data.extend_from_slice(&i.to_le_bytes());
    }
    let data = create_gguf_with_array_metadata(10, &arr_data); // u64 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_string_array_in_skip() {
    // Create GGUF with a string array that's not tokens/scores (should be skipped)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&7u64.to_le_bytes()); // 7 metadata entries

    // Extra string array (should be skipped)
    let key0 = b"general.tags";
    data.extend_from_slice(&(key0.len() as u64).to_le_bytes());
    data.extend_from_slice(key0);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&8u32.to_le_bytes()); // string element type
    let tags = ["tag1", "tag2"];
    data.extend_from_slice(&(tags.len() as u64).to_le_bytes());
    for tag in &tags {
        let bytes = tag.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Now add required tokenizer metadata
    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // BOS
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_truncated_key_length() {
    // GGUF where key length extends beyond data
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry

    // Key length that exceeds remaining data
    data.extend_from_slice(&1000u64.to_le_bytes());

    // Missing token metadata = error
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_gguf_truncated_value_type() {
    // GGUF where value type bytes are missing
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = b"test";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    // Missing value type - only 2 bytes instead of 4
    data.extend_from_slice(&[0u8, 0]);

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_gguf_missing_scores_uses_default() {
    // GGUF with tokens but without scores
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes()); // 4 metadata entries

    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // BOS
    let key2 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key3 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key4 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    // No scores - should use default
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_parse_string_array_wrong_element_type() {
    // Test parse_string_array with non-string element type
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 element type (wrong, should be 8)
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
    for _ in 0..3 {
        data.extend_from_slice(&0u32.to_le_bytes());
    }

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_parse_f32_array_wrong_element_type() {
    // Create GGUF with valid tokens but wrong scores array type
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes());

    // Valid tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores with wrong element type
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 element type (wrong, should be 6)
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0u32.to_le_bytes());
    }

    // Rest of required metadata
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    // Should error because scores array has wrong type
    assert!(result.is_err());
}

#[test]
fn test_gguf_string_metadata_skip() {
    // Test that string metadata (type 8) other than tokenizer.ggml.model is skipped
    let data = create_gguf_with_string_metadata("general.name", "TestModel");
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}
