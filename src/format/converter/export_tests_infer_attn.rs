use super::*;

#[test]
fn test_infer_model_config_attention_wq_pattern() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    // "attention.wq" pattern (Llama-style)
    tensors.insert(
        "model.layers.0.attention.wq.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 32);
}

// ========================================================================
// infer_model_config: Alternative k_proj name patterns for GQA
// ========================================================================

#[test]
fn test_infer_model_config_gqa_via_attn_k_proj() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    // "attn.k_proj" pattern
    tensors.insert(
        "model.layers.0.attn.k_proj.weight".to_string(),
        (vec![], vec![512, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // kv_dim=512, head_dim=128, kv_heads = 512/128 = 4
    assert_eq!(v["num_key_value_heads"], 4);
}

#[test]
fn test_infer_model_config_gqa_via_attention_wk() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    // "attention.wk" pattern (Llama-style)
    tensors.insert(
        "model.layers.0.attention.wk.weight".to_string(),
        (vec![], vec![1024, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // kv_dim=1024, head_dim=128, kv_heads = 1024/128 = 8
    assert_eq!(v["num_key_value_heads"], 8);
}

// ========================================================================
// infer_model_config: output.weight used for vocab instead of lm_head
// ========================================================================

#[test]
fn test_infer_model_config_output_weight_for_vocab() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF-style output.weight instead of lm_head
    tensors.insert("output.weight".to_string(), (vec![], vec![128256, 4096]));
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["vocab_size"], 128256);
}

// ========================================================================
// infer_model_config: 1D vocab tensor fallback
// ========================================================================

#[test]
fn test_infer_model_config_1d_lm_head_vocab() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D lm_head - uses first dim
    tensors.insert("lm_head.weight".to_string(), (vec![], vec![50000]));
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["vocab_size"], 50000);
}

// ========================================================================
// infer_model_config: 1D embedding + 1D lm_head (degenerate case)
// ========================================================================

#[test]
fn test_infer_model_config_empty_1d_embedding() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D embedding with no elements - shape.last() returns Some(0)
    tensors.insert("token_embd.weight".to_string(), (vec![], vec![0]));
    let config = infer_model_config(&tensors);
    // Should still produce valid JSON
    let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(&config);
    assert!(
        result.is_ok(),
        "Should produce valid JSON even with 0-dim tensor"
    );
}

// ========================================================================
// extract_user_metadata: Empty/short data
// ========================================================================

#[test]
fn test_extract_user_metadata_nonexistent_file() {
    let result = extract_user_metadata(Path::new("/tmp/nonexistent_apr_file_12345.apr"));
    assert!(result.is_empty());
}

#[test]
fn test_extract_user_metadata_short_data() {
    let dir = std::env::temp_dir().join("apr_test_short_data");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("short.apr");
    // Write less than 16 bytes
    fs::write(&path, &[0u8; 10]).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Invalid metadata JSON
// ========================================================================

#[test]
fn test_extract_user_metadata_invalid_json() {
    let dir = std::env::temp_dir().join("apr_test_invalid_json");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("invalid.apr");

    // Build APR-like bytes: magic(4) + version(4) + metadata_len(8) + garbage
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00"); // magic
    data.extend_from_slice(&2u32.to_le_bytes()); // version
    let garbage = b"not valid json{{{";
    let len = garbage.len() as u64;
    data.extend_from_slice(&len.to_le_bytes()); // metadata_len
    data.extend_from_slice(garbage);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Valid with source_metadata
// ========================================================================

#[test]
fn test_extract_user_metadata_with_source_metadata() {
    let dir = std::env::temp_dir().join("apr_test_source_metadata");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("with_meta.apr");

    let json = r#"{"custom":{"source_metadata":{"key1":"val1","key2":"val2"}}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00"); // magic
    data.extend_from_slice(&2u32.to_le_bytes()); // version
    let json_bytes = json.as_bytes();
    let len = json_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes()); // metadata_len
    data.extend_from_slice(json_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert_eq!(result.len(), 2);
    assert_eq!(result.get("key1").map(String::as_str), Some("val1"));
    assert_eq!(result.get("key2").map(String::as_str), Some("val2"));

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Valid without source_metadata
// ========================================================================

#[test]
fn test_extract_user_metadata_without_source_metadata() {
    let dir = std::env::temp_dir().join("apr_test_no_source_metadata");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("no_source_meta.apr");

    let json = r#"{"custom":{"other_key":"other_val"}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    let json_bytes = json.as_bytes();
    let len = json_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(json_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Non-string values in source_metadata skipped
// ========================================================================

#[test]
fn test_extract_user_metadata_non_string_values_skipped() {
    let dir = std::env::temp_dir().join("apr_test_non_string_values");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("mixed_types.apr");

    let json = r#"{"custom":{"source_metadata":{"str_key":"str_val","num_key":42,"bool_key":true,"null_key":null}}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    let json_bytes = json.as_bytes();
    let len = json_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(json_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    // Only the string value should be extracted
    assert_eq!(result.len(), 1);
    assert_eq!(result.get("str_key").map(String::as_str), Some("str_val"));

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Metadata length exceeds data length
// ========================================================================

#[test]
fn test_extract_user_metadata_metadata_len_exceeds_data() {
    let dir = std::env::temp_dir().join("apr_test_len_overflow");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("overflow.apr");

    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    // Claim metadata is 9999 bytes but only provide 5
    let len: u64 = 9999;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(b"hello");

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// extract_user_metadata: Invalid UTF-8 in metadata
// ========================================================================

#[test]
fn test_extract_user_metadata_invalid_utf8() {
    let dir = std::env::temp_dir().join("apr_test_invalid_utf8");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("bad_utf8.apr");

    let mut data = Vec::new();
    data.extend_from_slice(b"APR\x00");
    data.extend_from_slice(&2u32.to_le_bytes());
    // Invalid UTF-8 sequence
    let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81, 0x82];
    let len = bad_bytes.len() as u64;
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(bad_bytes);

    fs::write(&path, &data).expect("write failed");

    let result = extract_user_metadata(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// infer_tokenizer_json: Non-APR extension returns empty
// ========================================================================

#[test]
fn test_infer_tokenizer_json_non_apr_extension() {
    let result = infer_tokenizer_json(Path::new("/tmp/model.safetensors"));
    assert!(result.is_empty());
}

#[test]
fn test_infer_tokenizer_json_gguf_extension() {
    let result = infer_tokenizer_json(Path::new("/tmp/model.gguf"));
    assert!(result.is_empty());
}

#[test]
fn test_infer_tokenizer_json_no_extension() {
    let result = infer_tokenizer_json(Path::new("/tmp/model"));
    assert!(result.is_empty());
}

// ========================================================================
// infer_tokenizer_json: APR file that doesn't exist
// ========================================================================

#[test]
fn test_infer_tokenizer_json_nonexistent_apr() {
    let result = infer_tokenizer_json(Path::new("/tmp/nonexistent_12345.apr"));
    assert!(result.is_empty());
}

// ========================================================================
// infer_tokenizer_json: APR file too short
// ========================================================================

#[test]
fn test_infer_tokenizer_json_short_apr_file() {
    let dir = std::env::temp_dir().join("apr_test_short_apr_tokenizer");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("short.apr");
    // Write less than 44 bytes (header size)
    fs::write(&path, &[0u8; 20]).expect("write failed");

    let result = infer_tokenizer_json(&path);
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// infer_tokenizer_json: APR file with no tokenizer in metadata
// ========================================================================

#[test]
fn test_infer_tokenizer_json_apr_without_tokenizer() {
    let dir = std::env::temp_dir().join("apr_test_no_tokenizer");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("no_tok.apr");

    // Build fake APR: 44 bytes header + metadata JSON without tokenizer + terminator
    let mut data = vec![0u8; 44]; // header
    let metadata = r#"{"model_type": "qwen2", "hidden_size": 896}"#;
    data.extend_from_slice(metadata.as_bytes());
    data.extend_from_slice(b"}\n\n\n"); // terminator pattern

    fs::write(&path, &data).expect("write failed");

    let result = infer_tokenizer_json(&path);
    // Metadata doesn't contain "tokenizer" or "vocabulary"
    assert!(result.is_empty());

    let _ = fs::remove_file(&path);
}

// ========================================================================
// GH-253-4: ValidatedGgufMetadata tests
// ========================================================================

#[test]
fn test_validated_metadata_requires_architecture() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![(
        "general.name".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("general.architecture"));
}

#[test]
fn test_validated_metadata_tokens_require_model() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("qwen2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec!["a".to_string()]),
        ),
    ];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("tokenizer.ggml.model"));
}

#[test]
fn test_validated_metadata_model_requires_tokens() {
    use crate::format::gguf::GgufValue;
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("qwen2".to_string()),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        ),
    ];
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("tokenizer.ggml.tokens"));
}
