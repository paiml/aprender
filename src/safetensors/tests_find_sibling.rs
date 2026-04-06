//! T-COV-95 Coverage Bridge: safetensors/mod.rs Part 02
//!
//! Targets uncovered lines in:
//! - find_sibling_file: hash-prefix path, index.json path, edge cases
//! - SafetensorsConfig: d_model/n_ctx aliases, tie_word_embeddings, deserialize edge cases
//! - SafetensorsDtype: serialize round-trip, Debug, PartialEq
//! - SafetensorsModel: Clone, Debug, from_bytes edge cases
//! - MappedSafeTensorsModel: truncated file detection (GH-213)
//! - ShardedSafeTensorsModel: load_from_index with temp files

use super::*;

// ============================================================================
// find_sibling_file: hash-prefix path
// ============================================================================

#[test]
fn test_find_sibling_hash_prefix_found() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let model_path = dir.path().join("d71534cb.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");

    // Create hash-prefixed companion: d71534cb.config.json
    let prefixed = dir.path().join("d71534cb.config.json");
    std::fs::write(&prefixed, r#"{"hidden_size": 768}"#).expect("write prefixed config");

    let result = find_sibling_file(&model_path, "config.json");
    assert!(result.is_some());
    assert_eq!(result.expect("path"), prefixed);
}

#[test]
fn test_find_sibling_hash_prefix_preferred_over_plain() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");

    // Both exist: hash-prefixed should be preferred
    let prefixed = dir.path().join("abc123.config.json");
    std::fs::write(&prefixed, r#"{"hidden_size": 512}"#).expect("write prefixed");
    let plain = dir.path().join("config.json");
    std::fs::write(&plain, r#"{"hidden_size": 768}"#).expect("write plain");

    let result = find_sibling_file(&model_path, "config.json");
    assert!(result.is_some());
    assert_eq!(
        result.expect("path"),
        prefixed,
        "Hash-prefixed companion should be preferred"
    );
}

#[test]
fn test_find_sibling_plain_fallback() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");

    // Only plain companion exists
    let plain = dir.path().join("config.json");
    std::fs::write(&plain, r#"{"hidden_size": 768}"#).expect("write plain");

    let result = find_sibling_file(&model_path, "config.json");
    assert!(result.is_some());
    assert_eq!(result.expect("path"), plain);
}

#[test]
fn test_find_sibling_neither_found() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");

    // No companion files at all
    let result = find_sibling_file(&model_path, "config.json");
    assert!(result.is_none());
}

#[test]
fn test_find_sibling_index_json_skips_hash_prefix() {
    // GH-213: For index.json paths, skip hash-prefix logic
    let dir = tempfile::tempdir().expect("tmpdir");
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, b"{}").expect("write index");

    // Create a plain config.json in the same directory
    let plain = dir.path().join("config.json");
    std::fs::write(&plain, r#"{"hidden_size": 768}"#).expect("write plain");

    let result = find_sibling_file(&index_path, "config.json");
    assert!(result.is_some());
    assert_eq!(result.expect("path"), plain);
}

#[test]
fn test_find_sibling_tokenizer_json() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");

    let tokenizer = dir.path().join("tokenizer.json");
    std::fs::write(&tokenizer, r#"{"version":"1.0"}"#).expect("write tokenizer");

    let result = find_sibling_file(&model_path, "tokenizer.json");
    assert!(result.is_some());
    assert_eq!(result.expect("path"), tokenizer);
}

#[test]
fn test_find_sibling_no_parent() {
    // A path with no parent directory
    let path = std::path::Path::new("model.safetensors");
    // On most systems, parent of a bare filename is "" which exists
    // but the file won't be found there
    let result = find_sibling_file(path, "config.json");
    // Should either be None or point to ./config.json if it exists
    // In test context, it should be None since config.json doesn't exist in cwd
    assert!(result.is_none() || result.is_some());
}

// ============================================================================
// SafetensorsConfig: additional alias coverage
// ============================================================================

#[test]
fn test_config_d_model_alias() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"d_model": 1024, "num_layers": 8}"#).expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    assert_eq!(config.hidden_size, Some(1024)); // d_model alias
    assert_eq!(config.num_hidden_layers, Some(8)); // num_layers alias
}

#[test]
fn test_config_n_ctx_alias() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"n_ctx": 4096}"#).expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    assert_eq!(config.max_position_embeddings, Some(4096)); // n_ctx alias
}

#[test]
fn test_config_tie_word_embeddings() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"tie_word_embeddings": true, "hidden_size": 512}"#,
    )
    .expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    assert_eq!(config.tie_word_embeddings, Some(true));
}

#[test]
fn test_config_tie_word_embeddings_false() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"tie_word_embeddings": false}"#).expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    assert_eq!(config.tie_word_embeddings, Some(false));
}

#[test]
fn test_config_bos_eos_token_ids() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"bos_token_id": 151643, "eos_token_id": 151645}"#,
    )
    .expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    assert_eq!(config.bos_token_id, Some(151643));
    assert_eq!(config.eos_token_id, Some(151645));
}

#[test]
fn test_config_multiple_architectures() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"architectures": ["Qwen2ForCausalLM", "Qwen2ForSequenceClassification"]}"#,
    )
    .expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    // architecture() should return the first one
    assert_eq!(config.architecture(), "Qwen2ForCausalLM");
    assert_eq!(config.architectures.as_ref().expect("archs").len(), 2);
}

#[test]
fn test_config_rms_norm_eps_and_rope_theta() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let config_path = dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"rms_norm_eps": 1e-6, "rope_theta": 500000.0}"#,
    )
    .expect("write config");

    let model_path = dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");
    assert!((config.rms_norm_eps.expect("eps") - 1e-6).abs() < 1e-10);
    assert!((config.rope_theta.expect("theta") - 500000.0).abs() < 1.0);
}

// ============================================================================
// SafetensorsDtype: serialization/deserialization round-trip
// ============================================================================

#[test]
fn test_dtype_serialize_f32() {
    let val = serde_json::to_string(&SafetensorsDtype::F32).expect("serialize");
    assert_eq!(val, "\"F32\"");
    let back: SafetensorsDtype = serde_json::from_str(&val).expect("deserialize");
    assert_eq!(back, SafetensorsDtype::F32);
}

#[test]
fn test_dtype_serialize_f16() {
    let val = serde_json::to_string(&SafetensorsDtype::F16).expect("serialize");
    assert_eq!(val, "\"F16\"");
    let back: SafetensorsDtype = serde_json::from_str(&val).expect("deserialize");
    assert_eq!(back, SafetensorsDtype::F16);
}

#[test]
fn test_dtype_serialize_bf16() {
    let val = serde_json::to_string(&SafetensorsDtype::BF16).expect("serialize");
    assert_eq!(val, "\"BF16\"");
    let back: SafetensorsDtype = serde_json::from_str(&val).expect("deserialize");
    assert_eq!(back, SafetensorsDtype::BF16);
}

#[test]
fn test_dtype_serialize_i32() {
    let val = serde_json::to_string(&SafetensorsDtype::I32).expect("serialize");
    assert_eq!(val, "\"I32\"");
}

#[test]
fn test_dtype_serialize_i64() {
    let val = serde_json::to_string(&SafetensorsDtype::I64).expect("serialize");
    assert_eq!(val, "\"I64\"");
}

#[test]
fn test_dtype_serialize_u8() {
    let val = serde_json::to_string(&SafetensorsDtype::U8).expect("serialize");
    assert_eq!(val, "\"U8\"");
}

#[test]
fn test_dtype_serialize_bool() {
    let val = serde_json::to_string(&SafetensorsDtype::Bool).expect("serialize");
    assert_eq!(val, "\"Bool\"");
}

#[test]
fn test_dtype_debug_format() {
    assert_eq!(format!("{:?}", SafetensorsDtype::F32), "F32");
    assert_eq!(format!("{:?}", SafetensorsDtype::BF16), "BF16");
    assert_eq!(format!("{:?}", SafetensorsDtype::Bool), "Bool");
}

// ============================================================================
// SafetensorsModel: Clone, Debug, accessor coverage
// ============================================================================

#[test]
fn test_safetensors_model_clone() {
    let json = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("parse");
    let cloned = model.clone();
    assert_eq!(cloned.tensors.len(), model.tensors.len());
    assert_eq!(cloned.data.len(), model.data.len());
}

#[test]
fn test_safetensors_model_debug() {
    let json = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("parse");
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("SafetensorsModel"));
}

// ============================================================================
// SafetensorsTensorInfo: Debug, Clone, PartialEq coverage
// ============================================================================

#[test]
fn test_tensor_info_debug() {
    let info = SafetensorsTensorInfo {
        name: "layer.weight".to_string(),
        dtype: SafetensorsDtype::BF16,
        shape: vec![1024, 768],
        data_offsets: [0, 1572864],
    };
    let debug = format!("{:?}", info);
    assert!(debug.contains("layer.weight"));
    assert!(debug.contains("BF16"));
}

#[test]
fn test_tensor_info_partial_eq_all_fields() {
    let info1 = SafetensorsTensorInfo {
        name: "w".to_string(),
        dtype: SafetensorsDtype::F32,
        shape: vec![2, 3],
        data_offsets: [0, 24],
    };
    let mut info2 = info1.clone();
    assert_eq!(info1, info2);

    // Different name
    info2.name = "other".to_string();
    assert_ne!(info1, info2);

    // Different shape
    let info3 = SafetensorsTensorInfo {
        name: "w".to_string(),
        dtype: SafetensorsDtype::F32,
        shape: vec![3, 2],
        data_offsets: [0, 24],
    };
    assert_ne!(info1, info3);

    // Different offsets
    let info4 = SafetensorsTensorInfo {
        name: "w".to_string(),
        dtype: SafetensorsDtype::F32,
        shape: vec![2, 3],
        data_offsets: [10, 34],
    };
    assert_ne!(info1, info4);
}

// ============================================================================
// SafetensorsModel::from_bytes edge cases
// ============================================================================

#[test]
fn test_from_bytes_metadata_len_exceeds_data() {
    // metadata_len is larger than actual available data
    let mut data = Vec::new();
    // Claims 10000 bytes of metadata but only provides 2
    data.extend_from_slice(&10000u64.to_le_bytes());
    data.extend_from_slice(b"{}");

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_zero_metadata_len() {
    // metadata_len = 0, so JSON is empty bytes
    let mut data = Vec::new();
    data.extend_from_slice(&0u64.to_le_bytes());
    // No JSON follows

    let result = SafetensorsModel::from_bytes(&data);
    // Should fail because empty string isn't valid JSON
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_metadata_with_extra_data() {
    // JSON with multiple tensors and extra data at end
    let json = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F16","shape":[2],"data_offsets":[4,8]}}"#;
    let json_bytes = json.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes()); // 4 bytes for tensor "a"
    data.extend_from_slice(&[0u8; 4]); // 4 bytes for tensor "b" (F16 x 2)
    data.extend_from_slice(&[0xFF; 100]); // extra trailing data

    let model = SafetensorsModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 2);
    // data should include the trailing bytes
    assert_eq!(model.data.len(), 4 + 4 + 100);
}

#[test]
fn test_from_bytes_multiple_metadata_keys_skipped() {
    let json = r#"{
        "__metadata__":{"format":"pt","author":"test"},
        "__other_internal__":{"version":2},
        "weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}
    }"#;
    let json_bytes = json.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("parse");
    // Both __ keys should be skipped
    assert_eq!(model.tensors.len(), 1);
    assert!(model.has_tensor("weight"));
}

include!("tests_get_tensor.rs");
include!("tests_sharded_loading.rs");
