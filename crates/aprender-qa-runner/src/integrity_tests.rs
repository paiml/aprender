use super::*;

use std::io::Write;
use tempfile::TempDir;

fn create_test_config(dir: &Path, layers: usize, hidden: usize, vocab: usize) {
    let config = format!(
        r#"{{
            "num_hidden_layers": {layers},
            "hidden_size": {hidden},
            "vocab_size": {vocab},
            "num_attention_heads": 12
        }}"#
    );
    let path = dir.join("config.json");
    std::fs::write(path, config).expect("write config");
}

fn create_mock_safetensors(dir: &Path, layers: usize, hidden: usize, vocab: usize) {
    // Create a minimal valid safetensors header
    let mut header_obj = serde_json::Map::new();

    // Add embedding tensor
    let mut embed_info = serde_json::Map::new();
    embed_info.insert("shape".to_string(), serde_json::json!([vocab, hidden]));
    embed_info.insert(
        "dtype".to_string(),
        serde_json::Value::String("F32".to_string()),
    );
    embed_info.insert(
        "data_offsets".to_string(),
        serde_json::json!([0, vocab * hidden * 4]),
    );
    header_obj.insert(
        "model.embed_tokens.weight".to_string(),
        serde_json::Value::Object(embed_info),
    );

    // Add layer tensors
    for i in 0..layers {
        let mut layer_info = serde_json::Map::new();
        layer_info.insert("shape".to_string(), serde_json::json!([hidden, hidden]));
        layer_info.insert(
            "dtype".to_string(),
            serde_json::Value::String("F32".to_string()),
        );
        layer_info.insert("data_offsets".to_string(), serde_json::json!([0, 0]));
        header_obj.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            serde_json::Value::Object(layer_info),
        );
    }

    let header_json = serde_json::to_string(&header_obj).expect("serialize header");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let path = dir.join("model.safetensors");
    let mut file = File::create(path).expect("create safetensors");
    file.write_all(&header_len.to_le_bytes())
        .expect("write len");
    file.write_all(header_bytes).expect("write header");
    // Write minimal tensor data (just zeros to satisfy offsets)
    file.write_all(&[0u8; 1024]).expect("write data");
}


/// Verify find_config_for_model_file returns hash-prefixed config when available
#[test]
fn test_find_config_hash_prefix() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"fake").expect("write model");
    let config_path = dir.path().join("abc123.config.json");
    std::fs::write(&config_path, r#"{"num_hidden_layers": 4}"#).expect("write config");

    let found = find_config_for_model_file(&model_path);
    assert_eq!(found, Some(config_path));
}

/// Verify find_config_for_model_file falls back to config.json in same directory
#[test]
fn test_find_config_fallback_config_json() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"fake").expect("write model");
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, r#"{"num_hidden_layers": 4}"#).expect("write config");

    let found = find_config_for_model_file(&model_path);
    assert_eq!(found, Some(config_path));
}

/// Verify find_config_for_model_file returns None when no config exists
#[test]
fn test_find_config_no_config_returns_none() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"fake").expect("write model");

    let found = find_config_for_model_file(&model_path);
    assert!(found.is_none());
}

/// Verify find_config_for_model_file prefers hash-prefixed over generic config.json
#[test]
fn test_find_config_prefers_hash_prefix() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("abc123.safetensors");
    std::fs::write(&model_path, b"fake").expect("write model");
    let hash_config = dir.path().join("abc123.config.json");
    std::fs::write(&hash_config, r#"{"num_hidden_layers": 4}"#).expect("write hash config");
    let generic_config = dir.path().join("config.json");
    std::fs::write(&generic_config, r#"{"num_hidden_layers": 8}"#).expect("write generic config");

    let found = find_config_for_model_file(&model_path);
    assert_eq!(found, Some(hash_config));
}

/// Verify check_safetensors_file_integrity works end-to-end with hash-prefixed config
#[test]
fn test_check_safetensors_file_integrity_hash_config() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("abc123.safetensors");

    // Create minimal safetensors with matching config
    create_mock_safetensors(dir.path(), 4, 128, 32000);
    std::fs::rename(dir.path().join("model.safetensors"), &model_path).expect("rename");
    let config_path = dir.path().join("abc123.config.json");
    std::fs::write(
        &config_path,
        r#"{"num_hidden_layers": 4, "hidden_size": 128, "vocab_size": 32000, "num_attention_heads": 8}"#,
    )
    .expect("write config");

    let result = check_safetensors_file_integrity(&model_path);
    assert!(result.passed);
    assert!(result.config_found);
}

include!("integrity_tests_config_mismatch.rs");
include!("integrity_tests_part_b.rs");
