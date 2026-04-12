/// Verify integrity check passes when all config values match tensor dimensions
#[test]
fn test_integrity_check_all_match() {
    let dir = TempDir::new().expect("create temp dir");
    create_test_config(dir.path(), 24, 896, 151_936);
    create_mock_safetensors(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(
        result.passed,
        "Should pass when all values match: {:?}",
        result.errors
    );
    assert!(result.config_found);
    assert!(result.layer_count_match);
    assert!(result.hidden_size_match);
    assert!(result.vocab_size_match);
    assert!(result.errors.is_empty());
}

/// Verify integrity check fails when config layer count differs from tensors
#[test]
fn test_integrity_check_layer_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says 14 layers but tensors have 24
    create_test_config(dir.path(), 14, 896, 151_936);
    create_mock_safetensors(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed, "Should fail on layer mismatch");
    assert!(!result.layer_count_match);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-LAYERS"))
    );
}

/// Verify integrity check fails when config hidden_size differs from tensors
#[test]
fn test_integrity_check_hidden_size_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says hidden=4096 but tensors have 896
    create_test_config(dir.path(), 24, 4096, 151_936);
    create_mock_safetensors(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed, "Should fail on hidden_size mismatch");
    assert!(!result.hidden_size_match);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-HIDDEN"))
    );
}

/// Verify integrity check fails when config vocab_size differs from tensors
#[test]
fn test_integrity_check_vocab_size_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says vocab=896 (corrupted) but tensors have 151_936
    create_test_config(dir.path(), 24, 896, 896);
    create_mock_safetensors(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed, "Should fail on vocab_size mismatch");
    assert!(!result.vocab_size_match);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-VOCAB"))
    );
}

/// Verify integrity check fails when config.json is missing
#[test]
fn test_integrity_check_missing_config() {
    let dir = TempDir::new().expect("create temp dir");
    // No config.json, only safetensors
    create_mock_safetensors(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed, "Should fail when config.json missing");
    assert!(!result.config_found);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-CONFIG"))
    );
}

/// Verify integrity check fails when no .safetensors files are present
#[test]
fn test_integrity_check_no_safetensors() {
    let dir = TempDir::new().expect("create temp dir");
    // Only config.json, no safetensors files
    create_test_config(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed, "Should fail when no .safetensors files");
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("No .safetensors files"))
    );
}

/// Verify integrity check reports all three errors when layers, hidden, and vocab mismatch
#[test]
fn test_integrity_check_multiple_mismatches() {
    let dir = TempDir::new().expect("create temp dir");
    // All values wrong (the corrupted config case)
    create_test_config(dir.path(), 14, 4096, 896);
    create_mock_safetensors(dir.path(), 24, 896, 151_936);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed, "Should fail on multiple mismatches");
    assert!(!result.layer_count_match);
    assert!(!result.hidden_size_match);
    assert!(!result.vocab_size_match);
    assert_eq!(result.errors.len(), 3, "Should have 3 error messages");
}

/// Verify extract_layer_number parses layer indices from various tensor name patterns
#[test]
fn test_extract_layer_number() {
    assert_eq!(
        extract_layer_number("model.layers.23.self_attn.q_proj.weight"),
        Some(23)
    );
    assert_eq!(
        extract_layer_number("layers.0.mlp.gate_proj.weight"),
        Some(0)
    );
    assert_eq!(extract_layer_number("h.15.attn.c_attn.weight"), Some(15));
    assert_eq!(extract_layer_number("transformer.h.7.mlp.weight"), Some(7));
    assert_eq!(extract_layer_number("model.embed_tokens.weight"), None);
    assert_eq!(extract_layer_number("lm_head.weight"), None);
}

/// Verify ConfigValues serializes to JSON correctly
#[test]
fn test_config_values_serialization() {
    let values = ConfigValues {
        num_hidden_layers: Some(24),
        hidden_size: Some(896),
        vocab_size: Some(151_936),
        num_attention_heads: Some(14),
    };
    let json = serde_json::to_string(&values).expect("serialize");
    assert!(json.contains("24"));
    assert!(json.contains("896"));
}

/// Verify TensorDerivedValues serializes to JSON correctly
#[test]
fn test_tensor_derived_values_serialization() {
    let values = TensorDerivedValues {
        layer_count: Some(24),
        hidden_size: Some(896),
        vocab_size: Some(151_936),
    };
    let json = serde_json::to_string(&values).expect("serialize");
    assert!(json.contains("24"));
    assert!(json.contains("151936"));
}

/// Verify IntegrityResult serializes to JSON with error messages
#[test]
fn test_integrity_result_serialization() {
    let result = IntegrityResult {
        passed: false,
        config_found: true,
        layer_count_match: false,
        hidden_size_match: true,
        vocab_size_match: true,
        errors: vec!["G0-INTEGRITY-LAYERS: mismatch".to_string()],
        config_values: None,
        tensor_values: None,
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("G0-INTEGRITY-LAYERS"));
}

/// Verify gate_ids module exposes correct constant strings for each gate
#[test]
fn test_gate_ids() {
    assert_eq!(gate_ids::CONFIG, "G0-INTEGRITY-CONFIG");
    assert_eq!(gate_ids::LAYERS, "G0-INTEGRITY-LAYERS");
    assert_eq!(gate_ids::HIDDEN, "G0-INTEGRITY-HIDDEN");
    assert_eq!(gate_ids::VOCAB, "G0-INTEGRITY-VOCAB");
}

/// Verify IntegrityResult Debug implementation
#[test]
fn test_integrity_result_debug() {
    let result = IntegrityResult {
        passed: true,
        config_found: true,
        layer_count_match: true,
        hidden_size_match: true,
        vocab_size_match: true,
        errors: vec![],
        config_values: None,
        tensor_values: None,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("IntegrityResult"));
}

/// Verify ConfigValues Debug implementation
#[test]
fn test_config_values_debug() {
    let values = ConfigValues {
        num_hidden_layers: Some(24),
        hidden_size: Some(896),
        vocab_size: Some(151_936),
        num_attention_heads: Some(14),
    };
    let debug_str = format!("{values:?}");
    assert!(debug_str.contains("ConfigValues"));
}

/// Verify TensorDerivedValues Debug implementation
#[test]
fn test_tensor_derived_values_debug() {
    let values = TensorDerivedValues {
        layer_count: Some(24),
        hidden_size: Some(896),
        vocab_size: Some(151_936),
    };
    let debug_str = format!("{values:?}");
    assert!(debug_str.contains("TensorDerivedValues"));
}

/// Verify IntegrityResult clone preserves all fields including nested values
#[test]
fn test_integrity_result_clone() {
    let result = IntegrityResult {
        passed: true,
        config_found: true,
        layer_count_match: true,
        hidden_size_match: true,
        vocab_size_match: true,
        errors: vec!["test".to_string()],
        config_values: Some(ConfigValues {
            num_hidden_layers: Some(24),
            hidden_size: Some(896),
            vocab_size: Some(151_936),
            num_attention_heads: Some(14),
        }),
        tensor_values: Some(TensorDerivedValues {
            layer_count: Some(24),
            hidden_size: Some(896),
            vocab_size: Some(151_936),
        }),
    };
    let cloned = result.clone();
    assert_eq!(cloned.passed, result.passed);
    assert_eq!(cloned.errors.len(), result.errors.len());
}

// =========================================================================
// Additional coverage tests for uncovered paths
// =========================================================================

/// Verify read_safetensors_metadata returns error for a corrupted file
#[test]
fn test_read_safetensors_corrupted_file() {
    let dir = TempDir::new().expect("create temp dir");
    // Write a file that's too short to contain a valid header
    let path = dir.path().join("corrupt.safetensors");
    std::fs::write(&path, b"short").expect("write corrupt");
    let result = read_safetensors_metadata(&path);
    assert!(result.is_err());
}

/// Verify read_safetensors_metadata rejects headers exceeding MAX_HEADER_SIZE
#[test]
fn test_read_safetensors_oversized_header() {
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("oversize.safetensors");
    let mut file = std::fs::File::create(&path).expect("create file");
    // Header length of 200MB (exceeds MAX_HEADER_SIZE)
    let huge: u64 = 200_000_000;
    file.write_all(&huge.to_le_bytes()).expect("write len");
    drop(file);
    let result = read_safetensors_metadata(&path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("exceeds maximum"));
}

/// Verify read_safetensors_metadata skips __metadata__ key and returns real tensors
#[test]
fn test_read_safetensors_with_metadata_key() {
    let dir = TempDir::new().expect("create temp dir");
    // Create safetensors header that includes __metadata__ key
    let mut header_obj = serde_json::Map::new();

    // Add __metadata__ key (should be skipped)
    header_obj.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );

    // Add a real tensor
    let mut tensor_info = serde_json::Map::new();
    tensor_info.insert("shape".to_string(), serde_json::json!([100, 50]));
    tensor_info.insert(
        "dtype".to_string(),
        serde_json::Value::String("F32".to_string()),
    );
    tensor_info.insert("data_offsets".to_string(), serde_json::json!([0, 20000]));
    header_obj.insert(
        "model.weight".to_string(),
        serde_json::Value::Object(tensor_info),
    );

    let header_json = serde_json::to_string(&header_obj).expect("serialize header");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let path = dir.path().join("model.safetensors");
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(&header_len.to_le_bytes())
        .expect("write len");
    file.write_all(header_bytes).expect("write header");
    file.write_all(&[0u8; 128]).expect("write data padding");
    drop(file);

    let tensors = read_safetensors_metadata(&path).expect("should parse");
    // __metadata__ should NOT appear as a tensor
    assert!(!tensors.contains_key("__metadata__"));
    // But model.weight should
    assert!(tensors.contains_key("model.weight"));
    assert_eq!(tensors["model.weight"], vec![100, 50]);
}

/// Verify derive_values_from_tensors falls back to lm_head.weight for vocab/hidden
#[test]
fn test_derive_values_from_lm_head_fallback() {
    // No embed_tokens, only lm_head.weight — exercises the fallback path
    let mut tensors = HashMap::new();
    tensors.insert("lm_head.weight".to_string(), vec![32000, 4096]);
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![4096, 4096],
    );
    tensors.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        vec![4096, 4096],
    );

    let values = derive_values_from_tensors(&tensors);
    assert_eq!(values.vocab_size, Some(32000));
    assert_eq!(values.hidden_size, Some(4096));
    assert_eq!(values.layer_count, Some(2));
}

/// Verify derive_values_from_tensors uses model.lm_head.weight as secondary fallback
#[test]
fn test_derive_values_model_lm_head_fallback() {
    // No embed_tokens, uses model.lm_head.weight
    let mut tensors = HashMap::new();
    tensors.insert("model.lm_head.weight".to_string(), vec![50_000, 768]);

    let values = derive_values_from_tensors(&tensors);
    assert_eq!(values.vocab_size, Some(50_000));
    assert_eq!(values.hidden_size, Some(768));
}

/// Verify check_safetensors_integrity handles corrupt safetensors file gracefully
#[test]
fn test_check_safetensors_integrity_read_error() {
    let dir = TempDir::new().expect("create temp dir");
    create_test_config(dir.path(), 12, 768, 30_000);

    // Create a corrupt safetensors file (too short for header)
    let path = dir.path().join("model.safetensors");
    std::fs::write(&path, b"bad").expect("write corrupt");

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-CONFIG"))
    );
}

/// Verify check_safetensors_integrity detects hidden_size mismatch
#[test]
fn test_check_safetensors_integrity_hidden_size_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says hidden_size=1024 but tensor has 768
    create_test_config(dir.path(), 2, 1024, 30_000);
    create_mock_safetensors(dir.path(), 2, 768, 30_000);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed);
    assert!(result.errors.iter().any(|e| e.contains("HIDDEN")));
}

/// Verify check_safetensors_integrity detects vocab_size mismatch
#[test]
fn test_check_safetensors_integrity_vocab_size_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says vocab=50000 but tensor has 30000
    create_test_config(dir.path(), 2, 768, 50_000);
    create_mock_safetensors(dir.path(), 2, 768, 30_000);

    let result = check_safetensors_integrity(dir.path());
    assert!(!result.passed);
    assert!(result.errors.iter().any(|e| e.contains("VOCAB")));
}

// =========================================================================
// File-mode integrity tests (pacha cache with hash-prefixed files)
// =========================================================================

/// Create a named config JSON file in the given directory for hash-prefix testing
fn create_named_config(dir: &Path, name: &str, layers: usize, hidden: usize, vocab: usize) {
    let config = format!(
        r#"{{
            "num_hidden_layers": {layers},
            "hidden_size": {hidden},
            "vocab_size": {vocab},
            "num_attention_heads": 12
        }}"#
    );
    std::fs::write(dir.join(name), config).expect("write config");
}

/// Create a named safetensors file with given dimensions for hash-prefix testing
fn create_named_safetensors(
    dir: &Path,
    name: &str,
    layers: usize,
    hidden: usize,
    vocab: usize,
) {
    let mut header_obj = serde_json::Map::new();

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

    let path = dir.join(name);
    let mut file = File::create(path).expect("create safetensors");
    file.write_all(&header_len.to_le_bytes())
        .expect("write len");
    file.write_all(header_bytes).expect("write header");
    file.write_all(&[0u8; 1024]).expect("write data");
}

