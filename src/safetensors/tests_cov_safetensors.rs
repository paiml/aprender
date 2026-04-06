
// ========== Additional SafetensorsConfig coverage ==========

#[test]
fn test_cov_safetensors_config_load_from_sibling_invalid_json() {
    // Create a temp dir with an invalid config.json
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, "not valid json").expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path);
    assert!(config.is_none());
}

#[test]
fn test_cov_safetensors_config_load_from_sibling_valid() {
    // Create a temp dir with valid config.json
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    let config_json = r#"{
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 32000,
            "model_type": "llama"
        }"#;
    std::fs::write(&config_path, config_json).expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path);
    assert!(config.is_some());
    let config = config.expect("config");
    assert_eq!(config.hidden_size, Some(768));
    assert_eq!(config.num_hidden_layers, Some(12));
    assert_eq!(config.num_attention_heads, Some(12));
    assert_eq!(config.vocab_size, Some(32000));
    assert_eq!(config.model_type, Some("llama".to_string()));
}

#[test]
fn test_cov_safetensors_config_serde_aliases() {
    // Test that serde aliases work (n_embd, n_layer, n_head, etc.)
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    let config_json = r#"{
            "n_embd": 512,
            "n_layer": 6,
            "n_head": 8,
            "n_inner": 2048,
            "n_positions": 1024
        }"#;
    std::fs::write(&config_path, config_json).expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path);
    assert!(config.is_some());
    let config = config.expect("config");
    assert_eq!(config.hidden_size, Some(512)); // n_embd alias
    assert_eq!(config.num_hidden_layers, Some(6)); // n_layer alias
    assert_eq!(config.num_attention_heads, Some(8)); // n_head alias
    assert_eq!(config.intermediate_size, Some(2048)); // n_inner alias
    assert_eq!(config.max_position_embeddings, Some(1024)); // n_positions alias
}

#[test]
fn test_cov_safetensors_config_all_fields() {
    // Test config with all optional fields present
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    let config_json = r#"{
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "intermediate_size": 3072,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;
    std::fs::write(&config_path, config_json).expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");

    assert_eq!(config.hidden_size, Some(768));
    assert_eq!(config.num_hidden_layers, Some(12));
    assert_eq!(config.num_attention_heads, Some(12));
    assert_eq!(config.num_key_value_heads, Some(4));
    assert_eq!(config.vocab_size, Some(32000));
    assert_eq!(config.intermediate_size, Some(3072));
    assert_eq!(config.max_position_embeddings, Some(2048));
    assert!((config.rms_norm_eps.expect("eps") - 1e-5).abs() < 1e-10);
    assert!((config.rope_theta.expect("theta") - 10000.0).abs() < 1e-3);
    assert_eq!(
        config.architectures,
        Some(vec!["LlamaForCausalLM".to_string()])
    );
    assert_eq!(config.model_type, Some("llama".to_string()));
    assert_eq!(config.bos_token_id, Some(1));
    assert_eq!(config.eos_token_id, Some(2));
}

#[test]
fn test_cov_safetensors_config_architecture_empty_list() {
    // Test architecture() with empty architectures list
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: Some(vec![]), // Empty list
        model_type: Some("gpt2".to_string()),
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    // Should fall back to model_type when architectures is empty
    assert_eq!(config.architecture(), "gpt2");
}

// ========== Additional SafetensorsTensorInfo tests ==========

#[test]
fn test_cov_tensor_info_clone_and_eq() {
    let info1 = SafetensorsTensorInfo {
        name: "weight".to_string(),
        dtype: SafetensorsDtype::F32,
        shape: vec![2, 3],
        data_offsets: [0, 24],
    };

    let info2 = info1.clone();
    assert_eq!(info1, info2);

    let info3 = SafetensorsTensorInfo {
        name: "weight".to_string(),
        dtype: SafetensorsDtype::F16, // Different dtype
        shape: vec![2, 3],
        data_offsets: [0, 24],
    };
    assert_ne!(info1, info3);
}

#[test]
fn test_cov_safetensors_dtype_clone_and_eq() {
    let dtype1 = SafetensorsDtype::F32;
    let dtype2 = dtype1.clone();
    assert_eq!(dtype1, dtype2);

    let dtype3 = SafetensorsDtype::F16;
    assert_ne!(dtype1, dtype3);
}
