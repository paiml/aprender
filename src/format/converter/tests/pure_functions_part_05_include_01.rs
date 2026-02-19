
#[test]
fn test_load_config_gpt2_aliases_gh265() {
    // GPT-2 uses n_embd, n_layer, n_head, n_inner, n_positions
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");
    std::fs::write(
        &config_path,
        r#"{
            "model_type": "gpt2",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "n_positions": 1024,
            "vocab_size": 50257
        }"#,
    )
    .expect("write config");

    let config = load_model_config_from_json(&model_path).expect("should parse");
    assert_eq!(config.hidden_size, Some(768));
    assert_eq!(config.num_layers, Some(12));
    assert_eq!(config.num_heads, Some(12));
    assert_eq!(config.intermediate_size, Some(3072));
    assert_eq!(config.max_position_embeddings, Some(1024));
    assert_eq!(config.vocab_size, Some(50257));
}

#[test]
fn test_load_config_bloom_aliases_gh265() {
    // BLOOM uses n_embed (not n_embd!), n_layer, num_attention_heads
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");
    std::fs::write(
        &config_path,
        r#"{
            "model_type": "bloom",
            "n_embed": 1024,
            "n_layer": 24,
            "num_attention_heads": 16,
            "vocab_size": 250880
        }"#,
    )
    .expect("write config");

    let config = load_model_config_from_json(&model_path).expect("should parse");
    assert_eq!(config.hidden_size, Some(1024));
    assert_eq!(config.num_layers, Some(24));
    assert_eq!(config.num_heads, Some(16));
    // intermediate_size falls back to 4 * hidden_size
    assert_eq!(config.intermediate_size, Some(4096));
}

#[test]
fn test_load_config_opt_aliases_gh265() {
    // OPT uses hidden_size, num_hidden_layers, num_attention_heads, ffn_dim
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");
    std::fs::write(
        &config_path,
        r#"{
            "model_type": "opt",
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "ffn_dim": 2048,
            "vocab_size": 50272,
            "max_position_embeddings": 2048
        }"#,
    )
    .expect("write config");

    let config = load_model_config_from_json(&model_path).expect("should parse");
    assert_eq!(config.hidden_size, Some(512));
    assert_eq!(config.num_layers, Some(6));
    assert_eq!(config.num_heads, Some(8));
    assert_eq!(config.intermediate_size, Some(2048));
}

#[test]
fn test_load_config_gpt_neo_aliases_gh265() {
    // GPT-Neo uses hidden_size, num_layers (not num_hidden_layers), num_heads
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, b"dummy").expect("write model");
    std::fs::write(
        &config_path,
        r#"{
            "model_type": "gpt_neo",
            "hidden_size": 768,
            "num_layers": 6,
            "num_heads": 12,
            "vocab_size": 50257,
            "max_position_embeddings": 2048
        }"#,
    )
    .expect("write config");

    let config = load_model_config_from_json(&model_path).expect("should parse");
    assert_eq!(config.hidden_size, Some(768));
    assert_eq!(config.num_layers, Some(6));
    assert_eq!(config.num_heads, Some(12));
    // intermediate_size falls back to 4 * hidden_size
    assert_eq!(config.intermediate_size, Some(3072));
}
