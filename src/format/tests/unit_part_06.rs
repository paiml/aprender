
#[test]
fn test_gguf_model_config_clone() {
    let cfg = gguf::GgufModelConfig {
        architecture: Some("phi".to_string()),
        hidden_size: Some(2048),
        num_layers: Some(24),
        num_heads: Some(32),
        num_kv_heads: Some(32),
        vocab_size: Some(51200),
        intermediate_size: Some(8192),
        max_position_embeddings: Some(2048),
        rope_theta: Some(10000.0),
        rms_norm_eps: Some(1e-5),
        rope_type: Some(0),
    };
    let cloned = cfg.clone();
    assert_eq!(cloned.architecture, cfg.architecture);
    assert_eq!(cloned.hidden_size, cfg.hidden_size);
}

#[test]
fn test_gguf_model_config_debug() {
    let cfg = gguf::GgufModelConfig::default();
    let debug_str = format!("{cfg:?}");
    assert!(debug_str.contains("GgufModelConfig"));
}
