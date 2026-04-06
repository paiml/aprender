
// =============================================================================
// Config Clone and Debug Tests
// =============================================================================

#[test]
fn test_loader_part02_config_clone() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 256,
        num_layers: 8,
        num_heads: 16,
        num_kv_heads: 4,
        vocab_size: 32000,
        intermediate_dim: 1024,
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
            eos_token_id: None,
    };

    let cloned = config.clone();

    assert_eq!(cloned.architecture, config.architecture);
    assert_eq!(cloned.hidden_dim, config.hidden_dim);
    assert_eq!(cloned.num_layers, config.num_layers);
    assert_eq!(cloned.num_heads, config.num_heads);
    assert_eq!(cloned.num_kv_heads, config.num_kv_heads);
    assert_eq!(cloned.vocab_size, config.vocab_size);
    assert_eq!(cloned.intermediate_dim, config.intermediate_dim);
    assert_eq!(cloned.context_length, config.context_length);
    assert!((cloned.rope_theta - config.rope_theta).abs() < f32::EPSILON);
    assert!((cloned.eps - config.eps).abs() < f32::EPSILON);
    assert_eq!(cloned.rope_type, config.rope_type);
}

#[test]
fn test_loader_part02_config_debug() {
    let config = GGUFConfig {
        architecture: "debug_test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("debug_test"),
        hidden_dim: 512,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 2,
        vocab_size: 50000,
        intermediate_dim: 2048,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-6,
        rope_type: 2,
        bos_token_id: None,
            eos_token_id: None,
    };

    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("debug_test"));
    assert!(debug_str.contains("512"));
    assert!(debug_str.contains("GGUFConfig"));
}
