
// ============================================================================
// Part 15: GELU Activation Coverage
// ============================================================================

#[test]
fn test_forward_standard_mlp_gelu_path() {
    // Ensure GELU path is taken (no gate weight)
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Explicitly no gate weight
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = None;
    }

    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_standard_mlp_gelu_path() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // No gate weight - uses GELU path
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = None;
    }

    let mut cache = AprKVCache::new(&config);
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    }
}
