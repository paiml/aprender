
#[test]
fn test_generate_with_cache_repetition_penalty() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 5,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.5, // Strong repetition penalty
        trace: false,
        stop_tokens: vec![],
    };

    let result = transformer.generate_with_cache(&[0, 1, 2], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// Part 9: AprKVCache Edge Cases
// ============================================================================

#[test]
fn test_kv_cache_with_gqa_config() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let cache = AprKVCache::new(&config);

    assert_eq!(cache.num_kv_heads(), 2);
    assert_eq!(cache.head_dim(), 128 / 8);
    assert_eq!(cache.capacity(), 64);
}

#[test]
fn test_kv_cache_append_all_layers() {
    let config = create_test_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Append to all layers (last layer auto-advances)
    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }
    // No advance() needed - append() auto-advances on last layer

    // Cache length is counted per position, not per layer
    // After appending once to each layer at position 0
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_kv_cache_multiple_positions_all_layers() {
    let config = create_test_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

    // Simulate 5 positions
    for pos in 0..5 {
        let k = vec![(pos + 1) as f32; kv_size];
        let v = vec![(pos + 10) as f32; kv_size];

        for layer in 0..config.num_layers {
            cache.append(layer, &k, &v);
        }
        // No advance() needed - append() auto-advances on last layer
    }

    assert_eq!(cache.len(), 5);

    // Verify each layer's cache
    for layer in 0..config.num_layers {
        let (k_cache, v_cache) = cache.get(layer);
        assert_eq!(k_cache.len(), 5 * kv_size);
        assert_eq!(v_cache.len(), 5 * kv_size);
    }
}

// ============================================================================
// Part 10: AprTransformerLayer Edge Cases
// ============================================================================

#[test]
fn test_layer_with_all_optional_biases() {
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let mut layer = AprTransformerLayer::empty(hidden_dim, intermediate_dim);

    // Add all optional biases
    layer.attn_norm_bias = Some(vec![0.01; hidden_dim]);
    layer.qkv_bias = Some(vec![0.01; hidden_dim * 3]);
    layer.attn_output_bias = Some(vec![0.01; hidden_dim]);
    layer.ffn_gate_weight = Some(vec![0.1; hidden_dim * intermediate_dim]);
    layer.ffn_gate_bias = Some(vec![0.01; intermediate_dim]);
    layer.ffn_up_bias = Some(vec![0.01; intermediate_dim]);
    layer.ffn_down_bias = Some(vec![0.01; hidden_dim]);
    layer.ffn_norm_weight = Some(vec![1.0; hidden_dim]);
    layer.ffn_norm_bias = Some(vec![0.01; hidden_dim]);

    // Count parameters should include all optionals
    let params = layer.num_parameters();
    assert!(params > hidden_dim * 3 * hidden_dim); // More than just qkv
}

#[test]
fn test_layer_empty_gqa_various_ratios() {
    // Test various GQA ratios
    for (num_heads, num_kv_heads) in [(8, 4), (8, 2), (16, 4), (32, 8)] {
        let hidden_dim = num_heads * 16; // head_dim = 16
        let intermediate_dim = hidden_dim * 4;

        let layer =
            AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
    }
}

// ============================================================================
// Part 11: Forward Pass with Various Configurations
// ============================================================================

#[test]
fn test_forward_single_layer_single_head() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_many_layers() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 12, // Many layers
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_many_layers() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 8,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    for pos in 0..10 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 12: Predict Next Edge Cases
// ============================================================================

#[test]
fn test_predict_next_returns_valid_token() {
    let config = create_test_config();
    let vocab_size = config.vocab_size;
    let transformer = AprTransformer::new(config);

    for tokens in [vec![0], vec![0, 1], vec![0, 1, 2, 3, 4]] {
        let result = transformer.predict_next(&tokens);
        assert!(result.is_ok());
        assert!((result.expect("test value should be present") as usize) < vocab_size);
    }
}

#[test]
fn test_predict_next_deterministic() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    // Same input should produce same output (no randomness in predict_next)
    let r1 = transformer.predict_next(&[0, 1, 2]).expect("test value should be present");
    let r2 = transformer.predict_next(&[0, 1, 2]).expect("test value should be present");
    assert_eq!(r1, r2);
}

// ============================================================================
// Part 13: Num Parameters and Memory Size
// ============================================================================

#[test]
fn test_num_parameters_with_optionals() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    let base_params = transformer.num_parameters();

    // Add optional biases
    transformer.output_norm_bias = Some(vec![0.0; config.hidden_dim]);
    transformer.lm_head_bias = Some(vec![0.0; config.vocab_size]);

    let with_optionals = transformer.num_parameters();

    // Should be larger with optionals
    assert!(with_optionals > base_params);
    assert_eq!(
        with_optionals - base_params,
        config.hidden_dim + config.vocab_size
    );
}

#[test]
fn test_memory_size_consistency() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let params = transformer.num_parameters();
    let memory = transformer.memory_size();

    // Memory = params * 4 (f32 = 4 bytes)
    assert_eq!(memory, params * 4);
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
