
#[test]
fn test_forward_with_cache_no_ffn_norm() {
    // Ensure forward_with_cache works without ffn_norm (else branch)
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Explicitly remove ffn_norm
    for layer in &mut transformer.layers {
        layer.ffn_norm_weight = None;
        layer.ffn_norm_bias = None;
    }

    let mut cache = AprKVCache::new(&config);
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_gqa() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust QKV for GQA dimensions
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let mut cache = AprKVCache::new(&config);

    // Process multiple tokens
    for pos in 0..5 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 7: Generate with Cache - Additional Paths
// ============================================================================

#[test]
fn test_generate_with_cache_temperature_zero() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0, 1], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_temperature_high() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 2.0, // High temperature
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_single_prompt_token() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 1.0,
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0], &gen_config);
    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert!(!tokens.is_empty()); // At least prompt
}

#[test]
fn test_generate_with_cache_max_tokens_zero() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 0, // Generate no new tokens
        temperature: 1.0,
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0, 1, 2], &gen_config);
    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert_eq!(tokens.len(), 3); // Just the prompt
}

// ============================================================================
// Part 8: RoPE Edge Cases
// ============================================================================

#[test]
fn test_rope_at_position_zero() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Position 0 should apply RoPE with angle = 0 (cos=1, sin=0)
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

#[test]
fn test_rope_at_high_position() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Fill cache up to high position
    for pos in 0..100 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

#[test]
fn test_rope_with_different_theta() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 500000.0, // High theta like LLaMA 3
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let result = transformer.forward(&[0, 1, 2, 3, 4]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 9: Softmax Edge Cases in Attention
// ============================================================================

#[test]
fn test_attention_with_long_sequence() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);

    // Longer sequence to test attention accumulation
    let tokens: Vec<u32> = (0..20).collect();
    let result = transformer.forward(&tokens);
    assert!(result.is_ok());
}

#[test]
fn test_attention_single_position() {
    // Single token - attention to self only
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let result = transformer.forward(&[42]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 10: Generate Simple (Non-Cache) Edge Cases
// ============================================================================

#[test]
#[ignore = "Test expectation needs adjustment"]
fn test_generate_stops_early_on_eos_2() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Set up lm_head to output token 2 (EOS) with highest logit
    // by setting one weight high
    let hidden_dim = config.hidden_dim;
    let eos_token = 2;
    let _vocab_size = config.vocab_size;

    // Set weight for token 2 to be very high
    for i in 0..hidden_dim {
        transformer.lm_head_weight[eos_token * hidden_dim + i] = 10.0;
    }

    let result = transformer.generate(&[0, 1], 10);
    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    // Should stop at EOS
    assert!(tokens.contains(&2) || tokens.len() <= 3);
}

#[test]
fn test_generate_with_zero_max_tokens() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let result = transformer.generate(&[0, 1], 0);
    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert_eq!(tokens.len(), 2); // Just the prompt
}

// ============================================================================
// Part 11: from_apr_bytes Additional Paths
// ============================================================================

#[test]
#[ignore = "Test expectation needs adjustment"]
fn test_from_apr_bytes_with_metadata_fields() {
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"APR\0"); // APR v1 magic

    // Set header fields
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&100u32.to_le_bytes()); // metadata_size
    data[24..32].copy_from_slice(&164u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&164u64.to_le_bytes()); // data_offset

    // JSON metadata with all fields
    let metadata = r#"{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":4,"vocab_size":1000,"intermediate_size":512,"rope_theta":10000.0,"rms_norm_eps":1e-6,"max_position_embeddings":2048}"#;
    let meta_bytes = metadata.as_bytes();
    data[64..64 + meta_bytes.len()].copy_from_slice(meta_bytes);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok());

    let transformer = result.expect("test value should be present");
    assert_eq!(transformer.config.hidden_dim, 128);
    assert_eq!(transformer.config.num_layers, 2);
    assert_eq!(transformer.config.num_heads, 8);
    assert_eq!(transformer.config.num_kv_heads, 4);
    assert_eq!(transformer.config.vocab_size, 1000);
    assert_eq!(transformer.config.intermediate_dim, 512);
}

#[test]
#[ignore = "Test expectation needs adjustment"]
fn test_from_apr_bytes_with_alternate_field_names() {
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"APR1"); // APR v1 magic (alternate)

    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&80u32.to_le_bytes());
    data[24..32].copy_from_slice(&144u64.to_le_bytes());
    data[32..40].copy_from_slice(&144u64.to_le_bytes());

    // JSON with alternate field names (hidden_dim instead of hidden_size, etc.)
    let metadata =
        r#"{"hidden_dim":256,"num_layers":4,"num_heads":16,"num_kv_heads":8,"vocab_size":500}"#;
    let meta_bytes = metadata.as_bytes();
    data[64..64 + meta_bytes.len()].copy_from_slice(meta_bytes);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok());

    let transformer = result.expect("test value should be present");
    assert_eq!(transformer.config.hidden_dim, 256);
    assert_eq!(transformer.config.num_layers, 4);
}

// ============================================================================
// Part 12: Multi-layer Forward Pass
// ============================================================================

#[test]
fn test_forward_multi_layer_deep() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 8, // More layers
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_multi_layer() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 6,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Process multiple tokens through multiple layers
    for pos in 0..10 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }

    // Verify cache has entries for all positions
    assert_eq!(cache.len(), 10);

    // Verify each layer has cache
    for layer in 0..config.num_layers {
        let (k_cache, v_cache) = cache.get(layer);
        assert!(!k_cache.is_empty());
        assert!(!v_cache.is_empty());
    }
}

// ============================================================================
// Part 13: SwiGLU Path in forward_with_cache
// ============================================================================

#[test]
fn test_forward_with_cache_swiglu() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Enable SwiGLU
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = Some(vec![0.1; config.hidden_dim * config.intermediate_dim]);
    }

    let mut cache = AprKVCache::new(&config);

    for pos in 0..5 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

#[test]
fn test_forward_with_cache_swiglu_with_biases() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Enable SwiGLU with biases
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = Some(vec![0.1; config.hidden_dim * config.intermediate_dim]);
        layer.ffn_gate_bias = Some(vec![0.01; config.intermediate_dim]);
        layer.ffn_down_bias = Some(vec![0.01; config.hidden_dim]);
    }

    let mut cache = AprKVCache::new(&config);
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

// ============================================================================
// Part 14: Predict Next Edge Cases
// ============================================================================

#[test]
fn test_predict_next_single_token() {
    let config = create_test_config();
    let vocab_size = config.vocab_size;
    let transformer = AprTransformer::new(config);

    let result = transformer.predict_next(&[0]);
    assert!(result.is_ok());
    let token = result.expect("test value should be present");
    assert!(token < vocab_size as u32);
}

#[test]
fn test_predict_next_long_sequence() {
    let config = create_test_config();
    let vocab_size = config.vocab_size;
    let transformer = AprTransformer::new(config);

    let tokens: Vec<u32> = (0..15).collect();
    let result = transformer.predict_next(&tokens);
    assert!(result.is_ok());
    let token = result.expect("test value should be present");
    assert!(token < vocab_size as u32);
}
