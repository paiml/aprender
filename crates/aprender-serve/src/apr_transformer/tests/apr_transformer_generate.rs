
#[test]
fn test_generate_with_cache_with_temperature() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 2,
        temperature: 1.0,
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[0], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// Part 8: Utility Method Tests
// ============================================================================

#[test]
fn test_num_parameters() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let params = transformer.num_parameters();
    assert!(params > 0);

    // Should include embedding, layers, and lm_head
    // Minimum: vocab_size * hidden_dim (embedding)
    assert!(params >= 100 * 64);
}

#[test]
fn test_memory_size() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let size = transformer.memory_size();
    assert!(size > 0);

    // Memory size = num_parameters * 4 bytes
    assert_eq!(size, transformer.num_parameters() * 4);
}

#[test]
fn test_config_accessor() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());

    let returned_config = transformer.config();
    assert_eq!(returned_config.hidden_dim, config.hidden_dim);
    assert_eq!(returned_config.num_layers, config.num_layers);
    assert_eq!(returned_config.vocab_size, config.vocab_size);
}

// ============================================================================
// Part 9: from_apr_bytes() Error Tests
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let data = vec![0u8; 32]; // Less than 64 bytes
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_valid_magic_v1() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0"); // APR v1 magic

    // Set up minimal valid header
    // tensor_count at offset 8
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    // metadata_offset at offset 12
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    // metadata_size at offset 20
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // tensor_index_offset at offset 24
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    // data_offset at offset 32
    data[32..40].copy_from_slice(&66u64.to_le_bytes());

    // Add minimal JSON metadata
    data[64] = b'{';
    data[65] = b'}';

    let result = AprTransformer::from_apr_bytes(&data);
    // Magic check passes (v1), but may fail on subsequent parsing (no tensors).
    // Key assertion: error is NOT about invalid magic.
    if let Err(ref e) = result {
        let msg = format!("{e}");
        assert!(
            !msg.contains("Invalid APR magic"),
            "APR v1 magic should be accepted, got: {msg}"
        );
    }
}

#[test]
fn test_from_apr_bytes_valid_magic_v2() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR2"); // APR v2 magic

    // Set up minimal header
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    data[32..40].copy_from_slice(&66u64.to_le_bytes());

    // Minimal JSON
    data[64] = b'{';
    data[65] = b'}';

    let result = AprTransformer::from_apr_bytes(&data);
    // Magic check passes (v2), but may fail on subsequent parsing (no tensors).
    if let Err(ref e) = result {
        let msg = format!("{e}");
        assert!(
            !msg.contains("Invalid APR magic"),
            "APR v2 magic should be accepted, got: {msg}"
        );
    }
}

#[test]
fn test_from_apr_bytes_metadata_out_of_bounds() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");

    // metadata_offset points beyond file
    data[12..20].copy_from_slice(&1000u64.to_le_bytes());
    data[20..24].copy_from_slice(&100u32.to_le_bytes());

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// Part 10: Layer Tests
// ============================================================================

#[test]
fn test_layer_empty() {
    let layer = AprTransformerLayer::empty(64, 256);

    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
    assert_eq!(layer.attn_output_weight.len(), 64 * 64);
    assert_eq!(layer.ffn_up_weight.len(), 64 * 256);
    assert_eq!(layer.ffn_down_weight.len(), 256 * 64);
}

#[test]
fn test_layer_empty_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let intermediate_dim = 256;

    let layer =
        AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

    // QKV: Q (64) + K (16) + V (16) = 96 elements per hidden dim
    let head_dim = hidden_dim / num_heads; // 8
    let kv_dim = num_kv_heads * head_dim; // 16
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 64 + 32 = 96

    assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
}

#[test]
fn test_layer_num_parameters() {
    let layer = AprTransformerLayer::empty(64, 256);
    let params = layer.num_parameters();

    // Calculate expected
    let expected = 64  // attn_norm
        + 64 * 192  // qkv (3 * hidden)
        + 64 * 64   // attn_output
        + 64 * 256  // ffn_up
        + 256 * 64; // ffn_down

    assert_eq!(params, expected);
}

// ============================================================================
// Part 11: Q4K Layer Weights Tests
// ============================================================================

#[test]
fn test_q4k_layer_weights_default() {
    let weights = Q4KLayerWeights::default();

    assert!(weights.qkv_weight.is_none());
    assert!(weights.attn_q_weight.is_none());
    assert!(weights.attn_k_weight.is_none());
    assert!(weights.attn_v_weight.is_none());
    assert!(weights.attn_output_weight.is_none());
    assert!(weights.ffn_gate_weight.is_none());
    assert!(weights.ffn_up_weight.is_none());
    assert!(weights.ffn_down_weight.is_none());
}

#[test]
fn test_q4k_layer_weights_with_values() {
    let weights = Q4KLayerWeights {
        attn_q_weight: Some(vec![1, 2, 3]),
        attn_k_weight: Some(vec![4, 5, 6]),
        attn_v_weight: Some(vec![7, 8, 9]),
        ffn_down_weight: Some(vec![10, 11, 12]),
        ..Default::default()
    };

    assert!(weights.attn_q_weight.is_some());
    assert_eq!(weights.attn_q_weight.expect("test value should be present")[0], 1);
    assert!(weights.ffn_down_weight.is_some());
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
