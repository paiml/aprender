
// ============================================================================
// from_apr_bytes: Successful deserialization with layers
// ============================================================================

#[test]
fn test_from_apr_bytes_preserves_layer_data() {
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 16,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.5; 80],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.1; 8 * 24],
            qkv_bias: None,
            attn_output_weight: vec![0.2; 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.3; 128]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.4; 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.5; 128],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 8]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.6; 80],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("should serialize");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("should deserialize");

    // Verify layer count
    assert_eq!(restored.layers.len(), 1);

    // Verify token embedding values
    assert!((restored.token_embedding[0] - 0.5).abs() < 1e-6);

    // Verify layer weight values
    assert!((restored.layers[0].attn_norm_weight[0] - 1.0).abs() < 1e-6);
    assert!((restored.layers[0].qkv_weight[0] - 0.1).abs() < 1e-6);
    assert!((restored.layers[0].attn_output_weight[0] - 0.2).abs() < 1e-6);

    // Verify lm_head values
    assert!((restored.lm_head_weight[0] - 0.6).abs() < 1e-6);
}

// ============================================================================
// from_apr_bytes: Error path - truncated tensor index
// ============================================================================

#[test]
fn test_from_apr_bytes_truncated_at_tensor_index() {
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 8,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 16,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.0; 80],
        layers: vec![],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 80],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("should serialize");

    // Truncate right after the tensor index starts (at tensor_index_offset + 5 bytes)
    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().expect("slice")) as usize;
    let truncated = &bytes[..tensor_index_offset + 5];

    let result = GgufToAprConverter::from_apr_bytes(truncated);
    assert!(
        result.is_err(),
        "Should fail when tensor index is truncated"
    );
}

// ============================================================================
// GgufToAprConverter::stats with non-trivial model
// ============================================================================

#[test]
fn test_stats_with_converted_model() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let transformer = GgufToAprConverter::convert(&gguf_data).expect("should convert");
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.architecture, "llama");
    assert_eq!(stats.hidden_dim, 64);
    assert_eq!(stats.vocab_size, 32);
    assert_eq!(stats.num_layers, 1);
    assert!(stats.total_parameters > 0);
    assert!(stats.memory_bytes_f32 > 0);
    assert!(stats.memory_mb() > 0.0);
    assert!(stats.parameters_m() > 0.0);
}
