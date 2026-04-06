
#[test]
fn test_validated_transformer_rejects_wrong_ffn_gate_weight() {
    let mut t = make_valid_transformer(1);
    if let Some(ref mut w) = t.layers[0].ffn_gate_weight {
        for v in w.iter_mut() {
            *v = 0.0;
        }
    }
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_ffn_up_weight() {
    let mut t = make_valid_transformer(1);
    let len = t.layers[0].ffn_up_weight.len();
    t.layers[0].ffn_up_weight = vec![0.0; len];
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_ffn_down_weight() {
    let mut t = make_valid_transformer(1);
    let len = t.layers[0].ffn_down_weight.len();
    t.layers[0].ffn_down_weight = vec![0.0; len];
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_config_accessor() {
    let t = make_valid_transformer(2);
    let validated = ValidatedAprTransformer::validate(t).expect("should pass");
    assert_eq!(validated.config().hidden_dim, 16);
    assert_eq!(validated.config().num_layers, 2);
}

#[test]
fn test_validated_transformer_with_zero_num_heads() {
    // Edge case: num_heads = 0 should use hidden_dim as head_dim fallback
    let mut t = make_valid_transformer(1);
    t.config.num_heads = 0;
    t.config.num_kv_heads = 0;
    // This will fail on shape mismatch since QKV dimensions change
    let result = ValidatedAprTransformer::validate(t);
    // Should error due to shape mismatch, not panic
    assert!(result.is_err());
}

// ============================================================================
// Helpers
// ============================================================================

fn make_valid_transformer(num_layers: usize) -> AprTransformer {
    let hidden_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 4;
    let vocab_size = 32;
    let intermediate_dim = 64;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let make_data = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect()
    };

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-6,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: make_data(vocab_size * hidden_dim),
        layers: (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: make_data(qkv_out_dim * hidden_dim),
                qkv_bias: None,
                attn_output_weight: make_data(hidden_dim * hidden_dim),
                attn_output_bias: None,
                ffn_gate_weight: Some(make_data(intermediate_dim * hidden_dim)),
                ffn_gate_bias: None,
                ffn_up_weight: make_data(intermediate_dim * hidden_dim),
                ffn_up_bias: None,
                ffn_down_weight: make_data(hidden_dim * intermediate_dim),
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; hidden_dim]),
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
            })
            .collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: make_data(vocab_size * hidden_dim),
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}
