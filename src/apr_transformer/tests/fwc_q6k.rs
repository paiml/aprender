
// ============================================================================
// forward_with_cache: Q6K lm_head (different from Q4K lm_head path)
// ============================================================================

#[test]
fn test_fwc_q6k_lm_head_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.lm_head_weight_q6k = Some(q6k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q6K lm_head + trace: {}",
        result.unwrap_err()
    );
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward_with_cache: Q4K lm_head with trace
// ============================================================================

#[test]
fn test_fwc_q4k_lm_head_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.lm_head_weight_q4k = Some(q4k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K lm_head + trace: {}",
        result.unwrap_err()
    );
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward_with_cache: Q6K variants with trace (covers Q6K trace blocks)
// ============================================================================

#[test]
fn test_fwc_q6k_variants_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Q6K variants + trace: {}", r1.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward_with_cache: partial Q4K (some fields None) + trace
// Covers the F32 fallback trace blocks INSIDE the `if !force_f32` branch
// (e.g., "attn_output using F32 fallback (slow!)")
// ============================================================================

#[test]
fn test_fwc_partial_q4k_f32_fallback_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
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
    };

    // Partial Q4K: only Q and K populated, everything else None
    // This forces F32 fallback for V, attn_output, gate, up, down
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: None,      // F32 fallback
        attn_v_weight_q6k: None,  // no Q6K either
        attn_output_weight: None, // F32 fallback
        ffn_gate_weight: None,    // F32 fallback
        ffn_up_weight: None,      // F32 fallback
        ffn_down_weight: None,    // F32 fallback
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let mut cache = AprKVCache::new(&apr.config);
    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Partial Q4K + trace: {}", r1.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward (batch): out-of-vocab token + debug traces
// ============================================================================

#[test]
fn test_forward_batch_oov_token_with_debug() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_DEBUG", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    // Token 999 is way out of vocab (vocab=16)
    let result = apr.forward(&[999]);
    assert!(result.is_ok(), "OOV batch: {}", result.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_DEBUG");
    }
}

// ============================================================================
// forward_with_cache: GELU model (no gate) F32-only + trace
// Covers GELU branches in forward_with_cache with trace enabled
// ============================================================================

#[test]
fn test_fwc_gelu_f32_no_q4k_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.001; 3 * hidden * hidden],
        qkv_bias: Some(vec![0.0; 3 * hidden]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
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
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi2".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "GELU F32 + trace: {}", result.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward (batch): GELU + trace (covers batch GELU trace paths)
// ============================================================================

#[test]
fn test_forward_batch_gelu_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]),
        qkv_weight: vec![0.001; 3 * hidden * hidden],
        qkv_bias: Some(vec![0.0; 3 * hidden]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: Some(vec![0.0; intermediate]),
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: Some(vec![0.0; hidden]),
        ffn_norm_weight: Some(vec![1.0; hidden]),
        ffn_norm_bias: Some(vec![0.0; hidden]),
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
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi2".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2]);
    assert!(
        result.is_ok(),
        "GELU batch + trace: {}",
        result.unwrap_err()
    );
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

// ============================================================================
// forward (batch): partial Q4K + trace (batch forward F32 fallback trace)
// ============================================================================

#[test]
fn test_forward_batch_partial_q4k_with_trace() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
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
    };

    // Only attn_output Q4K populated — all others fall through to F32
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: None,
        attn_k_weight: None,
        attn_v_weight: None,
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: None,
        ffn_up_weight: None,
        ffn_down_weight: None,
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2, 3]);
    assert!(
        result.is_ok(),
        "Partial Q4K batch + trace: {}",
        result.unwrap_err()
    );
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}
