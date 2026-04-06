
// ============================================================================
// forward_with_cache: GELU model with Q4K fused (no gate weight)
// ============================================================================

#[test]
fn test_fwc_q4k_gelu_no_gate() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: Some(vec![0.0; hidden]), // LayerNorm (has bias)
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: Some(vec![0.0; hidden + 2 * kv_size]),
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: Some(vec![0.0; hidden]),
        ffn_gate_weight: None, // GELU: no gate
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

    // Q4K layers for GELU model (no gate)
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: None, // No gate for GELU
        ffn_up_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_down_weight: Some(q4k_bytes(hidden, intermediate)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "phi2".to_string(),
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
        output_norm_bias: Some(vec![0.0; hidden]),
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: Some(vec![0.0; vocab]),
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K GELU fwc: {}", result.unwrap_err());
}

// ============================================================================
// forward (batch): GELU model with Q4K
// ============================================================================

#[test]
fn test_forward_batch_q4k_gelu() {
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
        ffn_gate_weight: None,
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

    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: None, // No separate — falls through to combined QKV
        attn_k_weight: None,
        attn_v_weight: None,
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: None,
        ffn_up_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_down_weight: Some(q4k_bytes(hidden, intermediate)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
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
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2]);
    assert!(result.is_ok(), "Q4K GELU batch: {}", result.unwrap_err());
}

// ============================================================================
// Env-var-gated paths: REALIZE_TRACE, APR_FORCE_F32, REALIZE_DEBUG
// These tests cover trace/debug/force_f32 conditional blocks that are
// otherwise unreachable without setting environment variables.
// ============================================================================

#[test]
fn test_fwc_with_realize_trace() {
    // REALIZE_TRACE enables eprintln trace blocks in forward_with_cache
    // Safe in parallel: only affects stderr output, not computation
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Trace fwc first: {}", r1.unwrap_err());

    let r2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(r2.is_ok(), "Trace fwc second: {}", r2.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

#[test]
fn test_fwc_force_f32_with_q4k_layers() {
    // APR_FORCE_F32 forces F32 fallback even when Q4K layers exist
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("APR_FORCE_F32", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Force F32 fwc: {}", result.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("APR_FORCE_F32");
    }
}

#[test]
fn test_fwc_force_f32_with_trace() {
    // Both APR_FORCE_F32 and REALIZE_TRACE — covers the force_f32 trace blocks
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("APR_FORCE_F32", "1");
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Force F32 + trace: {}", r1.unwrap_err());

    let r2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(r2.is_ok(), "Force F32 + trace 2nd: {}", r2.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("APR_FORCE_F32");
        std::env::remove_var("REALIZE_TRACE");
    }
}

#[test]
fn test_forward_batch_with_realize_trace() {
    // REALIZE_TRACE for batch forward path
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_TRACE", "1");
    }
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.forward(&[1, 2, 3]);
    assert!(result.is_ok(), "Trace batch: {}", result.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_TRACE");
    }
}

#[test]
fn test_from_apr_bytes_with_realize_debug() {
    // REALIZE_DEBUG enables debug eprintln blocks in from_apr_bytes
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZE_DEBUG", "1");
    }

    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let head_dim = hidden / 4;
    let kv_size = 4 * head_dim;
    let mut tensors = vec![
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
    ];
    // Add Q4K separate weights to trigger Q4K debug path
    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.v_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12,
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors));
    assert!(apr.is_ok(), "Debug from_apr_bytes: {}", apr.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZE_DEBUG");
    }
}

// ============================================================================
// from_apr_bytes: GGUF naming (output.weight, blk.X) triggers is_gguf_model
// ============================================================================

#[test]
fn test_from_apr_bytes_gguf_naming() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let head_dim = hidden / 4;
    let kv_dim = 4 * head_dim; // kv_heads=4

    let tensors = vec![
        // GGUF-style embedding name
        TensorDef {
            name: "token_embd.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // GGUF-style output norm
        TensorDef {
            name: "output_norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // GGUF-style lm_head
        TensorDef {
            name: "output.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // GGUF-style layer norms
        TensorDef {
            name: "blk.0.attn_norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.ffn_norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // GGUF-style QKV (combined)
        TensorDef {
            name: "blk.0.attn_qkv.weight".to_string(),
            dtype: 0,
            dims: vec![(hidden + 2 * kv_dim) as u64, hidden as u64],
            data: make_f32_data((hidden + 2 * kv_dim) * hidden, 0.001),
        },
        // GGUF-style attn output
        TensorDef {
            name: "blk.0.attn_output.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.001),
        },
        // GGUF-style FFN (SwiGLU)
        TensorDef {
            name: "blk.0.ffn_gate.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "blk.0.ffn_up.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "blk.0.ffn_down.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_f32_data(hidden * intermediate, 0.001),
        },
    ];

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors));
    assert!(apr.is_ok(), "GGUF naming: {}", apr.unwrap_err());
    let model = apr.expect("test value should be present");
    assert_eq!(model.config.hidden_dim, hidden);
    assert_eq!(model.config.vocab_size, vocab);
    assert_eq!(model.layers.len(), 1);
}

// ============================================================================
// forward (batch): GELU F32-only (no Q4K) — covers ffn_up_bias/ffn_down_bias
// ============================================================================

#[test]
fn test_forward_batch_gelu_f32_no_q4k() {
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
        q4k_layers: None, // Pure F32 — exercises F32 GELU FFN with bias
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = apr.forward(&[1, 2]);
    assert!(result.is_ok(), "GELU F32 batch: {}", result.unwrap_err());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), vocab);
}
