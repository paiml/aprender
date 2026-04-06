
#[test]
fn test_from_apr_bytes_no_lm_head_no_embed_for_tying_fatal() {
    // Use "tok_embeddings.weight" which IS found for embed lookup but NOT tried
    // for weight tying (tying only tries model.embed_tokens.weight & token_embd.weight).
    // No lm_head.weight, no output.weight => hits line 896 error path.
    let hidden = 8;
    let intermediate = 32;
    let vocab = 4;
    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

    let kv_dim = 4 * (hidden / 4);
    let tensors = vec![
        // Embed: "tok_embeddings.weight" - found for embed, NOT for weight tying
        TensorDef {
            name: "tok_embeddings.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // Norm
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // Layer norms
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
        // Must provide QKV + attn_output + FFN for layer construction to succeed
        TensorDef {
            name: "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            dtype: 0,
            dims: vec![(hidden + 2 * kv_dim) as u64, hidden as u64],
            data: make_f32_data((hidden + 2 * kv_dim) * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.self_attn.o_proj.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.mlp.up_proj.weight".to_string(),
            dtype: 0,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.001),
        },
        TensorDef {
            name: "model.layers.0.mlp.down_proj.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_f32_data(hidden * intermediate, 0.001),
        },
    ];

    let data = build_apr_v2(&meta, &tensors);
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_err(),
        "Expected error for no lm_head + no embed for tying"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("FATAL") || err_msg.contains("lm_head"),
        "Error should mention FATAL or lm_head: {err_msg}"
    );
}

// ============================================================================
// from_apr_bytes: Q4K separate tensors → q4k_layers populated
// ============================================================================

#[test]
fn test_from_apr_bytes_separate_q4k_tensors_populates_q4k_layers() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":{},"num_key_value_heads":{},"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, heads, kv_heads, vocab, intermediate
    );

    let mut tensors = vec![
        // Embedding (F32)
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // Output norm (F32)
        TensorDef {
            name: "model.norm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // LM head (F32)
        TensorDef {
            name: "lm_head.weight".to_string(),
            dtype: 0,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
        // Attn norm (F32)
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        // FFN norm (F32)
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".to_string(),
            dtype: 0,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
    ];

    // Add SEPARATE Q4K attention weights
    // Q4K tensors: (name, out_dim, in_dim) — data size = out_dim * ceil(in_dim/256) * 144
    let q4k_tensors: Vec<(&str, usize, usize)> = vec![
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.v_proj.weight", kv_size, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ];

    for (name, out_dim, in_dim) in q4k_tensors {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12, // Q4_K
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Separate Q4K parse failed");

    // Verify q4k_layers is populated
    assert!(apr.q4k_layers.is_some(), "q4k_layers should be Some");
    let q4k = &apr.q4k_layers.as_ref().expect("test value should be present")[0];
    assert!(
        q4k.attn_q_weight.is_some(),
        "attn_q_weight should be populated"
    );
    assert!(
        q4k.attn_k_weight.is_some(),
        "attn_k_weight should be populated"
    );
    assert!(
        q4k.attn_v_weight.is_some(),
        "attn_v_weight should be populated"
    );
    assert!(
        q4k.attn_output_weight.is_some(),
        "attn_output_weight should be populated"
    );
    assert!(
        q4k.ffn_gate_weight.is_some(),
        "ffn_gate_weight should be populated"
    );
    assert!(
        q4k.ffn_up_weight.is_some(),
        "ffn_up_weight should be populated"
    );
    assert!(
        q4k.ffn_down_weight.is_some(),
        "ffn_down_weight should be populated"
    );
}

#[test]
fn test_from_apr_bytes_separate_q4k_then_forward_with_cache() {
    // Build APR with separate Q4K tensors, then run forward_with_cache
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":{},"num_key_value_heads":{},"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, heads, kv_heads, vocab, intermediate
    );

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

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors)).expect("Parse failed");
    assert!(apr.q4k_layers.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let r1 = apr.forward_with_cache(1, &mut cache, 0);
    assert!(r1.is_ok(), "Separate Q4K fwc: {}", r1.unwrap_err());

    let r2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(r2.is_ok(), "Separate Q4K fwc 2nd: {}", r2.unwrap_err());
}

// ============================================================================
// from_apr_bytes: Q6K separate tensors for mixed quant
// ============================================================================

#[test]
fn test_from_apr_bytes_mixed_q4k_q6k_tensors() {
    let hidden = 32;
    let intermediate = 64;
    let vocab = 16;

    let meta = format!(
        r#"{{"hidden_size":{},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":{},"intermediate_size":{},"rms_norm_eps":1e-5,"rope_theta":10000.0,"architecture":"llama","max_position_embeddings":2048}}"#,
        hidden, vocab, intermediate
    );

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

    // Q4K for Q, K, attn_output, gate
    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.q_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.k_proj.weight", hidden, hidden),
        ("model.layers.0.self_attn.o_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.gate_proj.weight", intermediate, hidden),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 12, // Q4_K
            dims: vec![out_dim as u64, in_dim as u64],
            data: q4k_bytes(out_dim, in_dim),
        });
    }

    // Q6K for V, up, down (mixed quantization)
    for (name, out_dim, in_dim) in [
        ("model.layers.0.self_attn.v_proj.weight", hidden, hidden),
        ("model.layers.0.mlp.up_proj.weight", intermediate, hidden),
        ("model.layers.0.mlp.down_proj.weight", hidden, intermediate),
    ] {
        tensors.push(TensorDef {
            name: name.to_string(),
            dtype: 14, // Q6_K
            dims: vec![out_dim as u64, in_dim as u64],
            data: q6k_bytes(out_dim, in_dim),
        });
    }

    let apr = AprTransformer::from_apr_bytes(&build_apr_v2(&meta, &tensors))
        .expect("Mixed Q4K/Q6K parse failed");

    assert!(apr.q4k_layers.is_some());
    let q4k = &apr.q4k_layers.as_ref().expect("test value should be present")[0];
    assert!(q4k.attn_q_weight.is_some());
    assert!(q4k.attn_v_weight.is_none(), "V should NOT be Q4K");
    assert!(q4k.attn_v_weight_q6k.is_some(), "V should be Q6K");
    assert!(q4k.ffn_up_weight.is_none(), "up should NOT be Q4K");
    assert!(q4k.ffn_up_weight_q6k.is_some(), "up should be Q6K");
    assert!(q4k.ffn_down_weight.is_none(), "down should NOT be Q4K");
    assert!(q4k.ffn_down_weight_q6k.is_some(), "down should be Q6K");
}

// ============================================================================
// generate: Q4K model
// ============================================================================

#[test]
fn test_generate_q4k_fused() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.generate(&[1, 2], 3);
    assert!(result.is_ok(), "Q4K generate: {}", result.unwrap_err());
    let tokens = result.expect("test value should be present");
    assert!(tokens.len() >= 2, "Should have at least input tokens");
}

#[test]
fn test_generate_q6k_variants() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let result = apr.generate(&[1], 2);
    assert!(result.is_ok(), "Q6K generate: {}", result.unwrap_err());
}
