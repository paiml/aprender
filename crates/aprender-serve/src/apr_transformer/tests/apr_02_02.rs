
// ============================================================================
// Fused QKV bias path
// ============================================================================

#[test]
fn test_from_apr_bytes_fused_qkv_bias() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Add a fused QKV bias tensor
    let qkv_out = hidden + 2 * hidden; // heads=4, kv_heads=4 => kv_dim=hidden
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64],
        data: make_f32_data(qkv_out, 0.0),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Fused QKV bias: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert!(apr.layers[0].qkv_bias.is_some());
    assert_eq!(apr.layers[0].qkv_bias.as_ref().expect("test value should be present").len(), qkv_out);
}

// ============================================================================
// Separate Q/K/V bias path
// ============================================================================

#[test]
fn test_from_apr_bytes_separate_qkv_biases() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let kv_dim = hidden; // kv_heads=4, head_dim=2
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Add separate Q/K/V biases
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.q_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![hidden as u64],
        data: make_f32_data(hidden, 0.1),
    });
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.k_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![kv_dim as u64],
        data: make_f32_data(kv_dim, 0.2),
    });
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.v_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![kv_dim as u64],
        data: make_f32_data(kv_dim, 0.3),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "Separate QKV biases: {}",
        result.unwrap_err()
    );

    let apr = result.expect("test value should be present");
    let bias = apr.layers[0].qkv_bias.as_ref().expect("test value should be present");
    assert_eq!(bias.len(), hidden + kv_dim + kv_dim);
    // Q bias = 0.1
    assert!((bias[0] - 0.1).abs() < 1e-6);
    // K bias = 0.2
    assert!((bias[hidden] - 0.2).abs() < 1e-6);
    // V bias = 0.3
    assert!((bias[hidden + kv_dim] - 0.3).abs() < 1e-6);
}

// ============================================================================
// Fused QKV weight path (single combined tensor)
// ============================================================================

#[test]
fn test_from_apr_bytes_fused_qkv_weight() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let kv_dim = hidden; // kv_heads=4
    let qkv_out = hidden + 2 * kv_dim;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Remove separate Q/K/V and add fused QKV
    tensors.retain(|t| {
        !t.name.contains("q_proj.weight")
            && !t.name.contains("k_proj.weight")
            && !t.name.contains("v_proj.weight")
    });
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64, hidden as u64],
        data: make_f32_data(qkv_out * hidden, 0.02),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Fused QKV: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.layers[0].qkv_weight.len(), qkv_out * hidden);
}

// ============================================================================
// GGUF naming with Q4K weights (tests get_q4k_raw_bytes with GGUF prefixes)
// ============================================================================

#[test]
fn test_from_apr_bytes_gguf_q4k_weights() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);

    let tensors = vec![
        TensorDef {
            name: "model.embed_tokens.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.1),
        },
        TensorDef {
            name: "blk.0.attn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.attn_q.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.attn_k.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.attn_v.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.attn_output.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, hidden as u64],
            data: make_q4k_data(hidden * hidden),
        },
        TensorDef {
            name: "blk.0.ffn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.ffn_gate.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_q4k_data(intermediate * hidden),
        },
        TensorDef {
            name: "blk.0.ffn_up.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_q4k_data(intermediate * hidden),
        },
        TensorDef {
            name: "blk.0.ffn_down.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_q4k_data(hidden * intermediate),
        },
        TensorDef {
            name: "output_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "output.weight".into(),
            dtype: DTYPE_Q4_K,
            dims: vec![vocab as u64, hidden as u64],
            data: make_q4k_data(vocab * hidden),
        },
    ];
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "GGUF Q4K: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert!(apr.q4k_layers.is_some());
    let q4k = apr.q4k_layers.as_ref().expect("test value should be present");
    assert!(q4k[0].attn_q_weight.is_some());
    assert!(q4k[0].attn_k_weight.is_some());
    assert!(q4k[0].attn_output_weight.is_some());
    assert!(q4k[0].ffn_gate_weight.is_some());
    assert!(q4k[0].ffn_up_weight.is_some());
    assert!(q4k[0].ffn_down_weight.is_some());
    assert!(apr.lm_head_weight_q4k.is_some());
}

// ============================================================================
// Multi-layer model
// ============================================================================

#[test]
fn test_from_apr_bytes_multi_layer() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let num_layers = 3;
    let meta = minimal_metadata(hidden, num_layers, 4, 4, vocab, intermediate);

    let mut tensors = vec![TensorDef {
        name: "model.embed_tokens.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![vocab as u64, hidden as u64],
        data: make_f32_data(vocab * hidden, 0.1),
    }];

    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");
        tensors.extend(vec![
            TensorDef {
                name: format!("{prefix}.input_layernorm.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64],
                data: make_f32_data(hidden, 1.0),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.q_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.k_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.v_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.self_attn.o_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, hidden as u64],
                data: make_f32_data(hidden * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.post_attention_layernorm.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64],
                data: make_f32_data(hidden, 1.0),
            },
            TensorDef {
                name: format!("{prefix}.mlp.up_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![intermediate as u64, hidden as u64],
                data: make_f32_data(intermediate * hidden, 0.01),
            },
            TensorDef {
                name: format!("{prefix}.mlp.down_proj.weight"),
                dtype: DTYPE_F32,
                dims: vec![hidden as u64, intermediate as u64],
                data: make_f32_data(hidden * intermediate, 0.01),
            },
        ]);
    }

    tensors.push(TensorDef {
        name: "model.norm.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![hidden as u64],
        data: make_f32_data(hidden, 1.0),
    });
    tensors.push(TensorDef {
        name: "lm_head.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![vocab as u64, hidden as u64],
        data: make_f32_data(vocab * hidden, 0.01),
    });
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Multi-layer: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.layers.len(), num_layers);
    assert_eq!(apr.config.num_layers, num_layers);
}

// ============================================================================
// GQA model (num_kv_heads < num_heads)
// ============================================================================

#[test]
fn test_from_apr_bytes_gqa_model() {
    let hidden = 16;
    let intermediate = 32;
    let vocab = 8;
    let heads = 4;
    let kv_heads = 2;
    let head_dim = hidden / heads; // 4
    let kv_dim = kv_heads * head_dim; // 8
    let meta = minimal_metadata(hidden, 1, heads, kv_heads, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        heads,
        kv_heads,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "GQA: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.config.num_kv_heads, kv_heads);
    // QKV weight should be hidden + 2*kv_dim = 16 + 2*8 = 32 rows * hidden=16 cols = 512
    assert_eq!(
        apr.layers[0].qkv_weight.len(),
        (hidden + 2 * kv_dim) * hidden
    );
}

// ============================================================================
// forward_with_cache coverage (AprTransformer::forward_with_cache)
// ============================================================================

/// Build a minimal executable F32 AprTransformer via from_apr_bytes
fn build_f32_apr(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);
    AprTransformer::from_apr_bytes(&data).expect("F32 APR build failed")
}

/// Build a minimal executable Q4K AprTransformer (with q4k_layers)
fn build_q4k_apr(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);
    AprTransformer::from_apr_bytes(&data).expect("Q4K APR build failed")
}

use crate::apr_transformer::AprKVCache;

#[test]
fn test_forward_with_cache_f32_first_token() {
    let apr = build_f32_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // First token — exercises cache_len == 0 path (V used directly)
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "F32 cache first token: {}",
        result.unwrap_err()
    );
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 16); // vocab_size
    assert!(logits.iter().all(|x| x.is_finite()));
}
