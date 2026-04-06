
#[test]
fn test_forward_with_cache_f32_multi_token() {
    let apr = build_f32_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // First token (cache_len == 0)
    let _ = apr.forward_with_cache(1, &mut cache, 0).expect("test value should be present");

    // Second token — exercises cache_len > 0 path (full attention with cache)
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "F32 cache second token: {}",
        result.unwrap_err()
    );
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_with_cache_f32_three_tokens() {
    let apr = build_f32_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // Multiple tokens to exercise the full attention score computation
    for pos in 0..3 {
        let result = apr.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(
            result.is_ok(),
            "F32 cache token {pos}: {}",
            result.unwrap_err()
        );
    }
}

#[test]
fn test_forward_with_cache_q4k_first_token() {
    // Q4K weights exercise the fused kernel paths in forward_with_cache
    let apr = build_q4k_apr(256, 256, 4);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K cache first token: {}",
        result.unwrap_err()
    );
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 4);
}

#[test]
fn test_forward_with_cache_q4k_multi_token() {
    let apr = build_q4k_apr(256, 256, 4);
    let mut cache = AprKVCache::new(&apr.config);

    // Exercise both cache_len == 0 and cache_len > 0 paths with Q4K
    let _ = apr.forward_with_cache(1, &mut cache, 0).expect("test value should be present");
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "Q4K cache second token: {}",
        result.unwrap_err()
    );
}

#[test]
fn test_forward_with_cache_q6k_weights() {
    // Q6K weights for v_proj, ffn_down, ffn_up exercise the Q6K matmul paths
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_F32,
    );
    // Override specific tensors to Q6_K
    for t in &mut tensors {
        if t.name.contains("v_proj") || t.name.contains("down_proj") || t.name.contains("up_proj") {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q6K APR build failed");
    let mut cache = AprKVCache::new(&apr.config);

    // First token exercises Q6K V projection and Q6K FFN paths
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q6K cache first token: {}",
        result.unwrap_err()
    );
}

#[test]
fn test_forward_with_cache_with_qkv_bias() {
    // Model with QKV biases exercises the bias-add paths in forward_with_cache
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
    // Add fused QKV bias
    let qkv_out = hidden + 2 * kv_dim;
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64],
        data: make_f32_data(qkv_out, 0.01),
    });
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("biased APR build failed");
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Biased cache: {}", result.unwrap_err());
}

// ============================================================================
// AprTransformer::forward coverage (non-cached full sequence)
// ============================================================================

#[test]
fn test_forward_f32_single_token() {
    let apr = build_f32_apr(8, 32, 16);
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Forward single: {}", result.unwrap_err());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_forward_f32_multi_token() {
    let apr = build_f32_apr(8, 32, 16);
    let result = apr.forward(&[1, 2, 3]);
    assert!(result.is_ok(), "Forward multi: {}", result.unwrap_err());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_forward_q4k_single_token() {
    let apr = build_q4k_apr(256, 256, 4);
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Q4K forward: {}", result.unwrap_err());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 4);
}

// ============================================================================
// AprTransformer::generate coverage
// ============================================================================

#[test]
fn test_generate_f32_greedy() {
    let apr = build_f32_apr(8, 32, 16);
    let result = apr.generate(&[1], 3);
    assert!(result.is_ok(), "Generate: {}", result.unwrap_err());
    let tokens = result.expect("test value should be present");
    assert!(tokens.len() >= 2); // at least prompt + 1 generated
    assert!(tokens.len() <= 4); // prompt + max 3 tokens
    assert_eq!(tokens[0], 1); // prompt preserved
}

// ============================================================================
// AprTransformer helper functions coverage
// ============================================================================

#[test]
fn test_apr_transformer_config_accessor() {
    let apr = build_f32_apr(8, 32, 16);
    let config = apr.config();
    assert_eq!(config.hidden_dim, 8);
    assert_eq!(config.intermediate_dim, 32);
    assert_eq!(config.vocab_size, 16);
}

#[test]
fn test_apr_transformer_num_parameters() {
    let apr = build_f32_apr(8, 32, 16);
    let params = apr.num_parameters();
    // Should be > 0
    assert!(params > 0);
    // Rough check: at minimum vocab*hidden*2 (embed + lm_head)
    assert!(params >= 16 * 8 * 2);
}

// ============================================================================
// Standard MLP (GELU, no gate) path coverage
// ============================================================================

/// Build a model WITHOUT ffn_gate_weight (GELU/standard MLP path, like phi-2)
fn build_gelu_apr(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
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
    // Remove gate_proj to make the model use standard MLP (GELU) instead of SwiGLU
    tensors.retain(|t| !t.name.contains("gate_proj"));
    let data = build_apr_v2(&meta, &tensors);
    AprTransformer::from_apr_bytes(&data).expect("GELU APR build failed")
}

#[test]
fn test_forward_gelu_model_single_token() {
    let apr = build_gelu_apr(8, 32, 16);
    // Verify no gate weight
    assert!(apr.layers[0].ffn_gate_weight.is_none());
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "GELU forward: {}", result.unwrap_err());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_forward_with_cache_gelu_model() {
    let apr = build_gelu_apr(8, 32, 16);
    let mut cache = AprKVCache::new(&apr.config);

    // First token (GELU + cache_len == 0)
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "GELU cache first: {}", result.unwrap_err());

    // Second token (GELU + cache_len > 0)
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(result.is_ok(), "GELU cache second: {}", result.unwrap_err());
}

// ============================================================================
// No ffn_norm path (hidden passed directly to FFN)
// ============================================================================

#[test]
fn test_forward_with_cache_no_ffn_norm() {
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
    // Remove post_attention_layernorm to exercise no-ffn-norm path
    tensors.retain(|t| !t.name.contains("post_attention_layernorm"));
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("no-ffn-norm APR build failed");
    assert!(apr.layers[0].ffn_norm_weight.is_none());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "No FFN norm: {}", result.unwrap_err());
}

// ============================================================================
// Q4K fused kernel with QKV bias in forward_with_cache
// ============================================================================

#[test]
fn test_forward_with_cache_q4k_with_bias() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
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
        DTYPE_Q4_K,
        DTYPE_F32,
    );
    // Add QKV bias (exercises the bias-split path when q4k_layer.is_some())
    tensors.push(TensorDef {
        name: "model.layers.0.self_attn.qkv_proj.bias".into(),
        dtype: DTYPE_F32,
        dims: vec![qkv_out as u64],
        data: make_f32_data(qkv_out, 0.01),
    });
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q4K+bias APR build failed");
    assert!(apr.q4k_layers.is_some());
    assert!(apr.layers[0].qkv_bias.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K+bias cache: {}", result.unwrap_err());
}

// ============================================================================
// Q4K lm_head in forward_with_cache
// ============================================================================

#[test]
fn test_forward_with_cache_q4k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q4K lm_head APR build failed");
    assert!(apr.lm_head_weight_q4k.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K lm_head cache: {}", result.unwrap_err());
}

#[test]
fn test_forward_with_cache_q6k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256;
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
    // Set lm_head to Q6_K
    for t in &mut tensors {
        if t.name == "lm_head.weight" {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);
    let apr = AprTransformer::from_apr_bytes(&data).expect("Q6K lm_head APR build failed");
    assert!(apr.lm_head_weight_q6k.is_some());

    let mut cache = AprKVCache::new(&apr.config);
    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q6K lm_head cache: {}", result.unwrap_err());
}
