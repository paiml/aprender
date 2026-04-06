
// =============================================================================
// Architecture Detection Tests
// =============================================================================

#[test]
fn test_architecture_detection_llama() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);

    // LLaMA contract: SwiGLU (gate weight), RMSNorm (no bias)
    let layer = &model.layers[0];
    assert!(
        layer.ffn_gate_weight.is_some(),
        "LLaMA should have gate weight (SwiGLU contract)"
    );
    assert!(
        layer.attn_norm_bias.is_none(),
        "LLaMA should not have norm bias (RMSNorm contract)"
    );
}

#[test]
fn test_architecture_detection_phi2() {
    let model = create_phi2_style_model(100, 64, 128, 4, 1);

    // phi-2 detection: ffn_gate_weight.is_none() || attn_norm_bias.is_some()
    let layer = &model.layers[0];
    assert!(
        layer.ffn_gate_weight.is_none(),
        "phi-2 should not have gate weight"
    );
    assert!(
        layer.attn_norm_bias.is_some(),
        "phi-2 should have norm bias"
    );
}

// =============================================================================
// KV Cache Integration Tests
// =============================================================================

#[test]
fn test_kv_cache_grows_correctly() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    assert_eq!(cache.len(), 0, "Cache should start empty");

    model.forward_cached(10, &mut cache, 0).expect("expected value");
    assert_eq!(
        cache.len(),
        1,
        "Cache should have 1 entry after first token"
    );

    model.forward_cached(20, &mut cache, 1).expect("expected value");
    assert_eq!(
        cache.len(),
        2,
        "Cache should have 2 entries after second token"
    );
}

#[test]
fn test_forward_vs_forward_cached_first_token() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    // Both should produce logits of same dimension
    let logits_forward = model.forward(&[42u32]).expect("logits_forward");
    let logits_cached = model.forward_cached(42, &mut cache, 0).expect("logits_cached");

    assert_eq!(logits_forward.len(), logits_cached.len());
    // Note: Values may differ slightly due to different code paths,
    // but both should be finite and reasonable
    assert!(logits_forward.iter().all(|x| x.is_finite()));
    assert!(logits_cached.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Separate QKV Weights Tests
// =============================================================================

#[test]
fn test_forward_with_separate_qkv() {
    let hidden_dim = 64;
    let head_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    // Create separate Q, K, V weights
    let q_weight = create_q4k_test_tensor(hidden_dim, hidden_dim);
    let k_weight = create_q4k_test_tensor(hidden_dim, kv_dim);
    let v_weight = create_q4k_test_tensor(hidden_dim, kv_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: create_q4k_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_tensor(hidden_dim, 128),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_tensor(128, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_tensor(hidden_dim, 128)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![0.1f32; 100 * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_tensor(hidden_dim, 100),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let logits = model.forward(&[42u32]).expect("logits");
    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Additional forward_cached coverage tests (130 uncov → target key branches)
// =============================================================================

#[test]
fn test_forward_cached_phi2_style() {
    let model = create_phi2_style_model(100, 64, 128, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    // phi2 path: LayerNorm + GELU (no gate weight, has norm bias)
    let logits = model
        .forward_cached(42, &mut cache, 0)
        .expect("phi2 forward_cached should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_forward_cached_sequential_5_tokens() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    // Decode 5 tokens sequentially
    for pos in 0..5 {
        let token = (pos as u32) + 1;
        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("sequential forward_cached should succeed");

        assert_eq!(logits.len(), 100);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_forward_cached_multi_layer() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 3);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model
        .forward_cached(10, &mut cache, 0)
        .expect("multi-layer forward_cached should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_cached_gqa() {
    // GQA: 8 Q heads, 2 KV heads
    let model = create_llama_style_model(100, 64, 128, 8, 2, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    // Two sequential tokens to exercise GQA cached attention
    let _ = model.forward_cached(5, &mut cache, 0).expect("_");
    let logits = model.forward_cached(10, &mut cache, 1).expect("logits");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_forward_cached_separate_qkv() {
    let hidden_dim = 64;
    let head_dim = 16;
    let num_kv_heads = 2;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: create_q4k_test_tensor(hidden_dim, hidden_dim),
            k: create_q4k_test_tensor(hidden_dim, kv_dim),
            v: create_q4k_test_tensor(hidden_dim, kv_dim),
        },
        qkv_bias: None,
        attn_output_weight: create_q4k_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_tensor(hidden_dim, 128),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_tensor(128, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_tensor(hidden_dim, 128)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; 100 * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_tensor(hidden_dim, 100),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);

    let logits = model
        .forward_cached(42, &mut cache, 0)
        .expect("separate QKV forward_cached should succeed");
    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Trace/debug env-var-gated coverage tests
// These set CPU_DEBUG_LAYERS and CPU_DEBUG to cover the trace/debug blocks
// in forward() and forward_cached().
// =============================================================================

#[test]
fn test_forward_with_cpu_debug_layers_llama() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("CPU_DEBUG_LAYERS", "1");
    }
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);

    let result = model.forward(&[5, 10, 15]);
    assert!(result.is_ok(), "debug forward: {}", result.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("CPU_DEBUG_LAYERS");
    }
}

#[test]
fn test_forward_with_cpu_debug_layers_phi() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("CPU_DEBUG_LAYERS", "1");
    }
    let model = create_phi2_style_model(100, 64, 128, 4, 1);

    let result = model.forward(&[5, 10]);
    assert!(result.is_ok(), "debug phi forward: {}", result.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("CPU_DEBUG_LAYERS");
    }
}

#[test]
fn test_forward_cached_with_cpu_debug() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("CPU_DEBUG", "1");
    }
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    for i in 0..5 {
        let r = model.forward_cached(5, &mut cache, i);
        assert!(r.is_ok(), "debug cached pos {}: {}", i, r.unwrap_err());
    }
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("CPU_DEBUG");
    }
}

#[test]
fn test_forward_cached_with_cpu_debug_phi() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("CPU_DEBUG", "1");
    }
    let model = create_phi2_style_model(100, 64, 128, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    for i in 0..5 {
        let r = model.forward_cached(10, &mut cache, i);
        assert!(r.is_ok(), "debug cached phi pos {}: {}", i, r.unwrap_err());
    }
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("CPU_DEBUG");
    }
}
