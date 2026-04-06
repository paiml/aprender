
// =============================================================================
// GQA (Grouped Query Attention) tests
// =============================================================================

#[test]
fn test_forward_single_gqa_config() {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    // Test GQA: num_kv_heads < num_heads (e.g., 4 Q heads share 2 KV heads)
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 2, // GQA: 2 KV heads for 4 Q heads
        num_layers: 1,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // Q is full, K/V reduced

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, config.intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(config.intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, config.intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);

    // First token (no cache)
    let logits1 = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("GQA forward (first token) should succeed");
    assert_eq!(logits1.len(), config.vocab_size);

    // Second token (uses cache with GQA)
    let logits2 = model
        .forward_single_with_cache(2, &mut cache, 1)
        .expect("GQA forward (cached) should succeed");
    assert_eq!(logits2.len(), config.vocab_size);
}

// =============================================================================
// Model with multiple layers
// =============================================================================

#[test]
fn test_forward_single_multiple_layers() {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 3, // 3 layers
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let hidden_dim = config.hidden_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layers: Vec<OwnedQuantizedLayer> = (0..config.num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
            qkv_bias: None,
            attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_test_data(hidden_dim, config.intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_test_data(config.intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, config.intermediate_dim)),
            ffn_gate_bias: None,
            ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        position_embedding: None,
        layers,
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
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
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Multi-layer forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// Additional forward_single_with_cache coverage (178 uncov)
// =============================================================================

#[test]
fn test_forward_single_phi_style_with_cache() {
    // phi-style: LayerNorm + GELU (no gate, has bias)
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let logits = model
        .forward_single_with_cache(42, &mut cache, 0)
        .expect("phi-style forward_single_with_cache should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_sequential_decode_10_tokens() {
    // Exercise the cached attention path with many positions
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    for pos in 0..10 {
        let token = (pos as u32) % (config.vocab_size as u32);
        let logits = model
            .forward_single_with_cache(token, &mut cache, pos)
            .expect("sequential decode should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_forward_single_with_scratch_multi_step() {
    // Exercise scratch buffer through multiple decode steps
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for pos in 0..5 {
        model
            .forward_single_with_scratch((pos as u32) + 1, &mut cache, pos, &mut scratch)
            .expect("multi-step scratch forward should succeed");

        assert_eq!(scratch.logits.len(), config.vocab_size);
        assert!(scratch.logits.iter().all(|&x| x.is_finite()));
    }
}

// =============================================================================
// forward_single_with_cache_adaptive tests (127 uncov, gpu feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_forward_single_adaptive_basic() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let metrics = Arc::new(DispatchMetrics::new());

    let logits = model
        .forward_single_with_cache_adaptive(42, &mut cache, 0, &metrics)
        .expect("adaptive forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[cfg(feature = "gpu")]
#[test]
fn test_forward_single_adaptive_sequential() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let metrics = Arc::new(DispatchMetrics::new());

    for pos in 0..5 {
        let token = (pos as u32) + 1;
        let logits = model
            .forward_single_with_cache_adaptive(token, &mut cache, pos, &metrics)
            .expect("sequential adaptive forward should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&x| x.is_finite()));
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_forward_single_adaptive_phi_style() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let metrics = Arc::new(DispatchMetrics::new());

    let logits = model
        .forward_single_with_cache_adaptive(10, &mut cache, 0, &metrics)
        .expect("phi-style adaptive forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// Trace/debug env-var-gated coverage tests
// These set REALIZAR_DEBUG_FORWARD and APR_TRACE_LAYERS to cover the
// trace/debug blocks in forward_single_with_cache and forward_single_with_scratch.
// =============================================================================

#[test]
fn test_forward_single_with_cache_debug_traces_llama() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // First token (position 0, layer 0) — triggers most trace blocks
    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Debug traces llama fwd: {}", r1.unwrap_err());

    // Second token
    let r2 = model.forward_single_with_cache(10, &mut cache, 1);
    assert!(r2.is_ok(), "Debug traces llama 2nd: {}", r2.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

#[test]
fn test_forward_single_with_cache_debug_traces_phi() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Debug traces phi fwd: {}", r1.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

#[test]
fn test_forward_single_with_cache_debug_traces_multi_token() {
    // Run multiple tokens with debug tracing to cover more trace branches
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    for i in 0..5 {
        let r = model.forward_single_with_cache(i as u32, &mut cache, i);
        assert!(r.is_ok(), "Debug multi pos {}: {}", i, r.unwrap_err());
    }
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}

// =============================================================================
// forward_single_with_scratch tests (zero-allocation variant)
// =============================================================================

#[test]
fn test_forward_single_with_scratch_llama_basic() {
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Scratch llama first: {}", r1.unwrap_err());
    assert!(
        scratch.logits.iter().any(|x| *x != 0.0),
        "logits should be non-zero"
    );

    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Scratch llama second: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_scratch_phi_basic() {
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Scratch phi first: {}", r1.unwrap_err());
    assert!(
        scratch.logits.iter().any(|x| *x != 0.0),
        "phi logits should be non-zero"
    );
}

#[test]
fn test_forward_single_with_scratch_multi_token() {
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for i in 0..10 {
        let r = model.forward_single_with_scratch(i as u32, &mut cache, i, &mut scratch);
        assert!(r.is_ok(), "Scratch multi pos {}: {}", i, r.unwrap_err());
    }
}

#[test]
fn test_forward_single_with_scratch_debug_traces() {
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("APR_TRACE_LAYERS", "1");
    }
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Scratch debug: {}", r1.unwrap_err());

    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Scratch debug 2: {}", r2.unwrap_err());
    // SAFETY: Preconditions verified by caller or enclosing context
    unsafe {
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("APR_TRACE_LAYERS");
    }
}
