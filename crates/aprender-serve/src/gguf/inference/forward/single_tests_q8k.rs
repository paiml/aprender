
// =============================================================================
// Q8K-optimized path (requires hidden_dim % 256 == 0)
// =============================================================================

/// Create a llama model with hidden_dim=256 to trigger Q8K paths
fn create_llama_256_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 4,
        num_kv_heads: 4,
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
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    OwnedQuantizedModel {
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
    }
}

#[test]
fn test_forward_single_with_scratch_q8k_path() {
    // hidden_dim=256 => Q8K-optimized path enabled
    let model = create_llama_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Q8K scratch: {}", r1.unwrap_err());
    assert!(scratch.logits.len() == config.vocab_size);

    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Q8K scratch 2nd: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_cache_q8k_path() {
    // Q8K path also used in forward_single_with_cache when hidden_dim % 256 == 0
    let model = create_llama_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Q8K cache: {}", r1.unwrap_err());
    assert_eq!(r1.expect("r1").len(), config.vocab_size);

    let r2 = model.forward_single_with_cache(10, &mut cache, 1);
    assert!(r2.is_ok(), "Q8K cache 2nd: {}", r2.unwrap_err());
}

/// Create a phi-style model with hidden_dim=256 to trigger Q8K GELU paths
fn create_phi_256_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 4,
        num_kv_heads: 4,
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
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: Some(vec![0.0f32; hidden_dim]), // LayerNorm
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: Some(vec![0.0f32; qkv_out_dim]),
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
        ffn_up_bias: Some(vec![0.0f32; intermediate_dim]),
        ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
        ffn_down_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_gate_weight: None, // GELU, no gate
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: Some(vec![0.0f32; hidden_dim]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; config.vocab_size * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: Some(vec![0.0f32; hidden_dim]),
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: Some(vec![0.0f32; config.vocab_size]),
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a GQA llama model with hidden_dim=256 (num_kv_heads < num_heads)
fn create_llama_256_gqa_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 8,    // 8 query heads
        num_kv_heads: 2, // 2 KV heads => GQA group_size=4
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
    let intermediate_dim = config.intermediate_dim;
    let head_dim = hidden_dim / config.num_heads; // 32
    let kv_dim = config.num_kv_heads * head_dim; // 64
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 256 + 128 = 384

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    OwnedQuantizedModel {
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
    }
}

#[test]
fn test_forward_single_with_scratch_phi_256_q8k_gelu() {
    // Phi model with hidden_dim=256: covers Q8K GELU up/down paths in scratch
    let model = create_phi_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "Phi-256 scratch first: {}", r1.unwrap_err());

    // Second token covers attention_with_cache path
    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "Phi-256 scratch second: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_cache_phi_256_q8k() {
    // Phi 256 through the cache path for Q8K GELU coverage
    let model = create_phi_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let r1 = model.forward_single_with_cache(5, &mut cache, 0);
    assert!(r1.is_ok(), "Phi-256 cache first: {}", r1.unwrap_err());

    let r2 = model.forward_single_with_cache(10, &mut cache, 1);
    assert!(r2.is_ok(), "Phi-256 cache second: {}", r2.unwrap_err());
}

#[test]
fn test_forward_single_with_scratch_gqa_256() {
    // GQA model (num_kv_heads=2, num_heads=8) with hidden_dim=256
    // Covers: GQA V-expansion on first token, attention_with_cache_gqa on subsequent
    let model = create_llama_256_gqa_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    // First token: covers GQA V-expansion branch in scratch
    let r1 = model.forward_single_with_scratch(5, &mut cache, 0, &mut scratch);
    assert!(r1.is_ok(), "GQA-256 scratch first: {}", r1.unwrap_err());

    // Second token: covers attention_with_cache_gqa_into branch
    let r2 = model.forward_single_with_scratch(10, &mut cache, 1, &mut scratch);
    assert!(r2.is_ok(), "GQA-256 scratch second: {}", r2.unwrap_err());

    // Third token: further exercises cached attention
    let r3 = model.forward_single_with_scratch(15, &mut cache, 2, &mut scratch);
    assert!(r3.is_ok(), "GQA-256 scratch third: {}", r3.unwrap_err());
}

#[test]
fn test_forward_single_with_cache_gqa_256() {
    // GQA model through cache path
    let model = create_llama_256_gqa_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    for i in 0..5 {
        let r = model.forward_single_with_cache(i as u32, &mut cache, i);
        assert!(r.is_ok(), "GQA-256 cache pos {}: {}", i, r.unwrap_err());
    }
}

#[test]
fn test_forward_single_with_scratch_q8k_multi_position() {
    // Multiple positions through Q8K path to exercise cached attention
    let model = create_llama_256_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for i in 0..10 {
        let r = model.forward_single_with_scratch(i as u32, &mut cache, i, &mut scratch);
        assert!(r.is_ok(), "Q8K multi pos {}: {}", i, r.unwrap_err());
    }
}
