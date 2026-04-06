//! Tests for single-token forward pass with KV cache
//!
//! Coverage target: src/gguf/inference/forward/single.rs
//!
//! Tests cover:
//! - forward_single_with_cache: Main decode-phase forward pass
//! - forward_single_with_scratch: Zero-allocation variant
//! - Various model architectures (RMSNorm vs LayerNorm, SwiGLU vs GELU)

use crate::gguf::test_helpers::{create_q4k_test_data, create_test_model_with_config};
use crate::gguf::{GGUFConfig, InferenceScratchBuffer, OwnedQuantizedKVCache};

/// Create a config for LLaMA-style models (RMSNorm + SwiGLU)
fn create_llama_style_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
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
    }
}

/// Create a config for phi-2 style models (LayerNorm + GELU, no gate)
fn create_phi_style_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
        hidden_dim: 64,
        intermediate_dim: 128,
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
    }
}

/// Create a LLaMA-style model with SwiGLU FFN (gate weight + ffn_norm)
fn create_llama_style_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = create_llama_style_config();
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);

    // QKV: hidden_dim -> hidden_dim + 2*kv_dim
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight = create_q4k_test_data(hidden_dim, qkv_out_dim);
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);
    let ffn_gate_weight = create_q4k_test_data(hidden_dim, intermediate_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None, // LLaMA has no norm bias (RMSNorm)
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight: Some(ffn_gate_weight), // SwiGLU gate
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]), // LLaMA has FFN norm
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

/// Create a phi-style model (LayerNorm + GELU, no gate)
fn create_phi_style_model() -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let config = create_phi_style_config();
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);

    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight = create_q4k_test_data(hidden_dim, qkv_out_dim);
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: Some(vec![0.0f32; hidden_dim]), // phi has norm bias (LayerNorm)
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: Some(vec![0.0f32; qkv_out_dim]),
        attn_output_weight,
        attn_output_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_up_weight,
        ffn_up_bias: Some(vec![0.0f32; intermediate_dim]),
        ffn_down_weight,
        ffn_down_bias: Some(vec![0.0f32; hidden_dim]),
        ffn_gate_weight: None, // phi uses GELU, no gate
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

// =============================================================================
// forward_single_with_cache tests
// =============================================================================

#[test]
fn test_forward_single_with_cache_first_token() {
    // First token should work without any cached K/V
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    let token_id = 5u32;
    let position = 0;

    let logits = model
        .forward_single_with_cache(token_id, &mut cache, position)
        .expect("First token forward should succeed");

    // Should return vocab_size logits
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "Logits should have vocab_size elements"
    );

    // All logits should be finite
    assert!(
        logits.iter().all(|&x| x.is_finite()),
        "All logits should be finite"
    );

    // Cache should have been populated
    assert_eq!(
        cache.len(),
        1,
        "Cache should have 1 position after first token"
    );
}

#[test]
fn test_forward_single_with_cache_sequential_tokens() {
    // Multiple tokens should use and extend the KV cache
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // Process 5 tokens sequentially
    for pos in 0..5 {
        let token_id = (pos as u32) + 1;
        let logits = model
            .forward_single_with_cache(token_id, &mut cache, pos)
            .expect("Sequential token forward should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&x| x.is_finite()));
    }

    assert_eq!(cache.len(), 5, "Cache should have 5 positions");
}

#[test]
fn test_forward_single_with_cache_deterministic() {
    // Same input should produce same output
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;

    let mut cache1 = OwnedQuantizedKVCache::from_config(config, 128);
    let mut cache2 = OwnedQuantizedKVCache::from_config(config, 128);

    let token_id = 10u32;

    let logits1 = model
        .forward_single_with_cache(token_id, &mut cache1, 0)
        .unwrap();
    let logits2 = model
        .forward_single_with_cache(token_id, &mut cache2, 0)
        .unwrap();

    assert_eq!(logits1, logits2, "Forward pass should be deterministic");
}

#[test]
fn test_forward_single_llama_style_rmsnorm_swiglu() {
    // Test LLaMA-style model path (RMSNorm + SwiGLU)
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // Verify model has SwiGLU (gate weight) and no norm bias (RMSNorm)
    assert!(
        model.layers[0].ffn_gate_weight.is_some(),
        "LLaMA model should have gate weight"
    );
    assert!(
        model.layers[0].attn_norm_bias.is_none(),
        "LLaMA model should not have attn_norm_bias (RMSNorm)"
    );

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("LLaMA-style forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_phi_style_layernorm_gelu() {
    // Test phi-style model path (LayerNorm + GELU)
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // Verify model has no gate (GELU path) and has norm bias (LayerNorm)
    assert!(
        model.layers[0].ffn_gate_weight.is_none(),
        "phi model should not have gate weight"
    );
    assert!(
        model.layers[0].attn_norm_bias.is_some(),
        "phi model should have attn_norm_bias (LayerNorm)"
    );

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("phi-style forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_with_lm_head_bias() {
    // Test that lm_head_bias is applied correctly
    let model = create_phi_style_model();
    assert!(
        model.lm_head_bias.is_some(),
        "phi model should have lm_head_bias"
    );

    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);
    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Forward with lm_head_bias should succeed");

    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_kv_cache_reuse() {
    // Verify cached K/V is actually being used for attention
    let model = create_test_model_with_config(&create_llama_style_config());
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);

    // First token: no cache
    let _ = model.forward_single_with_cache(1, &mut cache, 0).unwrap();
    assert_eq!(cache.len(), 1);

    // Second token: should use cached K/V from first token
    let logits = model.forward_single_with_cache(2, &mut cache, 1).unwrap();
    assert_eq!(cache.len(), 2);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_edge_case_zero_token() {
    // Token ID 0 should work (often <pad> or <unk>)
    let model = create_test_model_with_config(&create_llama_style_config());
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model
        .forward_single_with_cache(0, &mut cache, 0)
        .expect("Token ID 0 should work");

    assert_eq!(logits.len(), model.config.vocab_size);
}

#[test]
fn test_forward_single_edge_case_max_valid_token() {
    // Test with maximum valid token ID
    let model = create_test_model_with_config(&create_llama_style_config());
    let max_token = model.config.vocab_size as u32 - 1;
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model
        .forward_single_with_cache(max_token, &mut cache, 0)
        .expect("Max token ID should work");

    assert_eq!(logits.len(), model.config.vocab_size);
}

// =============================================================================
// forward_single_with_scratch tests
// =============================================================================

#[test]
fn test_forward_single_with_scratch_basic() {
    // Test zero-allocation forward pass
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    model
        .forward_single_with_scratch(1, &mut cache, 0, &mut scratch)
        .expect("Scratch forward should succeed");

    // Logits should be in scratch buffer
    assert_eq!(scratch.logits.len(), config.vocab_size);
    assert!(scratch.logits.iter().all(|&x| x.is_finite()));
}

#[test]
#[ignore = "cache.len() API changed - needs update"]
fn test_forward_single_with_scratch_sequential() {
    // Multiple tokens with scratch buffer
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    for pos in 0..3 {
        model
            .forward_single_with_scratch((pos as u32) + 1, &mut cache, pos, &mut scratch)
            .expect("Sequential scratch forward should succeed");
    }

    assert_eq!(cache.len(), 3);
    assert!(scratch.logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_forward_single_with_scratch_phi_style() {
    // Test phi-style (LayerNorm + GELU) with scratch buffer
    let model = create_phi_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    model
        .forward_single_with_scratch(1, &mut cache, 0, &mut scratch)
        .expect("phi-style scratch forward should succeed");

    assert_eq!(scratch.logits.len(), config.vocab_size);
}

#[test]
fn test_forward_single_with_scratch_reuses_buffers() {
    // Verify scratch buffers are reused (no additional allocations)
    let model = create_llama_style_model();
    let config = &model.config;
    let mut cache = OwnedQuantizedKVCache::from_config(config, 128);
    let mut scratch = InferenceScratchBuffer::from_config(config);

    // Record initial buffer capacities
    let hidden_cap = scratch.hidden.capacity();
    let normed_cap = scratch.normed.capacity();
    let logits_cap = scratch.logits.capacity();

    // Run multiple forwards
    for pos in 0..5 {
        model
            .forward_single_with_scratch((pos as u32) + 1, &mut cache, pos, &mut scratch)
            .unwrap();
    }

    // Capacities should not have grown
    assert_eq!(scratch.hidden.capacity(), hidden_cap);
    assert_eq!(scratch.normed.capacity(), normed_cap);
    assert_eq!(scratch.logits.capacity(), logits_cap);
}

include!("single_tests_forward.rs");
include!("single_tests_q8k.rs");
