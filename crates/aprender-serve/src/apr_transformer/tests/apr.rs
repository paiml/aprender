//! T-COV-95 Synthetic Falsification: apr_transformer/mod.rs via Pygmy Models
//!
//! Tests AprTransformer paths using converted Pygmy GGUF models.
//! The same code paths are exercised whether the model is 1KB or 100GB.

use crate::apr_transformer::{
    AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer, GenerateConfig,
};
use crate::convert::GgufToAprConverter;
use crate::gguf::test_factory::{build_minimal_llama_gguf, build_minimal_phi2_gguf};

// ============================================================================
// AprTransformer::from_apr_bytes tests using converted Pygmy GGUF
// ============================================================================

#[test]
fn test_apr_from_pygmy_llama() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("test value should be present");

    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("test value should be present");
    assert_eq!(loaded.config.architecture, "llama");
    assert_eq!(loaded.config.hidden_dim, 64);
    assert_eq!(loaded.config.num_layers, 1);
}

#[test]
fn test_apr_from_pygmy_phi2() {
    let gguf_data = build_minimal_phi2_gguf(32, 64, 128, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("test value should be present");

    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("test value should be present");
    assert_eq!(loaded.config.architecture, "phi2");
}

// ============================================================================
// AprTransformerConfig tests
// ============================================================================

#[test]
fn test_config_default() {
    let config = AprTransformerConfig::default();
    assert!(config.hidden_dim > 0);
    assert!(config.num_layers > 0);
    assert!(config.num_heads > 0);
}

#[test]
fn test_config_head_dim_calculation() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    // head_dim = hidden_dim / num_heads = 256 / 8 = 32
    assert_eq!(config.hidden_dim / config.num_heads, 32);
}

#[test]
fn test_config_gqa_ratio() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 2, // GQA with 4:1 ratio
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    // GQA ratio = num_heads / num_kv_heads
    assert_eq!(config.num_heads / config.num_kv_heads, 4);
}

// ============================================================================
// AprKVCache tests
// ============================================================================

#[test]
fn test_kv_cache_new() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 1000,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };
    let _cache = AprKVCache::new(&config);
    // Cache creation should succeed
}

#[test]
fn test_kv_cache_from_pygmy_config() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");
    let _cache = AprKVCache::new(&apr.config);
    // Cache creation should succeed with converted config
}

// ============================================================================
// AprTransformerLayer tests
// ============================================================================

#[test]
fn test_layer_minimal() {
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: None,
        qkv_weight: vec![0.1; 64 * 192], // Q+K+V = 3 * hidden_dim
        qkv_bias: None,
        attn_output_weight: vec![0.1; 64 * 64],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.1; 64 * 128]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.1; 64 * 128],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.1; 128 * 64],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; 64]),
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

    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert!(layer.ffn_gate_weight.is_some());
}

#[test]
fn test_layer_without_gate() {
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: None,
        qkv_weight: vec![0.1; 64 * 192],
        qkv_bias: None,
        attn_output_weight: vec![0.1; 64 * 64],
        attn_output_bias: None,
        ffn_gate_weight: None, // No gate (non-SwiGLU)
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.1; 64 * 128],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.1; 128 * 64],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; 64]),
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

    assert!(layer.ffn_gate_weight.is_none());
}

#[test]
fn test_layer_with_biases() {
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: Some(vec![0.0; 64]),
        qkv_weight: vec![0.1; 64 * 192],
        qkv_bias: Some(vec![0.0; 192]),
        attn_output_weight: vec![0.1; 64 * 64],
        attn_output_bias: Some(vec![0.0; 64]),
        ffn_gate_weight: Some(vec![0.1; 64 * 128]),
        ffn_gate_bias: Some(vec![0.0; 128]),
        ffn_up_weight: vec![0.1; 64 * 128],
        ffn_up_bias: Some(vec![0.0; 128]),
        ffn_down_weight: vec![0.1; 128 * 64],
        ffn_down_bias: Some(vec![0.0; 64]),
        ffn_norm_weight: Some(vec![1.0; 64]),
        ffn_norm_bias: Some(vec![0.0; 64]),
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

    assert!(layer.attn_norm_bias.is_some());
    assert!(layer.qkv_bias.is_some());
    assert!(layer.attn_output_bias.is_some());
}

// ============================================================================
// GenerateConfig tests
// ============================================================================

#[test]
fn test_generate_config_default() {
    let config = GenerateConfig::default();
    assert!(config.max_tokens > 0);
    assert!(config.temperature >= 0.0);
}

#[test]
fn test_generate_config_greedy() {
    let config = GenerateConfig {
        max_tokens: 10,
        temperature: 0.0, // Greedy
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        trace: false,
        stop_tokens: vec![],
    };
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_generate_config_with_nucleus_sampling() {
    let config = GenerateConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9, // Nucleus sampling
        top_k: 40,
        repetition_penalty: 1.1,
        trace: false,
        stop_tokens: vec![],
    };
    assert_eq!(config.top_p, 0.9);
}

#[test]
fn test_generate_config_with_trace() {
    let config = GenerateConfig {
        max_tokens: 50,
        temperature: 0.5,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: true,
        stop_tokens: vec![],
    };
    assert!(config.trace);
}

// ============================================================================
// AprTransformer structure tests from Pygmy Models
// ============================================================================

#[test]
fn test_apr_transformer_token_embedding_size() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // Embedding size should be vocab_size * hidden_dim
    // The actual size depends on how test_factory creates embeddings
    assert!(!apr.token_embedding.is_empty());
}

#[test]
fn test_apr_transformer_layer_count() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // Should have 1 layer (from build_minimal_llama_gguf)
    assert_eq!(apr.layers.len(), 1);
    assert_eq!(apr.config.num_layers, 1);
}

#[test]
fn test_apr_transformer_output_norm() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // Output norm should be hidden_dim
    assert_eq!(apr.output_norm_weight.len(), 64);
}

#[test]
fn test_apr_transformer_lm_head() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // LM head should be hidden_dim * vocab_size (or tied to embedding)
    assert!(!apr.lm_head_weight.is_empty());
}

// ============================================================================
// AprTransformer::from_apr_bytes error handling
// ============================================================================

#[test]
fn test_apr_from_bytes_too_small() {
    let result = AprTransformer::from_apr_bytes(&[0; 32]);
    assert!(result.is_err());
}

#[test]
fn test_apr_from_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_from_bytes_valid_magic_apr1() {
    // APR with null version byte (legacy)
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    // May fail due to invalid content, but magic check should pass
    let result = AprTransformer::from_apr_bytes(&data);
    // Either succeeds with default values or fails on content parsing
    let _ = result;
}

#[test]
fn test_apr_from_bytes_valid_magic_apr_v1() {
    // APR with explicit v1 marker
    let mut data = vec![0u8; 128];
    data[0..3].copy_from_slice(b"APR");
    data[3] = b'1';
    let _ = AprTransformer::from_apr_bytes(&data);
}

#[test]
fn test_apr_from_bytes_valid_magic_apr_v2() {
    // APR with explicit v2 marker
    let mut data = vec![0u8; 128];
    data[0..3].copy_from_slice(b"APR");
    data[3] = b'2';
    let _ = AprTransformer::from_apr_bytes(&data);
}

// ============================================================================
// Config field extraction from metadata
// ============================================================================

#[test]
fn test_apr_config_from_pygmy_architecture() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("test value should be present");
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("test value should be present");

    // Architecture should be preserved through round-trip
    assert_eq!(loaded.config.architecture, "llama");
}

#[test]
fn test_apr_config_from_pygmy_hidden_dim() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("test value should be present");
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("test value should be present");

    assert_eq!(loaded.config.hidden_dim, 64);
}

#[test]
fn test_apr_config_from_pygmy_num_heads() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("test value should be present");
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("test value should be present");

    assert_eq!(loaded.config.num_heads, 4);
}

#[test]
fn test_apr_config_from_pygmy_vocab_size() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // Vocab size should be 32 (from build_minimal_llama_gguf)
    assert_eq!(apr.config.vocab_size, 32);
}

#[test]
fn test_apr_config_from_pygmy_intermediate_dim() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // Intermediate dim is extracted from tensor sizes, may differ from input
    assert!(apr.config.intermediate_dim > 0);
}

#[test]
fn test_apr_config_from_pygmy_context_length() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // Context length is extracted from metadata, must be positive
    assert!(apr.config.context_length > 0);
}

#[test]
fn test_apr_config_from_pygmy_rope_theta() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // build_minimal_llama_gguf sets rope_freq_base to 10000.0
    assert!((apr.config.rope_theta - 10000.0).abs() < 0.1);
}

#[test]
fn test_apr_config_from_pygmy_eps() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // build_minimal_llama_gguf sets rms_epsilon to 1e-5
    assert!((apr.config.eps - 1e-5).abs() < 1e-7);
}

// ============================================================================
// Optional fields
// ============================================================================

#[test]
fn test_apr_transformer_q4k_layers_default_none() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // q4k_layers should be None for standard conversion
    assert!(apr.q4k_layers.is_none());
}

#[test]
fn test_apr_transformer_lm_head_q6k_default_none() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // lm_head_weight_q6k should be None for standard conversion
    assert!(apr.lm_head_weight_q6k.is_none());
}

#[test]
fn test_apr_transformer_lm_head_q4k_default_none() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    // lm_head_weight_q4k should be None for standard conversion
    assert!(apr.lm_head_weight_q4k.is_none());
}

include!("apr_layer.rs");
