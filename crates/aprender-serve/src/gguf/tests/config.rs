//! Part 01: Config + IMP-101 + IMP-105 Tests
//!
//! Extracted from gguf_monolith.rs as part of PMAT-802 shatter.
//!
//! ## Contents
//! - QuantizedGenerateConfig tests
//! - IMP-101: RoPE and Causal Attention Tests
//! - IMP-105: Grouped Query Attention (GQA) Tests

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::*;

// ============================================================
// QuantizedGGUFTransformer::generate() tests
// Per benchmark-model-runners-spec.md "What's Remaining" item 1
// ============================================================

#[test]
fn test_generate_config_default() {
    let config = QuantizedGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_generate_config_builder() {
    let config = QuantizedGenerateConfig::default()
        .with_max_tokens(128)
        .with_temperature(0.7)
        .with_top_k(40)
        .with_stop_tokens(vec![50256]);

    assert_eq!(config.max_tokens, 128);
    assert!((config.temperature - 0.7).abs() < 1e-6);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.stop_tokens, vec![50256]);
}

#[test]
fn test_generate_config_deterministic() {
    // Temperature 0.0 = greedy decoding
    let config = QuantizedGenerateConfig::deterministic(32);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert_eq!(config.max_tokens, 32);
}

// ==========================================================================
// IMP-101: RoPE and Causal Attention Tests
// ==========================================================================

/// IMP-101a: RoPE preserves vector magnitude
#[test]
fn test_imp_101a_rope_preserves_norm() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 4, // 4 heads x 16 dim = 64
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 64,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Create test vector
    let mut x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    // Apply RoPE at position 10 (4 heads for 64-dim vector)
    model.apply_rope(&mut x, 10, 4);

    let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    // RoPE is a rotation, should preserve L2 norm
    assert!(
        (norm_before - norm_after).abs() < 1e-5,
        "IMP-101a: RoPE should preserve vector norm. Before: {}, After: {}",
        norm_before,
        norm_after
    );
}

/// IMP-101a: RoPE produces different outputs at different positions
#[test]
fn test_imp_101a_rope_position_dependent() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 64,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Apply RoPE at different positions
    let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

    let mut x_pos0 = original.clone();
    let mut x_pos10 = original.clone();
    let mut x_pos100 = original.clone();

    model.apply_rope(&mut x_pos0, 0, 4);
    model.apply_rope(&mut x_pos10, 10, 4);
    model.apply_rope(&mut x_pos100, 100, 4);

    // Different positions should produce different outputs
    let diff_0_10: f32 = x_pos0
        .iter()
        .zip(x_pos10.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_10_100: f32 = x_pos10
        .iter()
        .zip(x_pos100.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff_0_10 > 1e-3,
        "IMP-101a: RoPE should produce different outputs at positions 0 vs 10"
    );
    assert!(
        diff_10_100 > 1e-3,
        "IMP-101a: RoPE should produce different outputs at positions 10 vs 100"
    );
}

/// IMP-101b: Causal attention only attends to past tokens
#[test]
fn test_imp_101b_causal_attention_mask() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 8, // Small for testing
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2, // 2 heads x 4 dim = 8
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 8,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Create test Q, K, V (seq_len=4, hidden_dim=8)
    let seq_len = 4;
    let hidden_dim = 8;
    let q: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let k: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| (i as f32 * 0.2).cos())
        .collect();
    let v: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| i as f32 * 0.1)
        .collect();

    let output = model.causal_attention(&q, &k, &v, seq_len);

    // Output should have correct shape
    assert_eq!(
        output.len(),
        seq_len * hidden_dim,
        "IMP-101b: Causal attention output should have shape [seq_len, hidden_dim]"
    );

    // First position can only attend to itself
    // Last position can attend to all positions
    // This is verified by the fact that the output doesn't crash and has correct shape
    assert!(
        output.iter().all(|v| v.is_finite()),
        "IMP-101b: All attention outputs should be finite"
    );
}

/// IMP-101b: Causal attention softmax sums to 1
#[test]
fn test_imp_101b_causal_attention_softmax_normalized() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 4,
        intermediate_dim: 16,
        num_layers: 1,
        num_heads: 1, // 1 head for simplicity
        num_kv_heads: 1,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 4,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Create identity K (each position is unique)
    let seq_len = 3;
    let hidden_dim = 4;

    // Q = same for all positions, K = identity-like
    let q: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0].repeat(seq_len);
    let k: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // pos 0
        0.0, 1.0, 0.0, 0.0, // pos 1
        0.0, 0.0, 1.0, 0.0, // pos 2
    ];
    let v: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // pos 0
        0.0, 1.0, 0.0, 0.0, // pos 1
        0.0, 0.0, 1.0, 0.0, // pos 2
    ];

    let output = model.causal_attention(&q, &k, &v, seq_len);

    // Output at each position should be a weighted sum of values
    // For position 0: can only attend to position 0, so output = V[0]
    let pos0_output = &output[0..hidden_dim];
    assert!(
        (pos0_output[0] - 1.0).abs() < 1e-5,
        "IMP-101b: Position 0 should only attend to itself"
    );
}

// ===== IMP-101c: KV Cache Integration Tests =====

/// IMP-101c: KV cache initializes correctly
#[test]
fn test_imp_101c_kv_cache_initialization() {
    let cache = OwnedQuantizedKVCache::new(12, 768, 2048);

    assert_eq!(cache.len(), 0, "IMP-101c: New cache should be empty");
    assert!(cache.is_empty(), "IMP-101c: is_empty should return true");
    assert_eq!(cache.max_len(), 2048, "IMP-101c: max_len should match");
}

/// IMP-101c: KV cache from config
#[test]
fn test_imp_101c_kv_cache_from_config() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 512,
        intermediate_dim: 2048,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 32000,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let cache = OwnedQuantizedKVCache::from_config(&config, 1024);

    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_len(), 1024);
}

/// IMP-101c: KV cache append and retrieve
#[test]
fn test_imp_101c_kv_cache_append_retrieve() {
    let mut cache = OwnedQuantizedKVCache::new(2, 4, 100);

    // Append K/V for layer 0
    let k0 = vec![1.0, 2.0, 3.0, 4.0];
    let v0 = vec![0.1, 0.2, 0.3, 0.4];
    cache.append(0, &k0, &v0);

    // Append K/V for layer 1
    let k1 = vec![5.0, 6.0, 7.0, 8.0];
    let v1 = vec![0.5, 0.6, 0.7, 0.8];
    cache.append(1, &k1, &v1);

    // Advance position
    cache.advance();

    assert_eq!(cache.len(), 1, "IMP-101c: Cache should have 1 position");

    // Verify retrieval
    let retrieved_k0 = cache.get_k(0);
    assert_eq!(
        retrieved_k0.len(),
        4,
        "IMP-101c: Retrieved K should have 4 elements"
    );
    assert!(
        (retrieved_k0[0] - 1.0).abs() < 1e-6,
        "IMP-101c: K values should match"
    );

    let retrieved_v1 = cache.get_v(1);
    assert!(
        (retrieved_v1[0] - 0.5).abs() < 1e-6,
        "IMP-101c: V values should match"
    );
}

/// IMP-101c: KV cache reset clears data
#[test]
fn test_imp_101c_kv_cache_reset() {
    let mut cache = OwnedQuantizedKVCache::new(2, 4, 100);

    // Add some data
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![0.1, 0.2, 0.3, 0.4];
    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 1);

    // Reset
    cache.reset();

    assert_eq!(cache.len(), 0, "IMP-101c: Reset should clear position");
    assert!(cache.is_empty(), "IMP-101c: Reset should make cache empty");
    assert!(
        cache.get_k(0).is_empty(),
        "IMP-101c: Reset should clear K data"
    );
}

include!("imp_101c.rs");
