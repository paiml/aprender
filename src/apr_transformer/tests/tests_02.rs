//! APR Transformer Coverage Tests Part 2 (PMAT-803)
//!
//! Additional comprehensive tests for `src/apr_transformer/mod.rs` to drive coverage.
//!
//! # Coverage Targets
//!
//! - `AprTransformer::forward()` edge cases and internal paths
//! - `AprTransformer::forward_with_cache()` edge cases
//! - Attention computation with GQA
//! - KV cache edge cases
//! - Matmul fallback paths
//! - Layer norm with bias
//! - Q4K layer weights paths

use crate::apr_transformer::{AprKVCache, AprTransformer, AprTransformerConfig, GenerateConfig};

// ============================================================================
// Part 1: Matmul Edge Cases
// ============================================================================

#[test]
fn test_forward_triggers_matmul_remainder_path() {
    // Use dimensions where out_dim % 4 != 0 to exercise remainder path
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 63, // Not divisible by 4
        num_layers: 1,
        num_heads: 3,
        num_kv_heads: 3,
        vocab_size: 50,
        intermediate_dim: 127, // Not divisible by 4
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_matmul_scalar_fallback_with_mismatched_weights() {
    // Create transformer with intentionally mismatched weights to trigger scalar fallback
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Intentionally set wrong weight size to trigger matmul_scalar
    transformer.layers[0].ffn_up_weight = vec![0.1; 100]; // Wrong size

    // Forward should still work (scalar fallback handles mismatched sizes)
    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 2: Layer Norm with Bias
// ============================================================================

#[test]
fn test_forward_with_layer_norm_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add attention norm bias to layers
    for layer in &mut transformer.layers {
        layer.attn_norm_bias = Some(vec![0.1; config.hidden_dim]);
    }

    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_output_norm_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add output norm bias
    transformer.output_norm_bias = Some(vec![0.05; config.hidden_dim]);

    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_ffn_norm_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add FFN norm and its bias
    for layer in &mut transformer.layers {
        layer.ffn_norm_weight = Some(vec![1.0; config.hidden_dim]);
        layer.ffn_norm_bias = Some(vec![0.01; config.hidden_dim]);
    }

    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 3: Forward with Various Bias Configurations
// ============================================================================

#[test]
fn test_forward_with_all_biases() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    let qkv_dim = config.hidden_dim * 3;
    for layer in &mut transformer.layers {
        layer.attn_norm_bias = Some(vec![0.01; config.hidden_dim]);
        layer.qkv_bias = Some(vec![0.01; qkv_dim]);
        layer.attn_output_bias = Some(vec![0.01; config.hidden_dim]);
        layer.ffn_up_bias = Some(vec![0.01; config.intermediate_dim]);
        layer.ffn_down_bias = Some(vec![0.01; config.hidden_dim]);
        layer.ffn_norm_weight = Some(vec![1.0; config.hidden_dim]);
        layer.ffn_norm_bias = Some(vec![0.01; config.hidden_dim]);
    }
    transformer.output_norm_bias = Some(vec![0.01; config.hidden_dim]);
    transformer.lm_head_bias = Some(vec![0.01; config.vocab_size]);

    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_swiglu_and_gate_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Enable SwiGLU with gate bias
    for layer in &mut transformer.layers {
        layer.ffn_gate_weight = Some(vec![0.1; config.hidden_dim * config.intermediate_dim]);
        layer.ffn_gate_bias = Some(vec![0.01; config.intermediate_dim]);
    }

    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 4: Attention with Different Head Configurations
// ============================================================================

#[test]
fn test_forward_with_single_head() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32, // single head with dim 32
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2, 3]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_many_heads() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 16,
        num_kv_heads: 16,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_gqa_4_to_1_ratio() {
    // 4 query heads, 1 KV head
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust QKV for GQA dimensions
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_gqa_8_to_2_ratio() {
    // 8 query heads, 2 KV heads
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust QKV for GQA dimensions
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 5: KV Cache Edge Cases
// ============================================================================

#[test]
fn test_kv_cache_near_capacity() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        context_length: 8, // Small capacity
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Fill cache to near capacity
    for pos in 0..7 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
        assert_eq!(cache.len(), pos + 1);
    }

    // One more should still work
    let result = transformer.forward_with_cache(7, &mut cache, 7);
    assert!(result.is_ok());
    assert_eq!(cache.len(), 8);
    assert_eq!(cache.capacity(), 8);
}

#[test]
fn test_kv_cache_clear_and_reuse() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Fill some entries
    for pos in 0..5 {
        let _ = transformer.forward_with_cache(pos as u32, &mut cache, pos);
    }
    assert_eq!(cache.len(), 5);

    // Clear
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Reuse
    for pos in 0..3 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
    assert_eq!(cache.len(), 3);
}

#[test]
fn test_kv_cache_get_layer_bounds() {
    let config = create_test_config();
    let mut cache = AprKVCache::new(&config);

    // Append some data
    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }
    // No advance() needed - append() auto-advances on last layer

    // Get for each layer
    for layer in 0..config.num_layers {
        let (k_cache, v_cache) = cache.get(layer);
        assert_eq!(k_cache.len(), kv_size);
        assert_eq!(v_cache.len(), kv_size);
    }
}

#[test]
fn test_kv_cache_multi_position_get() {
    let config = create_test_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

    // Append multiple positions
    for pos in 0..5 {
        let k = vec![(pos + 1) as f32; kv_size];
        let v = vec![(pos + 10) as f32; kv_size];
        for layer in 0..config.num_layers {
            cache.append(layer, &k, &v);
        }
        // No advance() needed - append() auto-advances on last layer
    }

    // Get should return all positions
    let (k_cache, v_cache) = cache.get(0);
    assert_eq!(k_cache.len(), 5 * kv_size);
    assert_eq!(v_cache.len(), 5 * kv_size);

    // Verify first position values
    assert!((k_cache[0] - 1.0).abs() < 1e-6);
    assert!((v_cache[0] - 10.0).abs() < 1e-6);

    // Verify last position values
    assert!((k_cache[4 * kv_size] - 5.0).abs() < 1e-6);
}

// ============================================================================
// Part 6: Forward with Cache - Additional Paths
// ============================================================================

#[test]
fn test_forward_with_cache_qkv_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add QKV bias
    let qkv_dim = config.hidden_dim * 3;
    for layer in &mut transformer.layers {
        layer.qkv_bias = Some(vec![0.01; qkv_dim]);
    }

    let mut cache = AprKVCache::new(&config);
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_attn_output_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add attention output bias
    for layer in &mut transformer.layers {
        layer.attn_output_bias = Some(vec![0.01; config.hidden_dim]);
    }

    let mut cache = AprKVCache::new(&config);
    for pos in 0..3 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

#[test]
fn test_forward_with_cache_ffn_biases() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add FFN biases
    for layer in &mut transformer.layers {
        layer.ffn_up_bias = Some(vec![0.01; config.intermediate_dim]);
        layer.ffn_down_bias = Some(vec![0.01; config.hidden_dim]);
    }

    let mut cache = AprKVCache::new(&config);
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_lm_head_bias() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add LM head bias
    transformer.lm_head_bias = Some(vec![0.01; config.vocab_size]);

    let mut cache = AprKVCache::new(&config);
    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());
}

include!("forward.rs");
include!("forward_standard.rs");
