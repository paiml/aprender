//! Q4_0 SIMD Transformer Tests (PMAT-803)
//!
//! Comprehensive tests for `src/apr_transformer/q4_simd.rs` to drive coverage.
//!
//! # Coverage Targets
//!
//! - `QuantizedAprTensorQ4` - Q4_0 tensor struct
//! - `QuantizedAprLayerQ4` - Quantized layer struct
//! - `QuantizedAprTransformerQ4` - Full transformer with SIMD matmul
//! - `AprInferenceScratch` - Zero-allocation scratch buffer
//!
//! # Test Organization
//!
//! - Part 1: QuantizedAprTensorQ4 tests
//! - Part 2: AprInferenceScratch tests
//! - Part 3: QuantizedAprLayerQ4 tests
//! - Part 4: QuantizedAprTransformerQ4 construction tests
//! - Part 5: Forward pass tests
//! - Part 6: KV cache tests
//! - Part 7: RoPE and attention tests
//! - Part 8: Memory and utility tests

use crate::apr_transformer::{
    AprInferenceScratch, AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4,
    QuantizedAprTransformerQ4,
};

// ============================================================================
// Part 1: QuantizedAprTensorQ4 Tests
// ============================================================================

#[test]
fn test_quantized_tensor_new() {
    let data = vec![0u8; 18]; // 1 Q4_0 block
    let tensor = QuantizedAprTensorQ4::new(data.clone(), 32, 1);

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 1);
}

#[test]
fn test_quantized_tensor_new_empty() {
    let tensor = QuantizedAprTensorQ4::new(vec![], 0, 0);

    assert!(tensor.data.is_empty());
    assert_eq!(tensor.in_dim, 0);
    assert_eq!(tensor.out_dim, 0);
}

#[test]
fn test_quantized_tensor_new_large() {
    // 1024 x 4096 matrix
    let num_elements: usize = 1024 * 4096;
    let num_blocks = num_elements.div_ceil(32);
    let data = vec![0u8; num_blocks * 18];
    let tensor = QuantizedAprTensorQ4::new(data, 1024, 4096);

    assert_eq!(tensor.in_dim, 1024);
    assert_eq!(tensor.out_dim, 4096);
}

#[test]
fn test_quantized_tensor_zeros_basic() {
    let tensor = QuantizedAprTensorQ4::zeros(64, 128);

    assert_eq!(tensor.in_dim, 64);
    assert_eq!(tensor.out_dim, 128);

    // 64 * 128 = 8192 elements, 256 blocks, 256 * 18 = 4608 bytes
    let expected_bytes = QuantizedAprTensorQ4::expected_bytes(64 * 128);
    assert_eq!(tensor.data.len(), expected_bytes);
    assert!(tensor.data.iter().all(|&b| b == 0));
}

#[test]
fn test_quantized_tensor_zeros_single_block() {
    // Exactly 32 elements = 1 block
    let tensor = QuantizedAprTensorQ4::zeros(32, 1);

    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 1);
    assert_eq!(tensor.data.len(), 18); // 1 block = 18 bytes
}

#[test]
fn test_quantized_tensor_zeros_partial_block() {
    // 33 elements = 2 blocks (ceil)
    let tensor = QuantizedAprTensorQ4::zeros(33, 1);

    assert_eq!(tensor.in_dim, 33);
    assert_eq!(tensor.out_dim, 1);
    assert_eq!(tensor.data.len(), 36); // 2 blocks = 36 bytes
}

#[test]
fn test_quantized_tensor_expected_bytes() {
    // Exact block boundary
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(96), 54);

    // Partial blocks (ceil division)
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(1), 18); // 1 element still needs 1 block
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(31), 18);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(65), 54);
}

#[test]
fn test_quantized_tensor_expected_bytes_zero() {
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
}

#[test]
fn test_quantized_tensor_expected_bytes_large() {
    // 1M elements = 31250 blocks = 562500 bytes
    let expected = (1_000_000_usize.div_ceil(32)) * 18;
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(1_000_000), expected);
}

#[test]
fn test_quantized_tensor_clone() {
    let tensor = QuantizedAprTensorQ4::zeros(64, 64);
    let cloned = tensor.clone();

    assert_eq!(cloned.data.len(), tensor.data.len());
    assert_eq!(cloned.in_dim, tensor.in_dim);
    assert_eq!(cloned.out_dim, tensor.out_dim);
}

// ============================================================================
// Part 2: AprInferenceScratch Tests
// ============================================================================

#[test]
fn test_scratch_from_config_basic() {
    let config = create_test_config(64, 128, 4, 4, 100);
    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 64);
    assert_eq!(scratch.normed.len(), 64);
    assert_eq!(scratch.qkv_out.len(), 64 * 3); // Conservative estimate
    assert_eq!(scratch.q.len(), 64);
    assert_eq!(scratch.k.len(), 64);
    assert_eq!(scratch.v.len(), 64);
    assert_eq!(scratch.attn_out.len(), 64);
    assert_eq!(scratch.ffn_input.len(), 64);
    assert_eq!(scratch.ffn_up.len(), 128);
    assert_eq!(scratch.ffn_gate.len(), 128);
    assert_eq!(scratch.ffn_out.len(), 64);
}

#[test]
fn test_scratch_from_config_large() {
    let config = create_test_config(2048, 8192, 32, 8, 32000);
    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 2048);
    assert_eq!(scratch.ffn_up.len(), 8192);
    assert_eq!(scratch.ffn_gate.len(), 8192);
}

#[test]
fn test_scratch_clear() {
    let config = create_test_config(64, 128, 4, 4, 100);
    let mut scratch = AprInferenceScratch::from_config(&config);

    // Set some values
    scratch.hidden[0] = 1.0;
    scratch.normed[0] = 2.0;
    scratch.qkv_out[0] = 3.0;
    scratch.q[0] = 4.0;
    scratch.k[0] = 5.0;
    scratch.v[0] = 6.0;
    scratch.attn_out[0] = 7.0;
    scratch.ffn_input[0] = 8.0;
    scratch.ffn_up[0] = 9.0;
    scratch.ffn_gate[0] = 10.0;
    scratch.ffn_out[0] = 11.0;

    scratch.clear();

    // Verify all cleared
    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.normed.iter().all(|&x| x == 0.0));
    assert!(scratch.qkv_out.iter().all(|&x| x == 0.0));
    assert!(scratch.q.iter().all(|&x| x == 0.0));
    assert!(scratch.k.iter().all(|&x| x == 0.0));
    assert!(scratch.v.iter().all(|&x| x == 0.0));
    assert!(scratch.attn_out.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_input.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_gate.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_out.iter().all(|&x| x == 0.0));
}

#[test]
fn test_scratch_multiple_clears() {
    let config = create_test_config(32, 64, 2, 2, 50);
    let mut scratch = AprInferenceScratch::from_config(&config);

    for i in 0..10 {
        scratch.hidden.fill(i as f32);
        scratch.clear();
        assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    }
}

// ============================================================================
// Part 3: QuantizedAprLayerQ4 Tests
// ============================================================================

#[test]
fn test_layer_q4_construction() {
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    };

    assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
    assert_eq!(layer.qkv_weight.out_dim, hidden_dim * 3);
    assert!(layer.ffn_gate_weight.is_none());
    assert!(layer.ffn_norm_weight.is_some());
}

#[test]
fn test_layer_q4_with_swiglu_gate() {
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    };

    assert!(layer.ffn_gate_weight.is_some());
    let gate = layer.ffn_gate_weight.as_ref().expect("test value should be present");
    assert_eq!(gate.in_dim, hidden_dim);
    assert_eq!(gate.out_dim, intermediate_dim);
}

#[test]
fn test_layer_q4_clone() {
    let layer = create_test_layer_q4(64, 128, true);
    let cloned = layer.clone();

    assert_eq!(cloned.attn_norm_weight.len(), layer.attn_norm_weight.len());
    assert_eq!(cloned.qkv_weight.data.len(), layer.qkv_weight.data.len());
    assert!(cloned.ffn_gate_weight.is_some());
}

// ============================================================================
// Part 4: QuantizedAprTransformerQ4 Construction Tests
// ============================================================================

#[test]
fn test_transformer_q4_construction() {
    let transformer = create_minimal_q4_transformer(1);

    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.layers.len(), 1);
    assert_eq!(transformer.token_embedding.len(), 100 * 64);
    assert_eq!(transformer.output_norm_weight.len(), 64);
}

#[test]
fn test_transformer_q4_multi_layer() {
    let num_layers = 4;
    let transformer = create_minimal_q4_transformer(num_layers);

    assert_eq!(transformer.config.num_layers, num_layers);
    assert_eq!(transformer.layers.len(), num_layers);
}

#[test]
fn test_transformer_q4_config_accessor() {
    let transformer = create_minimal_q4_transformer(2);
    let config = transformer.config();

    assert_eq!(config.hidden_dim, 64);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.vocab_size, 100);
    assert_eq!(config.intermediate_dim, 128);
    assert!((config.rope_theta - 10000.0).abs() < 1e-6);
}

#[test]
fn test_transformer_q4_create_scratch() {
    let transformer = create_minimal_q4_transformer(1);
    let scratch = transformer.create_scratch();

    assert_eq!(scratch.hidden.len(), 64);
    assert_eq!(scratch.ffn_up.len(), 128);
}

#[test]
fn test_transformer_q4_create_kv_cache() {
    let transformer = create_minimal_q4_transformer(2);
    let cache = transformer.create_kv_cache();

    assert_eq!(cache.len(), 0); // Empty cache initially
}

// ============================================================================
// Part 5: Forward Pass Tests
// ============================================================================

#[test]
fn test_forward_empty_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[]);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        crate::error::RealizarError::InvalidShape { .. }
    ));
}

#[test]
fn test_forward_single_token() {
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[0]);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 100); // vocab_size
}

#[test]
fn test_forward_multiple_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[0, 1, 2]);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 100); // Only last token logits returned
}

#[test]
fn test_forward_oov_token() {
    // Out-of-vocabulary token should be handled gracefully
    let transformer = create_minimal_q4_transformer(1);
    let result = transformer.forward(&[999]); // vocab_size = 100

    assert!(result.is_ok()); // Should use zero embedding
}

#[test]
fn test_forward_multi_layer() {
    let transformer = create_minimal_q4_transformer(4);
    let result = transformer.forward(&[42]);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_with_swiglu() {
    let transformer = create_q4_transformer_with_gate(2);
    let result = transformer.forward(&[1, 2, 3]);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_single_with_scratch() {
    let transformer = create_minimal_q4_transformer(1);
    let mut scratch = transformer.create_scratch();

    let result = transformer.forward_single_with_scratch(0, &mut scratch);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_forward_single_with_scratch_oov() {
    let transformer = create_minimal_q4_transformer(1);
    let mut scratch = transformer.create_scratch();

    let result = transformer.forward_single_with_scratch(999, &mut scratch);

    assert!(result.is_ok()); // Should handle OOV gracefully
}

#[test]
fn test_forward_single_with_scratch_reuse() {
    let transformer = create_minimal_q4_transformer(1);
    let mut scratch = transformer.create_scratch();

    // Multiple forward passes with same scratch
    for token_id in 0..10u32 {
        let result = transformer.forward_single_with_scratch(token_id, &mut scratch);
        assert!(result.is_ok());
    }
}

#[test]
fn test_forward_single_with_scratch_swiglu() {
    let transformer = create_q4_transformer_with_gate(1);
    let mut scratch = transformer.create_scratch();

    let result = transformer.forward_single_with_scratch(5, &mut scratch);
    assert!(result.is_ok());
}

// ============================================================================
// Part 6: KV Cache Tests
// ============================================================================

#[test]
fn test_forward_with_cache_empty_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    let result = transformer.forward_with_cache(&[], &mut cache);

    assert!(result.is_err());
}

include!("q4_simd_forward.rs");
include!("q4_simd_gguf.rs");
