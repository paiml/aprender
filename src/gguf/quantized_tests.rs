//! Comprehensive tests for quantized tensor types
//!
//! These tests target the ~37% uncovered code in quantized.rs:
//! - OwnedQKVWeights::from_borrowed with Separate variant
//! - OwnedQKVWeights::out_dim for Separate variant
//! - OwnedQKVWeights::q_dim for Separate variant
//! - OwnedQuantizedLayer::from_borrowed (full function)
//! - Edge cases and error paths

use super::*;
use crate::gguf::types::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create test data buffer with known pattern
fn create_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Create a QuantizedTensorRef with given parameters
fn tensor_ref(
    offset: usize,
    byte_size: usize,
    num_elements: usize,
    qtype: u32,
) -> QuantizedTensorRef {
    QuantizedTensorRef {
        offset,
        byte_size,
        num_elements,
        qtype,
    }
}

/// Create minimal GGUFConfig for testing
fn test_config(hidden_dim: usize, intermediate_dim: usize) -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

// ============================================================================
// OwnedQKVWeights::from_borrowed - Separate variant (UNCOVERED)
// ============================================================================

#[test]
fn test_owned_qkv_weights_from_borrowed_separate() {
    // Test Separate variant conversion (was previously uncovered)
    let hidden_dim = 64;
    let kv_dim = 32; // GQA: different K/V dims

    let q = tensor_ref(0, 128, hidden_dim * hidden_dim, GGUF_TYPE_Q4_K);
    let k = tensor_ref(128, 64, hidden_dim * kv_dim, GGUF_TYPE_Q4_K);
    let v = tensor_ref(192, 64, hidden_dim * kv_dim, GGUF_TYPE_Q4_K);

    let borrowed = QKVWeights::Separate { q, k, v };
    let data = create_test_data(300);

    let owned = OwnedQKVWeights::from_borrowed(&borrowed, &data, hidden_dim);

    // Verify Separate variant was created
    match owned {
        OwnedQKVWeights::Separate {
            ref q,
            ref k,
            ref v,
        } => {
            // Check Q tensor
            assert_eq!(q.in_dim, hidden_dim);
            assert_eq!(q.out_dim, hidden_dim); // q_dim = num_elements / hidden_dim
            assert_eq!(q.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(q.data.len(), 128);

            // Check K tensor (GQA)
            assert_eq!(k.in_dim, hidden_dim);
            assert_eq!(k.out_dim, kv_dim);
            assert_eq!(k.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(k.data.len(), 64);

            // Check V tensor (GQA)
            assert_eq!(v.in_dim, hidden_dim);
            assert_eq!(v.out_dim, kv_dim);
            assert_eq!(v.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(v.data.len(), 64);
        },
        OwnedQKVWeights::Fused(_) => panic!("Expected Separate variant"),
    }
}

#[test]
fn test_owned_qkv_weights_separate_out_dim() {
    // Test out_dim() for Separate variant (was previously uncovered)
    let hidden_dim = 64;
    let q_dim = 64;
    let k_dim = 32;
    let v_dim = 32;

    let q = tensor_ref(0, 100, hidden_dim * q_dim, GGUF_TYPE_Q4_K);
    let k = tensor_ref(100, 50, hidden_dim * k_dim, GGUF_TYPE_Q4_K);
    let v = tensor_ref(150, 50, hidden_dim * v_dim, GGUF_TYPE_Q4_K);

    let borrowed = QKVWeights::Separate { q, k, v };
    let data = create_test_data(250);
    let owned = OwnedQKVWeights::from_borrowed(&borrowed, &data, hidden_dim);

    // out_dim = q.out_dim + k.out_dim + v.out_dim = 64 + 32 + 32 = 128
    assert_eq!(owned.out_dim(), q_dim + k_dim + v_dim);
}

#[test]
fn test_owned_qkv_weights_separate_q_dim() {
    // Test q_dim() for Separate variant (was previously uncovered)
    let hidden_dim = 64;
    let q_dim = 128; // Q can be larger (multi-head)
    let k_dim = 32;
    let v_dim = 32;

    let q = tensor_ref(0, 200, hidden_dim * q_dim, GGUF_TYPE_Q4_K);
    let k = tensor_ref(200, 50, hidden_dim * k_dim, GGUF_TYPE_Q4_K);
    let v = tensor_ref(250, 50, hidden_dim * v_dim, GGUF_TYPE_Q4_K);

    let borrowed = QKVWeights::Separate { q, k, v };
    let data = create_test_data(350);
    let owned = OwnedQKVWeights::from_borrowed(&borrowed, &data, hidden_dim);

    // q_dim should return just the Q dimension
    assert_eq!(owned.q_dim(), q_dim);
}

// ============================================================================
// OwnedQuantizedLayer::from_borrowed (UNCOVERED)
// ============================================================================

#[test]
fn test_owned_quantized_layer_from_borrowed_minimal() {
    // Test from_borrowed with minimal layer (no optional fields)
    // Use gpt2 arch: GeluMlp has no gate weight, so ffn_up is NOT fused gate+up
    let mut config = test_config(64, 128);
    config.constraints = crate::gguf::ArchConstraints::from_architecture("gpt2");

    let layer = QuantizedGGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: None,
        qkv_weight: QKVWeights::Fused(tensor_ref(0, 100, 64 * 192, GGUF_TYPE_Q4_K)),
        qkv_bias: None,
        attn_output_weight: tensor_ref(100, 50, 64 * 64, GGUF_TYPE_Q4_K),
        attn_output_bias: None,
        ffn_up_weight: tensor_ref(150, 80, 64 * 128, GGUF_TYPE_Q4_K),
        ffn_up_bias: None,
        ffn_down_weight: tensor_ref(230, 80, 128 * 64, GGUF_TYPE_Q4_K),
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let data = create_test_data(400);
    let owned = OwnedQuantizedLayer::from_borrowed(&layer, &data, &config);

    // Verify attn_norm_weight copied
    assert_eq!(owned.attn_norm_weight.len(), 64);
    assert!(owned.attn_norm_bias.is_none());

    // Verify QKV weight converted
    assert_eq!(owned.qkv_weight.out_dim(), 192);

    // Verify attn_output_weight dimensions
    assert_eq!(owned.attn_output_weight.in_dim, 64);
    assert_eq!(owned.attn_output_weight.out_dim, 64);

    // Verify FFN weights dimensions
    assert_eq!(owned.ffn_up_weight.in_dim, 64);
    assert_eq!(owned.ffn_up_weight.out_dim, 128);
    assert_eq!(owned.ffn_down_weight.in_dim, 128);
    assert_eq!(owned.ffn_down_weight.out_dim, 64);

    // Verify optional fields are None
    assert!(owned.ffn_gate_weight.is_none());
    assert!(owned.ffn_gate_bias.is_none());
    assert!(owned.ffn_norm_weight.is_none());
    assert!(owned.ffn_norm_bias.is_none());
}

#[test]
fn test_owned_quantized_layer_from_borrowed_with_gate() {
    // Test from_borrowed with SwiGLU gate weights (LLaMA-style)
    let config = test_config(64, 128);

    let layer = QuantizedGGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: Some(vec![0.0; 64]),
        qkv_weight: QKVWeights::Fused(tensor_ref(0, 100, 64 * 192, GGUF_TYPE_Q4_K)),
        qkv_bias: Some(vec![0.0; 192]),
        attn_output_weight: tensor_ref(100, 50, 64 * 64, GGUF_TYPE_Q4_K),
        attn_output_bias: Some(vec![0.0; 64]),
        ffn_up_weight: tensor_ref(150, 80, 64 * 128, GGUF_TYPE_Q4_K),
        ffn_up_bias: Some(vec![0.0; 128]),
        ffn_down_weight: tensor_ref(230, 80, 128 * 64, GGUF_TYPE_Q4_K),
        ffn_down_bias: Some(vec![0.0; 64]),
        ffn_gate_weight: Some(tensor_ref(310, 80, 64 * 128, GGUF_TYPE_Q4_K)),
        ffn_gate_bias: Some(vec![0.0; 128]),
        ffn_norm_weight: Some(vec![1.0; 64]),
        ffn_norm_bias: Some(vec![0.0; 64]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let data = create_test_data(450);
    let owned = OwnedQuantizedLayer::from_borrowed(&layer, &data, &config);

    // Verify all optional fields present
    assert!(owned.attn_norm_bias.is_some());
    assert_eq!(owned.attn_norm_bias.as_ref().unwrap().len(), 64);

    assert!(owned.qkv_bias.is_some());
    assert_eq!(owned.qkv_bias.as_ref().unwrap().len(), 192);

    assert!(owned.attn_output_bias.is_some());
    assert!(owned.ffn_up_bias.is_some());
    assert!(owned.ffn_down_bias.is_some());

    // Verify gate weight
    assert!(owned.ffn_gate_weight.is_some());
    let gate = owned.ffn_gate_weight.as_ref().unwrap();
    assert_eq!(gate.in_dim, 64);
    assert_eq!(gate.out_dim, 128);

    assert!(owned.ffn_gate_bias.is_some());

    // Verify FFN norm
    assert!(owned.ffn_norm_weight.is_some());
    assert_eq!(owned.ffn_norm_weight.as_ref().unwrap().len(), 64);
    assert!(owned.ffn_norm_bias.is_some());
}

#[test]
fn test_owned_quantized_layer_from_borrowed_separate_qkv() {
    // Test from_borrowed with separate Q/K/V weights (LLaMA-style GQA)
    let hidden_dim = 64;
    let kv_dim = 32;
    let config = test_config(hidden_dim, 128);

    let layer = QuantizedGGUFTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: QKVWeights::Separate {
            q: tensor_ref(0, 80, hidden_dim * hidden_dim, GGUF_TYPE_Q4_K),
            k: tensor_ref(80, 40, hidden_dim * kv_dim, GGUF_TYPE_Q4_K),
            v: tensor_ref(120, 40, hidden_dim * kv_dim, GGUF_TYPE_Q4_K),
        },
        qkv_bias: None,
        attn_output_weight: tensor_ref(160, 50, hidden_dim * hidden_dim, GGUF_TYPE_Q4_K),
        attn_output_bias: None,
        ffn_up_weight: tensor_ref(210, 80, hidden_dim * 128, GGUF_TYPE_Q4_K),
        ffn_up_bias: None,
        ffn_down_weight: tensor_ref(290, 80, 128 * hidden_dim, GGUF_TYPE_Q4_K),
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let data = create_test_data(450);
    let owned = OwnedQuantizedLayer::from_borrowed(&layer, &data, &config);

    // Verify QKV is Separate variant
    match &owned.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => {
            assert_eq!(q.out_dim, hidden_dim);
            assert_eq!(k.out_dim, kv_dim);
            assert_eq!(v.out_dim, kv_dim);
        },
        OwnedQKVWeights::Fused(_) => panic!("Expected Separate variant"),
    }

    // Verify total QKV dim
    assert_eq!(owned.qkv_weight.out_dim(), hidden_dim + kv_dim + kv_dim);
    assert_eq!(owned.qkv_weight.q_dim(), hidden_dim);
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_owned_quantized_tensor_exact_bounds() {
    // Test with exact bounds (offset + byte_size = data.len())
    let tensor_ref = QuantizedTensorRef {
        offset: 5,
        byte_size: 5,
        num_elements: 10,
        qtype: GGUF_TYPE_Q8_0,
    };
    let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    let owned = OwnedQuantizedTensor::from_ref_with_dims(&tensor_ref, &data, 2, 5);

    assert_eq!(owned.data, &[5, 6, 7, 8, 9]);
    assert_eq!(owned.in_dim, 2);
    assert_eq!(owned.out_dim, 5);
    assert_eq!(owned.qtype, GGUF_TYPE_Q8_0);
}

#[test]
fn test_owned_quantized_tensor_zero_offset() {
    // Test with zero offset
    let tensor_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 4,
        num_elements: 4,
        qtype: GGUF_TYPE_Q6_K,
    };
    let data = vec![10, 20, 30, 40, 50];

    let owned = OwnedQuantizedTensor::from_ref_with_dims(&tensor_ref, &data, 2, 2);

    assert_eq!(owned.data, &[10, 20, 30, 40]);
    assert_eq!(owned.qtype, GGUF_TYPE_Q6_K);
}

#[test]
fn test_qkv_weights_fused_large_dimensions() {
    // Test Fused variant with realistic LLM dimensions
    let hidden_dim = 4096;
    let tensor = QuantizedTensorRef {
        offset: 0,
        byte_size: 1024 * 1024,
        num_elements: hidden_dim * hidden_dim * 3, // Q + K + V
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = QKVWeights::Fused(tensor);

    assert_eq!(qkv.out_dim(hidden_dim), hidden_dim * 3);
    assert_eq!(qkv.q_dim(hidden_dim), hidden_dim);
}

#[test]
fn test_qkv_weights_separate_gqa_dimensions() {
    // Test Separate variant with GQA (different K/V dims)
    let hidden_dim = 4096;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let q = QuantizedTensorRef {
        offset: 0,
        byte_size: 1024,
        num_elements: hidden_dim * hidden_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k = QuantizedTensorRef {
        offset: 1024,
        byte_size: 256,
        num_elements: hidden_dim * kv_dim,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v = QuantizedTensorRef {
        offset: 1280,
        byte_size: 256,
        num_elements: hidden_dim * kv_dim,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = QKVWeights::Separate { q, k, v };

    // out_dim = Q + K + V = 4096 + 1024 + 1024 = 6144
    assert_eq!(qkv.out_dim(hidden_dim), hidden_dim + kv_dim + kv_dim);
    assert_eq!(qkv.q_dim(hidden_dim), hidden_dim);
}

#[test]
fn test_owned_qkv_weights_fused_large_dim() {
    // Test Fused variant conversions with larger dimensions
    let hidden_dim = 256;
    let qkv_dim = hidden_dim * 3;

    let tensor = QuantizedTensorRef {
        offset: 0,
        byte_size: 500,
        num_elements: hidden_dim * qkv_dim,
        qtype: GGUF_TYPE_Q4_K,
    };

    let borrowed = QKVWeights::Fused(tensor);
    let data = create_test_data(600);
    let owned = OwnedQKVWeights::from_borrowed(&borrowed, &data, hidden_dim);

    assert_eq!(owned.out_dim(), qkv_dim);
    assert_eq!(owned.q_dim(), hidden_dim);
}

#[test]
fn test_quantized_tensor_ref_clone() {
    // Verify Clone implementation
    let original = QuantizedTensorRef {
        offset: 100,
        byte_size: 200,
        num_elements: 300,
        qtype: GGUF_TYPE_Q4_K,
    };

    let cloned = original.clone();

    assert_eq!(cloned.offset, original.offset);
    assert_eq!(cloned.byte_size, original.byte_size);
    assert_eq!(cloned.num_elements, original.num_elements);
    assert_eq!(cloned.qtype, original.qtype);
}

#[test]
fn test_qkv_weights_clone() {
    // Verify Clone implementation for QKVWeights
    let tensor = QuantizedTensorRef {
        offset: 0,
        byte_size: 100,
        num_elements: 100,
        qtype: GGUF_TYPE_Q4_K,
    };

    let original = QKVWeights::Fused(tensor);
    let cloned = original.clone();

    assert_eq!(cloned.out_dim(10), original.out_dim(10));
}

include!("quantized_tests_owned.rs");
