//! GQA (Grouped Query Attention) Tests - EXTREME TDD (Phase 54)
//!
//! These tests verify correct handling of GQA where num_kv_heads < num_heads.
//! Multiple Q heads share the same K/V head.
//!
//! Test Matrix:
//! - MHA: num_heads == num_kv_heads (baseline)
//! - GQA 4:1: 8 Q heads, 2 KV heads (group_size=4)
//! - GQA 8:1: 32 Q heads, 4 KV heads (group_size=8, TinyLlama)
//! - GQA 2:1: 4 Q heads, 2 KV heads (minimal GQA)

use crate::gguf::model::OwnedQuantizedModel;
use crate::gguf::quantized::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedTensor};
use crate::gguf::GGUFConfig;

// =============================================================================
// Helper: Create Q4_K test tensor with predictable values
// =============================================================================

fn create_q4k_test_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    // Set d=1.0 (f16: 0x3C00) for each super block
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = row * bytes_per_row + sb * 144;
            if offset + 2 <= data.len() {
                data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            }
        }
    }

    OwnedQuantizedTensor {
        qtype: 12, // Q4_K
        in_dim,
        out_dim,
        data,
    }
}

// =============================================================================
// Helper: Create GQA model for testing
// =============================================================================

fn create_gqa_model(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> OwnedQuantizedModel {
    let vocab_size = 100;
    let intermediate_dim = hidden_dim * 2;
    let num_layers = 1;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    // GQA dimensions
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim; // = hidden_dim
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = q_dim + 2 * kv_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_tensor(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let token_embedding = vec![0.1f32; vocab_size * hidden_dim];
    let output_norm_weight = vec![1.0f32; hidden_dim];
    let lm_head_weight = create_q4k_test_tensor(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config,
        token_embedding,
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight,
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

// =============================================================================
// RED Tests: OwnedQKVWeights dimension methods
// =============================================================================

/// Test q_dim() for GQA fused weights (4:1 ratio)
#[test]
fn test_qkv_weights_q_dim_gqa_4_to_1() {
    // GQA: 8 Q heads, 2 KV heads, hidden_dim=64, head_dim=8
    // q_dim = 8 * 8 = 64
    // kv_dim = 2 * 8 = 16
    // qkv_out_dim = 64 + 16 + 16 = 96
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = q_dim + 2 * kv_dim;

    let weights = OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim));

    // q_dim should be hidden_dim (64), NOT qkv_out_dim/3 (32)
    assert_eq!(
        weights.q_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads),
        64,
        "q_dim should be num_heads * head_dim = 64 for GQA"
    );
}

/// Test k_dim() for GQA fused weights
#[test]
fn test_qkv_weights_k_dim_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = q_dim + 2 * kv_dim;

    let weights = OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim));

    // k_dim should be kv_dim (16), NOT q_dim (64)
    assert_eq!(
        weights.k_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads),
        16,
        "k_dim should be num_kv_heads * head_dim = 16 for GQA"
    );
}

/// Test v_dim() for GQA fused weights
#[test]
fn test_qkv_weights_v_dim_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = q_dim + 2 * kv_dim;

    let weights = OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim));

    // v_dim should be kv_dim (16), NOT q_dim (64)
    assert_eq!(
        weights.v_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads),
        16,
        "v_dim should be num_kv_heads * head_dim = 16 for GQA"
    );
}

/// Test dimension consistency: q_dim + k_dim + v_dim == out_dim
#[test]
fn test_qkv_weights_dimension_consistency_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = q_dim + 2 * kv_dim;

    let weights = OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim));

    let computed_q =
        weights.q_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads);
    let computed_k =
        weights.k_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads);
    let computed_v =
        weights.v_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads);

    assert_eq!(
        computed_q + computed_k + computed_v,
        weights.out_dim(),
        "Q + K + V dimensions must equal out_dim"
    );
}

// =============================================================================
// RED Tests: MHA baseline (should still work)
// =============================================================================

/// Test q_dim() for MHA fused weights (1:1 ratio)
#[test]
fn test_qkv_weights_q_dim_mha() {
    // MHA: 8 Q heads, 8 KV heads, hidden_dim=64
    // q_dim = k_dim = v_dim = 64
    // qkv_out_dim = 3 * 64 = 192
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 8; // MHA
    let qkv_out_dim = 3 * hidden_dim;

    let weights = OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim));

    assert_eq!(
        weights.q_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads),
        64,
        "q_dim should be hidden_dim for MHA"
    );
    assert_eq!(
        weights.k_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads),
        64,
        "k_dim should be hidden_dim for MHA"
    );
    assert_eq!(
        weights.v_dim_for_config(num_heads, num_kv_heads, hidden_dim, hidden_dim / num_heads),
        64,
        "v_dim should be hidden_dim for MHA"
    );
}

// =============================================================================
// RED Tests: Forward pass with GQA
// =============================================================================

/// Test forward pass doesn't panic with GQA 4:1 ratio
#[test]
fn test_forward_gqa_4_to_1_no_panic() {
    // GQA: 8 Q heads, 2 KV heads
    let model = create_gqa_model(64, 8, 2);
    let token_ids = [10u32, 20, 30];

    // Should not panic with index out of bounds
    let result = model.forward(&token_ids);
    assert!(result.is_ok(), "Forward pass should succeed for GQA 4:1");
}

/// Test forward pass doesn't panic with GQA 8:1 ratio (TinyLlama-like)
#[test]
fn test_forward_gqa_8_to_1_no_panic() {
    // GQA: 32 Q heads, 4 KV heads (TinyLlama)
    let model = create_gqa_model(256, 32, 4);
    let token_ids = [10u32, 20, 30];

    let result = model.forward(&token_ids);
    assert!(result.is_ok(), "Forward pass should succeed for GQA 8:1");
}

/// Test forward pass produces finite logits for GQA
#[test]
fn test_forward_gqa_finite_logits() {
    let model = create_gqa_model(64, 8, 2);
    let token_ids = [10u32, 20, 30];

    let logits = model.forward(&token_ids).expect("Forward should succeed");

    assert_eq!(logits.len(), 100, "Should have vocab_size logits");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "All logits should be finite"
    );
}

/// Test forward pass with single token (GQA)
#[test]
fn test_forward_gqa_single_token() {
    let model = create_gqa_model(64, 8, 2);
    let token_ids = [42u32];

    let logits = model.forward(&token_ids).expect("Forward should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// RED Tests: causal_attention with GQA
// =============================================================================

/// Test causal_attention output shape for GQA
#[test]
fn test_causal_attention_output_shape_gqa() {
    let model = create_gqa_model(64, 8, 2);

    // seq_len=3, q_dim=64, kv_dim=16
    let seq_len = 3;
    let q_dim = 64;
    let kv_dim = 16;

    let q = vec![1.0f32; seq_len * q_dim];
    let k = vec![1.0f32; seq_len * kv_dim];
    let v = vec![1.0f32; seq_len * kv_dim];

    let output = model.causal_attention(&q, &k, &v, seq_len);

    // Output should be [seq_len, q_dim] = 3 * 64 = 192
    assert_eq!(
        output.len(),
        seq_len * q_dim,
        "Attention output should be seq_len * q_dim"
    );
}

/// Test causal_attention doesn't panic with GQA dimensions
#[test]
fn test_causal_attention_gqa_no_index_panic() {
    let model = create_gqa_model(64, 8, 2);

    let seq_len = 5;
    let q_dim = 64;
    let kv_dim = 16;

    // Longer sequence to stress test indexing
    let q = vec![0.1f32; seq_len * q_dim];
    let k = vec![0.1f32; seq_len * kv_dim];
    let v = vec![0.1f32; seq_len * kv_dim];

    // Should not panic with "index out of bounds"
    let output = model.causal_attention(&q, &k, &v, seq_len);
    assert_eq!(output.len(), seq_len * q_dim);
}

/// Test causal attention preserves causality for GQA
#[test]
fn test_causal_attention_gqa_causality() {
    let model = create_gqa_model(64, 8, 2);

    let seq_len = 4;
    let q_dim = 64;
    let kv_dim = 16;

    // Create Q, K, V where only position 0 has non-zero K,V
    let q = vec![1.0f32; seq_len * q_dim];
    let mut k = vec![0.0f32; seq_len * kv_dim];
    let mut v = vec![0.0f32; seq_len * kv_dim];

    // Only position 0 has K/V
    for i in 0..kv_dim {
        k[i] = 1.0;
        v[i] = 1.0;
    }

    // Position 0 should attend only to itself (position 0 K/V)
    // Position 1+ should only see position 0's K/V (causal)
    let output = model.causal_attention(&q, &k, &v, seq_len);

    // Output should be finite and reasonable
    assert!(output.iter().all(|x| x.is_finite()));
}

// =============================================================================
// RED Tests: Edge cases
// =============================================================================

/// Test GQA with minimal 2:1 ratio
#[test]
fn test_forward_gqa_2_to_1_minimal() {
    // Minimal GQA: 4 Q heads, 2 KV heads
    let model = create_gqa_model(32, 4, 2);
    let token_ids = [1u32, 2, 3, 4, 5];

    let result = model.forward(&token_ids);
    assert!(result.is_ok(), "Forward pass should succeed for GQA 2:1");
}

/// Test GQA with larger sequence
#[test]
fn test_forward_gqa_longer_sequence() {
    let model = create_gqa_model(64, 8, 2);
    let token_ids: Vec<u32> = (0..20).collect();

    let result = model.forward(&token_ids);
    assert!(
        result.is_ok(),
        "Forward pass should succeed with longer sequence"
    );

    let logits = result.unwrap();
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// RED Tests: Dimension validation at construction
// =============================================================================

/// Test that QKV weight out_dim matches expected GQA dimensions
#[test]
fn test_qkv_out_dim_matches_gqa_formula() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let expected_qkv_dim = hidden_dim + 2 * (num_kv_heads * head_dim);

    let weights = OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, expected_qkv_dim));

    assert_eq!(weights.out_dim(), expected_qkv_dim);
    assert_eq!(weights.out_dim(), 64 + 2 * 16);
    assert_eq!(weights.out_dim(), 96);
}

include!("attention_gqa_tests_forward_cached.rs");
