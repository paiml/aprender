//! GGUF Tests Part 27: T-COV-95 Deep Coverage Bridge
//!
//! Tests for loader.rs uncovered paths:
//! - get_tensor_f32: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K dequantization branches
//! - get_tensor_f32: tensor not found, unsupported qtype errors
//! - rope_type: architecture-based inference (NORM vs NEOX)
//! - rope_type: explicit rope.scaling.type metadata
//! - decode: no-vocabulary fallback to ASCII
//! - GGUFTransformer: tied embeddings (no output.weight)
//!
//! Refs PMAT-802: Protocol T-COV-95

use crate::gguf::test_factory::*;
use crate::gguf::GGUFModel;

// ============================================================================
// get_tensor_f32: Q4_0 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q4_0() {
    let n = 32; // one block
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_0_tensor("test.weight", &[n as u64], &create_q4_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q4_0 get_tensor_f32 failed: {:?}",
        values.err()
    );
    let v = values.expect("test value should be present");
    assert_eq!(v.len(), n);
}

#[test]
fn test_get_tensor_f32_q4_0_multi_block() {
    let n = 128; // 4 blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_0_tensor("test.weight", &[n as u64], &create_q4_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q8_0 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q8_0() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q8_0_tensor("test.weight", &[n as u64], &create_q8_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q8_0 get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

#[test]
fn test_get_tensor_f32_q8_0_multi_block() {
    let n = 256; // 8 blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q8_0_tensor("test.weight", &[n as u64], &create_q8_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q4_K dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q4_k() {
    let n = 256; // one super-block
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_k_tensor("test.weight", &[n as u64], &create_q4_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q4_K get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

#[test]
fn test_get_tensor_f32_q4_k_multi_block() {
    let n = 512; // 2 super-blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_k_tensor("test.weight", &[n as u64], &create_q4_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q5_K dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q5_k() {
    let n = 256;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_k_tensor("test.weight", &[n as u64], &create_q5_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q5_K get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

// ============================================================================
// get_tensor_f32: Q6_K dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q6_k() {
    let n = 256;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q6_k_tensor("test.weight", &[n as u64], &create_q6_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q6_K get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

// ============================================================================
// get_tensor_f32: F32 branch (already partially covered, additional edge case)
// ============================================================================

#[test]
fn test_get_tensor_f32_f32_2d() {
    let rows = 4u64;
    let cols = 8u64;
    let n = (rows * cols) as usize;
    let f32_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("test.matrix", &[rows, cols], &f32_data)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.matrix", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
    // Verify values are preserved
    assert!((values[0] - 0.0).abs() < 1e-6);
    assert!((values[1] - 0.1).abs() < 1e-6);
    assert!((values[n - 1] - (n - 1) as f32 * 0.1).abs() < 1e-4);
}

// ============================================================================
// get_tensor_f32: error paths
// ============================================================================

#[test]
fn test_get_tensor_f32_not_found() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("existing.weight", &[4], &vec![1.0f32; 4])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let result = model.get_tensor_f32("nonexistent.weight", &data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not found") || err.contains("Tensor"));
}

#[test]
fn test_get_tensor_f32_multiple_tensors_select() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("alpha.weight", &[4], &vec![1.0f32; 4])
        .add_f32_tensor("beta.weight", &[8], &vec![2.0f32; 8])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    let alpha = model
        .get_tensor_f32("alpha.weight", &data)
        .expect("test value should be present");
    assert_eq!(alpha.len(), 4);
    assert!((alpha[0] - 1.0).abs() < 1e-6);

    let beta = model
        .get_tensor_f32("beta.weight", &data)
        .expect("test value should be present");
    assert_eq!(beta.len(), 8);
    assert!((beta[0] - 2.0).abs() < 1e-6);
}

// ============================================================================
// rope_type: architecture-based inference
// ============================================================================

#[test]
fn test_rope_type_llama_returns_norm() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let rope = model.rope_type();
    assert_eq!(rope, Some(0)); // NORM style
}

#[test]
fn test_rope_type_qwen2_returns_neox() {
    let data = GGUFBuilder::new()
        .architecture("qwen2")
        .hidden_dim("qwen2", 32)
        .num_layers("qwen2", 1)
        .num_heads("qwen2", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let rope = model.rope_type();
    assert_eq!(rope, Some(2)); // NEOX style
}

#[test]
fn test_rope_type_phi3_returns_neox() {
    let data = GGUFBuilder::new()
        .architecture("phi3")
        .hidden_dim("phi3", 32)
        .num_layers("phi3", 1)
        .num_heads("phi3", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2));
}

#[test]
fn test_rope_type_gemma_returns_neox() {
    let data = GGUFBuilder::new()
        .architecture("gemma")
        .hidden_dim("gemma", 32)
        .num_layers("gemma", 1)
        .num_heads("gemma", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2));
}

#[test]
fn test_rope_type_falcon_returns_neox() {
    let data = GGUFBuilder::new()
        .architecture("falcon")
        .hidden_dim("falcon", 32)
        .num_layers("falcon", 1)
        .num_heads("falcon", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2));
}

#[test]
fn test_rope_type_stablelm_returns_neox() {
    let data = GGUFBuilder::new()
        .architecture("stablelm")
        .hidden_dim("stablelm", 32)
        .num_layers("stablelm", 1)
        .num_heads("stablelm", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2));
}

#[test]
fn test_rope_type_deepseek2_returns_neox() {
    let data = GGUFBuilder::new()
        .architecture("deepseek2")
        .hidden_dim("deepseek2", 32)
        .num_layers("deepseek2", 1)
        .num_heads("deepseek2", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(2));
}

#[test]
fn test_rope_type_unknown_arch_defaults_to_norm() {
    let data = GGUFBuilder::new()
        .architecture("custom_model")
        .hidden_dim("custom_model", 32)
        .num_layers("custom_model", 1)
        .num_heads("custom_model", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.rope_type(), Some(0)); // defaults to NORM
}

// ============================================================================
// decode: no vocabulary fallback
// ============================================================================

#[test]
fn test_decode_no_vocabulary_fallback() {
    // Model with no vocabulary metadata → fallback to ASCII
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert!(model.vocabulary().is_none());

    // decode should use ASCII fallback
    let text = model.decode(&[72, 101, 108, 108, 111]); // H, e, l, l, o
    assert_eq!(text, "Hello");
}

#[test]
fn test_decode_no_vocabulary_high_values() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");

    // Values > 127 are clamped to 127 then converted
    let text = model.decode(&[200, 300]);
    assert_eq!(text.len(), 2);
}

#[test]
fn test_encode_no_vocabulary_returns_none() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert!(model.encode("Hello").is_none());
}

include!("transformer.rs");
