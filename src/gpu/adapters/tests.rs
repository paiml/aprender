//! Phase 42 - Adapter Coverage Tests
//!
//! Tests CPU-only logic in adapters without requiring CUDA.

use super::apr::{transpose_matrix, AprGpuError, AprToGpuAdapter};
use crate::apr_transformer::{AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4};
use crate::gpu::scheduler::GpuModelConfig;

/// Build a minimal GpuModelConfig for tests (no explicit_head_dim, no MoE)
fn test_gpu_config(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
) -> GpuModelConfig {
    GpuModelConfig {
        vocab_size: 32000,
        hidden_dim,
        num_heads,
        num_kv_heads,
        num_layers: 1,
        intermediate_dim,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    }
}

// ============================================================================
// AprGpuError Tests
// ============================================================================

#[test]
fn test_apr_gpu_error_dequant() {
    let err = AprGpuError::DequantError("test error".to_string());
    assert!(err.to_string().contains("dequantize"));
    assert!(err.to_string().contains("test error"));
}

#[test]
fn test_apr_gpu_error_dimension_mismatch() {
    let err = AprGpuError::DimensionMismatch {
        expected: 100,
        actual: 50,
    };
    assert!(err.to_string().contains("100"));
    assert!(err.to_string().contains("50"));
}

#[test]
fn test_apr_gpu_error_gpu_model() {
    let err = AprGpuError::GpuModelError("allocation failed".to_string());
    assert!(err.to_string().contains("GpuModel"));
    assert!(err.to_string().contains("allocation failed"));
}

#[test]
fn test_apr_gpu_error_debug() {
    let err = AprGpuError::DequantError("test".to_string());
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("DequantError"));
}

// ============================================================================
// transpose_matrix Tests (Phase 42)
// ============================================================================

#[test]
fn test_transpose_square() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_matrix(&data, 2, 2);
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_rectangular() {
    // 2x3 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_matrix(&data, 2, 3);
    // Becomes 3x2
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_tall() {
    // 3x2 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_matrix(&data, 3, 2);
    // Becomes 2x3
    assert_eq!(result, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_transpose_single_row() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_matrix(&data, 1, 4);
    // 1x4 becomes 4x1
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_single_col() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = transpose_matrix(&data, 4, 1);
    // 4x1 becomes 1x4
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_identity() {
    let data = vec![1.0, 0.0, 0.0, 1.0];
    let result = transpose_matrix(&data, 2, 2);
    // Identity transposes to itself
    assert_eq!(result, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_transpose_large() {
    // 4x8 matrix
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let result = transpose_matrix(&data, 4, 8);

    // Verify dimensions preserved (8x4)
    assert_eq!(result.len(), 32);

    // Verify specific elements: (0,0) -> (0,0), (0,7) -> (7,0), (3,0) -> (0,3)
    assert_eq!(result[0], 0.0); // (0,0)
    assert_eq!(result[4 * 7], 7.0); // (7,0) was (0,7)
    assert_eq!(result[3], 24.0); // (0,3) was (3,0)
}

#[test]
#[should_panic(expected = "input is empty")]
fn test_transpose_empty() {
    let data: Vec<f32> = vec![];
    let _result = transpose_matrix(&data, 0, 0);
}

#[test]
fn test_transpose_single_element() {
    let data = vec![42.0];
    let result = transpose_matrix(&data, 1, 1);
    assert_eq!(result, vec![42.0]);
}

// ============================================================================
// AprToGpuAdapter Config Tests
// ============================================================================

#[test]
fn test_config_to_gpu_all_fields() {
    let apr_config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128256,
        intermediate_dim: 14336,
        context_length: 4096,
        rope_theta: 500000.0,
        eps: 1e-6,
        eos_token_id: None,
        ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.vocab_size, 128256);
    assert_eq!(gpu_config.hidden_dim, 4096);
    assert_eq!(gpu_config.num_heads, 32);
    assert_eq!(gpu_config.num_kv_heads, 8);
    assert_eq!(gpu_config.num_layers, 32);
    assert_eq!(gpu_config.intermediate_dim, 14336);
    assert_eq!(gpu_config.eps, 1e-6);
    assert_eq!(gpu_config.rope_theta, 500000.0);
}

#[test]
fn test_config_to_gpu_gqa() {
    // GQA: num_kv_heads < num_heads
    let apr_config = AprTransformerConfig {
        architecture: "mistral".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        vocab_size: 32000,
        intermediate_dim: 14336,
        context_length: 32768,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    // GQA should be preserved
    assert_eq!(gpu_config.num_heads, 32);
    assert_eq!(gpu_config.num_kv_heads, 8);
}

#[test]
fn test_config_to_gpu_mha() {
    // MHA: num_kv_heads == num_heads
    let apr_config = AprTransformerConfig {
        architecture: "gpt2".to_string(),
        hidden_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 12, // MHA
        vocab_size: 50257,
        intermediate_dim: 3072,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.num_heads, gpu_config.num_kv_heads);
}

#[test]
fn test_config_to_gpu_tiny_model() {
    let apr_config = AprTransformerConfig {
        architecture: "tiny".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.hidden_dim, 64);
    assert_eq!(gpu_config.vocab_size, 100);
}

// ============================================================================
// AprToGpuAdapter Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_tensor_empty() {
    let result = AprToGpuAdapter::dequantize_tensor(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_dequantize_tensor_padding() {
    // Q4_0 block: 2 bytes scale + 16 bytes nibbles = 18 bytes for 32 values
    // Create minimal valid Q4_0 data
    let mut data = vec![0u8; 18];
    // Set scale to 1.0 (f16)
    data[0] = 0x00;
    data[1] = 0x3c; // f16 1.0

    let result = AprToGpuAdapter::dequantize_tensor(&data, 64);
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 64); // Padded to expected
}

#[test]
fn test_dequantize_tensor_truncation() {
    // Create 2 Q4_0 blocks (64 values)
    let mut data = vec![0u8; 36];
    data[0] = 0x00;
    data[1] = 0x3c;
    data[18] = 0x00;
    data[19] = 0x3c;

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 32); // Truncated to expected
}

#[test]
fn test_dequantize_tensor_exact_size() {
    // Create 1 Q4_0 block (32 values)
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x3c;

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 32);
}

// ============================================================================
// AprToGpuAdapter Weight Extraction Tests
// ============================================================================

#[test]
fn test_extract_qkv_weights_dimensions() {
    let layer = create_test_q4_layer(256, 4, 4, 512);
    let config = test_gpu_config(256, 4, 4, 512);

    // QKV: num_heads*head_dim + 2 * kv_dim = 256 + 2*256 = 768
    // Total: hidden_dim * qkv_out = 256 * 768 = 196608
    let result = AprToGpuAdapter::extract_qkv_weights(&layer, &config);
    assert!(result.is_ok());

    let weights = result.unwrap();
    assert_eq!(weights.len(), 256 * 768);
}

#[test]
fn test_extract_qkv_weights_gqa() {
    let layer = create_test_q4_layer(256, 8, 2, 512); // GQA: 8 heads, 2 kv heads
    let config = test_gpu_config(256, 8, 2, 512);

    // QKV: num_heads*head_dim + 2 * kv_dim = 256 + 2*64 = 384
    let result = AprToGpuAdapter::extract_qkv_weights(&layer, &config);
    assert!(result.is_ok());

    let weights = result.unwrap();
    // hidden_dim * qkv_out = 256 * 384 = 98304
    assert_eq!(weights.len(), 256 * 384);
}

#[test]
fn test_extract_out_weights() {
    let layer = create_test_q4_layer(256, 4, 4, 512);
    let config = test_gpu_config(256, 4, 4, 512);

    let result = AprToGpuAdapter::extract_out_weights(&layer, &config);
    assert!(result.is_ok());

    let weights = result.unwrap();
    // q_dim = num_heads * head_dim = 4 * 64 = 256 = hidden_dim (no explicit head_dim)
    assert_eq!(weights.len(), 256 * 256);
}

#[test]
fn test_extract_ffn_weights() {
    let layer = create_test_q4_layer(256, 4, 4, 512);

    let result = AprToGpuAdapter::extract_ffn_weights(&layer, 256, 512);
    assert!(result.is_ok());

    let (fc1, fc2) = result.unwrap();
    assert_eq!(fc1.len(), 256 * 512); // up projection
    assert_eq!(fc2.len(), 512 * 256); // down projection
}

#[test]
fn test_extract_ffn_weights_large() {
    let layer = create_test_q4_layer(1024, 16, 16, 4096);

    let result = AprToGpuAdapter::extract_ffn_weights(&layer, 1024, 4096);
    assert!(result.is_ok());

    let (fc1, fc2) = result.unwrap();
    assert_eq!(fc1.len(), 1024 * 4096);
    assert_eq!(fc2.len(), 4096 * 1024);
}

include!("tests_gpu_model_q4.rs");
