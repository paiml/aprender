//! Integration Tests: GPU APR Model Loading (PMAT-106 + PMAT-803)
//!
//! These tests verify GPU inference with APR models while driving coverage for:
#![allow(clippy::manual_div_ceil)]
//! - `api/openai_handlers.rs` - API request handling
//! - `apr_transformer/q4_simd.rs` - Q4 transformer inference
//! - `gpu/scheduler/batch.rs` - GPU batch scheduling
//! - `gpu/adapters/apr.rs` - APR to GPU conversion
//!
//! # Coverage Strategy
//!
//! By testing the full inference path from API to GPU, we achieve broad coverage
//! without writing individual unit tests for each module.

#[allow(unused_imports)]
use realizar::apr_transformer::{
    AprInferenceScratch, AprKVCache, AprTransformerConfig, QuantizedAprLayerQ4,
    QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};
use realizar::gpu::adapters::{AprGpuError, AprToGpuAdapter};
use realizar::gpu::scheduler::{BlockWeights, GpuModelConfig};
#[allow(unused_imports)]
use realizar::quantize::dequantize_q4_0;

// ============================================================================
// Config Conversion Tests
// ============================================================================

#[test]
fn test_apr_config_to_gpu_config() {
    let apr_config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 4, // GQA
        vocab_size: 32000,
        intermediate_dim: 5632,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.vocab_size, 32000);
    assert_eq!(gpu_config.hidden_dim, 2048);
    assert_eq!(gpu_config.num_heads, 32);
    assert_eq!(gpu_config.num_kv_heads, 4);
    assert_eq!(gpu_config.num_layers, 22);
    assert_eq!(gpu_config.intermediate_dim, 5632);
    assert_eq!(gpu_config.eps, 1e-5);

    // Derived dimensions
    assert_eq!(gpu_config.head_dim(), 64); // 2048 / 32
    assert_eq!(gpu_config.kv_dim(), 256); // 4 * 64
}

#[test]
fn test_apr_config_mha_to_gpu() {
    // Standard MHA (not GQA)
    let apr_config = AprTransformerConfig {
        architecture: "phi2".to_string(),
        hidden_dim: 2560,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32, // MHA: same as num_heads
        vocab_size: 51200,
        intermediate_dim: 10240,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.num_heads, gpu_config.num_kv_heads);
    assert_eq!(gpu_config.kv_dim(), gpu_config.hidden_dim); // For MHA, kv_dim == hidden_dim
}

// ============================================================================
// Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_tensor_basic() {
    // Create a minimal Q4_0 block (18 bytes = 2 scale + 16 data)
    // This represents 32 values
    let mut data = vec![0u8; 18];
    // Set scale to 1.0 (f16: 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // All quants are 0 (which maps to -8 after dequant with 4-bit signed)

    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());

    let values = result.unwrap();
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_tensor_padding() {
    // Test that we pad if dequantized result is too short
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x3C;

    // Request more elements than available
    let result = AprToGpuAdapter::dequantize_tensor(&data, 64);
    assert!(result.is_ok());

    let values = result.unwrap();
    assert_eq!(values.len(), 64);
    // Extra values should be zero-padded
    for &v in &values[32..] {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_dequantize_tensor_truncation() {
    // Test that we truncate if dequantized result is too long
    let mut data = vec![0u8; 36]; // 2 blocks = 64 values
                                  // First block
    data[0] = 0x00;
    data[1] = 0x3C;
    // Second block
    data[18] = 0x00;
    data[19] = 0x3C;

    // Request fewer elements than available
    let result = AprToGpuAdapter::dequantize_tensor(&data, 32);
    assert!(result.is_ok());

    let values = result.unwrap();
    assert_eq!(values.len(), 32);
}

// ============================================================================
// Weight Extraction Tests
// ============================================================================

#[test]
fn test_extract_qkv_weights_gqa() {
    // Create a minimal layer for GQA (num_heads=8, num_kv_heads=2, head_dim=64)
    let hidden_dim = 512; // 8 heads * 64 dim
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim; // 128
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 512 + 256 = 768

    let total_elements = hidden_dim * qkv_out_dim;
    let num_blocks = (total_elements + 31) / 32;
    let data_size = num_blocks * 18;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; data_size], hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], hidden_dim, 1024),
        ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], 1024, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: None,
    };

    let config = GpuModelConfig {
        hidden_dim,
        intermediate_dim: 1024,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size: 50,
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
    };
    let result = AprToGpuAdapter::extract_qkv_weights(&layer, &config);
    assert!(result.is_ok());

    let weights = result.unwrap();
    assert_eq!(weights.len(), total_elements);
}

#[test]
fn test_extract_out_weights() {
    let hidden_dim = 256;
    let total_elements = hidden_dim * hidden_dim;
    let num_blocks = (total_elements + 31) / 32;
    let data_size = num_blocks * 18;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; data_size], hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], hidden_dim, 512),
        ffn_down_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], 512, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: None,
    };

    let config = GpuModelConfig {
        hidden_dim,
        intermediate_dim: 512,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
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
    };
    let result = AprToGpuAdapter::extract_out_weights(&layer, &config);
    assert!(result.is_ok());

    let weights = result.unwrap();
    assert_eq!(weights.len(), total_elements);
}

#[test]
fn test_extract_ffn_weights() {
    let hidden_dim = 256;
    let intermediate_dim = 1024;

    let fc1_elements = hidden_dim * intermediate_dim;
    let fc2_elements = intermediate_dim * hidden_dim;
    let fc1_blocks = (fc1_elements + 31) / 32;
    let fc2_blocks = (fc2_elements + 31) / 32;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::new(vec![0u8; 18], hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::new(
            vec![0u8; fc1_blocks * 18],
            hidden_dim,
            intermediate_dim,
        ),
        ffn_down_weight: QuantizedAprTensorQ4::new(
            vec![0u8; fc2_blocks * 18],
            intermediate_dim,
            hidden_dim,
        ),
        ffn_gate_weight: None,
        ffn_norm_weight: None,
    };

    let result = AprToGpuAdapter::extract_ffn_weights(&layer, hidden_dim, intermediate_dim);
    assert!(result.is_ok());

    let (fc1, fc2) = result.unwrap();
    assert_eq!(fc1.len(), fc1_elements);
    assert_eq!(fc2.len(), fc2_elements);
}

// ============================================================================
// Block Weights Creation Tests
// ============================================================================

#[test]
fn test_block_weights_creation() {
    let hidden_dim = 128;
    let intermediate_dim = 512;

    let block = BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
        qkv_bias: vec![],
        out_weight: vec![0.0; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.0; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.0; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: None, // No SwiGLU gate for this test
        linear_attn: None,
        moe_experts: None,
    };

    assert_eq!(block.attn_norm_weight.len(), hidden_dim);
    assert_eq!(block.qkv_weight.len(), hidden_dim * 3 * hidden_dim);
    assert_eq!(block.ffn_fc1_weight.len(), hidden_dim * intermediate_dim);
}

// ============================================================================
// GPU Model Config Tests (Coverage for gpu/scheduler/model.rs)
// ============================================================================

#[test]
fn test_gpu_model_config_derived() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 32,
        intermediate_dim: 11008,
        eps: 1e-6,
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
    };

    // Test derived dimensions
    assert_eq!(config.head_dim(), 128); // 4096 / 32
    assert_eq!(config.kv_dim(), 1024); // 8 * 128
    assert_eq!(config.qkv_dim(), 4096 + 2 * 1024); // hidden + 2*kv = 6144
}

#[test]
fn test_gpu_model_config_mha() {
    let config = GpuModelConfig {
        vocab_size: 50257,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12, // MHA
        num_layers: 12,
        intermediate_dim: 3072,
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
    };

    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.kv_dim(), 768);
    assert_eq!(config.qkv_dim(), 768 * 3); // MHA: 3 * hidden_dim
}

// ============================================================================
// APR Infrastructure Tests (Coverage for apr_transformer modules)
// ============================================================================

#[test]
fn test_apr_inference_scratch_creation() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 512,
        num_layers: 6,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 2048,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 512);
    assert_eq!(scratch.normed.len(), 512);
    assert_eq!(scratch.attn_out.len(), 512);
    assert_eq!(scratch.ffn_up.len(), 2048);
    assert_eq!(scratch.ffn_gate.len(), 2048);
    assert_eq!(scratch.ffn_out.len(), 512);
}

#[test]
fn test_apr_kv_cache_creation() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let cache = AprKVCache::new(&config);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 128);
    assert_eq!(cache.num_kv_heads(), 2);
    assert_eq!(cache.head_dim(), 64); // 256 / 4
}

#[test]
fn test_apr_kv_cache_append() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        vocab_size: 1000,
        intermediate_dim: 256,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let mut cache = AprKVCache::new(&config);
    let kv_size = 2 * 32; // num_kv_heads * head_dim = 2 * (128/4) = 64

    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Append to all layers
    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Verify retrieval
    let (cached_k, cached_v) = cache.get(0);
    assert_eq!(cached_k.len(), kv_size);
    assert_eq!(cached_v.len(), kv_size);
}

// ============================================================================
// Q4_0 Tensor Tests (Coverage for apr_transformer/q4_simd.rs)
// ============================================================================

#[test]
fn test_quantized_tensor_zeros() {
    let tensor = QuantizedAprTensorQ4::zeros(256, 512);

    assert_eq!(tensor.in_dim, 256);
    assert_eq!(tensor.out_dim, 512);

    // Verify all data is zero
    assert!(tensor.data.iter().all(|&b| b == 0));

    // Verify expected bytes
    let expected_bytes = QuantizedAprTensorQ4::expected_bytes(256 * 512);
    assert_eq!(tensor.data.len(), expected_bytes);
}

#[test]
fn test_quantized_tensor_expected_bytes() {
    // Q4_0: 18 bytes per 32 values
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(1), 18); // 1 block
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36); // 2 blocks
}

#[test]
fn test_quantized_layer_creation() {
    let hidden_dim = 256;
    let intermediate_dim = 1024;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    };

    assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
    assert!(layer.ffn_gate_weight.is_some());
    assert!(layer.ffn_norm_weight.is_some());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_dequantize_empty_data() {
    let result = AprToGpuAdapter::dequantize_tensor(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_apr_gpu_error_display() {
    let err = AprGpuError::DequantError("test error".to_string());
    assert!(err.to_string().contains("test error"));

    let err = AprGpuError::DimensionMismatch {
        expected: 100,
        actual: 50,
    };
    assert!(err.to_string().contains("100"));
    assert!(err.to_string().contains("50"));

    let err = AprGpuError::GpuModelError("gpu error".to_string());
    assert!(err.to_string().contains("gpu error"));
}

// ============================================================================
// AprF32ToGpuAdapter Tests (Coverage for apr.rs F32 path)
// ============================================================================

use realizar::apr_transformer::{AprTransformer, AprTransformerLayer};
use realizar::gpu::adapters::AprF32ToGpuAdapter;

#[test]
fn test_f32_adapter_to_gpu_model_basic() {
    let apr = create_f32_transformer(1);
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok());
    let model = result.unwrap();

    assert_eq!(model.config().hidden_dim, 64);
    assert_eq!(model.config().vocab_size, 100);
    assert_eq!(model.config().num_layers, 1);
}

#[test]
fn test_f32_adapter_to_gpu_model_multi_layer() {
    let apr = create_f32_transformer(4);
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok());
    let model = result.unwrap();

    assert_eq!(model.config().num_layers, 4);
}

#[test]
fn test_f32_adapter_with_all_biases() {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    // Create layer with all optional biases
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: Some(vec![0.1; hidden_dim]),
        qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
        qkv_bias: Some(vec![0.02; 3 * hidden_dim]),
        attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
        attn_output_bias: Some(vec![0.03; hidden_dim]),
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
        ffn_up_bias: Some(vec![0.04; intermediate_dim]),
        ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
        ffn_down_bias: Some(vec![0.05; hidden_dim]),
        ffn_norm_weight: None,
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

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: Some(vec![0.06; hidden_dim]),
        lm_head_weight: vec![0.0; hidden_dim * vocab_size],
        lm_head_bias: Some(vec![0.07; vocab_size]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());
}

#[test]
fn test_f32_adapter_without_biases() {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    // Create layer without optional biases (should use defaults)
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
        qkv_bias: None,
        attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
        ffn_down_bias: None,
        ffn_norm_weight: None,
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

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());
}

#[test]
fn test_f32_adapter_gqa_config() {
    // Test GQA configuration (num_kv_heads < num_heads)
    let hidden_dim = 128;
    let vocab_size = 200;
    let intermediate_dim = 512;
    let num_heads = 8;
    let num_kv_heads = 2; // GQA: 4x fewer KV heads
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: vec![0.0; hidden_dim * qkv_out_dim],
        qkv_bias: None,
        attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
        ffn_down_bias: None,
        ffn_norm_weight: None,
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

    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 2,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers: vec![layer.clone(), layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.config().num_heads, num_heads);
    assert_eq!(model.config().num_kv_heads, num_kv_heads);
}

#[test]
fn test_f32_adapter_larger_model() {
    // Test with dimensions closer to real models
    let apr = create_f32_transformer_large();
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok());
    let model = result.unwrap();

    assert_eq!(model.config().hidden_dim, 512);
    assert_eq!(model.config().num_layers, 6);
}

// ============================================================================
// F32 Adapter Helper Functions
// ============================================================================

fn create_f32_transformer(num_layers: usize) -> AprTransformer {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    let layers: Vec<AprTransformerLayer> = (0..num_layers)
        .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
        .collect();

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

fn create_f32_transformer_large() -> AprTransformer {
    let hidden_dim = 512;
    let vocab_size = 1000;
    let intermediate_dim = 2048;
    let num_layers = 6;

    let layers: Vec<AprTransformerLayer> = (0..num_layers)
        .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
        .collect();

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size,
            intermediate_dim,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}
