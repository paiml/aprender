//! Part 08: GPU Adapter Extended Coverage Tests
//!
//! Additional tests for `gpu/adapters/` module covering:
//! - AprF32ToGpuAdapter full conversion path
//! - AprToGpuAdapter Q4 full conversion
//! - MockExecutor integration with adapters
//! - GpuModel creation from adapters
//! - Error edge cases
//! - Backend dispatch paths

use crate::apr_transformer::{
    AprTransformer, AprTransformerConfig, AprTransformerLayer, QuantizedAprLayerQ4,
    QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};
use crate::gpu::adapters::{transpose_matrix, AprF32ToGpuAdapter, AprToGpuAdapter};
use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::GpuGenerateConfig;

// ============================================================================
// Test Helper Functions
// ============================================================================

fn create_minimal_apr_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    }
}

fn create_gqa_apr_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "llama_gqa".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 4 Q heads per KV head
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    }
}

fn create_f32_layer(config: &AprTransformerConfig) -> AprTransformerLayer {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let intermediate_dim = config.intermediate_dim;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; hidden_dim * qkv_out_dim],
        qkv_bias: None,
        attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
        attn_output_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
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
        ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_down_bias: None,
        ffn_gate_weight: Some(vec![0.01; hidden_dim * intermediate_dim]),
        ffn_gate_bias: None,
    }
}

fn create_minimal_f32_apr() -> AprTransformer {
    let config = create_minimal_apr_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let num_layers = config.num_layers;

    AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: (0..num_layers).map(|_| create_f32_layer(&config)).collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

fn create_gqa_f32_apr() -> AprTransformer {
    let config = create_gqa_apr_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let num_layers = config.num_layers;

    AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: (0..num_layers).map(|_| create_f32_layer(&config)).collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

fn create_q4_layer(config: &AprTransformerConfig) -> QuantizedAprLayerQ4 {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let intermediate_dim = config.intermediate_dim;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    }
}

fn create_q4_layer_without_gate(config: &AprTransformerConfig) -> QuantizedAprLayerQ4 {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let intermediate_dim = config.intermediate_dim;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    }
}

fn create_minimal_q4_apr() -> QuantizedAprTransformerQ4 {
    let config = create_minimal_apr_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let num_layers = config.num_layers;

    QuantizedAprTransformerQ4 {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: (0..num_layers).map(|_| create_q4_layer(&config)).collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

fn create_q4_apr_without_gate() -> QuantizedAprTransformerQ4 {
    let config = create_minimal_apr_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let num_layers = config.num_layers;

    QuantizedAprTransformerQ4 {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: (0..num_layers)
            .map(|_| create_q4_layer_without_gate(&config))
            .collect(),
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

// ============================================================================
// AprF32ToGpuAdapter Comprehensive Tests
// ============================================================================

#[test]
fn test_apr_f32_to_gpu_basic_conversion() {
    let apr = create_minimal_f32_apr();
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok(), "F32 conversion should succeed");
    let gpu_model = result.expect("test value should be present");

    // Verify config is correctly transferred
    assert_eq!(gpu_model.config.vocab_size, 100);
    assert_eq!(gpu_model.config.hidden_dim, 64);
    assert_eq!(gpu_model.config.num_heads, 4);
    assert_eq!(gpu_model.config.num_kv_heads, 4);
    assert_eq!(gpu_model.config.num_layers, 2);
    assert_eq!(gpu_model.config.intermediate_dim, 128);
}

#[test]
fn test_apr_f32_to_gpu_gqa_config_preserved() {
    let apr = create_gqa_f32_apr();
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok());
    let gpu_model = result.expect("test value should be present");

    // GQA config should be preserved
    assert_eq!(gpu_model.config.num_heads, 8);
    assert_eq!(gpu_model.config.num_kv_heads, 2);
    assert!(gpu_model.config.is_gqa());
}

#[test]
fn test_apr_f32_to_gpu_with_optional_biases() {
    let mut config = create_minimal_apr_config();
    config.num_layers = 1; // PMAT-284: match actual layer count
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    // Create APR with all optional biases filled in
    let apr = AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: Some(vec![0.1; hidden_dim]),
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: Some(vec![0.01; 3 * hidden_dim]),
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: Some(vec![0.02; hidden_dim]),
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
            ffn_norm_bias: Some(vec![0.1; hidden_dim]),
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
            ffn_up_weight: vec![0.01; hidden_dim * 128],
            ffn_up_bias: Some(vec![0.01; 128]),
            ffn_down_weight: vec![0.01; 128 * hidden_dim],
            ffn_down_bias: Some(vec![0.01; hidden_dim]),
            ffn_gate_weight: None,
            ffn_gate_bias: None,
        }],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: Some(vec![0.1; hidden_dim]),
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: Some(vec![0.001; vocab_size]),
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok(), "Should handle optional biases");
}

#[test]
fn test_apr_f32_to_gpu_without_optional_biases() {
    let mut config = create_minimal_apr_config();
    config.num_layers = 1; // PMAT-284: match actual layer count
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    // Create APR with no optional biases (all None)
    let apr = AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_norm_weight: None, // Tests default creation
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
            ffn_up_weight: vec![0.01; hidden_dim * 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 128 * hidden_dim],
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
        }],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok(), "Should handle missing optional biases");
}

#[test]
fn test_apr_f32_to_gpu_with_swiglu_gate() {
    let apr = create_minimal_f32_apr(); // Has gate weight
    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok());
    // Model should be created successfully with SwiGLU gate
}

#[test]
fn test_apr_f32_to_gpu_without_swiglu_gate() {
    let mut config = create_minimal_apr_config();
    config.num_layers = 1; // PMAT-284: match actual layer count
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    let apr = AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
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
            ffn_up_weight: vec![0.01; hidden_dim * 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 128 * hidden_dim],
            ffn_down_bias: None,
            ffn_gate_weight: None, // No gate
            ffn_gate_bias: None,
        }],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = AprF32ToGpuAdapter::to_gpu_model(&apr);
    assert!(result.is_ok());
}

// ============================================================================
// AprToGpuAdapter Q4 Comprehensive Tests
// ============================================================================

#[test]
fn test_apr_q4_to_gpu_basic_conversion() {
    let apr = create_minimal_q4_apr();
    let result = AprToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok(), "Q4 conversion should succeed");
    let gpu_model = result.expect("test value should be present");

    assert_eq!(gpu_model.config.vocab_size, 100);
    assert_eq!(gpu_model.config.hidden_dim, 64);
}

#[test]
fn test_apr_q4_to_gpu_without_gate() {
    let apr = create_q4_apr_without_gate();
    let result = AprToGpuAdapter::to_gpu_model(&apr);

    assert!(result.is_ok());
}

#[test]
fn test_apr_q4_config_conversion_all_fields() {
    let apr_config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 512,
        num_layers: 8,
        num_heads: 16,
        num_kv_heads: 4,
        vocab_size: 32000,
        intermediate_dim: 2048,
        context_length: 4096,
        rope_theta: 500000.0,
        eps: 1e-6,
            eos_token_id: None,
    ..Default::default()
    };

    let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

    assert_eq!(gpu_config.vocab_size, 32000);
    assert_eq!(gpu_config.hidden_dim, 512);
    assert_eq!(gpu_config.num_heads, 16);
    assert_eq!(gpu_config.num_kv_heads, 4);
    assert_eq!(gpu_config.num_layers, 8);
    assert_eq!(gpu_config.intermediate_dim, 2048);
    assert_eq!(gpu_config.eps, 1e-6);
    assert_eq!(gpu_config.rope_theta, 500000.0);
}

#[test]
fn test_apr_q4_extract_qkv_mha() {
    let config = create_minimal_apr_config();
    let gpu_config = AprToGpuAdapter::config_to_gpu(&config);
    let layer = create_q4_layer(&config);

    let result = AprToGpuAdapter::extract_qkv_weights(&layer, &gpu_config);
    assert!(result.is_ok());

    let qkv_out_dim = gpu_config.qkv_dim();
    let expected_len = config.hidden_dim * qkv_out_dim;
    assert_eq!(result.expect("test value should be present").len(), expected_len);
}

#[test]
fn test_apr_q4_extract_qkv_gqa() {
    let config = create_gqa_apr_config();
    let gpu_config = AprToGpuAdapter::config_to_gpu(&config);
    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; config.hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.hidden_dim + 2 * 16),
        attn_output_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(config.hidden_dim, config.intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(config.intermediate_dim, config.hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: Some(vec![1.0; config.hidden_dim]),
    };

    let result = AprToGpuAdapter::extract_qkv_weights(&layer, &gpu_config);
    assert!(result.is_ok());

    // GQA: kv_dim = 2 * 8 = 16
    let qkv_out_dim = gpu_config.qkv_dim(); // 64 + 32 = 96
    let expected_len = config.hidden_dim * qkv_out_dim;
    assert_eq!(result.expect("test value should be present").len(), expected_len);
}

include!("transpose_3x3_2x4.rs");
