//! Phase 49 - GpuModel Coverage Tests (gpu/scheduler/model.rs)
//!
//! Comprehensive tests for GpuModel struct and methods:
//! - GpuModelConfig: head_dim, kv_dim, qkv_dim, is_gqa
//! - GpuGenerateConfig: constructors, with_stop_tokens
//! - AttentionBuffers: new, reset
//! - GpuModel: constructors, test_executor methods, matmul_split, forward/generate
//! - BlockWeights structure
//! - WeightType enum
//!
//! Strategy: Use MockExecutor for CPU logic paths without GPU hardware.

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig, WeightType,
};
use crate::gpu::StreamingKVCache;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test model configuration (MHA - Multi-Head Attention)
fn create_test_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4, // MHA (not GQA)
        vocab_size: 100,
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

/// Create a GQA test configuration (num_heads > num_kv_heads)
fn create_gqa_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 4 Q heads per KV head
        vocab_size: 100,
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

/// Create a minimal single-layer config for fast tests
fn create_minimal_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
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
    }
}

// ============================================================================
// GpuModelConfig Tests
// ============================================================================

#[test]
fn test_gpu_model_config_head_dim() {
    let config = create_test_config();
    // hidden_dim=64, num_heads=4 -> head_dim=16
    assert_eq!(config.head_dim(), 16);
}

#[test]
fn test_gpu_model_config_head_dim_gqa() {
    let config = create_gqa_config();
    // hidden_dim=64, num_heads=8 -> head_dim=8
    assert_eq!(config.head_dim(), 8);
}

#[test]
fn test_gpu_model_config_kv_dim_mha() {
    let config = create_test_config();
    // MHA: num_kv_heads=4, head_dim=16 -> kv_dim=64
    assert_eq!(config.kv_dim(), 64);
}

#[test]
fn test_gpu_model_config_kv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: num_kv_heads=2, head_dim=8 -> kv_dim=16
    assert_eq!(config.kv_dim(), 16);
}

#[test]
fn test_gpu_model_config_qkv_dim_mha() {
    let config = create_test_config();
    // MHA: hidden_dim + 2*kv_dim = 64 + 2*64 = 192
    assert_eq!(config.qkv_dim(), 192);
}

#[test]
fn test_gpu_model_config_qkv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: hidden_dim + 2*kv_dim = 64 + 2*16 = 96
    assert_eq!(config.qkv_dim(), 96);
}

#[test]
fn test_gpu_model_config_is_gqa_false() {
    let config = create_test_config();
    // MHA: num_heads == num_kv_heads
    assert!(!config.is_gqa());
}

#[test]
fn test_gpu_model_config_is_gqa_true() {
    let config = create_gqa_config();
    // GQA: num_heads > num_kv_heads
    assert!(config.is_gqa());
}

// ============================================================================
// GpuGenerateConfig Tests
// ============================================================================

#[test]
fn test_gpu_generate_config_default() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_deterministic() {
    let config = GpuGenerateConfig::deterministic(128);
    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_sampling() {
    let config = GpuGenerateConfig::with_sampling(64, 0.7, 40);
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_stop_tokens() {
    let config = GpuGenerateConfig::deterministic(32).with_stop_tokens(vec![0, 2, 3]);
    assert_eq!(config.max_tokens, 32);
    assert_eq!(config.stop_tokens, vec![0, 2, 3]);
}

#[test]
fn test_gpu_generate_config_with_sampling_and_stop_tokens() {
    let config =
        GpuGenerateConfig::with_sampling(100, 0.9, 50).with_stop_tokens(vec![1, 2, 3, 4, 5]);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.9);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens.len(), 5);
}

// ============================================================================
// AttentionBuffers Tests
// ============================================================================

#[test]
fn test_attention_buffers_new() {
    let config = create_test_config();
    let buffers = AttentionBuffers::new(&config, 512);

    assert_eq!(buffers.q_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.scores_buffer.len(), config.num_heads * 512);
    assert_eq!(buffers.output_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.kv_proj_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.ffn_buffer.len(), config.intermediate_dim);
    assert_eq!(buffers.max_seq_len, 512);
}

#[test]
fn test_attention_buffers_new_gqa() {
    let config = create_gqa_config();
    let buffers = AttentionBuffers::new(&config, 256);

    assert_eq!(buffers.q_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.scores_buffer.len(), config.num_heads * 256);
    assert_eq!(buffers.max_seq_len, 256);
}

#[test]
fn test_attention_buffers_reset() {
    let config = create_minimal_config();
    let mut buffers = AttentionBuffers::new(&config, 64);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);
    buffers.kv_proj_buffer.fill(4.0);
    buffers.ffn_buffer.fill(5.0);

    // Reset should clear all
    buffers.reset();

    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
}

// ============================================================================
// GpuModel Construction Tests
// ============================================================================

#[test]
fn test_gpu_model_new_basic() {
    let config = create_test_config();
    let model = GpuModel::new(config.clone());
    assert!(model.is_ok());

    let model = model.expect("test value should be present");
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.num_layers, 2);
    assert!(!model.has_test_executor());
}

#[test]
fn test_gpu_model_new_gqa() {
    let config = create_gqa_config();
    let model = GpuModel::new(config);
    assert!(model.is_ok());

    let model = model.expect("test value should be present");
    assert!(model.config.is_gqa());
}

#[test]
fn test_gpu_model_from_gguf_config() {
    let config = create_test_config();
    let model = GpuModel::from_gguf_config(config.clone());
    assert!(model.is_ok());

    let model = model.expect("test value should be present");
    assert_eq!(model.config.vocab_size, config.vocab_size);
}

#[test]
fn test_gpu_model_config_getter() {
    let config = create_test_config();
    let model = GpuModel::new(config).expect("test value should be present");

    let retrieved_config = model.config();
    assert_eq!(retrieved_config.hidden_dim, 64);
    assert_eq!(retrieved_config.vocab_size, 100);
}

// ============================================================================
// GpuModel Test Executor Tests
// ============================================================================

#[test]
fn test_gpu_model_with_test_executor() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    assert!(!model.has_test_executor());

    let mock = MockExecutor::new("test_executor");
    model.with_test_executor(Box::new(mock));

    assert!(model.has_test_executor());
}

#[test]
fn test_gpu_model_clear_test_executor() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("test_executor");
    model.with_test_executor(Box::new(mock));
    assert!(model.has_test_executor());

    model.clear_test_executor();
    assert!(!model.has_test_executor());
}

#[test]
fn test_gpu_model_has_gpu() {
    let config = create_minimal_config();
    let model = GpuModel::new(config).expect("test value should be present");
    // has_gpu() delegates to scheduler, should not panic
    let _ = model.has_gpu();
}

// ============================================================================
// GpuModel with_attention_buffers Tests
// ============================================================================

#[test]
fn test_gpu_model_with_attention_buffers() {
    let config = create_test_config();
    let model = GpuModel::with_attention_buffers(config, 512);
    assert!(model.is_ok());

    let model = model.expect("test value should be present");
    assert!(model.has_attention_buffers());
}

#[test]
fn test_gpu_model_has_attention_buffers_false() {
    let config = create_test_config();
    let model = GpuModel::new(config).expect("test value should be present");
    assert!(!model.has_attention_buffers());
}

// ============================================================================
// GpuModel Feature Flag Tests (has_fused_*)
// ============================================================================

#[test]
fn test_gpu_model_has_fused_qkv() {
    let config = create_test_config();
    let model = GpuModel::new(config).expect("test value should be present");

    // Test checks if QKV weights have expected dimensions
    let has_fused = model.has_fused_qkv();
    // New model has initialized weights
    let _ = has_fused; // Just verify it doesn't panic
}

#[test]
fn test_gpu_model_has_fused_attn_proj() {
    let config = create_test_config();
    let model = GpuModel::new(config).expect("test value should be present");

    // Test checks if attention projection weights have expected dimensions
    let has_fused = model.has_fused_attn_proj();
    assert!(has_fused); // Should be true for properly initialized model
}

#[test]
fn test_gpu_model_has_fused_output_residual() {
    let config = create_test_config();
    let model = GpuModel::new(config).expect("test value should be present");

    // Requires attention_buffers to be present
    assert!(!model.has_fused_output_residual());

    // With attention buffers
    let model_with_buffers = GpuModel::with_attention_buffers(create_test_config(), 128).expect("test value should be present");
    assert!(model_with_buffers.has_fused_output_residual());
}

// ============================================================================
// GpuModel do_matmul Tests with MockExecutor
// ============================================================================

#[test]
fn test_gpu_model_do_matmul_with_mock() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("matmul_test").with_matmul_result(vec![1.0, 2.0, 3.0, 4.0]);
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 2 * 2];
    let b = vec![1.0f32; 2 * 2];
    let result = model.do_matmul(&a, &b, 2, 2, 2);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_gpu_model_do_matmul_failure() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("failing_matmul").with_matmul_failure();
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 4];
    let b = vec![1.0f32; 4];
    let result = model.do_matmul(&a, &b, 2, 2, 2);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_do_matmul_transpose_b_with_mock() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    // For transpose_b: a = m*k, b = n*k (b is transposed so n rows, k cols)
    // With m=1, k=2, n=2: a = 1*2 = 2, b = 2*2 = 4, output = m*n = 2
    let mock = MockExecutor::new("transpose_test").with_matmul_result(vec![10.0, 20.0]);
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 2]; // m=1, k=2
    let b = vec![1.0f32; 4]; // n=2, k=2
    let result = model.do_matmul_transpose_b(&a, &b, 1, 2, 2);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output, vec![10.0, 20.0]);
}

// ============================================================================
// GpuModel matmul_split Tests
// ============================================================================

#[test]
fn test_gpu_model_matmul_split_qkv() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("split_qkv").with_matmul_result(vec![0.0f32; config.qkv_dim()]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Qkv);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.qkv_dim());
}

include!("gpu_model_02.rs");
include!("block_weights.rs");
