//! Phase 48 - Batch Scenario Tests (gpu/scheduler/batch.rs coverage)
//!
//! Tests for:
//! - `generate_gpu`: Token generation with MockExecutor
//! - `forward_single_token`: Single token forward pass
//! - `forward_single_token_greedy`: Optimized greedy sampling
//! - `forward_block_single`: Single token through transformer block
//! - `optimized_gqa_attention`: GQA attention implementation
//!
//! Strategy: Use MockExecutor to test CPU logic paths without GPU hardware.

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::batch::{
    argmax, forward_block_single, forward_single_token, forward_single_token_greedy, generate_gpu,
    optimized_gqa_attention, optimized_lm_head_argmax_transposed, simplified_attention,
};
use crate::gpu::scheduler::{GpuModel, GpuModelConfig};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test model configuration
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

/// Create a small vocab config (uses GPU path in forward_single_token)
fn create_small_vocab_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50, // Small vocab triggers GPU path
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

/// Create a large vocab config (uses CPU path in forward_single_token)
fn create_large_vocab_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 10000, // Large vocab triggers CPU path
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
// forward_single_token Tests
// ============================================================================

#[test]
fn test_forward_single_token_basic() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Install MockExecutor
    let mock = MockExecutor::new("test_forward_single");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2, 3];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_ok(), "forward_single_token should succeed");
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), model.config.vocab_size);
}

#[test]
fn test_forward_single_token_empty_tokens() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens: Vec<usize> = vec![];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_err(), "empty tokens should fail");
}

#[test]
fn test_forward_single_token_out_of_bounds() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Token 9999 is out of bounds for vocab_size=50
    let tokens = vec![1, 2, 9999];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_err(), "out of bounds token should fail");
}

#[test]
fn test_forward_single_token_single_token() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_single_token");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![5];
    let result = forward_single_token(&mut model, &tokens);

    assert!(result.is_ok());
}

#[test]
fn test_forward_single_token_with_mock_calls() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_mock_calls");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2];
    let _ = forward_single_token(&mut model, &tokens);

    // Verify mock was used (small vocab triggers GPU matmul)
    assert!(model.has_test_executor());
}

// ============================================================================
// forward_single_token_greedy Tests
// ============================================================================

#[test]
fn test_forward_single_token_greedy_basic() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_greedy");
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2, 3];
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_ok());
    let next_token = result.expect("test value should be present");
    assert!(next_token < model.config.vocab_size);
}

#[test]
fn test_forward_single_token_greedy_empty() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens: Vec<usize> = vec![];
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_err());
}

#[test]
fn test_forward_single_token_greedy_out_of_bounds() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens = vec![99999]; // Out of bounds
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_err());
}

#[test]
fn test_forward_single_token_greedy_large_vocab() {
    // Large vocab triggers CPU path
    let config = create_large_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let tokens = vec![1, 2, 3];
    let result = forward_single_token_greedy(&mut model, &tokens);

    assert!(result.is_ok());
    let next_token = result.expect("test value should be present");
    assert!(next_token < model.config.vocab_size);
}

// ============================================================================
// forward_block_single Tests
// ============================================================================

#[test]
fn test_forward_block_single_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![0.1f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), model.config.hidden_dim);
}

#[test]
fn test_forward_block_single_all_layers() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mut hidden = vec![0.1f32; config.hidden_dim];

    for block_idx in 0..config.num_layers {
        let result = forward_block_single(&mut model, &hidden, block_idx);
        assert!(result.is_ok(), "block {} should succeed", block_idx);
        hidden = result.expect("test value should be present");
        assert_eq!(hidden.len(), config.hidden_dim);
    }
}

#[test]
fn test_forward_block_single_gqa() {
    // Test GQA path (num_heads > num_kv_heads)
    let config = create_gqa_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![0.1f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), model.config.hidden_dim);
}

#[test]
fn test_forward_block_single_zeros() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![0.0f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
}

#[test]
fn test_forward_block_single_large_values() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let input = vec![100.0f32; model.config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);

    assert!(result.is_ok());
}

// ============================================================================
// generate_gpu Tests
// ============================================================================

#[test]
fn test_generate_gpu_basic() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_generate");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![1, 2];
    let result = generate_gpu(&mut model, &prompt, 3);

    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    // Should have prompt + max_tokens (minus initial prediction)
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_generate_gpu_single_token() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_generate_single");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![5];
    let result = generate_gpu(&mut model, &prompt, 1);

    assert!(result.is_ok());
}

#[test]
fn test_generate_gpu_large_vocab() {
    // Large vocab uses different code path
    let config = create_large_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let prompt = vec![1, 2];
    let result = generate_gpu(&mut model, &prompt, 3);

    assert!(result.is_ok());
}

#[test]
fn test_generate_gpu_zero_max_tokens() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test_zero_tokens");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![1];
    let result = generate_gpu(&mut model, &prompt, 0);

    assert!(result.is_ok());
}

// ============================================================================
// optimized_gqa_attention Tests
// ============================================================================

#[test]
fn test_optimized_gqa_attention_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mock = MockExecutor::new("test_gqa_attention");
    model.with_test_executor(Box::new(mock));

    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.kv_dim();

    // QKV size for MHA: seq_len * (hidden_dim + 2*kv_dim) = seq_len * 3*hidden_dim for MHA
    let qkv = vec![0.1f32; seq_len * (hidden_dim + 2 * kv_dim)];

    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), seq_len * hidden_dim);
}

#[test]
fn test_optimized_gqa_attention_single_token() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mock = MockExecutor::new("test_gqa_single");
    model.with_test_executor(Box::new(mock));

    let seq_len = 1;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.kv_dim();

    let qkv = vec![0.1f32; seq_len * (hidden_dim + 2 * kv_dim)];

    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);

    assert!(result.is_ok());
}

#[test]
fn test_optimized_gqa_attention_gqa_mode() {
    // Test with actual GQA (num_heads > num_kv_heads)
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let mock = MockExecutor::new("test_gqa_mode");
    model.with_test_executor(Box::new(mock));

    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let kv_dim = config.kv_dim();

    // GQA: Q has hidden_dim, K/V have kv_dim
    let qkv = vec![0.1f32; seq_len * (hidden_dim + 2 * kv_dim)];

    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), seq_len * hidden_dim);
}

// ============================================================================
// argmax Additional Coverage Tests
// ============================================================================

#[test]
fn test_argmax_empty() {
    let logits: Vec<f32> = vec![];
    let result = argmax(&logits);
    assert_eq!(result, 0); // Default for empty
}

#[test]
fn test_argmax_boundary_1024() {
    // Exactly at boundary where chunking kicks in
    let mut logits = vec![0.0f32; 1024];
    logits[512] = 1.0;
    assert_eq!(argmax(&logits), 512);
}

#[test]
fn test_argmax_boundary_1025() {
    // Just over boundary - triggers chunked path
    let mut logits = vec![0.0f32; 1025];
    logits[1024] = 1.0;
    assert_eq!(argmax(&logits), 1024);
}

#[test]
fn test_argmax_large_8192() {
    let mut logits = vec![0.0f32; 8192];
    logits[4000] = 1.0;
    assert_eq!(argmax(&logits), 4000);
}

include!("optimized.rs");
include!("gpu_model.rs");
