//! Additional KV Cache Tests (Part 02) - Coverage Enhancement
//!
//! Targets uncovered paths in gpu/scheduler/kv.rs:
//! - RoPE application edge cases
//! - Error handling paths
//! - Sampling strategies (argmax, top-k)
//! - SwiGLU vs GELU FFN branches
//! - GQA incremental attention paths
//! - Large sequence and edge dimension cases

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::{
    BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig,
};
use crate::gpu::StreamingKVCache;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal config for fast testing
fn minimal_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        eps: 1e-5,
        rope_theta: 10000.0,
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

/// Create a GQA config (num_heads > num_kv_heads)
fn gqa_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // 4 Q heads per KV head
        vocab_size: 100,
        eps: 1e-5,
        rope_theta: 10000.0,
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

/// Create config with SwiGLU FFN (needs gate weights)
fn swiglu_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        eps: 1e-6,
        rope_theta: 10000.0,
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

/// Create model with SwiGLU gate weights
fn create_swiglu_model() -> GpuModel {
    let config = swiglu_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    // Add gate weights to enable SwiGLU path
    if !model.block_weights.is_empty() {
        let gate_weight = vec![0.01f32; config.hidden_dim * config.intermediate_dim];
        model.block_weights[0].ffn_gate_weight = Some(gate_weight);
    }

    model
}

// ============================================================================
// forward_gpu_with_cache Error Path Tests
// ============================================================================

#[test]
fn test_forward_gpu_with_cache_empty_tokens_error() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[], &mut kv_cache);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);
    assert!(err_msg.contains("empty") || err_msg.contains("Empty"));
}

#[test]
fn test_forward_gpu_with_cache_out_of_bounds_token() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Token 9999 exceeds vocab_size=50
    let result = model.forward_gpu_with_cache(&[9999], &mut kv_cache);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);
    assert!(err_msg.contains("out of bounds") || err_msg.contains("Token"));
}

// ============================================================================
// forward_gpu_incremental Error Path Tests
// ============================================================================

#[test]
fn test_forward_gpu_incremental_out_of_bounds_token() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Token 9999 exceeds vocab_size=50
    let result = model.forward_gpu_incremental(9999, &mut kv_cache);
    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_incremental_boundary_token() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("boundary_token");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Token at boundary (vocab_size - 1 = 49)
    let result = model.forward_gpu_incremental(49, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// generate_with_cache Error Path Tests
// ============================================================================

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let result = model.generate_with_cache(&[], &gen_config);

    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("empty") || err_msg.contains("Prompt"));
}

#[test]
fn test_generate_with_cache_single_token_prompt() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("single_prompt");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let result = model.generate_with_cache(&[1], &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("tokens");
    assert!(!tokens.is_empty());
}

// ============================================================================
// Sampling Strategy Tests (argmax and sample_topk paths)
// ============================================================================

#[test]
fn test_generate_with_cache_greedy_sampling() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("greedy");
    model.with_test_executor(Box::new(mock));

    // temperature=0.0 triggers argmax path
    let gen_config = GpuGenerateConfig::deterministic(3);
    assert_eq!(gen_config.temperature, 0.0);
    assert_eq!(gen_config.top_k, 1);

    let result = model.generate_with_cache(&[1, 2], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_topk_sampling() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("topk");
    model.with_test_executor(Box::new(mock));

    // temperature > 0 and top_k > 1 triggers sample_topk path
    let gen_config = GpuGenerateConfig::with_sampling(3, 0.8, 5);
    assert!(gen_config.temperature > 0.0);
    assert!(gen_config.top_k > 1);

    let result = model.generate_with_cache(&[1], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_high_temperature() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("high_temp");
    model.with_test_executor(Box::new(mock));

    // High temperature increases entropy in sampling
    let gen_config = GpuGenerateConfig::with_sampling(2, 1.5, 10);
    let result = model.generate_with_cache(&[1], &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_topk_1_with_temperature() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("topk_1");
    model.with_test_executor(Box::new(mock));

    // top_k=1 should also trigger argmax path
    let gen_config = GpuGenerateConfig {
        max_tokens: 2,
        temperature: 0.7,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    let result = model.generate_with_cache(&[1], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// Stop Token Tests
// ============================================================================

#[test]
fn test_generate_with_cache_stop_token_first_iteration() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    // MockExecutor returns zeros, so argmax returns 0
    let mock = MockExecutor::new("stop_first");
    model.with_test_executor(Box::new(mock));

    // Stop on token 0 (which is what argmax of zeros returns)
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let result = model.generate_with_cache(&[1], &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("tokens");
    // Should stop early due to stop token
    assert!(tokens.len() <= 2); // prompt + at most 1 generated
}

#[test]
fn test_generate_with_cache_multiple_stop_tokens() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("multi_stop");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(5)
        .with_stop_tokens(vec![0, 1, 2, 3]);

    let result = model.generate_with_cache(&[5], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// GQA (Grouped Query Attention) Path Tests
// ============================================================================

#[test]
fn test_forward_gpu_with_cache_gqa() {
    let config = gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("gqa_cache");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2, 3], &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_forward_gpu_incremental_gqa() {
    let config = gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("gqa_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First populate cache
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Then incremental
    let result = model.forward_gpu_incremental(2, &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_gqa_multiple_iterations() {
    let config = gqa_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("gqa_gen");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(5);
    let result = model.generate_with_cache(&[1, 2], &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("tokens");
    assert!(tokens.len() >= 2); // At least the prompt
}

// ============================================================================
// SwiGLU FFN Path Tests
// ============================================================================

#[test]
fn test_forward_gpu_with_cache_swiglu_path() {
    let mut model = create_swiglu_model();
    let config = model.config().clone();

    let mock = MockExecutor::new("swiglu");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2], &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_forward_gpu_incremental_swiglu_path() {
    let mut model = create_swiglu_model();
    let config = model.config().clone();

    let mock = MockExecutor::new("swiglu_inc");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initialize cache
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Incremental with SwiGLU
    let result = model.forward_gpu_incremental(2, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// Multi-token Sequence Tests (RoPE with multiple positions)
// ============================================================================

#[test]
fn test_forward_gpu_with_cache_long_sequence() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("long_seq");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Test with longer sequence to exercise RoPE over multiple positions
    let tokens: Vec<usize> = (0..10).collect();
    let result = model.forward_gpu_with_cache(&tokens, &mut kv_cache);

    assert!(result.is_ok());
}

include!("kv_tests_incremental_multiple.rs");
