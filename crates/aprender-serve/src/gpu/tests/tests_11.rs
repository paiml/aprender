//! Part 11: Additional Batch Generation Tests (PMAT-803)
//!
//! Tests for uncovered paths in `gpu/scheduler/batch.rs`:
//! - Large vocab path in `generate_gpu` (vocab_size > 8192)
//! - CPU fallback paths when GPU buffer limits exceeded
//! - SwiGLU FFN path in `forward_block_single`
//! - Edge cases for GQA attention with various head ratios

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::batch::{
    argmax, forward_block_single, forward_single_token, forward_single_token_greedy, generate_gpu,
    optimized_gqa_attention, optimized_lm_head_argmax_transposed, simplified_attention,
};
use crate::gpu::scheduler::{BlockWeights, GpuModel, GpuModelConfig};

// ============================================================================
// Test Configuration Helpers
// ============================================================================

/// Large vocab config (> 8192) triggers greedy path in generate_gpu
fn create_large_vocab_greedy_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 10000,
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

/// Config that triggers CPU fallback (exceeds 256MB buffer limit)
fn create_cpu_fallback_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 2048,
        intermediate_dim: 4096,
        num_layers: 1,
        num_heads: 16,
        num_kv_heads: 16,
        vocab_size: 50000, // 2048 * 50000 * 4 > 256MB
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

/// Small config for quick tests
fn create_small_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
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

/// GQA config with 4:1 head ratio
fn create_gqa_4_to_1_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 2,
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

/// Create model with SwiGLU gate weight enabled
fn create_model_with_swiglu_gate() -> GpuModel {
    let config = create_small_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let gate_weight = vec![0.01f32; config.hidden_dim * config.intermediate_dim];
    model.block_weights[0].ffn_gate_weight = Some(gate_weight);
    model
}

// ============================================================================
// generate_gpu Tests - Large Vocab Greedy Path
// ============================================================================

#[test]
fn test_generate_gpu_large_vocab_greedy_path() {
    let config = create_large_vocab_greedy_config();
    let mut model = GpuModel::new(config).expect("model creation");
    let prompt = vec![1, 2, 3];
    let result = generate_gpu(&mut model, &prompt, 3);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").len() >= prompt.len());
}

#[test]
fn test_generate_gpu_large_vocab_many_tokens() {
    let config = create_large_vocab_greedy_config();
    let mut model = GpuModel::new(config).expect("model creation");
    let prompt = vec![1, 2];
    let result = generate_gpu(&mut model, &prompt, 5);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").len() > prompt.len());
}

// ============================================================================
// forward_single_token Tests - CPU Fallback Path
// ============================================================================

#[test]
fn test_forward_single_token_cpu_fallback_large_vocab() {
    let config = create_cpu_fallback_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let tokens = vec![1, 2, 3];
    let result = forward_single_token(&mut model, &tokens);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), config.vocab_size);
}

#[test]
fn test_forward_single_token_cpu_fallback_single_token() {
    let config = create_cpu_fallback_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let tokens = vec![100];
    let result = forward_single_token(&mut model, &tokens);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), config.vocab_size);
}

// ============================================================================
// forward_single_token_greedy Tests - CPU Fallback Path
// ============================================================================

#[test]
fn test_forward_single_token_greedy_cpu_fallback() {
    let config = create_cpu_fallback_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let tokens = vec![1, 2, 3];
    let result = forward_single_token_greedy(&mut model, &tokens);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present") < config.vocab_size);
}

#[test]
fn test_forward_single_token_greedy_cpu_fallback_boundary() {
    let config = create_cpu_fallback_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let tokens = vec![config.vocab_size - 1];
    let result = forward_single_token_greedy(&mut model, &tokens);
    assert!(result.is_ok());
}

// ============================================================================
// forward_block_single Tests - SwiGLU FFN Path
// ============================================================================

#[test]
fn test_forward_block_single_swiglu_path() {
    let mut model = create_model_with_swiglu_gate();
    let hidden_dim = model.config.hidden_dim;
    let input = vec![0.1f32; hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), hidden_dim);
}

#[test]
fn test_forward_block_single_swiglu_zeros() {
    let mut model = create_model_with_swiglu_gate();
    let input = vec![0.0f32; model.config.hidden_dim];
    assert!(forward_block_single(&mut model, &input, 0).is_ok());
}

#[test]
fn test_forward_block_single_swiglu_negative_values() {
    let mut model = create_model_with_swiglu_gate();
    let input = vec![-0.5f32; model.config.hidden_dim];
    assert!(forward_block_single(&mut model, &input, 0).is_ok());
}

#[test]
fn test_forward_block_single_swiglu_large_values() {
    let mut model = create_model_with_swiglu_gate();
    let input = vec![10.0f32; model.config.hidden_dim];
    assert!(forward_block_single(&mut model, &input, 0).is_ok());
}

#[test]
fn test_forward_block_single_swiglu_mixed_values() {
    let mut model = create_model_with_swiglu_gate();
    let hidden_dim = model.config.hidden_dim;
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    assert!(forward_block_single(&mut model, &input, 0).is_ok());
}

// ============================================================================
// forward_block_single Tests - GQA Head Expansion
// ============================================================================

#[test]
fn test_forward_block_single_gqa_4_to_1() {
    let config = create_gqa_4_to_1_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let input = vec![0.1f32; config.hidden_dim];
    let result = forward_block_single(&mut model, &input, 0);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), config.hidden_dim);
}

// ============================================================================
// optimized_gqa_attention Tests
// ============================================================================

#[test]
fn test_optimized_gqa_attention_gqa_4_to_1() {
    let config = create_gqa_4_to_1_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    model.with_test_executor(Box::new(MockExecutor::new("gqa_4_to_1")));

    let seq_len = 4;
    let qkv = vec![0.1f32; seq_len * (config.hidden_dim + 2 * config.kv_dim())];
    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), seq_len * config.hidden_dim);
}

#[test]
fn test_optimized_gqa_attention_long_sequence() {
    let config = create_small_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    model.with_test_executor(Box::new(MockExecutor::new("long_seq")));

    let seq_len = 16;
    let qkv = vec![0.1f32; seq_len * (config.hidden_dim + 2 * config.kv_dim())];
    let result = optimized_gqa_attention(&mut model, &qkv, seq_len);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), seq_len * config.hidden_dim);
}

#[test]
fn test_optimized_gqa_attention_varying_values() {
    let config = create_small_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    model.with_test_executor(Box::new(MockExecutor::new("varying")));

    let seq_len = 4;
    let total_size = seq_len * (config.hidden_dim + 2 * config.kv_dim());
    let qkv: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.01).collect();
    assert!(optimized_gqa_attention(&mut model, &qkv, seq_len).is_ok());
}

// ============================================================================
// simplified_attention Tests
// ============================================================================

#[test]
fn test_simplified_attention_longer_sequence() {
    let config = create_small_config();
    let seq_len = 8;
    let qkv = vec![0.1f32; seq_len * 3 * config.hidden_dim];
    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), seq_len * config.hidden_dim);
}

#[test]
fn test_simplified_attention_varying_qkv() {
    let config = create_small_config();
    let seq_len = 4;
    let mut qkv = Vec::with_capacity(seq_len * 3 * config.hidden_dim);
    for section in 0..3 {
        for i in 0..seq_len * config.hidden_dim {
            qkv.push((i as f32) * 0.01 * (section + 1) as f32);
        }
    }
    assert!(simplified_attention(&config, &qkv, seq_len).is_ok());
}

// ============================================================================
// argmax Tests - Chunk Boundary Edge Cases
// ============================================================================

#[test]
fn test_argmax_exactly_chunk_size() {
    let mut logits = vec![0.0f32; 4096];
    logits[2048] = 100.0;
    assert_eq!(argmax(&logits), 2048);
}

#[test]
fn test_argmax_multiple_chunks() {
    let mut logits = vec![0.0f32; 12288];
    logits[12287] = 100.0;
    assert_eq!(argmax(&logits), 12287);
}

#[test]
fn test_argmax_all_negative() {
    let logits = vec![-10.0, -5.0, -1.0, -20.0, -0.5];
    assert_eq!(argmax(&logits), 4);
}

#[test]
fn test_argmax_cross_chunk_max() {
    let mut logits = vec![0.0f32; 8192];
    logits[4095] = 50.0;
    logits[4096] = 100.0;
    assert_eq!(argmax(&logits), 4096);
}

// ============================================================================
// optimized_lm_head_argmax_transposed Tests
// ============================================================================

#[test]
fn test_optimized_lm_head_argmax_large_vocab_cross_chunk() {
    let hidden_dim = 32;
    let vocab_size = 8192;
    let hidden = vec![0.1f32; hidden_dim];
    let mut weight_t = vec![0.0f32; vocab_size * hidden_dim];
    let bias = vec![0.0f32; vocab_size];
    for i in 0..hidden_dim {
        weight_t[4096 * hidden_dim + i] = 1.0;
    }
    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert_eq!(result, 4096);
}

#[test]
fn test_optimized_lm_head_argmax_negative_bias_dominates() {
    let hidden_dim = 16;
    let vocab_size = 100;
    let hidden = vec![1.0f32; hidden_dim];
    let weight_t = vec![0.1f32; vocab_size * hidden_dim];
    let mut bias = vec![-100.0f32; vocab_size];
    bias[42] = 100.0;
    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert_eq!(result, 42);
}

#[test]
fn test_optimized_lm_head_argmax_tie_handling() {
    let hidden_dim = 8;
    let vocab_size = 4;
    let hidden = vec![1.0f32; hidden_dim];
    let weight_t = vec![1.0f32; vocab_size * hidden_dim];
    let bias = vec![0.0f32; vocab_size];
    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert!(result < vocab_size);
}

// ============================================================================
// BlockWeights with SwiGLU Tests
// ============================================================================

#[test]
fn test_block_weights_swiglu_gate_construction() {
    let weights = BlockWeights {
        attn_norm_weight: vec![1.0; 32],
        attn_norm_bias: vec![0.0; 32],
        qkv_weight: vec![0.01; 32 * 96],
        qkv_bias: vec![0.0; 96],
        out_weight: vec![0.01; 32 * 32],
        out_bias: vec![0.0; 32],
        ffn_norm_weight: vec![1.0; 32],
        ffn_norm_bias: vec![0.0; 32],
        ffn_fc1_weight: vec![0.01; 32 * 64],
        ffn_fc1_bias: vec![0.0; 64],
        ffn_fc2_weight: vec![0.01; 64 * 32],
        ffn_fc2_bias: vec![0.0; 32],
        ffn_gate_weight: Some(vec![0.01; 32 * 64]),
        linear_attn: None,
    moe_experts: None,
    };
    assert!(weights.ffn_gate_weight.is_some());
    assert_eq!(weights.ffn_gate_weight.as_ref().expect("test value should be present").len(), 32 * 64);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_generation_flow_large_vocab() {
    let config = create_large_vocab_greedy_config();
    let mut model = GpuModel::new(config).expect("model creation");
    let prompt = vec![1, 2, 3, 4, 5];
    let result = generate_gpu(&mut model, &prompt, 3);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").starts_with(&prompt));
}

#[test]
fn test_full_generation_flow_gqa_config() {
    let config = create_gqa_4_to_1_config();
    let mut model = GpuModel::new(config).expect("model creation");
    let prompt = vec![1, 2];
    assert!(generate_gpu(&mut model, &prompt, 2).is_ok());
}

#[test]
fn test_forward_block_chain_with_swiglu() {
    let config = create_small_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let gate_weight = vec![0.01f32; config.hidden_dim * config.intermediate_dim];
    model.block_weights[0].ffn_gate_weight = Some(gate_weight);

    let mut hidden = vec![0.1f32; config.hidden_dim];
    for block_idx in 0..config.num_layers {
        let result = forward_block_single(&mut model, &hidden, block_idx);
        assert!(result.is_ok());
        hidden = result.expect("test value should be present");
    }
    assert_eq!(hidden.len(), config.hidden_dim);
}
