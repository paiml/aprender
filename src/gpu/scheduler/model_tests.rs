//! Phase 50 - GpuModel Coverage Tests Part 02 (gpu/scheduler/model.rs)
//!
//! Additional tests for improving coverage on model.rs (~24% uncovered):
//! - from_apr_weights constructor
//! - generate_gpu wrapper
//! - Edge cases for GQA attention paths
//! - SwiGLU FFN activation paths
//! - Error handling edge cases
//! - GpuModelConfig Debug/Clone traits
//! - Large vocab CPU fallback paths

use crate::gpu::executor::MockExecutor;
use crate::gpu::scheduler::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig, WeightType,
};
use crate::gpu::StreamingKVCache;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a minimal test model configuration
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

/// Create config with large vocab for CPU fallback path testing
fn create_large_vocab_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 32000, // Large vocab triggers CPU fallback
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

/// Create BlockWeights with SwiGLU gate (for SwiGLU FFN path)
fn create_block_weights_with_swiglu(config: &GpuModelConfig) -> BlockWeights {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let qkv_dim = config.qkv_dim();

    BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * qkv_dim],
        qkv_bias: vec![0.0; qkv_dim],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: Some(vec![0.01; hidden_dim * intermediate_dim]), // Enable SwiGLU
        linear_attn: None,
    moe_experts: None,
    }
}

// ============================================================================
// from_apr_weights Tests
// ============================================================================

#[test]
fn test_gpu_model_from_apr_weights_basic() {
    let config = create_minimal_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let intermediate_dim = config.intermediate_dim;
    let qkv_dim = config.qkv_dim();

    // Create minimal weight vectors
    let embedding_weights = vec![0.01f32; vocab_size * hidden_dim];
    let block_weights = vec![BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * qkv_dim],
        qkv_bias: vec![0.0; qkv_dim],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: None,
        linear_attn: None,
    moe_experts: None,
    }];
    let final_norm_weight = vec![1.0; hidden_dim];
    let final_norm_bias = vec![0.0; hidden_dim];
    let lm_head_weight = vec![0.01; hidden_dim * vocab_size];
    let lm_head_weight_t = vec![0.01; vocab_size * hidden_dim];
    let lm_head_bias = vec![0.0; vocab_size];

    let result = GpuModel::from_apr_weights(
        config.clone(),
        embedding_weights,
        block_weights,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
        lm_head_weight_t,
        lm_head_bias,
    );

    assert!(result.is_ok(), "from_apr_weights should succeed");
    let model = result.unwrap();
    assert_eq!(model.config().hidden_dim, hidden_dim);
    assert_eq!(model.config().vocab_size, vocab_size);
}

#[test]
fn test_gpu_model_from_apr_weights_with_swiglu() {
    let config = create_minimal_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let intermediate_dim = config.intermediate_dim;

    let embedding_weights = vec![0.01f32; vocab_size * hidden_dim];
    let block_weights = vec![create_block_weights_with_swiglu(&config)];
    let final_norm_weight = vec![1.0; hidden_dim];
    let final_norm_bias = vec![0.0; hidden_dim];
    let lm_head_weight = vec![0.01; hidden_dim * vocab_size];
    let lm_head_weight_t = vec![0.01; vocab_size * hidden_dim];
    let lm_head_bias = vec![0.0; vocab_size];

    let result = GpuModel::from_apr_weights(
        config,
        embedding_weights,
        block_weights,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
        lm_head_weight_t,
        lm_head_bias,
    );

    assert!(result.is_ok());
    let model = result.unwrap();
    // Verify the model has gate weights
    assert!(model.block_weights[0].ffn_gate_weight.is_some());
    assert_eq!(
        model.block_weights[0].ffn_gate_weight.as_ref().unwrap().len(),
        hidden_dim * intermediate_dim
    );
}

#[test]
fn test_gpu_model_from_apr_weights_gqa() {
    let config = create_gqa_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let intermediate_dim = config.intermediate_dim;
    let qkv_dim = config.qkv_dim();

    let embedding_weights = vec![0.01f32; vocab_size * hidden_dim];
    let mut block_weights = Vec::new();
    for _ in 0..config.num_layers {
        block_weights.push(BlockWeights {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: vec![0.0; hidden_dim],
            qkv_weight: vec![0.01; hidden_dim * qkv_dim],
            qkv_bias: vec![0.0; qkv_dim],
            out_weight: vec![0.01; hidden_dim * hidden_dim],
            out_bias: vec![0.0; hidden_dim],
            ffn_norm_weight: vec![1.0; hidden_dim],
            ffn_norm_bias: vec![0.0; hidden_dim],
            ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_fc1_bias: vec![0.0; intermediate_dim],
            ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_fc2_bias: vec![0.0; hidden_dim],
            ffn_gate_weight: None,
            linear_attn: None,
        moe_experts: None,
        });
    }
    let final_norm_weight = vec![1.0; hidden_dim];
    let final_norm_bias = vec![0.0; hidden_dim];
    let lm_head_weight = vec![0.01; hidden_dim * vocab_size];
    let lm_head_weight_t = vec![0.01; vocab_size * hidden_dim];
    let lm_head_bias = vec![0.0; vocab_size];

    let result = GpuModel::from_apr_weights(
        config.clone(),
        embedding_weights,
        block_weights,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
        lm_head_weight_t,
        lm_head_bias,
    );

    assert!(result.is_ok());
    let model = result.unwrap();
    assert!(model.config().is_gqa());
    assert_eq!(model.config().num_kv_heads, 2);
}

// ============================================================================
// generate_gpu Tests
// ============================================================================

#[test]
fn test_gpu_model_generate_gpu_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_gpu_test");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![1, 2];
    let max_tokens = 3;
    let result = model.generate_gpu(&prompt, max_tokens);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    // Should include prompt + generated tokens
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_gpu_model_generate_gpu_single_token() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_gpu_single");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![5];
    let max_tokens = 2;
    let result = model.generate_gpu(&prompt, max_tokens);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_generate_gpu_max_tokens_zero() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).unwrap();

    let mock = MockExecutor::new("generate_gpu_zero");
    model.with_test_executor(Box::new(mock));

    let prompt = vec![1];
    let max_tokens = 0;
    let result = model.generate_gpu(&prompt, max_tokens);

    assert!(result.is_ok());
    // Should return at least the prompt
    let tokens = result.unwrap();
    assert!(!tokens.is_empty());
}

// ============================================================================
// SwiGLU FFN Path Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_with_swiglu() {
    let config = create_minimal_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;
    let intermediate_dim = config.intermediate_dim;

    let embedding_weights = vec![0.01f32; vocab_size * hidden_dim];
    let block_weights = vec![create_block_weights_with_swiglu(&config)];
    let final_norm_weight = vec![1.0; hidden_dim];
    let final_norm_bias = vec![0.0; hidden_dim];
    let lm_head_weight = vec![0.01; hidden_dim * vocab_size];
    let lm_head_weight_t = vec![0.01; vocab_size * hidden_dim];
    let lm_head_bias = vec![0.0; vocab_size];

    let mut model = GpuModel::from_apr_weights(
        config.clone(),
        embedding_weights,
        block_weights,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
        lm_head_weight_t,
        lm_head_bias,
    )
    .unwrap();

    let mock = MockExecutor::new("swiglu_forward");
    model.with_test_executor(Box::new(mock));

    let result = model.forward_gpu(&[1, 2]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 2 * vocab_size);
}

#[test]
fn test_gpu_model_incremental_with_swiglu() {
    let config = create_minimal_config();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    let embedding_weights = vec![0.01f32; vocab_size * hidden_dim];
    let block_weights = vec![create_block_weights_with_swiglu(&config)];
    let final_norm_weight = vec![1.0; hidden_dim];
    let final_norm_bias = vec![0.0; hidden_dim];
    let lm_head_weight = vec![0.01; hidden_dim * vocab_size];
    let lm_head_weight_t = vec![0.01; vocab_size * hidden_dim];
    let lm_head_bias = vec![0.0; vocab_size];

    let mut model = GpuModel::from_apr_weights(
        config.clone(),
        embedding_weights,
        block_weights,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
        lm_head_weight_t,
        lm_head_bias,
    )
    .unwrap();

    let mock = MockExecutor::new("swiglu_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_incremental_optimized(1, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModelConfig Debug/Clone Tests
// ============================================================================

#[test]
fn test_gpu_model_config_debug() {
    let config = create_minimal_config();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("GpuModelConfig"));
    assert!(debug_str.contains("hidden_dim"));
    assert!(debug_str.contains("vocab_size"));
    assert!(debug_str.contains("num_heads"));
}

#[test]
fn test_gpu_model_config_clone() {
    let config = create_minimal_config();
    let cloned = config.clone();

    assert_eq!(config.hidden_dim, cloned.hidden_dim);
    assert_eq!(config.vocab_size, cloned.vocab_size);
    assert_eq!(config.num_layers, cloned.num_layers);
    assert_eq!(config.eps, cloned.eps);
    assert_eq!(config.rope_theta, cloned.rope_theta);
}

#[test]
fn test_gpu_generate_config_debug() {
    let config = GpuGenerateConfig::with_sampling(64, 0.8, 40);
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("GpuGenerateConfig"));
    assert!(debug_str.contains("max_tokens"));
    assert!(debug_str.contains("temperature"));
}

#[test]
fn test_gpu_generate_config_clone() {
    let config = GpuGenerateConfig::with_sampling(64, 0.8, 40).with_stop_tokens(vec![1, 2]);
    let cloned = config.clone();

    assert_eq!(config.max_tokens, cloned.max_tokens);
    assert_eq!(config.temperature, cloned.temperature);
    assert_eq!(config.top_k, cloned.top_k);
    assert_eq!(config.stop_tokens, cloned.stop_tokens);
}

// ============================================================================
// AttentionBuffers Debug Tests
// ============================================================================

#[test]
fn test_attention_buffers_debug() {
    let config = create_minimal_config();
    let buffers = AttentionBuffers::new(&config, 64);
    let debug_str = format!("{:?}", buffers);

    assert!(debug_str.contains("AttentionBuffers"));
    assert!(debug_str.contains("max_seq_len"));
}

// ============================================================================
// Large Vocab CPU Fallback Tests
// ============================================================================

#[test]
fn test_gpu_model_large_vocab_forward() {
    let config = create_large_vocab_config();
    let mut model = GpuModel::new(config.clone()).unwrap();

    let mock = MockExecutor::new("large_vocab_forward");
    model.with_test_executor(Box::new(mock));

    // This should hit the CPU fallback path for LM head
    let result = model.forward_gpu(&[1]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

include!("model_tests_gpu.rs");
