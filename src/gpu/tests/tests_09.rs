//! Phase 50 - KV Cache Scheduler Coverage Tests (gpu/scheduler/kv.rs)
//!
//! Comprehensive tests for KV cache operations:
//! - forward_gpu_with_cache: Forward pass with KV cache population
//! - forward_gpu_incremental: Incremental forward using cached KV
//! - generate_with_cache: Full generation loop with KV caching
//! - apply_rope: Rotary position embedding (internal)
//! - gqa_attention_with_kv / gqa_incremental_attention: GQA attention patterns
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
fn create_kv_test_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4, // MHA (not GQA)
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

/// Create a GQA test configuration (num_heads > num_kv_heads)
fn create_kv_gqa_config() -> GpuModelConfig {
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

/// Create a single-layer config for fast iteration tests
fn create_kv_single_layer_config() -> GpuModelConfig {
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

/// Create a deeper model for stress testing
fn create_kv_deep_config() -> GpuModelConfig {
    GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 4,
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

// ============================================================================
// StreamingKVCache Construction Tests
// ============================================================================

#[test]
fn test_kv_cache_new_basic() {
    let config = create_kv_test_config();
    let cache = StreamingKVCache::new(
        config.num_layers,
        256,
        config.num_kv_heads,
        config.head_dim(),
    );

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 256);
}

#[test]
fn test_kv_cache_new_gqa() {
    let config = create_kv_gqa_config();
    let cache = StreamingKVCache::new(
        config.num_layers,
        512,
        config.num_kv_heads,
        config.head_dim(),
    );

    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 512);
}

#[test]
fn test_kv_cache_memory_calculation() {
    let config = create_kv_test_config();
    let max_positions = 128;
    let cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Memory = num_layers * max_positions * num_kv_heads * head_dim * 2 (K+V) * 4 bytes
    let expected = config.num_layers * max_positions * config.num_kv_heads * config.head_dim() * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected);
}

#[test]
fn test_kv_cache_memory_mb_calculation() {
    let config = create_kv_test_config();
    let cache = StreamingKVCache::new(
        config.num_layers,
        1024,
        config.num_kv_heads,
        config.head_dim(),
    );

    let bytes = cache.memory_bytes();
    let expected_mb = bytes as f64 / (1024.0 * 1024.0);
    assert!((cache.memory_mb() - expected_mb).abs() < 0.001);
}

// ============================================================================
// StreamingKVCache Append and Get Tests
// ============================================================================

#[test]
fn test_kv_cache_append_single_position() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();
    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    // Append to all layers for one position
    for layer in 0..config.num_layers {
        cache.append(layer, &key, &value);
    }

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_append_multiple_positions() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Append 5 positions
    for pos in 0..5 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![(pos as f32) * 2.0; kv_dim];

        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_kv_cache_get_valid_single_layer() {
    let config = create_kv_single_layer_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        32,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();
    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    cache.append(0, &key, &value);

    let (keys, values) = cache.get_valid(0);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);

    // Verify values
    assert!(keys.iter().all(|&k| (k - 1.0).abs() < 1e-6));
    assert!(values.iter().all(|&v| (v - 2.0).abs() < 1e-6));
}

#[test]
fn test_kv_cache_get_range() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Append 10 positions
    for pos in 0..10 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![(pos as f32) * 10.0; kv_dim];

        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Get range 2..5
    let (keys, values) = cache.get_range(0, 2, 5);
    assert_eq!(keys.len(), 3 * kv_dim);
    assert_eq!(values.len(), 3 * kv_dim);

    // First element should be from position 2
    assert!((keys[0] - 2.0).abs() < 1e-6);
    // Last element should be from position 4
    assert!((keys[(2 * kv_dim)] - 4.0).abs() < 1e-6);
}

#[test]
fn test_kv_cache_clear() {
    let config = create_kv_test_config();
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();
    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    // Add some positions
    for _ in 0..5 {
        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }
    assert_eq!(cache.len(), 5);

    // Clear
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_circular_buffer_wraparound() {
    let config = create_kv_single_layer_config();
    let max_positions = 4;
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Append more positions than max_positions (should wrap around)
    for pos in 0..10 {
        let key = vec![pos as f32; kv_dim];
        let value = vec![(pos as f32) * 10.0; kv_dim];

        for layer in 0..config.num_layers {
            cache.append(layer, &key, &value);
        }
    }

    // Cache should be capped at max_positions
    assert_eq!(cache.len(), max_positions);
}

// ============================================================================
// forward_gpu_with_cache Tests
// ============================================================================

#[test]
fn test_forward_gpu_with_cache_single_token() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_with_cache_single");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids = vec![1];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);

    // Cache should have 1 position after forward
    assert_eq!(kv_cache.len(), 1);
}

#[test]
fn test_forward_gpu_with_cache_multiple_tokens() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_with_cache_multi");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids = vec![1, 2, 3, 4, 5];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    // Output is only for final position
    assert_eq!(logits.len(), config.vocab_size);

    // Cache should have all positions
    assert_eq!(kv_cache.len(), token_ids.len());
}

#[test]
fn test_forward_gpu_with_cache_empty_tokens_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids: Vec<usize> = vec![];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_out_of_bounds_token_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // vocab_size is 50, so token 999 is out of bounds
    let token_ids = vec![1, 999];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_gqa_model() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_gqa_cache");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let token_ids = vec![1, 2, 3];
    let result = model.forward_gpu_with_cache(&token_ids, &mut kv_cache);

    assert!(result.is_ok());
    assert_eq!(kv_cache.len(), 3);
}

include!("forward_gpu.rs");
include!("gqa_cache.rs");
