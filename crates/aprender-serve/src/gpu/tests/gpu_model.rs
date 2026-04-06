
#[test]
fn test_gpu_model_config_edge_case_single_head() {
    let config = GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 1000,
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

    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.kv_dim(), 64);
    assert!(!config.is_gqa());
}

// ============================================================================
// BlockWeights Tests
// ============================================================================

use crate::gpu::scheduler::BlockWeights;

#[test]
fn test_block_weights_construction() {
    let weights = BlockWeights {
        attn_norm_weight: vec![1.0; 64],
        attn_norm_bias: vec![0.0; 64],
        qkv_weight: vec![0.01; 64 * 192],
        qkv_bias: vec![0.0; 192],
        out_weight: vec![0.01; 64 * 64],
        out_bias: vec![0.0; 64],
        ffn_norm_weight: vec![1.0; 64],
        ffn_norm_bias: vec![0.0; 64],
        ffn_fc1_weight: vec![0.01; 64 * 256],
        ffn_fc1_bias: vec![0.0; 256],
        ffn_fc2_weight: vec![0.01; 256 * 64],
        ffn_fc2_bias: vec![0.0; 64],
        ffn_gate_weight: None,
        linear_attn: None,
    moe_experts: None,
    };

    assert_eq!(weights.attn_norm_weight.len(), 64);
    assert!(weights.ffn_gate_weight.is_none());
}

#[test]
fn test_block_weights_with_gate() {
    let weights = BlockWeights {
        attn_norm_weight: vec![1.0; 32],
        attn_norm_bias: vec![0.0; 32],
        qkv_weight: vec![0.01; 32 * 96],
        qkv_bias: vec![0.0; 96],
        out_weight: vec![0.01; 32 * 32],
        out_bias: vec![0.0; 32],
        ffn_norm_weight: vec![1.0; 32],
        ffn_norm_bias: vec![0.0; 32],
        ffn_fc1_weight: vec![0.01; 32 * 128],
        ffn_fc1_bias: vec![0.0; 128],
        ffn_fc2_weight: vec![0.01; 128 * 32],
        ffn_fc2_bias: vec![0.0; 32],
        ffn_gate_weight: Some(vec![0.01; 32 * 128]), // SwiGLU gate
        linear_attn: None,
    moe_experts: None,
    };

    assert!(weights.ffn_gate_weight.is_some());
    assert_eq!(weights.ffn_gate_weight.as_ref().expect("test value should be present").len(), 32 * 128);
}

// ============================================================================
// KV Cache Forward Pass Tests (gpu/scheduler/kv.rs)
// ============================================================================

use crate::gpu::scheduler::{forward_gpu_incremental, forward_gpu_with_cache, generate_with_cache};
use crate::gpu::StreamingKVCache;

fn create_kv_cache_for_model(config: &GpuModelConfig, max_seq_len: usize) -> StreamingKVCache {
    let head_dim = config.hidden_dim / config.num_heads;
    StreamingKVCache::new(
        config.num_layers,
        max_seq_len,
        config.num_kv_heads,
        head_dim,
    )
}

#[test]
fn test_forward_gpu_with_cache_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    let tokens = vec![1, 2, 3];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_gpu_with_cache_single_token() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    let tokens = vec![5];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_ok());
}

#[test]
fn test_forward_gpu_with_cache_empty_tokens() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    let tokens: Vec<usize> = vec![];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_out_of_bounds() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    // Token 9999 exceeds vocab_size (100)
    let tokens = vec![9999];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_with_cache_gqa() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    let tokens = vec![1, 2, 3, 4];
    let result = forward_gpu_with_cache(&mut model, &tokens, &mut kv_cache);

    assert!(result.is_ok());
}

// ============================================================================
// Forward GPU Incremental Tests
// ============================================================================

#[test]
fn test_forward_gpu_incremental_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    // First, populate cache with initial tokens
    let _ = forward_gpu_with_cache(&mut model, &[1, 2], &mut kv_cache);

    // Then do incremental forward for new token
    let result = forward_gpu_incremental(&mut model, 3, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_gpu_incremental_multiple() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    // Populate cache
    let _ = forward_gpu_with_cache(&mut model, &[1], &mut kv_cache);

    // Multiple incremental forwards
    for token in [2, 3, 4, 5] {
        let result = forward_gpu_incremental(&mut model, token, &mut kv_cache);
        assert!(result.is_ok(), "token {} forward should work", token);
    }
}

#[test]
fn test_forward_gpu_incremental_out_of_bounds() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 32);

    // Populate cache first
    let _ = forward_gpu_with_cache(&mut model, &[1], &mut kv_cache);

    // Token out of bounds
    let result = forward_gpu_incremental(&mut model, 9999, &mut kv_cache);
    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_incremental_gqa() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");
    let mut kv_cache = create_kv_cache_for_model(&config, 64);

    // Populate cache
    let _ = forward_gpu_with_cache(&mut model, &[1, 2, 3], &mut kv_cache);

    // Incremental forward
    let result = forward_gpu_incremental(&mut model, 4, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// Generate with Cache Tests
// ============================================================================

#[test]
fn test_generate_with_cache_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt = vec![1, 2];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert!(tokens.len() >= prompt.len()); // At least prompt length
}

#[test]
fn test_generate_with_cache_empty_prompt() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt: Vec<usize> = vec![];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_err());
}

#[test]
fn test_generate_with_cache_with_stop_tokens() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Stop on token 0 (which might be generated from uniform random weights)
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let prompt = vec![1];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_sampling() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Use sampling with temperature
    let gen_config = GpuGenerateConfig::with_sampling(3, 0.5, 10);
    let prompt = vec![1, 2, 3];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_zero_max_tokens() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(0);
    let prompt = vec![1, 2];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
    // Should at least return prompt + 1 token (or just prompt if first token is stop)
}

#[test]
fn test_generate_with_cache_gqa() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_single_token_prompt() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![5];

    let result = generate_with_cache(&mut model, &prompt, &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// StreamingKVCache Tests
// ============================================================================

#[test]
fn test_streaming_kv_cache_new() {
    let num_layers = 4;
    let max_seq_len = 128;
    let num_kv_heads = 4;
    let head_dim = 16;

    let cache = StreamingKVCache::new(num_layers, max_seq_len, num_kv_heads, head_dim);

    // Cache should be created and initially empty
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_positions(), max_seq_len);
}

#[test]
fn test_streaming_kv_cache_append_and_get() {
    let cache_config = create_test_config();
    let head_dim = cache_config.hidden_dim / cache_config.num_heads;
    let kv_dim = cache_config.num_kv_heads * head_dim;

    let mut cache = StreamingKVCache::new(
        cache_config.num_layers,
        64,
        cache_config.num_kv_heads,
        head_dim,
    );

    // Append K/V data to ALL layers (position only increments after last layer)
    let k = vec![0.1f32; kv_dim];
    let v = vec![0.2f32; kv_dim];

    for layer in 0..cache_config.num_layers {
        cache.append(layer, &k, &v);
    }

    // Now cache has 1 valid position
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);

    // Get cached data for layer 0
    let (cached_k, cached_v) = cache.get_valid(0);
    assert_eq!(cached_k.len(), kv_dim);
    assert_eq!(cached_v.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_multiple_layers() {
    let config = create_test_config();
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;

    let mut cache = StreamingKVCache::new(config.num_layers, 32, config.num_kv_heads, head_dim);

    let k = vec![0.1f32; kv_dim];
    let v = vec![0.2f32; kv_dim];

    // Append to all layers
    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }

    // Verify each layer has data
    for layer in 0..config.num_layers {
        let (cached_k, _) = cache.get_valid(layer);
        assert_eq!(cached_k.len(), kv_dim);
    }
}

#[test]
fn test_streaming_kv_cache_clear() {
    let config = create_test_config();
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;

    let mut cache = StreamingKVCache::new(config.num_layers, 32, config.num_kv_heads, head_dim);

    // Populate cache - must append to ALL layers for position to increment
    let k = vec![0.1f32; kv_dim];
    let v = vec![0.2f32; kv_dim];
    for layer in 0..config.num_layers {
        cache.append(layer, &k, &v);
    }
    assert!(!cache.is_empty());

    // Clear
    cache.clear();

    // Should be empty after clear
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}
