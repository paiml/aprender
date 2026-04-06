
#[test]
fn test_incremental_multiple_steps() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("multi_step");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initialize
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Multiple incremental steps to test RoPE position advancement
    for i in 2..8 {
        let result = model.forward_gpu_incremental(i, &mut kv_cache);
        assert!(result.is_ok(), "Failed at step {}", i);
    }
}

// ============================================================================
// KV Cache State Tests
// ============================================================================

#[test]
fn test_kv_cache_population_during_forward() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("cache_pop");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initially cache should be empty
    let (k, v) = kv_cache.get_valid(0);
    assert!(k.is_empty());
    assert!(v.is_empty());

    // Forward should populate cache
    let _ = model.forward_gpu_with_cache(&[1, 2, 3], &mut kv_cache);

    let (k, v) = kv_cache.get_valid(0);
    // Cache should now have 3 positions worth of K/V
    let kv_dim = config.num_kv_heads * config.head_dim();
    assert_eq!(k.len(), 3 * kv_dim);
    assert_eq!(v.len(), 3 * kv_dim);
}

#[test]
fn test_kv_cache_growth_during_incremental() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("cache_grow");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let kv_dim = config.num_kv_heads * config.head_dim();

    // Initialize with 1 token
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);
    let (k1, _) = kv_cache.get_valid(0);
    assert_eq!(k1.len(), kv_dim);

    // Incremental should grow cache
    let _ = model.forward_gpu_incremental(2, &mut kv_cache);
    let (k2, _) = kv_cache.get_valid(0);
    assert_eq!(k2.len(), 2 * kv_dim);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_forward_single_token_only() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("single");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[0], &mut kv_cache);
    assert!(result.is_ok());

    let logits = result.expect("logits");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_generate_max_tokens_zero() {
    let config = minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let mock = MockExecutor::new("zero_max");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(0);
    let result = model.generate_with_cache(&[1, 2], &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("tokens");
    // Should return at least prompt
    assert!(!tokens.is_empty());
}

#[test]
fn test_generate_with_all_stop_tokens() {
    let config = minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("all_stop");
    model.with_test_executor(Box::new(mock));

    // All possible tokens as stop tokens
    let stop_tokens: Vec<usize> = (0..config.vocab_size).collect();
    let gen_config = GpuGenerateConfig::deterministic(10)
        .with_stop_tokens(stop_tokens);

    let result = model.generate_with_cache(&[25], &gen_config);
    assert!(result.is_ok());

    let tokens = result.expect("tokens");
    // Should stop immediately after first generation
    assert!(tokens.len() <= 2);
}

// ============================================================================
// Multi-layer Tests
// ============================================================================

#[test]
fn test_forward_multi_layer_model() {
    let config = GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 4, // Multiple layers
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
    };

    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("multi_layer");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2], &mut kv_cache);
    assert!(result.is_ok());

    // Verify all layers have cached values
    for layer in 0..config.num_layers {
        let (k, v) = kv_cache.get_valid(layer);
        assert!(!k.is_empty(), "Layer {} K cache empty", layer);
        assert!(!v.is_empty(), "Layer {} V cache empty", layer);
    }
}
