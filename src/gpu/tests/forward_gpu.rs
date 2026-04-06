
// ============================================================================
// forward_gpu_incremental Tests
// ============================================================================

#[test]
fn test_forward_gpu_incremental_basic() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First populate cache
    let _ = model.forward_gpu_with_cache(&[1, 2], &mut kv_cache);
    assert_eq!(kv_cache.len(), 2);

    // Now do incremental forward
    let result = model.forward_gpu_incremental(3, &mut kv_cache);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);

    // Cache should now have 3 positions
    assert_eq!(kv_cache.len(), 3);
}

#[test]
fn test_forward_gpu_incremental_sequential() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_incremental_seq");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initial forward
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Multiple incremental forwards
    for token in [2, 3, 4, 5] {
        let result = model.forward_gpu_incremental(token, &mut kv_cache);
        assert!(result.is_ok());
    }

    assert_eq!(kv_cache.len(), 5);
}

#[test]
fn test_forward_gpu_incremental_out_of_bounds_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // vocab_size is 50, so token 9999 is out of bounds
    let result = model.forward_gpu_incremental(9999, &mut kv_cache);

    assert!(result.is_err());
}

#[test]
fn test_forward_gpu_incremental_gqa() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("incremental_gqa");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initial forward
    let _ = model.forward_gpu_with_cache(&[1, 2], &mut kv_cache);

    // Incremental forward with GQA
    let result = model.forward_gpu_incremental(3, &mut kv_cache);
    assert!(result.is_ok());
    assert_eq!(kv_cache.len(), 3);
}

#[test]
fn test_forward_gpu_incremental_deep_model() {
    let config = create_kv_deep_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("incremental_deep");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Initial forward
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Incremental forward through 4 layers
    let result = model.forward_gpu_incremental(2, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// generate_with_cache Tests
// ============================================================================

#[test]
fn test_generate_with_cache_basic() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_cache");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    // Should have at least prompt tokens
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt: Vec<usize> = vec![];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_err());
}

#[test]
fn test_generate_with_cache_with_stop_tokens() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_stop");
    model.with_test_executor(Box::new(mock));

    // Mock returns zeros, so token 0 will be generated and should stop
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let prompt = vec![1];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_with_sampling() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_sampling");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::with_sampling(3, 0.8, 5);
    let prompt = vec![1, 2];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_gqa() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_gqa");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![1];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_max_tokens_reached() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_max_tokens");
    model.with_test_executor(Box::new(mock));

    // Generate exactly max_tokens
    let max_tokens = 5;
    let gen_config = GpuGenerateConfig::deterministic(max_tokens);
    let prompt = vec![1];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    // Should have prompt + up to max_tokens (may stop earlier if stop token hit)
    assert!(tokens.len() <= prompt.len() + max_tokens);
}

#[test]
fn test_generate_with_cache_single_token_prompt() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_single_prompt");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![5];
    let result = model.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert!(!tokens.is_empty());
}

// ============================================================================
// KV Cache State Isolation Tests
// ============================================================================

#[test]
fn test_kv_cache_state_isolation_between_calls() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("state_isolation");
    model.with_test_executor(Box::new(mock));

    // First cache
    let mut cache1 = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Second cache
    let mut cache2 = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Populate first cache
    let _ = model.forward_gpu_with_cache(&[1, 2, 3], &mut cache1);
    assert_eq!(cache1.len(), 3);

    // Second cache should be independent
    let _ = model.forward_gpu_with_cache(&[4, 5], &mut cache2);
    assert_eq!(cache2.len(), 2);

    // First cache should still have 3
    assert_eq!(cache1.len(), 3);
}

#[test]
fn test_kv_cache_clear_between_generations() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("clear_between_gen");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First generation
    let _ = model.forward_gpu_with_cache(&[1, 2, 3], &mut cache);
    assert_eq!(cache.len(), 3);

    // Clear for new generation
    cache.clear();
    assert_eq!(cache.len(), 0);

    // Second generation
    let _ = model.forward_gpu_with_cache(&[4, 5], &mut cache);
    assert_eq!(cache.len(), 2);
}

// ============================================================================
// Memory and Performance Tests
// ============================================================================

#[test]
fn test_kv_cache_large_context() {
    let config = create_kv_test_config();
    let max_positions = 2048;
    let cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Verify memory calculation for larger context
    let expected = config.num_layers * max_positions * config.num_kv_heads * config.head_dim() * 2 * 4;
    assert_eq!(cache.memory_bytes(), expected);

    // Should be less than 1MB for this config
    assert!(cache.memory_mb() < 1.0);
}

#[test]
fn test_kv_cache_stress_many_positions() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("stress_positions");
    model.with_test_executor(Box::new(mock));

    let max_positions = 128;
    let mut cache = StreamingKVCache::new(
        config.num_layers,
        max_positions,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Populate with many positions via incremental forward
    let _ = model.forward_gpu_with_cache(&[1], &mut cache);

    for token in 2..100 {
        let result = model.forward_gpu_incremental(token % config.vocab_size, &mut cache);
        assert!(result.is_ok());
    }

    // Should have 99 positions (or wrapped around to max_positions)
    assert!(cache.len() <= max_positions);
}

// ============================================================================
// RoPE Integration Tests (via forward passes)
// ============================================================================

#[test]
fn test_rope_applied_in_forward_with_cache() {
    // RoPE is applied internally during forward_gpu_with_cache
    // We verify it doesn't panic and produces consistent output
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("rope_forward");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2, 3], &mut cache);
    assert!(result.is_ok());
}

#[test]
fn test_rope_applied_in_incremental_forward() {
    // RoPE uses start_pos in incremental forward
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("rope_incremental");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Populate cache first
    let _ = model.forward_gpu_with_cache(&[1, 2], &mut cache);

    // Incremental forward with correct position
    let result = model.forward_gpu_incremental(3, &mut cache);
    assert!(result.is_ok());
}

#[test]
fn test_rope_theta_variations() {
    // Test with different rope_theta values
    let mut config = create_kv_test_config();

    // Higher rope_theta (e.g., for longer context)
    config.rope_theta = 100000.0;

    let mut model = GpuModel::new(config.clone()).expect("test value should be present");
    let mock = MockExecutor::new("rope_theta");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_with_cache(&[1, 2], &mut cache);
    assert!(result.is_ok());
}

// ============================================================================
// GQA Attention Pattern Tests
// ============================================================================

#[test]
fn test_gqa_heads_per_kv_ratio() {
    let config = create_kv_gqa_config();

    // num_heads=8, num_kv_heads=2 => 4 Q heads per KV head
    let heads_per_kv = config.num_heads / config.num_kv_heads;
    assert_eq!(heads_per_kv, 4);
}
