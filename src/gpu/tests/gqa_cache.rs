
#[test]
fn test_gqa_kv_dim_calculation() {
    let config = create_kv_gqa_config();

    // head_dim = hidden_dim / num_heads = 64 / 8 = 8
    assert_eq!(config.head_dim(), 8);

    // kv_dim = num_kv_heads * head_dim = 2 * 8 = 16
    assert_eq!(config.kv_dim(), 16);
}

#[test]
fn test_gqa_cache_size_smaller() {
    // GQA should use less cache memory
    let mha_config = create_kv_test_config();
    let gqa_config = create_kv_gqa_config();

    let max_positions = 512;

    let mha_cache = StreamingKVCache::new(
        mha_config.num_layers,
        max_positions,
        mha_config.num_kv_heads,
        mha_config.head_dim(),
    );

    let gqa_cache = StreamingKVCache::new(
        gqa_config.num_layers,
        max_positions,
        gqa_config.num_kv_heads,
        gqa_config.head_dim(),
    );

    // GQA cache should be smaller (fewer kv_heads)
    assert!(gqa_cache.memory_bytes() < mha_cache.memory_bytes());
}

#[test]
fn test_gqa_forward_with_cache_full_sequence() {
    let config = create_kv_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("gqa_full_seq");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Longer sequence to exercise GQA
    let token_ids: Vec<usize> = (0..20).map(|i| i % config.vocab_size).collect();
    let result = model.forward_gpu_with_cache(&token_ids, &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 20);
}

// ============================================================================
// Error Path Tests
// ============================================================================

#[test]
fn test_forward_with_cache_boundary_token() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("boundary_token");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Use maximum valid token (vocab_size - 1)
    let max_valid_token = config.vocab_size - 1;
    let result = model.forward_gpu_with_cache(&[max_valid_token], &mut cache);

    assert!(result.is_ok());
}

#[test]
fn test_incremental_with_empty_cache() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("empty_cache");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Incremental forward with empty cache (first token)
    let result = model.forward_gpu_incremental(1, &mut cache);

    // Should work - effectively same as first token
    assert!(result.is_ok());
    assert_eq!(cache.len(), 1);
}

// ============================================================================
// Integration Tests (combining multiple operations)
// ============================================================================

#[test]
fn test_full_inference_workflow() {
    let config = create_kv_test_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("full_workflow");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // 1. Prefill with prompt
    let prompt = vec![1, 2, 3, 4, 5];
    let prefill_result = model.forward_gpu_with_cache(&prompt, &mut cache);
    assert!(prefill_result.is_ok());
    assert_eq!(cache.len(), prompt.len());

    // 2. Decode multiple tokens
    for i in 0..10 {
        let token = (i + 6) % config.vocab_size;
        let decode_result = model.forward_gpu_incremental(token, &mut cache);
        assert!(decode_result.is_ok());
    }

    // Cache should have prompt + decoded tokens
    assert_eq!(cache.len(), prompt.len() + 10);
}

#[test]
fn test_repeated_generations_with_cache_clear() {
    let config = create_kv_single_layer_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("repeated_gen");
    model.with_test_executor(Box::new(mock));

    let mut cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Multiple generation runs
    for run in 0..3 {
        cache.clear();
        assert!(cache.is_empty());

        let prompt = vec![run % config.vocab_size];
        let result = model.forward_gpu_with_cache(&prompt, &mut cache);
        assert!(result.is_ok(), "Run {} failed", run);

        for i in 0..5 {
            let token = (run + i + 1) % config.vocab_size;
            let result = model.forward_gpu_incremental(token, &mut cache);
            assert!(result.is_ok(), "Incremental {} in run {} failed", i, run);
        }
    }
}

#[test]
fn test_kv_cache_with_different_configs() {
    // Test that KV cache works correctly with various model configs
    let configs = vec![
        create_kv_test_config(),
        create_kv_gqa_config(),
        create_kv_single_layer_config(),
        create_kv_deep_config(),
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let mut model = GpuModel::new(config.clone()).expect("test value should be present");

        let mock = MockExecutor::new(&format!("config_{}", i));
        model.with_test_executor(Box::new(mock));

        let mut cache = StreamingKVCache::new(
            config.num_layers,
            64,
            config.num_kv_heads,
            config.head_dim(),
        );

        let result = model.forward_gpu_with_cache(&[1, 2], &mut cache);
        assert!(result.is_ok(), "Config {} failed", i);
        assert_eq!(cache.len(), 2, "Config {} cache length mismatch", i);
    }
}
