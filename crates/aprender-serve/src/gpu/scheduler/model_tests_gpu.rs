
#[test]
fn test_gpu_model_large_vocab_incremental() {
    let config = create_large_vocab_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("large_vocab_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // This should hit the CPU fallback path for LM head in incremental
    let result = model.forward_gpu_incremental_optimized(1, &mut kv_cache);
    assert!(result.is_ok());
    let logits = result.expect("logits");
    assert_eq!(logits.len(), config.vocab_size);
}

// ============================================================================
// Multi-layer Forward Tests
// ============================================================================

#[test]
fn test_gpu_model_multi_layer_forward() {
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

    let mock = MockExecutor::new("multi_layer_forward");
    model.with_test_executor(Box::new(mock));

    let result = model.forward_gpu(&[1, 2, 3]);
    assert!(result.is_ok());
    let logits = result.expect("logits");
    assert_eq!(logits.len(), 3 * config.vocab_size);
}

#[test]
fn test_gpu_model_all_blocks_forward_idx() {
    let config = GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 3,
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

    let mock = MockExecutor::new("all_blocks");
    model.with_test_executor(Box::new(mock));

    let mut hidden = vec![0.1f32; config.hidden_dim];
    for block_idx in 0..config.num_layers {
        let result = model.forward_block_idx(&hidden, 1, block_idx);
        assert!(result.is_ok(), "Block {} should succeed", block_idx);
        hidden = result.expect("expected value");
        assert_eq!(hidden.len(), config.hidden_dim);
    }
}

// ============================================================================
// GQA Edge Cases Tests
// ============================================================================

#[test]
fn test_gpu_model_gqa_multiple_q_per_kv() {
    // Test GQA where multiple Q heads share one KV head
    let config = GpuModelConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 16,  // 16 Q heads
        num_kv_heads: 4, // 4 KV heads -> 4 Q heads per KV head
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

    let mock = MockExecutor::new("gqa_multiple");
    model.with_test_executor(Box::new(mock));

    let result = model.forward_gpu(&[1]);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_gqa_incremental_with_cache() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("gqa_cache");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Process multiple tokens to populate cache
    let _ = model.forward_gpu_with_cache(&[1, 2, 3], &mut kv_cache);

    // Do incremental forward with populated cache
    let result = model.forward_gpu_incremental_optimized(4, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// Error Edge Cases Tests
// ============================================================================

#[test]
fn test_gpu_model_empty_generate() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("model");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let result = model.generate(&[], &gen_config);

    assert!(result.is_err(), "Generate with empty prompt should fail");
}

#[test]
fn test_gpu_model_token_at_vocab_boundary() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("boundary");
    model.with_test_executor(Box::new(mock));

    // Test token at vocab boundary (vocab_size - 1)
    let result = model.forward_gpu(&[config.vocab_size - 1]);
    assert!(result.is_ok());

    // Test token exactly at vocab_size (should fail)
    let result = model.forward_gpu(&[config.vocab_size]);
    assert!(result.is_err());
}

// ============================================================================
// WeightType All Variants Tests
// ============================================================================

#[test]
fn test_weight_type_copy_semantics() {
    let types = [
        WeightType::Qkv,
        WeightType::Output,
        WeightType::FfnFc1,
        WeightType::FfnFc2,
        WeightType::LmHead,
    ];

    for wt in types {
        let copied = wt; // Copy
        assert!(matches!(
            (wt, copied),
            (WeightType::Qkv, WeightType::Qkv)
                | (WeightType::Output, WeightType::Output)
                | (WeightType::FfnFc1, WeightType::FfnFc1)
                | (WeightType::FfnFc2, WeightType::FfnFc2)
                | (WeightType::LmHead, WeightType::LmHead)
        ));
    }
}

// ============================================================================
// BlockWeights Field Access Tests
// ============================================================================

#[test]
fn test_block_weights_all_fields() {
    let config = create_minimal_config();
    let block = create_block_weights_with_swiglu(&config);

    // Verify all fields can be accessed
    assert!(!block.attn_norm_weight.is_empty());
    assert!(!block.attn_norm_bias.is_empty());
    assert!(!block.qkv_weight.is_empty());
    assert!(!block.qkv_bias.is_empty());
    assert!(!block.out_weight.is_empty());
    assert!(!block.out_bias.is_empty());
    assert!(!block.ffn_norm_weight.is_empty());
    assert!(!block.ffn_norm_bias.is_empty());
    assert!(!block.ffn_fc1_weight.is_empty());
    assert!(!block.ffn_fc1_bias.is_empty());
    assert!(!block.ffn_fc2_weight.is_empty());
    assert!(!block.ffn_fc2_bias.is_empty());
    assert!(block.ffn_gate_weight.is_some());
}

// ============================================================================
// RoPE Theta Variation Tests
// ============================================================================

#[test]
fn test_gpu_model_custom_rope_theta() {
    // Test with non-default rope_theta
    let config = GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        eps: 1e-5,
        rope_theta: 500000.0, // Different rope_theta (like Llama 3)
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("custom_rope");
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
// Epsilon Variation Tests
// ============================================================================

#[test]
fn test_gpu_model_different_eps() {
    let configs = [1e-5, 1e-6, 1e-8];

    for eps in configs {
        let config = GpuModelConfig {
            hidden_dim: 32,
            intermediate_dim: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 50,
            eps,
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

        let mock = MockExecutor::new(&format!("eps_{}", eps));
        model.with_test_executor(Box::new(mock));

        let result = model.forward_gpu(&[1]);
        assert!(result.is_ok(), "Forward with eps={} should succeed", eps);
    }
}

// ============================================================================
// Sequential Forward Tests (Cache State)
// ============================================================================

#[test]
fn test_gpu_model_sequential_incremental_forwards() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("model");

    let mock = MockExecutor::new("sequential");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        128,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Process multiple incremental forwards
    for token_id in 0..10 {
        let token = token_id % config.vocab_size;
        let result = model.forward_gpu_incremental_optimized(token, &mut kv_cache);
        assert!(result.is_ok(), "Token {} should succeed", token_id);
    }
}

// ============================================================================
// Forward Block Incremental Optimized with All Layers
// ============================================================================

#[test]
fn test_gpu_model_block_incremental_all_layers() {
    let config = GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 3,
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

    let mock = MockExecutor::new("block_all_layers");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let mut hidden = vec![0.1f32; config.hidden_dim];
    for block_idx in 0..config.num_layers {
        let result = model.forward_block_incremental_optimized(&hidden, block_idx, &mut kv_cache);
        assert!(result.is_ok(), "Block {} incremental should succeed", block_idx);
        hidden = result.expect("expected value");
    }
}
