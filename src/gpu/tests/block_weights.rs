
#[test]
fn test_block_weights_with_gate() {
    let hidden_dim = 32;
    let intermediate_dim = 64;

    let block = BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * 96],
        qkv_bias: vec![0.0; 96],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: Some(vec![0.01; hidden_dim * intermediate_dim]),
        linear_attn: None,
    moe_experts: None,
    };

    assert!(block.ffn_gate_weight.is_some());
    assert_eq!(
        block.ffn_gate_weight.as_ref().expect("test value should be present").len(),
        hidden_dim * intermediate_dim
    );
}

// ============================================================================
// WeightType Debug/Clone Tests
// ============================================================================

#[test]
fn test_weight_type_debug() {
    let qkv = WeightType::Qkv;
    let output = WeightType::Output;
    let fc1 = WeightType::FfnFc1;
    let fc2 = WeightType::FfnFc2;
    let lm_head = WeightType::LmHead;

    // Test Debug trait
    let qkv_debug = format!("{:?}", qkv);
    assert!(qkv_debug.contains("Qkv"));

    let output_debug = format!("{:?}", output);
    assert!(output_debug.contains("Output"));

    let fc1_debug = format!("{:?}", fc1);
    assert!(fc1_debug.contains("FfnFc1"));

    let fc2_debug = format!("{:?}", fc2);
    assert!(fc2_debug.contains("FfnFc2"));

    let lm_head_debug = format!("{:?}", lm_head);
    assert!(lm_head_debug.contains("LmHead"));
}

#[test]
fn test_weight_type_clone() {
    let original = WeightType::Qkv;
    let cloned = original;

    // Clone should work (Copy trait)
    assert!(matches!(cloned, WeightType::Qkv));
}

// ============================================================================
// GpuModel layer_norm_static Tests (via forward operations)
// ============================================================================

#[test]
fn test_gpu_model_layer_norm_via_forward() {
    // layer_norm_static is tested indirectly via forward_block_idx
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("layer_norm_test");
    model.with_test_executor(Box::new(mock));

    // This exercises layer_norm_static internally
    let input = vec![1.0f32; config.hidden_dim];
    let result = model.forward_block_idx(&input, 1, 0);

    assert!(result.is_ok());
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_single_token_only() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("single_token");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![5];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_gpu_model_generate_max_tokens_zero() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("zero_max_tokens");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(0);
    let prompt = vec![1];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
    // Should return at least the prompt
    let tokens = result.expect("test value should be present");
    assert!(!tokens.is_empty());
}

#[test]
fn test_gpu_model_forward_block_incremental_optimized() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("block_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.forward_block_incremental_optimized(&input, 0, &mut kv_cache);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.hidden_dim);
}

// ============================================================================
// GpuModel GQA Attention Path Tests
// ============================================================================

#[test]
fn test_gpu_model_gqa_forward() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("gqa_forward");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![1, 2];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_gqa_incremental() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("gqa_incremental");
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

#[test]
fn test_gpu_model_gqa_generate() {
    let config = create_gqa_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("gqa_generate");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let result = model.generate(&[1], &gen_config);

    assert!(result.is_ok());
}
