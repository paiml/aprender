
#[test]
fn test_gpu_model_matmul_split_output() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock =
        MockExecutor::new("split_output").with_matmul_result(vec![0.0f32; config.hidden_dim]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Output);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_gpu_model_matmul_split_ffn_fc1() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock =
        MockExecutor::new("split_fc1").with_matmul_result(vec![0.0f32; config.intermediate_dim]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc1);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.intermediate_dim);
}

#[test]
fn test_gpu_model_matmul_split_ffn_fc2() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("split_fc2").with_matmul_result(vec![0.0f32; config.hidden_dim]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.intermediate_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc2);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_gpu_model_matmul_split_lm_head() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock =
        MockExecutor::new("split_lm_head").with_matmul_result(vec![0.0f32; config.vocab_size]);
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::LmHead);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.vocab_size);
}

// ============================================================================
// GpuModel forward_gpu Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_gpu_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_gpu");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![1, 2, 3];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    // Output should be seq_len * vocab_size
    assert_eq!(logits.len(), token_ids.len() * config.vocab_size);
}

#[test]
fn test_gpu_model_forward_gpu_empty_tokens() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let token_ids: Vec<usize> = vec![];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_forward_gpu_out_of_bounds_token() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    // Token 9999 is out of bounds for vocab_size=50
    let token_ids = vec![1, 9999];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_forward_gpu_owned() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_owned");
    model.with_test_executor(Box::new(mock));

    let token_ids = vec![1, 2];
    let result = model.forward_gpu_owned(&token_ids);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), token_ids.len() * config.vocab_size);
}

// ============================================================================
// GpuModel forward_block_idx Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_block_idx_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_block");
    model.with_test_executor(Box::new(mock));

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.forward_block_idx(&input, 1, 0);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_gpu_model_forward_block_idx_all_layers() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_all_layers");
    model.with_test_executor(Box::new(mock));

    let mut hidden = vec![0.1f32; config.hidden_dim];
    for block_idx in 0..config.num_layers {
        let result = model.forward_block_idx(&hidden, 1, block_idx);
        assert!(result.is_ok(), "Block {} should succeed", block_idx);
        hidden = result.expect("test value should be present");
    }
}

// ============================================================================
// GpuModel generate Tests
// ============================================================================

#[test]
fn test_gpu_model_generate_basic() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let prompt = vec![1, 2];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
    let tokens = result.expect("test value should be present");
    assert!(tokens.len() >= prompt.len());
}

#[test]
fn test_gpu_model_generate_optimized() {
    let config = create_minimal_config();
    let mut model = GpuModel::with_attention_buffers(config, 64).expect("test value should be present");

    let mock = MockExecutor::new("generate_optimized");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let prompt = vec![1];
    let result = model.generate_optimized(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_generate_optimized_empty_prompt() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let gen_config = GpuGenerateConfig::deterministic(5);
    let prompt: Vec<usize> = vec![];
    let result = model.generate_optimized(&prompt, &gen_config);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_generate_with_sampling() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_sampling");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::with_sampling(2, 0.8, 5);
    let prompt = vec![1, 2];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_generate_with_stop_tokens() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("generate_stop");
    model.with_test_executor(Box::new(mock));

    // Stop token 0 will likely be generated since mock returns zeros
    let gen_config = GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);
    let prompt = vec![1];
    let result = model.generate(&prompt, &gen_config);

    assert!(result.is_ok());
}

// ============================================================================
// GpuModel forward_gpu_with_cache Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_gpu_with_cache() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_with_cache");
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
}

#[test]
fn test_gpu_model_forward_gpu_incremental() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("forward_incremental");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // First populate cache with a token
    let _ = model.forward_gpu_with_cache(&[1], &mut kv_cache);

    // Then do incremental forward
    let result = model.forward_gpu_incremental(2, &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_gpu_incremental_optimized() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("incremental_optimized");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_gpu_incremental_optimized(1, &mut kv_cache);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.vocab_size);
}

#[test]
fn test_gpu_model_forward_gpu_incremental_optimized_out_of_bounds() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    // Token 9999 is out of bounds
    let result = model.forward_gpu_incremental_optimized(9999, &mut kv_cache);
    assert!(result.is_err());
}

// ============================================================================
// GpuModel Fused Operations Tests
// ============================================================================

#[test]
fn test_gpu_model_fused_qkv_projection() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.fused_qkv_projection(&input);

    assert!(result.is_ok());
    let (q, k, v) = result.expect("test value should be present");
    assert_eq!(q.len(), config.hidden_dim);
    assert_eq!(k.len(), config.kv_dim());
    assert_eq!(v.len(), config.kv_dim());
}

#[test]
fn test_gpu_model_generate_with_fused_qkv() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("fused_qkv");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(2);
    let result = model.generate_with_fused_qkv(&[1], &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_with_fused_attn_proj() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("fused_attn");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_with_fused_attn_proj(1, &mut kv_cache);
    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_forward_with_fused_output_residual() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config.clone()).expect("test value should be present");

    let mock = MockExecutor::new("fused_residual");
    model.with_test_executor(Box::new(mock));

    let mut kv_cache = StreamingKVCache::new(
        config.num_layers,
        64,
        config.num_kv_heads,
        config.head_dim(),
    );

    let result = model.forward_with_fused_output_residual(1, &mut kv_cache);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModel generate_with_cache Tests
// ============================================================================

#[test]
fn test_gpu_model_generate_with_cache() {
    let config = create_minimal_config();
    let mut model = GpuModel::new(config).expect("test value should be present");

    let mock = MockExecutor::new("gen_with_cache");
    model.with_test_executor(Box::new(mock));

    let gen_config = GpuGenerateConfig::deterministic(3);
    let result = model.generate_with_cache(&[1], &gen_config);

    assert!(result.is_ok());
}

// ============================================================================
// BlockWeights Debug/Clone Tests
// ============================================================================

#[test]
fn test_block_weights_structure() {
    // Verify BlockWeights can be constructed manually
    let hidden_dim = 64;
    let intermediate_dim = 128;
    let qkv_dim = 192;

    let block = BlockWeights {
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
    };

    assert_eq!(block.attn_norm_weight.len(), hidden_dim);
    assert!(block.ffn_gate_weight.is_none());
}
