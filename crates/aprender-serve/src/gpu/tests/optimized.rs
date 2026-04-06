
// ============================================================================
// optimized_lm_head_argmax_transposed Additional Tests
// ============================================================================

#[test]
fn test_optimized_lm_head_argmax_deterministic() {
    let hidden_dim = 64;
    let vocab_size = 1000;

    let hidden = vec![0.1f32; hidden_dim];
    let weight_t = vec![0.01f32; vocab_size * hidden_dim];
    let bias = vec![0.0f32; vocab_size];

    // Run multiple times to verify determinism
    let r1 = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    let r2 = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);

    assert_eq!(r1, r2, "results should be deterministic");
}

#[test]
fn test_optimized_lm_head_argmax_with_varied_bias() {
    let hidden_dim = 32;
    let vocab_size = 100;

    let hidden = vec![0.1f32; hidden_dim];
    let weight_t = vec![0.0f32; vocab_size * hidden_dim]; // Zero weights

    // Bias determines the winner
    let mut bias = vec![0.0f32; vocab_size];
    bias[42] = 10.0;

    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert_eq!(result, 42);
}

// ============================================================================
// simplified_attention Additional Tests
// ============================================================================

#[test]
fn test_simplified_attention_multiple_heads() {
    let config = GpuModelConfig {
        hidden_dim: 64,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        intermediate_dim: 128,
        num_layers: 1,
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
    };

    let seq_len = 4;
    // MHA: qkv has 3 * hidden_dim per position
    let qkv = vec![0.1f32; seq_len * 3 * config.hidden_dim];

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), seq_len * config.hidden_dim);
}

#[test]
fn test_simplified_attention_longer_sequence() {
    let config = create_test_config();
    let seq_len = 16;
    let qkv = vec![0.1f32; seq_len * 3 * config.hidden_dim];

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModelConfig Additional Coverage
// ============================================================================

#[test]
fn test_gpu_model_config_kv_dim_mha() {
    let config = create_test_config();
    // MHA: kv_dim = num_kv_heads * head_dim = num_kv_heads * (hidden_dim / num_heads)
    // For our config: 4 * (64/4) = 64 = hidden_dim
    assert_eq!(config.kv_dim(), config.hidden_dim);
}

#[test]
fn test_gpu_model_config_kv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: 2 * (64/8) = 2 * 8 = 16
    let expected_kv_dim = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    assert_eq!(config.kv_dim(), expected_kv_dim);
}

#[test]
fn test_gpu_model_config_qkv_dim_mha() {
    let config = create_test_config();
    // MHA: qkv_dim = hidden_dim + 2*kv_dim = 3*hidden_dim
    assert_eq!(config.qkv_dim(), 3 * config.hidden_dim);
}

#[test]
fn test_gpu_model_config_qkv_dim_gqa() {
    let config = create_gqa_config();
    // GQA: qkv_dim = hidden_dim + 2*kv_dim = 64 + 2*16 = 96
    let expected = config.hidden_dim + 2 * config.kv_dim();
    assert_eq!(config.qkv_dim(), expected);
}

#[test]
fn test_gpu_model_config_head_dim() {
    let config = create_test_config();
    assert_eq!(config.head_dim(), config.hidden_dim / config.num_heads);
}

// ============================================================================
// GpuModel with MockExecutor Tests
// ============================================================================

#[test]
fn test_gpu_model_with_test_executor() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    assert!(!model.has_test_executor());

    let mock = MockExecutor::new("test");
    model.with_test_executor(Box::new(mock));

    assert!(model.has_test_executor());
}

#[test]
fn test_gpu_model_clear_test_executor() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("test");
    model.with_test_executor(Box::new(mock));
    assert!(model.has_test_executor());

    model.clear_test_executor();
    assert!(!model.has_test_executor());
}

#[test]
fn test_mock_executor_failure() {
    let config = create_small_vocab_config();
    let mut model = GpuModel::new(config).expect("model creation");

    // Create mock that fails on matmul
    let mock = MockExecutor::new("failing").with_matmul_failure();
    model.with_test_executor(Box::new(mock));

    let tokens = vec![1, 2];
    let result = forward_single_token(&mut model, &tokens);

    // Should fail due to mock failure
    assert!(result.is_err());
}

// ============================================================================
// GpuGenerateConfig Tests
// ============================================================================

use crate::gpu::scheduler::GpuGenerateConfig;

#[test]
fn test_gpu_generate_config_default() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_deterministic() {
    let config = GpuGenerateConfig::deterministic(100);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_gpu_generate_config_with_sampling() {
    let config = GpuGenerateConfig::with_sampling(50, 0.7, 40);
    assert_eq!(config.max_tokens, 50);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
}

#[test]
fn test_gpu_generate_config_with_stop_tokens() {
    let config = GpuGenerateConfig::deterministic(32).with_stop_tokens(vec![1, 2, 50256]);
    assert_eq!(config.stop_tokens, vec![1, 2, 50256]);
}

#[test]
fn test_gpu_generate_config_chained() {
    let config = GpuGenerateConfig::with_sampling(128, 0.9, 50).with_stop_tokens(vec![0, 1]);

    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.9);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens, vec![0, 1]);
}

// ============================================================================
// AttentionBuffers Tests
// ============================================================================

use crate::gpu::scheduler::AttentionBuffers;

#[test]
fn test_attention_buffers_new() {
    let config = create_test_config();
    let max_seq_len = 128;

    let buffers = AttentionBuffers::new(&config, max_seq_len);

    assert_eq!(buffers.q_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.scores_buffer.len(), config.num_heads * max_seq_len);
    assert_eq!(buffers.output_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.kv_proj_buffer.len(), config.hidden_dim);
    assert_eq!(buffers.ffn_buffer.len(), config.intermediate_dim);
    assert_eq!(buffers.max_seq_len, max_seq_len);
}

#[test]
fn test_attention_buffers_reset() {
    let config = create_test_config();
    let mut buffers = AttentionBuffers::new(&config, 64);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);
    buffers.kv_proj_buffer.fill(4.0);
    buffers.ffn_buffer.fill(5.0);

    // Reset should zero everything
    buffers.reset();

    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
}

#[test]
fn test_attention_buffers_gqa_config() {
    let config = create_gqa_config();
    let max_seq_len = 256;

    let buffers = AttentionBuffers::new(&config, max_seq_len);

    // GQA config has 8 heads
    assert_eq!(buffers.scores_buffer.len(), 8 * max_seq_len);
}

// ============================================================================
// WeightType and matmul_split Tests
// ============================================================================

use crate::gpu::scheduler::WeightType;

#[test]
fn test_weight_type_enum() {
    // Verify all variants exist and can be cloned/debugged
    let qkv = WeightType::Qkv;
    let output = WeightType::Output;
    let fc1 = WeightType::FfnFc1;
    let fc2 = WeightType::FfnFc2;
    let lm_head = WeightType::LmHead;

    // Clone test
    let _cloned = qkv;

    // Debug test
    let _ = format!("{:?}", output);
    let _ = format!("{:?}", fc1);
    let _ = format!("{:?}", fc2);
    let _ = format!("{:?}", lm_head);
}

#[test]
fn test_matmul_split_qkv() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Qkv);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.qkv_dim());
}

#[test]
fn test_matmul_split_output() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::Output);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_matmul_split_ffn_fc1() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc1);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.intermediate_dim);
}

#[test]
fn test_matmul_split_ffn_fc2() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.intermediate_dim];
    let result = model.matmul_split(&input, 0, WeightType::FfnFc2);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_matmul_split_lm_head() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];
    let result = model.matmul_split(&input, 0, WeightType::LmHead);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), config.vocab_size);
}

#[test]
fn test_matmul_split_all_layers() {
    let config = create_test_config();
    let mut model = GpuModel::new(config.clone()).expect("model creation");

    let input = vec![0.1f32; config.hidden_dim];

    for layer_idx in 0..config.num_layers {
        let result = model.matmul_split(&input, layer_idx, WeightType::Qkv);
        assert!(result.is_ok(), "layer {} QKV should work", layer_idx);
    }
}

// ============================================================================
// do_matmul and do_matmul_transpose_b Tests
// ============================================================================

#[test]
fn test_do_matmul_basic() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let a = vec![1.0f32; 64];
    let b = vec![0.1f32; 64 * 128];

    let result = model.do_matmul(&a, &b, 1, 64, 128);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 128);
}

#[test]
fn test_do_matmul_with_test_executor() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("matmul_test");
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 32];
    let b = vec![0.1f32; 32 * 64];

    let result = model.do_matmul(&a, &b, 1, 32, 64);
    assert!(result.is_ok());
}

#[test]
fn test_do_matmul_transpose_b() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let a = vec![1.0f32; 32];
    // b is transposed: [n, k] = [64, 32]
    let b = vec![0.1f32; 64 * 32];

    let result = model.do_matmul_transpose_b(&a, &b, 1, 32, 64);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 64);
}

#[test]
fn test_do_matmul_transpose_b_with_mock() {
    let config = create_test_config();
    let mut model = GpuModel::new(config).expect("model creation");

    let mock = MockExecutor::new("transpose_test");
    model.with_test_executor(Box::new(mock));

    let a = vec![1.0f32; 16];
    let b = vec![0.1f32; 32 * 16];

    let result = model.do_matmul_transpose_b(&a, &b, 1, 16, 32);
    assert!(result.is_ok());
}

// ============================================================================
// GpuModelConfig Additional Tests
// ============================================================================

#[test]
fn test_gpu_model_config_is_gqa_true() {
    let config = create_gqa_config();
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_is_gqa_false() {
    let config = create_test_config();
    assert!(!config.is_gqa());
}
