//! Coverage Assault 01: gpu/scheduler/batch.rs
//!
//! Target: Increase batch.rs coverage from 20.97% to >40%
//! Focus: argmax, optimized_lm_head_argmax_transposed, simplified_attention

use realizar::gpu::scheduler::GpuModelConfig;

// ============================================================================
// argmax tests
// ============================================================================

#[test]
fn test_argmax_small_vocab() {
    use realizar::gpu::scheduler::batch::argmax;

    // Small vocab (<= 1024) uses simple iterator path
    let logits = vec![0.1, 0.5, 0.3, 0.2];
    assert_eq!(argmax(&logits), 1);

    // Max at first position
    let logits = vec![1.0, 0.5, 0.3, 0.2];
    assert_eq!(argmax(&logits), 0);

    // Max at last position
    let logits = vec![0.1, 0.5, 0.3, 2.0];
    assert_eq!(argmax(&logits), 3);

    // Single element
    let logits = vec![1.0];
    assert_eq!(argmax(&logits), 0);

    // Negative values
    let logits = vec![-1.0, -0.5, -2.0, -0.1];
    assert_eq!(argmax(&logits), 3);

    // Equal values - max_by returns last with equal comparison
    let logits = vec![1.0, 1.0, 1.0, 1.0];
    assert_eq!(argmax(&logits), 3);
}

#[test]
fn test_argmax_exactly_1024() {
    use realizar::gpu::scheduler::batch::argmax;

    // Boundary case: exactly 1024 elements (still uses simple path)
    let mut logits = vec![0.0f32; 1024];
    logits[512] = 1.0;
    assert_eq!(argmax(&logits), 512);
}

#[test]
fn test_argmax_large_vocab() {
    use realizar::gpu::scheduler::batch::argmax;

    // Large vocab (> 1024) uses chunked parallel path
    let mut logits = vec![0.0f32; 4096];
    logits[2048] = 1.0;
    assert_eq!(argmax(&logits), 2048);

    // Max in last chunk
    let mut logits = vec![0.0f32; 8192];
    logits[7000] = 1.0;
    assert_eq!(argmax(&logits), 7000);

    // Max at very end
    let mut logits = vec![0.0f32; 32000];
    logits[31999] = 1.0;
    assert_eq!(argmax(&logits), 31999);

    // Max at start with large vocab
    let mut logits = vec![0.0f32; 32000];
    logits[0] = 1.0;
    assert_eq!(argmax(&logits), 0);
}

#[test]
fn test_argmax_chunk_boundary() {
    use realizar::gpu::scheduler::batch::argmax;

    // Max exactly at chunk boundary (4096)
    let mut logits = vec![0.0f32; 8192];
    logits[4095] = 1.0; // Last of first chunk
    assert_eq!(argmax(&logits), 4095);

    // Max at start of second chunk
    let mut logits = vec![0.0f32; 8192];
    logits[4096] = 1.0; // First of second chunk
    assert_eq!(argmax(&logits), 4096);
}

// ============================================================================
// optimized_lm_head_argmax_transposed tests
// ============================================================================

#[test]
fn test_optimized_lm_head_argmax_basic() {
    use realizar::gpu::scheduler::batch::optimized_lm_head_argmax_transposed;

    // Simple 4-token vocab, hidden_dim=2
    let hidden = vec![1.0, 0.0];
    // weight_t is [vocab_size, hidden_dim] = [4, 2]
    // Each row is a vocab entry: [w0, w1]
    let weight_t = vec![
        0.0, 0.0, // vocab 0: dot = 0
        1.0, 0.0, // vocab 1: dot = 1
        2.0, 0.0, // vocab 2: dot = 2 (winner)
        0.5, 0.0, // vocab 3: dot = 0.5
    ];
    let bias = vec![0.0, 0.0, 0.0, 0.0];

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 4);
    assert_eq!(result, 2);
}

#[test]
fn test_optimized_lm_head_argmax_with_bias() {
    use realizar::gpu::scheduler::batch::optimized_lm_head_argmax_transposed;

    // Bias changes the winner
    let hidden = vec![1.0, 0.0];
    let weight_t = vec![
        0.0, 0.0, // vocab 0: dot = 0
        1.0, 0.0, // vocab 1: dot = 1
        2.0, 0.0, // vocab 2: dot = 2
        0.5, 0.0, // vocab 3: dot = 0.5
    ];
    let bias = vec![10.0, 0.0, 0.0, 0.0]; // vocab 0 gets +10 bias

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 4);
    assert_eq!(result, 0); // bias makes vocab 0 win
}

#[test]
fn test_optimized_lm_head_argmax_larger_vocab() {
    use realizar::gpu::scheduler::batch::optimized_lm_head_argmax_transposed;

    // 8192 vocab (spans multiple chunks)
    let hidden_dim = 64;
    let vocab_size = 8192;

    let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();

    // All weights are zero except for vocab entry 5000
    let mut weight_t = vec![0.0f32; vocab_size * hidden_dim];
    // Set vocab entry 5000 to all 1.0s
    for i in 0..hidden_dim {
        weight_t[5000 * hidden_dim + i] = 1.0;
    }

    let bias = vec![0.0f32; vocab_size];

    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert_eq!(result, 5000);
}

// ============================================================================
// simplified_attention tests
// ============================================================================

#[test]
fn test_simplified_attention_single_token() {
    use realizar::gpu::scheduler::batch::simplified_attention;

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 8,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 16,
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

    // Single token, seq_len=1
    // QKV layout: [seq_len * hidden_dim] for Q, K, V each
    // Total: 3 * 1 * 8 = 24 elements
    let qkv = vec![
        // Q (8 elements)
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // K (8 elements)
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // V (8 elements)
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ];

    let result = simplified_attention(&config, &qkv, 1);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 8); // seq_len * hidden_dim = 1 * 8
}

#[test]
fn test_simplified_attention_two_tokens() {
    use realizar::gpu::scheduler::batch::simplified_attention;

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 4,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 8,
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

    // Two tokens, seq_len=2
    // QKV layout: [seq_len * hidden_dim] for each
    // Total: 3 * 2 * 4 = 24 elements
    let qkv = vec![
        // Q: 2 positions x 4 dim
        1.0, 0.0, 1.0, 0.0, // pos 0
        0.0, 1.0, 0.0, 1.0, // pos 1
        // K: 2 positions x 4 dim
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, // V: 2 positions x 4 dim
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
    ];

    let result = simplified_attention(&config, &qkv, 2);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 8); // seq_len * hidden_dim = 2 * 4
}

// ============================================================================
// GpuModelConfig helper method tests
// ============================================================================

#[test]
fn test_gpu_model_config_helpers() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        num_layers: 32,
        intermediate_dim: 11008,
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

    // Test head_dim
    assert_eq!(config.head_dim(), 128); // 4096 / 32

    // Test kv_dim
    assert_eq!(config.kv_dim(), 1024); // 8 * 128

    // Test qkv_dim
    assert_eq!(config.qkv_dim(), 6144); // 4096 + 2*1024

    // Test is_gqa
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_mha() {
    // Standard MHA (not GQA)
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12, // Same as num_heads
        num_layers: 12,
        intermediate_dim: 3072,
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

    // Test is_gqa returns false for MHA
    assert!(!config.is_gqa());

    // Test head_dim
    assert_eq!(config.head_dim(), 64); // 768 / 12

    // Test kv_dim equals hidden_dim for MHA
    assert_eq!(config.kv_dim(), 768); // 12 * 64
}
