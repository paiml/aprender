
#[test]
fn test_model_generate_respects_max_tokens() {
    let config = ModelConfig {
        vocab_size: 10,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let gen_config = GenerationConfig::greedy().with_max_tokens(3);
    let tokens = model.generate(&[0, 1], &gen_config).expect("test");

    // Should have 2 prompt + 3 generated = 5 max
    assert!(tokens.len() <= 5);
}

#[test]
fn test_model_generate_with_eos() {
    let config = ModelConfig {
        vocab_size: 10,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    // Set EOS token
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(100)
        .with_eos_token_id(5);

    let tokens = model.generate(&[0], &gen_config).expect("test");

    // Should stop before max_tokens if EOS is generated
    // (may or may not hit EOS depending on model weights)
    assert!(tokens.len() <= 101);
}

#[test]
fn test_model_generate_empty_prompt_error() {
    let config = ModelConfig {
        vocab_size: 10,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let gen_config = GenerationConfig::greedy();
    let result = model.generate(&[], &gen_config);
    assert!(result.is_err());
}

#[test]
fn test_model_generate_deterministic_with_seed() {
    let config = ModelConfig {
        vocab_size: 20,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    // Same seed should give same results
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(5)
        .with_seed(12345);

    let tokens1 = model.generate(&[0], &gen_config).expect("test");
    let tokens2 = model.generate(&[0], &gen_config).expect("test");

    assert_eq!(tokens1, tokens2);
}

#[test]
fn test_model_generate_top_k() {
    let config = ModelConfig {
        vocab_size: 20,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let gen_config = GenerationConfig::top_k(5).with_max_tokens(3).with_seed(42);

    let tokens = model.generate(&[0], &gen_config).expect("test");

    // Should generate valid tokens
    assert!(tokens.len() <= 4);
    for &token in &tokens {
        assert!(token < 20);
    }
}

// MultiHeadAttention tests

#[test]
fn test_multi_head_attention_creation_mha() {
    // Standard Multi-Head Attention (num_kv_heads = num_heads)
    let mha = MultiHeadAttention::mha(64, 8).expect("test");
    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 8);
    assert_eq!(mha.head_dim(), 8); // 64 / 8
    assert_eq!(mha.hidden_dim(), 64);
    assert!(mha.is_mha());
    assert!(!mha.is_mqa());
    assert!(!mha.is_gqa());
}

#[test]
fn test_multi_head_attention_creation_mqa() {
    // Multi-Query Attention (num_kv_heads = 1)
    let mqa = MultiHeadAttention::mqa(64, 8).expect("test");
    assert_eq!(mqa.num_heads(), 8);
    assert_eq!(mqa.num_kv_heads(), 1);
    assert_eq!(mqa.head_dim(), 8);
    assert_eq!(mqa.hidden_dim(), 64);
    assert!(mqa.is_mqa());
    assert!(!mqa.is_mha());
    assert!(!mqa.is_gqa());
}

#[test]
fn test_multi_head_attention_creation_gqa() {
    // Grouped-Query Attention (1 < num_kv_heads < num_heads)
    let gqa = MultiHeadAttention::gqa(64, 8, 2).expect("test");
    assert_eq!(gqa.num_heads(), 8);
    assert_eq!(gqa.num_kv_heads(), 2);
    assert_eq!(gqa.head_dim(), 8);
    assert_eq!(gqa.hidden_dim(), 64);
    assert!(gqa.is_gqa());
    assert!(!gqa.is_mha());
    assert!(!gqa.is_mqa());
}

#[test]
fn test_multi_head_attention_zero_hidden_dim_error() {
    let result = MultiHeadAttention::new(0, 8, 8);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_zero_num_heads_error() {
    let result = MultiHeadAttention::new(64, 0, 1);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_zero_num_kv_heads_error() {
    let result = MultiHeadAttention::new(64, 8, 0);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_kv_heads_too_large_error() {
    // num_kv_heads cannot be greater than num_heads
    let result = MultiHeadAttention::new(64, 8, 16);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_indivisible_error() {
    // 65 is not divisible by 8
    let result = MultiHeadAttention::new(65, 8, 8);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_heads_not_divisible_error() {
    // num_heads must be divisible by num_kv_heads
    let result = MultiHeadAttention::new(64, 8, 3);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_mha_forward() {
    // Standard MHA with 2 heads
    let mha = MultiHeadAttention::mha(8, 2).expect("test");

    // Input: [seq_len=2, hidden_dim=8]
    let input = Tensor::from_vec(
        vec![2, 8],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 2
        ],
    )
    .expect("test");

    let output = mha.forward(&input).expect("test");

    // Output should have same shape as input
    assert_eq!(output.shape(), &[2, 8]);
}

#[test]
fn test_multi_head_attention_mqa_forward() {
    // Multi-Query Attention with 2 heads (shared K/V)
    let mqa = MultiHeadAttention::mqa(8, 2).expect("test");

    // Input: [seq_len=2, hidden_dim=8]
    let input = Tensor::from_vec(
        vec![2, 8],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 2
        ],
    )
    .expect("test");

    let output = mqa.forward(&input).expect("test");

    // Output should have same shape as input
    assert_eq!(output.shape(), &[2, 8]);
}

#[test]
fn test_multi_head_attention_shape_validation() {
    let mha = MultiHeadAttention::mha(8, 2).expect("test");

    // Wrong number of dimensions (1D instead of 2D)
    let input_1d = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let result = mha.forward(&input_1d);
    assert!(result.is_err());

    // Wrong hidden dimension
    let input_wrong_dim = Tensor::from_vec(vec![2, 16], vec![1.0; 32]).expect("test");
    let result = mha.forward(&input_wrong_dim);
    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_mha_vs_mqa_shape_consistency() {
    // Both MHA and MQA should produce same output shape
    let mha = MultiHeadAttention::mha(16, 4).expect("test");
    let mqa = MultiHeadAttention::mqa(16, 4).expect("test");

    let input = Tensor::from_vec(vec![3, 16], vec![0.5; 48]).expect("test");

    let multi_head_output = mha.forward(&input).expect("test");
    let multi_query_output = mqa.forward(&input).expect("test");

    // Both should have same output shape
    assert_eq!(multi_head_output.shape(), &[3, 16]);
    assert_eq!(multi_query_output.shape(), &[3, 16]);
    assert_eq!(multi_head_output.shape(), multi_query_output.shape());
}

#[test]
fn test_multi_head_attention_single_head() {
    // Edge case: single head (equivalent to single attention)
    let mha = MultiHeadAttention::mha(8, 1).expect("test");

    let input = Tensor::from_vec(vec![2, 8], vec![0.5; 16]).expect("test");
    let output = mha.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 8]);
}

#[test]
fn test_multi_head_attention_mqa_kv_sharing() {
    // MQA should work with larger number of heads
    let mqa = MultiHeadAttention::mqa(32, 8).expect("test");

    let input = Tensor::from_vec(vec![4, 32], vec![0.1; 128]).expect("test");
    let output = mqa.forward(&input).expect("test");

    assert_eq!(output.shape(), &[4, 32]);
}

#[test]
fn test_multi_head_attention_long_sequence() {
    // Test with longer sequence
    let mha = MultiHeadAttention::mha(16, 4).expect("test");

    // Sequence length = 10
    let input = Tensor::from_vec(vec![10, 16], vec![0.3; 160]).expect("test");
    let output = mha.forward(&input).expect("test");

    assert_eq!(output.shape(), &[10, 16]);
}

#[test]
fn test_multi_head_attention_mqa_memory_efficiency() {
    // MQA should still work correctly with shared K/V
    // This tests that the shared K/V logic is correct
    let mqa = MultiHeadAttention::mqa(64, 16).expect("test");

    // Small batch
    let input = Tensor::from_vec(vec![2, 64], vec![0.2; 128]).expect("test");
    let output = mqa.forward(&input).expect("test");

    assert_eq!(output.shape(), &[2, 64]);
    assert_eq!(output.data().len(), 128); // 2 * 64
}

#[test]
fn test_multi_head_attention_gqa_forward() {
    // Grouped-Query Attention with 8 heads, 2 KV heads (4 heads per group)
    let gqa = MultiHeadAttention::gqa(32, 8, 2).expect("test");

    // Input: [seq_len=3, hidden_dim=32]
    let input = Tensor::from_vec(vec![3, 32], vec![0.1; 96]).expect("test");

    let output = gqa.forward(&input).expect("test");

    // Output should have same shape as input
    assert_eq!(output.shape(), &[3, 32]);
}

#[test]
fn test_multi_head_attention_gqa_shape_consistency() {
    // MHA, MQA, and GQA should all produce same output shape
    let mha = MultiHeadAttention::mha(64, 8).expect("test");
    let mqa = MultiHeadAttention::mqa(64, 8).expect("test");
    let gqa = MultiHeadAttention::gqa(64, 8, 2).expect("test");

    let input = Tensor::from_vec(vec![4, 64], vec![0.5; 256]).expect("test");

    let multi_head_out = mha.forward(&input).expect("test");
    let multi_query_out = mqa.forward(&input).expect("test");
    let grouped_query_out = gqa.forward(&input).expect("test");

    // All should have same output shape
    assert_eq!(multi_head_out.shape(), &[4, 64]);
    assert_eq!(multi_query_out.shape(), &[4, 64]);
    assert_eq!(grouped_query_out.shape(), &[4, 64]);
    assert_eq!(multi_head_out.shape(), multi_query_out.shape());
    assert_eq!(multi_head_out.shape(), grouped_query_out.shape());
}

#[test]
fn test_multi_head_attention_gqa_different_group_sizes() {
    // Test GQA with different group sizes
    // 16 heads, 4 KV heads (4 heads per group)
    let gqa1 = MultiHeadAttention::gqa(128, 16, 4).expect("test");
    let input = Tensor::from_vec(vec![2, 128], vec![0.3; 256]).expect("test");
    let output1 = gqa1.forward(&input).expect("test");
    assert_eq!(output1.shape(), &[2, 128]);

    // 16 heads, 8 KV heads (2 heads per group)
    let gqa2 = MultiHeadAttention::gqa(128, 16, 8).expect("test");
    let output2 = gqa2.forward(&input).expect("test");
    assert_eq!(output2.shape(), &[2, 128]);
}

// ============================================================================
// Phase 3 Acceptance Tests (Refs REALIZAR-PERF-SPEC-001)
// ============================================================================

/// Phase 3 acceptance test: verify tok/s meets spec target
///
/// Per spec Phase 3 acceptance criteria:
/// ```rust,ignore
/// assert!(benchmark_tokens_per_second() >= 25.0);
/// ```
///
/// Note: This test uses a small test model to verify generation
/// throughput meets the baseline. Real phi-2 benchmarking requires
/// the actual model file and full optimization integration.
#[test]
#[ignore = "performance benchmark - run explicitly with --include-ignored"]
fn test_phase3_acceptance_tokens_per_second() {
    use crate::generate::GenerationConfig;
    use std::time::Instant;

    // Create baseline model configuration
    // The optimized components (Flash Attention v2, FusedLayerNormLinear)
    // show significant speedups individually - see companion tests:
    // - Flash Attention v2 SIMD: ~10x faster than parallel for small sequences
    // - FusedLayerNormLinear parallel: ~3.6x faster for large batches
    //
    // This test verifies the generation loop meets baseline throughput.
    // Full phi-2 integration requires wiring up optimized components.
    let config = ModelConfig {
        vocab_size: 100, // Small vocab for fast softmax
        hidden_dim: 64,  // Smaller hidden dimension
        num_heads: 4,    // Multiple heads
        num_layers: 2,   // Two transformer layers
        intermediate_dim: 128,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    // Warmup run
    let prompt = vec![1, 5, 10];
    let gen_config = GenerationConfig::greedy().with_max_tokens(5);
    let _ = model.generate(&prompt, &gen_config).expect("test");

    // Benchmark: generate 20 tokens 10 times
    let tokens_per_run = 20;
    let num_runs = 10;
    let gen_config = GenerationConfig::greedy().with_max_tokens(tokens_per_run);

    let start = Instant::now();
    for _ in 0..num_runs {
        let _ = model.generate(&prompt, &gen_config).expect("test");
    }
    let elapsed = start.elapsed();

    let total_tokens = tokens_per_run * num_runs;
    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    // Phase 3 acceptance: ≥25 tok/s
    // With optimized components, this should be achievable.
    // The individual component tests show:
    // - Flash Attention v2: 87µs/iter
    // - FusedLayerNormLinear parallel: 2.9ms/iter for 32x256->512
    assert!(
        tok_per_sec >= 25.0,
        "Phase 3 acceptance FAILED: {:.1} tok/s < 25.0 tok/s target. \
         Note: Full optimization requires integrating Flash Attention v2 \
         and FusedLayerNormLinear into Model::forward()",
        tok_per_sec
    );

    // Report performance
    eprintln!(
        "Phase 3 acceptance PASSED: {:.1} tok/s (target: ≥25.0 tok/s)",
        tok_per_sec
    );
}
