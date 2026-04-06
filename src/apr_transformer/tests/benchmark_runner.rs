//! APR Transformer Coverage Tests Part 3 (PMAT-803)
//!
//! Additional comprehensive tests for coverage gaps identified in:
//! - `AprBenchmarkRunner` methods
//! - `AprTransformer::from_apr_file` error paths
//! - Q4K layer weight handling
//! - SIMD helpers edge cases
//! - Matmul with various dimensions

use crate::apr_transformer::{
    AprBenchmarkRunner, AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer,
    GenerateConfig, Q4KLayerWeights,
};

// ============================================================================
// Part 1: AprBenchmarkRunner Tests
// ============================================================================

#[test]
fn test_benchmark_runner_new() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let runner = AprBenchmarkRunner::new(transformer);

    assert_eq!(runner.warmup_iterations(), 3);
    assert_eq!(runner.measure_iterations(), 10);
}

#[test]
fn test_benchmark_runner_set_warmup_iterations() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);

    runner.set_warmup_iterations(5);
    assert_eq!(runner.warmup_iterations(), 5);

    runner.set_warmup_iterations(0);
    assert_eq!(runner.warmup_iterations(), 0);
}

#[test]
fn test_benchmark_runner_set_measure_iterations() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);

    runner.set_measure_iterations(20);
    assert_eq!(runner.measure_iterations(), 20);

    // Minimum is 1
    runner.set_measure_iterations(0);
    assert_eq!(runner.measure_iterations(), 1);
}

#[test]
fn test_benchmark_runner_benchmark_decode() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);

    // Use minimal iterations for testing
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(2);

    let result = runner.benchmark_decode(&[0, 1], 3);
    assert!(result.is_ok());

    let bench = result.expect("test value should be present");
    assert!(bench.tokens_generated > 0);
    assert!(bench.total_time_ms > 0.0);
}

#[test]
fn test_benchmark_runner_debug() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);
    let runner = AprBenchmarkRunner::new(transformer);

    let debug_str = format!("{runner:?}");
    assert!(debug_str.contains("AprBenchmarkRunner"));
}

// ============================================================================
// Part 2: AprTransformer from_apr_file Error Paths
// ============================================================================

#[test]
fn test_from_apr_file_not_found() {
    let result = AprTransformer::from_apr_file("/nonexistent/path/to/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_from_apr_file_directory() {
    // Attempting to read a directory should fail
    let result = AprTransformer::from_apr_file("/tmp");
    assert!(result.is_err());
}

// ============================================================================
// Part 3: AprTransformer with Q4K Layer Weights
// ============================================================================

#[test]
fn test_transformer_with_q4k_layers() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add Q4K layers
    let q4k_weights = vec![
        Q4KLayerWeights {
            attn_q_weight: Some(vec![0u8; 144]), // 1 Q4K block
            attn_k_weight: Some(vec![0u8; 144]),
            attn_v_weight: Some(vec![0u8; 144]),
            ffn_gate_weight: Some(vec![0u8; 144]),
            ffn_up_weight: Some(vec![0u8; 144]),
            ffn_down_weight: Some(vec![0u8; 144]),
            ..Default::default()
        };
        config.num_layers
    ];

    transformer.q4k_layers = Some(q4k_weights);

    // Forward should still work (fallback to F32 for invalid Q4K data)
    // Note: The actual Q4K data is zeros, which will produce zeros output
    // but the code path should execute
    assert!(transformer.q4k_layers.is_some());
}

#[test]
fn test_transformer_with_q6k_lm_head() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add Q6K lm_head weight
    transformer.lm_head_weight_q6k = Some(vec![0u8; 210]); // 1 Q6K block

    assert!(transformer.lm_head_weight_q6k.is_some());
}

#[test]
fn test_transformer_with_q4k_lm_head() {
    let config = create_test_config();
    let mut transformer = AprTransformer::new(config.clone());

    // Add Q4K lm_head weight
    transformer.lm_head_weight_q4k = Some(vec![0u8; 144]); // 1 Q4K block

    assert!(transformer.lm_head_weight_q4k.is_some());
}

// ============================================================================
// Part 4: Matmul Edge Cases
// ============================================================================

#[test]
fn test_forward_with_odd_hidden_dim() {
    // Hidden dim not divisible by num_heads
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 65, // Not cleanly divisible by 4
        num_layers: 1,
        num_heads: 5,
        num_kv_heads: 5,
        vocab_size: 50,
        intermediate_dim: 130,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_large_intermediate_dim() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        intermediate_dim: 512, // Large intermediate
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_small_vocab() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 10, // Very small vocab
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), config.vocab_size);
}

// ============================================================================
// Part 5: RoPE Edge Cases
// ============================================================================

#[test]
fn test_forward_with_low_rope_theta() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 100.0, // Very low theta
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2, 3, 4]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_very_high_rope_theta() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 1_000_000.0, // Very high theta (like LLaMA 3.1)
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_rope_many_positions() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Process many positions to exercise RoPE at various positions
    for pos in 0..50 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 6: GQA Variations
// ============================================================================

#[test]
fn test_forward_gqa_8_to_1_ratio() {
    // 8 query heads, 1 KV head (aggressive GQA)
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut transformer = AprTransformer::new(config.clone());

    // Adjust QKV for GQA dimensions
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let result = transformer.forward(&[0, 1, 2]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_cache_gqa_8_to_1() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 1,
        vocab_size: 50,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut transformer = AprTransformer::new(config.clone());

    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let qkv_out_dim = config.hidden_dim + 2 * kv_dim;

    for layer in &mut transformer.layers {
        layer.qkv_weight = vec![0.01; config.hidden_dim * qkv_out_dim];
    }

    let mut cache = AprKVCache::new(&config);

    for pos in 0..5 {
        let result = transformer.forward_with_cache(pos as u32, &mut cache, pos);
        assert!(result.is_ok());
    }
}

// ============================================================================
// Part 7: Layer Norm Variations
// ============================================================================

#[test]
fn test_forward_with_very_small_eps() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-12, // Very small epsilon
        eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0, 1]);
    assert!(result.is_ok());
}

#[test]
fn test_forward_with_large_eps() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 0.1, // Large epsilon
        eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

// ============================================================================
// Part 8: GenerateConfig Variations
// ============================================================================

#[test]
fn test_generate_with_cache_top_k_sampling() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 1.0,
        top_k: 10, // Top-k sampling
        top_p: 1.0,
        repetition_penalty: 1.0,
        trace: false,
        stop_tokens: vec![],
    };

    let result = transformer.generate_with_cache(&[0, 1], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_cache_top_p_sampling() {
    let config = create_test_config();
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5, // Nucleus sampling
        repetition_penalty: 1.0,
        trace: false,
        stop_tokens: vec![],
    };

    let result = transformer.generate_with_cache(&[0], &gen_config);
    assert!(result.is_ok());
}

include!("generate.rs");
