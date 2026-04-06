//! Tests for batched forward pass implementations
//!
//! Coverage targets for batch.rs uncovered paths:
//! - Empty tokens/prompts error paths
//! - Bias handling in attention and FFN
//! - Temperature and top-k sampling branches
//! - batch_throughput_factor ranges
//! - Softmax variants (standard, online, tiled)
//! - Tiled attention with various tile sizes

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    GGUFConfig, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};

fn test_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        rope_theta: 10000.0,
        context_length: 512,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

// ============================================================================
// generate_with_smallvec tests
// ============================================================================

#[test]
fn test_generate_with_smallvec_empty_prompt_error() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate_with_smallvec(&[], &gen_config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(format!("{:?}", err).contains("empty"));
}

#[test]
fn test_generate_with_smallvec_greedy_sampling() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::deterministic(3);

    let result = model.generate_with_smallvec(&[1], &gen_config);
    assert!(result.is_ok());
    let tokens = result.unwrap();
    // Should have at least the prompt token
    assert!(!tokens.is_empty());
    assert_eq!(tokens[0], 1);
}

#[test]
fn test_generate_with_smallvec_temperature_sampling() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::default()
        .with_max_tokens(2)
        .with_temperature(1.0)
        .with_top_k(5);

    let result = model.generate_with_smallvec(&[1], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_generate_with_smallvec_stop_token() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    // Use token 0 as stop token (likely to be generated from zero weights)
    let gen_config = QuantizedGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);

    let result = model.generate_with_smallvec(&[1], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// batch_generate tests
// ============================================================================

#[test]
fn test_batch_generate_empty_prompts_error() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.batch_generate(&[], &gen_config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(format!("{:?}", err).contains("empty"));
}

#[test]
fn test_batch_generate_single_prompt_optimization() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::deterministic(2);

    let prompt: &[u32] = &[1, 2];
    let result = model.batch_generate(&[prompt], &gen_config);
    assert!(result.is_ok());
    let outputs = result.unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].len() >= 2);
}

#[test]
fn test_batch_generate_multiple_prompts() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::deterministic(2);

    let prompt1: &[u32] = &[1];
    let prompt2: &[u32] = &[2, 3];
    let result = model.batch_generate(&[prompt1, prompt2], &gen_config);
    assert!(result.is_ok());
    let outputs = result.unwrap();
    assert_eq!(outputs.len(), 2);
}

#[test]
fn test_batch_generate_with_stop_tokens() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::deterministic(10).with_stop_tokens(vec![0]);

    let prompt1: &[u32] = &[1];
    let prompt2: &[u32] = &[2];
    let result = model.batch_generate(&[prompt1, prompt2], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_batch_generate_with_temperature() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let gen_config = QuantizedGenerateConfig::default()
        .with_max_tokens(2)
        .with_temperature(0.8)
        .with_top_k(10);

    let prompt1: &[u32] = &[1];
    let prompt2: &[u32] = &[2];
    let result = model.batch_generate(&[prompt1, prompt2], &gen_config);
    assert!(result.is_ok());
}

// ============================================================================
// batch_throughput_factor tests
// ============================================================================

#[test]
fn test_batch_throughput_factor_all_ranges() {
    // 0 or 1 => 1.0
    assert!((OwnedQuantizedModel::batch_throughput_factor(0) - 1.0).abs() < f64::EPSILON);
    assert!((OwnedQuantizedModel::batch_throughput_factor(1) - 1.0).abs() < f64::EPSILON);

    // 2..=4 => 1.8
    assert!((OwnedQuantizedModel::batch_throughput_factor(2) - 1.8).abs() < f64::EPSILON);
    assert!((OwnedQuantizedModel::batch_throughput_factor(4) - 1.8).abs() < f64::EPSILON);

    // 5..=8 => 2.5
    assert!((OwnedQuantizedModel::batch_throughput_factor(5) - 2.5).abs() < f64::EPSILON);
    assert!((OwnedQuantizedModel::batch_throughput_factor(8) - 2.5).abs() < f64::EPSILON);

    // 9..=16 => 3.5
    assert!((OwnedQuantizedModel::batch_throughput_factor(9) - 3.5).abs() < f64::EPSILON);
    assert!((OwnedQuantizedModel::batch_throughput_factor(16) - 3.5).abs() < f64::EPSILON);

    // 17..=32 => 5.0
    assert!((OwnedQuantizedModel::batch_throughput_factor(17) - 5.0).abs() < f64::EPSILON);
    assert!((OwnedQuantizedModel::batch_throughput_factor(32) - 5.0).abs() < f64::EPSILON);

    // >32 => 6.0
    assert!((OwnedQuantizedModel::batch_throughput_factor(33) - 6.0).abs() < f64::EPSILON);
    assert!((OwnedQuantizedModel::batch_throughput_factor(100) - 6.0).abs() < f64::EPSILON);
}

// ============================================================================
// forward_batch tests
// ============================================================================

#[test]
fn test_forward_batch_single_token() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.forward_batch(&[1]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_batch_multiple_tokens() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.forward_batch(&[1, 2, 3]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    // batch_size * vocab_size
    assert_eq!(logits.len(), 3 * config.vocab_size);
}

// ============================================================================
// prefill_batch tests
// ============================================================================

#[test]
fn test_prefill_batch_empty_prompt_error() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 100);

    let result = model.prefill_batch(&[], &mut cache);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(format!("{:?}", err).contains("empty"));
}

#[test]
fn test_prefill_batch_single_token() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 100);

    let result = model.prefill_batch(&[1], &mut cache);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_prefill_batch_multiple_tokens() {
    let config = test_config();
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 100);

    let result = model.prefill_batch(&[1, 2, 3], &mut cache);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.vocab_size);
}

// ============================================================================
// standard_softmax tests
// ============================================================================

#[test]
fn test_standard_softmax_empty() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.standard_softmax(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_standard_softmax_single_element() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.standard_softmax(&[1.0]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_standard_softmax_sums_to_one() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.standard_softmax(&[1.0, 2.0, 3.0]);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_standard_softmax_ordering() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.standard_softmax(&[1.0, 2.0, 3.0]);
    // Larger input should have larger output
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

// ============================================================================
// online_softmax tests
// ============================================================================

#[test]
fn test_online_softmax_empty() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let result = model.online_softmax(&[], 4);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_online_softmax_matches_standard() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let standard = model.standard_softmax(&scores);
    let online = model.online_softmax(&scores, 2).unwrap();

    for (s, o) in standard.iter().zip(online.iter()) {
        assert!((s - o).abs() < 1e-5, "standard={}, online={}", s, o);
    }
}

#[test]
fn test_online_softmax_various_tile_sizes() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let standard = model.standard_softmax(&scores);

    for tile_size in [1, 2, 3, 4, 8, 16] {
        let online = model.online_softmax(&scores, tile_size).unwrap();
        for (s, o) in standard.iter().zip(online.iter()) {
            assert!((s - o).abs() < 1e-5, "tile_size={}", tile_size);
        }
    }
}

#[test]
fn test_online_softmax_tile_size_zero() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    // tile_size 0 should be treated as 1
    let result = model.online_softmax(&[1.0, 2.0], 0);
    assert!(result.is_ok());
    let sum: f32 = result.unwrap().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

// ============================================================================
// standard_single_head_attention tests
// ============================================================================

#[test]
fn test_standard_single_head_attention_basic() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let seq_len = 2;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q, K, V: [seq_len, head_dim]
    let q = vec![1.0; seq_len * head_dim];
    let k = vec![1.0; seq_len * head_dim];
    let v = vec![1.0; seq_len * head_dim];

    let result = model.standard_single_head_attention(&q, &k, &v, seq_len, head_dim, scale);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), seq_len * head_dim);
}

#[test]
fn test_standard_single_head_attention_identity() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    // Single position: attention weights will be 1.0, so output = V
    let seq_len = 1;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = vec![1.0; head_dim];
    let k = vec![1.0; head_dim];
    let v = vec![2.0, 3.0, 4.0, 5.0];

    let result = model.standard_single_head_attention(&q, &k, &v, seq_len, head_dim, scale);
    assert!(result.is_ok());
    let output = result.unwrap();
    // Output should be V (since softmax of single element is 1.0)
    for (o, expected) in output.iter().zip(v.iter()) {
        assert!((o - expected).abs() < 1e-5);
    }
}

// ============================================================================
// tiled_single_head_attention tests
// ============================================================================

#[test]
fn test_tiled_single_head_attention_matches_standard() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let seq_len = 4;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Random-ish values
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 3) as f32 * 0.1)
        .collect();

    let standard = model
        .standard_single_head_attention(&q, &k, &v, seq_len, head_dim, scale)
        .unwrap();
    let tiled = model
        .tiled_single_head_attention(&q, &k, &v, seq_len, head_dim, scale, 2)
        .unwrap();

    for (s, t) in standard.iter().zip(tiled.iter()) {
        assert!((s - t).abs() < 1e-4, "standard={}, tiled={}", s, t);
    }
}

include!("batch_tests_tiled_single.rs");
