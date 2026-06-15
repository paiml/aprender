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

// ============================================================================
// PMAT-783: dequantize_weight must NOT silently reinterpret quantized bytes as
// raw f32. Regression tests for the `_ => raw-f32` garbage fallback.
// ============================================================================

/// Build a single Q8_0 block (34 bytes: f16 scale + 32 int8 quants) from a
/// known scale and 32 integer quants.
#[cfg(feature = "gpu")]
fn make_q8_0_block(scale: f32, quants: [i8; 32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(34);
    let scale_bits = half::f16::from_f32(scale).to_bits();
    data.extend_from_slice(&scale_bits.to_le_bytes());
    for q in quants {
        data.push(q as u8);
    }
    data
}

/// A Q8_0 weight routed through `dequantize_weight` must dequantize to the same
/// values as the canonical `dequantize_q8_0` reference — NOT the raw-f32 garbage
/// the old `_ =>` fallback produced.
#[test]
#[cfg(feature = "gpu")]
fn test_dequantize_weight_q8_0_matches_reference_not_raw_f32() {
    use crate::gguf::types::GGUF_TYPE_Q8_0;
    use crate::quantize::dequantize_q8_0;

    let config = test_config();
    let model = create_test_model_with_config(&config);

    // One Q8_0 block = 32 elements. Use a non-trivial scale + quants.
    let quants: [i8; 32] = std::array::from_fn(|i| (i as i8) - 16);
    let scale = 0.125_f32;
    let block = make_q8_0_block(scale, quants);

    let weight = crate::gguf::OwnedQuantizedTensor {
        data: block.clone(),
        in_dim: 32,
        out_dim: 1,
        qtype: GGUF_TYPE_Q8_0,
    };

    let got = model
        .dequantize_weight(&weight)
        .expect("Q8_0 must dequantize, not error");
    let want = dequantize_q8_0(&block).expect("reference dequant");

    // Bit-identical to the canonical CPU reference dequantizer.
    assert_eq!(
        got, want,
        "dequantize_weight(Q8_0) must match dequantize_q8_0 reference"
    );

    // And it must differ from the OLD raw-f32 garbage interpretation: reading
    // the 34 packed bytes as 8 little-endian f32s.
    let raw_f32_garbage: Vec<f32> = block
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_ne!(
        got, raw_f32_garbage,
        "Q8_0 must NOT be reinterpreted as raw f32 bytes (the old garbage fallback)"
    );
    // Sanity: the dequantized values match scale * quant exactly.
    for (i, &q) in quants.iter().enumerate() {
        assert!(
            (got[i] - scale * f32::from(q)).abs() < 1e-6,
            "element {i}: got {} want {}",
            got[i],
            scale * f32::from(q)
        );
    }
}

/// A genuinely unsupported quantization type must FAIL LOUD (Err), never be
/// silently reinterpreted as raw f32.
#[test]
#[cfg(feature = "gpu")]
fn test_dequantize_weight_unsupported_type_errors_loud() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    // qtype 9999 has no dequantizer — must hard-error, not return f32-garbage.
    let weight = crate::gguf::OwnedQuantizedTensor {
        data: vec![0u8; 64],
        in_dim: 16,
        out_dim: 1,
        qtype: 9999,
    };

    let result = model.dequantize_weight(&weight);
    assert!(
        result.is_err(),
        "unsupported qtype must Err, not silently reinterpret bytes as f32"
    );
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("9999") && msg.to_lowercase().contains("unsupported"),
        "error must name the unsupported type: {msg}"
    );
}

include!("batch_tests_tiled_single.rs");
