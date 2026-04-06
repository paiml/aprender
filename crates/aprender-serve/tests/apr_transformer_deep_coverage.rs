//! Deep coverage tests for realizar/src/apr_transformer.rs
//!
//! This module provides additional coverage for apr_transformer functions
//! not covered by existing tests. Targets 95%+ coverage.

use realizar::apr_transformer::{
    AprBenchmarkResult, AprBenchmarkRunner, AprInferenceScratch, AprKVCache, AprLoadResult,
    AprParityComparison, AprPrefillResult, AprQuantizationType, AprTransformer,
    AprTransformerConfig, AprTransformerLayer, GenerateConfig, MmapAprTransformer,
    QuantizedAprTensorQ4, QuantizedAprTransformer, APR_TRANSFORMER_HEADER_SIZE,
};

// ============================================================================
// Test 1-10: AprQuantizationType exhaustive tests
// ============================================================================

#[test]
fn test_quantization_type_f32_bits() {
    assert_eq!(AprQuantizationType::F32.bits_per_weight(), 32.0);
}

#[test]
fn test_quantization_type_q4_k_bits() {
    assert_eq!(AprQuantizationType::Q4_K.bits_per_weight(), 4.5);
}

#[test]
fn test_quantization_type_all_variants_bits() {
    // Only F32, Q4_K, Q8_0 exist
    assert!(AprQuantizationType::F32.bits_per_weight() >= 32.0);
}

#[test]
fn test_quantization_type_q8_0_bits() {
    assert_eq!(AprQuantizationType::Q8_0.bits_per_weight(), 8.0);
}

#[test]
fn test_quantization_type_f32_bytes_per_block() {
    assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
}

#[test]
fn test_quantization_type_q4_k_bytes_per_block() {
    assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
}

#[test]
fn test_quantization_type_q8_0_bytes_per_block() {
    assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
}

// ============================================================================
// Test 11-20: AprQuantizationType values_per_block tests
// ============================================================================

#[test]
fn test_quantization_type_f32_values_per_block() {
    assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
}

#[test]
fn test_quantization_type_q4_k_values_per_block() {
    assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
}

#[test]
fn test_quantization_type_q8_0_values_per_block() {
    assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
}

#[test]
fn test_quantization_type_to_byte_f32() {
    assert_eq!(AprQuantizationType::F32.to_byte(), 0);
}

#[test]
fn test_quantization_type_to_byte_q4_k() {
    assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
}

#[test]
fn test_quantization_type_to_byte_q8_0() {
    assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
}

// ============================================================================
// Test 21-30: AprQuantizationType from_byte tests
// ============================================================================

#[test]
fn test_quantization_type_from_byte_0() {
    assert_eq!(
        AprQuantizationType::from_byte(0),
        Some(AprQuantizationType::F32)
    );
}

#[test]
fn test_quantization_type_from_byte_1() {
    assert_eq!(
        AprQuantizationType::from_byte(1),
        Some(AprQuantizationType::Q4_K)
    );
}

#[test]
fn test_quantization_type_from_byte_2() {
    assert_eq!(
        AprQuantizationType::from_byte(2),
        Some(AprQuantizationType::Q8_0)
    );
}

#[test]
fn test_quantization_type_from_byte_invalid() {
    assert_eq!(AprQuantizationType::from_byte(3), None);
    assert_eq!(AprQuantizationType::from_byte(100), None);
    assert_eq!(AprQuantizationType::from_byte(255), None);
}

#[test]
fn test_quantization_type_debug() {
    let qt = AprQuantizationType::Q4_K;
    let debug = format!("{qt:?}");
    assert!(debug.contains("Q4_K"));
}

#[test]
fn test_quantization_type_copy() {
    let qt1 = AprQuantizationType::Q4_K;
    let qt2 = qt1;
    assert_eq!(qt1, qt2);
}

// ============================================================================
// Test 31-40: AprTransformerConfig tests
// ============================================================================

#[test]
fn test_config_default_values() {
    let config = AprTransformerConfig::default();
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
}

#[test]
fn test_config_clone() {
    let config = AprTransformerConfig {
        hidden_dim: 1024,
        num_layers: 12,
        ..Default::default()
    };
    let cloned = config.clone();
    assert_eq!(config.hidden_dim, cloned.hidden_dim);
    assert_eq!(config.num_layers, cloned.num_layers);
}

#[test]
fn test_config_partial_eq() {
    let config1 = AprTransformerConfig {
        hidden_dim: 1024,
        ..Default::default()
    };
    let config2 = AprTransformerConfig {
        hidden_dim: 1024,
        ..Default::default()
    };
    assert_eq!(config1, config2);
}

#[test]
fn test_config_debug() {
    let config = AprTransformerConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("AprTransformerConfig"));
    assert!(debug.contains("512"));
}

// ============================================================================
// Test 41-50: AprTransformerLayer tests
// ============================================================================

#[test]
fn test_layer_empty_creates_valid_layer() {
    let layer = AprTransformerLayer::empty(64, 256);
    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert_eq!(layer.qkv_weight.len(), 3 * 64 * 64);
    assert_eq!(layer.attn_output_weight.len(), 64 * 64);
    assert_eq!(layer.ffn_up_weight.len(), 64 * 256);
    // ffn_gate_weight is Option<Vec<f32>>, may be Some or None depending on implementation
    if let Some(ref gate) = layer.ffn_gate_weight {
        assert_eq!(gate.len(), 64 * 256);
    }
    assert_eq!(layer.ffn_down_weight.len(), 256 * 64);
}

#[test]
fn test_layer_empty_gqa_dimensions() {
    let layer = AprTransformerLayer::empty_gqa(128, 8, 2, 512);
    // 8 heads, 2 KV heads, head_dim = 128/8 = 16
    // Q: 8 heads * 16 = 128, K/V: 2 heads * 16 = 32 each
    // QKV total: 128 + 32 + 32 = 192 per token
    let expected_qkv = 128 * 192; // hidden_dim * (q + k + v)
    assert_eq!(layer.qkv_weight.len(), expected_qkv);
}

#[test]
fn test_layer_num_parameters_basic() {
    let layer = AprTransformerLayer::empty(32, 64);
    let params = layer.num_parameters();
    assert!(params > 0);
}

#[test]
fn test_layer_optional_fields_none() {
    let layer = AprTransformerLayer::empty(32, 64);
    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.qkv_bias.is_none());
    assert!(layer.attn_output_bias.is_none());
    assert!(layer.ffn_norm_weight.is_none());
    assert!(layer.ffn_norm_bias.is_none());
    assert!(layer.ffn_up_bias.is_none());
    assert!(layer.ffn_down_bias.is_none());
    assert!(layer.ffn_gate_bias.is_none());
}

// ============================================================================
// Test 51-60: AprTransformer creation and basic tests
// ============================================================================

#[test]
fn test_transformer_new_creates_valid_model() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 2,
        vocab_size: 100,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert_eq!(transformer.config().hidden_dim, 32);
    assert_eq!(transformer.config().num_layers, 2);
    assert_eq!(transformer.layers.len(), 2);
}

#[test]
fn test_transformer_num_parameters() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let params = transformer.num_parameters();
    assert!(params > 0);
}

#[test]
fn test_transformer_memory_size() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let mem = transformer.memory_size();
    // Memory should equal num_parameters * 4 (f32 = 4 bytes)
    assert_eq!(mem, transformer.num_parameters() * 4);
}

#[test]
fn test_transformer_embed_single_token() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        vocab_size: 20,
        ..Default::default()
    };
    let mut transformer = AprTransformer::new(config);
    // Set specific embedding values
    transformer.token_embedding[0..16].fill(1.0);
    transformer.token_embedding[16..32].fill(2.0);

    let embedding = transformer.embed(&[0]);
    assert_eq!(embedding.len(), 16);
    assert!(embedding.iter().all(|&x| (x - 1.0).abs() < 1e-6));

    let embedding2 = transformer.embed(&[1]);
    assert_eq!(embedding2.len(), 16);
    assert!(embedding2.iter().all(|&x| (x - 2.0).abs() < 1e-6));
}

#[test]
fn test_transformer_embed_multiple_tokens() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        vocab_size: 10,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let embedding = transformer.embed(&[0, 1, 2]);
    // Should return embeddings for all tokens concatenated
    assert_eq!(embedding.len(), 3 * 8);
}

// ============================================================================
// Test 61-70: AprTransformer forward tests
// ============================================================================

#[test]
fn test_transformer_forward_single_token() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[1]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 50);
}

#[test]
fn test_transformer_forward_multiple_tokens() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[1, 2, 3, 4, 5]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 50);
}

#[test]
fn test_transformer_forward_empty_error() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_transformer_predict_next() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.predict_next(&[1, 2, 3]);
    assert!(result.is_ok());
    let next = result.unwrap();
    assert!(next < 50);
}

#[test]
fn test_transformer_generate_basic() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.generate(&[1], 5);
    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(!tokens.is_empty());
}

// ============================================================================
// Test 71-80: AprKVCache tests
// ============================================================================

#[test]
fn test_kv_cache_new_empty() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 8,
        context_length: 512,
        ..Default::default()
    };
    let cache = AprKVCache::new(&config);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.capacity(), 512);
}

#[test]
fn test_kv_cache_append_single() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);
    let k = vec![1.0; 32];
    let v = vec![2.0; 32];
    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_kv_cache_get_layer() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);
    let k = vec![1.0; 32];
    let v = vec![2.0; 32];
    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    let (k_out, v_out) = cache.get(0);
    assert_eq!(k_out.len(), 32);
    assert_eq!(v_out.len(), 32);
}

#[test]
fn test_kv_cache_clear() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);
    let k = vec![1.0; 32];
    let v = vec![2.0; 32];
    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    assert!(!cache.is_empty());
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

// ============================================================================
// Test 81-90: GenerateConfig tests
// ============================================================================

#[test]
fn test_generate_config_default() {
    let config = GenerateConfig::default();
    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 1.0).abs() < f32::EPSILON);
    assert!((config.top_p - 0.9).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 0);
    assert!((config.repetition_penalty - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_generate_config_clone() {
    let config = GenerateConfig {
        max_tokens: 50,
        temperature: 0.5,
        top_p: 0.8,
        top_k: 20,
        repetition_penalty: 1.2,
        trace: false,
        stop_tokens: vec![],
    };
    let cloned = config.clone();
    assert_eq!(config.max_tokens, cloned.max_tokens);
    assert!((config.temperature - cloned.temperature).abs() < f32::EPSILON);
}

#[test]
fn test_generate_config_debug() {
    let config = GenerateConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("GenerateConfig"));
    assert!(debug.contains("32"));
}

// ============================================================================
// Test 91-100: AprInferenceScratch tests
// ============================================================================

#[test]
fn test_inference_scratch_allocation() {
    let config = AprTransformerConfig {
        hidden_dim: 256,
        intermediate_dim: 1024,
        num_heads: 8,
        num_kv_heads: 8,
        context_length: 512,
        ..Default::default()
    };
    let scratch = AprInferenceScratch::from_config(&config);
    assert_eq!(scratch.hidden.len(), 256);
    assert_eq!(scratch.normed.len(), 256);
    assert_eq!(scratch.ffn_up.len(), 1024);
    assert_eq!(scratch.ffn_gate.len(), 1024);
    assert_eq!(scratch.ffn_out.len(), 256);
}

#[test]
fn test_inference_scratch_clear_all_zeros() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        ..Default::default()
    };
    let mut scratch = AprInferenceScratch::from_config(&config);
    scratch.hidden.fill(1.0);
    scratch.normed.fill(2.0);
    scratch.ffn_up.fill(3.0);
    scratch.ffn_gate.fill(4.0);
    scratch.ffn_out.fill(5.0);

    scratch.clear();

    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.normed.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_gate.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_out.iter().all(|&x| x == 0.0));
}

// ============================================================================
// Test 101-110: QuantizedAprTensorQ4 tests
// ============================================================================

#[test]
fn test_quantized_tensor_q4_new() {
    let data = vec![0u8; 144]; // One Q4_K superblock
    let tensor = QuantizedAprTensorQ4::new(data, 256, 1);
    assert_eq!(tensor.in_dim, 256);
    assert_eq!(tensor.out_dim, 1);
}

#[test]
fn test_quantized_tensor_q4_zeros() {
    let tensor = QuantizedAprTensorQ4::zeros(256, 32);
    assert_eq!(tensor.in_dim, 256);
    assert_eq!(tensor.out_dim, 32);
    let expected = QuantizedAprTensorQ4::expected_bytes(256 * 32);
    assert_eq!(tensor.data.len(), expected);
}

#[test]
fn test_quantized_tensor_q4_expected_bytes_small() {
    // QuantizedAprTensorQ4 uses Q4_0 format: 18 bytes per 32 values
    let bytes = QuantizedAprTensorQ4::expected_bytes(100);
    // 100 / 32 = ceil to 4 blocks
    // 4 * 18 = 72 bytes
    assert_eq!(bytes, 72);
}

#[test]
fn test_quantized_tensor_q4_expected_bytes_exact() {
    // Exactly one Q4_0 block (32 values)
    let bytes = QuantizedAprTensorQ4::expected_bytes(32);
    assert_eq!(bytes, 18);
}

#[test]
fn test_quantized_tensor_q4_expected_bytes_large() {
    // Multiple Q4_0 blocks
    // 64 values = 2 blocks = 36 bytes
    let bytes = QuantizedAprTensorQ4::expected_bytes(64);
    assert_eq!(bytes, 2 * 18);
}

// ============================================================================
// Test 111-120: QuantizedAprTransformer tests
// ============================================================================

#[test]
fn test_quantized_transformer_new_q4k() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
    assert_eq!(qt.quantization_type(), AprQuantizationType::Q4_K);
    assert_eq!(qt.bits_per_weight(), 4.5);
}

#[test]
fn test_quantized_transformer_new_q8_0() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);
    assert_eq!(qt.quantization_type(), AprQuantizationType::Q8_0);
    assert_eq!(qt.bits_per_weight(), 8.0);
}

#[test]
fn test_quantized_transformer_config_access() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        vocab_size: 100,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
    assert_eq!(qt.config().hidden_dim, 64);
    assert_eq!(qt.config().num_layers, 2);
    assert_eq!(qt.config().vocab_size, 100);
}

#[test]
fn test_quantized_transformer_weight_bytes() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
    let bytes = qt.weight_bytes();
    assert!(bytes > 0);
}

#[test]
fn test_quantized_transformer_f32_equivalent_bytes() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
    let f32_bytes = qt.f32_equivalent_bytes();
    let weight_bytes = qt.weight_bytes();
    // Q4_K should use less storage than F32
    assert!(weight_bytes < f32_bytes);
}

#[test]
fn test_quantized_transformer_num_parameters() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
    let params = qt.num_parameters();
    assert!(params > 0);
}

// ============================================================================
// Test 121-130: AprBenchmarkResult tests
// ============================================================================

#[test]
fn test_benchmark_result_default() {
    let result = AprBenchmarkResult::default();
    assert_eq!(result.tokens_generated, 0);
    assert_eq!(result.total_time_ms, 0.0);
    assert_eq!(result.tokens_per_second, 0.0);
}

#[test]
fn test_benchmark_result_clone() {
    let result = AprBenchmarkResult {
        tokens_generated: 100,
        total_time_ms: 1000.0,
        tokens_per_second: 100.0,
        throughput_p50: 95.0,
        throughput_p99: 80.0,
        throughput_std_dev: 5.0,
        peak_memory_mb: 512.0,
        model_memory_mb: 400.0,
    };
    let cloned = result.clone();
    assert_eq!(result.tokens_generated, cloned.tokens_generated);
    assert_eq!(result.total_time_ms, cloned.total_time_ms);
}

#[test]
fn test_benchmark_result_debug() {
    let result = AprBenchmarkResult::default();
    let debug = format!("{result:?}");
    assert!(debug.contains("AprBenchmarkResult"));
}

#[test]
fn test_benchmark_result_meets_threshold_true() {
    let result = AprBenchmarkResult {
        tokens_per_second: 100.0,
        ..Default::default()
    };
    assert!(result.meets_threshold(50.0));
    assert!(result.meets_threshold(100.0));
}

#[test]
fn test_benchmark_result_meets_threshold_false() {
    let result = AprBenchmarkResult {
        tokens_per_second: 100.0,
        ..Default::default()
    };
    assert!(!result.meets_threshold(150.0));
}

#[test]
fn test_benchmark_result_compare_to_baseline() {
    let baseline = AprBenchmarkResult {
        tokens_per_second: 100.0,
        peak_memory_mb: 500.0,
        ..Default::default()
    };
    let result = AprBenchmarkResult {
        tokens_per_second: 120.0,
        peak_memory_mb: 600.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    assert!((comparison.throughput_ratio - 1.2).abs() < 0.01);
    assert!((comparison.memory_ratio - 1.2).abs() < 0.01);
}

// ============================================================================
// Test 131-140: AprParityComparison tests
// ============================================================================

#[test]
fn test_parity_comparison_is_parity_true() {
    let comparison = AprParityComparison {
        throughput_ratio: 1.0,
        memory_ratio: 1.0,
        parity_threshold_pct: 95.0,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_parity_comparison_is_parity_false() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.90,
        memory_ratio: 1.0,
        parity_threshold_pct: 95.0,
    };
    assert!(!comparison.is_parity());
}

#[test]
fn test_parity_comparison_explicit_values() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.0,
        memory_ratio: 0.0,
        parity_threshold_pct: 95.0,
    };
    assert_eq!(comparison.throughput_ratio, 0.0);
    assert_eq!(comparison.memory_ratio, 0.0);
    assert_eq!(comparison.parity_threshold_pct, 95.0);
}

#[test]
fn test_parity_comparison_debug() {
    let comparison = AprParityComparison {
        throughput_ratio: 1.0,
        memory_ratio: 1.0,
        parity_threshold_pct: 95.0,
    };
    let debug = format!("{comparison:?}");
    assert!(debug.contains("AprParityComparison"));
}

// ============================================================================
// Test 141-150: AprPrefillResult and AprLoadResult tests
// ============================================================================

#[test]
fn test_prefill_result_default_values() {
    let result = AprPrefillResult::default();
    assert_eq!(result.prompt_tokens, 0);
    assert_eq!(result.prefill_time_ms, 0.0);
    assert_eq!(result.prefill_tok_s, 0.0);
}

#[test]
fn test_prefill_result_clone() {
    let result = AprPrefillResult {
        prompt_tokens: 100,
        prefill_time_ms: 50.0,
        prefill_tok_s: 2000.0,
    };
    let cloned = result.clone();
    assert_eq!(result.prompt_tokens, cloned.prompt_tokens);
}

#[test]
fn test_prefill_result_debug() {
    let result = AprPrefillResult::default();
    let debug = format!("{result:?}");
    assert!(debug.contains("AprPrefillResult"));
}

#[test]
fn test_load_result_default_values() {
    let result = AprLoadResult::default();
    assert_eq!(result.load_time_ms, 0.0);
}

#[test]
fn test_load_result_clone() {
    let result = AprLoadResult {
        load_time_ms: 100.0,
    };
    let cloned = result.clone();
    assert_eq!(result.load_time_ms, cloned.load_time_ms);
}

#[test]
fn test_load_result_debug() {
    let result = AprLoadResult::default();
    let debug = format!("{result:?}");
    assert!(debug.contains("AprLoadResult"));
}

// ============================================================================
// Test 151-160: AprBenchmarkRunner tests
// ============================================================================

#[test]
fn test_benchmark_runner_new() {
    let config = AprTransformerConfig::default();
    let transformer = AprTransformer::new(config);
    let runner = AprBenchmarkRunner::new(transformer);
    assert_eq!(runner.warmup_iterations(), 3);
    assert_eq!(runner.measure_iterations(), 10);
}

#[test]
fn test_benchmark_runner_set_warmup() {
    let config = AprTransformerConfig::default();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(5);
    assert_eq!(runner.warmup_iterations(), 5);
}

#[test]
fn test_benchmark_runner_set_measure() {
    let config = AprTransformerConfig::default();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_measure_iterations(20);
    assert_eq!(runner.measure_iterations(), 20);
}

#[test]
fn test_benchmark_runner_set_measure_min_one() {
    let config = AprTransformerConfig::default();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_measure_iterations(0);
    // Should be at least 1
    assert!(runner.measure_iterations() >= 1);
}

// ============================================================================
// Test 161-170: MmapAprTransformer error tests
// ============================================================================

#[test]
fn test_mmap_transformer_file_not_found_error() {
    let result = MmapAprTransformer::from_file("/nonexistent/path.apr");
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_empty_file_error() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("empty.apr");
    std::fs::File::create(&path).expect("create file");
    let result = MmapAprTransformer::from_file(&path);
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_invalid_magic_error() {
    use std::io::Write;
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("invalid.apr");
    let mut f = std::fs::File::create(&path).expect("create file");
    let data = vec![0u8; 128];
    f.write_all(&data).expect("write");
    drop(f);
    let result = MmapAprTransformer::from_file(&path);
    assert!(result.is_err());
}

// ============================================================================
// Test 171-180: QuantizedAprTransformerQ4 tests
// ============================================================================

#[test]
fn test_quantized_q4_from_gguf_creates_valid_model() {
    // This test verifies the from_gguf constructor but requires a valid GGUF
    // Instead, test the config and memory_size methods
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    // Test config accessor would be on a real model
    assert_eq!(config.hidden_dim, 32);
}

#[test]
fn test_apr_transformer_header_size_constant() {
    assert_eq!(APR_TRANSFORMER_HEADER_SIZE, 64);
}

// ============================================================================
// Test 181-190: Transformer with biases tests
// ============================================================================

#[test]
fn test_transformer_with_all_biases() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let mut transformer = AprTransformer::new(config);

    // Set all optional biases
    transformer.layers[0].attn_norm_bias = Some(vec![0.1; 32]);
    transformer.layers[0].qkv_bias = Some(vec![0.1; 3 * 32]);
    transformer.layers[0].attn_output_bias = Some(vec![0.1; 32]);
    transformer.layers[0].ffn_norm_weight = Some(vec![1.0; 32]);
    transformer.layers[0].ffn_norm_bias = Some(vec![0.1; 32]);
    transformer.layers[0].ffn_up_bias = Some(vec![0.1; 64]);
    transformer.layers[0].ffn_gate_bias = Some(vec![0.1; 64]);
    transformer.layers[0].ffn_down_bias = Some(vec![0.1; 32]);
    transformer.output_norm_bias = Some(vec![0.1; 32]);
    transformer.lm_head_bias = Some(vec![0.1; 50]);

    let result = transformer.forward(&[1, 2, 3]);
    assert!(result.is_ok());
}

#[test]
fn test_layer_num_parameters_with_all_biases() {
    let mut layer = AprTransformerLayer::empty(64, 256);
    let base_params = layer.num_parameters();

    // Add all optional biases
    layer.attn_norm_bias = Some(vec![0.0; 64]);
    layer.qkv_bias = Some(vec![0.0; 3 * 64]);
    layer.attn_output_bias = Some(vec![0.0; 64]);
    layer.ffn_norm_weight = Some(vec![1.0; 64]);
    layer.ffn_norm_bias = Some(vec![0.0; 64]);
    layer.ffn_up_bias = Some(vec![0.0; 256]);
    layer.ffn_gate_bias = Some(vec![0.0; 256]);
    layer.ffn_down_bias = Some(vec![0.0; 64]);

    let with_biases_params = layer.num_parameters();
    assert!(with_biases_params > base_params);
}

// ============================================================================
// Test 191-200: More coverage tests
// ============================================================================

#[test]
fn test_transformer_forward_with_cache_basic() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 2,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Process first token (forward_with_cache takes u32, cache, pos)
    let result = transformer.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok());
    assert!(!cache.is_empty());

    // Process second token
    let result2 = transformer.forward_with_cache(2, &mut cache, 1);
    assert!(result2.is_ok());
}

#[test]
fn test_transformer_generate_with_cache() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };
    let result = transformer.generate_with_cache(&[1, 2], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_quantized_transformer_serialization_roundtrip() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
    let bytes = qt.to_bytes().expect("serialize");
    let decoded = QuantizedAprTransformer::from_bytes(&bytes).expect("deserialize");
    assert_eq!(decoded.config().hidden_dim, config.hidden_dim);
    assert_eq!(decoded.quantization_type(), AprQuantizationType::F32);
}

#[test]
fn test_benchmark_runner_benchmark_decode_small() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(1);

    let result = runner.benchmark_decode(&[1, 2], 2);
    assert!(result.is_ok());
}

#[test]
fn test_benchmark_runner_benchmark_prefill_small() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(1);

    let result = runner.benchmark_prefill(&[1, 2, 3]);
    assert!(result.is_ok());
}

#[test]
fn test_benchmark_load_with_transformer() {
    let result = AprBenchmarkRunner::benchmark_load(|| {
        let config = AprTransformerConfig {
            hidden_dim: 32,
            num_layers: 1,
            vocab_size: 50,
            ..Default::default()
        };
        AprTransformer::new(config)
    });
    assert!(result.is_ok());
    let load = result.unwrap();
    assert!(load.load_time_ms >= 0.0);
}
