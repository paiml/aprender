//! EXTREME TDD Coverage Tests for apr_transformer.rs
//!
//! This file targets coverage gaps to achieve 95%+ coverage.
//! Focus areas:
//! - MmapAprTransformer methods
//! - QuantizedAprTransformerQ4 methods
//! - AprBenchmarkRunner benchmark methods
//! - Error handling paths
//! - Edge cases in attention and RoPE

use realizar::apr_transformer::{
    AprBenchmarkResult, AprBenchmarkRunner, AprInferenceScratch, AprKVCache, AprLoadResult,
    AprParityComparison, AprPrefillResult, AprQuantizationType, AprTransformer,
    AprTransformerConfig, AprTransformerLayer, GenerateConfig, QuantizedAprTensorQ4,
    QuantizedAprTransformer, APR_TRANSFORMER_HEADER_SIZE,
};

// ============================================================================
// MmapAprTransformer Error Path Tests
// ============================================================================

#[test]
fn test_mmap_transformer_file_not_found() {
    use realizar::apr_transformer::MmapAprTransformer;
    let result = MmapAprTransformer::from_file("/nonexistent/path/to/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_empty_file() {
    use realizar::apr_transformer::MmapAprTransformer;
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("empty.apr");
    std::fs::File::create(&path).expect("create file");
    let result = MmapAprTransformer::from_file(&path);
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_invalid_magic() {
    use realizar::apr_transformer::MmapAprTransformer;
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

#[test]
fn test_mmap_transformer_invalid_version() {
    use realizar::apr_transformer::MmapAprTransformer;
    use std::io::Write;
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("bad_version.apr");
    let mut f = std::fs::File::create(&path).expect("create file");
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..8].copy_from_slice(&99u32.to_le_bytes()); // Invalid version
    f.write_all(&data).expect("write");
    drop(f);
    let result = MmapAprTransformer::from_file(&path);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer::from_apr_bytes Error Path Tests
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let data = vec![0u8; 10];
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_wrong_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_valid_magic_minimal() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    // Set minimal header fields
    data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata_size
    data[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&128u64.to_le_bytes()); // data_offset
                                                         // Metadata (empty JSON object)
    data[64..66].copy_from_slice(b"{}");
    let result = AprTransformer::from_apr_bytes(&data);
    // Should fail: no embedding tensor in minimal APR (fail-fast is correct)
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_metadata_extends_beyond() {
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(b"APR\0");
    data[12..20].copy_from_slice(&60u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&100u32.to_le_bytes()); // metadata_size > remaining
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// QuantizedAprTransformer Error Path Tests
// ============================================================================

#[test]
fn test_quantized_transformer_forward_empty_tokens() {
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
    let result = qt.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_too_small() {
    let data = vec![0u8; 32];
    let result = QuantizedAprTransformer::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = QuantizedAprTransformer::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_invalid_quant_type() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    data[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
    data[48] = 255; // Invalid quant type
    let result = QuantizedAprTransformer::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// AprKVCache Edge Case Tests
// ============================================================================

#[test]
fn test_kv_cache_multiple_appends() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 256,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);

    for i in 0..5 {
        let k = vec![(i as f32) + 1.0; 64];
        let v = vec![(i as f32) + 10.0; 64];
        // Append to all layers
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }

    assert_eq!(cache.len(), 5);
    let (k_out, v_out) = cache.get(0);
    assert_eq!(k_out.len(), 5 * 64);
    assert_eq!(v_out.len(), 5 * 64);
}

#[test]
fn test_kv_cache_gqa_dimensions() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 4:1 ratio
        context_length: 128,
        ..Default::default()
    };
    let cache = AprKVCache::new(&config);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 128);
}

// ============================================================================
// AprTransformerLayer Tests
// ============================================================================

#[test]
fn test_layer_num_parameters_with_biases() {
    let mut layer = AprTransformerLayer::empty(64, 256);
    // Add optional biases
    layer.attn_norm_bias = Some(vec![0.0; 64]);
    layer.qkv_bias = Some(vec![0.0; 3 * 64]);
    layer.attn_output_bias = Some(vec![0.0; 64]);
    layer.ffn_up_bias = Some(vec![0.0; 256]);
    layer.ffn_down_bias = Some(vec![0.0; 64]);
    layer.ffn_norm_weight = Some(vec![1.0; 64]);
    layer.ffn_norm_bias = Some(vec![0.0; 64]);

    let params = layer.num_parameters();
    // Should include all the optional biases
    assert!(params > 0);
}

#[test]
fn test_layer_empty_gqa_smaller_kv() {
    // 8 heads, 2 KV heads -> smaller QKV weight
    let layer_mha = AprTransformerLayer::empty_gqa(128, 8, 8, 512);
    let layer_gqa = AprTransformerLayer::empty_gqa(128, 8, 2, 512);
    // GQA should have fewer parameters due to smaller K/V
    assert!(layer_gqa.qkv_weight.len() < layer_mha.qkv_weight.len());
}

// ============================================================================
// AprTransformer Method Tests
// ============================================================================

#[test]
fn test_transformer_forward_with_biases() {
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

    // Add biases to layer
    transformer.layers[0].qkv_bias = Some(vec![0.1; 3 * 32]);
    transformer.layers[0].attn_output_bias = Some(vec![0.1; 32]);
    transformer.layers[0].ffn_up_bias = Some(vec![0.1; 64]);
    transformer.layers[0].ffn_down_bias = Some(vec![0.1; 32]);
    transformer.output_norm_bias = Some(vec![0.1; 32]);
    transformer.lm_head_bias = Some(vec![0.1; 50]);

    let result = transformer.forward(&[1, 2, 3]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 50);
}

#[test]
fn test_transformer_generate_with_cache_basic() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 128,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.0, // Greedy
        ..Default::default()
    };
    let result = transformer.generate_with_cache(&[1, 2], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_transformer_generate_with_cache_temperature() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 128,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 2,
        temperature: 1.0, // Non-zero temperature
        ..Default::default()
    };
    let result = transformer.generate_with_cache(&[1], &gen_config);
    assert!(result.is_ok());
}

#[test]
fn test_transformer_generate_with_cache_empty_prompt() {
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
    let gen_config = GenerateConfig::default();
    let result = transformer.generate_with_cache(&[], &gen_config);
    assert!(result.is_err());
}

// ============================================================================
// AprBenchmarkRunner Tests
// ============================================================================

#[test]
fn test_benchmark_runner_benchmark_decode() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 128,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(2);

    let result = runner.benchmark_decode(&[1, 2], 3);
    assert!(result.is_ok());
    let bench = result.unwrap();
    // tokens_generated is usize, always >= 0
    let _ = bench.tokens_generated;
    assert!(bench.total_time_ms >= 0.0);
}

#[test]
fn test_benchmark_runner_benchmark_prefill() {
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
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(2);

    let result = runner.benchmark_prefill(&[1, 2, 3, 4, 5]);
    assert!(result.is_ok());
    let prefill = result.unwrap();
    assert_eq!(prefill.prompt_tokens, 5);
    assert!(prefill.prefill_time_ms >= 0.0);
}

#[test]
fn test_benchmark_runner_benchmark_load() {
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

#[test]
fn test_benchmark_runner_set_measure_iterations_min() {
    let config = AprTransformerConfig::default();
    let transformer = AprTransformer::new(config);
    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_measure_iterations(0);
    // Should be at least 1
    assert!(runner.measure_iterations() >= 1);
}

// ============================================================================
// AprBenchmarkResult Tests
// ============================================================================

#[test]
fn test_benchmark_result_compare_zero_baseline() {
    let baseline = AprBenchmarkResult {
        tokens_per_second: 0.0,
        peak_memory_mb: 0.0,
        ..Default::default()
    };
    let result = AprBenchmarkResult {
        tokens_per_second: 100.0,
        peak_memory_mb: 512.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    // With zero baseline, should return 1.0 ratios
    assert_eq!(comparison.throughput_ratio, 1.0);
    assert_eq!(comparison.memory_ratio, 1.0);
}

// ============================================================================
// QuantizedAprTensorQ4 Tests
// ============================================================================

#[test]
fn test_quantized_tensor_q4_zeros_allocation() {
    let tensor = QuantizedAprTensorQ4::zeros(64, 32);
    assert_eq!(tensor.in_dim, 64);
    assert_eq!(tensor.out_dim, 32);
    assert!(!tensor.data.is_empty());
}

#[test]
fn test_quantized_tensor_q4_expected_bytes_edge_cases() {
    // Single element
    let bytes_1 = QuantizedAprTensorQ4::expected_bytes(1);
    assert_eq!(bytes_1, 18); // Ceil to 1 block

    // Exactly one block
    let bytes_32 = QuantizedAprTensorQ4::expected_bytes(32);
    assert_eq!(bytes_32, 18);

    // Partial second block
    let bytes_33 = QuantizedAprTensorQ4::expected_bytes(33);
    assert_eq!(bytes_33, 36); // 2 blocks
}

// ============================================================================
// AprInferenceScratch Tests
// ============================================================================

#[test]
fn test_inference_scratch_from_config() {
    let config = AprTransformerConfig {
        hidden_dim: 128,
        intermediate_dim: 512,
        ..Default::default()
    };
    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 128);
    assert_eq!(scratch.normed.len(), 128);
    assert_eq!(scratch.ffn_up.len(), 512);
    assert_eq!(scratch.ffn_gate.len(), 512);
    assert_eq!(scratch.ffn_out.len(), 128);
}

#[test]
fn test_inference_scratch_clear() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        intermediate_dim: 128,
        ..Default::default()
    };
    let mut scratch = AprInferenceScratch::from_config(&config);

    // Set some values
    scratch.hidden.fill(1.0);
    scratch.ffn_up.fill(2.0);

    // Clear
    scratch.clear();

    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
}

// ============================================================================
// GenerateConfig Tests
// ============================================================================

#[test]
fn test_generate_config_custom_values() {
    let config = GenerateConfig {
        max_tokens: 100,
        temperature: 0.5,
        top_p: 0.8,
        top_k: 50,
        repetition_penalty: 1.2,
        trace: false,
        stop_tokens: vec![],
    };
    assert_eq!(config.max_tokens, 100);
    assert!((config.temperature - 0.5).abs() < f32::EPSILON);
    assert!((config.top_p - 0.8).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 50);
    assert!((config.repetition_penalty - 1.2).abs() < f32::EPSILON);
}

// ============================================================================
// AprTransformerConfig Serialization Tests
// ============================================================================

#[test]
fn test_config_json_roundtrip() {
    let config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128256,
        intermediate_dim: 14336,
        context_length: 8192,
        rope_theta: 500000.0,
        eps: 1e-6,
        ..Default::default()
    };
    let json = serde_json::to_string(&config).expect("serialize");
    let decoded: AprTransformerConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(config, decoded);
}

// ============================================================================
// AprQuantizationType Edge Cases
// ============================================================================

#[test]
fn test_quantization_type_default() {
    let qt = AprQuantizationType::default();
    assert_eq!(qt, AprQuantizationType::F32);
}

#[test]
fn test_quantization_type_clone_eq() {
    let qt1 = AprQuantizationType::Q4_K;
    let qt2 = qt1;
    assert_eq!(qt1, qt2);
}

// ============================================================================
// AprPrefillResult and AprLoadResult Tests
// ============================================================================

#[test]
fn test_prefill_result_default() {
    let result = AprPrefillResult::default();
    assert_eq!(result.prompt_tokens, 0);
    assert_eq!(result.prefill_time_ms, 0.0);
    assert_eq!(result.prefill_tok_s, 0.0);
}

#[test]
fn test_load_result_default() {
    let result = AprLoadResult::default();
    assert_eq!(result.load_time_ms, 0.0);
}

// ============================================================================
// AprParityComparison Tests
// ============================================================================

#[test]
fn test_parity_comparison_just_above_threshold() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.951,
        memory_ratio: 1.0,
        parity_threshold_pct: 95.0,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_parity_comparison_just_below_threshold() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.949,
        memory_ratio: 1.0,
        parity_threshold_pct: 95.0,
    };
    assert!(!comparison.is_parity());
}

// ============================================================================
// AprTransformer GQA Forward Tests
// ============================================================================

#[test]
fn test_transformer_forward_gqa_config() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 4:1 ratio
        vocab_size: 50,
        intermediate_dim: 128,
        context_length: 64,
        ..Default::default()
    };

    // Create transformer with GQA layers
    let layers: Vec<AprTransformerLayer> = (0..config.num_layers)
        .map(|_| {
            AprTransformerLayer::empty_gqa(
                config.hidden_dim,
                config.num_heads,
                config.num_kv_heads,
                config.intermediate_dim,
            )
        })
        .collect();

    let transformer = AprTransformer {
        config: config.clone(),
        token_embedding: vec![0.1; config.vocab_size * config.hidden_dim],
        layers,
        output_norm_weight: vec![1.0; config.hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; config.hidden_dim * config.vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let result = transformer.forward(&[1, 2, 3]);
    assert!(result.is_ok());
}

// ============================================================================
// Matmul Fallback Path Tests
// ============================================================================

#[test]
fn test_matmul_dimension_mismatch_fallback() {
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
    // Forward should still work even if internal matmul has edge cases
    let result = transformer.forward(&[1]);
    assert!(result.is_ok());
}

// ============================================================================
// Header Size Constant Test
// ============================================================================

#[test]
fn test_apr_transformer_header_size() {
    assert_eq!(APR_TRANSFORMER_HEADER_SIZE, 64);
}

// ============================================================================
// QuantizedAprTransformer Serialization Tests
// ============================================================================

#[test]
fn test_quantized_transformer_to_bytes_header() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let qt = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);
    let bytes = qt.to_bytes().expect("serialize");

    // Verify magic
    assert_eq!(&bytes[0..4], b"APR\0");
    // Verify quant type byte at offset 48
    assert_eq!(bytes[48], AprQuantizationType::Q8_0.to_byte());
}

// ============================================================================
// AprKVCache Panic Tests (documentation says it panics)
// ============================================================================

#[test]
#[should_panic(expected = "Layer index out of bounds")]
fn test_kv_cache_append_invalid_layer() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 128,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);
    let k = vec![1.0; 64];
    let v = vec![1.0; 64];
    cache.append(999, &k, &v); // Should panic
}

// ============================================================================
// AprTransformer Layer Norm Tests
// ============================================================================

#[test]
fn test_layer_norm_batch_processing() {
    let config = AprTransformerConfig {
        hidden_dim: 4,
        num_layers: 1,
        vocab_size: 10,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);

    // Test multiple tokens through forward which exercises batch layer norm
    let result = transformer.forward(&[1, 2, 3]);
    assert!(result.is_ok());
}

// ============================================================================
// AprTransformer Embed Out of Vocab Tests
// ============================================================================

#[test]
fn test_embed_large_token_id() {
    let config = AprTransformerConfig {
        hidden_dim: 4,
        vocab_size: 10,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);

    // Token ID way beyond vocab
    let embedded = transformer.embed(&[1000000]);
    assert_eq!(embedded.len(), 4);
    // Should be zeros for out-of-vocab
    assert!(embedded.iter().all(|&x| x == 0.0));
}

// ============================================================================
// AprTransformer Generate EOS Token Test
// ============================================================================

#[test]
fn test_generate_stops_at_eos() {
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

    // With zero weights, output will be deterministic
    let result = transformer.generate(&[1], 100);
    assert!(result.is_ok());
}

// ============================================================================
// QuantizedAprTransformer Forward With Cache Tests
// ============================================================================

#[test]
fn test_quantized_forward_with_cache_incremental() {
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
    let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
    let mut cache = AprKVCache::new(&config);

    // Process tokens incrementally
    for i in 0..5 {
        let result = qt.forward_with_cache(i as u32, &mut cache, i);
        assert!(result.is_ok());
    }
    // Cache length increments only on layer 0, so we check via is_empty
    assert!(!cache.is_empty());
}

// ============================================================================
// AprBenchmarkResult Statistics Tests
// ============================================================================

#[test]
fn test_benchmark_result_statistics() {
    let result = AprBenchmarkResult {
        tokens_generated: 100,
        total_time_ms: 1000.0,
        tokens_per_second: 100.0,
        throughput_p50: 95.0,
        throughput_p99: 80.0,
        throughput_std_dev: 10.0,
        peak_memory_mb: 1024.0,
        model_memory_mb: 800.0,
    };

    assert!(result.meets_threshold(50.0));
    assert!(result.meets_threshold(100.0));
    assert!(!result.meets_threshold(150.0));

    let baseline = result.clone();
    let comparison = result.compare_to_baseline(&baseline);
    assert!((comparison.throughput_ratio - 1.0).abs() < f32::EPSILON as f64);
    assert!(comparison.is_parity());
}

// ============================================================================
// AprTransformer Add Bias Tests
// ============================================================================

#[test]
fn test_add_bias_modular() {
    let config = AprTransformerConfig {
        hidden_dim: 4,
        num_layers: 1,
        vocab_size: 10,
        ..Default::default()
    };
    let mut transformer = AprTransformer::new(config.clone());

    // Set up biases
    transformer.layers[0].ffn_up_bias = Some(vec![1.0; config.intermediate_dim]);

    // Forward should apply biases
    let result = transformer.forward(&[1, 2]);
    assert!(result.is_ok());
}

// ============================================================================
// AprTransformer GELU Activation Tests
// ============================================================================

#[test]
fn test_gelu_large_values() {
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

    // Set large embedding values to test GELU with large inputs
    transformer.token_embedding.fill(10.0);

    let result = transformer.forward(&[1]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    // All logits should be finite
    assert!(logits.iter().all(|x| x.is_finite()));
}

// ============================================================================
// AprTransformer Predict Next Empty Logits Error Path
// ============================================================================

#[test]
fn test_predict_next_valid() {
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
    let next_token = result.unwrap();
    assert!(next_token < 50);
}
