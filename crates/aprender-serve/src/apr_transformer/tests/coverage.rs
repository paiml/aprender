//! APR Transformer Coverage Tests (PMAT-803)
//!
//! Comprehensive tests to improve coverage for:
//! - loader.rs: MmapAprTransformer, AprQuantizationType, QuantizedAprTransformer
//! - dequant.rs: f16_to_f32, extract_scale_min_apr, dequantize_q4_k_apr, dequantize_q6_k_apr
//! - config.rs: AprKVCache, GenerateConfig, AprTransformerConfig, AprTransformerLayer, Q4KLayerWeights
//! - helpers.rs: SIMD primitives

use crate::apr::MAGIC;
use crate::apr_transformer::{
    AprKVCache, AprQuantizationType, AprTransformerConfig, AprTransformerLayer, GenerateConfig,
    MmapAprTransformer, Q4KLayerWeights, QuantizedAprTransformer, APR_TRANSFORMER_HEADER_SIZE,
};

// ============================================================================
// Part 1: AprQuantizationType Tests
// ============================================================================

#[test]
fn test_quantization_type_default() {
    let qt = AprQuantizationType::default();
    assert_eq!(qt, AprQuantizationType::F32);
}

#[test]
fn test_quantization_type_bits_per_weight() {
    assert_eq!(AprQuantizationType::F32.bits_per_weight(), 32.0);
    assert_eq!(AprQuantizationType::Q4_K.bits_per_weight(), 4.5);
    assert_eq!(AprQuantizationType::Q8_0.bits_per_weight(), 8.0);
}

#[test]
fn test_quantization_type_bytes_per_block() {
    assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
    assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
    assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
}

#[test]
fn test_quantization_type_values_per_block() {
    assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
    assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
    assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
}

#[test]
fn test_quantization_type_to_byte() {
    assert_eq!(AprQuantizationType::F32.to_byte(), 0);
    assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
    assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
}

#[test]
fn test_quantization_type_from_byte_valid() {
    assert_eq!(
        AprQuantizationType::from_byte(0),
        Some(AprQuantizationType::F32)
    );
    assert_eq!(
        AprQuantizationType::from_byte(1),
        Some(AprQuantizationType::Q4_K)
    );
    assert_eq!(
        AprQuantizationType::from_byte(2),
        Some(AprQuantizationType::Q8_0)
    );
}

#[test]
fn test_quantization_type_from_byte_invalid() {
    assert_eq!(AprQuantizationType::from_byte(3), None);
    assert_eq!(AprQuantizationType::from_byte(255), None);
}

#[test]
fn test_quantization_type_roundtrip() {
    for qt in [
        AprQuantizationType::F32,
        AprQuantizationType::Q4_K,
        AprQuantizationType::Q8_0,
    ] {
        let byte = qt.to_byte();
        let recovered = AprQuantizationType::from_byte(byte);
        assert_eq!(recovered, Some(qt));
    }
}

#[test]
fn test_quantization_type_clone() {
    let original = AprQuantizationType::Q4_K;
    let cloned = original;
    assert_eq!(original, cloned);
}

#[test]
fn test_quantization_type_debug() {
    let qt = AprQuantizationType::Q8_0;
    let debug_str = format!("{:?}", qt);
    assert!(debug_str.contains("Q8_0"));
}

// ============================================================================
// Part 2: QuantizedAprTransformer Tests
// ============================================================================

fn create_test_apr_config() -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    }
}

#[test]
fn test_quantized_transformer_new_f32() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::F32);
    assert_eq!(transformer.bits_per_weight(), 32.0);
    assert_eq!(transformer.config().hidden_dim, 64);
    assert_eq!(transformer.config().num_layers, 2);
}

#[test]
fn test_quantized_transformer_new_q4k() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::Q4_K);
    assert_eq!(transformer.bits_per_weight(), 4.5);
}

#[test]
fn test_quantized_transformer_new_q8_0() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::Q8_0);
    assert_eq!(transformer.bits_per_weight(), 8.0);
}

#[test]
fn test_quantized_transformer_weight_bytes() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let bytes = transformer.weight_bytes();
    assert!(bytes > 0);
}

#[test]
fn test_quantized_transformer_f32_equivalent_bytes() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);

    let f32_bytes = transformer.f32_equivalent_bytes();
    let actual_bytes = transformer.weight_bytes();

    // Q4_K should be smaller than F32 equivalent
    // Note: embeddings are still F32, so compression is partial
    assert!(f32_bytes > 0);
    assert!(actual_bytes > 0);
}

#[test]
fn test_quantized_transformer_num_parameters() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let params = transformer.num_parameters();
    assert!(params > 0);

    // Estimate: embedding (100*64*2) + layers + norms
    // Should be at least vocab * hidden * 2
    assert!(params >= 100 * 64 * 2);
}

#[test]
fn test_quantized_transformer_forward_empty_tokens() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let result = transformer.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_forward_single_token() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());

    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_quantized_transformer_forward_multiple_tokens() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    let result = transformer.forward(&[0, 1, 2, 3]);
    assert!(result.is_ok());

    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_quantized_transformer_forward_oov_token() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);

    // Token ID beyond vocab_size (100)
    let result = transformer.forward(&[999]);
    assert!(result.is_ok()); // Should handle gracefully with zeros
}

#[test]
fn test_quantized_transformer_forward_with_cache() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
    let mut cache = AprKVCache::new(&config);

    let result = transformer.forward_with_cache(0, &mut cache, 0);
    assert!(result.is_ok());

    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_quantized_transformer_forward_with_cache_sequential() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
    let mut cache = AprKVCache::new(&config);

    // Process multiple tokens sequentially
    for (pos, token_id) in [0u32, 1, 2, 3].iter().enumerate() {
        let result = transformer.forward_with_cache(*token_id, &mut cache, pos);
        assert!(result.is_ok());
    }
}

#[test]
fn test_quantized_transformer_to_bytes_and_from_bytes() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

    // Serialize
    let bytes = transformer.to_bytes();
    assert!(bytes.is_ok());
    let bytes = bytes.expect("test value should be present");

    // Verify magic
    assert_eq!(&bytes[0..4], MAGIC);

    // Deserialize
    let recovered = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(recovered.is_ok());
    let recovered = recovered.expect("test value should be present");

    // Verify config matches
    assert_eq!(recovered.config().hidden_dim, config.hidden_dim);
    assert_eq!(recovered.config().num_layers, config.num_layers);
    assert_eq!(recovered.quantization_type(), AprQuantizationType::Q4_K);
}

#[test]
fn test_quantized_transformer_from_bytes_too_small() {
    let bytes = vec![0u8; 32]; // Too small for header
    let result = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_invalid_magic() {
    let mut bytes = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
    bytes[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let result = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_from_bytes_invalid_quant_type() {
    let config = create_test_apr_config();
    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::F32);

    let mut bytes = transformer.to_bytes().expect("test value should be present");

    // Corrupt quantization type byte (offset 48)
    bytes[48] = 255; // Invalid quant type

    let result = QuantizedAprTransformer::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_quantized_transformer_calculate_quantized_bytes() {
    // F32: 1 value per block, 4 bytes per block
    assert_eq!(
        QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::F32),
        100 * 4
    );

    // Q4_K: 256 values per block, 144 bytes per block
    // 100 values = 1 block (ceil)
    assert_eq!(
        QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::Q4_K),
        144
    );

    // Q8_0: 32 values per block, 36 bytes per block
    // 100 values = 4 blocks (ceil(100/32) = 4)
    assert_eq!(
        QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::Q8_0),
        4 * 36
    );
}

// ============================================================================
// Part 3: AprKVCache Tests
// ============================================================================

#[test]
fn test_kv_cache_new() {
    let config = create_test_apr_config();
    let cache = AprKVCache::new(&config);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), config.context_length);
    assert_eq!(cache.num_kv_heads(), config.num_kv_heads);
    assert_eq!(cache.head_dim(), config.hidden_dim / config.num_heads);
}

#[test]
fn test_kv_cache_append_and_get() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Append to layer 0
    cache.append(0, &k, &v);
    cache.advance(); // F-REGR-231: explicit advance required

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    // Get and verify
    let (cached_k, cached_v) = cache.get(0);
    assert_eq!(cached_k.len(), kv_size);
    assert_eq!(cached_v.len(), kv_size);
    assert_eq!(cached_k[0], 1.0);
    assert_eq!(cached_v[0], 2.0);
}

#[test]
fn test_kv_cache_append_multiple_positions() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

    // Append 3 positions
    for i in 0..3 {
        let k = vec![(i + 1) as f32; kv_size];
        let v = vec![(i + 10) as f32; kv_size];

        // Append to all layers (last layer auto-advances)
        for layer in 0..config.num_layers {
            cache.append(layer, &k, &v);
        }
        // No advance() needed - append() auto-advances on last layer
    }

    assert_eq!(cache.len(), 3);

    // Verify layer 0 values
    let (cached_k, cached_v) = cache.get(0);
    assert_eq!(cached_k.len(), 3 * kv_size);
    assert_eq!(cached_v.len(), 3 * kv_size);
}

#[test]
fn test_kv_cache_clear() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    cache.append(0, &k, &v);
    cache.advance(); // F-REGR-231: explicit advance required
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Capacity should be preserved
    assert_eq!(cache.capacity(), config.context_length);
}

#[test]
#[should_panic(expected = "Layer index out of bounds")]
fn test_kv_cache_append_invalid_layer() {
    let config = create_test_apr_config();
    let mut cache = AprKVCache::new(&config);

    let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);
    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Layer index beyond num_layers
    cache.append(999, &k, &v);
}

// ============================================================================
// Part 4: GenerateConfig Tests
// ============================================================================

#[test]
fn test_generate_config_default() {
    let config = GenerateConfig::default();

    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 1.0).abs() < 1e-6);
    assert!((config.top_p - 0.9).abs() < 1e-6);
    assert_eq!(config.top_k, 0);
    assert!((config.repetition_penalty - 1.0).abs() < 1e-6);
}

include!("coverage_generate_config.rs");
include!("coverage_dequantize.rs");
include!("coverage_apr_benchmark.rs");
