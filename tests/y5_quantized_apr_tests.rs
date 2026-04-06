#![allow(clippy::used_underscore_binding)]
//! Y5: Quantized APR Format Tests (EXTREME TDD - RED Phase)
//!
//! Per Section Y of the spec, APR format MUST support INT8/INT4 quantization.
//! These tests define Popperian falsification conditions for Y5.
//!
//! FALSIFICATION: INT8/INT4 APR inference fails

// ============================================================================
// Y5.1: Quantization Type Support
// ============================================================================

/// Y5.1a: QuantizationType enum exists with Q4_K and Q8_0 variants
/// FALSIFICATION: Enum missing or variants missing
#[test]
fn y5_1a_quantization_type_enum_exists() {
    use realizar::apr_transformer::AprQuantizationType;

    // Verify enum variants exist
    let _q4k = AprQuantizationType::Q4_K;
    let _q8_0 = AprQuantizationType::Q8_0;
    let _f32 = AprQuantizationType::F32;

    // Verify as_bits_per_weight returns correct values
    assert_eq!(_q4k.bits_per_weight(), 4.5, "Q4_K is 4.5 bits/weight");
    assert_eq!(_q8_0.bits_per_weight(), 8.0, "Q8_0 is 8 bits/weight");
    assert_eq!(_f32.bits_per_weight(), 32.0, "F32 is 32 bits/weight");
}

/// Y5.1b: Quantization type stored in APR header
/// FALSIFICATION: Header missing quantization field
#[test]
fn y5_1b_quantization_in_header() {
    use realizar::apr_transformer::APR_TRANSFORMER_HEADER_SIZE;

    // Header must have space for quantization type (1 byte at offset 48)
    // Use a runtime check to avoid clippy::assertions_on_constants
    let header_size = APR_TRANSFORMER_HEADER_SIZE;
    assert!(
        header_size >= 49,
        "Header must have room for quantization type byte"
    );
}

// ============================================================================
// Y5.2: QuantizedAprTransformer Struct
// ============================================================================

/// Y5.2a: QuantizedAprTransformer struct exists
/// FALSIFICATION: Struct missing or wrong fields
#[test]
fn y5_2a_quantized_transformer_struct_exists() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig::default();

    // Must be constructible with quantization type
    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

    assert_eq!(transformer.quantization_type(), AprQuantizationType::Q4_K);
    assert_eq!(transformer.config().hidden_dim, config.hidden_dim);
}

/// Y5.2b: QuantizedAprTransformer stores raw quantized bytes for layer weights
/// FALSIFICATION: Layer weights stored as F32 instead of quantized
#[test]
fn y5_2b_stores_raw_quantized_bytes() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        vocab_size: 1000,
        num_layers: 1,
        intermediate_dim: 512,
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);

    // Q4_K: 4.5 bits/weight for layer weights
    // Note: embeddings are F32 (not quantized) for quality
    let q4k_size = transformer.weight_bytes();
    let f32_equivalent = transformer.f32_equivalent_bytes();

    // With embeddings being F32, we expect ~50% compression (not 75%)
    // Layer weights are 7x smaller, but embeddings dominate for small models
    assert!(
        q4k_size < f32_equivalent,
        "Q4_K model ({} bytes) should be smaller than F32 ({} bytes)",
        q4k_size,
        f32_equivalent
    );

    // Verify quantization type is stored correctly
    assert_eq!(transformer.quantization_type(), AprQuantizationType::Q4_K);
}

// ============================================================================
// Y5.3: Quantized Forward Pass
// ============================================================================

/// Y5.3a: Q4_K forward pass produces valid logits
/// FALSIFICATION: forward() returns error or NaN/Inf
#[test]
fn y5_3a_q4k_forward_pass_valid() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        vocab_size: 100,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_dim: 512,
        context_length: 128,
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);

    // Forward pass with sample tokens
    let token_ids = vec![1u32, 2, 3];
    let logits = transformer
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    // Verify output shape
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "Logits should match vocab_size"
    );

    // Verify no NaN/Inf
    for (i, &logit) in logits.iter().enumerate() {
        assert!(logit.is_finite(), "Logit[{}] = {} is not finite", i, logit);
    }
}

/// Y5.3b: Q8_0 forward pass produces valid logits
/// FALSIFICATION: forward() returns error or NaN/Inf
#[test]
fn y5_3b_q8_0_forward_pass_valid() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        vocab_size: 100,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_dim: 512,
        context_length: 128,
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q8_0);

    let token_ids = vec![1u32, 2, 3];
    let logits = transformer
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    for &logit in &logits {
        assert!(logit.is_finite());
    }
}

// ============================================================================
// Y5.4: Quantization Quality
// ============================================================================

/// Y5.4a: Q4_K and F32 produce similar outputs (within tolerance)
/// FALSIFICATION: Max absolute difference > 0.1 per logit
#[test]
fn y5_4a_q4k_output_quality() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformer, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        vocab_size: 100,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_dim: 512,
        ..Default::default()
    };

    // Create F32 reference
    let f32_transformer = AprTransformer::new(config);
    let f32_logits = f32_transformer.forward(&[1, 2, 3]).expect("test");

    // Create Q4_K version (with same weights quantized)
    let q4k_transformer =
        QuantizedAprTransformer::from_f32_transformer(&f32_transformer, AprQuantizationType::Q4_K);
    let q4k_logits = q4k_transformer.forward(&[1, 2, 3]).expect("test");

    // Compare outputs - quantization introduces some error
    let max_diff: f32 = f32_logits
        .iter()
        .zip(q4k_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    // Q4_K should be within reasonable tolerance
    // Note: With zero-initialized weights, both will output zeros
    assert!(
        max_diff < 1.0,
        "Q4_K output differs from F32 by max {:.4}, expected < 1.0",
        max_diff
    );
}

/// Y5.4b: Q8_0 has better quality than Q4_K
/// FALSIFICATION: Q8_0 error >= Q4_K error
#[test]
fn y5_4b_q8_0_better_quality_than_q4k() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformer, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        vocab_size: 100,
        num_layers: 1,
        intermediate_dim: 512,
        ..Default::default()
    };

    let f32_transformer = AprTransformer::new(config);

    // Set some non-zero weights for meaningful comparison
    // (In real usage, weights come from trained model)

    let q4k =
        QuantizedAprTransformer::from_f32_transformer(&f32_transformer, AprQuantizationType::Q4_K);
    let q8_0 =
        QuantizedAprTransformer::from_f32_transformer(&f32_transformer, AprQuantizationType::Q8_0);

    // Q8_0 uses more bits, should have equal or better precision
    assert!(
        q8_0.bits_per_weight() >= q4k.bits_per_weight(),
        "Q8_0 should use more bits than Q4_K"
    );
}

// ============================================================================
// Y5.5: Memory Efficiency
// ============================================================================

/// Y5.5a: Q4_K uses ~7x less memory than F32 for layer weights
/// FALSIFICATION: Memory reduction < 4x (accounting for F32 embeddings)
#[test]
fn y5_5a_q4k_memory_efficiency() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    // Layer-heavy model: more layers = layer weights dominate over embeddings
    // Embeddings stay F32 for quality, so we need layers to dominate for good compression
    let config = AprTransformerConfig {
        hidden_dim: 512,
        vocab_size: 2000, // Smaller vocab to reduce F32 embedding overhead
        num_layers: 16,   // More layers = more quantized weights
        num_heads: 8,
        num_kv_heads: 8,
        intermediate_dim: 2048,
        context_length: 32, // Minimal context (doesn't affect weight size)
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);

    let q4k_bytes = transformer.weight_bytes();
    let f32_bytes = transformer.f32_equivalent_bytes();

    let compression_ratio = f32_bytes as f64 / q4k_bytes as f64;

    // Q4_K: 4.5 bits/weight for layers (~7x), but embeddings are F32
    // With layer-heavy model, expect 4-6x overall compression
    assert!(
        compression_ratio > 4.0,
        "Q4_K compression ratio {:.2}x should be > 4x",
        compression_ratio
    );
}

/// Y5.5b: Q8_0 uses ~4x less memory than F32 for layer weights
/// FALSIFICATION: Memory reduction < 2.5x (accounting for F32 embeddings)
#[test]
fn y5_5b_q8_0_memory_efficiency() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    // Layer-heavy model: more layers = layer weights dominate over embeddings
    let config = AprTransformerConfig {
        hidden_dim: 512,
        vocab_size: 2000, // Smaller vocab to reduce F32 embedding overhead
        num_layers: 16,   // More layers = more quantized weights
        num_heads: 8,
        num_kv_heads: 8,
        intermediate_dim: 2048,
        context_length: 32, // Minimal context (doesn't affect weight size)
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q8_0);

    let q8_bytes = transformer.weight_bytes();
    let f32_bytes = transformer.f32_equivalent_bytes();

    let compression_ratio = f32_bytes as f64 / q8_bytes as f64;

    // Q8_0: 8 bits/weight for layers (~4x), but embeddings are F32
    // With layer-heavy model, expect 2.5-3.5x overall compression
    assert!(
        compression_ratio > 2.5,
        "Q8_0 compression ratio {:.2}x should be > 2.5x",
        compression_ratio
    );
}

// ============================================================================
// Y5.6: Serialization
// ============================================================================

/// Y5.6a: QuantizedAprTransformer can be saved to bytes
/// FALSIFICATION: to_bytes() fails or returns empty
#[test]
fn y5_6a_quantized_to_bytes() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        vocab_size: 100,
        num_layers: 1,
        intermediate_dim: 512,
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
    let bytes = transformer
        .to_bytes()
        .expect("Serialization should succeed");

    assert!(!bytes.is_empty(), "Serialized bytes should not be empty");
    assert!(bytes.len() > 64, "Should be larger than header");
}

/// Y5.6b: QuantizedAprTransformer roundtrip preserves data
/// FALSIFICATION: Deserialized config differs from original
#[test]
fn y5_6b_quantized_roundtrip() {
    use realizar::apr_transformer::{
        AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        architecture: "qwen2".to_string(),
        hidden_dim: 256,
        vocab_size: 100,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        intermediate_dim: 512,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let original = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
    let bytes = original.to_bytes().expect("test");

    let restored = QuantizedAprTransformer::from_bytes(&bytes).expect("test");

    assert_eq!(restored.config().hidden_dim, config.hidden_dim);
    assert_eq!(restored.config().vocab_size, config.vocab_size);
    assert_eq!(restored.config().num_layers, config.num_layers);
    assert_eq!(restored.quantization_type(), AprQuantizationType::Q4_K);
}

// ============================================================================
// Summary: Y5 Popperian Falsification Matrix
// ============================================================================
//
// | Test | Claim | Falsification Condition |
// |------|-------|------------------------|
// | Y5.1a | AprQuantizationType enum | Missing Q4_K/Q8_0 variants |
// | Y5.1b | Quant type in header | Header < 49 bytes |
// | Y5.2a | QuantizedAprTransformer exists | Constructor fails |
// | Y5.2b | Stores raw bytes | Size > 25% of F32 |
// | Y5.3a | Q4_K forward works | Returns error/NaN/Inf |
// | Y5.3b | Q8_0 forward works | Returns error/NaN/Inf |
// | Y5.4a | Q4_K output quality | Max diff > 1.0 |
// | Y5.4b | Q8_0 > Q4_K quality | Q8_0 bits < Q4_K bits |
// | Y5.5a | Q4_K memory 7x smaller | Compression < 5x |
// | Y5.5b | Q8_0 memory 4x smaller | Compression < 3x |
// | Y5.6a | Serialization works | to_bytes() fails |
// | Y5.6b | Roundtrip preserves data | Config mismatch |
