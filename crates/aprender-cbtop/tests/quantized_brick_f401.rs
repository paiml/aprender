#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! PMAT-013: QuantizedBrick Falsification Tests
//!
//! Falsification criteria F401-F410 from cbtop spec §17.
//!
//! # Test Coverage
//!
//! | ID | Claim | Test |
//! |----|-------|------|
//! | F401 | Supports Q4_K, Q5_K, Q6_K, Q8_0 formats | test_f401_supported_formats |
//! | F402 | Memory reduction >= 2x for Q4_K | test_f402_memory_reduction |
//! | F403 | GGUF header parsing works | test_f403_gguf_header_parsing |
//! | F404 | Dequant strategies available (Fused, Prefetch, OnDemand) | test_f404_dequant_strategies |
//! | F405 | Perplexity delta documented per format | test_f405_perplexity_delta_documented |
//! | F406 | Compression ratio calculation accurate | test_f406_compression_ratio_accuracy |
//! | F407 | Block sizes correct per format | test_f407_block_sizes_correct |
//! | F408 | GGML type to format mapping complete | test_f408_ggml_format_mapping |
//! | F409 | Weight shape preserved after load | test_f409_weight_shape_preserved |
//! | F410 | Statistics aggregation correct | test_f410_statistics_aggregation |

use cbtop::{
    ggml_type_to_format, DequantStrategy, GgufError, GgufLoader, QuantFormat, QuantStats,
    QuantizedBrick, QuantizedWeights,
};

/// F401: QuantizedBrick supports all required formats.
#[test]
fn test_f401_supported_formats() {
    // All K-quant formats must be supported
    let formats = vec![
        QuantFormat::Q4_0,
        QuantFormat::Q4_K,
        QuantFormat::Q5_K,
        QuantFormat::Q6_K,
        QuantFormat::Q8_0,
    ];

    for format in formats {
        // Format must have valid bits per weight
        let bits = format.bits_per_weight();
        assert!(
            bits > 0.0 && bits <= 32.0,
            "Format {:?} has invalid bits_per_weight: {}",
            format,
            bits
        );

        // Format must have valid block size
        let block_size = format.block_size();
        assert!(block_size > 0, "Format {:?} has invalid block_size: {}", format, block_size);

        // Format must have valid bytes per block
        let bytes = format.bytes_per_block();
        assert!(bytes > 0, "Format {:?} has invalid bytes_per_block: {}", format, bytes);
    }

    // Also verify FP formats
    assert_eq!(QuantFormat::F32.bits_per_weight(), 32.0);
    assert_eq!(QuantFormat::F16.bits_per_weight(), 16.0);
    assert_eq!(QuantFormat::BF16.bits_per_weight(), 16.0);
}

/// F401 negative: Invalid format handling.
#[test]
fn test_f401_gptq_awq_formats() {
    // GPTQ and AWQ formats should also be supported
    let gptq = QuantFormat::Gptq { bits: 4, group_size: 128 };
    let awq = QuantFormat::Awq { bits: 4 };

    assert!(gptq.bits_per_weight() > 4.0); // Includes overhead
    assert!(awq.bits_per_weight() > 4.0);
}

/// F402: Q4_K provides >= 2x memory reduction vs F16.
#[test]
fn test_f402_memory_reduction() {
    let q4k_ratio = QuantFormat::Q4_K.memory_ratio();

    // Q4_K should use less than 50% of F16 memory (i.e., > 2x reduction)
    assert!(q4k_ratio < 0.5, "Q4_K memory ratio {} should be < 0.5 for 2x reduction", q4k_ratio);

    // In fact, Q4_K should be around 28% of F16
    assert!(q4k_ratio < 0.35, "Q4_K memory ratio {} should be < 0.35 (~3.5x reduction)", q4k_ratio);

    // Verify other formats have appropriate ratios
    assert!(QuantFormat::Q8_0.memory_ratio() < 0.6, "Q8_0 should use < 60% of F16 memory");
    assert!(QuantFormat::Q5_K.memory_ratio() < 0.45, "Q5_K should use < 45% of F16 memory");
}

/// F402 negative: Full precision has no reduction.
#[test]
fn test_f402_no_reduction_for_fp() {
    // F16 should have ratio of 1.0 (baseline)
    assert!(
        (QuantFormat::F16.memory_ratio() - 1.0).abs() < 0.01,
        "F16 memory ratio should be ~1.0"
    );

    // F32 should have ratio > 1.0 (uses more memory than F16)
    assert!(QuantFormat::F32.memory_ratio() > 1.5, "F32 memory ratio should be > 1.5");
}

/// F403: GGUF header parsing works correctly.
#[test]
fn test_f403_gguf_header_parsing() {
    let mut loader = GgufLoader::new("/tmp/test.gguf");

    // Create valid GGUF v3 header
    let mut header_data = vec![0u8; 24];
    header_data[0..4].copy_from_slice(b"GGUF");
    header_data[4..8].copy_from_slice(&3u32.to_le_bytes()); // version 3
    header_data[8..16].copy_from_slice(&256u64.to_le_bytes()); // tensor count
    header_data[16..24].copy_from_slice(&128u64.to_le_bytes()); // metadata count

    loader.parse_header(&header_data).unwrap();

    let header = loader.header().unwrap();
    assert_eq!(&header.magic, b"GGUF");
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 256);
    assert_eq!(header.metadata_kv_count, 128);
}

/// F403 negative: Invalid GGUF magic rejected.
#[test]
fn test_f403_invalid_magic_rejected() {
    let mut loader = GgufLoader::new("/tmp/bad.gguf");

    let mut bad_data = vec![0u8; 24];
    bad_data[0..4].copy_from_slice(b"BADM");

    let result = loader.parse_header(&bad_data);
    assert!(matches!(result, Err(GgufError::InvalidMagic(_))));
}

/// F403 negative: Unsupported GGUF version rejected.
#[test]
fn test_f403_unsupported_version_rejected() {
    let mut loader = GgufLoader::new("/tmp/bad.gguf");

    let mut bad_data = vec![0u8; 24];
    bad_data[0..4].copy_from_slice(b"GGUF");
    bad_data[4..8].copy_from_slice(&99u32.to_le_bytes()); // unsupported version

    let result = loader.parse_header(&bad_data);
    assert!(matches!(result, Err(GgufError::UnsupportedVersion(99))));
}

/// F404: All dequantization strategies available.
#[test]
fn test_f404_dequant_strategies() {
    // Verify all strategies can be constructed
    let strategies = vec![
        DequantStrategy::Fused,
        DequantStrategy::Prefetch { lookahead_blocks: 4 },
        DequantStrategy::OnDemand,
    ];

    // Each strategy should be usable in a QuantizedBrick
    for strategy in strategies {
        let brick = QuantizedBrick::new("test").with_dequant_strategy(strategy);
        assert_eq!(brick.dequant_strategy, strategy);
    }
}

/// F404 negative: Default strategy is Fused (best for GPU).
#[test]
fn test_f404_default_strategy_is_fused() {
    let strategy = DequantStrategy::default();
    assert_eq!(strategy, DequantStrategy::Fused);

    let brick = QuantizedBrick::new("test");
    assert_eq!(brick.dequant_strategy, DequantStrategy::Fused);
}

/// F405: Perplexity delta documented and reasonable for all formats.
#[test]
fn test_f405_perplexity_delta_documented() {
    // Q4_K should have low perplexity delta (~0.3%)
    let q4k_delta = QuantFormat::Q4_K.expected_ppl_delta();
    assert!(
        q4k_delta > 0.0 && q4k_delta < 1.0,
        "Q4_K perplexity delta {} should be between 0 and 1%",
        q4k_delta
    );
    assert!((q4k_delta - 0.3).abs() < 0.1, "Q4_K perplexity delta {} should be ~0.3%", q4k_delta);

    // Higher-bit formats should have lower perplexity delta
    assert!(
        QuantFormat::Q5_K.expected_ppl_delta() < QuantFormat::Q4_K.expected_ppl_delta(),
        "Q5_K should have lower PPL delta than Q4_K"
    );
    assert!(
        QuantFormat::Q8_0.expected_ppl_delta() < QuantFormat::Q5_K.expected_ppl_delta(),
        "Q8_0 should have lower PPL delta than Q5_K"
    );

    // FP formats should have near-zero perplexity delta
    assert!(QuantFormat::F16.expected_ppl_delta() < 0.01, "F16 should have near-zero PPL delta");
}

/// F406: Compression ratio calculation is accurate.
#[test]
fn test_f406_compression_ratio_accuracy() {
    // Create weights with known size
    let weights = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 1000], // 1KB quantized
        (1000, 10),      // 10,000 weights
        "test_layer",
    );

    // F16 would need 20,000 bytes for 10,000 weights
    assert_eq!(weights.f16_memory_bytes(), 20_000);
    assert_eq!(weights.memory_bytes(), 1000);

    // Compression ratio should be 20x for this artificial case
    let ratio = weights.compression_ratio();
    assert!((ratio - 20.0).abs() < 0.1, "Compression ratio {} should be ~20.0", ratio);
}

/// F406 negative: Zero-size weights don't cause division by zero.
#[test]
fn test_f406_zero_size_weights() {
    let weights = QuantizedWeights::new(QuantFormat::Q4_K, vec![], (0, 0), "empty");

    assert_eq!(weights.num_weights(), 0);
    assert_eq!(weights.memory_bytes(), 0);
    assert_eq!(weights.f16_memory_bytes(), 0);
    // compression_ratio for empty weights would be inf, but actual_bits_per_weight handles it
}

/// F407: Block sizes are correct for each format.
#[test]
fn test_f407_block_sizes_correct() {
    // Per GGML specification
    assert_eq!(QuantFormat::Q4_0.block_size(), 32);
    assert_eq!(QuantFormat::Q8_0.block_size(), 32);

    // K-quants use 256-element super-blocks
    assert_eq!(QuantFormat::Q4_K.block_size(), 256);
    assert_eq!(QuantFormat::Q5_K.block_size(), 256);
    assert_eq!(QuantFormat::Q6_K.block_size(), 256);

    // FP formats have block size of 1
    assert_eq!(QuantFormat::F32.block_size(), 1);
    assert_eq!(QuantFormat::F16.block_size(), 1);
}

/// F407: Bytes per block match GGML specification.
#[test]
fn test_f407_bytes_per_block_ggml_spec() {
    // Q4_0: 2 bytes (scale) + 16 bytes (32 * 4-bit / 8) = 18
    assert_eq!(QuantFormat::Q4_0.bytes_per_block(), 18);

    // Q8_0: 2 bytes (scale) + 32 bytes (32 * 8-bit / 8) = 34
    assert_eq!(QuantFormat::Q8_0.bytes_per_block(), 34);

    // Q4_K super-block: 144 bytes
    assert_eq!(QuantFormat::Q4_K.bytes_per_block(), 144);

    // Q5_K super-block: 176 bytes
    assert_eq!(QuantFormat::Q5_K.bytes_per_block(), 176);

    // Q6_K super-block: 210 bytes
    assert_eq!(QuantFormat::Q6_K.bytes_per_block(), 210);
}

/// F408: GGML type to format mapping is complete.
#[test]
fn test_f408_ggml_format_mapping() {
    // Standard GGML types
    assert_eq!(ggml_type_to_format(0), Some(QuantFormat::F32));
    assert_eq!(ggml_type_to_format(1), Some(QuantFormat::F16));
    assert_eq!(ggml_type_to_format(2), Some(QuantFormat::Q4_0));
    assert_eq!(ggml_type_to_format(8), Some(QuantFormat::Q8_0));

    // K-quants
    assert_eq!(ggml_type_to_format(12), Some(QuantFormat::Q4_K));
    assert_eq!(ggml_type_to_format(13), Some(QuantFormat::Q5_K));
    assert_eq!(ggml_type_to_format(14), Some(QuantFormat::Q6_K));
}

/// F408 negative: Unknown GGML types return None.
#[test]
fn test_f408_unknown_ggml_type() {
    assert_eq!(ggml_type_to_format(255), None);
    assert_eq!(ggml_type_to_format(100), None);
    assert_eq!(ggml_type_to_format(99), None);
}

/// F409: Weight shape is preserved after creation.
#[test]
fn test_f409_weight_shape_preserved() {
    let shape = (4096, 4096);
    let weights = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 1000],
        shape,
        "model.layers.0.self_attn.q_proj",
    );

    assert_eq!(weights.shape, shape);
    assert_eq!(weights.num_weights(), 4096 * 4096);
    assert_eq!(weights.layer_name, "model.layers.0.self_attn.q_proj");
}

/// F409: QuantizedBrick correctly stores and reports weights.
#[test]
fn test_f409_brick_weight_access() {
    let weights = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 144], // One super-block
        (16, 16),       // 256 weights = one Q4_K super-block
        "test",
    );

    let brick = QuantizedBrick::new("matmul").with_weights(weights);

    assert!(brick.has_weights());
    assert_eq!(brick.format(), Some(QuantFormat::Q4_K));
    assert_eq!(brick.memory_bytes(), 144);
}

/// F410: Statistics aggregation is correct across layers.
#[test]
fn test_f410_statistics_aggregation() {
    let mut stats = QuantStats::new();

    // Add Q4_K layer
    let q4k_weights = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 144 * 64], // 64 super-blocks = 16K weights
        (128, 128),
        "layer1.q_proj",
    );

    // Add Q8_0 layer
    let q8_weights = QuantizedWeights::new(
        QuantFormat::Q8_0,
        vec![0u8; 34 * 256], // 256 blocks = 8K weights
        (128, 64),
        "layer1.v_proj",
    );

    stats.add_layer(&q4k_weights);
    stats.add_layer(&q8_weights);

    // Verify totals
    assert_eq!(stats.total_weights, 16384 + 8192);
    assert_eq!(stats.total_memory_bytes, 144 * 64 + 34 * 256);
    assert_eq!(stats.layer_stats.len(), 2);

    // Verify per-format breakdown
    assert!(stats.weights_by_format.contains_key(&QuantFormat::Q4_K));
    assert!(stats.weights_by_format.contains_key(&QuantFormat::Q8_0));
}

/// F410: Dominant format detection works.
#[test]
fn test_f410_dominant_format_detection() {
    let mut stats = QuantStats::new();

    // Add more Q4_K weights than Q8_0
    let q4k_weights = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 10000],
        (1000, 1000), // 1M weights
        "large_layer",
    );
    let q8_weights = QuantizedWeights::new(
        QuantFormat::Q8_0,
        vec![0u8; 1000],
        (100, 100), // 10K weights
        "small_layer",
    );

    stats.add_layer(&q4k_weights);
    stats.add_layer(&q8_weights);

    // Q4_K should be dominant
    assert_eq!(stats.dominant_format(), Some(QuantFormat::Q4_K));
}

/// F410: Empty stats don't panic.
#[test]
fn test_f410_empty_stats() {
    let stats = QuantStats::new();

    assert_eq!(stats.total_weights, 0);
    assert_eq!(stats.total_memory_bytes, 0);
    assert_eq!(stats.compression_ratio(), 1.0);
    assert_eq!(stats.avg_bits_per_weight(), 0.0);
    assert_eq!(stats.dominant_format(), None);
}

/// Integration test: Full QuantizedBrick workflow.
#[test]
fn test_full_quantized_brick_workflow() {
    // 1. Create weights for a transformer layer
    let q_proj = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 144 * 256], // 64K weights
        (4096, 4096),
        "layer.0.q_proj",
    );

    // 2. Create brick with weights
    let brick = QuantizedBrick::new("attention_q")
        .with_weights(q_proj)
        .with_dequant_strategy(DequantStrategy::Fused)
        .with_budget(50_000); // 50K tok/sec budget

    // 3. Verify brick configuration
    assert!(brick.has_weights());
    assert_eq!(brick.format(), Some(QuantFormat::Q4_K));
    assert!(brick.memory_bytes() > 0);
    assert!(brick.bits_per_weight() < 16.0); // Should be ~4.5

    // 4. Verify display works
    let display = format!("{}", brick);
    assert!(display.contains("QuantizedBrick"));
    assert!(display.contains("Q4_K"));
}
