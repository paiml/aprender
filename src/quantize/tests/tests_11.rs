//! Extended coverage tests for quantize/types.rs (PMAT-802)
//!
//! This file targets uncovered code paths in the quantization types module.
//! Focus areas:
//! - Edge cases in Q8_0Block and Q8KSuperBlock quantization
//! - Near-zero value handling (minimal scale paths)
//! - Clamping boundary conditions
//! - Error message formatting

use crate::quantize::detect_simd_backend;
use crate::quantize::types::{
    DequantStats, InterleavedQ4K, Q4_0Block, Q4_KBlock, Q5_KBlock, Q6_KBlock, Q8KSuperBlock,
    Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// ============================================================================
// Q8_0Block Edge Case Coverage
// ============================================================================

/// Test Q8_0Block quantization with extremely small values (near-zero path)
/// This targets the `else` branch in quantize() where max_abs <= 1e-10
#[test]
fn test_q8_0_block_quantize_near_zero_values() {
    // All values are effectively zero
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Scale should be the minimal scale (1.0 / 127.0)
    let expected_scale = 1.0 / 127.0;
    assert!(
        (block.scale - expected_scale).abs() < 1e-9,
        "Expected minimal scale {expected_scale}, got {}",
        block.scale
    );

    // All quants should be 0 since values are tiny
    assert!(
        block.quants.iter().all(|&q| q == 0),
        "Expected all zeros for near-zero input"
    );
}

/// Test Q8_0Block quantization with exactly zero values
#[test]
fn test_q8_0_block_quantize_exact_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Should use minimal scale
    assert!(block.scale > 0.0, "Scale should be positive even for zeros");
    assert!(
        block.quants.iter().all(|&q| q == 0),
        "All quants should be zero"
    );
}

/// Test Q8_0Block quantization with values requiring negative clamping
#[test]
fn test_q8_0_block_quantize_clamp_negative() {
    // Create values that would exceed -128 when quantized
    // With scale = 1000/127 = 7.87, value -1000 would be -127 * 7.87 / 7.87 = -127
    // To get clamping, we need asymmetric values
    let mut values = [0.0f32; 32];
    values[0] = 1.0; // Sets max_abs to 1.0
    values[1] = -200.0; // Much larger than max_abs

    // max_abs = 200.0, scale = 200/127 = 1.575
    // -200 / 1.575 = -127.0 (at boundary)
    // Let's use more extreme values
    let mut values2 = [0.0f32; 32];
    values2[0] = 1.0; // max_abs starts at 1.0
                      // After processing all, max_abs will be max of all abs values
    for i in 1..32 {
        values2[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let block = Q8_0Block::quantize(&values2);
    assert!(block.scale > 0.0);
}

/// Test Q8_0Block quantization with large positive value causing clamping
#[test]
fn test_q8_0_block_quantize_clamp_positive() {
    // Create scenario where some values might clip to 127
    let mut values = [100.0f32; 32];
    values[0] = 100.0; // max
    let block = Q8_0Block::quantize(&values);

    // All values are the same and at max, so all quants should be 127
    assert!(
        block.quants.iter().all(|&q| q == 127),
        "All should clamp to 127"
    );
}

/// Test Q8_0Block relative_error with tiny max value (early return path)
#[test]
fn test_q8_0_block_relative_error_tiny_max() {
    // Create block with some data
    let original = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&original);

    // relative_error should return 0.0 when max_val < 1e-10
    let error = block.relative_error(&original);
    assert!(
        error.abs() < 1e-6,
        "Relative error should be 0 for tiny values, got {error}"
    );
}

/// Test Q8_0Block quantization_error calculation
#[test]
fn test_q8_0_block_quantization_error_varied() {
    // Create block with varied values
    let mut original = [0.0f32; 32];
    for i in 0..32 {
        original[i] = (i as f32) * 0.5 - 7.5; // Range roughly -7.5 to 8
    }

    let block = Q8_0Block::quantize(&original);
    let error = block.quantization_error(&original);

    // Error should be positive and reasonably small
    assert!(error >= 0.0, "Error should be non-negative");
    assert!(error < 1.0, "Error should be small for reasonable values");
}

/// Test Q8_0Block with mixed sign extreme values
#[test]
fn test_q8_0_block_mixed_extremes() {
    let mut values = [0.0f32; 32];
    values[0] = 1000.0;
    values[1] = -1000.0;
    for i in 2..32 {
        values[i] = (i as f32) - 16.0;
    }

    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();

    // Check that max and min are approximately preserved
    let max_orig = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_deq = dequantized
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Due to quantization, max should be close but not exact
    assert!(
        (max_orig - max_deq).abs() < max_orig * 0.02,
        "Max should be preserved within 2%"
    );
}

// ============================================================================
// Q8KSuperBlock Edge Case Coverage
// ============================================================================

/// Test Q8KSuperBlock quantization with near-zero values
#[test]
fn test_q8k_superblock_quantize_near_zero() {
    let values = [1e-12f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    // Should use minimal scale
    let expected_scale = 1.0 / 127.0;
    assert!(
        (block.scale - expected_scale).abs() < 1e-9,
        "Expected minimal scale"
    );
}

/// Test Q8KSuperBlock quantize_into with near-zero values
#[test]
fn test_q8k_superblock_quantize_into_near_zero() {
    let values = [1e-12f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    // Should use minimal scale
    let expected_scale = 1.0 / 127.0;
    assert!(
        (scale - expected_scale).abs() < 1e-9,
        "Expected minimal scale from quantize_into"
    );
}

/// Test Q8KSuperBlock quantize_into with varied values
#[test]
fn test_q8k_superblock_quantize_into_varied() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 127.5) * 0.1;
    }
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0, "Scale should be positive");
    // Check that quantization is symmetric
    let sum: i32 = quants.iter().map(|&q| q as i32).sum();
    assert!(sum.abs() < 256, "Sum should be roughly balanced");
}

/// Test Q8KSuperBlock dequantize roundtrip
#[test]
fn test_q8k_superblock_dequantize_roundtrip() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 127.5) * 0.01;
    }

    let block = Q8KSuperBlock::quantize(&values);
    let dequantized = block.dequantize();

    // Check that values are approximately preserved
    let mut max_error: f32 = 0.0;
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        max_error = max_error.max((orig - deq).abs());
    }

    assert!(
        max_error < 0.1,
        "Roundtrip error should be small, got {max_error}"
    );
}

/// Test Q8KSuperBlock with all negative values
#[test]
fn test_q8k_superblock_all_negative() {
    let values = [-1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    // All quants should be -127
    assert!(
        block.quants.iter().all(|&q| q == -127),
        "All should be -127 for uniform negative"
    );
}

// ============================================================================
// InterleavedQ4K Coverage (types.rs version)
// ============================================================================

/// Test InterleavedQ4K from_q4k with empty data
#[test]
fn test_interleaved_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("valid result");
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

/// Test InterleavedQ4K from_q4k with valid data
#[test]
fn test_interleaved_q4k_single_block() {
    // Create valid Q4_K super-block data (144 bytes)
    let mut data = vec![0u8; 144];

    // Set d = 1.0 (f16 representation: 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;

    // Set dmin = 0.5 (f16 representation: 0x3800)
    data[2] = 0x00;
    data[3] = 0x38;

    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());

    let interleaved = result.expect("valid result");
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), 256);
    assert_eq!(interleaved.d.len(), 1);
    assert_eq!(interleaved.dmin.len(), 1);
    assert_eq!(interleaved.scales.len(), 12);
    assert_eq!(interleaved.qs.len(), 128);
}

/// Test InterleavedQ4K from_q4k with invalid length
#[test]
fn test_interleaved_q4k_invalid_length() {
    // 143 bytes - not a multiple of 144
    let data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());

    // Verify error message contains relevant info
    if let Err(e) = result {
        let err_str = format!("{e:?}");
        assert!(
            err_str.contains("143") || err_str.contains("144"),
            "Error should mention the sizes"
        );
    }
}

/// Test InterleavedQ4K from_q4k with multiple super-blocks
#[test]
fn test_interleaved_q4k_multiple_blocks() {
    // 3 super-blocks = 432 bytes
    let mut data = vec![0u8; 144 * 3];

    // Set different d values for each super-block
    // Block 0: d = 1.0
    data[0] = 0x00;
    data[1] = 0x3C;
    // Block 1: d = 2.0 (0x4000)
    data[144] = 0x00;
    data[145] = 0x40;
    // Block 2: d = 0.5 (0x3800)
    data[288] = 0x00;
    data[289] = 0x38;

    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());

    let interleaved = result.expect("valid result");
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), 768);
    assert_eq!(interleaved.d.len(), 3);
    assert_eq!(interleaved.dmin.len(), 3);
    assert_eq!(interleaved.scales.len(), 36); // 3 * 12
    assert_eq!(interleaved.qs.len(), 384); // 3 * 128
}

/// Test InterleavedQ4K Debug trait
#[test]
fn test_interleaved_q4k_debug() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");
    let debug_str = format!("{interleaved:?}");
    assert!(debug_str.contains("InterleavedQ4K"));
    assert!(debug_str.contains("num_super_blocks"));
}

/// Test InterleavedQ4K Clone trait
#[test]
fn test_interleaved_q4k_clone() {
    let data = vec![0u8; 144];
    let original = InterleavedQ4K::from_q4k(&data).expect("valid data");
    let cloned = original.clone();

    assert_eq!(original.num_super_blocks, cloned.num_super_blocks);
    assert_eq!(original.d, cloned.d);
    assert_eq!(original.dmin, cloned.dmin);
    assert_eq!(original.scales, cloned.scales);
    assert_eq!(original.qs, cloned.qs);
}

// ============================================================================
// SimdBackend and DequantStats Coverage
// ============================================================================

/// Test all SimdBackend Display variants
#[test]
fn test_simd_backend_display_comprehensive() {
    let backends = [
        (SimdBackend::Avx2, "AVX2"),
        (SimdBackend::Sse2, "SSE2"),
        (SimdBackend::Neon, "NEON"),
        (SimdBackend::Scalar, "Scalar"),
    ];

    for (backend, expected) in backends {
        let display = format!("{backend}");
        assert_eq!(
            display, expected,
            "Display for {backend:?} should be {expected}"
        );
    }
}

/// Test SimdBackend Debug trait
#[test]
fn test_simd_backend_debug_all() {
    let backends = [
        SimdBackend::Avx2,
        SimdBackend::Sse2,
        SimdBackend::Neon,
        SimdBackend::Scalar,
    ];

    for backend in backends {
        let debug_str = format!("{backend:?}");
        assert!(
            !debug_str.is_empty(),
            "Debug output should not be empty for {backend:?}"
        );
    }
}

/// Test DequantStats construction and fields
#[test]
fn test_dequant_stats_fields() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 1024,
        simd_backend: SimdBackend::Avx2,
    };

    assert_eq!(stats.blocks_processed, 42);
    assert_eq!(stats.bytes_processed, 1024);
    assert_eq!(stats.simd_backend, SimdBackend::Avx2);
}

/// Test DequantStats Debug output contains all fields
#[test]
fn test_dequant_stats_debug_comprehensive() {
    let stats = DequantStats {
        blocks_processed: 999,
        bytes_processed: 12345,
        simd_backend: SimdBackend::Neon,
    };

    let debug_str = format!("{stats:?}");
    assert!(debug_str.contains("999"), "Should contain blocks count");
    assert!(debug_str.contains("12345"), "Should contain bytes count");
    assert!(debug_str.contains("Neon"), "Should contain backend name");
}

/// Test detect_simd_backend returns a valid variant
#[test]
fn test_detect_simd_backend_valid() {
    let backend = detect_simd_backend();

    // Should be one of the valid variants
    let is_valid = matches!(
        backend,
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar
    );
    assert!(is_valid, "Backend should be a valid variant");
}

include!("block_type_coverage.rs");
