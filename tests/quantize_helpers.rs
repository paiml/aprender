//! Quantize module helper tests for Phase 46 - The Soft Target Sweep
//!
//! Goal: Push quantize/mod.rs from 65% to >80% coverage.

use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q8_0, dequantize_q8_blocks, detect_simd_backend,
    f16_to_f32, quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_to_q8_blocks,
    DequantStats, InterleavedQ4K, Q8KSuperBlock, Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// ============================================================================
// Q8_0 Block Quantization Tests
// ============================================================================

#[test]
fn test_quantize_to_q8_blocks_basic() {
    // 32 values = 1 block
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization should succeed");
    assert_eq!(blocks.len(), 1);
}

#[test]
fn test_quantize_to_q8_blocks_multiple() {
    // 64 values = 2 blocks
    let values: Vec<f32> = (0..64).map(|i| i as f32 / 10.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization should succeed");
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_large() {
    // 1024 values = 32 blocks
    let values: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("quantization should succeed");
    assert_eq!(blocks.len(), 32);
}

#[test]
fn test_quantize_to_q8_blocks_invalid_length() {
    // 31 values - not a multiple of 32
    let values: Vec<f32> = (0..31).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_empty() {
    // Empty input is technically valid (0 blocks)
    let values: Vec<f32> = vec![];
    let blocks = quantize_to_q8_blocks(&values).expect("empty should succeed");
    assert!(blocks.is_empty());
}

#[test]
fn test_quantize_to_q8_blocks_one_extra() {
    // 33 values - one extra
    let values: Vec<f32> = (0..33).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

// ============================================================================
// Q8_0 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q8_blocks_basic() {
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
}

#[test]
fn test_dequantize_q8_blocks_roundtrip() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let blocks = quantize_to_q8_blocks(&original).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);

    // Check roundtrip error is small (quantization is lossy)
    for (orig, deq) in original.iter().zip(dequantized.iter()) {
        let error = (orig - deq).abs();
        assert!(
            error < 0.5,
            "Roundtrip error too large: {} vs {}",
            orig,
            deq
        );
    }
}

#[test]
fn test_dequantize_q8_blocks_empty() {
    let blocks: Vec<Q8_0Block> = vec![];
    let dequantized = dequantize_q8_blocks(&blocks);
    assert!(dequantized.is_empty());
}

#[test]
fn test_dequantize_q8_blocks_zeros() {
    let values: Vec<f32> = vec![0.0; 32];
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);

    for val in &dequantized {
        assert!((val.abs()) < 1e-6, "Expected zero, got {}", val);
    }
}

#[test]
fn test_dequantize_q8_blocks_large_values() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32) * 100.0).collect();
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);

    // Large values should still be representable
    assert_eq!(dequantized.len(), 32);
    // Check approximate preservation of magnitude
    let orig_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let deq_max = dequantized
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (orig_max - deq_max).abs() / orig_max < 0.1,
        "Large value not preserved"
    );
}

#[test]
fn test_dequantize_q8_blocks_negative() {
    let values: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);

    // Check that negative values are preserved
    for (i, val) in dequantized.iter().enumerate() {
        if i > 0 {
            assert!(*val < 0.0, "Expected negative at index {}, got {}", i, val);
        }
    }
}

// ============================================================================
// InterleavedQ4K Tests
// ============================================================================

#[test]
fn test_interleaved_q4k_invalid_length() {
    // Not a multiple of super-block size (144 bytes)
    let data = vec![0u8; 100];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_empty() {
    // Empty input
    let data = vec![];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("empty should succeed");
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_one_superblock() {
    // One super-block = 144 bytes
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("single superblock should succeed");
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), QK_K); // 256 values
}

#[test]
fn test_interleaved_q4k_multiple_superblocks() {
    // 3 super-blocks = 432 bytes
    let data = vec![0u8; 144 * 3];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("multiple superblocks should succeed");
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), QK_K * 3); // 768 values
}

#[test]
fn test_interleaved_q4k_dot_wrong_length() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    // Wrong activation length
    let activations = vec![0.0f32; 100]; // Need 256
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_dot_correct_length() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    let activations = vec![0.0f32; 256];
    let result = interleaved.dot(&activations);
    // Should succeed (result is 0 for zero inputs)
    assert!(result.is_ok());
    assert!((result.unwrap().abs()) < 1e-6);
}

#[test]
fn test_interleaved_q4k_dot_nonzero() {
    // Create proper Q4K data with actual values
    let mut data = vec![0u8; 144];

    // Set d (scale) to 1.0 in f16 format
    // f16 1.0 = 0x3C00
    data[0] = 0x00;
    data[1] = 0x3C;

    // dmin = 0
    data[2] = 0x00;
    data[3] = 0x00;

    // Set some scales
    for i in 0..12 {
        data[4 + i] = 16; // Small scale values
    }

    // Set some quantized values
    for i in 0..128 {
        data[16 + i] = 0x55; // Pattern: low=5, high=5
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).unwrap();

    // Create activations
    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
    // Result should be non-zero
    let dot_val = result.unwrap();
    // Just verify it doesn't panic and returns a reasonable value
    assert!(dot_val.is_finite());
}

// ============================================================================
// Q8_0Block Direct Tests
// ============================================================================

#[test]
fn test_q8_0_block_quantize() {
    let values: [f32; BLOCK_SIZE] = [1.0; BLOCK_SIZE];
    let block = Q8_0Block::quantize(&values);
    // Block should have non-zero scale
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8_0_block_dequantize() {
    let values: [f32; BLOCK_SIZE] = std::array::from_fn(|i| i as f32);
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();

    // Should have BLOCK_SIZE values
    assert_eq!(dequantized.len(), BLOCK_SIZE);

    // Values should be roughly preserved
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let error = (orig - deq).abs();
        assert!(error < 1.0, "Error too large: {} vs {}", orig, deq);
    }
}

#[test]
fn test_q8_0_block_all_same() {
    let values: [f32; BLOCK_SIZE] = [42.0; BLOCK_SIZE];
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();

    // All values should be similar after roundtrip
    for val in &dequantized {
        assert!((val - 42.0).abs() < 1.0, "Value not preserved: {}", val);
    }
}

#[test]
fn test_q8_0_block_mixed_signs() {
    let values: [f32; BLOCK_SIZE] = std::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();

    // Check signs are preserved
    for (i, val) in dequantized.iter().enumerate() {
        if i % 2 == 0 {
            assert!(*val > 0.0, "Expected positive at {}, got {}", i, val);
        } else {
            assert!(*val < 0.0, "Expected negative at {}, got {}", i, val);
        }
    }
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_quantize_denormals() {
    // Test with very small (denormal) values
    let values: Vec<f32> = (0..32).map(|_| 1e-38f32).collect();
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
}

#[test]
fn test_quantize_infinities() {
    // Test with infinity - should not panic
    let mut values = vec![0.0f32; 32];
    values[0] = f32::INFINITY;
    values[1] = f32::NEG_INFINITY;
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32);
}

#[test]
fn test_quantize_nans() {
    // Test with NaN - should not panic
    let mut values = vec![0.0f32; 32];
    values[0] = f32::NAN;
    let _blocks = quantize_to_q8_blocks(&values).unwrap();
    // NaN handling is implementation-defined, just verify no panic
}

#[test]
fn test_quantize_stress_many_blocks() {
    // Stress test with many blocks
    let values: Vec<f32> = (0..32768).map(|i| (i as f32).sin()).collect();
    let blocks = quantize_to_q8_blocks(&values).unwrap();
    assert_eq!(blocks.len(), 1024);
    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 32768);
}

// ============================================================================
// quantize_activations_q8_0 Tests
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_basic() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
    assert_eq!(quants.len(), activations.len());
}

#[test]
fn test_quantize_activations_q8_0_zeros() {
    let activations = vec![0.0f32; 256];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_ones() {
    let activations = vec![1.0f32; 256];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_negative() {
    let activations: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.01).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

// ============================================================================
// Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q4_0_basic() {
    // Q4_0 block: 2 bytes scale (f16) + 16 bytes quantized = 18 bytes per block
    // One block contains 32 values
    let data = vec![0u8; 18];
    let output = dequantize_q4_0(&data).expect("should dequantize");
    assert_eq!(output.len(), 32);
}

#[test]
fn test_dequantize_q4_0_multiple_blocks() {
    // Two blocks = 36 bytes = 64 values
    let data = vec![0u8; 36];
    let output = dequantize_q4_0(&data).expect("should dequantize");
    assert_eq!(output.len(), 64);
}

#[test]
fn test_dequantize_q8_0_basic() {
    // Q8_0 block: 2 bytes scale (f16) + 32 bytes quantized = 34 bytes per block
    let data = vec![0u8; 34];
    let output = dequantize_q8_0(&data).expect("should dequantize");
    assert_eq!(output.len(), 32);
}

#[test]
fn test_dequantize_q8_0_multiple_blocks() {
    // Two blocks = 68 bytes = 64 values
    let data = vec![0u8; 68];
    let output = dequantize_q8_0(&data).expect("should dequantize");
    assert_eq!(output.len(), 64);
}

#[test]
fn test_dequantize_f16_basic() {
    // f16 data: 2 bytes per value
    let data = vec![0u8; 8]; // 4 values
    let output = dequantize_f16(&data).expect("should dequantize");
    assert_eq!(output.len(), 4);
}

#[test]
fn test_f16_to_f32_zero() {
    // f16 zero = 0x0000
    let result = f16_to_f32(0x0000);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_one() {
    // f16 1.0 = 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_negative_one() {
    // f16 -1.0 = 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!((result - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_half() {
    // f16 0.5 = 0x3800
    let result = f16_to_f32(0x3800);
    assert!((result - 0.5).abs() < 1e-6);
}

// ============================================================================
// quantize_rmsnorm_q8_0 Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_basic() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 + 0.1).collect();
    let weights = vec![1.0f32; 256];
    let eps = 1e-6;

    let (normalized, quants) = quantize_rmsnorm_q8_0(&activations, &weights, eps);
    assert!(!normalized.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_quantize_rmsnorm_q8_0_ones() {
    let activations = vec![1.0f32; 256];
    let weights = vec![1.0f32; 256];
    let eps = 1e-6;

    let (normalized, quants) = quantize_rmsnorm_q8_0(&activations, &weights, eps);
    assert!(!normalized.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_quantize_rmsnorm_q8_0_different_weights() {
    let activations = vec![1.0f32; 256];
    let weights: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let eps = 1e-6;

    let (normalized, quants) = quantize_rmsnorm_q8_0(&activations, &weights, eps);
    assert!(!normalized.is_empty());
    assert!(!quants.is_empty());
}

// ============================================================================
// Q8_0Block Error Methods (quantize/types.rs coverage)
// ============================================================================

#[test]
fn test_q8_0_block_quantization_error() {
    let values: [f32; 32] = std::array::from_fn(|i| i as f32);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Error should be relatively small for well-behaved values
    assert!(error < 0.5, "quantization error too large: {error}");
}

#[test]
fn test_q8_0_block_quantization_error_zero_block() {
    let values: [f32; 32] = [0.0; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Zero values should have no error
    assert!(
        error < 1e-6,
        "zero block should have minimal error: {error}"
    );
}

#[test]
fn test_q8_0_block_quantization_error_large_values() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 100.0);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    // Large values will have larger absolute error, but should be bounded
    assert!(error < 50.0, "error too large for large values: {error}");
}

#[test]
fn test_q8_0_block_relative_error() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 + 1.0) * 10.0);
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Relative error should be small (< 1%)
    assert!(rel_error < 0.01, "relative error too large: {rel_error}");
}

#[test]
fn test_q8_0_block_relative_error_zero_block() {
    let values: [f32; 32] = [0.0; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Zero values special case should return 0
    assert_eq!(rel_error, 0.0, "zero block should have 0 relative error");
}

#[test]
fn test_q8_0_block_relative_error_small_values() {
    // Small values near epsilon threshold
    let values: [f32; 32] = std::array::from_fn(|i| i as f32 * 1e-11);
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    // Should handle near-zero gracefully
    assert!(rel_error >= 0.0, "relative error should be non-negative");
}

// ============================================================================
// Q8KSuperBlock Tests (quantize/types.rs coverage)
// ============================================================================

#[test]
fn test_q8k_superblock_quantize() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) / 128.0);
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0, "scale should be positive");
    assert_eq!(block.quants.len(), 256);
}

#[test]
fn test_q8k_superblock_quantize_zeros() {
    let values: [f32; 256] = [0.0; 256];
    let block = Q8KSuperBlock::quantize(&values);
    // Should use minimal scale and have all zero quants
    assert!(block.scale > 0.0, "scale should be positive even for zeros");
    for q in block.quants {
        assert_eq!(q, 0, "zero values should quantize to 0");
    }
}

#[test]
fn test_q8k_superblock_quantize_max_values() {
    let values: [f32; 256] = std::array::from_fn(|i| if i % 2 == 0 { 1000.0 } else { -1000.0 });
    let block = Q8KSuperBlock::quantize(&values);
    // Scale should be 1000/127
    let expected_scale = 1000.0 / 127.0;
    assert!((block.scale - expected_scale).abs() < 0.01);
    // Alternating +127/-127 (or close)
    assert!(block.quants[0] > 120 || block.quants[0] < -120);
    assert!(block.quants[1] > 120 || block.quants[1] < -120);
}

#[test]
fn test_q8k_superblock_dequantize() {
    let values: [f32; 256] = std::array::from_fn(|i| i as f32);
    let block = Q8KSuperBlock::quantize(&values);
    let deq = block.dequantize();
    assert_eq!(deq.len(), 256);
    // Check first few values are approximately correct
    for i in 0..10 {
        let diff = (deq[i] - values[i]).abs();
        assert!(
            diff < 3.0,
            "dequantized value {i} differs too much: {} vs {}",
            deq[i],
            values[i]
        );
    }
}

#[test]
fn test_q8k_superblock_dequantize_roundtrip() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.5);
    let block = Q8KSuperBlock::quantize(&values);
    let deq = block.dequantize();
    // Max error should be bounded
    let max_error = values
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 1.0, "roundtrip error too large: {max_error}");
}

#[test]
fn test_q8k_superblock_quantize_into() {
    let values: [f32; 256] = std::array::from_fn(|i| i as f32);
    let mut scale: f32 = 0.0;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // First value (0) should quantize to 0
    assert_eq!(quants[0], 0);
    // Values should increase (monotonic for monotonic input)
    assert!(
        quants[200] > quants[50],
        "quantized values should preserve order"
    );
}

#[test]
fn test_q8k_superblock_quantize_into_zeros() {
    let values = [0.0f32; 256];
    let mut scale: f32 = 0.0;
    let mut quants = [99i8; 256]; // Pre-fill with non-zero

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0); // Minimal scale
    for q in quants {
        assert_eq!(q, 0);
    }
}

// ============================================================================
// SimdBackend Tests (quantize/types.rs coverage)
// ============================================================================

#[test]
fn test_simd_backend_display_avx2() {
    let backend = SimdBackend::Avx2;
    assert_eq!(format!("{backend}"), "AVX2");
}

#[test]
fn test_simd_backend_display_sse2() {
    let backend = SimdBackend::Sse2;
    assert_eq!(format!("{backend}"), "SSE2");
}

#[test]
fn test_simd_backend_display_neon() {
    let backend = SimdBackend::Neon;
    assert_eq!(format!("{backend}"), "NEON");
}

#[test]
fn test_simd_backend_display_scalar() {
    let backend = SimdBackend::Scalar;
    assert_eq!(format!("{backend}"), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend: SimdBackend = Default::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_equality() {
    assert_eq!(SimdBackend::Avx2, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Sse2);
    assert_ne!(SimdBackend::Scalar, SimdBackend::Neon);
}

#[test]
fn test_detect_simd_backend() {
    // Should return a valid backend (we don't know which on this machine)
    let backend = detect_simd_backend();
    // Just verify it's one of the valid variants
    let valid = matches!(
        backend,
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar
    );
    assert!(valid, "detect_simd_backend returned invalid variant");
}

#[test]
fn test_detect_simd_backend_is_deterministic() {
    let b1 = detect_simd_backend();
    let b2 = detect_simd_backend();
    assert_eq!(b1, b2, "detect_simd_backend should be deterministic");
}

// ============================================================================
// DequantStats Tests (quantize/types.rs coverage)
// ============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats: DequantStats = Default::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_clone() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 1000,
        simd_backend: SimdBackend::Avx2,
    };
    let cloned = stats;
    assert_eq!(cloned.blocks_processed, 100);
    assert_eq!(cloned.bytes_processed, 1000);
    assert_eq!(cloned.simd_backend, SimdBackend::Avx2);
}

#[test]
fn test_dequant_stats_debug() {
    let stats = DequantStats::default();
    let debug_str = format!("{stats:?}");
    assert!(debug_str.contains("DequantStats"));
    assert!(debug_str.contains("blocks_processed"));
}

// ============================================================================
// InterleavedQ4K num_values Tests (quantize/types.rs coverage)
// ============================================================================

#[test]
fn test_interleaved_q4k_num_values_one() {
    // 1 super-block = 144 bytes
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");
    assert_eq!(interleaved.num_values(), 256); // QK_K = 256
}

#[test]
fn test_interleaved_q4k_num_values_multiple() {
    // 4 super-blocks = 576 bytes
    let data = vec![0u8; 144 * 4];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");
    assert_eq!(interleaved.num_values(), 256 * 4);
}

#[test]
fn test_interleaved_q4k_num_values_large() {
    // 100 super-blocks
    let data = vec![0u8; 144 * 100];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid data");
    assert_eq!(interleaved.num_values(), 256 * 100);
}
