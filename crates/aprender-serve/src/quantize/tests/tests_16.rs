//! Part 16: Comprehensive coverage for quantize/mod.rs uncovered paths
//!
//! Focus areas:
//! - F16_TO_F32_LUT static initialization and f16_to_f32_lut function
//! - Q8KSuperBlock::dequantize
//! - Q8_0Block::relative_error edge cases
//! - SimdBackend Display trait
//! - DequantStats struct
//! - InterleavedQ4K additional paths
//! - extract_scale_min edge cases for blocks 4-7
//! - fused_q4_0_q8_0_dot_simd boundary cases

use crate::quantize::{
    dequantize_q8_blocks, detect_simd_backend, fused_q4_0_q8_0_parallel_matvec,
    fused_q4_0_q8_0_parallel_matvec_into, fused_q8_0_q8_0_parallel_matvec,
    quantize_activations_q8k_into, quantize_to_q8_blocks, DequantStats, InterleavedQ4K,
    Q8KSuperBlock, Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// These are pub(crate), available within the crate
use crate::quantize::{
    extract_scale_min, extract_scale_min_from_slice, fused_q4_0_q8_0_dot_scalar,
    fused_q4_0_q8_0_dot_simd, fused_q8_0_q8_0_dot_scalar,
};

// =============================================================================
// F16_TO_F32_LUT and f16_to_f32_lut() Tests
// =============================================================================

#[test]
fn test_f16_to_f32_lut_special_values() {
    // The LUT is accessed via InterleavedQ4K::from_q4k which uses f16_to_f32_lut
    // Test that special f16 values are correctly converted

    // Zero: f16 bits 0x0000
    let zero_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&zero_data).expect("valid");
    assert_eq!(interleaved.d[0], 0.0, "f16 zero should convert to f32 zero");

    // One: f16 bits 0x3C00
    let mut one_data = vec![0u8; 144];
    one_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&one_data).expect("valid");
    assert!(
        (interleaved.d[0] - 1.0).abs() < 1e-6,
        "f16 1.0 should convert correctly"
    );

    // Negative one: f16 bits 0xBC00
    let mut neg_one_data = vec![0u8; 144];
    neg_one_data[0..2].copy_from_slice(&0xBC00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&neg_one_data).expect("valid");
    assert!(
        (interleaved.d[0] + 1.0).abs() < 1e-6,
        "f16 -1.0 should convert correctly"
    );
}

#[test]
fn test_f16_to_f32_lut_half_precision() {
    // 0.5 in f16: 0x3800
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        (interleaved.d[0] - 0.5).abs() < 1e-6,
        "f16 0.5 should convert correctly"
    );

    // 2.0 in f16: 0x4000
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        (interleaved.d[0] - 2.0).abs() < 1e-6,
        "f16 2.0 should convert correctly"
    );
}

#[test]
fn test_f16_to_f32_lut_subnormal_f16() {
    // Subnormal f16 value (very small, denormalized)
    // f16 subnormal: 0x0001 = smallest positive subnormal
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x0001u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(interleaved.d[0] > 0.0, "Subnormal f16 should be positive");
    assert!(
        interleaved.d[0] < 1e-6,
        "Subnormal f16 should be very small"
    );
}

#[test]
fn test_f16_to_f32_lut_infinity() {
    // Positive infinity in f16: 0x7C00
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(interleaved.d[0].is_infinite() && interleaved.d[0] > 0.0);

    // Negative infinity in f16: 0xFC00
    data[0..2].copy_from_slice(&0xFC00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(interleaved.d[0].is_infinite() && interleaved.d[0] < 0.0);
}

#[test]
fn test_f16_to_f32_lut_nan() {
    // NaN in f16: 0x7E00 (one possible NaN encoding)
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x7E00u16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        interleaved.d[0].is_nan(),
        "f16 NaN should convert to f32 NaN"
    );
}

#[test]
fn test_f16_to_f32_lut_max_f16() {
    // Max normal f16: 0x7BFF = 65504.0
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x7BFFu16.to_le_bytes());
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert!(
        (interleaved.d[0] - 65504.0).abs() < 1.0,
        "Max f16 should be ~65504"
    );
}

// =============================================================================
// Q8KSuperBlock::dequantize Tests
// =============================================================================

#[test]
fn test_q8k_superblock_dequantize_basic() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // All values should be approximately 1.0
    for (i, &v) in dequant.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Index {}: expected ~1.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_q8k_superblock_dequantize_zeros() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // All values should be exactly 0
    for (i, &v) in dequant.iter().enumerate() {
        assert!(v.abs() < 1e-6, "Index {}: expected 0, got {}", i, v);
    }
}

#[test]
fn test_q8k_superblock_dequantize_alternating() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = if i % 2 == 0 { 10.0 } else { -10.0 };
    }

    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // Check alternating signs are preserved
    for (i, &v) in dequant.iter().enumerate() {
        if i % 2 == 0 {
            assert!(v > 0.0, "Even index {} should be positive: {}", i, v);
        } else {
            assert!(v < 0.0, "Odd index {} should be negative: {}", i, v);
        }
    }
}

#[test]
fn test_q8k_superblock_dequantize_varying() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.5);
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // Check the trend is preserved (ascending values)
    for i in 1..256 {
        assert!(
            dequant[i] >= dequant[i - 1] - 1.0,
            "Values should be roughly ascending at index {}",
            i
        );
    }
}

#[test]
fn test_q8k_superblock_dequantize_extreme_values() {
    let mut values = [0.0f32; 256];
    values[0] = 1000.0;
    values[255] = -1000.0;

    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    // First value should be large positive (clamped to 127)
    assert!(dequant[0] > 500.0, "First value should be large positive");
    // Last value should be large negative (clamped to -127)
    assert!(dequant[255] < -500.0, "Last value should be large negative");
}

// =============================================================================
// Q8_0Block::relative_error Edge Cases
// =============================================================================

#[test]
fn test_q8_0block_relative_error_near_zero_max() {
    // Test the early return path when max_val < 1e-10
    let values = [1e-12f32; 32]; // Very small values
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    // Should return 0.0 when max_val is near zero
    assert_eq!(
        error, 0.0,
        "Relative error should be 0 for near-zero values"
    );
}

#[test]
fn test_q8_0block_relative_error_exactly_zero() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    assert_eq!(error, 0.0, "Relative error should be 0 for zero values");
}

#[test]
fn test_q8_0block_relative_error_mixed_small() {
    let mut values = [0.0f32; 32];
    values[0] = 1e-11; // Just at the threshold
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    // Should return 0.0 since max is < 1e-10
    assert_eq!(error, 0.0);
}

#[test]
fn test_q8_0block_relative_error_normal() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = Q8_0Block::quantize(&values);
    let error = block.relative_error(&values);

    // Should be a small but non-zero relative error
    assert!(
        error > 0.0,
        "Normal values should have some quantization error"
    );
    assert!(error < 0.1, "Relative error should be small: {}", error);
}

#[test]
fn test_q8_0block_quantization_error_basic() {
    let values = [5.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);

    // Quantization error should be small
    assert!(error < 0.1, "Quantization error should be small: {}", error);
}

// =============================================================================
// SimdBackend Display Trait Tests
// =============================================================================

#[test]
fn test_simd_backend_display_avx2() {
    let backend = SimdBackend::Avx2;
    assert_eq!(format!("{}", backend), "AVX2");
}

#[test]
fn test_simd_backend_display_sse2() {
    let backend = SimdBackend::Sse2;
    assert_eq!(format!("{}", backend), "SSE2");
}

#[test]
fn test_simd_backend_display_neon() {
    let backend = SimdBackend::Neon;
    assert_eq!(format!("{}", backend), "NEON");
}

#[test]
fn test_simd_backend_display_scalar() {
    let backend = SimdBackend::Scalar;
    assert_eq!(format!("{}", backend), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend = SimdBackend::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_equality() {
    assert_eq!(SimdBackend::Avx2, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Sse2);
    assert_ne!(SimdBackend::Neon, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_clone() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_copy() {
    let backend = SimdBackend::Neon;
    let copied: SimdBackend = backend;
    assert_eq!(backend, copied);
}

// =============================================================================
// DequantStats Tests
// =============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_clone() {
    let mut stats = DequantStats::default();
    stats.blocks_processed = 100;
    stats.bytes_processed = 3200;
    stats.simd_backend = SimdBackend::Avx2;

    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 100);
    assert_eq!(cloned.bytes_processed, 3200);
    assert_eq!(cloned.simd_backend, SimdBackend::Avx2);
}

#[test]
fn test_dequant_stats_debug() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 1344,
        simd_backend: SimdBackend::Sse2,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("1344"));
    assert!(debug_str.contains("Sse2"));
}

// =============================================================================
// detect_simd_backend Tests
// =============================================================================

#[test]
fn test_detect_simd_backend_runs() {
    let backend = detect_simd_backend();
    // On x86_64 with AVX2 (which is the development machine), expect AVX2
    // On other platforms, just verify it returns a valid variant
    match backend {
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar => {},
    }
}

#[test]
fn test_detect_simd_backend_consistent() {
    // Multiple calls should return the same result
    let backend1 = detect_simd_backend();
    let backend2 = detect_simd_backend();
    assert_eq!(backend1, backend2);
}

// =============================================================================
// extract_scale_min Tests for Blocks 4-7 (Packed Format)
// =============================================================================

#[test]
fn test_extract_scale_min_block_5() {
    let mut scales = [0u8; 12];
    // Block 5 uses: d = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4)
    //               m = (scales[9] >> 4) | ((scales[5] >> 6) << 4)
    scales[1] = 0b11_000000; // high 2 bits = 3
    scales[5] = 0b10_000000; // high 2 bits = 2
    scales[9] = 0b0011_0101; // low=5, high=3

    let (s, m) = extract_scale_min(&scales, 5);
    // scale = 5 | (3 << 4) = 5 + 48 = 53
    assert_eq!(s, 53.0, "Block 5 scale");
    // min = 3 | (2 << 4) = 3 + 32 = 35
    assert_eq!(m, 35.0, "Block 5 min");
}

#[test]
fn test_extract_scale_min_block_6() {
    let mut scales = [0u8; 12];
    scales[2] = 0b01_000000; // high 2 bits = 1
    scales[6] = 0b11_000000; // high 2 bits = 3
    scales[10] = 0b1001_0010; // low=2, high=9

    let (s, m) = extract_scale_min(&scales, 6);
    // scale = 2 | (1 << 4) = 2 + 16 = 18
    assert_eq!(s, 18.0, "Block 6 scale");
    // min = 9 | (3 << 4) = 9 + 48 = 57
    assert_eq!(m, 57.0, "Block 6 min");
}

#[test]
fn test_extract_scale_min_block_7() {
    let mut scales = [0u8; 12];
    scales[3] = 0b10_000000; // high 2 bits = 2
    scales[7] = 0b01_000000; // high 2 bits = 1
    scales[11] = 0b0100_1111; // low=15, high=4

    let (s, m) = extract_scale_min(&scales, 7);
    // scale = 15 | (2 << 4) = 15 + 32 = 47
    assert_eq!(s, 47.0, "Block 7 scale");
    // min = 4 | (1 << 4) = 4 + 16 = 20
    assert_eq!(m, 20.0, "Block 7 min");
}

include!("extract_scale_02.rs");
include!("quantize_activations_04.rs");
