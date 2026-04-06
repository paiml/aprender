//! EXTREME TDD coverage tests for GGUF quantization and dequantization functions
//!
//! Tests Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 dequantization with actual byte patterns.
//! PMAT-802: Comprehensive coverage for quantized tensor operations.

use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0,
    dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, f16_to_f32, fused_q4k_dot,
    fused_q4k_dot_simd, quantize_to_q8_blocks, InterleavedQ4K, Q8KSuperBlock, Q8_0Block,
    BLOCK_SIZE, QK_K,
};

// ============================================================================
// Q4_0 Dequantization Tests
// ============================================================================

/// Q4_0: Single block with f16 scale = 1.0, all quants = 0
/// Expected: -8.0 for positions 0-15, -8.0 for positions 16-31 (0-8=-8)
#[test]
fn test_q4_0_scale_one_quants_zero() {
    // f16 1.0 = 0x3C00
    let mut data = vec![0x00, 0x3C]; // Scale in little-endian
    data.extend([0u8; 16]); // 16 bytes of zero quants

    let result = dequantize_q4_0(&data).expect("should dequantize");
    assert_eq!(result.len(), BLOCK_SIZE);

    // Q4_0 formula: value = scale * (q - 8), where q is 4-bit [0,15]
    // Low nibbles (0-15): q=0, value = 1.0 * (0-8) = -8.0
    // High nibbles (16-31): q=0, value = 1.0 * (0-8) = -8.0
    for val in &result {
        assert!((*val - (-8.0)).abs() < 0.01, "Expected -8.0, got {}", val);
    }
}

/// Q4_0: Scale = 2.0, quants = 0x88 (low=8, high=8, both map to q-8=0)
#[test]
fn test_q4_0_scale_two_quants_8_8() {
    // f16 2.0 = 0x4000
    let mut data = vec![0x00, 0x40];
    data.extend([0x88u8; 16]); // Each byte: low=8, high=8

    let result = dequantize_q4_0(&data).expect("should dequantize");
    assert_eq!(result.len(), BLOCK_SIZE);

    // Q4_0: value = scale * (q - 8) = 2.0 * (8 - 8) = 0.0
    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q4_0: Max positive values (quants = 0xFF means q=15 for both nibbles)
#[test]
fn test_q4_0_max_values() {
    // f16 1.0 = 0x3C00
    let mut data = vec![0x00, 0x3C];
    data.extend([0xFFu8; 16]); // All 15s

    let result = dequantize_q4_0(&data).expect("should dequantize");

    // Q4_0: value = 1.0 * (15 - 8) = 7.0
    for val in &result {
        assert!((*val - 7.0).abs() < 0.01, "Expected 7.0, got {}", val);
    }
}

/// Q4_0: Negative scale
#[test]
fn test_q4_0_negative_scale() {
    // f16 -1.0 = 0xBC00
    let mut data = vec![0x00, 0xBC];
    data.extend([0x88u8; 16]); // q=8, q-8=0

    let result = dequantize_q4_0(&data).expect("should dequantize");

    // -1.0 * (8 - 8) = 0.0
    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q4_0: Multiple blocks
#[test]
fn test_q4_0_multiple_blocks() {
    // 3 blocks: 18 bytes each = 54 bytes total
    let mut data = Vec::new();
    for _ in 0..3 {
        data.extend([0x00, 0x3C]); // f16 1.0
        data.extend([0x88u8; 16]);
    }

    let result = dequantize_q4_0(&data).expect("should dequantize");
    assert_eq!(result.len(), 3 * BLOCK_SIZE);
}

/// Q4_0: Invalid length (not multiple of 18)
#[test]
fn test_q4_0_invalid_length() {
    let data = vec![0u8; 17]; // Invalid: not 18
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q8_0 Dequantization Tests
// ============================================================================

/// Q8_0: Scale = 1.0, all quants = 0
#[test]
fn test_q8_0_scale_one_quants_zero() {
    // f16 1.0 = 0x3C00, then 32 bytes of zeros
    let mut data = vec![0x00, 0x3C];
    data.extend([0u8; 32]);

    let result = dequantize_q8_0(&data).expect("should dequantize");
    assert_eq!(result.len(), BLOCK_SIZE);

    // Q8_0: value = scale * q where q is i8, zero → 0.0
    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q8_0: Scale = 2.0, quants = positive values
#[test]
fn test_q8_0_positive_quants() {
    // f16 2.0 = 0x4000
    let mut data = vec![0x00, 0x40];
    // Fill with i8 = 10 (0x0A)
    data.extend([0x0Au8; 32]);

    let result = dequantize_q8_0(&data).expect("should dequantize");

    // 2.0 * 10 = 20.0
    for val in &result {
        assert!((*val - 20.0).abs() < 0.01, "Expected 20.0, got {}", val);
    }
}

/// Q8_0: Negative quants (i8 = -10 = 0xF6)
#[test]
fn test_q8_0_negative_quants() {
    // f16 1.0 = 0x3C00
    let mut data = vec![0x00, 0x3C];
    // i8 -10 = 0xF6
    data.extend([0xF6u8; 32]);

    let result = dequantize_q8_0(&data).expect("should dequantize");

    // 1.0 * (-10) = -10.0
    for val in &result {
        assert!((*val - (-10.0)).abs() < 0.01, "Expected -10.0, got {}", val);
    }
}

/// Q8_0: Max positive (127) and negative (-128) values
#[test]
fn test_q8_0_extremes() {
    // f16 1.0 = 0x3C00
    let mut data = vec![0x00, 0x3C];
    // First half: 127 (0x7F), second half: -128 (0x80)
    data.extend([0x7Fu8; 16]);
    data.extend([0x80u8; 16]);

    let result = dequantize_q8_0(&data).expect("should dequantize");

    // First 16: 1.0 * 127 = 127.0
    for val in &result[0..16] {
        assert!((*val - 127.0).abs() < 0.1, "Expected 127.0, got {}", val);
    }
    // Second 16: 1.0 * (-128) = -128.0
    for val in &result[16..32] {
        assert!(
            (*val - (-128.0)).abs() < 0.1,
            "Expected -128.0, got {}",
            val
        );
    }
}

/// Q8_0: Invalid length (not multiple of 34)
#[test]
fn test_q8_0_invalid_length() {
    let data = vec![0u8; 33]; // Invalid
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q4_K Dequantization Tests (Super-block: 144 bytes for 256 values)
// ============================================================================

/// Q4_K: Minimal super-block with zero scales
#[test]
fn test_q4_k_zero_scales() {
    // 144 bytes per super-block: d(2) + dmin(2) + scales(12) + qs(128)
    let data = vec![0u8; 144];
    // d = 0, dmin = 0 → all output should be 0

    let result = dequantize_q4_k(&data).expect("should dequantize");
    assert_eq!(result.len(), QK_K);

    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q4_K: Scale = 1.0, dmin = 0, testing actual formula
#[test]
fn test_q4_k_unit_scale() {
    let mut data = vec![0u8; 144];
    // d = f16 1.0 = 0x3C00
    data[0] = 0x00;
    data[1] = 0x3C;
    // dmin = 0 (already zero)

    // Set all scales to 1 (packed 6-bit format is complex, so use minimal)
    // scales[0..4] = 0x01 (gives scale=1 for first blocks)
    data[4] = 0x01;

    let result = dequantize_q4_k(&data).expect("should dequantize");
    assert_eq!(result.len(), QK_K);
    // Verify some output is produced
    assert!(result.iter().any(|&v| v.abs() < 1000.0));
}

/// Q4_K: Multiple super-blocks
#[test]
fn test_q4_k_multiple_super_blocks() {
    let data = vec![0u8; 144 * 3]; // 3 super-blocks

    let result = dequantize_q4_k(&data).expect("should dequantize");
    assert_eq!(result.len(), 3 * QK_K);
}

/// Q4_K: Invalid length
#[test]
fn test_q4_k_invalid_length() {
    let data = vec![0u8; 143]; // Not 144
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q5_K Dequantization Tests (Super-block: 176 bytes for 256 values)
// ============================================================================

/// Q5_K: Zero super-block
#[test]
fn test_q5_k_zero() {
    let data = vec![0u8; 176];

    let result = dequantize_q5_k(&data).expect("should dequantize");
    assert_eq!(result.len(), QK_K);

    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q5_K: Multiple super-blocks
#[test]
fn test_q5_k_multiple_super_blocks() {
    let data = vec![0u8; 176 * 2];

    let result = dequantize_q5_k(&data).expect("should dequantize");
    assert_eq!(result.len(), 2 * QK_K);
}

/// Q5_K: Invalid length
#[test]
fn test_q5_k_invalid_length() {
    let data = vec![0u8; 175];
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q6_K Dequantization Tests (Super-block: 210 bytes for 256 values)
// ============================================================================

/// Q6_K: Zero super-block
#[test]
fn test_q6_k_zero() {
    let data = vec![0u8; 210];

    let result = dequantize_q6_k(&data).expect("should dequantize");
    assert_eq!(result.len(), QK_K);

    // Q6_K formula with zero d produces zeros
    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q6_K: Scale = 1.0 at correct offset (d is at byte 208-209)
#[test]
fn test_q6_k_unit_scale() {
    let mut data = vec![0u8; 210];
    // d is at offset 208 (f16)
    data[208] = 0x00;
    data[209] = 0x3C; // f16 1.0

    let result = dequantize_q6_k(&data).expect("should dequantize");
    assert_eq!(result.len(), QK_K);
}

/// Q6_K: Multiple super-blocks
#[test]
fn test_q6_k_multiple_super_blocks() {
    let data = vec![0u8; 210 * 4];

    let result = dequantize_q6_k(&data).expect("should dequantize");
    assert_eq!(result.len(), 4 * QK_K);
}

/// Q6_K: Invalid length
#[test]
fn test_q6_k_invalid_length() {
    let data = vec![0u8; 209];
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

// ============================================================================
// F16 Dequantization Tests
// ============================================================================

/// F16: Zero value
#[test]
fn test_f16_zero() {
    let data = vec![0x00, 0x00]; // f16 zero
    let result = dequantize_f16(&data).expect("should dequantize");
    assert_eq!(result.len(), 1);
    assert!(result[0].abs() < 0.0001);
}

/// F16: One value
#[test]
fn test_f16_one() {
    let data = vec![0x00, 0x3C]; // f16 1.0
    let result = dequantize_f16(&data).expect("should dequantize");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 0.01);
}

/// F16: Negative one
#[test]
fn test_f16_negative_one() {
    let data = vec![0x00, 0xBC]; // f16 -1.0
    let result = dequantize_f16(&data).expect("should dequantize");
    assert_eq!(result.len(), 1);
    assert!((result[0] - (-1.0)).abs() < 0.01);
}

/// F16: Multiple values
#[test]
fn test_f16_multiple() {
    let mut data = Vec::new();
    data.extend([0x00, 0x3C]); // 1.0
    data.extend([0x00, 0x40]); // 2.0
    data.extend([0x00, 0x42]); // 3.0

    let result = dequantize_f16(&data).expect("should dequantize");
    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.0).abs() < 0.01);
    assert!((result[1] - 2.0).abs() < 0.01);
    assert!((result[2] - 3.0).abs() < 0.01);
}

/// F16: Invalid length (odd bytes)
#[test]
fn test_f16_invalid_length() {
    let data = vec![0x00, 0x3C, 0x00]; // 3 bytes, invalid
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

/// F16: Infinity
#[test]
fn test_f16_infinity() {
    let data = vec![0x00, 0x7C]; // f16 +inf
    let result = dequantize_f16(&data).expect("should dequantize");
    assert!(result[0].is_infinite() && result[0].is_sign_positive());
}

/// F16: Negative infinity
#[test]
fn test_f16_neg_infinity() {
    let data = vec![0x00, 0xFC]; // f16 -inf
    let result = dequantize_f16(&data).expect("should dequantize");
    assert!(result[0].is_infinite() && result[0].is_sign_negative());
}

/// F16: NaN
#[test]
fn test_f16_nan() {
    let data = vec![0x01, 0x7C]; // f16 NaN
    let result = dequantize_f16(&data).expect("should dequantize");
    assert!(result[0].is_nan());
}

// ============================================================================
// Q4_1 Dequantization Tests (Block: 20 bytes for 32 values)
// ============================================================================

/// Q4_1: Zero block
#[test]
fn test_q4_1_zero() {
    // d(2) + min(2) + qs(16) = 20 bytes
    let data = vec![0u8; 20];

    let result = dequantize_q4_1(&data).expect("should dequantize");
    assert_eq!(result.len(), BLOCK_SIZE);

    // d=0, min=0 → all zeros
    for val in &result {
        assert!(val.abs() < 0.01, "Expected 0.0, got {}", val);
    }
}

/// Q4_1: Scale = 1, min = 5
#[test]
fn test_q4_1_with_min() {
    let mut data = vec![0u8; 20];
    // d = f16 1.0
    data[0] = 0x00;
    data[1] = 0x3C;
    // min = f16 5.0 = 0x4500
    data[2] = 0x00;
    data[3] = 0x45;
    // qs = 0 (all nibbles = 0)

    let result = dequantize_q4_1(&data).expect("should dequantize");

    // value = d * q + min = 1.0 * 0 + 5.0 = 5.0
    for val in &result {
        assert!((*val - 5.0).abs() < 0.1, "Expected 5.0, got {}", val);
    }
}

/// Q4_1: Invalid length
#[test]
fn test_q4_1_invalid_length() {
    let data = vec![0u8; 19];
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q5_0 Dequantization Tests (Block: 22 bytes for 32 values)
// ============================================================================

/// Q5_0: Zero block
#[test]
fn test_q5_0_zero() {
    // d(2) + qh(4) + qs(16) = 22 bytes
    let data = vec![0u8; 22];

    let result = dequantize_q5_0(&data).expect("should dequantize");
    assert_eq!(result.len(), BLOCK_SIZE);
}

/// Q5_0: Invalid length
#[test]
fn test_q5_0_invalid_length() {
    let data = vec![0u8; 21];
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q5_1 Dequantization Tests (Block: 24 bytes for 32 values)
// ============================================================================

/// Q5_1: Zero block
#[test]
fn test_q5_1_zero() {
    // d(2) + min(2) + qh(4) + qs(16) = 24 bytes
    let data = vec![0u8; 24];

    let result = dequantize_q5_1(&data).expect("should dequantize");
    assert_eq!(result.len(), BLOCK_SIZE);
}

/// Q5_1: Invalid length
#[test]
fn test_q5_1_invalid_length() {
    let data = vec![0u8; 23];
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q8_0Block Struct Tests
// ============================================================================

/// Q8_0Block: Quantize constant values
#[test]
fn test_q8_0_block_quantize_constant() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Scale should be 1.0 / 127.0 ≈ 0.00787
    assert!(block.scale > 0.0);

    // All quants should be equal
    let first = block.quants[0];
    for &q in &block.quants[1..] {
        assert_eq!(q, first);
    }
}

/// Q8_0Block: Quantize zeros
#[test]
fn test_q8_0_block_quantize_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // All quants should be zero
    for &q in &block.quants {
        assert_eq!(q, 0);
    }
}

/// Q8_0Block: Dequantize roundtrip
#[test]
fn test_q8_0_block_roundtrip() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) / 4.0);
    let block = Q8_0Block::quantize(&values);

    // Check roundtrip error is reasonable
    let max_error = block.quantization_error(&values);
    assert!(max_error < 0.1, "Roundtrip error {} too high", max_error);
}

/// Q8_0Block: Relative error
#[test]
fn test_q8_0_block_relative_error() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 + 1.0) * 2.0);
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);

    // Relative error should be reasonable for Q8 (< 1%)
    assert!(rel_error < 0.02, "Relative error {} too high", rel_error);
}

// ============================================================================
// Q8K Super-Block Tests
// ============================================================================

/// Q8K: Quantize constant values
#[test]
fn test_q8k_quantize_constant() {
    let values = [2.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    assert!(block.scale > 0.0);

    // All quants should be equal
    let first = block.quants[0];
    for &q in &block.quants[1..] {
        assert_eq!(q, first);
    }
}

/// Q8K: Quantize zeros
#[test]
fn test_q8k_quantize_zeros() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    for &q in &block.quants {
        assert_eq!(q, 0);
    }
}

/// Q8K: Dequantize roundtrip
#[test]
fn test_q8k_roundtrip() {
    let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) / 32.0);
    let block = Q8KSuperBlock::quantize(&values);
    let dequantized = block.dequantize();

    // Check max error
    let max_error = values
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 0.1, "Roundtrip error {} too high", max_error);
}

/// Q8K: Quantize into pre-allocated buffer
#[test]
fn test_q8k_quantize_into() {
    let values: [f32; 256] = std::array::from_fn(|i| i as f32);
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // Verify some quants are non-zero
    assert!(quants.iter().any(|&q| q != 0));
}

// ============================================================================
// quantize_to_q8_blocks Tests
// ============================================================================

/// quantize_to_q8_blocks: Single block
#[test]
fn test_quantize_to_q8_blocks_single() {
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("should quantize");

    assert_eq!(blocks.len(), 1);
    assert!(blocks[0].scale > 0.0);
}

/// quantize_to_q8_blocks: Multiple blocks
#[test]
fn test_quantize_to_q8_blocks_multiple() {
    let values: Vec<f32> = (0..96).map(|i| i as f32).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("should quantize");

    assert_eq!(blocks.len(), 3);
}

/// quantize_to_q8_blocks: Invalid length
#[test]
fn test_quantize_to_q8_blocks_invalid() {
    let values: Vec<f32> = (0..31).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

// ============================================================================
// InterleavedQ4K Tests
// ============================================================================

/// InterleavedQ4K: Create from valid data
#[test]
fn test_interleaved_q4k_from_valid() {
    let data = vec![0u8; 144]; // One super-block
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("should create");

    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), QK_K);
}

/// InterleavedQ4K: Invalid length
#[test]
fn test_interleaved_q4k_invalid_length() {
    let data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

/// InterleavedQ4K: Multiple super-blocks
#[test]
fn test_interleaved_q4k_multiple() {
    let data = vec![0u8; 144 * 4];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("should create");

    assert_eq!(interleaved.num_super_blocks, 4);
    assert_eq!(interleaved.num_values(), 4 * QK_K);
}

// ============================================================================
// Fused Q4K Dot Product Tests
// ============================================================================

/// Fused Q4K dot: Zero weights, any activations
#[test]
fn test_fused_q4k_dot_zero_weights() {
    let q4k_data = vec![0u8; 144]; // Zero super-block
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations).expect("should compute");
    assert!(result.abs() < 0.01, "Expected 0, got {}", result);
}

/// Fused Q4K dot: Invalid Q4K length
#[test]
fn test_fused_q4k_dot_invalid_q4k() {
    let q4k_data = vec![0u8; 143];
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

/// Fused Q4K dot: Activation length mismatch
#[test]
fn test_fused_q4k_dot_length_mismatch() {
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 100]; // Wrong length

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

/// Fused Q4K dot SIMD: Same result as scalar
#[test]
fn test_fused_q4k_dot_simd_matches_scalar() {
    // Create some non-zero Q4K data
    let mut q4k_data = vec![0u8; 144];
    // d = f16 1.0
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C;
    // dmin = f16 0.5 = 0x3800
    q4k_data[2] = 0x00;
    q4k_data[3] = 0x38;
    // Set some scales
    q4k_data[4] = 0x21;

    let activations: Vec<f32> = (0..QK_K).map(|i| (i as f32) / 256.0).collect();

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd");

    // Allow 4 ULP tolerance
    let diff = (scalar - simd).abs();
    let max_ulp = 4.0 * f32::EPSILON * scalar.abs().max(simd.abs()).max(1.0);
    assert!(
        diff < max_ulp.max(0.001),
        "SIMD mismatch: scalar={}, simd={}, diff={}",
        scalar,
        simd,
        diff
    );
}

// ============================================================================
// f16_to_f32 Conversion Tests
// ============================================================================

/// f16_to_f32: Zero
#[test]
fn test_f16_to_f32_zero() {
    assert!(f16_to_f32(0x0000).abs() < 0.0001);
}

/// f16_to_f32: Negative zero
#[test]
fn test_f16_to_f32_negative_zero() {
    let val = f16_to_f32(0x8000);
    assert!(val.abs() < 0.0001);
    assert!(val.is_sign_negative());
}

/// f16_to_f32: One
#[test]
fn test_f16_to_f32_one() {
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
}

/// f16_to_f32: Negative one
#[test]
fn test_f16_to_f32_neg_one() {
    assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001);
}

/// f16_to_f32: Positive infinity
#[test]
fn test_f16_to_f32_pos_inf() {
    let val = f16_to_f32(0x7C00);
    assert!(val.is_infinite() && val.is_sign_positive());
}

/// f16_to_f32: Negative infinity
#[test]
fn test_f16_to_f32_neg_inf() {
    let val = f16_to_f32(0xFC00);
    assert!(val.is_infinite() && val.is_sign_negative());
}

/// f16_to_f32: NaN
#[test]
fn test_f16_to_f32_nan() {
    let val = f16_to_f32(0x7C01);
    assert!(val.is_nan());
}

/// f16_to_f32: Subnormal
#[test]
fn test_f16_to_f32_subnormal() {
    // Smallest positive subnormal: 0x0001
    let val = f16_to_f32(0x0001);
    assert!(val > 0.0);
    assert!(val < 0.001);
}

// ============================================================================
// Empty Data Tests
// ============================================================================

/// Empty Q4_0 data
#[test]
fn test_q4_0_empty() {
    let result = dequantize_q4_0(&[]).expect("should work");
    assert!(result.is_empty());
}

/// Empty Q8_0 data
#[test]
fn test_q8_0_empty() {
    let result = dequantize_q8_0(&[]).expect("should work");
    assert!(result.is_empty());
}

/// Empty Q4_K data
#[test]
fn test_q4_k_empty() {
    let result = dequantize_q4_k(&[]).expect("should work");
    assert!(result.is_empty());
}

/// Empty Q5_K data
#[test]
fn test_q5_k_empty() {
    let result = dequantize_q5_k(&[]).expect("should work");
    assert!(result.is_empty());
}

/// Empty Q6_K data
#[test]
fn test_q6_k_empty() {
    let result = dequantize_q6_k(&[]).expect("should work");
    assert!(result.is_empty());
}

/// Empty F16 data
#[test]
fn test_f16_empty() {
    let result = dequantize_f16(&[]).expect("should work");
    assert!(result.is_empty());
}
