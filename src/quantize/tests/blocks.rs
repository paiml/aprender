//! T-COV-95 Extended Coverage: quantize/mod.rs
//!
//! Targets: Additional dequantization coverage, block size validation,
//! error paths, edge cases with extreme values.

use crate::quantize::{
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q5_1, dequantize_q8_0,
    dequantize_q8_blocks, quantize_activations_q8k_into, quantize_to_q8_blocks, BLOCK_SIZE, QK_K,
};

// ============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks coverage
// ============================================================================

#[test]
fn test_q8_blocks_single_block() {
    let values = vec![1.0f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test value should be present");
    assert_eq!(blocks.len(), 1);
}

#[test]
fn test_q8_blocks_multiple_blocks() {
    let values = vec![0.5f32; 96]; // 3 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    let blocks = result.expect("test value should be present");
    assert_eq!(blocks.len(), 3);
}

#[test]
fn test_q8_blocks_roundtrip() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);

    assert_eq!(dequantized.len(), values.len());
    // Check approximate match (quantization introduces error)
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.1, "orig={}, deq={}", orig, deq);
    }
}

#[test]
fn test_q8_blocks_zero_values() {
    let values = vec![0.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);

    for v in &dequantized {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_q8_blocks_positive_values() {
    let values = vec![1.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);

    for v in &dequantized {
        assert!(*v > 0.9);
    }
}

#[test]
fn test_q8_blocks_negative_values() {
    let values = vec![-1.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);

    for v in &dequantized {
        assert!(*v < -0.9);
    }
}

#[test]
fn test_q8_blocks_mixed_values() {
    let mut values = vec![0.0f32; 32];
    for i in 0..32 {
        values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);

    for (i, v) in dequantized.iter().enumerate() {
        if i % 2 == 0 {
            assert!(*v > 0.5, "i={}, v={}", i, v);
        } else {
            assert!(*v < -0.5, "i={}, v={}", i, v);
        }
    }
}

#[test]
fn test_q8_blocks_empty_dequant() {
    let blocks: Vec<crate::quantize::Q8_0Block> = vec![];
    let dequantized = dequantize_q8_blocks(&blocks);
    assert!(dequantized.is_empty());
}

// ============================================================================
// dequantize_q4_0 additional coverage
// ============================================================================

#[test]
fn test_dequant_q4_0_varied_scales() {
    // Q4_0: 18 bytes per block (2 byte scale + 16 bytes quants)
    let mut data = vec![0u8; 36]; // 2 blocks

    // Block 1: scale = 0.5 in f16
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5 in f16

    // Block 2: scale = 2.0 in f16
    data[18..20].copy_from_slice(&0x4000u16.to_le_bytes()); // 2.0 in f16

    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 64);
}

#[test]
fn test_dequant_q4_0_max_quant_values() {
    let mut data = vec![0u8; 18];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    // Set all quant nibbles to max (0xF)
    for i in 2..18 {
        data[i] = 0xFF;
    }
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    // All values should be non-zero with max quants
    assert!(values.iter().any(|&v| v != 0.0));
}

#[test]
fn test_dequant_q4_0_zero_scale() {
    let mut data = vec![0u8; 18];
    data[0..2].copy_from_slice(&0x0000u16.to_le_bytes()); // scale = 0
    // Non-zero quants, but scale=0 should give zeros
    for i in 2..18 {
        data[i] = 0x55;
    }
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    for v in values {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

// ============================================================================
// dequantize_q8_0 additional coverage
// ============================================================================

#[test]
fn test_dequant_q8_0_varied_quants() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0

    // Set alternating positive/negative quants
    for i in 2..34 {
        data[i] = if (i - 2) % 2 == 0 { 64 } else { 192 }; // 64 = +64, 192 = -64
    }

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");

    // Check alternating signs
    for (i, v) in values.iter().enumerate() {
        if i % 2 == 0 {
            assert!(*v > 0.0, "i={}, v={}", i, v);
        } else {
            assert!(*v < 0.0, "i={}, v={}", i, v);
        }
    }
}

#[test]
fn test_dequant_q8_0_large_scale() {
    let mut data = vec![0u8; 34];
    // Large scale: 100.0 in f16 (approximately 0x5640)
    data[0..2].copy_from_slice(&0x5640u16.to_le_bytes());
    // Small quant value
    data[2] = 1; // +1

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    // First value should be approximately 100.0
    assert!(values[0] > 50.0);
}

// ============================================================================
// dequantize_q4_1 additional coverage
// ============================================================================

#[test]
fn test_dequant_q4_1_with_min() {
    // Q4_1: 20 bytes per block (2 byte delta + 2 byte min + 16 bytes quants)
    let mut data = vec![0u8; 20];
    // delta = 1.0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // min = -2.0 (0xC000 in f16)
    data[2..4].copy_from_slice(&0xC000u16.to_le_bytes());

    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequant_q4_1_zero_min() {
    let mut data = vec![0u8; 20];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // delta = 1.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // min = 0

    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
}

// ============================================================================
// dequantize_q5_0 additional coverage
// ============================================================================

#[test]
fn test_dequant_q5_0_with_high_bits() {
    // Q5_0: 22 bytes per block
    let mut data = vec![0u8; 22];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    // Set high bits
    data[2..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    // Set low quants
    for i in 6..22 {
        data[i] = 0x55;
    }

    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequant_q5_0_zero_high_bits() {
    let mut data = vec![0u8; 22];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    // All high bits zero
    data[2..6].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
}

// ============================================================================
// dequantize_q5_1 additional coverage
// ============================================================================

#[test]
fn test_dequant_q5_1_full_range() {
    // Q5_1: 24 bytes per block
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // delta = 1.0
    data[2..4].copy_from_slice(&0x4000u16.to_le_bytes()); // min = 2.0
    // High bits
    data[4..8].copy_from_slice(&[0xAA, 0xAA, 0xAA, 0xAA]);
    // Low quants
    for i in 8..24 {
        data[i] = 0x33;
    }

    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 32);
}

// ============================================================================
// quantize_activations_q8k_into edge cases
// ============================================================================

#[test]
fn test_q8k_into_max_positive() {
    let activations = vec![f32::MAX / 2.0; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    // Scale should be very large
    assert!(scales[0] > 1e30);
}

#[test]
fn test_q8k_into_max_negative() {
    let activations = vec![f32::MIN / 2.0; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_q8k_into_alternating() {
    let mut activations = vec![0.0f32; 256];
    for i in 0..256 {
        activations[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());

    // Quants should alternate in sign
    for i in 0..256 {
        if i % 2 == 0 {
            assert!(quants[i] > 0, "i={}, quant={}", i, quants[i]);
        } else {
            assert!(quants[i] < 0, "i={}, quant={}", i, quants[i]);
        }
    }
}

#[test]
fn test_q8k_into_linear_ramp() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());

    // Quants should be monotonically increasing
    for i in 1..256 {
        assert!(
            quants[i] >= quants[i - 1] || quants[i - 1] == 127,
            "i={}, prev={}, curr={}",
            i,
            quants[i - 1],
            quants[i]
        );
    }
}

#[test]
fn test_q8k_into_four_superblocks() {
    let activations = vec![0.25f32; 1024]; // 4 superblocks
    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 1024];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());

    // All scales should be similar
    for s in &scales {
        assert!((*s - scales[0]).abs() < 0.001);
    }
}

// ============================================================================
// Constants verification
// ============================================================================

#[test]
fn test_block_size_is_32() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_is_256() {
    assert_eq!(QK_K, 256);
}

#[test]
fn test_qk_k_is_multiple_of_block_size() {
    assert_eq!(QK_K % BLOCK_SIZE, 0);
    assert_eq!(QK_K / BLOCK_SIZE, 8);
}

// ============================================================================
// Block boundary tests
// ============================================================================

#[test]
fn test_dequant_q4_0_at_block_boundary() {
    // Exactly 3 blocks
    let data = vec![0u8; 54];
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 96);
}

#[test]
fn test_dequant_q8_0_at_block_boundary() {
    // Exactly 3 blocks
    let data = vec![0u8; 102];
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 96);
}

#[test]
fn test_dequant_q4_1_at_block_boundary() {
    // Exactly 3 blocks
    let data = vec![0u8; 60];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 96);
}

#[test]
fn test_dequant_q5_0_at_block_boundary() {
    // Exactly 3 blocks
    let data = vec![0u8; 66];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 96);
}

#[test]
fn test_dequant_q5_1_at_block_boundary() {
    // Exactly 3 blocks
    let data = vec![0u8; 72];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 96);
}
