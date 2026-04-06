
#[test]
fn test_quantize_activations_q8k_into_zeros_cov() {
    let activations = vec![0.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    // Should succeed even with zeros
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: quantize_activations_q8_0
// =========================================================================

// =========================================================================
// PMAT-097: Reference Comparison Tests for Q4_1, Q5_0, Q5_1
// These tests verify CORRECTNESS of dequantized values, not just code coverage.
// Based on GGML/llama.cpp reference implementation and candle layout specification.
// =========================================================================

/// Q4_1 Reference Test: Verify candle layout with known values
/// GGUF candle layout for Q4_1:
/// - Positions 0-15: low nibbles (byte & 0xF) from bytes 0-15
/// - Positions 16-31: high nibbles (byte >> 4) from bytes 0-15
///
/// Formula: value = scale * nibble + min
#[test]
fn test_q4_1_candle_layout_correctness() {
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes quants = 20 bytes
    let mut block = vec![0u8; 20];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // Set quants with known pattern:
    // byte[0] = 0x10 -> low=0, high=1
    // byte[1] = 0x32 -> low=2, high=3
    // byte[2] = 0x54 -> low=4, high=5
    // byte[3] = 0x76 -> low=6, high=7
    block[4] = 0x10; // low=0, high=1
    block[5] = 0x32; // low=2, high=3
    block[6] = 0x54; // low=4, high=5
    block[7] = 0x76; // low=6, high=7
    block[8] = 0x98; // low=8, high=9
    block[9] = 0xBA; // low=10 (0xA), high=11 (0xB)
    block[10] = 0xDC; // low=12 (0xC), high=13 (0xD)
    block[11] = 0xFE; // low=14 (0xE), high=15 (0xF)
                      // Remaining bytes are zeros

    let result = dequantize_q4_1(&block).expect("dequantization failed");

    assert_eq!(result.len(), 32);

    // Verify candle layout - low nibbles at positions 0-15
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "pos 0: expected 0.0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 2.0).abs() < 1e-5,
        "pos 1: expected 2.0, got {}",
        result[1]
    );
    assert!(
        (result[2] - 4.0).abs() < 1e-5,
        "pos 2: expected 4.0, got {}",
        result[2]
    );
    assert!(
        (result[3] - 6.0).abs() < 1e-5,
        "pos 3: expected 6.0, got {}",
        result[3]
    );
    assert!(
        (result[4] - 8.0).abs() < 1e-5,
        "pos 4: expected 8.0, got {}",
        result[4]
    );
    assert!(
        (result[5] - 10.0).abs() < 1e-5,
        "pos 5: expected 10.0, got {}",
        result[5]
    );
    assert!(
        (result[6] - 12.0).abs() < 1e-5,
        "pos 6: expected 12.0, got {}",
        result[6]
    );
    assert!(
        (result[7] - 14.0).abs() < 1e-5,
        "pos 7: expected 14.0, got {}",
        result[7]
    );

    // Verify candle layout - high nibbles at positions 16-31
    assert!(
        (result[16] - 1.0).abs() < 1e-5,
        "pos 16: expected 1.0, got {}",
        result[16]
    );
    assert!(
        (result[17] - 3.0).abs() < 1e-5,
        "pos 17: expected 3.0, got {}",
        result[17]
    );
    assert!(
        (result[18] - 5.0).abs() < 1e-5,
        "pos 18: expected 5.0, got {}",
        result[18]
    );
    assert!(
        (result[19] - 7.0).abs() < 1e-5,
        "pos 19: expected 7.0, got {}",
        result[19]
    );
    assert!(
        (result[20] - 9.0).abs() < 1e-5,
        "pos 20: expected 9.0, got {}",
        result[20]
    );
    assert!(
        (result[21] - 11.0).abs() < 1e-5,
        "pos 21: expected 11.0, got {}",
        result[21]
    );
    assert!(
        (result[22] - 13.0).abs() < 1e-5,
        "pos 22: expected 13.0, got {}",
        result[22]
    );
    assert!(
        (result[23] - 15.0).abs() < 1e-5,
        "pos 23: expected 15.0, got {}",
        result[23]
    );
}

/// Q4_1 Reference Test: Verify scale and min application
#[test]
fn test_q4_1_scale_and_min_correctness() {
    let mut block = vec![0u8; 20];

    // Scale = 2.0, Min = 10.0
    block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(10.0).to_le_bytes());

    // byte[0] = 0x31 -> low=1, high=3
    block[4] = 0x31;

    let result = dequantize_q4_1(&block).expect("dequantization failed");

    // Position 0: 2.0 * 1 + 10.0 = 12.0
    assert!(
        (result[0] - 12.0).abs() < 0.1,
        "pos 0: expected 12.0, got {}",
        result[0]
    );

    // Position 16: 2.0 * 3 + 10.0 = 16.0
    assert!(
        (result[16] - 16.0).abs() < 0.1,
        "pos 16: expected 16.0, got {}",
        result[16]
    );
}

/// Q5_0 Reference Test: Verify candle layout with 5th bit
/// GGUF candle layout for Q5_0:
/// - Positions 0-15: (low nibble | (qh bit << 4)) - 16
/// - Positions 16-31: (high nibble | (qh bit << 4)) - 16
///
/// Formula: value = scale * ((nibble | high_bit) - 16)
#[test]
fn test_q5_0_candle_layout_correctness() {
    // Q5_0 block: 2 bytes scale + 4 bytes qh + 16 bytes quants = 22 bytes
    let mut block = vec![0u8; 22];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // qh = 0 (no high bits set) for simplicity
    block[2..6].copy_from_slice(&[0, 0, 0, 0]);

    // Set quants with pattern
    // With qh=0 and offset=-16, nibble 0 -> -16, nibble 8 -> -8, nibble 15 -> -1
    block[6] = 0x80; // low=0, high=8
    block[7] = 0xF1; // low=1, high=15

    let result = dequantize_q5_0(&block).expect("dequantization failed");

    assert_eq!(result.len(), 32);

    // Position 0: 1.0 * (0 | 0 - 16) = -16.0
    assert!(
        (result[0] - (-16.0)).abs() < 1e-5,
        "pos 0: expected -16.0, got {}",
        result[0]
    );

    // Position 1: 1.0 * (1 | 0 - 16) = -15.0
    assert!(
        (result[1] - (-15.0)).abs() < 1e-5,
        "pos 1: expected -15.0, got {}",
        result[1]
    );

    // Position 16: 1.0 * (8 | 0 - 16) = -8.0
    assert!(
        (result[16] - (-8.0)).abs() < 1e-5,
        "pos 16: expected -8.0, got {}",
        result[16]
    );

    // Position 17: 1.0 * (15 | 0 - 16) = -1.0
    assert!(
        (result[17] - (-1.0)).abs() < 1e-5,
        "pos 17: expected -1.0, got {}",
        result[17]
    );
}

/// Q5_0 Reference Test: Verify high bit handling
/// qh bits 0-15 apply to positions 0-15, bits 16-31 apply to positions 16-31
#[test]
fn test_q5_0_high_bit_correctness() {
    let mut block = vec![0u8; 22];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // qh = 0x00010001 -> bit 0 and bit 16 are set
    // This means positions 0 and 16 get +16 from the high bit
    block[2..6].copy_from_slice(&0x00010001u32.to_le_bytes());

    // Quants: byte[0] = 0x00 -> low=0, high=0
    block[6] = 0x00;

    let result = dequantize_q5_0(&block).expect("dequantization failed");

    // Position 0: nibble=0, high_bit=1 -> (0 | 16) - 16 = 0
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "pos 0: expected 0.0, got {}",
        result[0]
    );

    // Position 16: nibble=0, high_bit=1 -> (0 | 16) - 16 = 0
    assert!(
        (result[16] - 0.0).abs() < 1e-5,
        "pos 16: expected 0.0, got {}",
        result[16]
    );
}

/// Q5_1 Reference Test: Verify candle layout with scale and min
/// Formula: value = scale * (nibble | high_bit) + min
#[test]
fn test_q5_1_candle_layout_correctness() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes qh + 16 bytes quants = 24 bytes
    let mut block = vec![0u8; 24];

    // Scale = 1.0, Min = 0.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // qh = 0 (no high bits set)
    block[4..8].copy_from_slice(&[0, 0, 0, 0]);

    // Quants with known pattern
    block[8] = 0x10; // low=0, high=1
    block[9] = 0x32; // low=2, high=3

    let result = dequantize_q5_1(&block).expect("dequantization failed");

    assert_eq!(result.len(), 32);

    // Position 0: 1.0 * (0 | 0) + 0 = 0.0
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "pos 0: expected 0.0, got {}",
        result[0]
    );

    // Position 1: 1.0 * (2 | 0) + 0 = 2.0
    assert!(
        (result[1] - 2.0).abs() < 1e-5,
        "pos 1: expected 2.0, got {}",
        result[1]
    );

    // Position 16: 1.0 * (1 | 0) + 0 = 1.0
    assert!(
        (result[16] - 1.0).abs() < 1e-5,
        "pos 16: expected 1.0, got {}",
        result[16]
    );

    // Position 17: 1.0 * (3 | 0) + 0 = 3.0
    assert!(
        (result[17] - 3.0).abs() < 1e-5,
        "pos 17: expected 3.0, got {}",
        result[17]
    );
}

/// Q5_1 Reference Test: Verify scale, min, and high bit together
#[test]
fn test_q5_1_scale_min_high_bit_correctness() {
    let mut block = vec![0u8; 24];

    // Scale = 2.0, Min = 5.0
    block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(5.0).to_le_bytes());

    // qh = 0x00000001 -> only bit 0 set
    block[4..8].copy_from_slice(&0x00000001u32.to_le_bytes());

    // byte[0] = 0x21 -> low=1, high=2
    block[8] = 0x21;

    let result = dequantize_q5_1(&block).expect("dequantization failed");

    // Position 0: 2.0 * (1 | 16) + 5.0 = 2.0 * 17 + 5 = 39.0
    assert!(
        (result[0] - 39.0).abs() < 0.1,
        "pos 0: expected 39.0, got {}",
        result[0]
    );

    // Position 16: 2.0 * (2 | 0) + 5.0 = 2.0 * 2 + 5 = 9.0
    assert!(
        (result[16] - 9.0).abs() < 0.1,
        "pos 16: expected 9.0, got {}",
        result[16]
    );
}

/// Q4_0 Reference Test: Verify candle layout (for comparison - this one works)
/// Q4_0 uses signed offset: value = scale * (nibble - 8)
#[test]
fn test_q4_0_candle_layout_reference() {
    let mut block = vec![0u8; 18];

    // Scale = 1.0
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // Quants with known pattern
    block[2] = 0x80; // low=0, high=8
    block[3] = 0xF1; // low=1, high=15

    let result = dequantize_q4_0(&block).expect("dequantization failed");

    // Position 0: 1.0 * (0 - 8) = -8.0
    assert!(
        (result[0] - (-8.0)).abs() < 1e-5,
        "pos 0: expected -8.0, got {}",
        result[0]
    );

    // Position 1: 1.0 * (1 - 8) = -7.0
    assert!(
        (result[1] - (-7.0)).abs() < 1e-5,
        "pos 1: expected -7.0, got {}",
        result[1]
    );

    // Position 16: 1.0 * (8 - 8) = 0.0
    assert!(
        (result[16] - 0.0).abs() < 1e-5,
        "pos 16: expected 0.0, got {}",
        result[16]
    );

    // Position 17: 1.0 * (15 - 8) = 7.0
    assert!(
        (result[17] - 7.0).abs() < 1e-5,
        "pos 17: expected 7.0, got {}",
        result[17]
    );
}

/// Multi-block test: Verify Q4_1 works across block boundaries
#[test]
fn test_q4_1_multi_block_correctness() {
    // Two Q4_1 blocks = 40 bytes
    let mut data = vec![0u8; 40];

    // Block 0: scale=1.0, min=0.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data[4] = 0x21; // low=1, high=2

    // Block 1: scale=2.0, min=10.0
    data[20..22].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    data[22..24].copy_from_slice(&half::f16::from_f32(10.0).to_le_bytes());
    data[24] = 0x43; // low=3, high=4

    let result = dequantize_q4_1(&data).expect("dequantization failed");

    assert_eq!(result.len(), 64); // 2 blocks * 32 values

    // Block 0, Position 0: 1.0 * 1 + 0.0 = 1.0
    assert!(
        (result[0] - 1.0).abs() < 1e-5,
        "block 0 pos 0: expected 1.0, got {}",
        result[0]
    );

    // Block 0, Position 16: 1.0 * 2 + 0.0 = 2.0
    assert!(
        (result[16] - 2.0).abs() < 1e-5,
        "block 0 pos 16: expected 2.0, got {}",
        result[16]
    );

    // Block 1, Position 0: 2.0 * 3 + 10.0 = 16.0
    assert!(
        (result[32] - 16.0).abs() < 0.1,
        "block 1 pos 0: expected 16.0, got {}",
        result[32]
    );

    // Block 1, Position 16: 2.0 * 4 + 10.0 = 18.0
    assert!(
        (result[48] - 18.0).abs() < 0.1,
        "block 1 pos 16: expected 18.0, got {}",
        result[48]
    );
}
