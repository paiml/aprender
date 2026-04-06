
// ============== EXTREME TDD: Q5_1 Dequantization Tests ==============

#[test]
fn test_dequantize_q5_1_single_block() {
    // Q5_1 block: 24 bytes (2 scale + 2 min + 4 high bits + 16 quants)
    let mut data = Vec::new();

    // d = 1.0 (f16: 0x3C00)
    data.extend_from_slice(&0x3C00_u16.to_le_bytes());
    // min = 0.0 (f16: 0x0000)
    data.extend_from_slice(&0x0000_u16.to_le_bytes());
    // qh: 4 bytes of high bits (all zeros)
    data.extend_from_slice(&[0x00; 4]);
    // qs: 16 bytes of low 4 bits (all zeros)
    data.extend_from_slice(&[0x00; 16]);

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // All values should be d * q + min = 1.0 * 0 + 0.0 = 0.0
    for v in &result {
        assert!((v - 0.0).abs() < 1e-3);
    }
}

#[test]
fn test_dequantize_q5_1_with_min() {
    let mut data = Vec::new();

    // d = 0.0 (f16: 0x0000)
    data.extend_from_slice(&0x0000_u16.to_le_bytes());
    // min = 2.0 (f16: 0x4000)
    data.extend_from_slice(&0x4000_u16.to_le_bytes());
    // qh: 4 bytes of high bits (all zeros)
    data.extend_from_slice(&[0x00; 4]);
    // qs: 16 bytes of low 4 bits (all zeros)
    data.extend_from_slice(&[0x00; 16]);

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // All values should be d * q + min = 0 + 2.0 = 2.0
    for v in &result {
        assert!((v - 2.0).abs() < 1e-3);
    }
}

#[test]
fn test_dequantize_q5_1_with_high_bits() {
    let mut data = Vec::new();

    // d = 1.0 (f16: 0x3C00)
    data.extend_from_slice(&0x3C00_u16.to_le_bytes());
    // min = 0.0 (f16: 0x0000)
    data.extend_from_slice(&0x0000_u16.to_le_bytes());
    // qh: all 1s (every high bit set)
    data.extend_from_slice(&[0xFF; 4]);
    // qs: all zeros
    data.extend_from_slice(&[0x00; 16]);

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // With high bit = 1, q = 0 | (1 << 4) = 16, value = 1.0 * 16 + 0 = 16.0
    for v in &result {
        assert!((v - 16.0).abs() < 1e-3);
    }
}

#[test]
fn test_dequantize_q5_1_invalid_length() {
    let data = vec![0u8; 23]; // Not a multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_multiple_blocks() {
    let mut data = Vec::new();

    // Block 1
    data.extend_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0
    data.extend_from_slice(&0x0000_u16.to_le_bytes()); // min=0.0
    data.extend_from_slice(&[0x00; 4]);
    data.extend_from_slice(&[0x00; 16]);

    // Block 2
    data.extend_from_slice(&0x4000_u16.to_le_bytes()); // d=2.0
    data.extend_from_slice(&0x3C00_u16.to_le_bytes()); // min=1.0
    data.extend_from_slice(&[0x00; 4]);
    data.extend_from_slice(&[0x00; 16]);

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 64); // 2 blocks * 32 values
}

// ========================================================================
// SIMD-PARALLEL DEQUANTIZATION TESTS (EXTREME TDD)
// ========================================================================

#[test]
fn test_dequantize_q4_k_parallel_matches_scalar() {
    // Create 2 super-blocks (288 bytes)
    let mut data = vec![0u8; 288];

    // Super-block 0: d=1.0, dmin=0.0, all zeros
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d=1.0 (f16)
    data[2..4].copy_from_slice(&0x0000_u16.to_le_bytes()); // dmin=0.0

    // Super-block 1: d=2.0, dmin=0.5
    data[144..146].copy_from_slice(&0x4000_u16.to_le_bytes()); // d=2.0
    data[146..148].copy_from_slice(&0x3800_u16.to_le_bytes()); // dmin=0.5

    let scalar = dequantize_q4_k(&data).expect("test");
    let parallel = dequantize_q4_k_parallel(&data).expect("test");

    assert_eq!(scalar.len(), parallel.len());
    for (s, p) in scalar.iter().zip(parallel.iter()) {
        assert!((s - p).abs() < 1e-5, "Mismatch: scalar={s}, parallel={p}");
    }
}

#[test]
fn test_dequantize_q4_k_simd_matches_scalar() {
    // Create a single super-block
    let mut data = vec![0u8; 144];

    // d=1.5, dmin=0.25
    data[0..2].copy_from_slice(&0x3E00_u16.to_le_bytes()); // d≈1.5
    data[2..4].copy_from_slice(&0x3400_u16.to_le_bytes()); // dmin≈0.25

    // Set some non-zero quantized values
    for (idx, byte) in data[16..144].iter_mut().enumerate() {
        *byte = (idx % 16) as u8 | ((idx % 8) << 4) as u8;
    }

    let scalar = dequantize_q4_k(&data).expect("test");
    let simd = dequantize_q4_k_simd(&data).expect("test");

    assert_eq!(scalar.len(), simd.len());
    assert_eq!(simd.len(), 256);

    for (i, (s, p)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            (s - p).abs() < 1e-4,
            "Mismatch at index {i}: scalar={s}, simd={p}"
        );
    }
}

#[test]
fn test_dequantize_q4_k_parallel_invalid_length() {
    let data = vec![0u8; 143]; // Not a multiple of 144
    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_simd_invalid_length() {
    let data = vec![0u8; 145]; // Not a multiple of 144
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_err());
}
