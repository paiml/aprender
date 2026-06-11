//! Quantize Tests Part 29: T-COV-95 Coverage Bridge (B7)
//!
//! Tests for uncovered dequantization functions:
//! - dequantize_q4_1: valid data, invalid block size
//! - dequantize_q5_0: valid data, invalid block size
//! - dequantize_q5_1: valid data, invalid block size
//! - dequantize_q2_k: valid data, invalid block size
//! - Error paths for all format-specific dequant functions
//!
//! Refs PMAT-802: Protocol T-COV-95 Batch B7

use crate::quantize::*;

// ============================================================================
// Q4_1 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q4_1_single_block() {
    // Q4_1: 20 bytes per block (2 f16 scale + 2 f16 min + 16 quants)
    let mut data = vec![0u8; 20];
    // Scale = 1.0 as f16
    let scale_f16 = 0x3C00u16; // f16 for 1.0
    data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    // Min = 0.0 as f16
    data[2..4].copy_from_slice(&0u16.to_le_bytes());
    // Quants: all zeros
    // Each byte encodes two values: low nibble and high nibble
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 32);
    // All values should be 0.0 (0 * 1.0 + 0.0)
    for v in &values {
        assert!(v.abs() < 1e-3, "Expected ~0.0, got {}", v);
    }
}

#[test]
fn test_dequantize_q4_1_with_min_offset() {
    let mut data = vec![0u8; 20];
    let scale_f16 = 0x3C00u16; // 1.0
    let min_f16 = 0x4000u16; // 2.0
    data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    data[2..4].copy_from_slice(&min_f16.to_le_bytes());
    // All quants zero: each value = 0 * 1.0 + 2.0 = 2.0
    let result = dequantize_q4_1(&data).expect("test value should be present");
    for v in &result {
        assert!((v - 2.0).abs() < 0.1, "Expected ~2.0, got {}", v);
    }
}

#[test]
fn test_dequantize_q4_1_multiple_blocks() {
    let data = vec![0u8; 20 * 3]; // 3 blocks
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32 * 3);
}

#[test]
fn test_dequantize_q4_1_invalid_size() {
    let data = vec![0u8; 19]; // Not multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_invalid_size_21() {
    let data = vec![0u8; 21];
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_1_empty() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

// ============================================================================
// Q5_0 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q5_0_single_block() {
    // Q5_0: 22 bytes per block (2 f16 scale + 4 high bits + 16 quants)
    let mut data = vec![0u8; 22];
    let scale_f16 = 0x3C00u16; // 1.0
    data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    // qh (4 bytes) = 0, qs (16 bytes) = 0
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 32);
    // Q5_0 is centered (value - 16), so all zeros become -16 * scale
    for v in &values {
        assert!((v - (-16.0)).abs() < 0.5, "Expected ~-16.0, got {}", v);
    }
}

#[test]
fn test_dequantize_q5_0_multiple_blocks() {
    let data = vec![0u8; 22 * 2];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32 * 2);
}

#[test]
fn test_dequantize_q5_0_invalid_size() {
    let data = vec![0u8; 21]; // Not multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_invalid_size_23() {
    let data = vec![0u8; 23];
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_0_empty() {
    let result = dequantize_q5_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

// ============================================================================
// Q5_1 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q5_1_single_block() {
    // Q5_1: 24 bytes per block (2 f16 scale + 2 f16 min + 4 high bits + 16 quants)
    let mut data = vec![0u8; 24];
    let scale_f16 = 0x3C00u16; // 1.0
    let min_f16 = 0x4000u16; // 2.0
    data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    data[2..4].copy_from_slice(&min_f16.to_le_bytes());
    // All quants zero: value = 0 * 1.0 + 2.0 = 2.0
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 32);
    for v in &values {
        assert!((v - 2.0).abs() < 0.5, "Expected ~2.0, got {}", v);
    }
}

#[test]
fn test_dequantize_q5_1_multiple_blocks() {
    let data = vec![0u8; 24 * 4];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 32 * 4);
}

#[test]
fn test_dequantize_q5_1_invalid_size() {
    let data = vec![0u8; 23];
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_invalid_size_25() {
    let data = vec![0u8; 25];
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_empty() {
    let result = dequantize_q5_1(&[]);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

// ============================================================================
// Q2_K Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q2_k_single_block() {
    // Q2_K: 84 bytes per super-block
    // Layout: 16 bytes scales + 64 bytes quants + 2 f16 d + 2 f16 dmin
    let mut data = vec![0u8; 84];
    // Set d (scale) at offset 80-81
    let d_f16 = 0x3C00u16; // 1.0
    data[80..82].copy_from_slice(&d_f16.to_le_bytes());
    // Set dmin at offset 82-83
    data[82..84].copy_from_slice(&0u16.to_le_bytes()); // min = 0
    let result = dequantize_q2_k(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 256); // QK_K = 256
}

/// FT-Q2K-002 (inference path): serve dequantize_q2_k must match ggml
/// `dequantize_row_q2_K` / candle byte-for-byte. The prior scheme applied the
/// wrong scale to the wrong 2-bit lanes (185/256 elements wrong → corrupt Q2_K
/// inference).
#[test]
fn test_dequantize_q2_k_ggml_golden() {
    fn reference(scales: &[u8], qs: &[u8], d: f32, dmin: f32) -> Vec<f32> {
        let mut y = Vec::with_capacity(256);
        let mut is = 0usize;
        for group in 0..2 {
            let chunk = &qs[group * 32..group * 32 + 32];
            let mut shift = 0u8;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                for &q in &chunk[0..16] {
                    y.push(dl * f32::from((q >> shift) & 0x03) - ml);
                }
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                for &q in &chunk[16..32] {
                    y.push(dl * f32::from((q >> shift) & 0x03) - ml);
                }
                shift += 2;
            }
        }
        y
    }
    let scales: Vec<u8> = (0u8..16).map(|i| i.wrapping_mul(9).wrapping_add(5)).collect();
    let qs: Vec<u8> = (0u8..64).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
    let mut data = Vec::with_capacity(84);
    data.extend_from_slice(&scales);
    data.extend_from_slice(&qs);
    data.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    data.extend_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
    let expected = reference(&scales, &qs, 1.0, 0.5);
    let got = dequantize_q2_k(&data).expect("dequant");
    assert_eq!(got.len(), 256);
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "Q2_K elem {i}: got {g}, want {e} (ggml ordering)"
        );
    }
}

#[test]
fn test_dequantize_q2_k_multiple_blocks() {
    let data = vec![0u8; 84 * 2];
    let result = dequantize_q2_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 256 * 2);
}

#[test]
fn test_dequantize_q2_k_invalid_size() {
    let data = vec![0u8; 83]; // Not multiple of 84
    let result = dequantize_q2_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q2_k_invalid_size_85() {
    let data = vec![0u8; 85];
    let result = dequantize_q2_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q2_k_empty() {
    let result = dequantize_q2_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

#[test]
fn test_dequantize_q2_k_with_nonzero_scales() {
    // Set scales to non-zero to exercise scale extraction logic
    let mut data = vec![0u8; 84];
    // Scales occupy bytes 0-15 (packed as nibbles)
    for i in 0..16 {
        data[i] = 0x11; // Scale nibbles: low=1, high=1
    }
    let d_f16 = 0x3C00u16; // 1.0
    data[80..82].copy_from_slice(&d_f16.to_le_bytes());
    let dmin_f16 = 0x3800u16; // 0.5
    data[82..84].copy_from_slice(&dmin_f16.to_le_bytes());
    let result = dequantize_q2_k(&data);
    assert!(result.is_ok());
    let values = result.expect("test value should be present");
    assert_eq!(values.len(), 256);
    // Values should be non-zero due to scale
    let has_nonzero = values.iter().any(|&v| v.abs() > 1e-6);
    assert!(
        has_nonzero,
        "Expected some non-zero values with non-zero scales"
    );
}

// ============================================================================
// Q4_1 nibble pattern tests
// ============================================================================

#[test]
fn test_dequantize_q4_1_nibble_pattern() {
    let mut data = vec![0u8; 20];
    let scale_f16 = 0x3C00u16; // 1.0
    data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    data[2..4].copy_from_slice(&0u16.to_le_bytes()); // min = 0
    // Set first quant byte to 0xAB (low=B=11, high=A=10)
    data[4] = 0xAB;
    let result = dequantize_q4_1(&data).expect("test value should be present");
    // Position 0 (low nibble): 11 * 1.0 + 0.0 = 11.0
    assert!((result[0] - 11.0).abs() < 0.5, "Got {}", result[0]);
    // Position 16 (high nibble): 10 * 1.0 + 0.0 = 10.0
    assert!((result[16] - 10.0).abs() < 0.5, "Got {}", result[16]);
}

// ============================================================================
// Q5_0 high-bit tests
// ============================================================================

#[test]
fn test_dequantize_q5_0_with_high_bits() {
    let mut data = vec![0u8; 22];
    let scale_f16 = 0x3C00u16; // 1.0
    data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    // Set high bits: qh = 0x00000001 (bit 0 set)
    data[2..6].copy_from_slice(&1u32.to_le_bytes());
    // First quant = 0x00 (both nibbles 0)
    let result = dequantize_q5_0(&data).expect("test value should be present");
    // Position 0: q_low = 0 | (1 << 4) = 16, value = (16 - 16) * 1.0 = 0
    assert!(result[0].abs() < 0.5, "Got {}", result[0]);
}

// ============================================================================
// Cross-format consistency: all empty inputs produce empty output
// ============================================================================

#[test]
fn test_all_dequant_empty_input_empty_output() {
    assert!(dequantize_q4_1(&[]).expect("test value should be present").is_empty());
    assert!(dequantize_q5_0(&[]).expect("test value should be present").is_empty());
    assert!(dequantize_q5_1(&[]).expect("test value should be present").is_empty());
    assert!(dequantize_q2_k(&[]).expect("test value should be present").is_empty());
}

// ============================================================================
// Cross-format consistency: all invalid sizes produce errors
// ============================================================================

#[test]
fn test_all_dequant_single_byte_errors() {
    assert!(dequantize_q4_1(&[0]).is_err());
    assert!(dequantize_q5_0(&[0]).is_err());
    assert!(dequantize_q5_1(&[0]).is_err());
    assert!(dequantize_q2_k(&[0]).is_err());
}
