//! Part 13: Additional coverage for quantize/mod.rs uncovered paths
//!
//! Focus areas:
//! - InterleavedQ4K::dot with various input sizes (mod.rs impl)
//! - extract_scale_min and extract_scale_min_from_slice edge cases
//! - fused_q4_0_q8_0_dot_scalar with edge cases
//! - fused_q8_0_q8_0_dot_scalar with edge cases
//! - fused_q4_0_q8_0_parallel_matvec_into error paths
//! - fused_q8_0_q8_0_parallel_matvec_into error paths

use crate::quantize::{
    dequantize_q8_blocks, extract_scale_min, extract_scale_min_from_slice,
    fused_q4_0_q8_0_dot_scalar, fused_q4_0_q8_0_parallel_matvec,
    fused_q4_0_q8_0_parallel_matvec_into, fused_q8_0_q8_0_dot_scalar,
    fused_q8_0_q8_0_parallel_matvec, fused_q8_0_q8_0_parallel_matvec_into,
    quantize_activations_q8k_into, quantize_to_q8_blocks, InterleavedQ4K, Q8KSuperBlock, Q8_0Block,
};

// =============================================================================
// InterleavedQ4K::dot Tests - Exercising scalar fallback path (mod.rs impl)
// =============================================================================

#[test]
fn test_interleaved_q4k_mod_dot_single_superblock_zeros() {
    // 1 super-block with all zeros
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations).expect("dot should work");

    // All zeros in weights means result should be approximately 0
    assert!(result.is_finite(), "Result should be finite: {}", result);
}

#[test]
fn test_interleaved_q4k_mod_dot_single_superblock_ones() {
    // Create super-block with d=1.0, all scales=1, all qs=0x11
    let mut data = vec![0u8; 144];

    // d = 1.0 (f16: 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // dmin = 0.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    // Set scale byte 0 to 1 (for first block scale)
    data[4] = 1;

    // Set qs to 0x11 (low=1, high=1)
    for i in 16..144 {
        data[i] = 0x11;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_two_superblocks() {
    // 2 super-blocks
    let mut data = vec![0u8; 288];

    // Super-block 0: d=1.0, dmin=0.0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    // Super-block 1: d=2.0, dmin=0.0
    data[144..146].copy_from_slice(&0x4000u16.to_le_bytes());
    data[146..148].copy_from_slice(&0x0000u16.to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![0.5f32; 512];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_length_error() {
    let data = vec![0u8; 144]; // 256 values
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    // Wrong length
    let activations = vec![1.0f32; 128];
    let result = interleaved.dot(&activations);
    assert!(result.is_err(), "Should error on length mismatch");

    let activations = vec![1.0f32; 300];
    let result = interleaved.dot(&activations);
    assert!(result.is_err(), "Should error on length mismatch");
}

#[test]
fn test_interleaved_q4k_mod_dot_varied_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0

    // Set scales
    for i in 4..16 {
        data[i] = 10;
    }

    // Set varied qs
    for i in 16..144 {
        data[i] = ((i - 16) % 16) as u8 | (((i - 16 + 1) % 16) << 4) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    // Varied activations
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_negative_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 32; // scale

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![-1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_mod_dot_with_dmin() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    data[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin=1.0 (nonzero!)

    // Set mins in scales (bytes 4-7 are for mins of first 4 blocks)
    data[4] = 0; // scale for block 0
    data[8] = 10; // min for block 0

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    // With dmin nonzero, we subtract dmin*m from the result
    assert!(result.is_finite());
}

// =============================================================================
// extract_scale_min Tests - All block indices
// =============================================================================

#[test]
fn test_extract_scale_min_mod_all_blocks() {
    let scales: [u8; 12] = [
        0b00_000001, // byte 0: scale0=1, high bits=0
        0b00_000010, // byte 1: scale1=2
        0b00_000011, // byte 2: scale2=3
        0b00_000100, // byte 3: scale3=4
        0b00_000101, // byte 4: min0=5
        0b00_000110, // byte 5: min1=6
        0b00_000111, // byte 6: min2=7
        0b00_001000, // byte 7: min3=8
        0b0010_0001, // byte 8: low=1 (scale4), high=2 (min4)
        0b0100_0011, // byte 9: low=3 (scale5), high=4 (min5)
        0b0110_0101, // byte 10: low=5 (scale6), high=6 (min6)
        0b1000_0111, // byte 11: low=7 (scale7), high=8 (min7)
    ];

    // Blocks 0-3: simple extraction
    let (s0, m0) = extract_scale_min(&scales, 0);
    assert_eq!(s0, 1.0);
    assert_eq!(m0, 5.0);

    let (s1, m1) = extract_scale_min(&scales, 1);
    assert_eq!(s1, 2.0);
    assert_eq!(m1, 6.0);

    let (s2, m2) = extract_scale_min(&scales, 2);
    assert_eq!(s2, 3.0);
    assert_eq!(m2, 7.0);

    let (s3, m3) = extract_scale_min(&scales, 3);
    assert_eq!(s3, 4.0);
    assert_eq!(m3, 8.0);

    // Blocks 4-7: packed extraction
    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = 1 | (0 << 4) = 1
    //          m = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = 2 | (0 << 4) = 2
    let (s4, m4) = extract_scale_min(&scales, 4);
    assert_eq!(s4, 1.0);
    assert_eq!(m4, 2.0);

    let (s5, m5) = extract_scale_min(&scales, 5);
    assert_eq!(s5, 3.0);
    assert_eq!(m5, 4.0);

    let (s6, m6) = extract_scale_min(&scales, 6);
    assert_eq!(s6, 5.0);
    assert_eq!(m6, 6.0);

    let (s7, m7) = extract_scale_min(&scales, 7);
    assert_eq!(s7, 7.0);
    assert_eq!(m7, 8.0);
}

#[test]
fn test_extract_scale_min_mod_high_bits_contribution() {
    // Test that high bits from bytes 0-3 contribute to blocks 4-7
    let mut scales: [u8; 12] = [0; 12];

    // Set high bits in byte 0 (contributes to scale4)
    scales[0] = 0b11_000000; // high bits = 3
                             // Set low bits in byte 8 (contributes to scale4)
    scales[8] = 0b0000_0001; // low = 1

    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = 1 | (3 << 4) = 49
    let (s4, _) = extract_scale_min(&scales, 4);
    assert_eq!(s4, 49.0, "Scale4 with high bits contribution");
}

#[test]
fn test_extract_scale_min_from_slice_mod_all_indices() {
    let scales: [u8; 12] = [
        10, 20, 30, 40, // bytes 0-3
        5, 15, 25, 35, // bytes 4-7
        0, 0, 0, 0, // bytes 8-11
    ];

    // Even indices use simple extraction
    // idx=0: scale_idx=0, min_idx=4
    let (s0, m0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(s0, 10.0);
    assert_eq!(m0, 5.0);

    // idx=2: scale_idx=1, min_idx=5
    let (s2, m2) = extract_scale_min_from_slice(&scales, 2);
    assert_eq!(s2, 20.0);
    assert_eq!(m2, 15.0);

    // idx=4: scale_idx=2, min_idx=6
    let (s4, m4) = extract_scale_min_from_slice(&scales, 4);
    assert_eq!(s4, 30.0);
    assert_eq!(m4, 25.0);

    // idx=6: scale_idx=3, min_idx=7
    let (s6, m6) = extract_scale_min_from_slice(&scales, 6);
    assert_eq!(s6, 40.0);
    assert_eq!(m6, 35.0);
}

#[test]
fn test_extract_scale_min_from_slice_mod_odd_indices_formula() {
    // For odd indices:
    // scale_idx = idx / 2
    // min_idx = scale_idx + 4
    // scale = (scales[scale_idx] >> 6) | ((scales[scale_idx + 2] & 0x0F) << 2)
    // min = (scales[min_idx] >> 6) | ((scales[min_idx + 2] & 0x0F) << 2)

    let mut scales: [u8; 12] = [0; 12];
    // For idx=1: scale_idx=0, uses bytes 0 and 2
    scales[0] = 0b11_000000; // high 2 bits = 3
    scales[2] = 0b0000_1111; // low 4 bits = 15
                             // scale = 3 | (15 << 2) = 3 + 60 = 63

    scales[4] = 0b10_000000; // high 2 bits = 2
    scales[6] = 0b0000_0101; // low 4 bits = 5
                             // min = 2 | (5 << 2) = 2 + 20 = 22

    let (s1, m1) = extract_scale_min_from_slice(&scales, 1);
    assert_eq!(s1, 63.0, "idx=1 scale");
    assert_eq!(m1, 22.0, "idx=1 min");
}

// =============================================================================
// fused_q4_0_q8_0_dot_scalar Edge Cases
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_empty() {
    let q4_data: Vec<u8> = vec![];
    let q8_scales: Vec<f32> = vec![];
    let q8_quants: Vec<i8> = vec![];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_partial_block() {
    // Q4_0 block is 18 bytes (2 scale + 16 quants)
    // If we only have 10 bytes, it's a partial block
    let mut q4_data = vec![0u8; 10];
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    // Should not panic, processes what's available
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_two_complete_blocks() {
    // 2 Q4_0 blocks = 36 bytes
    let mut q4_data = vec![0u8; 36];

    // Block 0: scale = 1.0
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // quants centered at 8, so 0x88 gives q=0 after -8 offset
    for i in 2..18 {
        q4_data[i] = 0x88;
    }

    // Block 1: scale = 2.0
    q4_data[18..20].copy_from_slice(&0x4000u16.to_le_bytes());
    for i in 20..36 {
        q4_data[i] = 0x88;
    }

    let q8_scales = vec![1.0f32, 1.0f32];
    let q8_quants = vec![1i8; 64];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // With centered quants (0x88 = q_val 0 after -8), result should be ~0
    assert!(result.abs() < 1.0, "Expected near 0, got {}", result);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_negative_quants() {
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    // Set q4 quants to 0x00 (low=0, high=0 -> values = -8 after offset)
    for i in 2..18 {
        q4_data[i] = 0x00;
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![10i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // Each dequantized Q4 value is -8 (0 - 8 offset), scaled by 1.0
    assert!(result < 0.0, "Should be negative with 0x00 q4 quants");
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_large_scale() {
    let mut q4_data = vec![0u8; 18];
    // scale = 1000.0 in f16 (approximate: 0x63D0)
    q4_data[0..2].copy_from_slice(&0x63D0u16.to_le_bytes());
    for i in 2..18 {
        q4_data[i] = 0xFF; // max q4 values
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar Edge Cases
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_empty() {
    let q8_weight_data: Vec<u8> = vec![];
    let q8_act_scales: Vec<f32> = vec![];
    let q8_act_quants: Vec<i8> = vec![];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_partial_block() {
    // Q8_0 block is 34 bytes (2 scale + 32 quants)
    let mut q8_weight_data = vec![0u8; 20]; // partial
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_two_blocks() {
    // 2 Q8_0 blocks = 68 bytes
    let mut q8_weight_data = vec![0u8; 68];

    // Block 0: scale = 1.0, all quants = 5
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 2..34 {
        q8_weight_data[i] = 5u8;
    }

    // Block 1: scale = 0.5, all quants = 10
    q8_weight_data[34..36].copy_from_slice(&0x3800u16.to_le_bytes());
    for i in 36..68 {
        q8_weight_data[i] = 10u8;
    }

    let q8_act_scales = vec![1.0f32, 1.0f32];
    let q8_act_quants = vec![2i8; 64];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 64);
    // Block 0: 1.0 * 1.0 * (5 * 2 * 32) = 320
    // Block 1: 0.5 * 1.0 * (10 * 2 * 32) = 320
    // Total: 640
    assert!(
        (result - 640.0).abs() < 10.0,
        "Expected ~640, got {}",
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_mixed_signs() {
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0

    // Mix of positive and negative quants
    for i in 2..18 {
        q8_weight_data[i] = 10u8; // positive
    }
    for i in 18..34 {
        q8_weight_data[i] = (-10i8) as u8; // negative
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![5i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // 16 positive: 10 * 5 * 16 = 800
    // 16 negative: -10 * 5 * 16 = -800
    // Total: 0
    assert!(
        result.abs() < 10.0,
        "Mixed signs should nearly cancel: {}",
        result
    );
}

include!("fused_02.rs");
include!("fused_02_02.rs");
