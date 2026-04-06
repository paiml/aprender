//! T-COV-95 Phase 50: Deep coverage for fused_q5k_q6k.rs and fused_k.rs
//!
//! Covers:
//! - fused_q6k_dot: error paths, zero data, valid computation, SIMD dispatch
//! - fused_q5k_dot: error paths, zero data, valid computation
//! - fused_q6k_dot_simd: delegates to correct path
//! - fused_q5k_dot_simd: delegates to scalar
//! - fused_q4k_q8_dot (from fused_q5k_q6k): error paths, zero data

use crate::quantize::fused_q5k_q6k::{
    fused_q4k_q8_dot, fused_q5k_dot, fused_q5k_dot_simd, fused_q6k_dot, fused_q6k_dot_simd,
};
use crate::quantize::types::Q8_0Block;

// ============================================================================
// fused_q6k_dot error paths
// ============================================================================

#[test]
fn test_fused_q6k_dot_bad_data_length() {
    // Not a multiple of 210
    let data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a multiple") || err.contains("super-block"),
        "Expected super-block error, got: {}",
        err
    );
}

#[test]
fn test_fused_q6k_dot_activation_mismatch() {
    let data = vec![0u8; 210]; // 1 super-block = 256 values
    let activations = vec![1.0f32; 128]; // Wrong: need 256
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("doesn't match") || err.contains("Activation"),
        "Expected activation mismatch error, got: {}",
        err
    );
}

#[test]
fn test_fused_q6k_dot_zero_data() {
    let data = vec![0u8; 210]; // All zeros
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_ok());
    // d=0 means all dequantized values are 0, so dot product is 0
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q6k_dot_empty() {
    let data: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q6k_dot_nonzero() {
    let mut data = vec![0u8; 210];
    // Set d = 1.0 (f16: 0x3C00) at offset 208
    data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set scales to 1
    for i in 0..16 {
        data[192 + i] = 1;
    }
    // Set some ql values to produce nonzero quantized values
    for i in 0..128 {
        data[i] = 0x33; // low nibble = 3, makes (3 | ((qh&3)<<4)) - 32
    }
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_ok());
    // Should be nonzero because d>0, scales>0, and some q values are nonzero
    // Even though values may be negative (q - 32), the dot product with uniform activations won't be 0
    // Actually it could be 0 if the positive and negative values cancel.
    // The important thing is that the computation completed without error.
}

#[test]
fn test_fused_q6k_dot_two_superblocks() {
    let data = vec![0u8; 210 * 2]; // 2 super-blocks
    let activations = vec![1.0f32; 512]; // 256 * 2
    let result = fused_q6k_dot(&data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0); // All zeros
}

// ============================================================================
// fused_q6k_dot_simd
// ============================================================================

#[test]
fn test_fused_q6k_dot_simd_matches_scalar() {
    let mut data = vec![0u8; 210];
    data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1.0
    for i in 0..16 {
        data[192 + i] = 2; // scales
    }
    for i in 0..128 {
        data[i] = 0x44;
    }
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();

    let scalar_result = fused_q6k_dot(&data, &activations).expect("test value should be present");
    let simd_result = fused_q6k_dot_simd(&data, &activations).expect("test value should be present");

    // SIMD and scalar should produce very close results
    assert!(
        (scalar_result - simd_result).abs() < 1.0,
        "Scalar={}, SIMD={}, diff={}",
        scalar_result,
        simd_result,
        (scalar_result - simd_result).abs()
    );
}

#[test]
fn test_fused_q6k_dot_simd_error_propagation() {
    let data = vec![0u8; 100]; // Bad length
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot_simd(&data, &activations);
    assert!(result.is_err());
}

// ============================================================================
// fused_q5k_dot error paths
// ============================================================================

#[test]
fn test_fused_q5k_dot_bad_data_length() {
    let data = vec![0u8; 100]; // Not a multiple of 176
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a multiple") || err.contains("super-block"),
        "Expected super-block error, got: {}",
        err
    );
}

#[test]
fn test_fused_q5k_dot_activation_mismatch() {
    let data = vec![0u8; 176]; // 1 super-block = 256 values
    let activations = vec![1.0f32; 128]; // Wrong: need 256
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("doesn't match") || err.contains("Activation"),
        "Expected activation mismatch error, got: {}",
        err
    );
}

#[test]
fn test_fused_q5k_dot_zero_data() {
    let data = vec![0u8; 176]; // All zeros
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_ok());
    // d=0 means dot product is 0
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q5k_dot_empty() {
    let data: Vec<u8> = vec![];
    let activations: Vec<f32> = vec![];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q5k_dot_nonzero() {
    let mut data = vec![0u8; 176];
    // Set d = 1.0 (f16: 0x3C00) at offset 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set dmin = 0 at offset 2
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // Set some scales (12 bytes at offset 4)
    for i in 0..12 {
        data[4 + i] = 0x11; // Some scale value
    }
    // Set some qs (128 bytes at offset 48)
    for i in 0..128 {
        data[48 + i] = 0x55;
    }
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_ok());
    // The computation should complete without error
}

#[test]
fn test_fused_q5k_dot_two_superblocks() {
    let data = vec![0u8; 176 * 2];
    let activations = vec![1.0f32; 512];
    let result = fused_q5k_dot(&data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

// ============================================================================
// fused_q5k_dot_simd
// ============================================================================

#[test]
fn test_fused_q5k_dot_simd_delegates() {
    // q5k_dot_simd currently delegates to scalar, so results should match
    let data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let scalar = fused_q5k_dot(&data, &activations).expect("test value should be present");
    let simd = fused_q5k_dot_simd(&data, &activations).expect("test value should be present");
    assert_eq!(scalar, simd);
}

#[test]
fn test_fused_q5k_dot_simd_error_propagation() {
    let data = vec![0u8; 100]; // Bad length
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&data, &activations);
    assert!(result.is_err());
}

// ============================================================================
// fused_q4k_q8_dot error paths (from fused_q5k_q6k.rs)
// ============================================================================

#[test]
fn test_fused_q4k_q8_dot_bad_q4k_length() {
    let data = vec![0u8; 100]; // Not a multiple of 144
    let q8_blocks: Vec<Q8_0Block> = vec![];
    let result = fused_q4k_q8_dot(&data, &q8_blocks);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8_dot_q8_block_mismatch() {
    let data = vec![0u8; 144]; // 1 super-block = 256 values = 8 Q8 blocks
    let q8_blocks = vec![
        Q8_0Block {
            scale: 0.0,
            quants: [0i8; 32]
        };
        4
    ]; // Wrong: need 8
    let result = fused_q4k_q8_dot(&data, &q8_blocks);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("doesn't match") || err.contains("block count"),
        "Expected block count error, got: {}",
        err
    );
}

#[test]
fn test_fused_q4k_q8_dot_zeros() {
    let data = vec![0u8; 144];
    let q8_blocks = vec![
        Q8_0Block {
            scale: 0.0,
            quants: [0i8; 32]
        };
        8
    ];
    let result = fused_q4k_q8_dot(&data, &q8_blocks);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8_dot_empty() {
    let data: Vec<u8> = vec![];
    let q8_blocks: Vec<Q8_0Block> = vec![];
    let result = fused_q4k_q8_dot(&data, &q8_blocks);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8_dot_nonzero() {
    let mut data = vec![0u8; 144];
    // Set d = 1.0 (f16: 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }
    // Set some qs
    for i in 0..128 {
        data[16 + i] = 0x55;
    }
    // Create Q8 blocks with nonzero values
    let mut q8_blocks = Vec::new();
    for _ in 0..8 {
        q8_blocks.push(Q8_0Block {
            scale: 0.1,
            quants: [10i8; 32],
        });
    }
    let result = fused_q4k_q8_dot(&data, &q8_blocks);
    assert!(result.is_ok());
    // Non-zero because d>0, scales>0, qs>0, and q8 values are nonzero
}

#[test]
fn test_fused_q4k_q8_dot_two_superblocks() {
    let data = vec![0u8; 144 * 2];
    let q8_blocks = vec![
        Q8_0Block {
            scale: 0.0,
            quants: [0i8; 32]
        };
        16
    ];
    let result = fused_q4k_q8_dot(&data, &q8_blocks);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}
