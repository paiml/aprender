use super::*;

// ============================================================================
// FUSED Q4_K DOT PRODUCT TESTS (SCALAR)
// ============================================================================

#[test]
fn test_fused_q4k_dot_invalid_data_length() {
    // Q4_K super-block is 144 bytes
    let data = vec![0u8; 100]; // Not a multiple of 144
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not a multiple"));
}

#[test]
fn test_fused_q4k_dot_activation_length_mismatch() {
    // One super-block = 144 bytes = 256 values
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 128]; // Should be 256

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("doesn't match"));
}

#[test]
fn test_fused_q4k_dot_zero_data() {
    // All zeros should produce zero dot product
    let data = vec![0u8; 144];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations).unwrap();
    // d = 0, so all values dequantize to 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_dot_single_super_block() {
    // Create valid Q4_K data: d (2) + dmin (2) + scales (12) + qs (128)
    let mut data = vec![0u8; 144];

    // Set d = 1.0 (f16: 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // Set dmin = 0.5 (f16: 0x3800)
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales (12 bytes packed)
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs values to a pattern
    for i in 0..128 {
        data[16 + i] = 0x55; // Pattern: low=5, high=5
    }

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_multiple_super_blocks() {
    // Two super-blocks = 288 bytes = 512 values
    let mut data = vec![0u8; 288];

    // Set d = 0.5 for both blocks
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    data[144..146].copy_from_slice(&0x3800u16.to_le_bytes());

    let activations = vec![0.5f32; 512];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_empty_data() {
    let data = vec![];
    let activations = vec![];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0);
}

// ============================================================================
// FUSED Q4_K DOT PRODUCT TESTS (SIMD)
// ============================================================================

#[test]
fn test_fused_q4k_dot_simd_invalid_input() {
    let data = vec![0u8; 100]; // Invalid length
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_matches_scalar() {
    // Create valid Q4_K data
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs to deterministic pattern
    for i in 0..128 {
        data[16 + i] = ((i * 3) % 256) as u8;
    }

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

    let scalar_result = fused_q4k_dot(&data, &activations).unwrap();
    let simd_result = fused_q4k_dot_simd(&data, &activations).unwrap();

    // Allow small tolerance for SIMD vs scalar (FMA ordering differences)
    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.001,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_dot_simd_zero_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot_simd(&data, &activations).unwrap();
    // Product of anything with zero should be zero
    assert_eq!(result, 0.0);
}

// ============================================================================
// FUSED Q4_K × Q8_K DOT PRODUCT TESTS
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_invalid_data_length() {
    let data = vec![0u8; 100]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_scales_too_small() {
    let data = vec![0u8; 144]; // 1 super-block
    let q8k_scales = vec![]; // Should be >= 1
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_quants_too_small() {
    let data = vec![0u8; 144]; // 1 super-block = 256 values
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![10i8; 128]; // Should be 256

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_zero_data() {
    let data = vec![0u8; 144];
    let q8k_scales = vec![0.0f32; 1]; // Zero scale
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).unwrap();
    // With zero Q4_K d and zero Q8_K scale, result should be 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_basic() {
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs
    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_invalid_input() {
    let data = vec![0u8; 100]; // Invalid
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![10i8; 256];

    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_matches_scalar() {
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.5
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set scales
    for i in 0..12 {
        data[4 + i] = 0x11;
    }

    // Set qs to pattern
    for i in 0..128 {
        data[16 + i] = ((i * 7) % 256) as u8;
    }

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants: Vec<i8> = (0..256).map(|i| ((i % 64) - 32) as i8).collect();

    let scalar_result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).unwrap();
    let simd_result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).unwrap();

    // Allow tolerance for SIMD vs scalar
    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.02,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_multiple_super_blocks() {
    // Two super-blocks = 288 bytes = 512 values
    let mut data = vec![0u8; 288];

    // Set d for both blocks
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());

    let q8k_scales = vec![0.1f32; 2];
    let q8k_quants = vec![5i8; 512];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_fused_q4k_dot_max_nibble_values() {
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // Set qs to max nibble (0xFF = 15 low, 15 high)
    for i in 0..128 {
        data[16 + i] = 0xFF;
    }

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_negative_quants() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    let q8k_scales = vec![0.1f32; 1];
    let q8k_quants = vec![-10i8; 256]; // All negative

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_simd_large_input() {
    // 8 super-blocks = 1152 bytes = 2048 values
    let mut data = vec![0u8; 1152];

    for sb in 0..8 {
        let offset = sb * 144;
        // Set d = 0.1
        data[offset..offset + 2].copy_from_slice(&0x2E66u16.to_le_bytes());
    }

    let activations = vec![0.5f32; 2048];

    let scalar_result = fused_q4k_dot(&data, &activations).unwrap();
    let simd_result = fused_q4k_dot_simd(&data, &activations).unwrap();

    let rel_err = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        rel_err < 0.001,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_mixed_signs() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // Set qs to alternating pattern
    for i in 0..128 {
        data[16 + i] = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }

    let q8k_scales = vec![0.1f32; 1];
    // Alternating positive and negative quants
    let q8k_quants: Vec<i8> = (0..256)
        .map(|i| if i % 2 == 0 { 10 } else { -10 })
        .collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_scale_extraction() {
    // Test that scales are correctly extracted and applied
    let mut data = vec![0u8; 144];

    // Set d = 2.0 (f16: 0x4000)
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());

    // Set dmin = 0 (no offset)
    data[2..4].copy_from_slice(&[0x00, 0x00]);

    // Set first scale to max (63) - packed format
    data[4] = 0x3F; // First scale = 63

    // Set qs to 1 (low nibble only)
    for i in 0..128 {
        data[16 + i] = 0x01;
    }

    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&data, &activations);
    assert!(result.is_ok());
    // Result should be non-zero since d > 0, scale > 0, qs > 0
    assert!(result.unwrap().abs() > 0.0);
}

include!("fused_k_tests_q4k.rs");
include!("fused_k_tests_dot_errors.rs");
include!("fused_k_tests_avx2_q4k.rs");
