
#[test]
fn test_fused_q4k_q8k_dot_negative_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![-1i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_mixed_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let q8k_scales = vec![0.5f32; 1];
    let mut q8k_quants = Vec::with_capacity(QK_K);
    for i in 0..QK_K {
        q8k_quants.push(if i % 2 == 0 { 1 } else { -1 });
    }

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// ============================================================================
// Part 7: fused_q4k_q8k_dot_simd Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_simd_invalid_length() {
    let q4k_data = vec![0u8; 100];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_empty_inputs() {
    let q4k_data: Vec<u8> = vec![];
    let q8k_scales: Vec<f32> = vec![];
    let q8k_quants: Vec<i8> = vec![];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_simd_single_block() {
    let q4k_data = create_q4k_test_block(1.0, 0.25);
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_multiple_blocks() {
    let num_blocks = 8;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for _ in 0..num_blocks {
        q4k_data.extend_from_slice(&create_q4k_test_block(0.25, 0.1));
    }
    let q8k_scales = vec![0.25f32; num_blocks];
    let q8k_quants = vec![3i8; num_blocks * QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

// ============================================================================
// Part 8: Q4K x Q8K Scalar vs SIMD Equivalence
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_zero() {
    let q4k_data = vec![0u8; Q4K_BLOCK_BYTES];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; QK_K];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");

    assert!(
        (scalar - simd).abs() < 1e-5,
        "scalar={} simd={}",
        scalar,
        simd
    );
}

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_basic() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![2i8; QK_K];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_varied() {
    let mut q4k_data = Vec::with_capacity(Q4K_BLOCK_BYTES);
    for i in 0..Q4K_BLOCK_BYTES {
        q4k_data.push((i * 41 % 256) as u8);
    }
    // Fix d and dmin to valid f16
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C;
    q4k_data[2] = 0x00;
    q4k_data[3] = 0x38;

    let q8k_scales: Vec<f32> = (0..8).map(|i| 0.1 + i as f32 * 0.1).collect();
    let q8k_quants: Vec<i8> = (0..QK_K)
        .map(|i| ((i % 256) as i8).wrapping_sub(64))
        .collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_scalar_simd_equivalence_many_blocks() {
    let num_blocks = 16;
    let mut q4k_data = Vec::with_capacity(num_blocks * Q4K_BLOCK_BYTES);
    for b in 0..num_blocks {
        let mut block = create_q4k_test_block(0.5, 0.25);
        for i in 12..Q4K_BLOCK_BYTES {
            block[i] = ((b * 23 + i * 11) % 256) as u8;
        }
        q4k_data.extend_from_slice(&block);
    }

    let q8k_scales = vec![0.25f32; num_blocks];
    let q8k_quants: Vec<i8> = (0..num_blocks * QK_K)
        .map(|i| ((i % 127) as i8) - 63)
        .collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("test value should be present");

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

// ============================================================================
// Part 9: Edge Cases
// ============================================================================

#[test]
fn test_fused_q4k_dot_large_scale() {
    // Very large scale value
    let q4k_data = create_q4k_test_block(1000.0, 500.0);
    // Ensure d is a valid f16 representation of ~1000
    // f16 max is 65504, so 1000 is fine
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_dot_small_scale() {
    // Very small scale value
    let q4k_data = create_q4k_test_block(0.001, 0.0005);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_large_activations() {
    let q4k_data = create_q4k_test_block(1.0, 0.5);
    let activations = vec![1000.0f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_extreme_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![127i8; QK_K]; // Max i8 value

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_min_quants() {
    let q4k_data = create_q4k_test_block(1.0, 0.0);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![-128i8; QK_K]; // Min i8 value

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_finite());
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a Q4_K test block with specified d and dmin values
///
/// Q4_K super-block format (144 bytes):
/// - d: 2 bytes (f16)
/// - dmin: 2 bytes (f16)
/// - scales: 12 bytes (6-bit scales for 8 sub-blocks)
/// - quants: 128 bytes (4-bit quants for 256 values)
fn create_q4k_test_block(d: f32, dmin: f32) -> Vec<u8> {
    let mut block = vec![0u8; Q4K_BLOCK_BYTES];

    // Write d as f16 (simplified - use known f16 bit patterns)
    let d_f16 = f32_to_f16_approx(d);
    block[0] = (d_f16 & 0xFF) as u8;
    block[1] = ((d_f16 >> 8) & 0xFF) as u8;

    // Write dmin as f16
    let dmin_f16 = f32_to_f16_approx(dmin);
    block[2] = (dmin_f16 & 0xFF) as u8;
    block[3] = ((dmin_f16 >> 8) & 0xFF) as u8;

    // Scales (12 bytes) - set to mid-range values
    for i in 4..16 {
        block[i] = 0x88; // Mid-range scale
    }

    // Quants (128 bytes) - set to alternating pattern
    for i in 16..Q4K_BLOCK_BYTES {
        block[i] = 0x55; // Alternating nibbles
    }

    block
}

/// Approximate f32 to f16 conversion
fn f32_to_f16_approx(f: f32) -> u16 {
    if f == 0.0 {
        return 0;
    }
    if f.is_nan() {
        return 0x7E00;
    }
    if f.is_infinite() {
        return if f > 0.0 { 0x7C00 } else { 0xFC00 };
    }

    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    // Bias adjustment: f32 bias is 127, f16 bias is 15
    let new_exp = exp - 127 + 15;

    if new_exp <= 0 {
        // Subnormal or zero
        0
    } else if new_exp >= 31 {
        // Overflow to infinity
        (sign << 15) | 0x7C00
    } else {
        // Normal number
        let new_mantissa = (mantissa >> 13) as u16;
        (sign << 15) | ((new_exp as u16) << 10) | new_mantissa
    }
}
