
// =========================================================================
// Coverage Tests: extract_scale_min_from_slice
// =========================================================================

/// Test extract_scale_min_from_slice with various block indices
#[test]
fn test_extract_scale_min_from_slice_all_blocks() {
    let scales = [0u8; 12];
    // Test all 8 blocks
    for idx in 0..8 {
        let (scale, min) = extract_scale_min_from_slice(&scales, idx);
        assert!(scale >= 0.0);
        assert!(min >= 0.0);
    }
}

// =========================================================================
// Coverage Tests: fused_q4_0_q8_0_parallel_matvec_into
// =========================================================================

/// Test fused_q4_0_q8_0_parallel_matvec_into basic operation
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_basic() {
    let in_dim = 32;
    let out_dim = 2;
    let mut weight_data = vec![0u8; 18 * out_dim];
    weight_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    weight_data[18..20].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output)
        .expect("should succeed");
    assert_eq!(output.len(), out_dim);
}

// =========================================================================
// Coverage Tests: fused_q8_0_q8_0_parallel_matvec_into
// =========================================================================

/// Test fused_q8_0_q8_0_parallel_matvec_into basic operation
#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_basic() {
    let in_dim = 32;
    let out_dim = 2;
    // Q8_0: 34 bytes per 32 values
    let mut weight_data = vec![0u8; 34 * out_dim];
    weight_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    weight_data[34..36].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, out_dim, &mut output)
        .expect("should succeed");
    assert_eq!(output.len(), out_dim);
}

// =========================================================================
// Coverage Tests: Multiple super-block scenarios
// =========================================================================

/// Test fused_q4k_q8k_dot with multiple super-blocks
#[test]
fn test_fused_q4k_q8k_dot_multiple_superblocks() {
    let num_superblocks = 2;
    let mut q4k_data = vec![0u8; 144 * num_superblocks];
    for i in 0..num_superblocks {
        let offset = i * 144;
        q4k_data[offset..offset + 2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    }

    let q8k_scales = vec![1.0f32; 8 * num_superblocks];
    let q8k_quants = vec![1i8; 256 * num_superblocks];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// =========================================================================
// Coverage Tests: f16_to_f32_lut
// =========================================================================

/// Test f16_to_f32_lut lookup table
#[test]
fn test_f16_to_f32_lut_values() {
    // Test specific known values
    let one_bits = half::f16::from_f32(1.0).to_bits();
    let result = f16_to_f32_lut(one_bits);
    assert!((result - 1.0).abs() < 1e-3);

    let neg_bits = half::f16::from_f32(-1.0).to_bits();
    let result = f16_to_f32_lut(neg_bits);
    assert!((result - (-1.0)).abs() < 1e-3);

    let zero_bits = half::f16::from_f32(0.0).to_bits();
    let result = f16_to_f32_lut(zero_bits);
    assert_eq!(result, 0.0);
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock
// =========================================================================

/// Test Q8KSuperBlock quantize basic
#[test]
fn test_q8k_superblock_quantize_basic() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
    assert_eq!(block.quants.len(), 256);
}

/// Test Q8KSuperBlock quantize with varying values
#[test]
fn test_q8k_superblock_quantize_varying() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 128.0) / 128.0;
    }
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
    // Check that extreme values are captured
    assert_ne!(block.quants[0], 0);
    assert_ne!(block.quants[255], 0);
}

/// Test Q8KSuperBlock quantize with near-zero values (edge case)
#[test]
fn test_q8k_superblock_quantize_near_zero() {
    let values = [1e-12f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    // Should use fallback scale
    assert!((block.scale - 1.0 / 127.0).abs() < 1e-6);
}
