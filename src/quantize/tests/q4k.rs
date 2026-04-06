
// ============================================================================
// Tests for numerical edge cases
// ============================================================================

#[test]
fn test_q4k_with_inf_scale_p22() {
    // f16 infinity = 0x7C00
    let mut data = generate_q4k_with_scales(1, [0x00, 0x7C], [0x00, 0x00]);

    // Set some non-zero quants
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    // With infinite scale, results will be inf
    for val in &output {
        // Should be inf or finite (not NaN unless quants are 0)
        assert!(!val.is_nan() || val.is_infinite() || val.is_finite());
    }
}

#[test]
fn test_q4k_with_nan_scale_p22() {
    // f16 NaN = 0x7C01
    let mut data = generate_q4k_with_scales(1, [0x01, 0x7C], [0x00, 0x00]);

    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    // With NaN scale, all results should be NaN
    for val in &output {
        assert!(val.is_nan(), "Expected NaN, got {}", val);
    }
}

#[test]
fn test_q8_0_with_inf_scale_p22() {
    let mut data = vec![0u8; 34];

    // Scale = inf (f16 = 0x7C00)
    data[0] = 0x00;
    data[1] = 0x7C;

    // Set non-zero quants
    for i in 0..32 {
        data[2 + i] = 64;
    }

    let output = dequantize_q8_0_block(&data);

    for val in &output {
        assert!(val.is_infinite(), "Expected inf, got {}", val);
    }
}

#[test]
fn test_rope_with_nan_inputs_p22() {
    let half_dim = 4;
    let mut x1 = vec![f32::NAN, 1.0, 2.0, 3.0];
    let mut x2 = vec![4.0, f32::NAN, 5.0, 6.0];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // NaN propagates
    assert!(x1[0].is_nan());
    assert!(x2[1].is_nan());
}

#[test]
fn test_rope_with_inf_inputs_p22() {
    let half_dim = 4;
    let mut x1 = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 3.0];
    let mut x2 = vec![4.0, 5.0, 6.0, f32::INFINITY];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Inf propagates in computations
    assert!(x1[0].is_infinite() || x1[0].is_nan());
    assert!(x1[2].is_infinite() || x1[2].is_nan());
}

// ============================================================================
// Tests for parallel consistency under load
// ============================================================================

#[test]
fn test_q4k_parallel_stress_consistency_p22() {
    // Run many times to catch race conditions
    let data = generate_q4k_with_scales(64, [0x00, 0x3C], [0x00, 0x38]);
    let reference = dequantize_q4_k_simd(&data).expect("test value should be present");

    for iteration in 0..20 {
        let result = dequantize_q4_k_simd(&data).expect("test value should be present");
        for i in 0..result.len() {
            assert!(
                (result[i] - reference[i]).abs() < 1e-10
                    || (result[i].is_nan() && reference[i].is_nan()),
                "Iteration {}: Mismatch at {}: {} vs {}",
                iteration,
                i,
                result[i],
                reference[i]
            );
        }
    }
}

#[test]
fn test_q8_0_parallel_stress_consistency_p22() {
    let data = generate_q8_0_with_scale(128, [0x00, 0x3C]);
    let reference = dequantize_q8_0_simd(&data).expect("test value should be present");

    for iteration in 0..20 {
        let result = dequantize_q8_0_simd(&data).expect("test value should be present");
        for i in 0..result.len() {
            assert!(
                (result[i] - reference[i]).abs() < 1e-10,
                "Iteration {}: Mismatch at {}: {} vs {}",
                iteration,
                i,
                result[i],
                reference[i]
            );
        }
    }
}

// ============================================================================
// Tests for scale extraction edge cases in dequantization
// ============================================================================

#[test]
fn test_q4k_scale_extraction_blocks_4_to_7_p22() {
    // Blocks 4-7 use packed scale extraction with high bits
    let mut sb_data = vec![0u8; 144];

    // d = 1.0, dmin = 0.5
    sb_data[0] = 0x00;
    sb_data[1] = 0x3C;
    sb_data[2] = 0x00;
    sb_data[3] = 0x38;

    // Set scales bytes 0-3 with high bits set (affects blocks 4-7)
    sb_data[4] = 0b11_000001; // scale[0] = 1, high bits = 3
    sb_data[5] = 0b10_000010; // scale[1] = 2, high bits = 2
    sb_data[6] = 0b01_000011; // scale[2] = 3, high bits = 1
    sb_data[7] = 0b00_000100; // scale[3] = 4, high bits = 0

    // Set scales bytes 4-7 (used for mins of blocks 0-3, and combined with high bits for 4-7)
    sb_data[8] = 0b00_010000;  // For block 4: d = (low) | (high << 4) = 0 | 48 = 48
    sb_data[9] = 0b00_100000;  // For block 5
    sb_data[10] = 0b00_110000; // For block 6
    sb_data[11] = 0b00_000001; // For block 7

    // Set remaining scale bytes
    for i in 8..12 {
        sb_data[4 + i] = ((i * 5) % 64) as u8;
    }

    // Fill quants
    for i in 0..128 {
        sb_data[16 + i] = 0x77; // Both nibbles = 7
    }

    let output = dequantize_q4_k_superblock(&sb_data);
    assert_eq!(output.len(), QK_K);

    // Verify output is finite and varies based on scale extraction
    for val in &output {
        assert!(val.is_finite(), "Non-finite value: {}", val);
    }
}

// ============================================================================
// Tests for exact size boundaries
// ============================================================================

#[test]
fn test_q4k_parallel_boundary_1_superblock_p22() {
    let data = generate_q4k_with_scales(1, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_parallel(&data).expect("test value should be present");
    assert_eq!(result.len(), QK_K);
}

#[test]
fn test_q4k_parallel_boundary_2_superblocks_p22() {
    let data = generate_q4k_with_scales(2, [0x00, 0x3C], [0x00, 0x38]);
    let result = dequantize_q4_k_parallel(&data).expect("test value should be present");
    assert_eq!(result.len(), 2 * QK_K);
}

#[test]
fn test_q8_0_parallel_boundary_1_block_p22() {
    let data = generate_q8_0_with_scale(1, [0x00, 0x3C]);
    let result = dequantize_q8_0_parallel(&data).expect("test value should be present");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_q8_0_parallel_boundary_2_blocks_p22() {
    let data = generate_q8_0_with_scale(2, [0x00, 0x3C]);
    let result = dequantize_q8_0_parallel(&data).expect("test value should be present");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_rope_boundary_size_1_p22() {
    let mut x1 = vec![1.0];
    let mut x2 = vec![2.0];
    let cos_vals = vec![0.0];
    let sin_vals = vec![1.0];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // 90 degree: x1' = -x2, x2' = x1
    assert!((x1[0] - (-2.0)).abs() < 1e-6);
    assert!((x2[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_rope_boundary_size_2_p22() {
    let mut x1 = vec![1.0, 3.0];
    let mut x2 = vec![2.0, 4.0];
    let cos_vals = vec![1.0, 0.0];
    let sin_vals = vec![0.0, 1.0];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // First: identity
    assert!((x1[0] - 1.0).abs() < 1e-6);
    assert!((x2[0] - 2.0).abs() < 1e-6);
    // Second: 90 degree
    assert!((x1[1] - (-4.0)).abs() < 1e-6);
    assert!((x2[1] - 3.0).abs() < 1e-6);
}
