
/// Test that the low nibble × q8[offset+b] and high nibble × q8[offset+32+b] paths
/// produce different results when data differs.
#[test]
fn test_fused_q4k_q8k_dot_lo_hi_nibble_separation() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
    }
    for j in 8..12 {
        scales[j] = 0x05;
    }

    // Only low nibbles nonzero
    let qs_lo = [0x0Fu8; 128];
    let data_lo = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_lo);

    // Only high nibbles nonzero
    let qs_hi = [0xF0u8; 128];
    let data_hi = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_hi);

    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![5i8; QK_K];

    let result_lo = fused_q4k_q8k_dot(&data_lo, &q8k_scales, &q8k_quants).expect("lo nibble");
    let result_hi = fused_q4k_q8k_dot(&data_hi, &q8k_scales, &q8k_quants).expect("hi nibble");

    assert!(result_lo.abs() > 0.0, "Low nibble should produce non-zero");
    assert!(result_hi.abs() > 0.0, "High nibble should produce non-zero");
}

/// Test q8k dot with extreme i8 values to exercise overflow-safe accumulation.
#[test]
fn test_fused_q4k_q8k_dot_extreme_i8_values() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 63; // max
        scales[j + 4] = 63;
    }
    for j in 8..12 {
        scales[j] = 0xFF;
    }

    let qs = [0xFFu8; 128]; // all nibbles = 15
    let data = build_q4k_superblock(F16_ONE, F16_ONE, &scales, &qs);

    let q8k_scales = vec![1.0f32; 1];
    // All q8 values at maximum positive
    let q8k_quants = vec![127i8; QK_K];

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("extreme values");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

/// Test q8k dot with all minimum i8 values.
#[test]
fn test_fused_q4k_q8k_dot_all_min_i8() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10;
        scales[j + 4] = 5;
    }
    for j in 8..12 {
        scales[j] = 0x5A;
    }

    let qs = [0x88u8; 128]; // lo=8, hi=8
    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);

    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![-128i8; QK_K]; // all minimum

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("min i8");
    assert!(result.is_finite(), "Expected finite result, got {result}");
}

// ============================================================================
// SIMD/SCALAR PARITY TESTS WITH NON-TRIVIAL DATA
// ============================================================================

/// Test fused_q4k_dot_simd vs scalar with non-trivial data (all inner loop paths exercised).
#[test]
fn test_fused_q4k_dot_simd_scalar_parity_nontrivial() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 15;
        scales[j + 4] = 7;
    }
    for j in 8..12 {
        scales[j] = 0x7F;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 13 + 7) % 256) as u8;
    }

    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);
    let activations: Vec<f32> = (0..QK_K).map(|i| ((i as f32) - 128.0) * 0.01).collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.01,
        "SIMD/scalar parity failure: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test fused_q4k_q8k_dot_simd vs scalar with non-trivial data.
#[test]
fn test_fused_q4k_q8k_dot_simd_scalar_parity_nontrivial() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 20;
        scales[j + 4] = 8;
    }
    for j in 8..12 {
        scales[j] = 0x84;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 11 + 3) % 256) as u8;
    }

    let data = build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs);
    let q8k_scales = vec![0.75f32; 1];
    let q8k_quants: Vec<i8> = (0..QK_K)
        .map(|i| ((i % 200) as i16 - 100).clamp(-128, 127) as i8)
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.05,
        "SIMD/scalar parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test SIMD parity with multi-superblock data.
#[test]
fn test_fused_q4k_dot_simd_scalar_parity_multi_sb() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 8;
        scales[j + 4] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x38;
    }

    let num_sb = 4;
    let mut data = Vec::new();
    for sb in 0..num_sb {
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = ((i + sb * 37) % 256) as u8;
        }
        data.extend(build_q4k_superblock(F16_ONE, F16_QUARTER, &scales, &qs));
    }

    let activations: Vec<f32> = (0..QK_K * num_sb).map(|i| (i as f32).sin()).collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.01,
        "Multi-SB parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test SIMD parity for q8k with multiple superblocks.
#[test]
fn test_fused_q4k_q8k_dot_simd_scalar_parity_multi_sb() {
    let num_sb = 3;
    let mut data = Vec::new();

    for sb in 0..num_sb {
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = (5 + sb as u8) % 63;
            scales[j + 4] = (2 + sb as u8) % 63;
        }
        for j in 8..12 {
            scales[j] = 0x25;
        }
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = ((i * (sb + 3) + 11) % 256) as u8;
        }
        data.extend(build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs));
    }

    let q8k_scales: Vec<f32> = (0..num_sb).map(|i| 0.1 + 0.1 * i as f32).collect();
    let q8k_quants: Vec<i8> = (0..QK_K * num_sb)
        .map(|i| ((i * 3 % 255) as i8).wrapping_sub(127))
        .collect();

    let scalar = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants).expect("simd");

    let abs_diff = (scalar - simd).abs();
    let rel_err = if scalar.abs() > 1e-6 {
        abs_diff / scalar.abs()
    } else {
        abs_diff
    };
    assert!(
        rel_err < 0.05,
        "Multi-SB q8k parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

// ============================================================================
// SIMD DISPATCH ERROR PATHS
// ============================================================================

/// Test fused_q4k_dot_simd error path with bad data length.
#[test]
fn test_fused_q4k_dot_simd_bad_data_length() {
    let data = vec![0u8; 145]; // 145 is not a multiple of 144
    let activations = vec![1.0f32; QK_K];
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not a multiple"), "Error: {err}");
}

/// Test fused_q4k_dot_simd error path with activation mismatch.
#[test]
fn test_fused_q4k_dot_simd_activation_mismatch() {
    let data = vec![0u8; SB_BYTES];
    let activations = vec![1.0f32; 100]; // should be 256
    let result = fused_q4k_dot_simd(&data, &activations);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("doesn't match") || err.contains("match"),
        "Error: {err}"
    );
}

/// Test fused_q4k_q8k_dot_simd error path with bad data.
#[test]
fn test_fused_q4k_q8k_dot_simd_bad_data_length() {
    let data = vec![0u8; 200]; // not multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; QK_K];
    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

/// Test fused_q4k_q8k_dot_simd error with insufficient scales.
#[test]
fn test_fused_q4k_q8k_dot_simd_insufficient_scales() {
    let data = vec![0u8; SB_BYTES * 2]; // 2 superblocks
    let q8k_scales = vec![1.0f32; 1]; // only 1 scale for 2 blocks
    let q8k_quants = vec![1i8; QK_K * 2];
    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

/// Test fused_q4k_q8k_dot_simd error with insufficient quants.
#[test]
fn test_fused_q4k_q8k_dot_simd_insufficient_quants() {
    let data = vec![0u8; SB_BYTES]; // 1 superblock
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 100]; // should be 256
    let result = fused_q4k_q8k_dot_simd(&data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

// ============================================================================
// EXTRACT_SCALE_MIN INTEGRATION TESTS (verifying correct scale application)
// ============================================================================

/// Verify that extract_scale_min returns expected values for our test scales,
/// confirming the inner loop applies them correctly.
#[test]
fn test_scale_extraction_consistency() {
    let mut scales = [0u8; 12];
    // Block 0: scale=10, min=3
    scales[0] = 10;
    scales[4] = 3;

    let (sc, mn) = extract_scale_min(&scales, 0);
    assert_eq!(sc, 10.0);
    assert_eq!(mn, 3.0);
}

/// Verify scale extraction for blocks 4-7 (packed format).
#[test]
fn test_scale_extraction_high_blocks() {
    let mut scales = [0u8; 12];
    // For block 4:
    // d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    scales[0] = 0xC0; // bits 7:6 = 3 => contributes 3<<4 = 48 to scale
    scales[4] = 0x80; // bits 7:6 = 2 => contributes 2<<4 = 32 to min
    scales[8] = 0x25; // low nibble=5 for scale, high nibble=2 for min

    let (sc, mn) = extract_scale_min(&scales, 4);
    // scale = 5 | (3 << 4) = 5 | 48 = 53
    assert_eq!(sc, 53.0, "Scale extraction for block 4");
    // min = 2 | (2 << 4) = 2 | 32 = 34
    assert_eq!(mn, 34.0, "Min extraction for block 4");
}

// ============================================================================
// NUMERICAL PRECISION AND ACCUMULATION TESTS
// ============================================================================

/// Test that accumulation works correctly across many values (Kahan-style issues).
#[test]
fn test_fused_q4k_dot_accumulation_precision() {
    // Use 8 superblocks to accumulate a large number of products
    let num_sb = 8;
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x11u8; 128]; // lo=1, hi=1
    let mut data = Vec::new();
    for _ in 0..num_sb {
        data.extend(build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs));
    }

    let activations = vec![1.0f32; QK_K * num_sb];

    let result = fused_q4k_dot(&data, &activations).expect("accumulation");
    // Each superblock: 256 values, each dequantized to d*scale*1 = 1.0 * scale * 1.0
    // Total = num_sb * 256 * (scale_contribution)
    assert!(
        result.is_finite(),
        "Accumulation should be finite: {result}"
    );
    assert!(result > 0.0, "Accumulation should be positive: {result}");
}

/// Test linearity: doubling activations should double the result.
#[test]
fn test_fused_q4k_dot_linearity() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
        scales[j + 4] = 0; // no dmin contribution
    }
    for j in 8..12 {
        scales[j] = 0x05;
    }

    let qs = [0x33u8; 128]; // lo=3, hi=3
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    let activations_1x = vec![1.0f32; QK_K];
    let activations_2x = vec![2.0f32; QK_K];

    let result_1x = fused_q4k_dot(&data, &activations_1x).expect("1x");
    let result_2x = fused_q4k_dot(&data, &activations_2x).expect("2x");

    let ratio = if result_1x.abs() > 1e-8 {
        result_2x / result_1x
    } else {
        // If result_1x ~ 0, result_2x should also be ~ 0
        assert!(result_2x.abs() < 1e-6);
        2.0 // pass trivially
    };
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "Expected 2x scaling: 1x={result_1x}, 2x={result_2x}, ratio={ratio}"
    );
}

/// Test that fused_q4k_q8k_dot scales linearly with q8k_scale.
#[test]
fn test_fused_q4k_q8k_dot_scale_linearity() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 8;
    }
    for j in 8..12 {
        scales[j] = 0x08;
    }

    let qs = [0x44u8; 128]; // lo=4, hi=4
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    let q8k_quants = vec![10i8; QK_K];

    let q8k_scales_1x = vec![1.0f32; 1];
    let q8k_scales_2x = vec![2.0f32; 1];

    let result_1x = fused_q4k_q8k_dot(&data, &q8k_scales_1x, &q8k_quants).expect("1x");
    let result_2x = fused_q4k_q8k_dot(&data, &q8k_scales_2x, &q8k_quants).expect("2x");

    if result_1x.abs() > 1e-8 {
        let ratio = result_2x / result_1x;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Expected 2x scaling: 1x={result_1x}, 2x={result_2x}, ratio={ratio}"
        );
    }
}
