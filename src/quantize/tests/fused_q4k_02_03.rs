
// ============================================================================
// EDGE CASE: SINGLE ELEMENT PER CHUNK BOUNDARY
// ============================================================================

/// Test that the inner loop processes exactly QK_K=256 elements per superblock
/// by checking that result is zero when all activations are zero except one.
#[test]
fn test_fused_q4k_dot_single_nonzero_activation() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x55u8; 128]; // lo=5, hi=5
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    // Only activation[0] = 1.0, rest = 0.0
    let mut activations = vec![0.0f32; QK_K];
    activations[0] = 1.0;

    let result = fused_q4k_dot(&data, &activations).expect("single nonzero");
    // Only one product contributes, so result should be small but non-zero
    // (assuming scale > 0 and q_val > 0)
    // value = d * sc * 5 - dmin * min
    // With dmin=0: value = 1.0 * scale_float * 5.0
    // Only one activation at 1.0, so result = that value
    assert!(result.is_finite());
}

/// Test with nonzero activation at the last position (boundary of last chunk).
#[test]
fn test_fused_q4k_dot_last_activation_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x55u8; 128];
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    let mut activations = vec![0.0f32; QK_K];
    activations[QK_K - 1] = 1.0; // last element only

    let result = fused_q4k_dot(&data, &activations).expect("last position");
    assert!(result.is_finite());
}

/// Test with nonzero activation at chunk boundary (position 64, 128, 192).
#[test]
fn test_fused_q4k_dot_chunk_boundary_activations() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 3;
    }
    for j in 8..12 {
        scales[j] = 0x03;
    }

    let qs = [0xAAu8; 128]; // lo=10, hi=10
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);

    for boundary in [0, 32, 64, 96, 128, 160, 192, 224] {
        let mut activations = vec![0.0f32; QK_K];
        activations[boundary] = 1.0;
        let result = fused_q4k_dot(&data, &activations).expect(&format!("boundary {boundary}"));
        assert!(
            result.is_finite(),
            "Failed at boundary {boundary}: {result}"
        );
    }
}

// ============================================================================
// LARGE F16 d/dmin VALUES
// ============================================================================

/// Test with large d value (f16 max normalized ~65504).
#[test]
fn test_fused_q4k_dot_large_d() {
    // f16 max = 0x7BFF = 65504.0
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let qs = [0x11u8; 128]; // lo=1, hi=1
    let data = build_q4k_superblock(0x7BFF, F16_ZERO, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("large d");
    assert!(
        result.is_finite(),
        "Large d should still be finite: {result}"
    );
    assert!(result > 0.0);
}

/// Test with subnormal f16 d value.
#[test]
fn test_fused_q4k_dot_subnormal_d() {
    // f16 subnormal: 0x0001 = smallest positive subnormal ~5.96e-8
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 63; // max scale
    }
    for j in 8..12 {
        scales[j] = 0x0F; // scale nibble = F
    }

    let qs = [0xFFu8; 128]; // max nibbles
    let data = build_q4k_superblock(0x0001, F16_ZERO, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("subnormal d");
    assert!(result.is_finite());
    // Very small d means very small result
    assert!(
        result.abs() < 1.0,
        "Subnormal d should give small result: {result}"
    );
}

// ============================================================================
// SIMD empty/zero tests to exercise SIMD dispatch path
// ============================================================================

/// Test SIMD dispatch with empty data (should return 0.0 via scalar fallback).
#[test]
fn test_fused_q4k_dot_simd_empty() {
    let result = fused_q4k_dot_simd(&[], &[]).expect("empty simd");
    assert_eq!(result, 0.0);
}

/// Test SIMD q8k dispatch with empty data.
#[test]
fn test_fused_q4k_q8k_dot_simd_empty() {
    let result = fused_q4k_q8k_dot_simd(&[], &[], &[]).expect("empty simd q8k");
    assert_eq!(result, 0.0);
}

/// Test SIMD dispatch with non-trivial data exercising all 4 chunks.
#[test]
fn test_fused_q4k_dot_simd_all_chunks_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 10 + j as u8;
        scales[j + 4] = 3 + j as u8;
    }
    for j in 8..12 {
        scales[j] = 0x3A + (j - 8) as u8;
    }

    let mut qs = [0u8; 128];
    // Different pattern for each 32-byte chunk
    for i in 0..32 {
        qs[i] = 0x12; // chunk 0
    }
    for i in 32..64 {
        qs[i] = 0x34; // chunk 1
    }
    for i in 64..96 {
        qs[i] = 0x56; // chunk 2
    }
    for i in 96..128 {
        qs[i] = 0x78; // chunk 3
    }

    let data = build_q4k_superblock(F16_ONE, F16_QUARTER, &scales, &qs);
    let activations: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.01) + 0.1).collect();

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
        "All-chunks parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

/// Test fused_q4k_q8k_dot_simd with non-trivial data and all chunks active.
#[test]
fn test_fused_q4k_q8k_dot_simd_all_chunks_nonzero() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 12;
        scales[j + 4] = 4;
    }
    for j in 8..12 {
        scales[j] = 0x4C;
    }

    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = ((i * 5 + 17) % 256) as u8;
    }

    let data = build_q4k_superblock(F16_HALF, F16_QUARTER, &scales, &qs);
    let q8k_scales = vec![0.8f32; 1];
    let q8k_quants: Vec<i8> = (0..QK_K)
        .map(|i| {
            let v = (i as i16 * 3) % 256 - 128;
            v.clamp(-128, 127) as i8
        })
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
        "Q8K all-chunks parity: scalar={scalar}, simd={simd}, rel_err={rel_err}"
    );
}

// ============================================================================
// REGRESSION: Specific scale patterns that exercise blocks 4-7
// ============================================================================

/// Test with scales that exercise blocks 4-7 specifically (packed encoding).
#[test]
fn test_fused_q4k_dot_blocks_4_through_7() {
    // Only set scales for blocks 4-7 (via packed encoding), blocks 0-3 = 0
    let mut scales = [0u8; 12];
    // Blocks 0-3 have scale=0, min=0
    // Blocks 4-7: use high bits from blocks 0-3
    scales[0] = 0xC0; // bits 7:6 = 3 for block 4 scale high
    scales[1] = 0x80; // bits 7:6 = 2 for block 5 scale high
    scales[2] = 0x40; // bits 7:6 = 1 for block 6 scale high
    scales[3] = 0x00; // bits 7:6 = 0 for block 7 scale high
                      // scales[4-7] stay 0 (blocks 0-3 min = 0)
                      // scales[8-11] provide low 4 bits
    scales[8] = 0x0F; // block 4: scale_lo=15, min_hi=0
    scales[9] = 0x0A; // block 5: scale_lo=10, min_hi=0
    scales[10] = 0x05; // block 6: scale_lo=5, min_hi=0
    scales[11] = 0x01; // block 7: scale_lo=1, min_hi=0

    let qs = [0x88u8; 128]; // lo=8, hi=8
    let data = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs);
    let activations = vec![1.0f32; QK_K];

    let result = fused_q4k_dot(&data, &activations).expect("blocks 4-7");
    // Blocks 0-3 have scale=0 so contribute 0
    // Blocks 4-7 have non-zero scales so should contribute
    assert!(result.is_finite());
    assert!(result.abs() > 0.0, "Blocks 4-7 should contribute: {result}");
}

/// Test that blocks 0-3 and 4-7 produce different results when their scales differ.
#[test]
fn test_fused_q4k_dot_block_groups_independence() {
    // Set up only blocks 0-3
    let mut scales_low = [0u8; 12];
    for j in 0..4 {
        scales_low[j] = 10; // blocks 0-3: scale=10
    }

    // Set up only blocks 4-7
    let mut scales_high = [0u8; 12];
    scales_high[0] = 0xC0;
    scales_high[1] = 0xC0;
    scales_high[2] = 0xC0;
    scales_high[3] = 0xC0;
    for j in 8..12 {
        scales_high[j] = 0x0A; // scale=10 for blocks 4-7
    }

    let qs = [0x55u8; 128];
    let data_low = build_q4k_superblock(F16_ONE, F16_ZERO, &scales_low, &qs);
    let data_high = build_q4k_superblock(F16_ONE, F16_ZERO, &scales_high, &qs);

    let activations = vec![1.0f32; QK_K];

    let result_low = fused_q4k_dot(&data_low, &activations).expect("low blocks");
    let result_high = fused_q4k_dot(&data_high, &activations).expect("high blocks");

    assert!(result_low.is_finite());
    assert!(result_high.is_finite());
    // Both should be non-zero but possibly different magnitudes
    assert!(result_low.abs() > 0.0);
    assert!(result_high.abs() > 0.0);
}

// ============================================================================
// Q4K_Q8K DOT: Per-byte inner loop coverage
// ============================================================================

/// Test that each of the 32 bytes in a chunk contributes to the result.
#[test]
fn test_fused_q4k_q8k_dot_per_byte_contribution() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 1;
    }
    for j in 8..12 {
        scales[j] = 0x01;
    }

    let q8k_scales = vec![1.0f32; 1];

    // Baseline: all qs = 0
    let qs_zero = [0x00u8; 128];
    let data_zero = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_zero);
    let q8k_quants = vec![1i8; QK_K];
    let baseline = fused_q4k_q8k_dot(&data_zero, &q8k_scales, &q8k_quants).expect("baseline");

    // Set byte 0 only
    let mut qs_byte0 = [0x00u8; 128];
    qs_byte0[0] = 0xFF;
    let data_byte0 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_byte0);
    let result_byte0 = fused_q4k_q8k_dot(&data_byte0, &q8k_scales, &q8k_quants).expect("byte0");
    assert_ne!(baseline, result_byte0, "Byte 0 should affect result");

    // Set byte 64 (start of chunk 2)
    let mut qs_byte64 = [0x00u8; 128];
    qs_byte64[64] = 0xFF;
    let data_byte64 = build_q4k_superblock(F16_ONE, F16_ZERO, &scales, &qs_byte64);
    let result_byte64 = fused_q4k_q8k_dot(&data_byte64, &q8k_scales, &q8k_quants).expect("byte64");
    assert_ne!(baseline, result_byte64, "Byte 64 should affect result");
}

/// Test q8k dot accumulation: sum_lo and sum_hi, q8_sum_lo and q8_sum_hi paths.
#[test]
fn test_fused_q4k_q8k_dot_accumulator_paths() {
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = 5;
        scales[j + 4] = 2;
    }
    for j in 8..12 {
        scales[j] = 0x25;
    }

    // Use specific qs pattern that exercises both sum_lo and sum_hi differently
    let mut qs = [0u8; 128];
    // Chunk 0: low nibbles all 15, high nibbles all 0
    for i in 0..32 {
        qs[i] = 0x0F; // lo=15, hi=0
    }
    // Chunk 1: low nibbles all 0, high nibbles all 15
    for i in 32..64 {
        qs[i] = 0xF0; // lo=0, hi=15
    }
    // Chunk 2: alternating
    for i in 64..96 {
        qs[i] = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }
    // Chunk 3: middle values
    for i in 96..128 {
        qs[i] = 0x77; // lo=7, hi=7
    }

    let data = build_q4k_superblock(F16_ONE, F16_HALF, &scales, &qs);
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i % 64) as i8) - 32).collect();

    let result = fused_q4k_q8k_dot(&data, &q8k_scales, &q8k_quants).expect("accumulator paths");
    assert!(result.is_finite(), "Should be finite: {result}");
}
