
#[test]
fn test_fused_q4_0_q8_0_dot_simd_mod_odd_blocks() {
    // in_dim = 96 (3 blocks) - tests remainder handling
    let in_dim = 96;
    let num_blocks = 3;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar additional edge cases
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_zero_scales() {
    let mut q8_weight_data = vec![0u8; 34];
    // scale = 0.0 (f16: 0x0000)
    q8_weight_data[0..2].copy_from_slice(&0x0000u16.to_le_bytes());
    for i in 2..34 {
        q8_weight_data[i] = 127u8; // max positive
    }

    let q8_act_scales = vec![0.0f32];
    let q8_act_quants = vec![127i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    assert_eq!(result, 0.0, "Zero scales should give zero result");
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_max_positive() {
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    for i in 2..34 {
        q8_weight_data[i] = 127u8; // max positive i8
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![127i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // Expected: 1.0 * 1.0 * (127 * 127 * 32) = 516128
    let expected = 127.0 * 127.0 * 32.0;
    assert!(
        (result - expected).abs() < 1.0,
        "Max positive: expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_all_negative() {
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    for i in 2..34 {
        q8_weight_data[i] = (-100i8) as u8;
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![-50i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // Expected: 1.0 * 1.0 * (-100 * -50 * 32) = 160000 (positive)
    assert!(result > 0.0, "Negative × negative should be positive");
}

// =============================================================================
// Parallel matvec success paths with boundary dimensions
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_exact_threshold() {
    // out_dim = 1024 exactly (at parallel threshold)
    let in_dim = 32;
    let out_dim = 1024;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![0.1f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_small_success() {
    // Small matrix - sequential path
    let in_dim = 64;
    let out_dim = 8;
    let bytes_per_row = 36; // 2 blocks per row

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        for block in 0..2 {
            let start = row * bytes_per_row + block * 18;
            weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);

    // All outputs should be finite
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Output {} should be finite: {}", i, v);
    }
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_mod_success() {
    let in_dim = 32;
    let out_dim = 16;
    let bytes_per_row = 34;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        // Set scale to 0.5
        weight_data[start..start + 2].copy_from_slice(&0x3800u16.to_le_bytes());
        // Set quants
        for i in 2..34 {
            weight_data[start + i] = ((row + i) % 128) as u8;
        }
    }

    let activations = vec![0.5f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_mod_success() {
    let in_dim = 64;
    let out_dim = 8;
    let bytes_per_row = 68; // 2 blocks

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        for block in 0..2 {
            let start = row * bytes_per_row + block * 34;
            weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
    }

    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Output {} should be finite", i);
    }
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_mod_success() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            weight_data[start + i] = 0x88;
        }
    }

    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Output {} should be finite", i);
    }
}

// =============================================================================
// extract_scale_min boundary tests
// =============================================================================

#[test]
fn test_extract_scale_min_mod_zeros() {
    let scales = [0u8; 12];

    for i in 0..8 {
        let (s, m) = extract_scale_min(&scales, i);
        assert_eq!(s, 0.0, "Block {} scale should be 0", i);
        assert_eq!(m, 0.0, "Block {} min should be 0", i);
    }
}

#[test]
fn test_extract_scale_min_mod_block_4() {
    // Block 4 uses packed format
    let mut scales = [0u8; 12];
    // For block 4:
    // d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    scales[0] = 0b11_000000; // high 2 bits = 3
    scales[4] = 0b10_000000; // high 2 bits = 2
    scales[8] = 0b0101_0111; // low=7, high=5

    let (s, m) = extract_scale_min(&scales, 4);
    // scale = 7 | (3 << 4) = 7 + 48 = 55
    assert_eq!(s, 55.0, "Block 4 scale");
    // min = 5 | (2 << 4) = 5 + 32 = 37
    assert_eq!(m, 37.0, "Block 4 min");
}

// =============================================================================
// InterleavedQ4K dot with various scale values
// =============================================================================

#[test]
fn test_interleaved_q4k_dot_large_scales() {
    let mut data = vec![0u8; 144];

    // d = 100.0 (large scale) - f16 for 100.0 is approximately 0x5640
    data[0..2].copy_from_slice(&0x5640u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0

    // Set some scales
    for i in 4..16 {
        data[i] = 32; // mid-range scale
    }

    // Set qs
    for i in 16..144 {
        data[i] = 0x77; // q_low=7, q_high=7
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![0.01f32; 256]; // small activations to prevent overflow

    let result = interleaved.dot(&activations).expect("test value should be present");
    assert!(result.is_finite(), "Large scale result should be finite");
}

#[test]
fn test_interleaved_q4k_dot_small_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    // Very small activations
    let activations = vec![1e-6f32; 256];
    let result = interleaved.dot(&activations).expect("test value should be present");
    assert!(result.is_finite());
    assert!(
        result.abs() < 0.01,
        "Small activations should give small result"
    );
}

// =============================================================================
// Q8_0Block roundtrip with uniform values
// =============================================================================

#[test]
fn test_q8_0block_mod_uniform_values() {
    let values = [5.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    // All values should be approximately equal
    let first = dequant[0];
    for (i, &v) in dequant.iter().enumerate() {
        assert!(
            (v - first).abs() < 0.1,
            "Uniform values: index {} differs: {} vs {}",
            i,
            v,
            first
        );
    }
}

#[test]
fn test_q8_0block_mod_alternating_values() {
    let values: [f32; 32] = std::array::from_fn(|i| if i % 2 == 0 { 10.0 } else { -10.0 });
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    // Check alternating pattern is preserved
    for (i, &v) in dequant.iter().enumerate() {
        if i % 2 == 0 {
            assert!(v > 0.0, "Even index {} should be positive: {}", i, v);
        } else {
            assert!(v < 0.0, "Odd index {} should be negative: {}", i, v);
        }
    }
}

// =============================================================================
// Q8KSuperBlock additional edge cases
// =============================================================================

#[test]
fn test_q8k_superblock_mod_single_value_dominates() {
    let mut values = [0.0f32; 256];
    values[0] = 1000.0; // One very large value

    let block = Q8KSuperBlock::quantize(&values);

    // The large value should clamp to 127
    assert_eq!(block.quants[0], 127);

    // Other values should be near zero
    for i in 1..256 {
        assert!(
            block.quants[i].abs() <= 1,
            "Index {} should be near zero: {}",
            i,
            block.quants[i]
        );
    }
}

#[test]
fn test_q8k_superblock_mod_negative_dominates() {
    let mut values = [0.0f32; 256];
    values[100] = -500.0;

    let block = Q8KSuperBlock::quantize(&values);

    // The large negative should clamp to extreme negative (-127 or -128)
    assert!(block.quants[100] <= -127);
}

// =============================================================================
// Empty input edge cases
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_mod_empty() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

#[test]
fn test_dequantize_q8_blocks_mod_empty() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = dequantize_q8_blocks(&blocks);
    assert!(result.is_empty());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_mod_single_value() {
    // Test with single partial element (edge case)
    let q4_data: Vec<u8> = vec![0; 18];
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![0i8; 32];

    // in_dim = 1 means we only use first value
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 1);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_mod_single_value() {
    let q8_weight_data: Vec<u8> = vec![0; 34];
    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![0i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 1);
    assert!(result.is_finite());
}
