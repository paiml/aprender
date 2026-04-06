
#[test]
fn test_extract_scale_min_all_blocks_max_values() {
    // Test with maximum 6-bit values (63)
    let scales: [u8; 12] = [
        0b11_111111, // block 0 scale = 63, high bits = 3 for block 4
        0b11_111111, // block 1 scale = 63, high bits = 3 for block 5
        0b11_111111, // block 2 scale = 63, high bits = 3 for block 6
        0b11_111111, // block 3 scale = 63, high bits = 3 for block 7
        0b11_111111, // block 0 min = 63, high bits = 3 for block 4 min
        0b11_111111, // block 1 min = 63, high bits = 3 for block 5 min
        0b11_111111, // block 2 min = 63, high bits = 3 for block 6 min
        0b11_111111, // block 3 min = 63, high bits = 3 for block 7 min
        0b1111_1111, // block 4 low bits
        0b1111_1111, // block 5 low bits
        0b1111_1111, // block 6 low bits
        0b1111_1111, // block 7 low bits
    ];

    // Blocks 0-3: simple extraction, scale = min = 63
    for i in 0..4 {
        let (s, m) = extract_scale_min(&scales, i);
        assert_eq!(s, 63.0, "Block {} scale should be 63", i);
        assert_eq!(m, 63.0, "Block {} min should be 63", i);
    }

    // Blocks 4-7: packed extraction with max values
    for i in 4..8 {
        let (s, m) = extract_scale_min(&scales, i);
        // scale = 15 | (3 << 4) = 15 + 48 = 63
        assert_eq!(s, 63.0, "Block {} scale should be 63", i);
        // min = 15 | (3 << 4) = 63
        assert_eq!(m, 63.0, "Block {} min should be 63", i);
    }
}

// =============================================================================
// extract_scale_min_from_slice Additional Tests
// =============================================================================

#[test]
fn test_extract_scale_min_from_slice_odd_index_3() {
    let mut scales = [0u8; 12];
    // For idx=3: scale_idx=1, min_idx=5
    // scale = (scales[1] >> 6) | ((scales[3] & 0x0F) << 2)
    // min = (scales[5] >> 6) | ((scales[7] & 0x0F) << 2)
    scales[1] = 0b10_000000; // high 2 bits = 2
    scales[3] = 0b0000_0111; // low 4 bits = 7
    scales[5] = 0b11_000000; // high 2 bits = 3
    scales[7] = 0b0000_1010; // low 4 bits = 10

    let (s, m) = extract_scale_min_from_slice(&scales, 3);
    // scale = 2 | (7 << 2) = 2 + 28 = 30
    assert_eq!(s, 30.0, "idx=3 scale");
    // min = 3 | (10 << 2) = 3 + 40 = 43
    assert_eq!(m, 43.0, "idx=3 min");
}

#[test]
fn test_extract_scale_min_from_slice_odd_index_5() {
    let mut scales = [0u8; 12];
    // For idx=5: scale_idx=2, min_idx=6
    scales[2] = 0b01_000000; // high 2 bits = 1
    scales[4] = 0b0000_1100; // low 4 bits = 12
    scales[6] = 0b11_000000; // high 2 bits = 3
    scales[8] = 0b0000_0001; // low 4 bits = 1

    let (s, m) = extract_scale_min_from_slice(&scales, 5);
    // scale = 1 | (12 << 2) = 1 + 48 = 49
    assert_eq!(s, 49.0, "idx=5 scale");
    // min = 3 | (1 << 2) = 3 + 4 = 7
    assert_eq!(m, 7.0, "idx=5 min");
}

#[test]
fn test_extract_scale_min_from_slice_odd_index_7() {
    let mut scales = [0u8; 12];
    // For idx=7: scale_idx=3, min_idx=7
    scales[3] = 0b00_000000; // high 2 bits = 0
    scales[5] = 0b0000_1111; // low 4 bits = 15
    scales[7] = 0b10_000000; // high 2 bits = 2
    scales[9] = 0b0000_0011; // low 4 bits = 3

    let (s, m) = extract_scale_min_from_slice(&scales, 7);
    // scale = 0 | (15 << 2) = 60
    assert_eq!(s, 60.0, "idx=7 scale");
    // min = 2 | (3 << 2) = 2 + 12 = 14
    assert_eq!(m, 14.0, "idx=7 min");
}

// =============================================================================
// InterleavedQ4K Additional Tests
// =============================================================================

#[test]
fn test_interleaved_q4k_scales_and_dmin_extraction() {
    // Test that scales and dmin are correctly extracted
    let mut data = vec![0u8; 144];

    // d = 0.5 (f16: 0x3800)
    data[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    // dmin = 0.25 (f16: 0x3400)
    data[2..4].copy_from_slice(&0x3400u16.to_le_bytes());

    // Set scale bytes
    for i in 4..16 {
        data[i] = (i - 4) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    assert!((interleaved.d[0] - 0.5).abs() < 1e-3);
    assert!((interleaved.dmin[0] - 0.25).abs() < 1e-3);

    // Check scales were copied
    for (i, &s) in interleaved.scales.iter().enumerate() {
        assert_eq!(s, i as u8);
    }
}

#[test]
fn test_interleaved_q4k_qs_copy() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0

    // Set qs with recognizable pattern
    for i in 16..144 {
        data[i] = ((i - 16) % 256) as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    // Check qs were copied correctly
    for (i, &q) in interleaved.qs.iter().enumerate() {
        assert_eq!(q, (i % 256) as u8);
    }
}

#[test]
fn test_interleaved_q4k_dot_with_all_max_nibbles() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0

    // Set all scales to 1
    for i in 4..16 {
        data[i] = 1;
    }

    // Set all qs to 0xFF (max nibbles: low=15, high=15)
    for i in 16..144 {
        data[i] = 0xFF;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite(), "Result should be finite: {}", result);
    assert!(result > 0.0, "Result should be positive with max nibbles");
}

#[test]
fn test_interleaved_q4k_dot_with_zero_d() {
    let mut data = vec![0u8; 144];
    // d = 0.0 (f16: 0x0000)
    data[0..2].copy_from_slice(&0x0000u16.to_le_bytes());
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot should work");
    assert_eq!(result, 0.0, "Zero d should give zero result");
}

#[test]
fn test_interleaved_q4k_dot_three_superblocks() {
    let mut data = vec![0u8; 144 * 3];

    // Super-block 0: d = 1.0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // Super-block 1: d = 2.0
    data[144..146].copy_from_slice(&0x4000u16.to_le_bytes());

    // Super-block 2: d = 0.5
    data[288..290].copy_from_slice(&0x3800u16.to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), 768);

    let activations = vec![0.1f32; 768];
    let result = interleaved.dot(&activations).expect("dot should work");
    assert!(result.is_finite());
}

// =============================================================================
// fused_q4_0_q8_0_dot_simd Boundary Tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_simd_exactly_256_elements() {
    // 256 elements = 8 blocks, triggers 4-block unrolling
    let in_dim = 256;
    let num_blocks = 8;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            q4_data[start + i] = 0x44; // q_low=4, q_high=4
        }
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_exactly_128_elements() {
    // 128 elements = 4 blocks, at the 4-block unroll threshold
    let in_dim = 128;
    let num_blocks = 4;
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

#[test]
fn test_fused_q4_0_q8_0_dot_simd_5_blocks() {
    // 5 blocks = 160 elements, tests remainder handling after 4-block
    let in_dim = 160;
    let num_blocks = 5;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            q4_data[start + i] = 0x55;
        }
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![2i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_9_blocks() {
    // 9 blocks = 288 elements, tests 4-block unroll with remainder
    let in_dim = 288;
    let num_blocks = 9;
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
// Constants Tests
// =============================================================================

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant() {
    assert_eq!(QK_K, 256);
}

// =============================================================================
// Q8_0Block Additional Tests
// =============================================================================

#[test]
fn test_q8_0block_quantize_with_zero_block() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Fallback scale = 1.0 / 127.0
    assert!((block.scale - 1.0 / 127.0).abs() < 1e-10);

    // All quants should be 0
    for q in &block.quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_q8_0block_quantize_negative_dominant() {
    let mut values = [0.0f32; 32];
    values[0] = -100.0;

    let block = Q8_0Block::quantize(&values);

    // Scale should be based on the max abs value
    assert!((block.scale - 100.0 / 127.0).abs() < 1e-3);

    // First quant should be -127 (clamped)
    assert_eq!(block.quants[0], -127);
}

#[test]
fn test_q8_0block_debug() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let debug_str = format!("{:?}", block);

    assert!(debug_str.contains("Q8_0Block"));
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

#[test]
fn test_q8_0block_clone() {
    let values = [5.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let cloned = block.clone();

    assert_eq!(block.scale, cloned.scale);
    assert_eq!(block.quants, cloned.quants);
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar Additional Tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_boundary_in_dim() {
    // Test with in_dim that isn't a multiple of 32
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    for i in 2..34 {
        q8_weight_data[i] = 10u8;
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![5i8; 32];

    // Test with in_dim = 20 (less than block size)
    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 20);
    assert!(result.is_finite());
    // Should only sum first 20 products
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_exact_block() {
    // in_dim exactly 32
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 2..34 {
        q8_weight_data[i] = 1u8;
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // Expected: 1.0 * 1.0 * (1 * 1 * 32) = 32
    assert!((result - 32.0).abs() < 1.0);
}

// =============================================================================
// Parallel Matvec Large Dimension Tests (Parallel Path)
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_above_threshold() {
    // 2048 outputs should trigger parallel path (threshold is 1024)
    let in_dim = 32;
    let out_dim = 2048;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![0.01f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);

    // All outputs should be finite
    for v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_large() {
    // 1024 outputs with parallel path
    let in_dim = 32;
    let out_dim = 1024;
    let bytes_per_row = 34;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        let start = row * bytes_per_row;
        weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![0.1f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}
