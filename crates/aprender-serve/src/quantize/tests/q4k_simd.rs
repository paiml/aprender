
#[test]
fn test_q8_0_parallel_thread_consistency_p19() {
    let data = generate_q8_0_block_data(32);
    let reference = dequantize_q8_0_parallel(&data).expect("test value should be present");

    for run in 0..10 {
        let result = dequantize_q8_0_parallel(&data).expect("test value should be present");
        for i in 0..result.len() {
            assert_eq!(
                result[i], reference[i],
                "Thread inconsistency on run {} at index {}",
                run, i
            );
        }
    }
}

#[test]
fn test_q4k_simd_chunking_boundary_128_p19() {
    // Test at the CHUNK_SIZE boundary (64 super-blocks)
    // Just below threshold
    let data_below = generate_q4k_superblock_data(127);
    let result_below = dequantize_q4_k_simd(&data_below);
    assert!(result_below.is_ok());

    // Just above threshold
    let data_above = generate_q4k_superblock_data(129);
    let result_above = dequantize_q4_k_simd(&data_above);
    assert!(result_above.is_ok());

    // Exactly at threshold
    let data_at = generate_q4k_superblock_data(128);
    let result_at = dequantize_q4_k_simd(&data_at);
    assert!(result_at.is_ok());
}

// ============================================================================
// Tests for numerical edge cases
// ============================================================================

#[test]
fn test_q4k_dequant_with_large_scale_p19() {
    let mut data = generate_q4k_superblock_data(1);

    // Set d to a large f16 value (near max ~65504)
    // 0x7BFF is max normal f16
    data[0] = 0xFF;
    data[1] = 0x7B;

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    for val in &output {
        assert!(val.is_finite(), "Large scale produced non-finite: {}", val);
    }
}

#[test]
fn test_q4k_dequant_with_small_scale_p19() {
    let mut data = generate_q4k_superblock_data(1);

    // Set d to a very small f16 value (subnormal)
    data[0] = 0x01;
    data[1] = 0x00;

    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q8_0_dequant_with_negative_scale_p19() {
    let mut data = vec![0u8; 34];

    // Set scale to -1.0 (f16 = 0xBC00)
    data[0] = 0x00;
    data[1] = 0xBC;

    // Set quants to positive values
    for i in 0..32 {
        data[2 + i] = 64; // +64
    }

    let output = dequantize_q8_0_block(&data);

    // Result should be 64 * (-1.0) = -64.0
    for val in &output {
        assert!((val - (-64.0)).abs() < 0.01, "Expected -64.0, got {}", val);
    }
}

#[test]
fn test_rope_rotation_with_subnormal_values_p19() {
    let half_dim = 4;
    let mut x1 = vec![f32::MIN_POSITIVE / 2.0; half_dim]; // Subnormal
    let mut x2 = vec![f32::MIN_POSITIVE / 2.0; half_dim];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Should produce finite results (possibly zero due to underflow)
    for val in x1.iter().chain(x2.iter()) {
        assert!(
            val.is_finite(),
            "Subnormal rotation produced non-finite: {}",
            val
        );
    }
}

#[test]
fn test_rope_rotation_with_large_values_p19() {
    let half_dim = 4;
    let mut x1 = vec![1e30; half_dim];
    let mut x2 = vec![1e30; half_dim];
    let cos_vals = vec![0.5; half_dim];
    let sin_vals = vec![0.5; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(
            val.is_finite(),
            "Large value rotation produced non-finite: {}",
            val
        );
    }
}

// ============================================================================
// Tests for AVX-512 path (if available)
// ============================================================================

#[test]
fn test_rope_rotation_size_32_avx512_candidate_p19() {
    // 32 elements - 2 AVX-512 iterations if available
    let half_dim = 32;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_rope_rotation_size_64_avx512_multiple_p19() {
    // 64 elements - 4 AVX-512 iterations
    let half_dim = 64;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.01).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.01 + 0.5).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.02).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.02).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_rope_rotation_size_65_avx512_remainder_p19() {
    // 65 elements - 4 AVX-512 + 1 remainder
    let half_dim = 65;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.03).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.03).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}
