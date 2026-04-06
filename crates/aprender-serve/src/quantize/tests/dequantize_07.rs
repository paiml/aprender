
// ============================================================================
// Tests for dequantize_q8_0_simd
// ============================================================================

#[test]
fn test_dequantize_q8_0_simd_single_block_p19() {
    let data = generate_q8_0_block_data(1);
    let result = dequantize_q8_0_simd(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), 32);
}

#[test]
fn test_dequantize_q8_0_simd_multiple_blocks_p19() {
    let num_blocks = 16;
    let data = generate_q8_0_block_data(num_blocks);
    let result = dequantize_q8_0_simd(&data);

    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), num_blocks * 32);
}

#[test]
fn test_dequantize_q8_0_simd_parity_with_parallel_p19() {
    let data = generate_q8_0_block_data(8);

    let simd_result = dequantize_q8_0_simd(&data).expect("test value should be present");
    let parallel_result = dequantize_q8_0_parallel(&data).expect("test value should be present");

    assert_eq!(simd_result.len(), parallel_result.len());

    for i in 0..simd_result.len() {
        let diff = (simd_result[i] - parallel_result[i]).abs();
        assert!(
            diff < 1e-6,
            "SIMD/parallel mismatch at {}: simd={}, parallel={}",
            i,
            simd_result[i],
            parallel_result[i]
        );
    }
}

#[test]
fn test_dequantize_q8_0_simd_invalid_size_p19() {
    let data = vec![0u8; 50]; // Not 34 * n
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_err());
}

// ============================================================================
// Tests for dequantize_q8_0_block (internal helper)
// ============================================================================

#[test]
fn test_dequantize_q8_0_block_basic_p19() {
    let data = generate_q8_0_block_data(1);
    let block_data = &data[0..34];

    let output = dequantize_q8_0_block(block_data);
    assert_eq!(output.len(), 32);

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q8_0_block_zero_scale_p19() {
    // Block with zero scale
    let mut block_data = vec![0u8; 34];
    // Scale = 0 (f16)
    block_data[0] = 0x00;
    block_data[1] = 0x00;

    // Set some non-zero quants
    for i in 0..32 {
        block_data[2 + i] = i as u8;
    }

    let output = dequantize_q8_0_block(&block_data);
    assert_eq!(output.len(), 32);

    // With zero scale, all outputs should be 0
    for val in &output {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_dequantize_q8_0_block_max_values_p19() {
    let mut block_data = vec![0u8; 34];

    // Scale = 1.0 (f16 = 0x3C00)
    block_data[0] = 0x00;
    block_data[1] = 0x3C;

    // Set all quants to 127 (max positive i8)
    for i in 0..32 {
        block_data[2 + i] = 127;
    }

    let output = dequantize_q8_0_block(&block_data);

    // All values should be 127 * 1.0 = 127.0
    for val in &output {
        assert!((val - 127.0).abs() < 0.01, "Expected 127.0, got {}", val);
    }
}

#[test]
fn test_dequantize_q8_0_block_min_values_p19() {
    let mut block_data = vec![0u8; 34];

    // Scale = 1.0 (f16 = 0x3C00)
    block_data[0] = 0x00;
    block_data[1] = 0x3C;

    // Set all quants to -128 (min i8 as u8 = 0x80 = 128)
    for i in 0..32 {
        block_data[2 + i] = 0x80;
    }

    let output = dequantize_q8_0_block(&block_data);

    // All values should be -128 * 1.0 = -128.0
    for val in &output {
        assert!(
            (val - (-128.0)).abs() < 0.01,
            "Expected -128.0, got {}",
            val
        );
    }
}

#[test]
fn test_dequantize_q8_0_block_alternating_signs_p19() {
    let mut block_data = vec![0u8; 34];

    // Scale = 0.5 (f16 = 0x3800)
    block_data[0] = 0x00;
    block_data[1] = 0x38;

    // Alternating positive (64) and negative (-64 = 0xC0 = 192)
    for i in 0..32 {
        block_data[2 + i] = if i % 2 == 0 { 64 } else { 192 };
    }

    let output = dequantize_q8_0_block(&block_data);

    for (i, &val) in output.iter().enumerate() {
        if i % 2 == 0 {
            assert!((val - 32.0).abs() < 0.01, "Expected 32.0, got {}", val);
        } else {
            assert!((val - (-32.0)).abs() < 0.01, "Expected -32.0, got {}", val);
        }
    }
}

// ============================================================================
// Tests for apply_rope_rotation_simd
// ============================================================================

#[test]
fn test_apply_rope_rotation_simd_basic_p19() {
    let half_dim = 4;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Verify finite results
    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_size_8_p19() {
    // Exactly 8 elements - minimum for AVX2 SIMD path
    let half_dim = 8;
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
fn test_apply_rope_rotation_simd_size_16_p19() {
    // 16 elements - 2 SIMD iterations
    let half_dim = 16;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32 * 0.1).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1) + 1.0).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_size_17_remainder_p19() {
    // 17 elements - tests remainder handling (2*8 + 1)
    let half_dim = 17;
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
fn test_apply_rope_rotation_simd_size_7_scalar_fallback_p19() {
    // 7 elements - should use scalar fallback (< 8)
    let half_dim = 7;
    let mut x1: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..half_dim).map(|i| (i + half_dim) as f32).collect();
    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_simd_identity_p19() {
    // Identity rotation: cos=1, sin=0
    let half_dim = 8;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let original_x1 = x1.clone();
    let original_x2 = x2.clone();

    let cos_vals = vec![1.0; half_dim];
    let sin_vals = vec![0.0; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: x1' = x1*1 - x2*0 = x1, x2' = x1*0 + x2*1 = x2
    for i in 0..half_dim {
        assert!((x1[i] - original_x1[i]).abs() < 1e-6, "x1 changed at {}", i);
        assert!((x2[i] - original_x2[i]).abs() < 1e-6, "x2 changed at {}", i);
    }
}

#[test]
fn test_apply_rope_rotation_simd_90_degrees_p19() {
    // 90 degree rotation: cos=0, sin=1
    let half_dim = 8;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let original_x1 = x1.clone();
    let original_x2 = x2.clone();

    let cos_vals = vec![0.0; half_dim];
    let sin_vals = vec![1.0; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=0, sin=1: x1' = x1*0 - x2*1 = -x2, x2' = x1*1 + x2*0 = x1
    for i in 0..half_dim {
        assert!(
            (x1[i] - (-original_x2[i])).abs() < 1e-6,
            "x1[{}]: expected {}, got {}",
            i,
            -original_x2[i],
            x1[i]
        );
        assert!(
            (x2[i] - original_x1[i]).abs() < 1e-6,
            "x2[{}]: expected {}, got {}",
            i,
            original_x1[i],
            x2[i]
        );
    }
}

#[test]
fn test_apply_rope_rotation_simd_180_degrees_p19() {
    // 180 degree rotation: cos=-1, sin=0
    let half_dim = 8;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x2 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let original_x1 = x1.clone();
    let original_x2 = x2.clone();

    let cos_vals = vec![-1.0; half_dim];
    let sin_vals = vec![0.0; half_dim];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=-1, sin=0: x1' = x1*(-1) - x2*0 = -x1, x2' = x1*0 + x2*(-1) = -x2
    for i in 0..half_dim {
        assert!((x1[i] - (-original_x1[i])).abs() < 1e-6);
        assert!((x2[i] - (-original_x2[i])).abs() < 1e-6);
    }
}

#[test]
fn test_apply_rope_rotation_simd_parity_with_scalar_p19() {
    let half_dim = 32;
    let mut simd_x1: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.1).collect();
    let mut simd_x2: Vec<f32> = (0..half_dim).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let mut scalar_x1 = simd_x1.clone();
    let mut scalar_x2 = simd_x2.clone();

    let cos_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_vals: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    apply_rope_rotation_simd(&mut simd_x1, &mut simd_x2, &cos_vals, &sin_vals);
    apply_rope_rotation_scalar(&mut scalar_x1, &mut scalar_x2, &cos_vals, &sin_vals);

    for i in 0..half_dim {
        let diff_x1 = (simd_x1[i] - scalar_x1[i]).abs();
        let diff_x2 = (simd_x2[i] - scalar_x2[i]).abs();
        assert!(
            diff_x1 < 1e-5,
            "x1 SIMD/scalar mismatch at {}: simd={}, scalar={}",
            i,
            simd_x1[i],
            scalar_x1[i]
        );
        assert!(
            diff_x2 < 1e-5,
            "x2 SIMD/scalar mismatch at {}: simd={}, scalar={}",
            i,
            simd_x2[i],
            scalar_x2[i]
        );
    }
}

// ============================================================================
// Tests for apply_rope_rotation_scalar (internal helper)
// ============================================================================

#[test]
fn test_apply_rope_rotation_scalar_basic_p19() {
    let _half_dim = 4;
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals = vec![1.0, 0.866, 0.5, 0.0]; // cos(0), cos(30), cos(60), cos(90)
    let sin_vals = vec![0.0, 0.5, 0.866, 1.0]; // sin(0), sin(30), sin(60), sin(90)

    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

    for val in x1.iter().chain(x2.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_apply_rope_rotation_scalar_empty_p19() {
    let mut x1: Vec<f32> = vec![];
    let mut x2: Vec<f32> = vec![];
    let cos_vals: Vec<f32> = vec![];
    let sin_vals: Vec<f32> = vec![];

    // Should not panic on empty input
    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);
    assert!(x1.is_empty());
    assert!(x2.is_empty());
}

#[test]
fn test_apply_rope_rotation_scalar_single_element_p19() {
    let mut x1 = vec![1.0];
    let mut x2 = vec![2.0];
    let cos_vals = vec![0.5];
    let sin_vals = vec![0.866];

    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // x1' = 1*0.5 - 2*0.866 = 0.5 - 1.732 = -1.232
    // x2' = 1*0.866 + 2*0.5 = 0.866 + 1.0 = 1.866
    assert!((x1[0] - (-1.232)).abs() < 0.01);
    assert!((x2[0] - 1.866).abs() < 0.01);
}

#[test]
fn test_apply_rope_rotation_scalar_negative_values_p19() {
    let mut x1 = vec![-1.0, -2.0, -3.0, -4.0];
    let mut x2 = vec![-5.0, -6.0, -7.0, -8.0];
    let cos_vals = vec![1.0; 4];
    let sin_vals = vec![0.0; 4];

    apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Identity rotation preserves values
    assert!((x1[0] - (-1.0)).abs() < 1e-6);
    assert!((x2[0] - (-5.0)).abs() < 1e-6);
}

// ============================================================================
// Tests for thread pool and chunking behavior
// ============================================================================

#[test]
fn test_q4k_parallel_thread_consistency_p19() {
    // Run many times to catch potential race conditions
    let data = generate_q4k_superblock_data(16);
    let reference = dequantize_q4_k_parallel(&data).expect("test value should be present");

    for run in 0..10 {
        let result = dequantize_q4_k_parallel(&data).expect("test value should be present");
        for i in 0..result.len() {
            assert_eq!(
                result[i], reference[i],
                "Thread inconsistency on run {} at index {}",
                run, i
            );
        }
    }
}
