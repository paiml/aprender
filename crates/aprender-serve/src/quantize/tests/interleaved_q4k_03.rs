
// =========================================================================
// Deep Coverage Tests: Scalar Fallback Paths (_deep_qcov_ prefix)
// =========================================================================
//
// These tests explicitly exercise scalar fallback code paths that are normally
// bypassed when AVX2/AVX-512 are available. They achieve coverage by:
// 1. Directly calling *_scalar functions
// 2. Testing error handling paths
// 3. Exercising boundary conditions in quantization code
// =========================================================================

// -------------------------------------------------------------------------
// InterleavedQ4K::dot_scalar Direct Tests
// -------------------------------------------------------------------------

#[test]
fn test_interleaved_q4k_dot_scalar_explicitly_deep_qcov_001() {
    // Force use of scalar path by calling dot_scalar directly
    // Single super-block with simple test data
    let mut q4k_data = Vec::with_capacity(144);

    // d = 1.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // dmin = 0.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // scales: 12 bytes (all zeros -> scale=0, min=0)
    q4k_data.extend_from_slice(&[0u8; 12]);
    // qs: 128 bytes (alternating pattern 0x12 for low=2, high=1)
    q4k_data.extend_from_slice(&[0x12u8; 128]);

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations = vec![1.0f32; 256];

    // Directly call scalar implementation
    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
    // With scale=0 and min=0 from scales array, result should be 0
    let dot_result = result.expect("quantization failed");
    assert!(dot_result.abs() < 1e-6, "Expected ~0, got {}", dot_result);
}

#[test]
fn test_interleaved_q4k_dot_scalar_with_scales_deep_qcov_002() {
    // Test scalar path with non-zero scales
    let mut q4k_data = Vec::with_capacity(144);

    // d = 2.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    // dmin = 0.5 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // scales: 12 bytes with first scale=1, first min=1
    let mut scales = [0u8; 12];
    scales[0] = 0x01; // scale[0] = 1 (6-bit)
    scales[4] = 0x01; // min[0] = 1 (6-bit)
    q4k_data.extend_from_slice(&scales);
    // qs: 128 bytes (all 0x55 for varied nibbles)
    q4k_data.extend_from_slice(&[0x55u8; 128]);

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
    // Result should be non-zero due to scales
    let _dot_result = result.expect("quantization failed");
}

#[test]
fn test_interleaved_q4k_dot_scalar_multiple_superblocks_deep_qcov_003() {
    // Test scalar path with 4 super-blocks
    let num_sb = 4;
    let mut q4k_data = Vec::with_capacity(144 * num_sb);

    for i in 0..num_sb {
        // Vary d across super-blocks
        let d_val = 0.5 + (i as f32) * 0.25;
        q4k_data.extend_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
        q4k_data.extend_from_slice(&half::f16::from_f32(0.1).to_le_bytes());

        // Varied scales
        let mut scales = [0u8; 12];
        scales[0] = ((i + 1) % 64) as u8;
        q4k_data.extend_from_slice(&scales);

        // Varied qs
        let pattern = ((i * 17) % 256) as u8;
        q4k_data.extend_from_slice(&vec![pattern; 128]);
    }

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
}

#[test]
fn test_interleaved_q4k_dot_scalar_negative_activations_deep_qcov_004() {
    // Test with negative activations
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    let mut scales = [0u8; 12];
    scales[0] = 0x3F; // max scale
    q4k_data.extend_from_slice(&scales);
    q4k_data.extend_from_slice(&[0xFFu8; 128]); // max nibbles

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.01).collect();

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
    // Result should be negative
    let dot_result = result.expect("quantization failed");
    assert!(dot_result < 0.0, "Expected negative, got {}", dot_result);
}

#[test]
fn test_interleaved_q4k_dot_scalar_max_values_deep_qcov_005() {
    // Test with maximum scale and nibble values
    let mut q4k_data = Vec::with_capacity(144);
    // d = max f16 normal value (about 65504)
    q4k_data.extend_from_slice(&half::f16::from_f32(100.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(50.0).to_le_bytes());
    // Max scales (63)
    q4k_data.extend_from_slice(&[0x3Fu8; 12]);
    // Max nibbles (0xFF = low=15, high=15)
    q4k_data.extend_from_slice(&[0xFFu8; 128]);

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");
    let activations = vec![10.0f32; 256];

    let result = interleaved.dot_scalar(&activations);
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// InterleavedQ4K Error Handling Tests
// -------------------------------------------------------------------------

#[test]
fn test_interleaved_q4k_dot_mismatched_size_deep_qcov_006() {
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid data");

    // Wrong activation size (128 instead of 256)
    let activations = vec![1.0f32; 128];
    let result = interleaved.dot(&activations);
    assert!(result.is_err());

    // Verify error message contains useful info
    if let Err(e) = result {
        let msg = format!("{:?}", e);
        assert!(msg.contains("128") || msg.contains("256"));
    }
}

#[test]
fn test_interleaved_q4k_from_invalid_length_deep_qcov_007() {
    // Not a multiple of 144
    let q4k_data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&q4k_data);
    assert!(result.is_err());

    let q4k_data = vec![0u8; 145];
    let result = InterleavedQ4K::from_q4k(&q4k_data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_num_values_deep_qcov_008() {
    // Test num_values() for various super-block counts
    let q4k_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    assert_eq!(interleaved.num_values(), 256);

    let q4k_data = vec![0u8; 288];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    assert_eq!(interleaved.num_values(), 512);

    let q4k_data = vec![0u8; 144 * 10];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");
    assert_eq!(interleaved.num_values(), 2560);
}

// -------------------------------------------------------------------------
// fused_q4k_dot Scalar Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_scalar_all_zeros_deep_qcov_009() {
    // All zeros should produce zero dot product
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
    let dot = result.expect("quantization failed");
    assert!(dot.abs() < 1e-6, "Expected ~0, got {}", dot);
}

#[test]
fn test_fused_q4k_dot_invalid_q4k_length_deep_qcov_010() {
    // Q4K data not multiple of 144
    let q4k_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_mismatched_activation_length_deep_qcov_011() {
    // Activations don't match expected count
    let q4k_data = vec![0u8; 144]; // 256 values
    let activations = vec![1.0f32; 100];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_dot_simd_falls_back_correctly_deep_qcov_012() {
    // This test verifies that fused_q4k_dot_simd produces same results as scalar
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    q4k_data.extend_from_slice(&[0x21u8; 12]); // varied scales
    q4k_data.extend_from_slice(&[0x34u8; 128]); // varied qs

    let activations: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

    let scalar_result = fused_q4k_dot(&q4k_data, &activations).expect("scalar ok");
    let simd_result = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd ok");

    // Results should be within ULP tolerance
    let diff = (scalar_result - simd_result).abs();
    let tolerance = scalar_result.abs() * 1e-5 + 1e-6;
    assert!(
        diff < tolerance,
        "scalar={}, simd={}, diff={}",
        scalar_result,
        simd_result,
        diff
    );
}

// -------------------------------------------------------------------------
// fused_q4k_q8k_dot Scalar Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_dot_scalar_basic_deep_qcov_013() {
    // Test scalar Q4K×Q8K dot product
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_invalid_q4k_deep_qcov_014() {
    let q4k_data = vec![0u8; 100]; // Not multiple of 144
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_insufficient_scales_deep_qcov_015() {
    let q4k_data = vec![0u8; 288]; // 2 super-blocks
    let q8k_scales = vec![1.0f32; 1]; // Only 1 scale (need 2)
    let q8k_quants = vec![0i8; 512];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_insufficient_quants_deep_qcov_016() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 100]; // Need 256

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_with_data_deep_qcov_017() {
    // Test with actual non-zero data
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    let mut scales = [0u8; 12];
    scales[0] = 0x1F; // scale=31
    scales[4] = 0x0F; // min=15
    q4k_data.extend_from_slice(&scales);
    q4k_data.extend_from_slice(&[0xABu8; 128]);

    let q8k_scales = vec![1.5f32; 1];
    let q8k_quants: Vec<i8> = (0..256)
        .map(|i| ((i % 256) as i8).wrapping_sub(64))
        .collect();

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// fused_q4_0_q8_0_dot_scalar Direct Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_basic_deep_qcov_018() {
    // Test scalar Q4_0×Q8_0 dot directly
    // Q4_0: 18 bytes per 32 values (2 bytes scale + 16 bytes qs)
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // 16 bytes of qs with value 0x88 (low=8, high=8) -> centered at 0
    q4_data[2..18].copy_from_slice(&[0x88u8; 16]);

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // Should be near zero since Q4 values are centered at 8 (subtract 8 = 0)
    assert!(result.abs() < 1e-3, "Expected ~0, got {}", result);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_multiple_blocks_deep_qcov_019() {
    // Test with 2 blocks (64 values)
    let mut q4_data = vec![0u8; 36];
    q4_data[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    q4_data[2..18].copy_from_slice(&[0x99u8; 16]);
    q4_data[18..20].copy_from_slice(&half::f16::from_f32(1.5).to_le_bytes());
    q4_data[20..36].copy_from_slice(&[0x66u8; 16]);

    let q8_scales = vec![1.0f32, 2.0f32];
    let q8_quants: Vec<i8> = (0..64).map(|i| (i % 100) as i8).collect();

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // Just verify it runs without error
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_truncated_block_deep_qcov_020() {
    // Test with in_dim not a multiple of 32
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4_data[2..18].copy_from_slice(&[0x55u8; 16]);

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![2i8; 32];

    // Only use 24 values (less than 32)
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 24);
    assert!(result.is_finite());
}

// -------------------------------------------------------------------------
// extract_scale_min_from_slice Coverage Tests
// -------------------------------------------------------------------------

#[test]
fn test_extract_scale_min_from_slice_odd_indices_deep_qcov_021() {
    // Test odd index paths (different bit manipulation)
    let mut scales = [0u8; 12];
    // Set up for odd index extraction
    scales[0] = 0xC0; // High 2 bits for odd scale
    scales[2] = 0x0F; // Low 4 bits for odd scale
    scales[4] = 0x80; // High 2 bits for odd min
    scales[6] = 0x07; // Low 4 bits for odd min

    let (scale1, min1) = extract_scale_min_from_slice(&scales, 1);
    assert!(scale1 >= 0.0);
    assert!(min1 >= 0.0);

    // Test index 3, 5, 7
    for idx in [3, 5, 7] {
        let (s, m) = extract_scale_min_from_slice(&scales, idx);
        assert!(s >= 0.0);
        assert!(m >= 0.0);
    }
}

#[test]
fn test_extract_scale_min_from_slice_even_indices_deep_qcov_022() {
    // Test even index paths
    let mut scales = [0u8; 12];
    scales[0] = 0x3F; // max 6-bit scale for idx=0
    scales[4] = 0x20; // min for idx=0

    let (scale0, min0) = extract_scale_min_from_slice(&scales, 0);
    assert_eq!(scale0, 63.0);
    assert_eq!(min0, 32.0);

    // Test indices 2, 4, 6
    for idx in [2, 4, 6] {
        let (s, m) = extract_scale_min_from_slice(&scales, idx);
        assert!(s >= 0.0);
        assert!(m >= 0.0);
    }
}

// -------------------------------------------------------------------------
// fused_q4_0_q8_0_parallel_matvec Error Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_weight_too_small_deep_qcov_023() {
    let in_dim = 32;
    let out_dim = 4;
    // Need 4 rows * 18 bytes = 72 bytes, provide less
    let weight_data = vec![0u8; 50];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_weight_too_small_deep_qcov_024() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; 50]; // Too small
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}
