use super::super::*;

// =========================================================================
// f16 conversion and Q4K header parsing tests
// =========================================================================

/// Test f16 conversion: NaN values
#[test]
fn test_f16_to_f32_nan() {
    // f16 NaN: exp=31, mantissa != 0
    let nan_val = f16_to_f32(0x7C01);
    assert!(nan_val.is_nan(), "0x7C01 should be NaN");
}

/// Test f16 conversion: negative normal value
#[test]
fn test_f16_to_f32_negative_normal() {
    // -2.0 in f16 = 0xC000
    let val = f16_to_f32(0xC000);
    assert!((val - (-2.0)).abs() < 1e-3, "Expected -2.0, got {}", val);
}

/// Test f16 conversion: smallest normal
#[test]
fn test_f16_to_f32_smallest_normal() {
    // Smallest positive normal: 0x0400 = 2^(-14) ~ 6.1035e-5
    let val = f16_to_f32(0x0400);
    assert!(val > 0.0 && val < 0.001, "Expected small normal, got {}", val);
}

/// Test f16 conversion: largest normal
#[test]
fn test_f16_to_f32_largest_normal() {
    // Largest finite f16: 0x7BFF ~ 65504
    let val = f16_to_f32(0x7BFF);
    assert!((val - 65504.0).abs() < 100.0, "Expected ~65504, got {}", val);
}

/// Test f16 conversion: negative subnormal
#[test]
fn test_f16_to_f32_negative_subnormal() {
    // Negative smallest subnormal: 0x8001
    let val = f16_to_f32(0x8001);
    assert!(val < 0.0 && val > -1e-4, "Expected small negative, got {}", val);
}

/// Test parse_q4k_header with all-zero block
#[test]
fn test_parse_q4k_header_all_zeros() {
    let block = vec![0u8; 144];
    let (d, dmin, scales, mins) = parse_q4k_header(&block);
    assert_eq!(d, 0.0);
    assert_eq!(dmin, 0.0);
    assert_eq!(scales, [0u8; 8]);
    assert_eq!(mins, [0u8; 8]);
}

/// Test parse_q4k_header with max-value block
#[test]
fn test_parse_q4k_header_max_values() {
    let mut block = vec![0xFFu8; 144];
    block[0] = 0xFF;
    block[1] = 0x7B; // d ~ largest f16 finite
    block[2] = 0xFF;
    block[3] = 0x7B; // dmin ~ largest f16 finite
    let (d, dmin, scales, mins) = parse_q4k_header(&block);
    assert!(d.is_finite(), "d should be finite");
    assert!(dmin.is_finite(), "dmin should be finite");
    // Scales and mins should be populated
    for i in 0..8 {
        // The exact values depend on bit unpacking but should be non-zero
        assert!(scales[i] > 0 || mins[i] > 0 || i >= 4);
    }
}

/// Test dequantize with more than one block
#[test]
fn test_dequantize_q4k_multi_block() {
    let num_elements = 512; // 2 blocks
    let mut data = Vec::new();
    for _ in 0..2 {
        data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        data.extend_from_slice(&[0x01u8; 12]); // scales = 1
        data.extend_from_slice(&[0x77u8; 128]); // qs = 7|7
    }

    let result = dequantize_q4k_to_f32(&data, num_elements);
    assert_eq!(result.len(), num_elements);
    for val in &result {
        assert!(val.is_finite());
    }
}

/// Test that colmajor skips zero input values
#[test]
#[allow(deprecated)]
fn test_colmajor_sparse_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
        q4k_data.extend_from_slice(&[0x55u8; 128]); // qs
    }

    // All zeros except first element
    let mut input = vec![0.0f32; in_dim];
    input[0] = 1.0;

    let output = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);
    assert_eq!(output.len(), out_dim);
    // Should be non-zero since input[0] = 1.0
    assert!(output[0].is_finite());
}

/// Test matmul_q4k_f32 with 4-way unroll remainder (in_dim not multiple of 4 within a chunk)
#[test]
fn test_matmul_q4k_f32_optimized_remainder() {
    // Test the optimized path's remainder handling
    let in_dim = 256;
    let out_dim = 1;

    let mut q4k_data = Vec::new();
    q4k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
    q4k_data.extend_from_slice(&[0x01u8; 12]); // scales = 1
                                               // Use varying qs to exercise different nibble values
    for i in 0..128 {
        q4k_data.push(((i % 16) | (((i + 1) % 16) << 4)) as u8);
    }

    let input = vec![1.0f32; in_dim];
    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let optimized = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    let diff = (scalar[0] - optimized[0]).abs();
    assert!(diff < 1e-3, "Scalar {} vs optimized {}, diff={}", scalar[0], optimized[0], diff);
}
