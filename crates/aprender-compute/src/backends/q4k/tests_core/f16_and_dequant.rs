//! f16 conversion, dequantization, colmajor, parse header, and edge case tests.

use super::super::*;

/// Test f16 conversion correctness
#[test]
fn test_f16_to_f32() {
    // Test normal values
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-3); // 1.0
    assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-3); // 2.0
    assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-3); // 0.5

    // Test zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x8000), -0.0);

    // Test subnormals (small values)
    let small = f16_to_f32(0x0001);
    assert!(small > 0.0 && small < 1e-4);
}

#[test]
fn test_f16_to_f32_infinity_nan() {
    // Positive infinity = 0x7C00
    let inf = f16_to_f32(0x7C00);
    assert!(inf.is_infinite() && inf.is_sign_positive());

    // Negative infinity = 0xFC00
    let neg_inf = f16_to_f32(0xFC00);
    assert!(neg_inf.is_infinite() && neg_inf.is_sign_negative());

    // Negative value
    let neg_one = f16_to_f32(0xBC00); // -1.0
    assert!((neg_one + 1.0).abs() < 1e-3);
}

#[test]
fn test_dequantize_q4k_to_f32_basic() {
    // Create a single Q4K block (144 bytes for 256 elements)
    let mut block = vec![0u8; SUPER_BLOCK_BYTES];
    // d = 1.0 (0x3C00)
    block[0] = 0x00;
    block[1] = 0x3C;
    // dmin = 0 (0x0000)
    block[2] = 0x00;
    block[3] = 0x00;
    // scales = all zeros
    block[4..16].fill(0x00);
    // qs = 0x55 (5 | 5<<4) for all values
    block[16..144].fill(0x55);

    let result = dequantize_q4k_to_f32(&block, 256);
    assert_eq!(result.len(), 256);

    // All values should be finite
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4k_to_f32_varies_scales() {
    let mut block = vec![0u8; SUPER_BLOCK_BYTES];
    block[0] = 0x00;
    block[1] = 0x3C; // d = 1.0
    block[2] = 0x00;
    block[3] = 0x00; // dmin = 0

    // Set different scales for each group
    for i in 0..12 {
        block[4 + i] = (i * 10) as u8;
    }

    // Set quantized values
    block[16..144].fill(0x33); // 3 | 3<<4

    let result = dequantize_q4k_to_f32(&block, 256);
    assert_eq!(result.len(), 256);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
#[allow(deprecated)]
fn test_matmul_q4k_f32_colmajor_basic() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
        q4k_data.extend_from_slice(&[0x55u8; 128]); // qs
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
#[allow(deprecated)]
fn test_matmul_q4k_f32_colmajor_dispatch_basic() {
    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[(row as u8 + 1); 12]); // varying scales
        q4k_data.extend_from_slice(&[(row as u8 * 17).wrapping_add(0x44); 128]);
        // varying qs
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let output = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
#[allow(deprecated)]
fn test_matmul_q4k_colmajor_produces_finite() {
    // Column-major layout test: verify it produces valid finite outputs
    // Note: colmajor and rowmajor have different data layout assumptions
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
        q4k_data.extend_from_slice(&[0x01u8; 12]);
        q4k_data.extend_from_slice(&[0x55u8; 128]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();

    let rowmajor = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let colmajor = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);

    // Both should produce finite results
    for (i, r) in rowmajor.iter().enumerate() {
        assert!(r.is_finite(), "Row {}: rowmajor non-finite", i);
    }
    for (i, c) in colmajor.iter().enumerate() {
        assert!(c.is_finite(), "Row {}: colmajor non-finite", i);
    }
}

#[test]
fn test_matmul_q4k_unaligned_dimensions() {
    // Test with dimensions not aligned to block size (256)
    let in_dim = 300;
    let out_dim = 3;
    let num_blocks = (in_dim + 255) / 256; // = 2 blocks

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d
            q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin
            q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
            q4k_data.extend_from_slice(&[0x33u8; 128]); // qs
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_matmul_q4k_zero_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
        q4k_data.extend_from_slice(&[0x7Fu8; 12]); // max scales
        q4k_data.extend_from_slice(&[0xFFu8; 128]); // max qs
    }

    let input: Vec<f32> = vec![0.0; in_dim];
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert_eq!(*val, 0.0, "Output should be zero when input is zero");
    }
}

#[test]
fn test_matmul_q4k_large_dimensions() {
    let in_dim = 1024;
    let out_dim = 8;
    let num_blocks = in_dim / 256;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for blk in 0..num_blocks {
            let val = ((row * num_blocks + blk) as u8).wrapping_mul(17);
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
            q4k_data.extend_from_slice(&[0x33, 0x2A]); // dmin ~ 0.05
            q4k_data.extend_from_slice(&[(val.wrapping_add(1)); 12]);
            q4k_data.extend_from_slice(&[val.wrapping_add(0x55); 128]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 100) as f32) * 0.01).collect();
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_parse_q4k_header() {
    let mut block = vec![0u8; 144];
    // d = 1.0 (0x3C00), dmin = 0.5 (0x3800)
    block[0] = 0x00;
    block[1] = 0x3C;
    block[2] = 0x00;
    block[3] = 0x38;
    // scales_bytes[0..12] for llama.cpp format
    block[4..8].copy_from_slice(&[0x01, 0x02, 0x03, 0x04]); // scales[0-3] = 1,2,3,4
    block[8..12].copy_from_slice(&[0x0A, 0x0B, 0x0C, 0x0D]); // mins[0-3] = 10,11,12,13
    block[12..16].copy_from_slice(&[0x55, 0x66, 0x77, 0x88]); // combined lower nibbles

    let (d, dmin, scales, mins) = parse_q4k_header(&block);

    assert!((d - 1.0).abs() < 0.01, "d should be ~1.0, got {}", d);
    assert!((dmin - 0.5).abs() < 0.01, "dmin should be ~0.5, got {}", dmin);
    // Check first scales/mins have expected low 6-bit values
    assert_eq!(scales[0], 0x01, "scales[0] should be 1");
    assert_eq!(scales[1], 0x02, "scales[1] should be 2");
    assert_eq!(mins[0], 0x0A, "mins[0] should be 10");
    assert_eq!(mins[1], 0x0B, "mins[1] should be 11");
}

#[test]
fn test_matmul_q4k_single_row() {
    let in_dim = 256;
    let out_dim = 1;

    let mut q4k_data = Vec::new();
    q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
    q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
    q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
    q4k_data.extend_from_slice(&[0xAAu8; 128]); // qs

    let input: Vec<f32> = vec![1.0; in_dim];
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}
