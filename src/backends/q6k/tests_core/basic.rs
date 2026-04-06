//! Basic Q6K tests: basic functionality, f16 conversion, edge cases

use super::super::*;

#[test]
fn test_q6k_basic() {
    let in_dim = 256;
    let out_dim = 2;

    // Create Q6K test data (210 bytes per block)
    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        // ql: 128 bytes (all zeros = q4 part is 0)
        q6k_data.extend_from_slice(&[0x55u8; 128]); // 5 in each nibble
                                                    // qh: 64 bytes (all zeros = q2 part is 0)
        q6k_data.extend_from_slice(&[0x00u8; 64]);
        // scales: 16 bytes (all ones)
        q6k_data.extend_from_slice(&[0x01u8; 16]);
        // d: f16 = 1.0
        q6k_data.extend_from_slice(&[0x00, 0x3C]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q6k_avx2_vs_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let in_dim = 512;
    let out_dim = 4;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        for _ in 0..2 {
            // 2 blocks per row
            q6k_data.extend_from_slice(&[(row as u8 * 17).wrapping_add(0x33); 128]);
            q6k_data.extend_from_slice(&[(row as u8).wrapping_add(0x11); 64]);
            q6k_data.extend_from_slice(&[0x02u8; 16]);
            q6k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.002 - 0.5).collect();

    let scalar = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        assert!(diff < 1e-4, "Row {}: scalar {} vs dispatch {} (diff {})", i, s, d, diff);
    }
}

#[test]
fn test_f16_to_f32_normal() {
    // Normal f16 value: 1.0 = 0x3C00
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result);

    // 2.0 = 0x4000
    let result = f16_to_f32(0x4000);
    assert!((result - 2.0).abs() < 1e-6, "Expected 2.0, got {}", result);

    // -1.0 = 0xBC00
    let result = f16_to_f32(0xBC00);
    assert!((result + 1.0).abs() < 1e-6, "Expected -1.0, got {}", result);
}

#[test]
fn test_f16_to_f32_zero() {
    // Positive zero
    let result = f16_to_f32(0x0000);
    assert_eq!(result, 0.0, "Expected +0.0");
    assert!(result.is_sign_positive());

    // Negative zero
    let result = f16_to_f32(0x8000);
    assert_eq!(result, 0.0, "Expected -0.0");
    assert!(result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_infinity() {
    // Positive infinity = 0x7C00
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite() && result.is_sign_positive());

    // Negative infinity = 0xFC00
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite() && result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_subnormal() {
    // Smallest subnormal: 0x0001 ≈ 5.96e-8
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0 && result < 1e-6, "Expected small subnormal, got {}", result);

    // Larger subnormal: 0x03FF (largest subnormal)
    let result = f16_to_f32(0x03FF);
    assert!(result > 0.0 && result < 1e-4, "Expected subnormal, got {}", result);
}

#[test]
#[allow(deprecated)]
fn test_q6k_colmajor_basic() {
    let in_dim = 256;
    let out_dim = 2;

    // Create Q6K test data
    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x33u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q6k_f32_colmajor(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }
}

#[test]
#[allow(deprecated)]
fn test_q6k_colmajor_dispatch() {
    let in_dim = 256;
    let out_dim = 4;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        q6k_data.extend_from_slice(&[(row as u8).wrapping_add(0x22); 128]);
        q6k_data.extend_from_slice(&[(row as u8).wrapping_add(0x11); 64]);
        q6k_data.extend_from_slice(&[0x02u8; 16]);
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 1.0).collect();

    let result = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, out_dim, in_dim);
    assert_eq!(result.len(), out_dim);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_unaligned_dimensions() {
    // Test with dimensions not aligned to block size (256)
    let in_dim = 300; // Not a multiple of 256
    let out_dim = 3;
    let num_blocks = (in_dim + 255) / 256; // = 2 blocks

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q6k_data.extend_from_slice(&[0x11u8; 128]);
            q6k_data.extend_from_slice(&[0x00u8; 64]);
            q6k_data.extend_from_slice(&[0x01u8; 16]);
            q6k_data.extend_from_slice(&[0x00, 0x3C]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_single_row() {
    let in_dim = 256;
    let out_dim = 1;

    let mut q6k_data = Vec::new();
    q6k_data.extend_from_slice(&[0xAAu8; 128]); // ql
    q6k_data.extend_from_slice(&[0x55u8; 64]); // qh (alternating bits)
    q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
    q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0

    let input: Vec<f32> = vec![1.0; in_dim];
    let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

#[test]
fn test_q6k_large_dimensions() {
    let in_dim = 1024;
    let out_dim = 8;
    let num_blocks = in_dim / 256;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        for blk in 0..num_blocks {
            let val = ((row * num_blocks + blk) as u8).wrapping_mul(17);
            q6k_data.extend_from_slice(&[val; 128]);
            q6k_data.extend_from_slice(&[val.wrapping_add(1); 64]);
            q6k_data.extend_from_slice(&[0x02u8; 16]);
            q6k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 100) as f32) * 0.01).collect();
    let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_zero_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0xFFu8; 128]);
        q6k_data.extend_from_slice(&[0xFFu8; 64]);
        q6k_data.extend_from_slice(&[0x7Fu8; 16]); // max positive scale
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input: Vec<f32> = vec![0.0; in_dim];
    let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert_eq!(*val, 0.0, "Output should be zero when input is zero");
    }
}

#[test]
fn test_q6k_negative_scales() {
    let in_dim = 256;
    let out_dim = 1;

    let mut q6k_data = Vec::new();
    q6k_data.extend_from_slice(&[0x00u8; 128]); // ql = 0
    q6k_data.extend_from_slice(&[0x00u8; 64]); // qh = 0
    q6k_data.extend_from_slice(&[0x80u8; 16]); // scales = -128 (negative)
    q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0

    let input: Vec<f32> = vec![1.0; in_dim];
    let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
    // With negative scales and quant=0-32=-32, result should be positive
}
