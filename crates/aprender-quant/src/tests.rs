use super::*;

/// Standard test data: 256 floats centered around zero.
fn test_data_256() -> Vec<f32> {
    (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect()
}

/// Compute max absolute error between original and dequantized data.
fn max_abs_error(original: &[f32], dequantized: &[f32]) -> f32 {
    original.iter().zip(dequantized.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

/// Compute data range (max - min).
fn data_range(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |a, &b| a.max(b)) - data.iter().fold(0.0f32, |a, &b| a.min(b))
}

/// Assert roundtrip error is within a fraction of data range.
fn assert_roundtrip_within_range(
    original: &[f32],
    dequantized: &[f32],
    fraction: f32,
    label: &str,
) {
    let error = max_abs_error(original, dequantized);
    let threshold = data_range(original) * fraction;
    assert!(error < threshold, "{label} roundtrip error {error} exceeds threshold {threshold}");
}

#[test]
fn test_q4k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q4_k(&data);
    assert_eq!(quantized.len(), 144);
    let dequantized = dequantize_q4_k_to_f32(&quantized, 256);
    assert_roundtrip_within_range(&data, &dequantized, 0.5, "Q4K");
}

#[test]
fn test_q5k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q5_k(&data);
    assert_eq!(quantized.len(), 176);
    let dequantized = dequantize_q5_k_to_f32(&quantized, 256);
    assert_roundtrip_within_range(&data, &dequantized, 0.4, "Q5K");
}

#[test]
fn test_q6k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q6_k(&data);
    assert_eq!(quantized.len(), 210);
    let dequantized = dequantize_q6_k_to_f32(&quantized, 256);
    assert!(max_abs_error(&data, &dequantized) < 1.0, "Q6K roundtrip error too high");
}

#[test]
fn test_q4k_matrix() {
    let data: Vec<f32> = (0..512).map(|i| i as f32 / 100.0).collect();
    let shape = vec![2, 256];
    let quantized = quantize_q4_k_matrix(&data, &shape);
    assert_eq!(quantized.len(), 2 * 144);
}

#[test]
fn test_transpose_q4k() {
    let cols = 256;
    let rows = 2;
    let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 / 10.0).collect();
    let quantized = quantize_q4_k(&data);
    let shape = vec![cols, rows];
    let (transposed_data, new_shape) = transpose_q4k_for_matmul(&quantized, &shape);
    assert_eq!(new_shape, vec![rows, cols]);
    assert!(!transposed_data.is_empty());
}

#[test]
fn test_f16_min_normal() {
    let f16_val = half::f16::from_f32(F16_MIN_NORMAL);
    let roundtrip = f16_val.to_f32();
    assert!(roundtrip > 0.0, "F16_MIN_NORMAL should be positive after f16 roundtrip");
    assert!(roundtrip < 1e-4, "F16_MIN_NORMAL should be small");
}

#[test]
fn test_constants() {
    assert_eq!(Q4_K_BLOCK_SIZE, 256);
    assert_eq!(Q4_K_BLOCK_BYTES, 144);
    assert_eq!(Q5_K_BLOCK_SIZE, 256);
    assert_eq!(Q5_K_BLOCK_BYTES, 176);
    assert_eq!(Q6_K_BLOCK_SIZE, 256);
    assert_eq!(Q6_K_BLOCK_BYTES, 210);
}

// ===== Dequantize f16 scale sanitization tests =====
// Regression: Q5K and Q6K dequantize did not sanitize f16 scale values,
// unlike Q4K which guards against NaN/Inf/subnormal via sanitize_f16_scale().
// In clean-room containers (no SIMD flags), subnormal f16 values can propagate
// through dequantization and produce incorrect or non-finite results.

/// Construct a Q6K block with a hand-crafted f16 scale value and verify
/// dequantization produces only finite outputs.
#[test]
fn test_q6k_dequantize_subnormal_scale() {
    // Build a minimal Q6K block (210 bytes) with a subnormal f16 scale.
    // Layout: ql[128] + qh[64] + scales[16] + d(f16)[2]
    let mut block = vec![0u8; Q6_K_BLOCK_BYTES];

    // Set ql to non-zero pattern so dequant produces non-trivial values
    for i in 0..128 {
        block[i] = 0x12;
    }
    // Set scales to non-zero
    for i in 192..208 {
        block[i] = 1;
    }
    // Set d (f16) to a subnormal value: 0x0001 is the smallest positive subnormal f16
    block[208] = 0x01;
    block[209] = 0x00;

    let result = dequantize_q6_k_to_f32(&block, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "Q6K dequant produced non-finite value at index {}: {}", i, v);
    }
}

/// Construct a Q6K block with NaN f16 scale and verify dequantization
/// returns zeros (not NaN propagation).
#[test]
fn test_q6k_dequantize_nan_scale() {
    let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
    for i in 0..128 {
        block[i] = 0x55;
    }
    for i in 192..208 {
        block[i] = 2;
    }
    // f16 NaN: exponent all 1s, non-zero mantissa. 0x7C01 is a NaN.
    block[208] = 0x01;
    block[209] = 0x7C;

    let result = dequantize_q6_k_to_f32(&block, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "Q6K NaN scale propagated to index {}: {}", i, v);
        assert!(
            v.abs() < f32::EPSILON,
            "Q6K with NaN scale should produce 0.0 at index {}, got {}",
            i,
            v
        );
    }
}

/// Construct a Q5K block with a subnormal f16 scale and verify finite results.
#[test]
fn test_q5k_dequantize_subnormal_scale() {
    let mut block = vec![0u8; Q5_K_BLOCK_BYTES];
    // Set scales (bytes 4..16) to non-zero
    for i in 4..16 {
        block[i] = 0x21;
    }
    // Set qh (bytes 16..48) to non-zero
    for i in 16..48 {
        block[i] = 0x55;
    }
    // Set qs (bytes 48..176) to non-zero
    for i in 48..176 {
        block[i] = 0x33;
    }
    // d (f16) subnormal at bytes 0..1
    block[0] = 0x01;
    block[1] = 0x00;
    // dmin (f16) subnormal at bytes 2..3
    block[2] = 0x01;
    block[3] = 0x00;

    let result = dequantize_q5_k_to_f32(&block, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "Q5K dequant produced non-finite value at index {}: {}", i, v);
    }
}

/// Q6K roundtrip with SIMD-boundary-crossing data: values that span
/// SIMD lane widths (8-wide, 16-wide) to catch scaling mismatches.
#[test]
fn test_q6k_simd_scaling_roundtrip() {
    // Data with sharp transitions at SIMD lane boundaries (every 8 and 16 elements)
    let data: Vec<f32> = (0..256)
        .map(|i| {
            let base = (i as f32 - 128.0) / 10.0;
            // Introduce sharp scaling change at lane boundary
            if i % 16 < 8 {
                base * 0.01
            } else {
                base * 100.0
            }
        })
        .collect();

    let quantized = quantize_q6_k(&data);
    let dequantized = dequantize_q6_k_to_f32(&quantized, 256);

    // All values must be finite
    for (i, &v) in dequantized.iter().enumerate() {
        assert!(v.is_finite(), "Q6K SIMD scaling roundtrip: non-finite at index {}: {}", i, v);
    }

    // Roundtrip error should be bounded
    let max_err =
        data.iter().zip(dequantized.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let range = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        max_err < range * 0.15,
        "Q6K SIMD scaling roundtrip error {} exceeds 15% of range {}",
        max_err,
        range
    );
}
