use crate::quantize::*;
#[test]
fn test_dequantize_q8_blocks_roundtrip_cov() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("quantization failed");
    let restored = dequantize_q8_blocks(&blocks);
    assert_eq!(restored.len(), 64);
    // Should be close to original
    for (o, r) in original.iter().zip(restored.iter()) {
        assert!((o - r).abs() < 0.1);
    }
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock quantize_into
// =========================================================================

#[test]
fn test_q8ksuperblock_quantize_into_cov() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
}

// =========================================================================
// Coverage Tests: dequantize_f16 additional error paths
// =========================================================================

#[test]
fn test_dequantize_f16_size_3_cov() {
    let data = vec![0u8; 3]; // Odd size - invalid
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_f16_vec_empty_cov() {
    let data: Vec<u8> = vec![];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Coverage Tests: f16_to_f32 edge cases
// =========================================================================

#[test]
fn test_f16_to_f32_subnormal_cov() {
    // Subnormal f16 value (exponent 0, non-zero mantissa)
    let subnormal = 0x0001u16; // Smallest positive subnormal
    let f32_val = f16_to_f32(subnormal);
    assert!(f32_val > 0.0);
    assert!(f32_val < 1e-5);
}

#[test]
fn test_f16_to_f32_infinity_cov() {
    // Positive infinity (exponent all 1s, mantissa 0)
    let pos_inf = 0x7C00u16;
    let f32_val = f16_to_f32(pos_inf);
    assert!(f32_val.is_infinite() && f32_val > 0.0);
}

#[test]
fn test_f16_to_f32_neg_infinity_cov() {
    // Negative infinity
    let neg_inf = 0xFC00u16;
    let f32_val = f16_to_f32(neg_inf);
    assert!(f32_val.is_infinite() && f32_val < 0.0);
}

// ============================================================================
// Coverage Tests for Dequantize Functions (Refs PMAT-802)
// ============================================================================

#[test]
fn test_cov_dequantize_f16_valid() {
    // Create F16 bytes for values: 1.0, 2.0
    let f16_one = half::f16::from_f32(1.0).to_le_bytes();
    let f16_two = half::f16::from_f32(2.0).to_le_bytes();
    let data = [f16_one[0], f16_one[1], f16_two[0], f16_two[1]];

    let result = dequantize_f16(&data).expect("should succeed");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_cov_dequantize_f16_invalid_odd() {
    let data = [0u8; 3]; // Odd length
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_f16_empty() {
    let data: [u8; 0] = [];
    let result = dequantize_f16(&data).expect("empty should succeed");
    assert_eq!(result.len(), 0);
}

#[test]
fn test_cov_dequantize_q4_1_invalid() {
    // Q4_1 block is 20 bytes, test with 19
    let data = vec![0u8; 19];
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_q4_1_one_block() {
    // Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes data = 20 bytes
    let mut data = Vec::new();
    // Scale = 1.0 as f16
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0 as f16
    data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // 16 bytes of zeros (32 4-bit values)
    data.extend_from_slice(&[0u8; 16]);

    let result = dequantize_q4_1(&data).expect("should succeed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_cov_dequantize_q4_1_two_blocks() {
    let mut data = Vec::new();
    // Two blocks of 20 bytes each
    for _ in 0..2 {
        data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        data.extend_from_slice(&[0u8; 16]);
    }

    let result = dequantize_q4_1(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_cov_dequantize_q5_0_invalid() {
    // Q5_0 block is 22 bytes, test with 21
    let data = vec![0u8; 21];
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_q5_0_one_block() {
    // Q5_0 block: 2 bytes scale + 4 bytes high bits + 16 bytes low bits = 22 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&[0u8; 4]); // qh (high bits)
    data.extend_from_slice(&[0u8; 16]); // qs (low bits)

    let result = dequantize_q5_0(&data).expect("should succeed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_cov_dequantize_q5_0_two_blocks() {
    // Two blocks = 44 bytes -> 64 floats
    let data = vec![0u8; 44];
    let result = dequantize_q5_0(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_cov_dequantize_q5_1_invalid() {
    // Q5_1 block is 24 bytes, test with 23
    let data = vec![0u8; 23];
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_dequantize_q5_1_one_block() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes high + 16 bytes low = 24 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data.extend_from_slice(&[0u8; 4]); // qh
    data.extend_from_slice(&[0u8; 16]); // qs

    let result = dequantize_q5_1(&data).expect("should succeed");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_cov_dequantize_q5_1_two_blocks() {
    // Two blocks = 48 bytes -> 64 floats
    let data = vec![0u8; 48];
    let result = dequantize_q5_1(&data).expect("should succeed");
    assert_eq!(result.len(), 64);
}

// =========================================================================
// IMP-XXX: Direct Scalar Fallback Tests
// =========================================================================
// These tests directly call the scalar implementations, bypassing the
// SIMD dispatcher. This ensures the scalar fallbacks are tested even on
// machines with AVX2 support.

#[test]
fn test_scalar_quantize_rmsnorm_q8_0_direct() {
    // Direct test of scalar implementation
    let input = vec![1.0f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);

    // Should produce 2 blocks of scales (64 / 32 = 2)
    assert_eq!(scales.len(), 2);
    // Should produce 64 quantized values
    assert_eq!(quants.len(), 64);
    // Scales should be positive
    assert!(scales.iter().all(|&s| s >= 0.0));
}

#[test]
fn test_scalar_quantize_rmsnorm_q8_0_various_sizes() {
    for size in [32, 64, 128, 256, 512] {
        let input = vec![0.5f32; size];
        let norm_weight = vec![2.0f32; size];
        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);

        assert_eq!(scales.len(), size / 32, "scales for size {}", size);
        assert_eq!(quants.len(), size, "quants for size {}", size);
    }
}

#[test]
fn test_scalar_fused_swiglu_direct() {
    // Direct test of scalar SwiGLU
    let mut gate = vec![1.0f32; 64];
    let up = vec![2.0f32; 64];

    fused_swiglu_simd(&mut gate, &up);

    // silu(1.0) ≈ 0.7311 (sigmoid(1) = ~0.7311)
    // result = silu(gate) * up ≈ 0.7311 * 2.0 ≈ 1.4623
    // Note: SIMD approximation may differ slightly
    for g in &gate {
        assert!(*g > 1.4 && *g < 1.6, "got {}", g);
    }
}

#[test]
fn test_scalar_fused_swiglu_zero() {
    let mut gate = vec![0.0f32; 32];
    let up = vec![1.0f32; 32];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    for g in &gate {
        assert!(g.abs() < 1e-6, "expected near zero, got {}", g);
    }
}

#[test]
fn test_scalar_softmax_direct() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_simd(&mut x);

    // Sum should be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);

    // Values should be monotonically increasing (since input was increasing)
    assert!(x[0] < x[1] && x[1] < x[2] && x[2] < x[3]);
}

#[test]
fn test_scalar_softmax_large_values() {
    // Test numerical stability with large values
    let mut x = vec![1000.0, 1001.0, 1002.0, 1003.0];
    softmax_simd(&mut x);

    // Should not overflow/underflow
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);
    assert!(
        x.iter().all(|&v| v.is_finite()),
        "all values should be finite"
    );
}

#[test]
fn test_scalar_apply_rope_rotation_direct() {
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals = vec![0.5f32; 4];
    let sin_vals = vec![0.866f32; 4]; // sin(60°) ≈ 0.866

    let x1_orig = x1.clone();
    let x2_orig = x2.clone();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Rotation formula: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
    for i in 0..4 {
        let expected_x1 = x1_orig[i] * 0.5 - x2_orig[i] * 0.866;
        let expected_x2 = x1_orig[i] * 0.866 + x2_orig[i] * 0.5;
        assert!(
            (x1[i] - expected_x1).abs() < 1e-5,
            "x1[{}]: expected {}, got {}",
            i,
            expected_x1,
            x1[i]
        );
        assert!(
            (x2[i] - expected_x2).abs() < 1e-5,
            "x2[{}]: expected {}, got {}",
            i,
            expected_x2,
            x2[i]
        );
    }
}

#[test]
fn test_scalar_fused_q4_0_q8_0_dot_direct() {
    // Create minimal Q4_0 data (1 block = 18 bytes)
    // Layout: 2 bytes f16 scale + 16 bytes quants
    let mut q4_data = vec![0u8; 18];
    // Set scale to 1.0 (f16 encoding)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    q4_data[0] = scale_bits as u8;
    q4_data[1] = (scale_bits >> 8) as u8;
    // Set quants to some pattern (each byte = 2 values)
    for i in 0..16 {
        q4_data[2 + i] = 0x88; // Both nibbles = 8 (after -8 offset = 0)
    }

    // Create Q8_0 activations (1 block = 32 values)
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);

    // With our setup, all dequantized values are 0, so dot product should be 0
    assert!(result.abs() < 1e-5, "expected near zero, got {}", result);
}

#[test]
fn test_scalar_fused_q8_0_q8_0_dot_direct() {
    // Create Q8_0 weight data (1 block = 34 bytes: 2 bytes scale + 32 bytes quants)
    let mut q8_weight_data = vec![0u8; 34];
    // Set scale to 1.0 (f16 encoding)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    q8_weight_data[0] = scale_bits as u8;
    q8_weight_data[1] = (scale_bits >> 8) as u8;
    // Set quants to 1
    for i in 0..32 {
        q8_weight_data[2 + i] = 1;
    }

    // Create Q8_0 activations
    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);

    // Dot product of 32 ones with scale 1.0 * 1.0 = 32
    assert!((result - 32.0).abs() < 1e-3, "expected ~32, got {}", result);
}

// =========================================================================
// Direct Tests for Dequantization Helpers
// =========================================================================

#[test]
fn test_dequantize_q4_k_cov_direct() {
    // Create a minimal Q4_K super-block (144 bytes)
    // Layout: d (2) + dmin (2) + scales (12) + qs (128) = 144 bytes
    let mut sb_data = vec![0u8; 144];

    // Set d = 1.0 (f16 encoding)
    let d_bits = half::f16::from_f32(1.0).to_bits();
    sb_data[0] = d_bits as u8;
    sb_data[1] = (d_bits >> 8) as u8;

    // Set dmin = 0.0 (no minimum offset)
    sb_data[2] = 0;
    sb_data[3] = 0;

    // Set scales to simple pattern (all 1s for testing)
    for i in 0..12 {
        sb_data[4 + i] = 0x11; // scale = 1, min = 1 (packed)
    }

    // Set qs to all zeros for simplicity
    for i in 0..128 {
        sb_data[16 + i] = 0x00;
    }

    let result = dequantize_q4_k(&sb_data).expect("test");

    // Should produce QK_K (256) values
    assert_eq!(
        result.len(),
        QK_K,
        "expected {} values, got {}",
        QK_K,
        result.len()
    );
    // All values should be finite
    assert!(
        result.iter().all(|&v| v.is_finite()),
        "all values should be finite"
    );
}

#[test]
fn test_dequantize_q4_k_cov_nonzero_quants() {
    // Test with non-zero quantized values
    let mut sb_data = vec![0u8; 144];

    // d = 0.5
    let d_bits = half::f16::from_f32(0.5).to_bits();
    sb_data[0] = d_bits as u8;
    sb_data[1] = (d_bits >> 8) as u8;

    // dmin = 0.25
    let dmin_bits = half::f16::from_f32(0.25).to_bits();
    sb_data[2] = dmin_bits as u8;
    sb_data[3] = (dmin_bits >> 8) as u8;

    // scales = 0x22 (scale=2, min=2 for each pair)
    for i in 0..12 {
        sb_data[4 + i] = 0x22;
    }

    // qs = 0x88 (nibbles = 8, 8)
    for i in 0..128 {
        sb_data[16 + i] = 0x88;
    }

    let result = dequantize_q4_k(&sb_data).expect("test");

    assert_eq!(result.len(), QK_K);
    // With our setup, values should follow pattern: d * scale * q - dmin * m
    // The exact values depend on scale extraction logic, but they should be consistent
    assert!(result.iter().all(|&v| v.is_finite()));
}

include!("dequantize_06.rs");
include!("f16_03.rs");
include!("interleaved_q4k_02.rs");
