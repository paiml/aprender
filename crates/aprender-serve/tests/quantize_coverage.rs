//! EXTREME TDD coverage tests for realizar/src/quantize.rs
//!
//! Target: Increase coverage from 76% to 90%+
//! Focus areas:
//! - Q4_K, Q5_K, Q6_K dequantization edge cases
//! - Fused dot product operations
//! - Parallel matrix-vector operations
//! - Error handling paths
//! - SIMD paths (scalar fallbacks)

use realizar::quantize::{
    apply_rope_rotation_simd, dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q4_k_parallel, dequantize_q4_k_simd, dequantize_q5_0, dequantize_q5_1,
    dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, dequantize_q8_0_parallel,
    dequantize_q8_0_simd, dequantize_q8_blocks, detect_simd_backend, f16_to_f32,
    fused_q4_0_q8_0_parallel_matvec, fused_q4_0_q8_0_parallel_matvec_into, fused_q4k_dot,
    fused_q4k_dot_simd, fused_q4k_parallel_matvec, fused_q4k_parallel_matvec_into,
    fused_q4k_q8_dot, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd, fused_q4k_q8k_ffn_up_gate_into,
    fused_q4k_q8k_parallel_matvec_into, fused_q4k_tiled_matvec, fused_q5k_dot, fused_q5k_dot_simd,
    fused_q5k_parallel_matvec, fused_q5k_parallel_matvec_into, fused_q6k_dot, fused_q6k_dot_simd,
    fused_q6k_parallel_matvec, fused_q6k_parallel_matvec_into, fused_q8_0_q8_0_parallel_matvec,
    fused_q8_0_q8_0_parallel_matvec_into, fused_rmsnorm_ffn_up_gate, fused_rmsnorm_q4_0_matmul,
    fused_swiglu_simd, quantize_activations_q8_0, quantize_activations_q8k_into,
    quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, quantize_to_q8_blocks, softmax_simd,
    DequantStats, Q4_0Block, Q4_KBlock, Q5_KBlock, Q6_KBlock, Q8KSuperBlock, Q8_0Block,
    SimdBackend, QK_K,
};

// ============================================================================
// Q4_0 Dequantization Tests
// ============================================================================

#[test]
fn test_q4_0_empty_data() {
    let result = dequantize_q4_0(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q4_0_negative_scale() {
    let mut data = Vec::new();
    // Scale: -1.0 as f16
    data.extend_from_slice(&half::f16::from_f32(-1.0).to_le_bytes());
    // 16 bytes of quants (all zeros for simplicity)
    data.extend_from_slice(&[0x88u8; 16]); // low=8, high=8

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // Values should be negative of positive scale
}

#[test]
fn test_q4_0_very_small_scale() {
    let mut data = Vec::new();
    // Scale: very small f16
    data.extend_from_slice(&half::f16::from_f32(1e-4).to_le_bytes());
    data.extend_from_slice(&[0xFFu8; 16]); // all 15s

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // All values should be very small
    for val in result {
        assert!(val.abs() < 1e-2);
    }
}

// ============================================================================
// Q8_0 Dequantization Tests
// ============================================================================

#[test]
fn test_q8_0_empty_data() {
    let result = dequantize_q8_0(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q8_0_negative_quants() {
    let mut data = Vec::new();
    // Scale: 1.0 as f16
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // 32 int8 values with negative values
    for i in 0..32_i8 {
        #[allow(clippy::cast_sign_loss)]
        data.push((i - 16) as u8);
    }

    let result = dequantize_q8_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    assert!((result[0] - (-16.0)).abs() < 1e-3);
}

#[test]
fn test_q8_0_large_scale() {
    let mut data = Vec::new();
    // Scale: 100.0 as f16
    data.extend_from_slice(&half::f16::from_f32(100.0).to_le_bytes());
    data.extend_from_slice(&[1u8; 32]);

    let result = dequantize_q8_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // All values should be ~100.0 (1 * 100)
    for val in &result {
        assert!((*val - 100.0).abs() < 1.0);
    }
}

// ============================================================================
// Q4_K Dequantization Tests
// ============================================================================

#[test]
fn test_q4k_varied_scales() {
    // Create a Q4_K block with varied scale values
    let mut data = vec![0u8; 144];

    // d = 2.0 (f16)
    let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];

    // dmin = 1.0 (f16)
    let dmin_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    data[2] = dmin_bytes[0];
    data[3] = dmin_bytes[1];

    // scales: set varied patterns
    for i in 0..12 {
        data[4 + i] = ((i * 5) % 64) as u8;
    }

    // qs: varied patterns
    for i in 0..128 {
        data[16 + i] = ((i * 3) % 256) as u8;
    }

    let result = dequantize_q4_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q4k_max_d_value() {
    let mut data = vec![0u8; 144];

    // d = max f16 (~65504)
    let d_bytes = half::f16::from_f32(65504.0).to_bits().to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];

    // dmin = 0
    data[2] = 0;
    data[3] = 0;

    let result = dequantize_q4_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Q5_K Dequantization Tests
// ============================================================================

#[test]
fn test_q5k_with_high_bits() {
    // Q5_K uses 5 bits: 4 bits from qs + 1 bit from qh
    let mut data = vec![0u8; 176];

    // d = 1.0
    let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    data[0] = d_bytes[0];
    data[1] = d_bytes[1];

    // dmin = 0.0
    data[2] = 0;
    data[3] = 0;

    // scales
    for i in 0..12 {
        data[4 + i] = 0x3F; // max scale
    }

    // qh: set all high bits to 1
    for i in 0..32 {
        data[16 + i] = 0xFF;
    }

    // qs: set to mid-range
    for i in 0..128 {
        data[48 + i] = 0x88;
    }

    let result = dequantize_q5_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q5k_negative_dmin() {
    let mut data = vec![0u8; 176];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    // dmin = -1.0 (negative)
    data[2..4].copy_from_slice(&half::f16::from_f32(-1.0).to_bits().to_le_bytes());

    let result = dequantize_q5_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q5k_multiple_blocks() {
    // 3 super-blocks
    let data = vec![0u8; 176 * 3];
    let result = dequantize_q5_k(&data).expect("valid data");
    assert_eq!(result.len(), 256 * 3);
}

// ============================================================================
// Q6_K Dequantization Tests
// ============================================================================

#[test]
fn test_q6k_with_varied_qh() {
    // Q6_K: ql (128) + qh (64) + scales (16) + d (2) = 210
    let mut data = vec![0u8; 210];

    // ql: varied
    for (i, byte) in data.iter_mut().enumerate().take(128) {
        *byte = (i % 256) as u8;
    }

    // qh: varied (upper 2 bits)
    for i in 0..64 {
        data[128 + i] = 0xAA; // alternating pattern
    }

    // scales: varied signed
    for i in 0..16 {
        #[allow(clippy::cast_sign_loss)]
        {
            data[192 + i] = (i as i8 - 8) as u8;
        }
    }

    // d = 0.5
    data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

    let result = dequantize_q6_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q6k_negative_scales() {
    let mut data = vec![0u8; 210];

    // Set negative scales
    for i in 0..16 {
        data[192 + i] = 0xFF; // -1 as i8
    }

    // d = 1.0
    data[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    let result = dequantize_q6_k(&data).expect("valid block");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Additional Q4_K Tests
// ============================================================================

#[test]
fn test_q4k_zero_dmin() {
    let mut data = vec![0u8; 144];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    // dmin = 0.0 (no minimum subtraction)
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    // Set varied scale pattern
    for i in 0..12 {
        data[4 + i] = ((i + 1) * 4) as u8;
    }

    let result = dequantize_q4_k(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_q4k_large_scale_values() {
    let mut data = vec![0u8; 144];

    // d = 10.0
    data[0..2].copy_from_slice(&half::f16::from_f32(10.0).to_bits().to_le_bytes());
    // dmin = 5.0
    data[2..4].copy_from_slice(&half::f16::from_f32(5.0).to_bits().to_le_bytes());

    // Max scales
    for i in 0..12 {
        data[4 + i] = 0xFF;
    }

    let result = dequantize_q4_k(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Fused Q4_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q4k_dot_all_zeros() {
    let q4k_data = vec![0u8; 144];
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations).expect("valid");
    assert!(result.abs() < 1e-6);
}

#[test]
fn test_fused_q4k_dot_simd_error_paths() {
    // Invalid data length
    let bad_data = vec![0u8; 100]; // not multiple of 144
    let activations = vec![0.0f32; 256];
    assert!(fused_q4k_dot_simd(&bad_data, &activations).is_err());

    // Activation length mismatch
    let good_data = vec![0u8; 144];
    let bad_activations = vec![0.0f32; 100];
    assert!(fused_q4k_dot_simd(&good_data, &bad_activations).is_err());
}

// ============================================================================
// Fused Q5_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q5k_dot_basic() {
    let mut q5k_data = vec![0u8; 176];

    // d = 1.0
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    // dmin = 0.0
    q5k_data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    let activations: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();

    let result = fused_q5k_dot(&q5k_data, &activations).expect("valid");
    // Just verify it produces a finite result
    assert!(result.is_finite());
}

#[test]
fn test_fused_q5k_dot_simd_wrapper() {
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];

    // Should use scalar fallback (simd wrapper)
    let result = fused_q5k_dot_simd(&q5k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q5k_dot_invalid_data() {
    let bad_data = vec![0u8; 175]; // not multiple of 176
    let activations = vec![0.0f32; 256];
    assert!(fused_q5k_dot(&bad_data, &activations).is_err());
}

#[test]
fn test_fused_q5k_dot_activation_mismatch() {
    let data = vec![0u8; 176];
    let activations = vec![0.0f32; 128]; // wrong length
    assert!(fused_q5k_dot(&data, &activations).is_err());
}

// ============================================================================
// Fused Q6_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q6k_dot_basic() {
    let mut q6k_data = vec![0u8; 210];

    // d at offset 208
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    let activations: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();

    let result = fused_q6k_dot(&q6k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_dot_simd_wrapper() {
    let mut q6k_data = vec![0u8; 210];
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

    let activations = vec![1.0f32; 256];

    let result = fused_q6k_dot_simd(&q6k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_dot_invalid_data() {
    let bad_data = vec![0u8; 209]; // not multiple of 210
    let activations = vec![0.0f32; 256];
    assert!(fused_q6k_dot(&bad_data, &activations).is_err());
}

// ============================================================================
// Q6_K x Q8_K Fused Dot Product Tests
// ============================================================================

// ============================================================================
// Tiled Matrix-Vector Multiply Tests
// ============================================================================

#[test]
fn test_fused_q4k_tiled_matvec_error_weight_too_small() {
    let weight_data = vec![0u8; 100]; // too small
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_tiled_matvec(&weight_data, &activations, 256, 4, None);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_tiled_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![0.0f32; 128]; // wrong length

    let result = fused_q4k_tiled_matvec(&weight_data, &activations, 256, 4, None);
    assert!(result.is_err());
}

// ============================================================================
// Parallel Matrix-Vector Multiply Tests
// ============================================================================

#[test]
fn test_fused_q4k_parallel_matvec_small() {
    // Test sequential path (out_dim < 256)
    let weight_data = vec![0u8; 144 * 10]; // 10 rows
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_parallel_matvec(&weight_data, &activations, 256, 10).expect("valid");
    assert_eq!(result.len(), 10);
}

#[test]
fn test_fused_q4k_parallel_matvec_large() {
    // Test parallel path (out_dim >= 256)
    let weight_data = vec![0u8; 144 * 300]; // 300 rows
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_parallel_matvec(&weight_data, &activations, 256, 300).expect("valid");
    assert_eq!(result.len(), 300);
}

#[test]
fn test_fused_q4k_parallel_matvec_into_errors() {
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2]; // too small

    let result = fused_q4k_parallel_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_parallel_matvec_basic() {
    let weight_data = vec![0u8; 176 * 10];
    let activations = vec![1.0f32; 256];

    let result = fused_q5k_parallel_matvec(&weight_data, &activations, 256, 10).expect("valid");
    assert_eq!(result.len(), 10);
}

#[test]
fn test_fused_q5k_parallel_matvec_into_errors() {
    let weight_data = vec![0u8; 176 * 4];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2]; // too small

    let result = fused_q5k_parallel_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_basic() {
    let weight_data = vec![0u8; 210 * 10];
    let activations = vec![1.0f32; 256];

    let result = fused_q6k_parallel_matvec(&weight_data, &activations, 256, 10).expect("valid");
    assert_eq!(result.len(), 10);
}

#[test]
fn test_fused_q6k_parallel_matvec_into_errors() {
    // Weight too small
    let weight_data = vec![0u8; 100];
    let activations = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 4];

    let result = fused_q6k_parallel_matvec_into(&weight_data, &activations, 256, 4, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Q4_K x Q8_K Parallel Matvec Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_basic() {
    let weight_data = vec![0u8; 144 * 4];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut output = vec![0.0f32; 4];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut output,
    );
    assert!(result.is_ok());
}

// ============================================================================
// Quantize Activations Tests
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_basic() {
    let activations: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_all_zeros() {
    let activations = vec![0.0f32; 64];

    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2);
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_activations_q8_0_partial_block() {
    // 50 values: 1 full block + partial
    let activations: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();

    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 2); // ceil(50/32) = 2
    assert_eq!(quants.len(), 64); // padded to 2*32
}

#[test]
fn test_quantize_activations_q8k_into() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants)
        .expect("valid quantization");

    assert!(scales[0] > 0.0);
}

// ============================================================================
// RMSNorm + Q8_0 Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_large_input() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let norm_weight = vec![1.0f32; 256];

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);

    assert_eq!(scales.len(), 8); // 256/32 = 8 blocks
    assert_eq!(quants.len(), 256);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 64];
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, 1e-6, &mut scales, &mut quants);

    assert!(scales[0] > 0.0 || scales[1] > 0.0);
}

// ============================================================================
// Fused RMSNorm + Matmul Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_q4_0_matmul_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18 * 4]; // 4 output rows

    let result =
        fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-6, &weight_data, 32, 4).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_error_input_mismatch() {
    let input = vec![1.0f32; 16]; // wrong
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18 * 4];

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-6, &weight_data, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_error_weight_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 10]; // too small

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-6, &weight_data, 32, 4);
    assert!(result.is_err());
}

// ============================================================================
// Fused FFN Up/Gate Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 18 * 8];

    let (up, gate) =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-6, &up_weight, &gate_weight, 32, 8)
            .expect("valid");

    assert_eq!(up.len(), 8);
    assert_eq!(gate.len(), 8);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_error_input_mismatch() {
    let input = vec![1.0f32; 16]; // wrong
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 18 * 8];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-6, &up_weight, &gate_weight, 32, 8);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_error_up_weight_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 10]; // too small
    let gate_weight = vec![0u8; 18 * 8];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-6, &up_weight, &gate_weight, 32, 8);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_error_gate_weight_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18 * 8];
    let gate_weight = vec![0u8; 10]; // too small

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-6, &up_weight, &gate_weight, 32, 8);
    assert!(result.is_err());
}

// ============================================================================
// SwiGLU Tests
// ============================================================================

#[test]
fn test_fused_swiglu_simd_basic() {
    let mut gate = vec![0.0f32, 1.0, -1.0, 2.0];
    let up = vec![1.0f32, 2.0, 3.0, 4.0];

    fused_swiglu_simd(&mut gate, &up);

    // silu(x) = x * sigmoid(x)
    // gate[0] = silu(0) * 1 = 0 * 0.5 * 1 = 0
    assert!(gate[0].abs() < 1e-6);
    // silu(1) ~ 0.731
    assert!((gate[1] - 0.731 * 2.0).abs() < 0.1);
}

#[test]
fn test_fused_swiglu_simd_large() {
    // Test SIMD path (>8 elements)
    let mut gate: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let up: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.5).collect();

    fused_swiglu_simd(&mut gate, &up);

    // Verify all values are finite
    for v in &gate {
        assert!(v.is_finite());
    }
}

#[test]
fn test_fused_swiglu_simd_negative() {
    let mut gate = vec![-5.0f32; 8];
    let up = vec![1.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(-5) is very small (close to 0)
    for v in &gate {
        assert!(v.abs() < 0.1);
    }
}

// ============================================================================
// Softmax Tests
// ============================================================================

#[test]
fn test_softmax_simd_basic() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];

    softmax_simd(&mut x);

    // Verify sum to 1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify monotonicity
    for i in 1..x.len() {
        assert!(x[i] > x[i - 1]);
    }
}

#[test]
fn test_softmax_simd_empty() {
    let mut x: Vec<f32> = vec![];
    softmax_simd(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_simd_single() {
    let mut x = vec![5.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_large() {
    // Test SIMD path (>8 elements)
    let mut x: Vec<f32> = (0..32).map(|i| i as f32).collect();

    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_numerical_stability() {
    // Large values that could overflow without max subtraction
    let mut x = vec![1000.0f32, 1001.0, 1002.0];

    softmax_simd(&mut x);

    // Should still sum to 1
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All values should be finite
    for v in &x {
        assert!(v.is_finite());
    }
}

// ============================================================================
// QK_K Constant Test
// ============================================================================

#[test]
fn test_qk_k_constant() {
    // QK_K should be 256 (super-block size for K-quants)
    assert_eq!(QK_K, 256);
}

// ============================================================================
// Q8_0Block Method Tests
// ============================================================================

#[test]
fn test_q8_0_block_quantize_basic() {
    let values: [f32; 32] = core::array::from_fn(|i| (i as f32 - 16.0) / 16.0);
    let block = Q8_0Block::quantize(&values);

    assert!(block.scale > 0.0);
    assert_eq!(block.quants.len(), 32);
}

#[test]
fn test_q8_0_block_dequantize() {
    let values: [f32; 32] = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    assert_eq!(dequant.len(), 32);
    for val in &dequant {
        assert!((*val - 1.0).abs() < 0.02);
    }
}

#[test]
fn test_q8_0_block_quantization_error() {
    let values: [f32; 32] = core::array::from_fn(|i| i as f32 * 0.1);
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);

    // Error should be small for well-distributed values
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_relative_error() {
    let values: [f32; 32] = core::array::from_fn(|i| (i as f32 + 1.0) * 0.5);
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);

    // Relative error should be small
    assert!(rel_error < 0.05);
}

#[test]
fn test_q8_0_block_relative_error_near_zero() {
    let values: [f32; 32] = [1e-15f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);

    // When all values are near zero, relative error should be 0
    assert!((rel_error - 0.0).abs() < 1e-6);
}

// ============================================================================
// Q8KSuperBlock Tests
// ============================================================================

#[test]
fn test_q8k_superblock_quantize() {
    let values: [f32; 256] = core::array::from_fn(|i| (i as f32 - 128.0) / 128.0);
    let block = Q8KSuperBlock::quantize(&values);

    assert!(block.scale > 0.0);
    assert_eq!(block.quants.len(), 256);
}

#[test]
fn test_q8k_superblock_quantize_into() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
}

#[test]
fn test_q8k_superblock_dequantize() {
    let values: [f32; 256] = [0.5f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();

    assert_eq!(dequant.len(), 256);
    for val in &dequant {
        assert!((*val - 0.5).abs() < 0.02);
    }
}

#[test]
fn test_q8k_superblock_all_zeros() {
    let values: [f32; 256] = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);

    // Scale should be minimal but non-zero
    assert!(block.scale > 0.0);
    // All quants should be zero
    for q in &block.quants {
        assert_eq!(*q, 0);
    }
}

// ============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks Tests
// ============================================================================

#[test]
fn test_quantize_to_q8_blocks_basic() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("valid");

    assert_eq!(blocks.len(), 2); // 64 / 32 = 2 blocks
}

#[test]
fn test_quantize_to_q8_blocks_error_not_multiple() {
    let values: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let result = quantize_to_q8_blocks(&values);

    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks_roundtrip() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let blocks = quantize_to_q8_blocks(&original).expect("valid");
    let recovered = dequantize_q8_blocks(&blocks);

    assert_eq!(recovered.len(), 64);
    // Check round-trip error is small
    for (o, r) in original.iter().zip(recovered.iter()) {
        assert!((o - r).abs() < 0.02);
    }
}

// ============================================================================
// f16_to_f32 and dequantize_f16 Tests
// ============================================================================

#[test]
fn test_f16_to_f32_basic_values() {
    // Test f16 representation of 1.0
    let one_f16 = half::f16::from_f32(1.0).to_bits();
    let result = f16_to_f32(one_f16);
    assert!((result - 1.0).abs() < 1e-6);

    // Test f16 representation of -1.0
    let neg_one_f16 = half::f16::from_f32(-1.0).to_bits();
    let result = f16_to_f32(neg_one_f16);
    assert!((result - (-1.0)).abs() < 1e-6);

    // Test zero
    let zero_f16 = half::f16::from_f32(0.0).to_bits();
    let result = f16_to_f32(zero_f16);
    assert!(result.abs() < 1e-10);
}

#[test]
fn test_dequantize_f16_basic() {
    // Create f16 data for [1.0, 2.0, 3.0]
    let mut data = Vec::new();
    for val in [1.0f32, 2.0, 3.0] {
        data.extend_from_slice(&half::f16::from_f32(val).to_le_bytes());
    }

    let result = dequantize_f16(&data).expect("valid");
    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.0).abs() < 1e-3);
    assert!((result[1] - 2.0).abs() < 1e-3);
    assert!((result[2] - 3.0).abs() < 1e-3);
}

#[test]
fn test_dequantize_f16_error_odd_bytes() {
    let data = vec![0u8; 5]; // Not multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_f16_empty() {
    let result = dequantize_f16(&[]).expect("valid");
    assert!(result.is_empty());
}

// ============================================================================
// Q4_1 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q4_1_basic() {
    // Q4_1 block: 2 (d) + 2 (min) + 16 (quants) = 20 bytes
    let mut data = vec![0u8; 20];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // min = 0.0
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // quants all 0x00 (low=0, high=0)
    // value = d * q + min = 1.0 * 0 + 0 = 0

    let result = dequantize_q4_1(&data).expect("valid");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q4_1_with_min() {
    let mut data = vec![0u8; 20];

    // d = 0.1
    data[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
    // min = 5.0
    data[2..4].copy_from_slice(&half::f16::from_f32(5.0).to_le_bytes());
    // quants all 0x00

    let result = dequantize_q4_1(&data).expect("valid");
    assert_eq!(result.len(), 32);
    // All values should be ~5.0 (d * 0 + min)
    for val in &result {
        assert!((*val - 5.0).abs() < 0.1);
    }
}

#[test]
fn test_dequantize_q4_1_error_invalid_length() {
    let data = vec![0u8; 19]; // Not multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q5_0 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q5_0_basic() {
    // Q5_0 block: 2 (d) + 4 (qh) + 16 (qs) = 22 bytes
    let mut data = vec![0u8; 22];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // qh = 0 (no high bits)
    // qs = 0 (all zeros)
    // value = d * (q - 16) = 1.0 * (-16) = -16.0

    let result = dequantize_q5_0(&data).expect("valid");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_0_with_high_bits() {
    let mut data = vec![0u8; 22];

    // d = 0.5
    data[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // qh = all 1s (high bits set)
    data[2..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    // qs = all 0xF (max low nibble)
    for byte in &mut data[6..22] {
        *byte = 0xFF;
    }

    let result = dequantize_q5_0(&data).expect("valid");
    assert_eq!(result.len(), 32);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q5_0_error_invalid_length() {
    let data = vec![0u8; 21]; // Not multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

// ============================================================================
// Q5_1 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q5_1_basic() {
    // Q5_1 block: 2 (d) + 2 (min) + 4 (qh) + 16 (qs) = 24 bytes
    let mut data = vec![0u8; 24];

    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // min = 2.0
    data[2..4].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    // qh = 0
    // qs = 0
    // value = d * q + min = 1.0 * 0 + 2.0 = 2.0

    let result = dequantize_q5_1(&data).expect("valid");
    assert_eq!(result.len(), 32);
    for val in &result {
        assert!((*val - 2.0).abs() < 0.1);
    }
}

#[test]
fn test_dequantize_q5_1_with_high_bits() {
    let mut data = vec![0u8; 24];

    // d = 0.1
    data[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
    // min = 0.0
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // qh = all 1s
    data[4..8].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    // qs = varied
    for (i, byte) in data[8..24].iter_mut().enumerate() {
        *byte = (i % 16) as u8;
    }

    let result = dequantize_q5_1(&data).expect("valid");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_1_error_invalid_length() {
    let data = vec![0u8; 23]; // Not multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// ============================================================================
// Fused Q4_K x Q8_0 Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8_dot_basic() {
    // Q4_K: 144 bytes per super-block (256 values)
    let q4k_data = vec![0u8; 144];
    // Q8_0: 8 blocks per super-block (256/32)
    let q8_blocks: Vec<Q8_0Block> = (0..8)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [1i8; 32],
        })
        .collect();

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4k_q8_dot_error_invalid_q4k_length() {
    let q4k_data = vec![0u8; 143]; // Not multiple of 144
    let q8_blocks: Vec<Q8_0Block> = vec![];

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8_dot_error_block_count_mismatch() {
    let q4k_data = vec![0u8; 144];
    let q8_blocks: Vec<Q8_0Block> = (0..4)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [1i8; 32],
        })
        .collect(); // Need 8, have 4

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_err());
}

// ============================================================================
// Fused Q4_K x Q8_K Dot Product Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_dot_basic() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_error_invalid_q4k_length() {
    let q4k_data = vec![0u8; 143];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_error_scales_too_small() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales: Vec<f32> = vec![]; // empty
    let q8k_quants = vec![1i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_error_quants_too_small() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 128]; // Need 256

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_dot_simd_basic() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![0.5f32; 1];
    let q8k_quants = vec![2i8; 256];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("valid");
    assert!(result.is_finite());
}

// ============================================================================
// Fused Q4_K x Q8_K FFN Up/Gate Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_basic() {
    let up_weight = vec![0u8; 144 * 4]; // 4 output rows
    let gate_weight = vec![0u8; 144 * 4];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut up_output = vec![0.0f32; 4];
    let mut gate_output = vec![0.0f32; 4];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weight,
        &gate_weight,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_error_weight_too_small() {
    let up_weight = vec![0u8; 100]; // Too small
    let gate_weight = vec![0u8; 144 * 4];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut up_output = vec![0.0f32; 4];
    let mut gate_output = vec![0.0f32; 4];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weight,
        &gate_weight,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// ============================================================================
// Parallel Q4_K Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q4_k_parallel_basic() {
    let data = vec![0u8; 144]; // 1 super-block
    let result = dequantize_q4_k_parallel(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4_k_parallel_multiple_blocks() {
    let data = vec![0u8; 144 * 4]; // 4 super-blocks
    let result = dequantize_q4_k_parallel(&data).expect("valid");
    assert_eq!(result.len(), 256 * 4);
}

#[test]
fn test_dequantize_q4_k_parallel_error_invalid_length() {
    let data = vec![0u8; 143];
    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_simd_basic() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_simd(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4_k_simd_large() {
    // Test with many super-blocks to trigger parallel path
    let data = vec![0u8; 144 * 200];
    let result = dequantize_q4_k_simd(&data).expect("valid");
    assert_eq!(result.len(), 256 * 200);
}

// ============================================================================
// Parallel Q8_0 Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q8_0_parallel_basic() {
    // Q8_0 block: 2 (f16 scale) + 32 (i8 quants) = 34 bytes
    let data = vec![0u8; 34];
    let result = dequantize_q8_0_parallel(&data).expect("valid");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q8_0_parallel_multiple_blocks() {
    let data = vec![0u8; 34 * 4];
    let result = dequantize_q8_0_parallel(&data).expect("valid");
    assert_eq!(result.len(), 32 * 4);
}

#[test]
fn test_dequantize_q8_0_parallel_error_invalid_length() {
    let data = vec![0u8; 33];
    let result = dequantize_q8_0_parallel(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_simd_basic() {
    let data = vec![0u8; 34];
    let result = dequantize_q8_0_simd(&data).expect("valid");
    assert_eq!(result.len(), 32);
}

// ============================================================================
// Q4_0 x Q8_0 Parallel Matvec Tests
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_basic() {
    // Q4_0: 18 bytes per 32 values
    // 4 rows x 32 columns = 4 * 18 bytes
    let weight_data = vec![0u8; 18 * 4];
    let activations = vec![1.0f32; 32];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_large() {
    // Test with large out_dim to trigger parallel path (>1024)
    let weight_data = vec![0u8; 18 * 1500];
    let activations = vec![1.0f32; 32];

    let result =
        fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1500).expect("valid");
    assert_eq!(result.len(), 1500);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_error_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 18 * 4];
    let activations = vec![1.0f32; 16]; // Wrong length

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_basic() {
    let weight_data = vec![0u8; 18 * 4];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 4];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_error_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 4];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_error_activation_mismatch() {
    let weight_data = vec![0u8; 18 * 4];
    let activations = vec![1.0f32; 16]; // Wrong
    let mut output = vec![0.0f32; 4];

    let result = fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Q8_0 x Q8_0 Parallel Matvec Tests
// ============================================================================

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_basic() {
    // Q8_0 weight: 34 bytes per 32 values
    let weight_data = vec![0u8; 34 * 4];
    let activations = vec![1.0f32; 32];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4).expect("valid");
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_error_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_error_activation_mismatch() {
    let weight_data = vec![0u8; 34 * 4];
    let activations = vec![1.0f32; 16]; // Wrong

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_basic() {
    let weight_data = vec![0u8; 34 * 4];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 4];

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 4, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_error_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 4];

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 4, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_error_activation_mismatch() {
    let weight_data = vec![0u8; 34 * 4];
    let activations = vec![1.0f32; 16]; // Wrong
    let mut output = vec![0.0f32; 4];

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 4, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_error_output_too_small() {
    let weight_data = vec![0u8; 34 * 4];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2]; // Too small

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 4, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// RoPE Rotation Tests
// ============================================================================

#[test]
fn test_apply_rope_rotation_simd_basic() {
    let mut x1 = vec![1.0f32, 0.0, 1.0, 0.0];
    let mut x2 = vec![0.0f32, 1.0, 0.0, 1.0];
    let cos_vals = vec![1.0f32; 4]; // cos(0) = 1
    let sin_vals = vec![0.0f32; 4]; // sin(0) = 0

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: x1' = x1*1 - x2*0 = x1, x2' = x1*0 + x2*1 = x2
    assert!((x1[0] - 1.0).abs() < 1e-6);
    assert!((x2[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_apply_rope_rotation_simd_90_degrees() {
    let mut x1 = vec![1.0f32; 8];
    let mut x2 = vec![0.0f32; 8];
    let cos_vals = vec![0.0f32; 8]; // cos(90) = 0
    let sin_vals = vec![1.0f32; 8]; // sin(90) = 1

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=0, sin=1: x1' = 1*0 - 0*1 = 0, x2' = 1*1 + 0*0 = 1
    for val in &x1 {
        assert!(val.abs() < 1e-6);
    }
    for val in &x2 {
        assert!((*val - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_apply_rope_rotation_simd_large() {
    // Test with large vectors to exercise SIMD path
    let mut x1 = vec![1.0f32; 64];
    let mut x2 = vec![0.5f32; 64];
    let cos_vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).cos()).collect();
    let sin_vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // Just verify all values are finite
    for val in &x1 {
        assert!(val.is_finite());
    }
    for val in &x2 {
        assert!(val.is_finite());
    }
}

// ============================================================================
// DequantStats and SimdBackend Tests
// ============================================================================

#[test]
fn test_dequant_stats_default() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
}

#[test]
fn test_simd_backend_display() {
    assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
    assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
}

#[test]
fn test_simd_backend_default() {
    let backend = SimdBackend::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_detect_simd_backend() {
    let backend = detect_simd_backend();
    // Just verify it returns a valid backend
    match backend {
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar => {},
    }
}

// ============================================================================
// Multiple Blocks Dequantization Tests
// ============================================================================

#[test]
fn test_dequantize_q4_1_multiple_blocks() {
    let data = vec![0u8; 20 * 3]; // 3 blocks
    let result = dequantize_q4_1(&data).expect("valid");
    assert_eq!(result.len(), 32 * 3);
}

#[test]
fn test_dequantize_q5_0_multiple_blocks() {
    let data = vec![0u8; 22 * 3]; // 3 blocks
    let result = dequantize_q5_0(&data).expect("valid");
    assert_eq!(result.len(), 32 * 3);
}

#[test]
fn test_dequantize_q5_1_multiple_blocks() {
    let data = vec![0u8; 24 * 3]; // 3 blocks
    let result = dequantize_q5_1(&data).expect("valid");
    assert_eq!(result.len(), 32 * 3);
}

// ============================================================================
// Edge Cases and Numerical Stability Tests
// ============================================================================

#[test]
fn test_q8_0_block_extreme_values() {
    let values: [f32; 32] = [127.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    for val in &dequant {
        assert!((*val - 127.0).abs() < 1.0);
    }
}

#[test]
fn test_q8_0_block_mixed_sign_values() {
    let mut values = [0.0f32; 32];
    for (i, val) in values.iter_mut().enumerate() {
        *val = if i % 2 == 0 { 50.0 } else { -50.0 };
    }
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    for (i, val) in dequant.iter().enumerate() {
        let expected = if i % 2 == 0 { 50.0 } else { -50.0 };
        assert!((val - expected).abs() < 1.0);
    }
}

#[test]
fn test_fused_q4k_q8_dot_multiple_superblocks() {
    let q4k_data = vec![0u8; 144 * 3]; // 3 super-blocks
    let q8_blocks: Vec<Q8_0Block> = (0..24)
        .map(|_| Q8_0Block {
            scale: 1.0,
            quants: [1i8; 32],
        })
        .collect();

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_multiple_superblocks() {
    let q4k_data = vec![0u8; 144 * 3];
    let q8k_scales = vec![1.0f32; 3];
    let q8k_quants = vec![1i8; 256 * 3];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("valid");
    assert!(result.is_finite());
}

// ============================================================================
// InterleavedQ4K Tests
// ============================================================================

use realizar::quantize::InterleavedQ4K;

#[test]
fn test_interleaved_q4k_from_q4k_basic() {
    // Create valid Q4_K data (144 bytes per super-block)
    let mut data = vec![0u8; 144];

    // d = 1.0 as f16
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // dmin = 0.5 as f16
    data[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    assert_eq!(interleaved.num_super_blocks, 1);
    assert_eq!(interleaved.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_from_q4k_multiple_blocks() {
    let data = vec![0u8; 144 * 4]; // 4 super-blocks
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");

    assert_eq!(interleaved.num_super_blocks, 4);
    assert_eq!(interleaved.num_values(), 256 * 4);
    assert_eq!(interleaved.d.len(), 4);
    assert_eq!(interleaved.dmin.len(), 4);
    assert_eq!(interleaved.scales.len(), 12 * 4);
    assert_eq!(interleaved.qs.len(), 128 * 4);
}

#[test]
fn test_interleaved_q4k_from_q4k_error_invalid_length() {
    let data = vec![0u8; 143]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_dot_basic() {
    let mut data = vec![0u8; 144];

    // d = 0.1
    data[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
    // dmin = 0.0
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_dot_error_activation_mismatch() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 128]; // Wrong length

    let result = interleaved.dot(&activations);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_dot_varied_activations() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    // Set some scale values
    data[4..16].iter_mut().enumerate().for_each(|(i, byte)| {
        *byte = ((i + 1) * 5) as u8;
    });

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();

    let result = interleaved.dot(&activations).expect("valid");
    assert!(result.is_finite());
}

// ============================================================================
// Q4_0Block and Q8_0Block Struct Field Tests
// ============================================================================

#[test]
fn test_q4_0_block_struct() {
    let block = Q4_0Block {
        scale: 0.5,
        quants: [0xAB; 16],
    };
    assert_eq!(block.scale, 0.5);
    assert_eq!(block.quants[0], 0xAB);
    assert_eq!(block.quants.len(), 16);
}

#[test]
fn test_q4_k_block_struct() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0x12; 12],
        qs: [0x34; 128],
    };
    assert_eq!(block.d, 1.0);
    assert_eq!(block.dmin, 0.5);
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qs.len(), 128);
}

#[test]
fn test_q5_k_block_struct() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 1.0,
        scales: [0x56; 12],
        qh: [0x78; 32],
        qs: [0x9A; 128],
    };
    assert_eq!(block.d, 2.0);
    assert_eq!(block.dmin, 1.0);
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qh.len(), 32);
    assert_eq!(block.qs.len(), 128);
}

#[test]
fn test_q6_k_block_struct() {
    let block = Q6_KBlock {
        d: 0.25,
        scales: [1; 16],
        qh: [0xBC; 64],
        qs: [0xDE; 128],
    };
    assert_eq!(block.d, 0.25);
    assert_eq!(block.scales.len(), 16);
    assert_eq!(block.qh.len(), 64);
    assert_eq!(block.qs.len(), 128);
}

// ============================================================================
// Additional Q4_0 Dequantization Edge Cases
// ============================================================================

#[test]
fn test_q4_0_zero_scale() {
    let mut data = Vec::new();
    // Scale: 0.0 as f16
    data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data.extend_from_slice(&[0xFF; 16]); // Max quant values

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // All values should be 0 when scale is 0
    for val in &result {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_q4_0_max_quant_values() {
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&[0xFF; 16]); // low=15, high=15

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // Values should be around 7 (15 - 8 = 7)
    for val in &result {
        assert!((*val - 7.0).abs() < 0.1);
    }
}

#[test]
fn test_q4_0_min_quant_values() {
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&[0x00; 16]); // low=0, high=0

    let result = dequantize_q4_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    // Values should be -8 (0 - 8 = -8)
    for val in &result {
        assert!((*val - (-8.0)).abs() < 0.1);
    }
}

#[test]
fn test_q4_0_multiple_blocks_varied() {
    let mut data = Vec::new();

    // Block 1: scale = 1.0, quants = 0x77 (low=7, high=7)
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&[0x77; 16]);

    // Block 2: scale = 2.0, quants = 0x88 (low=8, high=8)
    data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    data.extend_from_slice(&[0x88; 16]);

    let result = dequantize_q4_0(&data).expect("valid blocks");
    assert_eq!(result.len(), 64);
}

// ============================================================================
// Additional Q8_0 Dequantization Edge Cases
// ============================================================================

#[test]
fn test_q8_0_zero_scale() {
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data.extend_from_slice(&[127u8; 32]); // Max positive quants

    let result = dequantize_q8_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    for val in &result {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_q8_0_alternating_sign() {
    let mut data = Vec::new();
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // Alternating positive/negative values
    (0..32_i8).for_each(|i| {
        let val = if i % 2 == 0 { 64 } else { -64_i8 };
        #[allow(clippy::cast_sign_loss)]
        data.push(val as u8);
    });

    let result = dequantize_q8_0(&data).expect("valid block");
    assert_eq!(result.len(), 32);
    result.iter().enumerate().for_each(|(i, val)| {
        let expected = if i % 2 == 0 { 64.0 } else { -64.0 };
        assert!((val - expected).abs() < 0.1);
    });
}

// ============================================================================
// Additional Q4_K Dequantization Edge Cases
// ============================================================================

#[test]
fn test_q4k_empty_data() {
    let result = dequantize_q4_k(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q4k_error_invalid_length() {
    let data = vec![0u8; 143]; // Not multiple of 144
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_q4k_multiple_superblocks() {
    let data = vec![0u8; 144 * 5]; // 5 super-blocks
    let result = dequantize_q4_k(&data).expect("valid");
    assert_eq!(result.len(), 256 * 5);
}

// ============================================================================
// Additional Q5_K Dequantization Edge Cases
// ============================================================================

#[test]
fn test_q5k_empty_data() {
    let result = dequantize_q5_k(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q5k_error_invalid_length() {
    let data = vec![0u8; 175]; // Not multiple of 176
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_q5k_zero_high_bits() {
    let mut data = vec![0u8; 176];
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // qh = 0 (no high bits set)
    // This means the 5th bit is always 0

    let result = dequantize_q5_k(&data).expect("valid");
    assert_eq!(result.len(), 256);
}

// ============================================================================
// Additional Q6_K Dequantization Edge Cases
// ============================================================================

#[test]
fn test_q6k_empty_data() {
    let result = dequantize_q6_k(&[]).expect("empty should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_q6k_error_invalid_length() {
    let data = vec![0u8; 209]; // Not multiple of 210
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_q6k_multiple_superblocks() {
    let data = vec![0u8; 210 * 3]; // 3 super-blocks
    let result = dequantize_q6_k(&data).expect("valid");
    assert_eq!(result.len(), 256 * 3);
}

#[test]
fn test_q6k_all_max_scales() {
    let mut data = vec![0u8; 210];

    // Set all scales to max positive value (127)
    data[192..208].iter_mut().for_each(|byte| *byte = 127);

    // d = 1.0
    data[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    let result = dequantize_q6_k(&data).expect("valid");
    assert_eq!(result.len(), 256);
    for val in &result {
        assert!(val.is_finite());
    }
}

// ============================================================================
// Quantize Activations Q8K Error Paths
// ============================================================================

#[test]
fn test_quantize_activations_q8k_into_error_not_multiple_256() {
    let activations: Vec<f32> = (0..300).map(|i| i as f32).collect(); // Not multiple of 256
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_error_scales_too_small() {
    let activations: Vec<f32> = (0..512).map(|i| i as f32).collect(); // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Need 2, have 1
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_error_quants_too_small() {
    let activations: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 256]; // Need 512, have 256

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_large() {
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 1024];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("valid");

    // Verify scales are positive
    for s in &scales {
        assert!(*s > 0.0);
    }
}

// ============================================================================
// Fused Operations Additional Error Paths
// ============================================================================

#[test]
fn test_fused_q4k_tiled_matvec_in_dim_not_multiple_256() {
    let weight_data = vec![0u8; 144 * 4];
    let activations = vec![1.0f32; 200]; // Not multiple of 256

    // Note: function should handle padding internally
    let result = fused_q4k_tiled_matvec(&weight_data, &activations, 200, 4, None);
    // This may succeed or fail depending on implementation details
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_fused_q5k_parallel_matvec_error_weight_too_small() {
    let weight_data = vec![0u8; 100]; // Too small
    let activations = vec![1.0f32; 256];

    let result = fused_q5k_parallel_matvec(&weight_data, &activations, 256, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_q6k_parallel_matvec_error_weight_too_small() {
    let weight_data = vec![0u8; 100]; // Too small
    let activations = vec![1.0f32; 256];

    let result = fused_q6k_parallel_matvec(&weight_data, &activations, 256, 4);
    assert!(result.is_err());
}

// ============================================================================
// SIMD Dequantization Large Input Tests
// ============================================================================

#[test]
fn test_dequantize_q4_k_simd_error_invalid_length() {
    let data = vec![0u8; 143]; // Not multiple of 144
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_simd_large() {
    let data = vec![0u8; 34 * 100]; // 100 blocks
    let result = dequantize_q8_0_simd(&data).expect("valid");
    assert_eq!(result.len(), 32 * 100);
}

#[test]
fn test_dequantize_q8_0_simd_error_invalid_length() {
    let data = vec![0u8; 33]; // Not multiple of 34
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_err());
}

// ============================================================================
// Softmax Additional Edge Cases
// ============================================================================

#[test]
fn test_softmax_simd_two_elements() {
    let mut x = vec![0.0f32, 0.0];
    softmax_simd(&mut x);

    // Equal inputs -> equal outputs (each 0.5)
    assert!((x[0] - 0.5).abs() < 1e-6);
    assert!((x[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_negative_values() {
    let mut x = vec![-1.0f32, -2.0, -3.0];
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify monotonicity (less negative = higher probability)
    assert!(x[0] > x[1]);
    assert!(x[1] > x[2]);
}

#[test]
fn test_softmax_simd_very_large_difference() {
    let mut x = vec![100.0f32, 0.0, -100.0];
    softmax_simd(&mut x);

    // First element should be ~1.0, others ~0.0
    assert!(x[0] > 0.99);
    assert!(x[1] < 0.01);
    assert!(x[2] < 1e-30);
}

// ============================================================================
// SwiGLU Additional Edge Cases
// ============================================================================

#[test]
fn test_fused_swiglu_simd_empty() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];

    fused_swiglu_simd(&mut gate, &up);
    assert!(gate.is_empty());
}

#[test]
fn test_fused_swiglu_simd_single_element() {
    let mut gate = vec![1.0f32];
    let up = vec![2.0f32];

    fused_swiglu_simd(&mut gate, &up);

    // silu(1.0) = 1.0 * sigmoid(1.0) ~ 0.731
    // result = silu(1.0) * 2.0 ~ 1.462
    assert!((gate[0] - 1.462).abs() < 0.1);
}

#[test]
fn test_fused_swiglu_simd_all_zeros() {
    let mut gate = vec![0.0f32; 16];
    let up = vec![1.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0) = 0 * 0.5 = 0
    for v in &gate {
        assert!(v.abs() < 1e-6);
    }
}

// ============================================================================
// RoPE Additional Edge Cases
// ============================================================================

#[test]
fn test_apply_rope_rotation_simd_empty() {
    let mut x1: Vec<f32> = vec![];
    let mut x2: Vec<f32> = vec![];
    let cos_vals: Vec<f32> = vec![];
    let sin_vals: Vec<f32> = vec![];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);
    // Should not panic
}

#[test]
fn test_apply_rope_rotation_simd_single() {
    let mut x1 = vec![1.0f32];
    let mut x2 = vec![0.0f32];
    let cos_vals = vec![0.5f32];
    let sin_vals = vec![0.866f32]; // ~sin(60 degrees)

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // x1' = 1.0 * 0.5 - 0.0 * 0.866 = 0.5
    // x2' = 1.0 * 0.866 + 0.0 * 0.5 = 0.866
    assert!((x1[0] - 0.5).abs() < 0.01);
    assert!((x2[0] - 0.866).abs() < 0.01);
}

// ============================================================================
// Q4_K x Q8_K Parallel Matvec Additional Tests
// ============================================================================

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_large() {
    // Test parallel path with large output
    let weight_data = vec![0u8; 144 * 500];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut output = vec![0.0f32; 500];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        256,
        500,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_error_weight_too_small() {
    let weight_data = vec![0u8; 100]; // Too small
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];
    let mut output = vec![0.0f32; 4];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weight_data,
        &q8k_scales,
        &q8k_quants,
        256,
        4,
        &mut output,
    );
    assert!(result.is_err());
}

// ============================================================================
// Q4_0 Parallel Matvec Large Tests
// ============================================================================

// ============================================================================
// DequantStats Fields Test
// ============================================================================

#[test]
fn test_dequant_stats_with_values() {
    let stats = DequantStats {
        blocks_processed: 1000,
        bytes_processed: 36000,
        simd_backend: SimdBackend::Avx2,
    };
    assert_eq!(stats.blocks_processed, 1000);
    assert_eq!(stats.bytes_processed, 36000);
    assert_eq!(stats.simd_backend, SimdBackend::Avx2);

    // Test Clone trait by using cloned value alongside original
    #[allow(clippy::redundant_clone)]
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, stats.blocks_processed);
}

#[test]
fn test_simd_backend_all_variants() {
    // Test Debug trait
    assert!(format!("{:?}", SimdBackend::Avx2).contains("Avx2"));
    assert!(format!("{:?}", SimdBackend::Sse2).contains("Sse2"));
    assert!(format!("{:?}", SimdBackend::Neon).contains("Neon"));
    assert!(format!("{:?}", SimdBackend::Scalar).contains("Scalar"));

    // Test Copy trait
    let backend = SimdBackend::Neon;
    let copied = backend;
    assert_eq!(backend, copied);
}

// ============================================================================
// Q4_0 x Q8_0 Additional Tests
// ============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_medium() {
    // Test with medium size (between sequential and parallel thresholds)
    let weight_data = vec![0u8; 18 * 500];
    let activations = vec![1.0f32; 32];

    let result =
        fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 500).expect("valid");
    assert_eq!(result.len(), 500);
}

// ============================================================================
// Q8_0 x Q8_0 Additional Tests
// ============================================================================

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_large() {
    let weight_data = vec![0u8; 34 * 500];
    let activations = vec![1.0f32; 32];

    let result =
        fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 500).expect("valid");
    assert_eq!(result.len(), 500);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_large() {
    let weight_data = vec![0u8; 34 * 500];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 500];

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 500, &mut output);
    assert!(result.is_ok());
}

// ============================================================================
// Fused Dot Products with Non-zero Data
// ============================================================================

#[test]
fn test_fused_q4k_dot_nonzero_weights() {
    let mut q4k_data = vec![0u8; 144];

    // d = 0.5
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // dmin = 0.1
    q4k_data[2..4].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());

    // Set some non-zero scales and quants
    q4k_data[4..16]
        .iter_mut()
        .enumerate()
        .for_each(|(i, byte)| {
            *byte = ((i + 1) * 3) as u8;
        });
    q4k_data[16..144]
        .iter_mut()
        .enumerate()
        .for_each(|(i, byte)| {
            *byte = ((i * 7) % 256) as u8;
        });

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();

    let result = fused_q4k_dot(&q4k_data, &activations).expect("valid");
    assert!(result.is_finite());
    // With non-zero weights and activations, result should be non-zero
    assert!(result.abs() > 1e-6);
}

#[test]
fn test_fused_q5k_dot_nonzero_weights() {
    let mut q5k_data = vec![0u8; 176];

    // d = 0.5
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // dmin = 0.1
    q5k_data[2..4].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());

    // Set some non-zero values
    q5k_data[4..].iter_mut().enumerate().for_each(|(i, byte)| {
        *byte = ((i * 11) % 256) as u8;
    });

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();

    let result = fused_q5k_dot(&q5k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

#[test]
fn test_fused_q6k_dot_nonzero_weights() {
    let mut q6k_data = vec![0u8; 210];

    // Set varied ql values
    q6k_data[0..128]
        .iter_mut()
        .enumerate()
        .for_each(|(i, byte)| {
            *byte = ((i * 5) % 256) as u8;
        });

    // Set varied qh values
    q6k_data[128..192]
        .iter_mut()
        .enumerate()
        .for_each(|(i, byte)| {
            *byte = ((i * 3) % 256) as u8;
        });

    // Set positive scales
    q6k_data[192..208]
        .iter_mut()
        .enumerate()
        .for_each(|(i, byte)| {
            *byte = (i + 1) as u8;
        });

    // d = 0.5
    q6k_data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();

    let result = fused_q6k_dot(&q6k_data, &activations).expect("valid");
    assert!(result.is_finite());
}

// ============================================================================
// Parallel Dequantization Large Tests
// ============================================================================

#[test]
fn test_dequantize_q4_k_parallel_large() {
    let data = vec![0u8; 144 * 100]; // 100 super-blocks
    let result = dequantize_q4_k_parallel(&data).expect("valid");
    assert_eq!(result.len(), 256 * 100);
}

#[test]
fn test_dequantize_q8_0_parallel_large() {
    let data = vec![0u8; 34 * 100]; // 100 blocks
    let result = dequantize_q8_0_parallel(&data).expect("valid");
    assert_eq!(result.len(), 32 * 100);
}

// ============================================================================
// quantize_rmsnorm_q8_0 Edge Cases
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_all_zeros() {
    let input = vec![0.0f32; 64];
    let norm_weight = vec![1.0f32; 64];

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
    for q in &quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_varied_weights() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let norm_weight: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) / 64.0).collect();

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-6);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

// ============================================================================
// F16 Conversion Tests
// ============================================================================

#[test]
fn test_f16_to_f32_special_values() {
    // Test infinity
    let pos_inf = half::f16::INFINITY.to_bits();
    let result = f16_to_f32(pos_inf);
    assert!(result.is_infinite() && result > 0.0);

    // Test negative infinity
    let neg_inf = half::f16::NEG_INFINITY.to_bits();
    let result = f16_to_f32(neg_inf);
    assert!(result.is_infinite() && result < 0.0);

    // Test NaN
    let nan = half::f16::NAN.to_bits();
    let result = f16_to_f32(nan);
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_subnormal() {
    // Test smallest positive subnormal
    let subnormal = half::f16::from_f32(5.96e-8).to_bits(); // Smallest f16 subnormal
    let result = f16_to_f32(subnormal);
    assert!((0.0..1e-4).contains(&result));
}

#[test]
fn test_dequantize_f16_large() {
    let mut data = Vec::new();
    (0..1000).for_each(|i| {
        data.extend_from_slice(&half::f16::from_f32(i as f32 * 0.1).to_le_bytes());
    });

    let result = dequantize_f16(&data).expect("valid");
    assert_eq!(result.len(), 1000);
}
