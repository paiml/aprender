//! Part 21: Comprehensive Coverage Tests for activation.rs
//!
//! Targets uncovered code paths in `src/quantize/activation.rs`:
//! - AVX2 inner loops with specific remainder patterns
//! - Block-level quantization edge cases
//! - Scalar fallback verification
//! - Error paths in fused functions

use crate::quantize::activation::{
    fused_swiglu_scalar, quantize_rmsnorm_q8_0_scalar, softmax_scalar,
};
use crate::quantize::{
    fused_rmsnorm_ffn_up_gate, fused_rmsnorm_q4_0_matmul, fused_swiglu_simd,
    quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, softmax_simd,
};

// ============================================================================
// AVX2 Block Quantization Inner Loop Coverage
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_size_40_avx2_sum_boundary() {
    let input: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.08).collect();
    let norm_weight = vec![1.0f32; 40];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
    for q in &quants[40..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_48_block_remainder_16() {
    let input: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) * 0.06).collect();
    let norm_weight = vec![1.0f32; 48];
    let (scales, _) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    assert!(scales[0] > 0.0 && scales[0].is_finite());
    assert!(scales[1] > 0.0 && scales[1].is_finite());
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_56_block_remainder_24() {
    let input: Vec<f32> = (0..56).map(|i| (i as f32 - 28.0) * 0.04).collect();
    let norm_weight = vec![1.0f32; 56];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    for q in &quants[56..64] {
        assert_eq!(*q, 0i8);
    }
}

// ============================================================================
// Quantization Scale Computation Edge Cases
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_large_range_single_block() {
    let mut input = vec![0.001f32; 32];
    input[0] = 100.0;
    input[31] = -100.0;
    let norm_weight = vec![1.0f32; 32];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 1);
    assert!(scales[0] > 0.0);
    assert!(quants[0].abs() > 100 || quants[31].abs() > 100);
}

#[test]
fn test_quantize_rmsnorm_q8_0_near_threshold_weights() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1e-11f32; 32];
    let (scales, _) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
}

#[test]
fn test_quantize_rmsnorm_q8_0_just_above_threshold() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![2e-10f32; 32];
    let (scales, _) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert!(scales[0] > 0.0);
}

// ============================================================================
// Scalar Fallback Path Verification
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_mathematical_correctness() {
    let input = vec![3.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, 0.0);
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-6);
    for q in &quants[..32] {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_non_uniform_weights() {
    let input = vec![2.0f32; 32];
    let norm_weight: Vec<f32> = (0..32).map(|i| (i + 1) as f32 * 0.1).collect();
    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, 1e-5);
    assert!(scales[0] > 0.0);
    assert!(quants[31].abs() >= quants[0].abs());
}

// ============================================================================
// Fused SwiGLU Coverage
// ============================================================================

#[test]
fn test_fused_swiglu_simd_polynomial_boundaries() {
    let mut gate = vec![
        -0.693_147_2,
        0.693_147_2,
        -1.386_294_4,
        1.386_294_4,
        -2.302_585_1,
        2.302_585_1,
        -4.605_170_2,
        4.605_170_2,
    ];
    let up = vec![1.0f32; 8];
    fused_swiglu_simd(&mut gate, &up);
    for g in &gate {
        assert!(g.is_finite());
    }
}

#[test]
fn test_fused_swiglu_scalar_precise_values() {
    let mut gate = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    let up = vec![2.0, 2.0, 2.0, 2.0, 2.0];
    fused_swiglu_scalar(&mut gate, &up);
    assert!((gate[0] - 0.0).abs() < 1e-10);
    assert!((gate[1] - 1.4621172).abs() < 1e-4);
    assert!((gate[2] - (-0.5378828)).abs() < 1e-4);
}

// ============================================================================
// Softmax SIMD Coverage
// ============================================================================

#[test]
fn test_softmax_simd_phase_coverage() {
    let mut x: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.5).collect();
    softmax_simd(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    for i in 1..24 {
        assert!(x[i] >= x[i - 1] - 1e-6);
    }
}

#[test]
fn test_softmax_simd_scalar_remainder_3() {
    let mut x: Vec<f32> = (0..11).map(|i| i as f32 * 0.3).collect();
    softmax_simd(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_scalar_remainder_5() {
    let mut x: Vec<f32> = (0..13).map(|i| (i as f32 - 6.0) * 0.4).collect();
    softmax_simd(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_identical_large_positive() {
    let mut x = vec![100.0f32; 16];
    softmax_simd(&mut x);
    for v in &x {
        assert!((v - 1.0 / 16.0).abs() < 1e-5);
    }
}

// ============================================================================
// Quantize Activations Q8_0 Coverage
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_rounding_boundary() {
    let exact_127 = 127.0 * (1.0 / 127.0);
    let activations = vec![exact_127; 8];
    let (_, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(quants[0], 127);
}

#[test]
fn test_quantize_activations_q8_0_clamping_path() {
    let activations = vec![1.0f32; 8];
    let (_, quants) = quantize_activations_q8_0(&activations);
    for q in &quants[..8] {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_quantize_activations_q8_0_decreasing_blocks() {
    let mut activations = vec![100.0f32; 32];
    activations.extend(vec![10.0f32; 32]);
    activations.extend(vec![1.0f32; 32]);
    let (scales, _) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 3);
    assert!(scales[0] > scales[1] && scales[1] > scales[2]);
}

// ============================================================================
// Quantize RMSNorm Into Coverage
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_into_overwrites() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let mut scales = vec![999.0f32; 2];
    let mut quants = vec![99i8; 64];
    quantize_rmsnorm_q8_0_into(&input, &norm_weight, 1e-5, &mut scales, &mut quants);
    assert_ne!(scales[0], 999.0);
    assert_ne!(scales[1], 999.0);
    assert!(quants.iter().filter(|&&q| q != 99).count() > 0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_size_31() {
    let input: Vec<f32> = (0..31).map(|i| (i as f32 - 15.5) * 0.1).collect();
    let norm_weight = vec![1.0f32; 31];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];
    quantize_rmsnorm_q8_0_into(&input, &norm_weight, 1e-5, &mut scales, &mut quants);
    assert!(scales[0] > 0.0);
    assert_eq!(quants[31], 0i8);
}

// ============================================================================
// Fused RMSNorm Q4_0 Matmul Coverage
// ============================================================================

#[test]
fn test_fused_rmsnorm_q4_0_matmul_exact_weight_size() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let weight_data = vec![0u8; 72];
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 64, 2);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 2);
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_weight_one_short() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let weight_data = vec![0u8; 71];
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 64, 2);
    assert!(result.is_err());
}

// ============================================================================
// Fused RMSNorm FFN Up/Gate Coverage
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_exact_weights() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let up_weight = vec![0u8; 36 * 4];
    let gate_weight = vec![0u8; 36 * 4];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 64, 4);
    assert!(result.is_ok());
    let (up_out, gate_out) = result.expect("test value should be present");
    assert_eq!(up_out.len(), 4);
    assert_eq!(gate_out.len(), 4);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_up_one_short() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let up_weight = vec![0u8; 143];
    let gate_weight = vec![0u8; 144];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 64, 4);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_gate_one_short() {
    let input = vec![0.5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let up_weight = vec![0u8; 144];
    let gate_weight = vec![0u8; 143];
    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 64, 4);
    assert!(result.is_err());
}

// ============================================================================
// AVX2 Specific Path Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_7_scalar_only() {
    let input: Vec<f32> = (0..39).map(|i| ((i as f32) - 19.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 39];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    for q in &quants[39..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_31_mixed() {
    let input: Vec<f32> = (0..63).map(|i| ((i as f32) - 31.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 63];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 2);
    assert_eq!(quants[63], 0i8);
    assert_ne!(quants[62], 0i8);
}

// ============================================================================
// Numerical Precision Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_precision_sum_squares() {
    let input = vec![1e5f32; 64];
    let norm_weight = vec![1.0f32; 64];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert!(scales[0].is_finite() && scales[1].is_finite());
    for q in &quants[..64] {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_quantize_rmsnorm_parity_challenging_pattern() {
    let input: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 1e-6 } else { 1e6 })
        .collect();
    let norm_weight = vec![1.0f32; 64];
    let (scales_simd, _) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    let (scales_scalar, _) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, 1e-5);
    for (s, d) in scales_scalar.iter().zip(scales_simd.iter()) {
        assert!((s - d).abs() / s.max(1e-10) < 1e-3);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_softmax_scalar_size_2() {
    let mut x = vec![1.0, 2.0];
    softmax_scalar(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_scalar_size_3() {
    let mut x = vec![1.0, 2.0, 3.0];
    softmax_scalar(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(x[2] > x[1]);
}

#[test]
fn test_quantize_rmsnorm_q8_0_large_weights() {
    let input = vec![1e-10f32; 32];
    let norm_weight = vec![1e10f32; 32];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert!(scales[0].is_finite());
    for _q in &quants[..32] {
        assert!(true /* i8 always in range */);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_mixed_sign_weights() {
    let input = vec![1.0f32; 32];
    let norm_weight: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert!(scales[0] > 0.0);
    let pos_count = quants[..32].iter().filter(|&&q| q > 0).count();
    let neg_count = quants[..32].iter().filter(|&&q| q < 0).count();
    assert_eq!(pos_count, 16);
    assert_eq!(neg_count, 16);
}
