//! Part 12: Deep Coverage Tests for activation.rs
//!
//! Targets uncovered lines in `src/quantize/activation.rs` including:
//! - Scalar fallback paths for all activation functions
//! - Edge cases: zeros, near-zeros, extreme values, boundary sizes
//! - Partial block handling in quantization
//! - `quantize_rmsnorm_q8_0_into` edge cases
//! - AVX2 remainder loop paths

use crate::quantize::activation::{
    fused_swiglu_scalar, quantize_rmsnorm_q8_0_scalar, softmax_scalar,
};
use crate::quantize::{
    fused_rmsnorm_ffn_up_gate, fused_rmsnorm_q4_0_matmul, fused_swiglu_simd,
    quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, softmax_simd,
};

// ============================================================================
// quantize_rmsnorm_q8_0_scalar: Direct Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_basic() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // One block of 32
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Scale should be positive
    assert!(scales[0] > 0.0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_partial_block() {
    // 40 elements = 1 full block + 8 partial
    let input: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 40];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Should have 2 blocks (ceil(40/32) = 2)
    assert_eq!(scales.len(), 2);
    // Quants padded to 64 (2 * 32)
    assert_eq!(quants.len(), 64);
    // Last 24 values should be padding zeros
    for q in &quants[40..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_near_zero_max() {
    // All values extremely small -> triggers near-zero handling path
    let input = vec![1e-12f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Scale should be positive and finite (either fallback or computed)
    assert!(scales[0] > 0.0, "Scale should be positive");
    assert!(scales[0].is_finite(), "Scale should be finite");
    // All quantized values are i8, so range is guaranteed
    assert!(!quants.is_empty(), "Should have quantized values");
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_zeros() {
    let input = vec![0.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // inv_rms will be very large due to eps, but zero * inv_rms = 0
    // So max_abs = 0, triggering fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    for q in quants {
        assert_eq!(q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_extreme_values() {
    let input: Vec<f32> = vec![127.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // After RMSNorm, all values are 1.0 (same input, normalized)
    // Scale = 1.0 / 127.0
    assert!(scales[0] > 0.0);
    // All quants should be equal
    let first = quants[0];
    for q in &quants[1..32] {
        assert_eq!(*q, first);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_negative_values() {
    let input: Vec<f32> = (0..32).map(|i| -((i + 1) as f32)).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    // Most quantized values should be negative
    let negative_count = quants[..32].iter().filter(|&&q| q < 0).count();
    assert!(negative_count > 16);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_weight_variation() {
    let input = vec![1.0f32; 32];
    let norm_weight: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let eps = 1e-5;

    let (_scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Quantized values should vary due to weights
    assert_ne!(quants[0], quants[31]);
}

// ============================================================================
// fused_swiglu_scalar: Direct Tests
// ============================================================================

#[test]
fn test_fused_swiglu_scalar_basic() {
    let mut gate = vec![0.0, 1.0, -1.0, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];

    fused_swiglu_scalar(&mut gate, &up);

    // silu(0) * 1 = 0
    assert!((gate[0] - 0.0).abs() < 1e-5);
    // silu(1) * 1 = 1 / (1 + exp(-1)) ~= 0.7311
    assert!((gate[1] - 0.7310586).abs() < 1e-4);
    // silu(-1) * 1 = -1 / (1 + exp(1)) ~= -0.2689
    assert!((gate[2] - (-0.2689414)).abs() < 1e-4);
}

#[test]
fn test_fused_swiglu_scalar_zeros() {
    let mut gate = vec![0.0; 8];
    let up = vec![1.0; 8];

    fused_swiglu_scalar(&mut gate, &up);

    for g in gate {
        assert_eq!(g, 0.0);
    }
}

#[test]
fn test_fused_swiglu_scalar_up_zeros() {
    let mut gate = vec![1.0, 2.0, 3.0, 4.0];
    let up = vec![0.0; 4];

    fused_swiglu_scalar(&mut gate, &up);

    // silu(x) * 0 = 0
    for g in gate {
        assert_eq!(g, 0.0);
    }
}

#[test]
fn test_fused_swiglu_scalar_large_positive() {
    let mut gate = vec![10.0, 20.0, 50.0];
    let up = vec![1.0; 3];

    fused_swiglu_scalar(&mut gate, &up);

    // For large positive x, silu(x) ~= x (sigmoid(x) ~= 1)
    assert!((gate[0] - 10.0).abs() < 0.01);
    assert!((gate[1] - 20.0).abs() < 0.01);
    assert!((gate[2] - 50.0).abs() < 0.01);
}

#[test]
fn test_fused_swiglu_scalar_large_negative() {
    let mut gate = vec![-10.0, -20.0, -50.0];
    let up = vec![1.0; 3];

    fused_swiglu_scalar(&mut gate, &up);

    // For large negative x, silu(x) ~= 0 (sigmoid(x) ~= 0)
    assert!(gate[0].abs() < 0.001);
    assert!(gate[1].abs() < 1e-6);
    assert!(gate[2].abs() < 1e-10);
}

#[test]
fn test_fused_swiglu_scalar_empty() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];

    fused_swiglu_scalar(&mut gate, &up);

    assert!(gate.is_empty());
}

#[test]
fn test_fused_swiglu_scalar_single() {
    let mut gate = vec![0.5];
    let up = vec![2.0];

    fused_swiglu_scalar(&mut gate, &up);

    // silu(0.5) = 0.5 / (1 + exp(-0.5)) ~= 0.3112
    // silu(0.5) * 2.0 ~= 0.6225
    assert!((gate[0] - 0.6224593).abs() < 1e-4);
}

// ============================================================================
// softmax_scalar: Direct Tests
// ============================================================================

#[test]
fn test_softmax_scalar_basic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_scalar(&mut x);

    // Sum should be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Values should be ordered
    assert!(x[3] > x[2]);
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_scalar_uniform() {
    let mut x = vec![5.0; 8];
    softmax_scalar(&mut x);

    // All values should be equal = 1/8
    for v in &x {
        assert!((v - 0.125).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_scalar_large_values() {
    let mut x = vec![1000.0, 1001.0, 1002.0];
    softmax_scalar(&mut x);

    // Should be numerically stable
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All values should be valid (no NaN/Inf)
    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_softmax_scalar_very_negative() {
    let mut x = vec![-1000.0, -1001.0, -1002.0];
    softmax_scalar(&mut x);

    // Should still sum to 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_scalar_single() {
    let mut x = vec![42.0];
    softmax_scalar(&mut x);

    // Single element softmax = 1.0
    assert!((x[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_scalar_dominant() {
    let mut x = vec![0.0, 0.0, 100.0, 0.0];
    softmax_scalar(&mut x);

    // The large value should dominate
    assert!(x[2] > 0.99);
    assert!(x[0] < 0.01);
    assert!(x[1] < 0.01);
    assert!(x[3] < 0.01);
}

// ============================================================================
// quantize_rmsnorm_q8_0: SIMD/Scalar dispatch tests (sizes that hit remainders)
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_size_7_remainder() {
    // Size 7 hits remainder loop (not divisible by 8)
    let input: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 7];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // 1 block (ceil(7/32) = 1)
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32); // Padded to 32
    // Padding should be zeros
    for q in &quants[7..32] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_33_remainder() {
    // Size 33 = 1 full block + 1 element
    let input: Vec<f32> = (0..33).map(|i| i as f32 * 0.05).collect();
    let norm_weight = vec![1.0f32; 33];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_63() {
    // 63 = tests the AVX2 loop (7 * 8 = 56 processed) + 7 remainder
    let input: Vec<f32> = (0..63).map(|i| (i as f32 - 31.5) * 0.1).collect();
    let norm_weight = vec![1.0f32; 63];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2); // ceil(63/32)
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_256() {
    // Large size for AVX2 main loop testing
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 8); // 256/32
    assert_eq!(quants.len(), 256);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_260() {
    // 260 = 8 full blocks + 4 elements (tests partial block in AVX2 path)
    let input: Vec<f32> = (0..260).map(|i| i as f32 * 0.001).collect();
    let norm_weight = vec![1.0f32; 260];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 9); // ceil(260/32)
    assert_eq!(quants.len(), 288); // 9 * 32
}

// ============================================================================
// quantize_rmsnorm_q8_0_into: Edge cases
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert!(scales[0] > 0.0);
    // Values should be non-zero (except possibly zeros from input)
    let nonzero = quants.iter().filter(|&&q| q != 0).count();
    assert!(nonzero > 0);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_partial_block() {
    // 50 elements = 1 full block + 18 partial
    let input: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 50];
    let eps = 1e-5;

    // Need 2 scales and 64 quants (2 blocks)
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
    // Padding should be zeros (indices 50..64)
    for q in &quants[50..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_near_zero() {
    let input = vec![1e-12f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // With near-zero input, scale should be small but non-zero (fallback or computed)
    assert!(scales[0] > 0.0, "Scale should be positive");
    assert!(scales[0].is_finite(), "Scale should be finite");
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_matches_allocating_version() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    // Allocating version
    let (expected_scales, expected_quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Into version
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];
    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // Results should match
    assert_eq!(scales, expected_scales);
    assert_eq!(quants, expected_quants);
}

include!("fused_swiglu_02.rs");
include!("quantize_activations_rmsnorm.rs");
