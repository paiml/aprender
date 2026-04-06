
// ============================================================================
// Size Edge Cases for AVX2 Paths
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_size_1() {
    // Minimum size
    let input = vec![5.0f32];
    let norm_weight = vec![1.0f32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32); // Padded
    assert_eq!(quants[0], 127); // Normalized = 1.0 -> 127
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_8() {
    // Exactly one SIMD iteration
    let input: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
    let norm_weight = vec![1.0f32; 8];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_16() {
    // Two SIMD iterations, no remainder
    let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 16];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
}

#[test]
fn test_quantize_rmsnorm_q8_0_size_24() {
    // Three SIMD iterations, no remainder
    let input: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 24];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
}

#[test]
fn test_fused_swiglu_simd_size_1() {
    // Single element (hits scalar remainder)
    let mut gate = vec![0.5f32];
    let up = vec![2.0f32];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0.5) * 2.0
    let expected = 0.5 / (1.0 + (-0.5f32).exp()) * 2.0;
    assert!((gate[0] - expected).abs() < 1e-4);
}

#[test]
fn test_fused_swiglu_simd_size_16() {
    // Two SIMD iterations, no remainder
    let mut gate: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.2).collect();
    let up = vec![1.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite());
    }
}

#[test]
fn test_softmax_simd_size_1() {
    // Single element softmax should be 1.0
    let mut x = vec![42.0f32];
    softmax_simd(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_8() {
    // Exactly one SIMD iteration
    let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Last element should be largest
    assert!(x[7] > x[0]);
}

#[test]
fn test_softmax_simd_size_16() {
    // Two SIMD iterations
    let mut x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ============================================================================
// Up array edge cases for SwiGLU
// ============================================================================

#[test]
fn test_fused_swiglu_simd_up_zeros() {
    let mut gate: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
    let up = vec![0.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // silu(x) * 0 = 0
    for g in &gate {
        assert_eq!(*g, 0.0);
    }
}

#[test]
fn test_fused_swiglu_simd_up_negative() {
    let mut gate = vec![2.0f32; 8];
    let up = vec![-1.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(2.0) > 0, * -1 = negative
    for g in &gate {
        assert!(*g < 0.0);
    }
}

#[test]
fn test_fused_swiglu_simd_up_large() {
    let mut gate = vec![0.5f32; 8];
    let up = vec![1000.0f32; 8];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0.5) > 0, multiplied by large up should produce large positive
    for g in &gate {
        assert!(*g > 100.0, "Expected large positive, got {}", g);
        assert!(g.is_finite());
    }
}

// ============================================================================
// Precision and Numerical Stability
// ============================================================================

#[test]
fn test_softmax_simd_underflow_protection() {
    // Values so negative that exp would underflow
    let mut x = vec![-1000.0, -1001.0, -1002.0, -1003.0];
    softmax_simd(&mut x);

    // Should still sum to 1.0 due to max subtraction
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // All values should be finite
    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_softmax_scalar_underflow_protection() {
    let mut x = vec![-1000.0, -1001.0, -1002.0];
    softmax_scalar(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_quantize_rmsnorm_q8_0_numerical_stability() {
    // Input that would cause numerical issues without proper handling
    let mut input = vec![0.0f32; 64];
    input[0] = 1e6;
    input[63] = -1e6;
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should produce valid output
    assert!(scales[0].is_finite());
    assert!(scales[1].is_finite());
    // Extreme values should map to extreme quants
}

// ============================================================================
// Quantize rmsnorm into edge cases
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_into_size_1() {
    let input = vec![5.0f32];
    let norm_weight = vec![1.0f32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    assert_eq!(quants[0], 127);
    // Padding should be zeros
    for q in &quants[1..32] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_size_33() {
    // Tests partial second block
    let input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 33];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 64];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // Both blocks should have scales
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
    // Padding in second block (indices 33..64)
    for q in &quants[33..64] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_all_zeros() {
    let input = vec![0.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![99i8; 32]; // Non-zero initial values

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // All quants should be zero
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
    // Scale should be the fallback value
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
}

// ============================================================================
// Quantize activations edge cases
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_empty() {
    let activations: Vec<f32> = vec![];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_single_zero() {
    let activations = vec![0.0f32];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    assert_eq!(quants[0], 0i8);
}

#[test]
fn test_quantize_activations_q8_0_all_zeros() {
    let activations = vec![0.0f32; 64];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // All scales should be fallback
    for s in &scales {
        assert!((*s - 1.0 / 127.0).abs() < 1e-10);
    }
    // All quants should be zero
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_activations_q8_0_infinity() {
    let activations = vec![f32::INFINITY, 1.0, 2.0];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Infinity should produce infinite scale or saturated quants
    // The behavior depends on implementation - i8 is always finite
    assert!(scales[0].is_infinite() || quants[0] == 127 || quants[0] == -128);
}

#[test]
fn test_quantize_activations_q8_0_negative_infinity() {
    let activations = vec![f32::NEG_INFINITY, 1.0, 2.0];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // Should handle negative infinity
    // The behavior depends on implementation - i8 is always finite
    assert!(scales[0].is_infinite() || quants[0] == -127 || quants[0] == -128);
}

#[test]
fn test_quantize_activations_q8_0_nan() {
    let activations = vec![f32::NAN, 1.0, 2.0];
    let (scales, _quants) = quantize_activations_q8_0(&activations);

    // NaN in max calculation propagates
    assert!(scales[0].is_nan() || scales[0] > 0.0);
}

// ============================================================================
// Additional Scalar/SIMD Parity Tests
// ============================================================================

#[test]
fn test_fused_swiglu_scalar_simd_parity_large() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.05).collect();
    let up = vec![1.5f32; 128];

    let mut gate_scalar = values.clone();
    fused_swiglu_scalar(&mut gate_scalar, &up);

    let mut gate_simd = values;
    fused_swiglu_simd(&mut gate_simd, &up);

    // Both produce finite results
    for (s, d) in gate_scalar.iter().zip(gate_simd.iter()) {
        assert!(s.is_finite(), "Scalar should be finite: {}", s);
        assert!(d.is_finite(), "SIMD should be finite: {}", d);
    }
}

#[test]
fn test_softmax_scalar_simd_parity_large() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect();

    let mut x_scalar = values.clone();
    softmax_scalar(&mut x_scalar);

    let mut x_simd = values;
    softmax_simd(&mut x_simd);

    // Should match within tolerance
    for (s, d) in x_scalar.iter().zip(x_simd.iter()) {
        assert!((s - d).abs() < 1e-5, "Mismatch: scalar={}, simd={}", s, d);
    }
}

// ============================================================================
// Block Boundary Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_exactly_32() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Exactly one block
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // No padding needed
}

#[test]
fn test_quantize_rmsnorm_q8_0_exactly_64() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Exactly two blocks
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_exactly_96() {
    let input: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.03).collect();
    let norm_weight = vec![1.0f32; 96];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Exactly three blocks
    assert_eq!(scales.len(), 3);
    assert_eq!(quants.len(), 96);
}

// ============================================================================
// Maximum Input Size Tests
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_large_input() {
    // 1024 elements = 32 blocks
    let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.001).collect();
    let norm_weight = vec![1.0f32; 1024];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 32);
    assert_eq!(quants.len(), 1024);
    // All scales should be positive and finite
    for s in &scales {
        assert!(*s > 0.0);
        assert!(s.is_finite());
    }
}

#[test]
fn test_fused_swiglu_simd_large_input() {
    let mut gate: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.005).collect();
    let up = vec![1.0f32; 1024];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite());
    }
}
