
// ============================================================================
// fused_swiglu_simd: Size-based coverage
// ============================================================================

#[test]
fn test_fused_swiglu_simd_size_7() {
    // Not divisible by 8 - hits remainder path
    let mut gate: Vec<f32> = (0..7).map(|i| i as f32 * 0.5).collect();
    let up = vec![1.0f32; 7];

    fused_swiglu_simd(&mut gate, &up);

    // silu(0) = 0
    assert!((gate[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_fused_swiglu_simd_size_15() {
    // 8 + 7 remainder
    let mut gate: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) * 0.2).collect();
    let up = vec![1.0f32; 15];

    fused_swiglu_simd(&mut gate, &up);

    // All should be finite
    for g in &gate {
        assert!(g.is_finite());
    }
}

#[test]
fn test_fused_swiglu_simd_size_64() {
    // Perfectly divisible by 8
    let mut gate: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let up = vec![1.0f32; 64];

    fused_swiglu_simd(&mut gate, &up);

    // Should have proper sigmoid-like distribution
    assert!(gate.iter().all(|g| g.is_finite()));
}

// ============================================================================
// softmax_simd: Size-based coverage
// ============================================================================

#[test]
fn test_softmax_simd_size_7() {
    // Not divisible by 8
    let mut x: Vec<f32> = (0..7).map(|i| i as f32).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_15() {
    // 8 + 7 remainder
    let mut x: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_64() {
    let mut x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_size_100() {
    // 12 * 8 + 4 remainder
    let mut x: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ============================================================================
// quantize_activations_q8_0: Edge cases
// ============================================================================

#[test]
fn test_quantize_activations_q8_0_size_1() {
    let activations = vec![42.0f32];
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // 1 block
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32); // Padded

    // First element should be 127 (max value maps to max quant)
    assert_eq!(quants[0], 127);
    // Padding should be zeros
    for q in &quants[1..32] {
        assert_eq!(*q, 0i8);
    }
}

#[test]
fn test_quantize_activations_q8_0_symmetric() {
    let activations = vec![-10.0, 0.0, 10.0];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // Scale = 10.0 / 127.0
    // quants: -127, 0, 127
    assert_eq!(quants[0], -127);
    assert_eq!(quants[1], 0);
    assert_eq!(quants[2], 127);
}

#[test]
fn test_quantize_activations_q8_0_near_zero_max() {
    let activations = vec![1e-12f32; 10];
    let (scales, _quants) = quantize_activations_q8_0(&activations);

    // Fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
}

#[test]
fn test_quantize_activations_q8_0_exact_block() {
    let activations: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Max value is 31, so quants[31] = 127
    assert_eq!(quants[31], 127);
}

#[test]
fn test_quantize_activations_q8_0_multi_block() {
    let activations: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.5).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);

    // 4 blocks (ceil(100/32) = 4)
    assert_eq!(scales.len(), 4);
    assert_eq!(quants.len(), 128);
}

// ============================================================================
// fused_rmsnorm_q4_0_matmul: Error paths and edge cases
// ============================================================================

#[test]
fn test_fused_rmsnorm_q4_0_matmul_input_dim_mismatch() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18]; // 1 block

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_weight_too_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 10]; // Too small (need 18 bytes for 1 block)

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_zero_out_dim() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let weight_data = vec![0u8; 18];

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 32, 0);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present").is_empty());
}

// ============================================================================
// fused_rmsnorm_ffn_up_gate: Error paths and edge cases
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_input_dim_mismatch() {
    let input = vec![1.0f32; 16]; // Wrong size
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_up_weight_too_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 10]; // Too small
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_gate_weight_too_small() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 10]; // Too small

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_zero_out_dim() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let up_weight = vec![0u8; 18];
    let gate_weight = vec![0u8; 18];

    let result =
        fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight, 32, 0);
    assert!(result.is_ok());
    let (up, gate) = result.expect("test value should be present");
    assert!(up.is_empty());
    assert!(gate.is_empty());
}

// ============================================================================
// Scalar vs SIMD parity tests
// ============================================================================

#[test]
fn test_swiglu_scalar_produces_output() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect();
    let up = vec![1.0f32; 32];

    // Test that scalar swiglu produces valid output
    let mut gate_scalar = values.clone();
    fused_swiglu_scalar(&mut gate_scalar, &up);

    // All values should be finite
    for v in &gate_scalar {
        assert!(v.is_finite(), "SwiGLU output should be finite");
    }

    // Output should differ from input (transformation applied)
    let different = gate_scalar
        .iter()
        .zip(values.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(different, "SwiGLU should transform input");
}

#[test]
fn test_softmax_scalar_simd_parity() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

    // Scalar
    let mut x_scalar = values.clone();
    softmax_scalar(&mut x_scalar);

    // SIMD (via dispatch)
    let mut x_simd = values.clone();
    softmax_simd(&mut x_simd);

    // Should match within tolerance
    for (s, d) in x_scalar.iter().zip(x_simd.iter()) {
        assert!((s - d).abs() < 1e-5, "Mismatch: scalar={}, simd={}", s, d);
    }
}

#[test]
fn test_quantize_rmsnorm_scalar_simd_parity() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    // Scalar
    let (scales_scalar, quants_scalar) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // SIMD (via dispatch) - may use AVX2
    let (scales_simd, quants_simd) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Scales should be very close
    for (s, d) in scales_scalar.iter().zip(scales_simd.iter()) {
        assert!(
            (s - d).abs() < 1e-5,
            "Scale mismatch: scalar={}, simd={}",
            s,
            d
        );
    }

    // Quants may differ by 1 due to rounding
    for (s, d) in quants_scalar.iter().zip(quants_simd.iter()) {
        assert!(
            (*s as i32 - *d as i32).abs() <= 1,
            "Quant mismatch: scalar={}, simd={}",
            s,
            d
        );
    }
}

// ============================================================================
// Additional edge case tests for full coverage
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_size_1() {
    let input = vec![1.0f32];
    let norm_weight = vec![1.0f32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // First element should be 127 (normalized = 1.0)
    assert_eq!(quants[0], 127);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_size_31() {
    // Just under block boundary
    let input: Vec<f32> = (0..31).map(|i| i as f32 * 0.1).collect();
    let norm_weight = vec![1.0f32; 31];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // Padding at position 31
    assert_eq!(quants[31], 0i8);
}

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_size_64() {
    // Exactly 2 blocks
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64);
    // Both blocks should have valid scales
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

#[test]
fn test_fused_swiglu_scalar_symmetry() {
    // silu(-x) * 1 = -silu(x) for silu
    let mut gate_pos = vec![1.0, 2.0, 3.0];
    let mut gate_neg = vec![-1.0, -2.0, -3.0];
    let up = vec![1.0, 1.0, 1.0];

    fused_swiglu_scalar(&mut gate_pos, &up);
    fused_swiglu_scalar(&mut gate_neg, &up);

    // silu(-x) = -x * sigmoid(-x), silu(x) = x * sigmoid(x)
    // They are NOT equal in magnitude, but have opposite signs
    for i in 0..3 {
        assert!(gate_pos[i] > 0.0);
        assert!(gate_neg[i] < 0.0);
    }
}

#[test]
fn test_softmax_scalar_two_elements() {
    let mut x = vec![0.0, 0.0];
    softmax_scalar(&mut x);

    // Equal inputs should give equal outputs
    assert!((x[0] - 0.5).abs() < 1e-5);
    assert!((x[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_softmax_scalar_diff_10() {
    // Large difference should make smaller almost 0
    let mut x = vec![0.0, 10.0];
    softmax_scalar(&mut x);

    assert!(x[0] < 0.001);
    assert!(x[1] > 0.999);
}

#[test]
fn test_quantize_activations_q8_0_all_negative() {
    let activations = vec![-5.0f32; 16];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // All quants should be -127 (max negative)
    for q in &quants[..16] {
        assert_eq!(*q, -127);
    }
    // Padding should be 0
    for q in &quants[16..32] {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_quantize_activations_q8_0_alternating() {
    let activations = vec![1.0, -1.0, 1.0, -1.0];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // Scale = 1.0 / 127.0
    // quants should alternate 127, -127
    assert_eq!(quants[0], 127);
    assert_eq!(quants[1], -127);
    assert_eq!(quants[2], 127);
    assert_eq!(quants[3], -127);
}

// ============================================================================
// Test clamping behavior
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_scalar_clamping() {
    // Create input that would exceed i8 range after quantization
    let input = vec![1000.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // All quants should be 127 (clamped max)
    for q in quants {
        assert_eq!(q, 127);
    }
    assert!(scales[0] > 0.0);
}
