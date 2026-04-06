
#[test]
fn test_quantize_activations_q8_0_clamping_negative() {
    // Very large negative values
    let activations = vec![-1000.0f32; 8];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // All should clamp to -127 (not -128)
    for q in &quants[..8] {
        assert_eq!(*q, -127);
    }
}

// ============================================================================
// Test various epsilon values
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_tiny_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-15; // Very small epsilon

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert!(scales[0] > 0.0);
    // All quants should be equal (uniform input)
    let first = quants[0];
    for q in &quants[1..32] {
        assert_eq!(*q, first);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_large_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1.0; // Large epsilon

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    // Large epsilon affects the inv_rms calculation
    assert!(scales[0] > 0.0);
}

// ============================================================================
// AVX2 specific path tests (block sizes that exercise remainder loops)
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_size_9() {
    // 9 elements in block = 1 SIMD iteration + 1 remainder
    let input: Vec<f32> = (0..9).map(|i| (i as f32) * 0.1).collect();
    let norm_weight = vec![1.0f32; 9];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_size_17() {
    // 17 elements in block = 2 SIMD iterations + 1 remainder
    let input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 17];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_rmsnorm_q8_0_avx2_block_size_25() {
    // 25 elements in block = 3 SIMD iterations + 1 remainder
    let input: Vec<f32> = (0..25).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let norm_weight = vec![1.0f32; 25];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

// ============================================================================
// Test with specific patterns that might cause numerical issues
// ============================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_alternating_signs() {
    let input: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (_scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // All normalized values have same magnitude, so should quantize to same abs value
    let first_abs = quants[0].abs();
    for q in &quants[1..32] {
        assert_eq!(q.abs(), first_abs);
    }
}

#[test]
fn test_fused_swiglu_simd_very_large_values() {
    let mut gate = vec![100.0f32; 16];
    let up = vec![1.0f32; 16];

    fused_swiglu_simd(&mut gate, &up);

    // For large positive, silu(x) ~= x
    for g in &gate {
        assert!((g - 100.0).abs() < 0.1);
    }
}

#[test]
fn test_softmax_simd_all_same_large() {
    let mut x = vec![500.0f32; 16];
    softmax_simd(&mut x);

    // All same -> uniform distribution
    for v in &x {
        assert!((v - 1.0 / 16.0).abs() < 1e-5);
    }
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_multi_block() {
    // 128 elements = 4 blocks
    let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let norm_weight = vec![1.0f32; 128];
    let eps = 1e-5;

    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 128];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

    // All 4 blocks should have positive scales
    for s in &scales {
        assert!(*s > 0.0);
    }
}
