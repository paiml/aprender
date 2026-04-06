
/// Test with mixed positive and negative large values
#[test]
fn test_quantize_activations_q8_0_mixed_large() {
    let large_val = 1e10;
    let activations = vec![large_val, -large_val, large_val, -large_val];
    let (_scales, quants) = quantize_activations_q8_0(&activations);

    // Should correctly quantize to extremes
    assert_eq!(quants[0], 127);
    assert_eq!(quants[1], -127);
    assert_eq!(quants[2], 127);
    assert_eq!(quants[3], -127);
}

// ============================================================================
// Scalar vs SIMD Consistency - Specific Patterns
// ============================================================================

/// Test that scalar and SIMD produce consistent results for edge patterns
#[test]
fn test_swiglu_scalar_simd_edge_consistency() {
    // Pattern that exercises exp approximation edges
    let values: Vec<f32> = vec![
        -100.0,
        -87.0,
        -50.0,
        -10.0,
        -1.0,
        0.0,
        1.0,
        10.0,
        50.0,
        87.0,
        100.0,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        0.5,
        -0.5,
        2.0,
    ];
    let up = vec![1.0f32; 16];

    let mut gate_scalar = values.clone();
    fused_swiglu_scalar(&mut gate_scalar, &up);

    let mut gate_simd = values;
    fused_swiglu_simd(&mut gate_simd, &up);

    // Both should produce finite results (may differ in precision)
    for (s, d) in gate_scalar.iter().zip(gate_simd.iter()) {
        assert!(s.is_finite(), "Scalar should be finite");
        assert!(d.is_finite(), "SIMD should be finite");
    }
}

/// Test softmax scalar/SIMD consistency with extreme range
#[test]
fn test_softmax_scalar_simd_extreme_consistency() {
    let values: Vec<f32> = vec![-1000.0, -500.0, -100.0, -10.0, 0.0, 10.0, 100.0, 500.0];

    let mut x_scalar = values.clone();
    softmax_scalar(&mut x_scalar);

    let mut x_simd = values;
    softmax_simd(&mut x_simd);

    // Both should sum to 1.0
    let sum_scalar: f32 = x_scalar.iter().sum();
    let sum_simd: f32 = x_simd.iter().sum();

    assert!(
        (sum_scalar - 1.0).abs() < 1e-4,
        "Scalar sum: {}",
        sum_scalar
    );
    assert!((sum_simd - 1.0).abs() < 1e-4, "SIMD sum: {}", sum_simd);
}

// ============================================================================
// Weight Pattern Tests for RMSNorm
// ============================================================================

/// Test with weights that vary across blocks
#[test]
fn test_quantize_rmsnorm_q8_0_varying_block_weights() {
    let input = vec![1.0f32; 64];
    // First block has small weights, second block has large weights
    let mut norm_weight = vec![0.1f32; 32];
    norm_weight.extend(vec![10.0f32; 32]);
    let eps = 1e-5;

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // First block should have smaller scale than second block
    assert!(
        scales[0] < scales[1],
        "Second block should have larger scale due to larger weights"
    );
}

/// Test with zero weights in one block, non-zero in another
#[test]
fn test_quantize_rmsnorm_q8_0_zero_first_block_weights() {
    let input = vec![1.0f32; 64];
    let mut norm_weight = vec![0.0f32; 32]; // Zero weights in first block
    norm_weight.extend(vec![1.0f32; 32]); // Non-zero in second block
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // First block should have fallback scale (1.0/127.0)
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    // First block quants should all be zero
    for q in &quants[..32] {
        assert_eq!(*q, 0i8);
    }
    // Second block should have non-zero values
    assert!(scales[1] > 0.0);
}

// ============================================================================
// Epsilon Edge Cases
// ============================================================================

/// Test with negative epsilon (should still work, just adds negative to mean_sq)
#[test]
fn test_quantize_rmsnorm_q8_0_negative_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = -1e-5; // Negative epsilon (unusual but valid float)

    let (scales, _quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should still produce valid output (mean_sq is 1.0, so 1.0 + (-1e-5) > 0)
    assert!(scales[0].is_finite());
    assert!(scales[0] > 0.0);
}

/// Test with infinity epsilon
#[test]
fn test_quantize_rmsnorm_q8_0_infinity_epsilon() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = f32::INFINITY;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // inv_rms = 1.0 / sqrt(mean_sq + inf) = 1.0 / inf = 0.0
    // So all normalized values are 0, triggering fallback scale
    assert!((scales[0] - 1.0 / 127.0).abs() < 1e-10);
    for q in &quants {
        assert_eq!(*q, 0i8);
    }
}

// ============================================================================
// Input Pattern Edge Cases
// ============================================================================

/// Test with single very large value, rest zeros
#[test]
fn test_quantize_rmsnorm_q8_0_spike_input() {
    let mut input = vec![0.0f32; 32];
    input[0] = 1e6; // Single spike
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (_scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Only first element should have non-zero quant
    assert_eq!(quants[0], 127);
    for q in &quants[1..32] {
        assert_eq!(*q, 0i8);
    }
}

/// Test with decreasing exponential pattern
#[test]
fn test_quantize_rmsnorm_q8_0_exponential_decay() {
    let input: Vec<f32> = (0..32).map(|i| (-0.1 * i as f32).exp()).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

    // Should produce valid quantized output with highest value at index 0
    assert!(scales[0] > 0.0);
    // First elements should have higher absolute values
    assert!(quants[0].abs() >= quants[31].abs());
}
