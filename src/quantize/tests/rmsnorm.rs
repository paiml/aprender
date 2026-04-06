
#[test]
fn test_rmsnorm_q8_scalar_varying_weights() {
    let input = vec![2.0f32; 64];
    let norm_weight: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) / 32.0).collect();
    let eps = 1e-5;
    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 2); // 64 / 32
    assert_eq!(quants.len(), 64);
    // Verify that quants are valid (within i8 range implicitly by type)
    for &q in &quants {
        assert!(q >= -128 && q <= 127);
    }
}

#[test]
fn test_rmsnorm_q8_scalar_partial_block() {
    // 48 elements: 1 full block + 1 partial (16 elements + 16 padding)
    let input = vec![1.0f32; 48];
    let norm_weight = vec![1.0f32; 48];
    let eps = 1e-5;
    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 2); // ceil(48/32) = 2
    assert_eq!(quants.len(), 64); // 2 * 32
    // Padding elements should be zero
    for i in 48..64 {
        assert_eq!(quants[i], 0, "Padding at index {} should be 0", i);
    }
}

// ============================================================================
// quantize_rmsnorm_q8_0_into
// ============================================================================

#[test]
fn test_rmsnorm_q8_into_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];
    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);
    assert!(scales[0] > 0.0);
    let first_q = quants[0];
    for &q in &quants {
        assert_eq!(q, first_q, "Uniform input should give uniform quants");
    }
}

#[test]
fn test_rmsnorm_q8_into_matches_allocating() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
    let norm_weight: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) / 64.0).collect();
    let eps = 1e-5;

    // Allocating version
    let (scales_alloc, quants_alloc) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

    // Zero-allocation version
    let mut scales_into = vec![0.0f32; 2];
    let mut quants_into = vec![0i8; 64];
    quantize_rmsnorm_q8_0_into(
        &input,
        &norm_weight,
        eps,
        &mut scales_into,
        &mut quants_into,
    );

    // Should match
    for i in 0..2 {
        assert!(
            (scales_alloc[i] - scales_into[i]).abs() < 1e-7,
            "Scale mismatch at {}: alloc={}, into={}",
            i,
            scales_alloc[i],
            scales_into[i]
        );
    }
    for i in 0..64 {
        assert_eq!(
            quants_alloc[i], quants_into[i],
            "Quant mismatch at {}: alloc={}, into={}",
            i, quants_alloc[i], quants_into[i]
        );
    }
}
