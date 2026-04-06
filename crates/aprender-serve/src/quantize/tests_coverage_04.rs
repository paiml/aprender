
#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_dot_negative_activations() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4_0_test_block(1.0, 0xF); // nibble=15 → 15-8=7
    let mut q4_data = Vec::with_capacity(18 * 2);
    for _ in 0..2 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![-3i8; 64];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4_0_q8_0_dot_avx2(&q4_data, &q8_scales, &q8_quants, 64) };

    let diff = (scalar - avx2).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "negative act: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_4block_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 8 blocks = 256 elements (≥ 256, triggers 4-block unrolling)
    let block = build_q4_0_test_block(0.5, 3);
    let mut q4_data = Vec::with_capacity(18 * 8);
    for _ in 0..8 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 8];
    let q8_quants = vec![4i8; 256];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 256);
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2_4b = unsafe { fused_q4_0_q8_0_dot_avx2_4block(&q4_data, &q8_scales, &q8_quants, 256) };

    let diff = (scalar - avx2_4b).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "4block: scalar={scalar} vs avx2_4b={avx2_4b}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_4block_dot_large_dim() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 16 blocks = 512 elements
    let block = build_q4_0_test_block(1.0, 10);
    let mut q4_data = Vec::with_capacity(18 * 16);
    for _ in 0..16 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.1).collect();
    let q8_quants = vec![1i8; 512];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 512);
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2_4b = unsafe { fused_q4_0_q8_0_dot_avx2_4block(&q4_data, &q8_scales, &q8_quants, 512) };

    let diff = (scalar - avx2_4b).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "large dim: scalar={scalar} vs avx2_4b={avx2_4b}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_4block_dot_varying_scales() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 12 blocks = 384 elements (tests non-power-of-2 with 4-block unrolling)
    let block = build_q4_0_test_block(2.0, 7);
    let mut q4_data = Vec::with_capacity(18 * 12);
    for _ in 0..12 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales: Vec<f32> = (0..12)
        .map(|i| if i % 2 == 0 { 1.0 } else { -0.5 })
        .collect();
    let q8_quants = vec![5i8; 384];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 384);
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2_4b = unsafe { fused_q4_0_q8_0_dot_avx2_4block(&q4_data, &q8_scales, &q8_quants, 384) };

    let diff = (scalar - avx2_4b).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "varying: scalar={scalar} vs avx2_4b={avx2_4b}, diff={diff}"
    );
}
