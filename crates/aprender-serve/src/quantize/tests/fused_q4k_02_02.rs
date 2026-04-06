
#[test]
fn test_fused_q4k_q8k_dot_large_scales() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let q8k_scales = vec![1000.0f32];
    let q8k_quants = vec![10i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

// -----------------------------------------------------------------------------
// Stress Tests
// -----------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_simd_32_blocks() {
    let q4k_data = gen_multi_q4k_blocks(32);
    let activations: Vec<f32> = (0..32 * QK_K).map(|i| (i as f32 * 0.001).sin()).collect();

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok(), "32-block stress test should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_simd_32_blocks() {
    let q4k_data = gen_multi_q4k_blocks(32);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(32);

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(
        result.is_ok(),
        "32-block Q4K×Q8K stress test should succeed"
    );
}
