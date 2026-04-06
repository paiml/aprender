
/// Test minimum valid Q8_0 matvec (1 block, 1 output)
#[test]
fn test_minimum_q8_0_matvec() {
    let in_dim = 32;
    let out_dim = 1;
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

/// Test exactly at parallel threshold (1024 rows)
#[test]
fn test_exactly_at_parallel_threshold() {
    let in_dim = 32;
    let out_dim = 1024; // Exactly at threshold
    let weight_data = vec![0u8; out_dim * 18];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

/// Test just below parallel threshold
#[test]
fn test_below_parallel_threshold() {
    let in_dim = 32;
    let out_dim = 1023; // Just below threshold
    let weight_data = vec![0u8; out_dim * 18];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

// =============================================================================
// Scalar Fallback Path Tests
// =============================================================================

/// Test scalar dot product handles block boundary correctly
#[test]
fn test_scalar_dot_block_boundary() {
    use crate::quantize::activation::quantize_activations_q8_0;
    use crate::quantize::fused_q4_0_q8_0_dot_scalar;

    // Create exactly 3 blocks (96 elements)
    let in_dim = 96;
    let q4_data = vec![0u8; 3 * 18];
    let activations: Vec<f32> = (0..in_dim).map(|i| i as f32 / 100.0).collect();

    let (q8_scales, q8_quants) = quantize_activations_q8_0(&activations);

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

/// Test scalar dot product with truncated data
#[test]
fn test_scalar_dot_truncated_data() {
    use crate::quantize::activation::quantize_activations_q8_0;
    use crate::quantize::fused_q4_0_q8_0_dot_scalar;

    let in_dim = 64;
    // Provide less data than needed - scalar should handle gracefully
    let q4_data = vec![0u8; 18]; // Only 1 block instead of 2
    let activations = vec![1.0f32; in_dim];

    let (q8_scales, q8_quants) = quantize_activations_q8_0(&activations);

    // Should complete without panic (may give partial result)
    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}
