
#[test]
fn test_fused_q4_0_q8_0_dot_scalar_multiple_blocks() {
    // 2 blocks = 64 elements
    let q4_data = vec![0u8; 36]; // 2 * 18 bytes
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 64];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    assert!(result.is_finite());
}

// =============================================================================
// fused_q8_0_q8_0_dot_scalar tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_zero() {
    // Q8_0 block: 2 bytes f16 scale + 32 bytes quants = 34 bytes
    let q8_weight = vec![0u8; 34];
    let q8_scales = vec![0.0f32];
    let q8_quants = vec![0i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_basic() {
    // Create Q8_0 weight with scale = 1.0
    let mut q8_weight = vec![0u8; 34];
    q8_weight[0] = 0x00;
    q8_weight[1] = 0x3C; // f16 1.0

    // Set quantized values
    for i in 2..34 {
        q8_weight[i] = 1; // i8 value 1
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
    // Should be 32 * 1 * 1 * 1 * 1 = 32
    assert!((result - 32.0).abs() < 1.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_negative() {
    let mut q8_weight = vec![0u8; 34];
    q8_weight[0] = 0x00;
    q8_weight[1] = 0x3C; // f16 1.0

    // Set quantized values to -1 (0xFF as i8)
    for i in 2..34 {
        q8_weight[i] = 0xFF; // -1 as i8
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
    // Should be 32 * (-1) * 1 * 1 * 1 = -32
    assert!((result - (-32.0)).abs() < 1.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_multiple_blocks() {
    // 2 blocks = 64 elements
    let q8_weight = vec![0u8; 68]; // 2 * 34 bytes
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 64];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 64);
    assert!(result.is_finite());
}

// =============================================================================
// Additional edge case tests for high coverage
// =============================================================================

#[test]
fn test_q4_0_dot_scalar_partial_block() {
    // Test with in_dim not multiple of 32 (partial final block)
    let q4_data = vec![0u8; 36]; // 2 blocks worth
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 50]; // Only 50 elements, not 64

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 50);
    assert!(result.is_finite());
}

#[test]
fn test_q8_0_dot_scalar_partial_block() {
    let q8_weight = vec![0u8; 68];
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 50];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight, &q8_scales, &q8_quants, 50);
    assert!(result.is_finite());
}

#[test]
fn test_interleaved_q4k_three_superblocks() {
    let data = vec![0u8; 432]; // 3 super-blocks
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 3);
    assert_eq!(interleaved.num_values(), 768);
}

#[test]
fn test_q8k_into_exact_buffer_sizes() {
    let activations = vec![0.5f32; 256];
    let mut scales = vec![0.0f32; 1]; // Exactly 1 needed
    let mut quants = vec![0i8; 256]; // Exactly 256 needed

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_extract_scale_min_all_blocks() {
    // Test all 8 blocks to ensure both branches are covered
    let scales: [u8; 12] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    ];

    for i in 0..8 {
        let (scale, min) = extract_scale_min(&scales, i);
        assert!(scale >= 0.0);
        assert!(min >= 0.0);
    }
}
