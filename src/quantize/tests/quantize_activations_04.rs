
// =============================================================================
// quantize_activations_q8k_into Tests
// =============================================================================

#[test]
fn test_quantize_activations_q8k_into_two_superblocks() {
    let activations: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.1).collect();
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("should work");

    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);

    // Check that first superblock has mostly negative values
    let negative_count: usize = quants[..256].iter().filter(|&&q| q < 0).count();
    assert!(
        negative_count > 100,
        "First superblock should have many negatives"
    );

    // Check that second superblock has mostly positive values
    let positive_count: usize = quants[256..].iter().filter(|&&q| q > 0).count();
    assert!(
        positive_count > 100,
        "Second superblock should have many positives"
    );
}

#[test]
fn test_quantize_activations_q8k_into_uniform() {
    let activations = vec![50.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("should work");

    // All quants should be 127 (max positive)
    for q in &quants {
        assert_eq!(*q, 127);
    }
}

// =============================================================================
// fused_q4_0_q8_0_dot_scalar Special Values
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_with_nan_scale() {
    let mut q4_data = vec![0u8; 18];
    // NaN in f16: 0x7E00
    q4_data[0..2].copy_from_slice(&0x7E00u16.to_le_bytes());

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_nan(), "NaN scale should propagate");
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_with_inf_scale() {
    let mut q4_data = vec![0u8; 18];
    // Infinity in f16: 0x7C00
    q4_data[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(
        result.is_infinite() || result.is_nan(),
        "Inf scale should produce inf/nan"
    );
}

// =============================================================================
// InterleavedQ4K::dot Scalar Fallback (non-x86_64 path)
// =============================================================================

#[test]
fn test_interleaved_q4k_dot_scalar_via_small_input() {
    // Even on x86_64 with AVX2, we test the scalar path through the dot() dispatcher
    // The scalar path is also tested indirectly - here we ensure correctness
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    data[4] = 1; // scale for first block

    // Set recognizable pattern in qs
    for i in 16..144 {
        data[i] = 0x21; // q_low=1, q_high=2
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];

    let result = interleaved.dot(&activations).expect("dot works");
    assert!(result.is_finite());
}

// =============================================================================
// Q8KSuperBlock::quantize_into Tests
// =============================================================================

#[test]
fn test_q8k_superblock_quantize_into_basic() {
    let values = [5.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // All should be 127
    for q in &quants {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_q8k_superblock_quantize_into_negative() {
    let values = [-5.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // All should be -127
    for q in &quants {
        assert_eq!(*q, -127);
    }
}

#[test]
fn test_q8k_superblock_quantize_into_mixed() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 128.0) * 0.5;
    }
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    assert!(scale > 0.0);
    // First half should be negative, second half positive
    assert!(quants[0] < 0);
    assert!(quants[255] > 0);
}

// =============================================================================
// Additional fused_q4_0_q8_0_parallel_matvec_into Tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_large_output() {
    let in_dim = 64;
    let out_dim = 256;
    let bytes_per_row = 36; // 2 blocks

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    for row in 0..out_dim {
        for block in 0..2 {
            let start = row * bytes_per_row + block * 18;
            weight_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
    }

    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());

    for v in &output {
        assert!(v.is_finite());
    }
}

// =============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks Additional Tests
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_four_blocks() {
    let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("valid");

    assert_eq!(blocks.len(), 4);

    let dequant = dequantize_q8_blocks(&blocks);
    assert_eq!(dequant.len(), 128);

    // Check trend preserved
    for i in 1..128 {
        assert!(
            dequant[i] > dequant[i - 1] - 0.5,
            "Values should be roughly ascending"
        );
    }
}

#[test]
fn test_dequantize_q8_blocks_single_block() {
    let values = [10.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let blocks = vec![block];

    let dequant = dequantize_q8_blocks(&blocks);

    assert_eq!(dequant.len(), 32);
    for v in &dequant {
        assert!((v - 10.0).abs() < 0.5);
    }
}
