
// =============================================================================
// Parallel matvec error path tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_weight_short() {
    // Weights too short for dimensions
    let in_dim = 64;
    let out_dim = 4;
    let bytes_per_row = 36; // 2 blocks per row
    let needed = out_dim * bytes_per_row; // 144 bytes
    let weight_data = vec![0u8; needed - 1]; // 1 byte short

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_mod_dim_mismatch() {
    let in_dim = 32;
    let bytes_per_row = 18;
    let weight_data = vec![0u8; 4 * bytes_per_row];
    let mut output = vec![0.0f32; 4];

    // Wrong activation length
    let activations = vec![1.0f32; 64];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_mod_weight_short() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let needed = out_dim * bytes_per_row;
    let weight_data = vec![0u8; needed - 1];

    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_mod_dim_mismatch() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let mut output = vec![0.0f32; out_dim];

    // Wrong activation length
    let activations = vec![1.0f32; 64];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_mod_output_short() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    // Output too short
    let mut output = vec![0.0f32; 2];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

// =============================================================================
// Sequential path tests (small dimensions)
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_sequential_path() {
    // Small dimensions to trigger sequential path (out_dim < 512)
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    // Set scales to 1.0 for each row
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        weight_data[row_start..row_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // Set quants to centered (0x88)
        for i in 2..18 {
            weight_data[row_start + i] = 0x88;
        }
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_mod_parallel_path() {
    // Large dimensions to trigger parallel path (out_dim >= 512)
    let in_dim = 32;
    let out_dim = 600;
    let bytes_per_row = 18;

    let mut weight_data = vec![0u8; out_dim * bytes_per_row];
    // Set all scales to 1.0
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        weight_data[row_start..row_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// Q8_0Block additional tests
// =============================================================================

#[test]
fn test_q8_0block_mod_quantize_extreme_values() {
    let mut values = [0.0f32; 32];
    values[0] = f32::MAX / 2.0;
    values[1] = -f32::MAX / 2.0;

    let block = Q8_0Block::quantize(&values);
    assert!(block.scale.is_finite());

    let dequant = block.dequantize();
    for v in &dequant {
        assert!(v.is_finite());
    }
}

#[test]
fn test_q8_0block_mod_roundtrip_small_values() {
    let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.001);
    let block = Q8_0Block::quantize(&values);
    let dequant = block.dequantize();

    // Small values may lose precision
    for (&orig, &deq) in values.iter().zip(dequant.iter()) {
        assert!((orig - deq).abs() < 0.01, "orig={}, deq={}", orig, deq);
    }
}

// =============================================================================
// Q8KSuperBlock additional tests
// =============================================================================

#[test]
fn test_q8k_superblock_mod_quantize_alternating() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }

    let block = Q8KSuperBlock::quantize(&values);

    // Check that alternating pattern is preserved in sign
    for i in 0..256 {
        if i % 2 == 0 {
            assert!(block.quants[i] > 0, "Even index should be positive");
        } else {
            assert!(block.quants[i] < 0, "Odd index should be negative");
        }
    }
}

#[test]
fn test_q8k_superblock_mod_quantize_into_with_overflow_values() {
    // Values that would overflow if not clamped
    let values = [500.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

    // All should clamp to 127
    for q in &quants {
        assert_eq!(*q, 127);
    }
}

// =============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks additional tests
// =============================================================================

#[test]
fn test_quantize_to_q8_blocks_mod_exact_multiple() {
    // Exactly 2 blocks = 64 values
    let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("valid");

    assert_eq!(blocks.len(), 2);

    // Dequantize and verify
    let dequant = dequantize_q8_blocks(&blocks);
    assert_eq!(dequant.len(), 64);

    for (&orig, &deq) in values.iter().zip(dequant.iter()) {
        assert!((orig - deq).abs() < 0.5, "orig={}, deq={}", orig, deq);
    }
}

#[test]
fn test_quantize_to_q8_blocks_mod_not_multiple() {
    // 50 values (not multiple of 32)
    let values = vec![1.0f32; 50];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_blocks_mod_preserves_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let blocks = vec![block];

    let dequant = dequantize_q8_blocks(&blocks);

    for v in &dequant {
        assert!(v.abs() < 1e-6, "Should be near zero: {}", v);
    }
}

// =============================================================================
// quantize_activations_q8k_into additional tests
// =============================================================================

#[test]
fn test_quantize_activations_q8k_into_mod_not_multiple() {
    let activations = vec![1.0f32; 300]; // Not multiple of 256
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 300];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_mod_scales_too_small() {
    let activations = vec![1.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Only 1, need 2
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_mod_quants_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 128]; // Only 128, need 256

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_mod_success() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    quantize_activations_q8k_into(&activations, &mut scales, &mut quants).expect("should work");

    assert!(scales[0] > 0.0);

    // Check that signs are preserved
    for i in 0..128 {
        assert!(quants[i] <= 0, "First half should be negative or zero");
    }
    for i in 128..256 {
        assert!(quants[i] >= 0, "Second half should be positive or zero");
    }
}

// =============================================================================
// Additional coverage: InterleavedQ4K::from_q4k error handling
// =============================================================================

#[test]
fn test_interleaved_q4k_from_q4k_invalid_length() {
    // 143 bytes - not multiple of 144
    let data = vec![0u8; 143];
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_from_q4k_empty() {
    let data: Vec<u8> = vec![];
    let result = InterleavedQ4K::from_q4k(&data);
    // Empty is technically valid (0 super-blocks)
    assert!(result.is_ok());
    let interleaved = result.expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 0);
    assert_eq!(interleaved.num_values(), 0);
}

#[test]
fn test_interleaved_q4k_num_values() {
    // 1 super-block = 256 values
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_values(), 256);

    // 3 super-blocks = 768 values
    let data = vec![0u8; 144 * 3];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_values(), 768);
}

#[test]
fn test_interleaved_q4k_from_q4k_multiple_superblocks() {
    // Test with 4 super-blocks
    let mut data = vec![0u8; 144 * 4];

    // Set different d values for each super-block
    for sb in 0..4 {
        let d_val = 1.0 + sb as f32 * 0.5;
        let d_bits = half::f16::from_f32(d_val).to_bits();
        let sb_start = sb * 144;
        data[sb_start..sb_start + 2].copy_from_slice(&d_bits.to_le_bytes());
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    assert_eq!(interleaved.num_super_blocks, 4);
    assert_eq!(interleaved.d.len(), 4);

    // Verify d values were extracted correctly
    for (i, &d) in interleaved.d.iter().enumerate() {
        let expected = 1.0 + i as f32 * 0.5;
        assert!(
            (d - expected).abs() < 0.01,
            "Super-block {} d: expected {}, got {}",
            i,
            expected,
            d
        );
    }
}

// =============================================================================
// fused_q4_0_q8_0_dot_simd various dimensions (trigger different paths)
// =============================================================================

use crate::quantize::fused_q4_0_q8_0_dot_simd;

#[test]
fn test_fused_q4_0_q8_0_dot_simd_mod_small_dim() {
    // in_dim = 32 (1 block) - tests single block path
    let in_dim = 32;
    let mut q4_data = vec![0u8; 18];
    q4_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    for i in 2..18 {
        q4_data[i] = 0x44; // q_low=4, q_high=4
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![2i8; 32];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_mod_medium_dim() {
    // in_dim = 128 (4 blocks) - tests 2-block path
    let in_dim = 128;
    let num_blocks = 4;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            q4_data[start + i] = 0x55;
        }
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_simd_mod_large_dim() {
    // in_dim = 512 (16 blocks) - tests 4-block unrolling path
    let in_dim = 512;
    let num_blocks = 16;
    let mut q4_data = vec![0u8; num_blocks * 18];

    for block in 0..num_blocks {
        let start = block * 18;
        q4_data[start..start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            q4_data[start + i] = 0x88;
        }
    }

    let q8_scales = vec![1.0f32; num_blocks];
    let q8_quants = vec![1i8; in_dim];

    let result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, in_dim);
    assert!(result.is_finite());
}
