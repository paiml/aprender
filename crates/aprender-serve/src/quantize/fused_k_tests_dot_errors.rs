
#[test]
fn test_fused_q4k_q8k_dot_error_messages() {
    // Test each error path explicitly
    let data = vec![0u8; 100];
    let err = fused_q4k_q8k_dot(&data, &[1.0], &[1; 256]).unwrap_err();
    assert!(err.to_string().contains("not a multiple"));

    let data = vec![0u8; 144];
    let err = fused_q4k_q8k_dot(&data, &[], &[1; 256]).unwrap_err();
    assert!(err.to_string().contains("scales"));

    let err = fused_q4k_q8k_dot(&data, &[1.0], &[1; 100]).unwrap_err();
    assert!(err.to_string().contains("quants"));
}

// --- fused_q4k_q8k_dot_simd error paths ---

#[test]
fn test_fused_q4k_q8k_dot_simd_error_paths() {
    // Invalid data length
    let err = fused_q4k_q8k_dot_simd(&[0u8; 100], &[1.0], &[1i8; 256]).unwrap_err();
    assert!(err.to_string().contains("not a multiple"));
}

#[test]
fn test_fused_q4k_dot_simd_error_paths() {
    // Activation length mismatch via simd path
    let data = vec![0u8; 144];
    let activations = vec![0.0f32; 100];
    let err = fused_q4k_dot_simd(&data, &activations).unwrap_err();
    assert!(err.to_string().contains("doesn't match"));
}

// --- fused_q4k_dot: packed scale blocks (blocks 4-7) ---

#[test]
fn test_fused_q4k_dot_packed_scale_blocks() {
    // Exercise blocks 4-7 which use packed scale layout
    let mut data = vec![0u8; 144];

    // d = 1.0, dmin = 0
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // For block 4 (is=4): packed layout
    // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    //
    // Set scales[0] = 0b11_000000 (high 2 bits = 3, low 6 bits = 0)
    // Set scales[8] = 0b0010_0101 (high 4 = 2, low 4 = 5)
    // scale = 5 | (3 << 4) = 5 | 48 = 53
    // min = 2 | (0 << 4) = 2
    data[4] = 0b1100_0000; // scales[0]
    data[12] = 0b0010_0101; // scales[8]

    // Set qs for chunk 2 (j=128, which reads qs[64..96])
    // These are for blocks 4 and 5
    for i in 64..96 {
        data[16 + i] = 0x22; // low=2, high=2
    }

    let activations = vec![1.0f32; 256];
    let result = fused_q4k_dot(&data, &activations).expect("should succeed");

    // Chunk 2 (j=128):
    //   is=4: sc1 = (scales[8]&0x0F)|((scales[0]>>6)<<4) = 5 | 48 = 53
    //         m1  = (scales[8]>>4)|((scales[4]>>6)<<4)    = 2 | 0 = 2
    //   is=5: sc2 = (scales[9]&0x0F)|((scales[1]>>6)<<4) = 0
    //
    //   d1 = 1.0 * 53 = 53.0, dm1 = 0 * 2 = 0 (dmin=0)
    //   Low nibbles (32): val = 53.0 * 2 - 0 = 106.0, sum = 32 * 106 = 3392.0
    //   High nibbles: sc2=0 so 0
    //
    // Total = 3392.0
    assert!(
        (result - 3392.0).abs() < 1.0,
        "Expected about 3392.0, got {}",
        result
    );
}

// --- Symmetry and sign tests ---

#[test]
fn test_fused_q4k_dot_sign_reversal() {
    // If we negate all activations, result should negate
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 1;
    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let pos_act = vec![1.0f32; 256];
    let neg_act = vec![-1.0f32; 256];

    let pos_result = fused_q4k_dot(&data, &pos_act).expect("pos");
    let neg_result = fused_q4k_dot(&data, &neg_act).expect("neg");

    assert!(
        (pos_result + neg_result).abs() < 0.01,
        "Negating activations should negate result: {} vs {}",
        pos_result,
        neg_result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_sign_reversal() {
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    data[4] = 1;
    for i in 0..128 {
        data[16 + i] = 0x55;
    }

    let q8k_scales = vec![1.0f32];
    let pos_quants = vec![10i8; 256];
    let neg_quants = vec![-10i8; 256];

    let pos_result = fused_q4k_q8k_dot(&data, &q8k_scales, &pos_quants).expect("pos");
    let neg_result = fused_q4k_q8k_dot(&data, &q8k_scales, &neg_quants).expect("neg");

    // With dmin=0, negating quants should negate result
    // (When dmin != 0, the min correction also inverts, so this holds)
    assert!(
        (pos_result + neg_result).abs() < 1.0,
        "Negating quants should negate result: {} vs {}",
        pos_result,
        neg_result
    );
}

// --- Large multi-block SIMD parity ---

#[test]
fn test_fused_q4k_dot_simd_16_super_blocks() {
    // 16 super-blocks = 2304 bytes = 4096 values
    let mut data = vec![0u8; 16 * 144];

    for sb in 0..16 {
        let offset = sb * 144;
        // d = 0.1 (f16 ~ 0x2E66)
        data[offset..offset + 2].copy_from_slice(&0x2E66u16.to_le_bytes());
        data[offset + 2..offset + 4].copy_from_slice(&0x2800u16.to_le_bytes());

        // Varied scales
        for i in 0..12 {
            data[offset + 4 + i] = ((sb + i * 5 + 1) % 63) as u8;
        }

        // Varied qs
        for i in 0..128 {
            data[offset + 16 + i] = ((sb * 37 + i * 23 + 5) % 256) as u8;
        }
    }

    let activations: Vec<f32> = (0..4096)
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.005 - 0.5)
        .collect();

    let scalar = fused_q4k_dot(&data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&data, &activations).expect("simd");

    let rel_err = if scalar.abs() > 1e-6 {
        (simd - scalar).abs() / scalar.abs()
    } else {
        (simd - scalar).abs()
    };
    assert!(
        rel_err < 0.01,
        "16-superblock parity: scalar={}, simd={}, rel_err={}",
        scalar,
        simd,
        rel_err
    );
}

#[test]
fn test_fused_q4k_q8k_dot_empty() {
    let result = fused_q4k_q8k_dot(&[], &[], &[]).expect("empty should work");
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_q8k_dot_simd_empty() {
    let result = fused_q4k_q8k_dot_simd(&[], &[], &[]).expect("empty should work");
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4k_dot_simd_empty() {
    let result = fused_q4k_dot_simd(&[], &[]).expect("empty should work");
    assert_eq!(result, 0.0);
}

// --- dequantize_q4_k consistency check ---

/// PMAT-170: Q4K Layout Consistency Test
///
/// Verifies that apr::dequantize_q4_k produces the same element ordering
/// as fused_q4k_parallel_matvec. This was the root cause of GPU explosion bug #170.
#[test]
fn test_q4k_layout_consistency_pmat170() {
    use crate::apr::dequantize_q4_k;
    use crate::quantize::fused_q4k_parallel_matvec;

    // Use 256x256 test matrix (1 super-block per row)
    let in_dim = 256;
    let out_dim = 256;
    let num_elements = in_dim * out_dim;

    // Create reproducible Q4K test data (144 bytes per row)
    let bytes_per_row = 144;
    let total_bytes = out_dim * bytes_per_row;
    let q4k_bytes: Vec<u8> = (0..total_bytes)
        .map(|i| ((i * 17 + 37) % 256) as u8)
        .collect();

    // Method 1: Direct dequantization
    let dequant = dequantize_q4_k(&q4k_bytes, num_elements);

    // Method 2: Extract columns via fused matmul with basis vectors
    let mut fused_matrix = vec![0.0f32; num_elements];
    for col in 0..in_dim {
        // Basis vector: e_col = [0, ..., 0, 1, 0, ..., 0]
        let mut basis = vec![0.0f32; in_dim];
        basis[col] = 1.0;

        // fused_q4k_parallel_matvec produces W @ basis = column col of W
        if let Ok(column) = fused_q4k_parallel_matvec(&q4k_bytes, &basis, in_dim, out_dim) {
            for row in 0..out_dim {
                fused_matrix[row * in_dim + col] = column[row];
            }
        }
    }

    // Compare element by element
    // Contract: cpu-q4k-activation-quant-v1.yaml (Refs realizar#96)
    // fused_q4k_parallel_matvec now uses Q8K activation quantization internally,
    // which introduces small rounding (< 0.2%) vs the f32 dequant path.
    // We use relative tolerance to catch layout bugs while allowing Q8K rounding.
    let mut mismatches = 0;
    let mut max_rel_err = 0.0f32;

    for i in 0..num_elements {
        let diff = (dequant[i] - fused_matrix[i]).abs();
        let rel_err = if dequant[i].abs() > 1e-6 {
            diff / dequant[i].abs()
        } else {
            diff
        };
        if rel_err > 0.01 {
            mismatches += 1;
            max_rel_err = max_rel_err.max(rel_err);
        }
    }

    assert_eq!(
        mismatches, 0,
        "Q4K layout mismatch: {} elements differ (max rel_err: {:.4}%). \
             This indicates dequantize_q4_k has different element ordering \
             than fused_q4k_parallel_matvec, which would cause GPU explosion.",
        mismatches,
        max_rel_err * 100.0
    );
}

// ============================================================================
// FUSED Q4_K × Q8_K DOT PRODUCT — AVX2 COVERAGE TESTS
// ============================================================================
// These tests call the unsafe fused_q4k_q8k_dot_avx2 directly to cover
// the AVX2 code path (which is unreachable through the public API on
// machines with AVX-512 VNNI).

/// Build valid Q4K super-block data for testing.
/// Each super-block: [d:f16(2), dmin:f16(2), scales:12, quants:128] = 144 bytes
fn build_q4k_test_block(d: f32, dmin: f32, nibble_val: u8) -> [u8; 144] {
    let mut block = [0u8; 144];
    // d as f16
    let d_bits = half::f16::from_f32(d).to_bits();
    block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    // dmin as f16
    let dmin_bits = half::f16::from_f32(dmin).to_bits();
    block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
    // scales: set all to give scale=1, min=0 (6-bit encoded)
    // For extract_scale_min, lower 4 bits = scale, upper 2 bits = min
    for i in 0..12 {
        block[4 + i] = 0x01; // scale=1, min=0 in packed format
    }
    // quants: 128 bytes, each byte has lo and hi nibble
    let packed = (nibble_val & 0x0F) | ((nibble_val & 0x0F) << 4);
    for i in 0..128 {
        block[16 + i] = packed;
    }
    block
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip on non-AVX2 hardware
    }

    // Build 1 super-block
    let block = build_q4k_test_block(1.0, 0.0, 3);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.expect("avx2");

    let diff = (scalar - avx2).abs();
    assert!(
        diff < 1.0,
        "scalar={scalar} vs avx2={avx2}, diff={diff} exceeds tolerance"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_zero_quants() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 0);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![0i8; 256];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.expect("result");
    assert!(
        result.abs() < 1e-6,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 4 super-blocks
    let block = build_q4k_test_block(1.0, 0.0, 5);
    let mut q4k_data = Vec::with_capacity(144 * 4);
    for _ in 0..4 {
        q4k_data.extend_from_slice(&block);
    }
    let q8k_scales = vec![1.0f32; 4];
    let q8k_quants = vec![2i8; 256 * 4];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.expect("avx2");

    let diff = (scalar - avx2).abs();
    // Allow larger tolerance for multi-block accumulation
    let rel_tolerance = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < rel_tolerance,
        "4-block: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_negative_quants() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 7);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-3i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.expect("avx2");

    let diff = (scalar - avx2).abs();
    let rel_tolerance = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < rel_tolerance,
        "neg quants: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_with_dmin() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // Non-zero dmin affects the min-subtraction path
    let block = build_q4k_test_block(1.0, 0.5, 4);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![2.0f32];
    let q8k_quants = vec![5i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.expect("avx2");

    let diff = (scalar - avx2).abs();
    let rel_tolerance = scalar.abs().max(1.0) * 0.05;
    assert!(
        diff < rel_tolerance,
        "dmin: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let q4k_data = vec![0u8; 100]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for non-144-aligned data");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_buffer_too_small() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 1);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 128]; // Too small (need 256)

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for too-small Q8K buffer");
}
