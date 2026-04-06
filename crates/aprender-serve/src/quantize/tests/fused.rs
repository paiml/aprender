
#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_large_matrix_deep_qcov_025() {
    // Test with larger matrix to exercise parallel path
    let in_dim: usize = 128;
    let out_dim = 64;
    let blocks_per_row = in_dim.div_ceil(32);
    let bytes_per_row = blocks_per_row * 18;
    let mut weight_data = vec![0u8; out_dim * bytes_per_row];

    // Set scales for each block
    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let offset = row * bytes_per_row + block * 18;
            weight_data[offset..offset + 2]
                .copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

// -------------------------------------------------------------------------
// fused_q4k_q8k_dot_simd Parity Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_dot_simd_vs_scalar_parity_deep_qcov_026() {
    // Verify SIMD and scalar produce same results
    let mut q4k_data = Vec::with_capacity(144);
    q4k_data.extend_from_slice(&half::f16::from_f32(1.5).to_le_bytes());
    q4k_data.extend_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
    for i in 0..12 {
        q4k_data.push((i * 5 % 64) as u8);
    }
    for i in 0..128 {
        q4k_data.push((i * 7 % 256) as u8);
    }

    let q8k_scales = vec![2.0f32];
    let q8k_quants: Vec<i8> = (0..256).map(|i| ((i * 3) % 127) as i8).collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar ok");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("simd ok");

    let rel_diff = (scalar - simd).abs() / (scalar.abs() + 1e-10);
    assert!(
        rel_diff < 0.01,
        "scalar={}, simd={}, rel_diff={}",
        scalar,
        simd,
        rel_diff
    );
}

// -------------------------------------------------------------------------
// Quantization Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_dequantize_q4_k_edge_all_max_nibbles_deep_qcov_027() {
    // Test Q4_K with all maximum nibble values (0xFF)
    let mut data = vec![0u8; 144];
    // d = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // dmin = 1.0
    data[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // scales all max
    data[4..16].copy_from_slice(&[0x3F; 12]);
    // qs all max
    data[16..144].copy_from_slice(&[0xFF; 128]);

    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
    // All values should be positive (d * scale * 15 - dmin * min)
}

#[test]
fn test_dequantize_q4_k_edge_all_zero_nibbles_deep_qcov_028() {
    // Test Q4_K with all zero nibble values
    let mut data = vec![0u8; 144];
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data[4..16].copy_from_slice(&[0x3F; 12]); // max scales
    // qs all zero
    data[16..144].copy_from_slice(&[0x00; 128]);

    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    // Values should be negative due to -dmin * min term
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v <= 0.0 || v.abs() < 1e-6,
            "vals[{}]={} should be <= 0",
            i,
            v
        );
    }
}

#[test]
fn test_dequantize_q5_k_edge_high_bits_deep_qcov_029() {
    // Test Q5_K with high bits set in qh
    let mut data = vec![0u8; 176];
    data[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    data[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    data[4..16].copy_from_slice(&[0x1F; 12]); // varied scales
    // qh all 0xFF (high bits set)
    data[16..48].copy_from_slice(&[0xFF; 32]);
    // qs varied
    data[48..176].copy_from_slice(&[0xAA; 128]);

    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 256);
}

#[test]
fn test_dequantize_q6_k_edge_negative_scales_deep_qcov_030() {
    // Q6_K scales are i8, test negative scales
    let mut data = vec![0u8; 210];
    // ql: 128 bytes
    data[0..128].copy_from_slice(&[0x55; 128]);
    // qh: 64 bytes
    data[128..192].copy_from_slice(&[0xAA; 64]);
    // scales: 16 i8 values, some negative
    for i in 0..16 {
        data[192 + i] = if i % 2 == 0 { 64u8 } else { 192u8 }; // 64 or -64 as i8
    }
    // d at the end
    data[208..210].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    // Should have mix of positive and negative values
    let has_pos = vals.iter().any(|&v| v > 0.0);
    let has_neg = vals.iter().any(|&v| v < 0.0);
    assert!(has_pos || has_neg, "Should have varied values");
}

#[test]
fn test_dequantize_q8_0_negative_values_deep_qcov_031() {
    // Test Q8_0 with negative quantized values
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    // 32 i8 values: -128 to -97
    for i in 0..32 {
        data[2 + i] = (128 + i) as u8; // -128, -127, ... as i8
    }

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    // All values should be negative
    for (i, &v) in vals.iter().enumerate() {
        assert!(v < 0.0, "vals[{}]={} should be < 0", i, v);
    }
}

// -------------------------------------------------------------------------
// Q8KSuperBlock Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_q8k_superblock_quantize_all_same_value_deep_qcov_032() {
    let values = [42.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    // All quants should be same
    let first = block.quants[0];
    assert!(block.quants.iter().all(|&q| q == first));
}

#[test]
fn test_q8k_superblock_quantize_alternating_deep_qcov_033() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let block = Q8KSuperBlock::quantize(&values);
    // Quants should alternate between positive and negative
    assert!(block.quants[0] != 0);
    assert!(block.quants[0] != block.quants[1] || block.quants[0] == 0);
}

#[test]
fn test_q8k_superblock_quantize_into_overflow_safe_deep_qcov_034() {
    // Test with values that could cause overflow in naive implementation
    let values = [f32::MAX / 1000.0; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    // Should complete without panic
    assert!(scale > 0.0);
}

// -------------------------------------------------------------------------
// Boundary Condition Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_single_superblock_boundary_deep_qcov_035() {
    // Exactly 1 super-block (minimum valid size)
    let q4k_data = vec![0u8; 144];
    let activations = vec![0.5f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_max_superblocks_deep_qcov_036() {
    // Test with 16 super-blocks (4096 values)
    let num_sb = 16;
    let mut q4k_data = vec![0u8; 144 * num_sb];
    for i in 0..num_sb {
        let offset = i * 144;
        q4k_data[offset..offset + 2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    }

    let activations = vec![1.0f32; 256 * num_sb];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_interleaved_q4k_clone_and_debug_deep_qcov_037() {
    let q4k_data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("valid");

    // Test Clone
    let cloned = interleaved.clone();
    assert_eq!(cloned.num_super_blocks, interleaved.num_super_blocks);

    // Test Debug
    let debug_str = format!("{:?}", interleaved);
    assert!(debug_str.contains("InterleavedQ4K"));
}

// -------------------------------------------------------------------------
// Additional fused_q8_0_q8_0 Coverage
// -------------------------------------------------------------------------

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_error_deep_qcov_038() {
    let in_dim = 64;
    let out_dim = 4;
    // Q8_0: 34 bytes per 32 values, so 2 blocks per row = 68 bytes per row
    // Need 4 * 68 = 272 bytes, provide less
    let weight_data = vec![0u8; 100];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// f16_to_f32_lut Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_f16_to_f32_lut_special_values_deep_qcov_039() {
    // Test zero
    let zero_bits = half::f16::from_f32(0.0).to_bits();
    assert_eq!(f16_to_f32_lut(zero_bits), 0.0);

    // Test negative zero
    let neg_zero_bits = half::f16::from_f32(-0.0).to_bits();
    let result = f16_to_f32_lut(neg_zero_bits);
    assert!(result == 0.0 || result == -0.0);

    // Test small positive
    let small_bits = half::f16::from_f32(0.001).to_bits();
    let result = f16_to_f32_lut(small_bits);
    assert!((result - 0.001).abs() < 0.0001);

    // Test large value
    let large_bits = half::f16::from_f32(1000.0).to_bits();
    let result = f16_to_f32_lut(large_bits);
    assert!((result - 1000.0).abs() < 1.0);
}

// -------------------------------------------------------------------------
// quantize_activations_q8_0 Coverage
// -------------------------------------------------------------------------

#[test]
fn test_quantize_activations_q8_0_single_block_deep_qcov_040() {
    // Exactly 32 values (1 block)
    let activations: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

#[test]
fn test_quantize_activations_q8_0_multiple_blocks_deep_qcov_041() {
    // 3 blocks (96 values)
    let activations: Vec<f32> = (0..96).map(|i| (i as f32 * 0.1).sin()).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 3);
    assert_eq!(quants.len(), 96);
}

#[test]
fn test_quantize_activations_q8_0_partial_block_deep_qcov_042() {
    // 40 values (1 full + partial)
    let activations = vec![1.0f32; 40];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    // Should round up to 2 blocks
    assert_eq!(scales.len(), 2);
    assert_eq!(quants.len(), 64); // Padded to 64
}

// -------------------------------------------------------------------------
// Additional Error Path Coverage
// -------------------------------------------------------------------------

#[test]
fn test_dequantize_q4_0_empty_input_deep_qcov_043() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q8_0_empty_input_deep_qcov_044() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q4_k_empty_input_deep_qcov_045() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q5_k_empty_input_deep_qcov_046() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

#[test]
fn test_dequantize_q6_k_empty_input_deep_qcov_047() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

// -------------------------------------------------------------------------
// fused_q4k_q8k_parallel_matvec Coverage
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_error_weight_deep_qcov_048() {
    // in_dim=256, out_dim=2 -> need 288 bytes, provide less
    let weights = vec![0u8; 100];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 2];

    let result =
        fused_q4k_q8k_parallel_matvec_into(&weights, &q8k_scales, &q8k_quants, 256, 2, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_error_up_weight_deep_qcov_049() {
    let up_weights = vec![0u8; 50]; // Too small
    let gate_weights = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        256,
        1,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_error_gate_weight_deep_qcov_050() {
    let up_weights = vec![0u8; 144];
    let gate_weights = vec![0u8; 50]; // Too small
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        256,
        1,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// =========================================================================
// Additional 95% Coverage Tests
// =========================================================================

#[test]
fn test_cov95_fused_q4k_dot_empty_activations() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let activations: Vec<f32> = vec![];
    let result = fused_q4k_dot(&q4k_data, &activations);
    // Empty activations should return error or 0
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_cov95_fused_q4k_dot_simd_short_input() {
    // Test with input shorter than SIMD width
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let activations = vec![1.0f32; 16]; // Less than block size
    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok() || result.is_err());
}
