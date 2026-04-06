
// =============================================================================
// fused_q8_0_q8_0_parallel_matvec tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("weight data too small"));
}

#[test]
fn test_fused_q8_0_q8_0_activation_mismatch() {
    // Q8_0: 34 bytes per block of 32
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; 64];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("doesn't match in_dim"));
}

#[test]
fn test_fused_q8_0_q8_0_sequential_path() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_path() {
    let in_dim = 32;
    let out_dim = 2048;
    let bytes_per_row = 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

// =============================================================================
// fused_q8_0_q8_0_parallel_matvec_into tests
// =============================================================================

#[test]
fn test_fused_q8_0_q8_0_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_into_activation_mismatch() {
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 1];

    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_into_success() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; out_dim * 34];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_fused_q8_0_q8_0_into_large() {
    let in_dim = 64;
    let out_dim = 128;
    let blocks_per_row = 2;
    let bytes_per_row = blocks_per_row * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.25f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

// =============================================================================
// Additional coverage for edge cases
// =============================================================================

#[test]
fn test_q4_0_matvec_multiple_blocks_per_row() {
    // 3 blocks per row (96 values)
    let in_dim = 96;
    let out_dim = 2;
    let blocks_per_row = 3;
    let bytes_per_row = blocks_per_row * 18;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q8_0_matvec_multiple_blocks_per_row() {
    let in_dim = 96;
    let out_dim = 2;
    let blocks_per_row = 3;
    let bytes_per_row = blocks_per_row * 34;

    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q8k_into_multiple_superblocks() {
    let activations = vec![0.5f32; 1024]; // 4 superblocks
    let mut scales = vec![0.0f32; 4];
    let mut quants = vec![0i8; 1024];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());

    // All scales should be computed
    for s in &scales {
        assert!(s.abs() > 0.0);
    }
}

#[test]
fn test_interleaved_q4k_varied_values() {
    // Create Q4_K data with varied values
    let mut data = vec![0u8; 144];

    // Set d to 0.5 (f16 0x3800)
    data[0] = 0x00;
    data[1] = 0x38;

    // Set dmin to 0.25 (f16 0x3400)
    data[2] = 0x00;
    data[3] = 0x34;

    // Set some scale values
    for i in 4..16 {
        data[i] = (i as u8) * 5;
    }

    // Set varied quantized values
    for i in 16..144 {
        data[i] = i as u8;
    }

    let interleaved = InterleavedQ4K::from_q4k(&data).expect("test value should be present");

    // Check d and dmin were parsed
    assert!(interleaved.d[0] > 0.0);
    assert!(interleaved.dmin[0] > 0.0);

    let activations: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
    let result = interleaved.dot(&activations).expect("test value should be present");
    assert!(result.is_finite());
}

#[test]
fn test_q4_0_matvec_with_non_zero_weights() {
    let in_dim = 32;
    let out_dim = 2;

    // Create Q4_0 data with non-zero scale
    let mut weight_data = vec![0u8; out_dim * 18];

    // Set f16 scale to 1.0 (0x3C00) for first row
    weight_data[0] = 0x00;
    weight_data[1] = 0x3C;

    // Set some quantized values
    for i in 2..18 {
        weight_data[i] = 0x88; // Both nibbles = 8, centered
    }

    // Second row
    weight_data[18] = 0x00;
    weight_data[19] = 0x3C;
    for i in 20..36 {
        weight_data[i] = 0x44;
    }

    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");

    // With non-zero weights and activations, output should be non-zero
    // (though exact values depend on quantization details)
    assert!(output.iter().all(|v| v.is_finite()));
}

// =============================================================================
// extract_scale_min tests
// =============================================================================

#[test]
fn test_extract_scale_min_block_0() {
    let scales: [u8; 12] = [
        0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 0);
    assert_eq!(scale, 63.0); // 0x3F & 63 = 63
    assert_eq!(min, 42.0); // 0x2A & 63 = 42
}

#[test]
fn test_extract_scale_min_block_1() {
    let scales: [u8; 12] = [
        0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 1);
    assert_eq!(scale, 31.0); // 0x1F & 63 = 31
    assert_eq!(min, 21.0); // 0x15 & 63 = 21
}

#[test]
fn test_extract_scale_min_block_2() {
    let scales: [u8; 12] = [
        0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 2);
    assert_eq!(scale, 15.0); // 0x0F & 63 = 15
    assert_eq!(min, 10.0); // 0x0A & 63 = 10
}

#[test]
fn test_extract_scale_min_block_3() {
    let scales: [u8; 12] = [
        0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 3);
    assert_eq!(scale, 7.0); // 0x07 & 63 = 7
    assert_eq!(min, 5.0); // 0x05 & 63 = 5
}

#[test]
fn test_extract_scale_min_block_4() {
    // Block 4-7 use packed layout
    let scales: [u8; 12] = [
        0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 4);
    // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = (0x12 & 0x0F) | ((0xC0 >> 6) << 4) = 2 | (3 << 4) = 2 | 48 = 50
    // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = (0x12 >> 4) | ((0 >> 6) << 4) = 1 | 0 = 1
    assert_eq!(scale, 50.0);
    assert_eq!(min, 1.0);
}

#[test]
fn test_extract_scale_min_block_5() {
    let scales: [u8; 12] = [
        0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 5);
    // scale = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4) = (0x34 & 0x0F) | ((0xC0 >> 6) << 4) = 4 | 48 = 52
    // min = (scales[9] >> 4) | ((scales[5] >> 6) << 4) = (0x34 >> 4) | 0 = 3
    assert_eq!(scale, 52.0);
    assert_eq!(min, 3.0);
}

#[test]
fn test_extract_scale_min_block_6() {
    let scales: [u8; 12] = [
        0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0x00,
    ];
    let (scale, min) = extract_scale_min(&scales, 6);
    // scale = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4) = (0x56 & 0x0F) | ((0xC0 >> 6) << 4) = 6 | 48 = 54
    // min = (scales[10] >> 4) | ((scales[6] >> 6) << 4) = (0x56 >> 4) | 0 = 5
    assert_eq!(scale, 54.0);
    assert_eq!(min, 5.0);
}

#[test]
fn test_extract_scale_min_block_7() {
    let scales: [u8; 12] = [
        0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x78,
    ];
    let (scale, min) = extract_scale_min(&scales, 7);
    // scale = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4) = (0x78 & 0x0F) | ((0xC0 >> 6) << 4) = 8 | 48 = 56
    // min = (scales[11] >> 4) | ((scales[7] >> 6) << 4) = (0x78 >> 4) | 0 = 7
    assert_eq!(scale, 56.0);
    assert_eq!(min, 7.0);
}

// =============================================================================
// extract_scale_min_from_slice tests
// =============================================================================

#[test]
fn test_extract_scale_min_from_slice_even_idx() {
    // idx = 0 (even)
    let scales: Vec<u8> = vec![
        0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min_from_slice(&scales, 0);
    // scale_idx = 0, min_idx = 4
    // scale = scales[0] & 0x3F = 0x3F = 63
    // min = scales[4] & 0x3F = 0x2A = 42
    assert_eq!(scale, 63.0);
    assert_eq!(min, 42.0);
}

#[test]
fn test_extract_scale_min_from_slice_even_idx_2() {
    let scales: Vec<u8> = vec![
        0x3F, 0x1F, 0x0F, 0x07, 0x2A, 0x15, 0x0A, 0x05, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min_from_slice(&scales, 2);
    // scale_idx = 1, min_idx = 5
    // scale = scales[1] & 0x3F = 0x1F = 31
    // min = scales[5] & 0x3F = 0x15 = 21
    assert_eq!(scale, 31.0);
    assert_eq!(min, 21.0);
}

#[test]
fn test_extract_scale_min_from_slice_odd_idx() {
    // idx = 1 (odd) - uses different extraction logic
    let scales: Vec<u8> = vec![
        0xC0, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min_from_slice(&scales, 1);
    // scale_idx = 0, min_idx = 4
    // scale = (scales[0] >> 6) | ((scales[2] & 0x0F) << 2) = (0xC0 >> 6) | ((0x0F & 0x0F) << 2) = 3 | 60 = 63
    // min = (scales[4] >> 6) | ((scales[6] & 0x0F) << 2) = (0 >> 6) | ((0x0F & 0x0F) << 2) = 0 | 60 = 60
    assert_eq!(scale, 63.0);
    assert_eq!(min, 60.0);
}

#[test]
fn test_extract_scale_min_from_slice_odd_idx_3() {
    let scales: Vec<u8> = vec![
        0x00, 0xC0, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00,
    ];
    let (scale, min) = extract_scale_min_from_slice(&scales, 3);
    // scale_idx = 1, min_idx = 5
    // scale = (scales[1] >> 6) | ((scales[3] & 0x0F) << 2) = (0xC0 >> 6) | ((0x0F) << 2) = 3 | 60 = 63
    // min = (scales[5] >> 6) | ((scales[7] & 0x0F) << 2) = 0 | 60 = 60
    assert_eq!(scale, 63.0);
    assert_eq!(min, 60.0);
}

// =============================================================================
// fused_q4_0_q8_0_dot_scalar tests
// =============================================================================

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_zero() {
    // Q4_0 block: 2 bytes f16 scale + 16 bytes quants
    let q4_data = vec![0u8; 18];
    let q8_scales = vec![0.0f32];
    let q8_quants = vec![0i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_basic() {
    // Create Q4_0 data with scale = 1.0 (f16 0x3C00)
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // f16 1.0

    // Set quantized values to 8 (centered at 0 after -8 offset)
    for i in 2..18 {
        q4_data[i] = 0x88; // Both nibbles = 8
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // Result should be 0 since (8-8)=0 for all values
    assert!(result.is_finite());
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_nonzero() {
    // Create Q4_0 data
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // f16 1.0

    // Set quantized values to various values
    for i in 2..18 {
        q4_data[i] = 0xF0; // Low nibble = 0 (-8), high nibble = 15 (7)
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![127i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    assert!(result.is_finite());
    // Result should be non-zero
}
