//! T-COV-95 Coverage Bridge: fused Q4_0/Q8_0 matvec functions
//!
//! Targets: fused_q4_0_q8_0_parallel_matvec, fused_q4_0_q8_0_parallel_matvec_into,
//! fused_q8_0_q8_0_parallel_matvec, fused_q8_0_q8_0_parallel_matvec_into,
//! extract_scale_min for blocks 4-7.

use crate::quantize::*;

// ============================================================================
// fused_q4_0_q8_0_parallel_matvec tests
// ============================================================================

#[test]
fn test_q4_0_q8_0_matvec_valid_small() {
    // Q4_0: 18 bytes per block of 32 values
    // For in_dim=32, out_dim=2: 2 rows × 1 block × 18 bytes = 36 bytes
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 18; // 1 block
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4_0_q8_0_matvec_weight_too_small() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small") || err.contains("need"),
        "got: {err}"
    );
}

#[test]
fn test_q4_0_q8_0_matvec_activation_mismatch() {
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; 16]; // Wrong size

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("match") || err.contains("length"),
        "got: {err}"
    );
}

#[test]
fn test_q4_0_q8_0_matvec_large_parallel() {
    // Test the parallel path (out_dim >= 1024)
    let in_dim = 32;
    let out_dim = 1024;
    let bytes_per_row = 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4_0_q8_0_matvec_multi_block() {
    // in_dim=64 requires 2 blocks per row
    let in_dim = 64;
    let out_dim = 4;
    let blocks_per_row = 2;
    let bytes_per_row = blocks_per_row * 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

// ============================================================================
// fused_q4_0_q8_0_parallel_matvec_into tests
// ============================================================================

#[test]
fn test_q4_0_q8_0_matvec_into_valid() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q4_0_q8_0_matvec_into_weight_too_small() {
    let in_dim = 32;
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 4];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_q4_0_q8_0_matvec_into_activation_mismatch() {
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; 16]; // Wrong
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4_0_q8_0_parallel_matvec_into(&weight_data, &activations, in_dim, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// fused_q8_0_q8_0_parallel_matvec tests
// ============================================================================

#[test]
fn test_q8_0_q8_0_matvec_valid_small() {
    // Q8_0: 34 bytes per block of 32 values
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 34; // 1 block
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q8_0_q8_0_matvec_weight_too_small() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; 20]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small") || err.contains("need"),
        "got: {err}"
    );
}

#[test]
fn test_q8_0_q8_0_matvec_activation_mismatch() {
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; 16]; // Wrong size

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_q8_0_q8_0_matvec_multi_block() {
    // in_dim=64 requires 2 blocks per row
    let in_dim = 64;
    let out_dim = 4;
    let blocks_per_row = 2;
    let bytes_per_row = blocks_per_row * 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

// ============================================================================
// fused_q8_0_q8_0_parallel_matvec_into tests
// ============================================================================

#[test]
fn test_q8_0_q8_0_matvec_into_valid() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
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
fn test_q8_0_q8_0_matvec_into_weight_too_small() {
    let in_dim = 32;
    let out_dim = 4;
    let weight_data = vec![0u8; 20]; // Too small
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

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
fn test_q8_0_q8_0_matvec_into_activation_mismatch() {
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; 16]; // Wrong
    let mut output = vec![0.0f32; out_dim];

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
fn test_q8_0_q8_0_matvec_into_output_too_small() {
    let in_dim = 32;
    let out_dim = 4;
    let bytes_per_row = 34;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 2]; // Too small

    let result = fused_q8_0_q8_0_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small") || err.contains("need"),
        "got: {err}"
    );
}

// ============================================================================
// extract_scale_min coverage for blocks 4-7
// ============================================================================

#[test]
fn test_extract_scale_min_block_0() {
    let scales: [u8; 12] = [0x3F, 0x20, 0x10, 0x08, 0x3E, 0x1F, 0x0F, 0x07, 0, 0, 0, 0];
    let (scale, min) = extract_scale_min(&scales, 0);
    assert_eq!(scale, 63.0); // 0x3F & 63 = 63
    assert_eq!(min, 62.0); // 0x3E & 63 = 62
}

#[test]
fn test_extract_scale_min_block_1() {
    let scales: [u8; 12] = [0x3F, 0x20, 0x10, 0x08, 0x3E, 0x1F, 0x0F, 0x07, 0, 0, 0, 0];
    let (scale, min) = extract_scale_min(&scales, 1);
    assert_eq!(scale, 32.0); // 0x20 & 63 = 32
    assert_eq!(min, 31.0); // 0x1F & 63 = 31
}

#[test]
fn test_extract_scale_min_block_4() {
    // Block 4 uses packed layout
    let mut scales: [u8; 12] = [0; 12];
    // For block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    // m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    scales[0] = 0xC0; // high 2 bits = 3
    scales[4] = 0x80; // high 2 bits = 2
    scales[8] = 0x52; // low 4 bits = 2, high 4 bits = 5
    let (scale, min) = extract_scale_min(&scales, 4);
    // d = (0x52 & 0x0F) | ((0xC0 >> 6) << 4) = 2 | (3 << 4) = 2 | 48 = 50
    assert_eq!(scale, 50.0);
    // m = (0x52 >> 4) | ((0x80 >> 6) << 4) = 5 | (2 << 4) = 5 | 32 = 37
    assert_eq!(min, 37.0);
}

#[test]
fn test_extract_scale_min_block_5() {
    let mut scales: [u8; 12] = [0; 12];
    scales[1] = 0x40; // high 2 bits = 1
    scales[5] = 0xC0; // high 2 bits = 3
    scales[9] = 0x31; // low 4 bits = 1, high 4 bits = 3
    let (scale, min) = extract_scale_min(&scales, 5);
    // d = (0x31 & 0x0F) | ((0x40 >> 6) << 4) = 1 | (1 << 4) = 1 | 16 = 17
    assert_eq!(scale, 17.0);
    // m = (0x31 >> 4) | ((0xC0 >> 6) << 4) = 3 | (3 << 4) = 3 | 48 = 51
    assert_eq!(min, 51.0);
}

#[test]
fn test_extract_scale_min_block_6() {
    let mut scales: [u8; 12] = [0; 12];
    scales[2] = 0x80; // high 2 bits = 2
    scales[6] = 0x40; // high 2 bits = 1
    scales[10] = 0xAB; // low 4 bits = 11, high 4 bits = 10
    let (scale, min) = extract_scale_min(&scales, 6);
    // d = (0xAB & 0x0F) | ((0x80 >> 6) << 4) = 11 | (2 << 4) = 11 | 32 = 43
    assert_eq!(scale, 43.0);
    // m = (0xAB >> 4) | ((0x40 >> 6) << 4) = 10 | (1 << 4) = 10 | 16 = 26
    assert_eq!(min, 26.0);
}

#[test]
fn test_extract_scale_min_block_7() {
    let mut scales: [u8; 12] = [0; 12];
    scales[3] = 0xC0; // high 2 bits = 3
    scales[7] = 0x80; // high 2 bits = 2
    scales[11] = 0xFF; // low 4 bits = 15, high 4 bits = 15
    let (scale, min) = extract_scale_min(&scales, 7);
    // d = (0xFF & 0x0F) | ((0xC0 >> 6) << 4) = 15 | (3 << 4) = 15 | 48 = 63
    assert_eq!(scale, 63.0);
    // m = (0xFF >> 4) | ((0x80 >> 6) << 4) = 15 | (2 << 4) = 15 | 32 = 47
    assert_eq!(min, 47.0);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_q4_0_q8_0_matvec_with_nonzero_weights() {
    // Create valid Q4_0 data with non-zero scale
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 18;
    let mut weight_data = vec![0u8; out_dim * bytes_per_row];

    // Set f16 scale = 1.0 (0x3C00) for first row
    weight_data[0] = 0x00;
    weight_data[1] = 0x3C;
    // Set some quant values
    for i in 2..18 {
        weight_data[i] = 0x88; // -8 + 8 = 0, 8 - 8 = 0 after offset
    }

    let activations = vec![1.0f32; in_dim];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q8_0_q8_0_matvec_with_nonzero_weights() {
    // Create valid Q8_0 data with non-zero scale
    let in_dim = 32;
    let out_dim = 2;
    let bytes_per_row = 34;
    let mut weight_data = vec![0u8; out_dim * bytes_per_row];

    // Set f16 scale = 1.0 (0x3C00) for first row
    weight_data[0] = 0x00;
    weight_data[1] = 0x3C;
    // Set some quant values (i8)
    for i in 2..34 {
        weight_data[i] = 10; // All positive
    }

    let activations = vec![1.0f32; in_dim];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q4_0_matvec_sequential_threshold() {
    // Test at exactly the parallel threshold boundary (1024)
    let in_dim = 32;
    let out_dim = 1023; // Just below threshold - sequential
    let bytes_per_row = 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

#[test]
fn test_q4_0_matvec_at_threshold() {
    // Test at exactly the parallel threshold (1024)
    let in_dim = 32;
    let out_dim = 1024; // Exactly at threshold - parallel
    let bytes_per_row = 18;
    let weight_data = vec![0u8; out_dim * bytes_per_row];
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

#[test]
fn test_extract_scale_min_all_zeros() {
    let scales: [u8; 12] = [0; 12];
    for block in 0..8 {
        let (scale, min) = extract_scale_min(&scales, block);
        assert_eq!(scale, 0.0);
        assert_eq!(min, 0.0);
    }
}

#[test]
fn test_extract_scale_min_all_max() {
    let scales: [u8; 12] = [0xFF; 12];
    // Block 0: scale = 0xFF & 63 = 63, min = 0xFF & 63 = 63
    let (scale, min) = extract_scale_min(&scales, 0);
    assert_eq!(scale, 63.0);
    assert_eq!(min, 63.0);

    // Block 4: d = (0xFF & 0x0F) | ((0xFF >> 6) << 4) = 15 | 48 = 63
    // m = (0xFF >> 4) | ((0xFF >> 6) << 4) = 15 | 48 = 63
    let (scale4, min4) = extract_scale_min(&scales, 4);
    assert_eq!(scale4, 63.0);
    assert_eq!(min4, 63.0);
}
