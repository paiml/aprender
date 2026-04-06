//! Phase 39: Parallel K-quantization coverage tests for `parallel_k.rs`
//!
//! Tests cover:
//! - `fused_q4k_tiled_matvec` - L2-aware tiled matmul
//! - `fused_q4k_parallel_matvec` / `_into` - Parallel Q4_K
//! - `fused_q5k_parallel_matvec` / `_into` - Parallel Q5_K
//! - `fused_q6k_parallel_matvec` / `_into` - Parallel Q6_K
//! - `fused_q4k_q8k_parallel_matvec_into` - Q4_K x Q8_K with TCB tiling
//! - `fused_q4k_q8k_ffn_up_gate_into` - Fused FFN up+gate
//! - Backward-compat aliases
//!
//! Focus areas:
//! - Sequential vs parallel paths (PARALLEL_THRESHOLD = 256)
//! - TCB midi-tile (64 rows) and micro-tile (4 rows) chunking
//! - Edge cases: single row, tile boundaries, partial tiles
//! - Error handling: invalid dimensions, buffer sizes

// LAYOUT-002: ROW-MAJOR ONLY - no colmajor/auto aliases
use crate::quantize::parallel_k::{
    fused_q4k_parallel_matvec, fused_q4k_parallel_matvec_into, fused_q4k_q8k_ffn_up_gate_into,
    fused_q4k_q8k_parallel_matvec_into, fused_q4k_tiled_matvec, fused_q5k_parallel_matvec,
    fused_q5k_parallel_matvec_into, fused_q6k_parallel_matvec, fused_q6k_parallel_matvec_into,
};
use crate::quantize::types::QK_K;

// ============================================================================
// Helper functions for generating valid quantized weight data
// ============================================================================

/// Generates valid Q4_K weight data for testing.
/// Q4_K format: 144 bytes per super-block (256 elements).
fn generate_q4k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144;
    let total_bytes = out_dim * bytes_per_row;

    let mut data = vec![0u8; total_bytes];

    // Fill with deterministic pattern that produces valid Q4_K blocks
    for (i, byte) in data.iter_mut().enumerate() {
        // Create semi-random but deterministic pattern
        *byte = ((i * 17 + 31) % 256) as u8;
    }

    // Set valid scale values (d, dmin) at appropriate offsets in each block
    // Q4_K layout: d (2 bytes), dmin (2 bytes), scales (12 bytes), qs (128 bytes)
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let block_start = row * bytes_per_row + sb * 144;

            // Set d as small positive value (half-precision)
            // 0x3c00 = 1.0 in FP16
            data[block_start] = 0x00;
            data[block_start + 1] = 0x3c;

            // Set dmin as small positive value
            data[block_start + 2] = 0x00;
            data[block_start + 3] = 0x38; // ~0.5 in FP16
        }
    }

    data
}

/// Generates valid Q5_K weight data for testing.
/// Q5_K format: 176 bytes per super-block (256 elements).
fn generate_q5k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 176;
    let total_bytes = out_dim * bytes_per_row;

    let mut data = vec![0u8; total_bytes];

    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i * 19 + 37) % 256) as u8;
    }

    // Set valid scale values for Q5_K
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let block_start = row * bytes_per_row + sb * 176;
            data[block_start] = 0x00;
            data[block_start + 1] = 0x3c;
            data[block_start + 2] = 0x00;
            data[block_start + 3] = 0x38;
        }
    }

    data
}

/// Generates valid Q6_K weight data for testing.
/// Q6_K format: 210 bytes per super-block (256 elements).
fn generate_q6k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 210;
    let total_bytes = out_dim * bytes_per_row;

    let mut data = vec![0u8; total_bytes];

    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i * 23 + 41) % 256) as u8;
    }

    // Set valid scale value for Q6_K (d at offset 208)
    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let block_start = row * bytes_per_row + sb * 210;
            // d at offset 208 (2 bytes FP16)
            data[block_start + 208] = 0x00;
            data[block_start + 209] = 0x3c;
        }
    }

    data
}

/// Generates Q8_K quantized activations for testing.
fn generate_q8k_activations(in_dim: usize) -> (Vec<f32>, Vec<i8>) {
    let super_blocks = in_dim.div_ceil(QK_K);
    let scales: Vec<f32> = (0..super_blocks).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let quants: Vec<i8> = (0..in_dim)
        .map(|i| ((i % 256) as i8).wrapping_sub(64))
        .collect();
    (scales, quants)
}

// ============================================================================
// Tests for fused_q4k_tiled_matvec
// ============================================================================

#[test]
fn test_q4k_tiled_matvec_single_row_pk14() {
    let in_dim = 256; // 1 super-block
    let out_dim = 1;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
    // Should produce a finite value
    assert!(output[0].is_finite());
}

#[test]
fn test_q4k_tiled_matvec_multiple_rows_pk14() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 32;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_tiled_matvec_custom_tile_size_pk14() {
    let in_dim = 256;
    let out_dim = 128;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];

    // Custom tile size = 16
    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(16));
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_partial_last_tile_pk14() {
    let in_dim = 256;
    let out_dim = 100; // Not a multiple of default tile size (64)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_many_tiles_pk14() {
    let in_dim = 256;
    let out_dim = 256; // 4 tiles of size 64
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.1f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_weight_too_small_pk14() {
    let in_dim = 256;
    let out_dim = 10;
    let weights = vec![0u8; 100]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("weight data too small"));
}

#[test]
fn test_q4k_tiled_matvec_activation_mismatch_pk14() {
    let in_dim = 256;
    let out_dim = 10;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 128]; // Wrong size

    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("doesn't match in_dim"));
}

#[test]
fn test_q4k_tiled_matvec_tile_size_1_pk14() {
    let in_dim = 256;
    let out_dim = 8;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    // Extreme: tile_size = 1
    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(1));
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_tiled_matvec_tile_larger_than_out_pk14() {
    let in_dim = 256;
    let out_dim = 10;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    // tile_size > out_dim - should still work (single partial tile)
    let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(128));
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

// ============================================================================
// Tests for fused_q4k_parallel_matvec (sequential path: out_dim < 256)
// ============================================================================

#[test]
fn test_q4k_parallel_matvec_sequential_path_pk14() {
    let in_dim = 256;
    let out_dim = 128; // < 256 threshold, uses sequential path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_parallel_matvec_at_threshold_pk14() {
    let in_dim = 256;
    let out_dim = 256; // Exactly at threshold, uses parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_parallel_path_pk14() {
    let in_dim = 512;
    let out_dim = 512; // > 256 threshold, uses parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.1f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), 1);
}

#[test]
fn test_q4k_parallel_matvec_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

#[test]
fn test_q4k_parallel_matvec_activation_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 64]; // Wrong size

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q4k_parallel_matvec_into
// ============================================================================

#[test]
fn test_q4k_parallel_matvec_into_sequential_pk14() {
    let in_dim = 256;
    let out_dim = 128; // Sequential path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_parallel_matvec_into_parallel_pk14() {
    let in_dim = 256;
    let out_dim = 512; // Parallel path with midi-tiles
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_into_midi_tile_boundary_pk14() {
    let in_dim = 256;
    let out_dim = 64; // Exactly one midi-tile (MIDI_TILE_M = 64)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    // But out_dim < 256, so this uses sequential path
    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q4k_parallel_matvec_into_partial_midi_tile_pk14() {
    let in_dim = 256;
    let out_dim = 300; // 4 midi-tiles + 44 remainder (300 = 64*4 + 44)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_parallel_matvec_into_output_buffer_too_small_pk14() {
    let in_dim = 256;
    let out_dim = 128;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 64]; // Too small

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Output buffer too small"));
}

include!("q4k_parallel.rs");
include!("q4k_q8k.rs");
