//! Generic Parallel Matrix-Vector Multiplication (Contract: quantized-dot-product-v1.yaml)
//!
//! Replaces ~300 lines of duplicated parallel matvec code (Q4_K, Q5_K, Q6_K variants)
//! with a single generic implementation parameterized by `QuantBlockFormat`.
//!
//! ## Design
//!
//! The generic matvec delegates to format-specific SIMD dot products for the inner
//! loop (those remain hand-optimized). The outer loop — validation, padding,
//! parallel dispatch, tiling — is format-independent and encoded here once.
//!
//! ## Tuning Constants (from existing code)
//!
//! - `PARALLEL_THRESHOLD = 256` — below this, use sequential path (PAR-126)
//! - `MIDI_TILE_M = 64` — TCB-style midi-tile for L2 cache reuse

use super::format_trait::QuantBlockFormat;
use crate::error::{RealizarError, Result};
use std::borrow::Cow;

/// Parallel threshold: use sequential path below this out_dim (PAR-126)
const PARALLEL_THRESHOLD: usize = 256;

/// TCB-style midi-tile size for parallel chunking (L2 cache reuse)
const MIDI_TILE_M: usize = 64;

/// Pad activations to super-block boundary when `in_dim % elements_per_superblock != 0`.
///
/// GH-202 FIX: Quantized weights are stored with per-row padding to super-block
/// boundaries. The fused dot kernels expect activations to match the padded length.
#[inline]
fn pad_activations_generic(activations: &[f32], padded_len: usize) -> Cow<'_, [f32]> {
    if activations.len() == padded_len {
        Cow::Borrowed(activations)
    } else {
        let mut padded = vec![0.0f32; padded_len];
        padded[..activations.len()].copy_from_slice(activations);
        Cow::Owned(padded)
    }
}

/// Type alias for format-specific SIMD dot product function
///
/// The generic matvec calls this per-row. Each format provides its own
/// optimized SIMD implementation (or scalar fallback).
pub type FusedDotFn = fn(&[u8], &[f32]) -> Result<f32>;

/// Generic parallel fused matrix-vector multiply for any blocked quantization format.
///
/// Writes results into a pre-allocated output buffer (zero-allocation hot path).
///
/// # Arguments
///
/// * `weight_data` - Raw quantized weight data, row-major [out_dim × bytes_per_row]
/// * `activations` - Input activations [in_dim]
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
/// * `output` - Pre-allocated output buffer [out_dim]
/// * `dot_fn` - Format-specific SIMD dot product function
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
/// - Output buffer length is less than out_dim
#[allow(clippy::similar_names)]
pub fn generic_parallel_matvec_into<F: QuantBlockFormat>(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
    dot_fn: FusedDotFn,
) -> Result<()> {
    let super_blocks_per_row = in_dim.div_ceil(F::ELEMENTS_PER_SUPERBLOCK);
    let bytes_per_row = super_blocks_per_row * F::SUPERBLOCK_BYTES;

    // Validate weight data size
    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{} weight data too small: need {} bytes for {}x{}, have {}",
                F::FORMAT_ID,
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    // Validate activation length
    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    // Validate output buffer
    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    // GH-202 FIX: Pad activations to super-block boundary
    let padded_in_dim = super_blocks_per_row * F::ELEMENTS_PER_SUPERBLOCK;
    let acts = pad_activations_generic(activations, padded_in_dim);

    if out_dim < PARALLEL_THRESHOLD {
        // Sequential path: avoids rayon overhead for small matrices
        for o in 0..out_dim {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];
            output[o] = dot_fn(row_data, &acts).unwrap_or(0.0);
        }
    } else {
        // Parallel path with TCB-style midi-tile chunking
        use rayon::prelude::*;

        output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .enumerate()
            .for_each(|(midi_idx, midi_chunk)| {
                let midi_start = midi_idx * MIDI_TILE_M;

                for (local_idx, out) in midi_chunk.iter_mut().enumerate() {
                    let row = midi_start + local_idx;
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &weight_data[row_start..row_end];
                    *out = dot_fn(row_data, &acts).unwrap_or(0.0);
                }
            });
    }

    Ok(())
}

/// Generic parallel fused matrix-vector multiply (allocating variant).
///
/// Convenience wrapper that allocates the output buffer.
///
/// # Errors
///
/// Same as `generic_parallel_matvec_into`.
#[allow(clippy::similar_names)]
pub fn generic_parallel_matvec<F: QuantBlockFormat>(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    dot_fn: FusedDotFn,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; out_dim];
    generic_parallel_matvec_into::<F>(
        weight_data,
        activations,
        in_dim,
        out_dim,
        &mut output,
        dot_fn,
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::format_trait::{Q4K, Q6K};
    use crate::quantize::generic_dot::generic_fused_dot_scalar;

    /// Helper: create Q4_K weight data (zeros with valid structure)
    fn create_q4k_test_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 144;
        vec![0u8; out_dim * bytes_per_row]
    }

    /// Scalar Q4K dot wrapper for testing
    fn q4k_scalar_dot(data: &[u8], acts: &[f32]) -> Result<f32> {
        generic_fused_dot_scalar::<Q4K>(data, acts)
    }

    /// Scalar Q6K dot wrapper for testing
    fn q6k_scalar_dot(data: &[u8], acts: &[f32]) -> Result<f32> {
        generic_fused_dot_scalar::<Q6K>(data, acts)
    }

    #[test]
    fn test_generic_matvec_q4k_basic() {
        let in_dim = 256;
        let out_dim = 64;
        let weights = create_q4k_test_weights(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = generic_parallel_matvec_into::<Q4K>(
            &weights,
            &acts,
            in_dim,
            out_dim,
            &mut output,
            q4k_scalar_dot,
        );
        assert!(result.is_ok());
        // All zero weights → all zero outputs
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_generic_matvec_q6k_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 32;
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 210;
        let weights = vec![0u8; out_dim * bytes_per_row];
        let acts = vec![1.0f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = generic_parallel_matvec_into::<Q6K>(
            &weights,
            &acts,
            in_dim,
            out_dim,
            &mut output,
            q6k_scalar_dot,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_generic_matvec_weight_too_small() {
        let weights = vec![0u8; 100]; // Too small
        let acts = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 64];

        let result = generic_parallel_matvec_into::<Q4K>(
            &weights,
            &acts,
            256,
            64,
            &mut output,
            q4k_scalar_dot,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_generic_matvec_activation_mismatch() {
        let weights = create_q4k_test_weights(64, 256);
        let acts = vec![1.0f32; 128]; // Wrong length
        let mut output = vec![0.0f32; 64];

        let result = generic_parallel_matvec_into::<Q4K>(
            &weights,
            &acts,
            256,
            64,
            &mut output,
            q4k_scalar_dot,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_generic_matvec_output_too_small() {
        let weights = create_q4k_test_weights(64, 256);
        let acts = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 32]; // Too small

        let result = generic_parallel_matvec_into::<Q4K>(
            &weights,
            &acts,
            256,
            64,
            &mut output,
            q4k_scalar_dot,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_generic_matvec_allocating_variant() {
        let weights = create_q4k_test_weights(64, 256);
        let acts = vec![1.0f32; 256];

        let result = generic_parallel_matvec::<Q4K>(&weights, &acts, 256, 64, q4k_scalar_dot);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed").len(), 64);
    }

    #[test]
    fn test_generic_matvec_parallel_threshold() {
        // Above threshold (512 > 256) — takes parallel path
        let in_dim = 256;
        let out_dim = 512;
        let weights = create_q4k_test_weights(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = generic_parallel_matvec_into::<Q4K>(
            &weights,
            &acts,
            in_dim,
            out_dim,
            &mut output,
            q4k_scalar_dot,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_generic_matvec_padding() {
        // in_dim not a multiple of 256 — requires padding (GH-202)
        let in_dim: usize = 200;
        let out_dim: usize = 16;
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 144;
        let weights = vec![0u8; out_dim * bytes_per_row];
        let acts = vec![1.0f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = generic_parallel_matvec_into::<Q4K>(
            &weights,
            &acts,
            in_dim,
            out_dim,
            &mut output,
            q4k_scalar_dot,
        );
        assert!(result.is_ok());
    }
}
