//! Fused Gate+Up Kernel for SwiGLU FFN (PMAT-FFN-FUSION)
//!
//! Eliminates 50% of rayon dispatches in the FFN block by computing both gate and up
//! projections in a single parallel region. Activations are loaded once per midi-tile,
//! giving L1/L2 cache reuse.
//!
//! ## Design
//!
//! Same outer loop structure as `generic_parallel_matvec_into` (validation, padding,
//! parallel dispatch, tiling), but processes TWO weight matrices per thread.
//!
//! ## Performance
//!
//! - Before: 2 rayon spawns per layer × 28 layers = 56 dispatches/token (gate+up)
//! - After: 1 rayon spawn per layer × 28 layers = 28 dispatches/token
//! - L1/L2 reuse: activation vector (16KB for 4096 hidden) loaded once, not twice

use super::format_trait::QuantBlockFormat;
use super::generic_matvec::FusedDotFn;
use crate::error::{RealizarError, Result};
use std::borrow::Cow;

/// Parallel threshold: use sequential path below this out_dim (PAR-126)
const PARALLEL_THRESHOLD: usize = 256;

/// TCB-style midi-tile size for parallel chunking (L2 cache reuse)
const MIDI_TILE_M: usize = 64;

/// Pad activations to super-block boundary when `in_dim % elements_per_superblock != 0`.
#[inline]
fn pad_activations(activations: &[f32], padded_len: usize) -> Cow<'_, [f32]> {
    if activations.len() == padded_len {
        Cow::Borrowed(activations)
    } else {
        let mut padded = vec![0.0f32; padded_len];
        padded[..activations.len()].copy_from_slice(activations);
        Cow::Owned(padded)
    }
}

/// Generic fused gate+up matrix-vector multiply for any blocked quantization format.
///
/// Computes BOTH gate and up projections in a single parallel region, halving
/// rayon dispatch overhead and improving cache utilization.
///
/// # Arguments
///
/// * `gate_weight_data` - Raw quantized weight data for gate projection [out_dim × bytes_per_row]
/// * `up_weight_data` - Raw quantized weight data for up projection [out_dim × bytes_per_row]
/// * `activations` - Input activations [in_dim]
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension (same for both gate and up)
/// * `gate_output` - Pre-allocated output buffer for gate projection [out_dim]
/// * `up_output` - Pre-allocated output buffer for up projection [out_dim]
/// * `dot_fn` - Format-specific SIMD dot product function
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Output buffer length is less than out_dim
#[allow(clippy::too_many_arguments)]
pub fn generic_fused_gate_up_matvec_into<F: QuantBlockFormat>(
    gate_weight_data: &[u8],
    up_weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    gate_output: &mut [f32],
    up_output: &mut [f32],
    dot_fn: FusedDotFn,
) -> Result<()> {
    let super_blocks_per_row = in_dim.div_ceil(F::ELEMENTS_PER_SUPERBLOCK);
    let bytes_per_row = super_blocks_per_row * F::SUPERBLOCK_BYTES;

    // Validate weight data sizes
    let expected_weight_bytes = out_dim * bytes_per_row;
    if gate_weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{} gate weight data too small: need {} bytes for {}x{}, have {}",
                F::FORMAT_ID,
                expected_weight_bytes,
                out_dim,
                in_dim,
                gate_weight_data.len()
            ),
        });
    }
    if up_weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{} up weight data too small: need {} bytes for {}x{}, have {}",
                F::FORMAT_ID,
                expected_weight_bytes,
                out_dim,
                in_dim,
                up_weight_data.len()
            ),
        });
    }

    // Validate output buffers
    if gate_output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Gate output buffer too small: need {}, have {}",
                out_dim,
                gate_output.len()
            ),
        });
    }
    if up_output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Up output buffer too small: need {}, have {}",
                out_dim,
                up_output.len()
            ),
        });
    }

    // GH-202 FIX: Pad activations to super-block boundary
    let padded_in_dim = super_blocks_per_row * F::ELEMENTS_PER_SUPERBLOCK;
    let acts = pad_activations(activations, padded_in_dim);

    if out_dim < PARALLEL_THRESHOLD {
        // Sequential path: avoids rayon overhead for small matrices
        for o in 0..out_dim {
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;

            let gate_row = &gate_weight_data[row_start..row_end];
            gate_output[o] = dot_fn(gate_row, &acts).unwrap_or(0.0);

            let up_row = &up_weight_data[row_start..row_end];
            up_output[o] = dot_fn(up_row, &acts).unwrap_or(0.0);
        }
    } else {
        // Parallel path: SINGLE rayon dispatch for both gate and up
        // Each thread handles a midi-tile of rows for BOTH projections
        use rayon::prelude::*;

        gate_output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .zip(up_output[..out_dim].par_chunks_mut(MIDI_TILE_M))
            .enumerate()
            .for_each(|(midi_idx, (gate_chunk, up_chunk))| {
                let midi_start = midi_idx * MIDI_TILE_M;

                for (local_idx, (gate_out, up_out)) in
                    gate_chunk.iter_mut().zip(up_chunk.iter_mut()).enumerate()
                {
                    let row = midi_start + local_idx;
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;

                    // Activations loaded once per midi-tile (L1/L2 cache reuse)
                    let gate_row = &gate_weight_data[row_start..row_end];
                    *gate_out = dot_fn(gate_row, &acts).unwrap_or(0.0);

                    let up_row = &up_weight_data[row_start..row_end];
                    *up_out = dot_fn(up_row, &acts).unwrap_or(0.0);
                }
            });
    }

    Ok(())
}

/// Fused gate+up matvec for Q4_K weights
///
/// Contract: cpu-q4k-activation-quant-v1.yaml (Refs realizar#96)
/// Pre-quantizes f32 activations to Q8_K once, then delegates to
/// `fused_q4k_q8k_ffn_up_gate_into` for integer-only inner loops.
/// Both gate and up projections share the same quantized activations.
pub fn fused_gate_up_q4k_into(
    gate_weight_data: &[u8],
    up_weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    gate_output: &mut [f32],
    up_output: &mut [f32],
) -> Result<()> {
    use super::parallel_k::fused_q4k_q8k_ffn_up_gate_into;
    use super::types::QK_K;

    // Phase 1: Quantize activations to Q8_K (shared by gate + up)
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let padded_in_dim = super_blocks_per_row * QK_K;

    let acts = pad_activations(activations, padded_in_dim);

    let num_superblocks = padded_in_dim / QK_K;
    let mut q8k_scales = vec![0.0f32; num_superblocks];
    let mut q8k_quants = vec![0i8; padded_in_dim];

    super::quantize_activations_q8k_into(&acts, &mut q8k_scales, &mut q8k_quants)?;

    // Phase 2: Q4K×Q8K fused gate+up with integer-only inner loops
    fused_q4k_q8k_ffn_up_gate_into(
        up_weight_data,
        gate_weight_data,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        up_output,
        gate_output,
    )
}

/// Fused gate+up matvec for Q5_K weights
pub fn fused_gate_up_q5k_into(
    gate_weight_data: &[u8],
    up_weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    gate_output: &mut [f32],
    up_output: &mut [f32],
) -> Result<()> {
    use super::format_trait::Q5K;
    use super::fused_q5k_q6k::fused_q5k_dot_simd;

    generic_fused_gate_up_matvec_into::<Q5K>(
        gate_weight_data,
        up_weight_data,
        activations,
        in_dim,
        out_dim,
        gate_output,
        up_output,
        fused_q5k_dot_simd,
    )
}

/// Fused gate+up matvec for Q6_K weights
pub fn fused_gate_up_q6k_into(
    gate_weight_data: &[u8],
    up_weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    gate_output: &mut [f32],
    up_output: &mut [f32],
) -> Result<()> {
    use super::format_trait::Q6K;
    use super::fused_q5k_q6k::fused_q6k_dot_simd;

    generic_fused_gate_up_matvec_into::<Q6K>(
        gate_weight_data,
        up_weight_data,
        activations,
        in_dim,
        out_dim,
        gate_output,
        up_output,
        fused_q6k_dot_simd,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::format_trait::{Q4K, Q5K, Q6K};
    use crate::quantize::generic_dot::generic_fused_dot_scalar;
    use crate::quantize::generic_matvec::generic_parallel_matvec_into;

    /// Helper: create test weights with valid structure for a given format
    fn create_test_weights<F: QuantBlockFormat>(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(F::ELEMENTS_PER_SUPERBLOCK);
        let bytes_per_row = super_blocks_per_row * F::SUPERBLOCK_BYTES;
        vec![0u8; out_dim * bytes_per_row]
    }

    /// Scalar dot wrappers for testing
    fn q4k_scalar_dot(data: &[u8], acts: &[f32]) -> Result<f32> {
        generic_fused_dot_scalar::<Q4K>(data, acts)
    }
    fn q5k_scalar_dot(data: &[u8], acts: &[f32]) -> Result<f32> {
        generic_fused_dot_scalar::<Q5K>(data, acts)
    }
    fn q6k_scalar_dot(data: &[u8], acts: &[f32]) -> Result<f32> {
        generic_fused_dot_scalar::<Q6K>(data, acts)
    }

    #[test]
    fn test_fused_gate_up_q4k_matches_separate() {
        let in_dim = 256;
        let out_dim = 128;
        let gate_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let up_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];

        // Fused path
        let mut fused_gate = vec![0.0f32; out_dim];
        let mut fused_up = vec![0.0f32; out_dim];
        generic_fused_gate_up_matvec_into::<Q4K>(
            &gate_weights,
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut fused_gate,
            &mut fused_up,
            q4k_scalar_dot,
        )
        .expect("fused should succeed");

        // Separate paths
        let mut sep_gate = vec![0.0f32; out_dim];
        let mut sep_up = vec![0.0f32; out_dim];
        generic_parallel_matvec_into::<Q4K>(
            &gate_weights,
            &acts,
            in_dim,
            out_dim,
            &mut sep_gate,
            q4k_scalar_dot,
        )
        .expect("separate gate should succeed");
        generic_parallel_matvec_into::<Q4K>(
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut sep_up,
            q4k_scalar_dot,
        )
        .expect("separate up should succeed");

        // Results must match exactly (same computation, same inputs)
        assert_eq!(fused_gate, sep_gate, "Q4K gate output mismatch");
        assert_eq!(fused_up, sep_up, "Q4K up output mismatch");
    }

    #[test]
    fn test_fused_gate_up_q6k_matches_separate() {
        let in_dim = 256;
        let out_dim = 64;
        let gate_weights = create_test_weights::<Q6K>(out_dim, in_dim);
        let up_weights = create_test_weights::<Q6K>(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];

        let mut fused_gate = vec![0.0f32; out_dim];
        let mut fused_up = vec![0.0f32; out_dim];
        generic_fused_gate_up_matvec_into::<Q6K>(
            &gate_weights,
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut fused_gate,
            &mut fused_up,
            q6k_scalar_dot,
        )
        .expect("fused should succeed");

        let mut sep_gate = vec![0.0f32; out_dim];
        let mut sep_up = vec![0.0f32; out_dim];
        generic_parallel_matvec_into::<Q6K>(
            &gate_weights,
            &acts,
            in_dim,
            out_dim,
            &mut sep_gate,
            q6k_scalar_dot,
        )
        .expect("separate gate should succeed");
        generic_parallel_matvec_into::<Q6K>(
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut sep_up,
            q6k_scalar_dot,
        )
        .expect("separate up should succeed");

        assert_eq!(fused_gate, sep_gate, "Q6K gate output mismatch");
        assert_eq!(fused_up, sep_up, "Q6K up output mismatch");
    }

    #[test]
    fn test_fused_gate_up_q5k_matches_separate() {
        let in_dim = 256;
        let out_dim = 64;
        let gate_weights = create_test_weights::<Q5K>(out_dim, in_dim);
        let up_weights = create_test_weights::<Q5K>(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];

        let mut fused_gate = vec![0.0f32; out_dim];
        let mut fused_up = vec![0.0f32; out_dim];
        generic_fused_gate_up_matvec_into::<Q5K>(
            &gate_weights,
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut fused_gate,
            &mut fused_up,
            q5k_scalar_dot,
        )
        .expect("fused should succeed");

        let mut sep_gate = vec![0.0f32; out_dim];
        let mut sep_up = vec![0.0f32; out_dim];
        generic_parallel_matvec_into::<Q5K>(
            &gate_weights,
            &acts,
            in_dim,
            out_dim,
            &mut sep_gate,
            q5k_scalar_dot,
        )
        .expect("separate gate should succeed");
        generic_parallel_matvec_into::<Q5K>(
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut sep_up,
            q5k_scalar_dot,
        )
        .expect("separate up should succeed");

        assert_eq!(fused_gate, sep_gate, "Q5K gate output mismatch");
        assert_eq!(fused_up, sep_up, "Q5K up output mismatch");
    }

    #[test]
    fn test_fused_gate_up_sequential_small() {
        // Below PARALLEL_THRESHOLD (128 < 256) → takes sequential path
        let in_dim = 256;
        let out_dim = 128;
        let gate_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let up_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];

        let mut gate_out = vec![0.0f32; out_dim];
        let mut up_out = vec![0.0f32; out_dim];
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &gate_weights,
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_gate_up_weight_too_small() {
        let acts = vec![1.0f32; 256];
        let mut gate_out = vec![0.0f32; 64];
        let mut up_out = vec![0.0f32; 64];

        // Gate weight too small
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &[0u8; 100], // too small
            &create_test_weights::<Q4K>(64, 256),
            &acts,
            256,
            64,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_err());

        // Up weight too small
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &create_test_weights::<Q4K>(64, 256),
            &[0u8; 100], // too small
            &acts,
            256,
            64,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_gate_up_output_too_small() {
        let weights = create_test_weights::<Q4K>(64, 256);
        let acts = vec![1.0f32; 256];

        // Gate output too small
        let mut gate_out = vec![0.0f32; 32]; // too small
        let mut up_out = vec![0.0f32; 64];
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &weights,
            &weights,
            &acts,
            256,
            64,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_err());

        // Up output too small
        let mut gate_out = vec![0.0f32; 64];
        let mut up_out = vec![0.0f32; 32]; // too small
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &weights,
            &weights,
            &acts,
            256,
            64,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_gate_up_padding() {
        // in_dim not a multiple of 256 — requires padding (GH-202)
        let in_dim: usize = 200;
        let out_dim: usize = 16;
        let gate_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let up_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];

        let mut gate_out = vec![0.0f32; out_dim];
        let mut up_out = vec![0.0f32; out_dim];
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &gate_weights,
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_gate_up_parallel_threshold() {
        // Above threshold (512 > 256) — takes parallel path
        let in_dim = 256;
        let out_dim = 512;
        let gate_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let up_weights = create_test_weights::<Q4K>(out_dim, in_dim);
        let acts = vec![1.0f32; in_dim];

        let mut gate_out = vec![0.0f32; out_dim];
        let mut up_out = vec![0.0f32; out_dim];
        let result = generic_fused_gate_up_matvec_into::<Q4K>(
            &gate_weights,
            &up_weights,
            &acts,
            in_dim,
            out_dim,
            &mut gate_out,
            &mut up_out,
            q4k_scalar_dot,
        );
        assert!(result.is_ok());
    }
}
