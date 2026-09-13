//! L0-1b (#2971, PMAT-1070): the f32-activation Q4_K matvec.
//!
//! The Q8_K path quantises the activation vector with one scale per 256
//! elements; on a massive-activation token one element sets the scale of 255
//! others and their coherent rounding error reaches 13 % of a neuron's output
//! (docs/audits/l0-1b-arms.md). When [`super::has_crushed_block`] says so, the
//! reference forward runs THIS matvec instead: Q4_K weights dequantised against
//! f32 activations (`fused_q4k_dot_simd`, the `DIRECT_FP32_GEMV` kernel), row
//! parallel. Same numerics as `DIRECT_FP32_GEMV=1` — arm A7 measured it at
//! cosine 0.999896 end to end.
use rayon::prelude::*;

use super::fused_k::fused_q4k_dot_simd;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

/// `output[row] = dot(dequant(weight_row), activations)` for every row, f32 activations.
///
/// # Errors
/// Activation length must equal `in_dim`; the weight must hold `out_dim` rows.
pub fn fused_q4k_parallel_matvec_f32_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    const SB_BYTES: usize = 144;
    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4K activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SB_BYTES;
    if weight_data.len() < out_dim * bytes_per_row || output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4K matvec: weight {} bytes / output {} for out_dim {} x {} bytes per row",
                weight_data.len(),
                output.len(),
                out_dim,
                bytes_per_row
            ),
        });
    }
    let padded_in_dim = super_blocks_per_row * QK_K;
    let mut padded = activations.to_vec();
    padded.resize(padded_in_dim, 0.0);
    let acts = &padded[..];
    output[..out_dim]
        .par_chunks_mut(64)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let row_base = chunk_idx * 64;
            for (i, out) in chunk.iter_mut().enumerate() {
                let row_start = (row_base + i) * bytes_per_row;
                *out = fused_q4k_dot_simd(&weight_data[row_start..row_start + bytes_per_row], acts)
                    .unwrap_or(0.0);
            }
        });
    Ok(())
}
