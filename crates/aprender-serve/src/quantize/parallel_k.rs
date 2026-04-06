//! Parallel and tiled matrix-vector operations for K-quantization (PMAT-802)
//!
//! Implements L2-aware tiled and parallel matvec operations:
//! - `fused_q4k_tiled_matvec` - L2-aware tiled matmul
//! - `fused_q4k_parallel_matvec`, `fused_q4k_parallel_matvec_into` - Parallel Q4_K
//! - `fused_q5k_parallel_matvec`, `fused_q5k_parallel_matvec_into` - Parallel Q5_K
//! - `fused_q6k_parallel_matvec`, `fused_q6k_parallel_matvec_into` - Parallel Q6_K
//!
//! Per Goto & Van Geijn "Anatomy of High-Performance Matrix Multiplication":
//! - GEBP (General Block Panel) tiling maximizes cache reuse
//! - Tile size should fit in L2 cache (~256KB-512KB typically)

use super::bsum_precompute::{fused_q4k_q8k_dot_with_bsums_simd, precompute_q8k_bsums};
use super::format_trait::{Q5K, Q6K};
#[cfg(target_arch = "x86_64")]
use super::fused_k::fused_q4k_q8k_dot_4rows_avx512vnni;
use super::fused_k::{fused_q4k_dot_simd, fused_q4k_q8k_dot_simd};
use super::fused_q5k_q6k::{fused_q5k_dot_simd, fused_q6k_dot_simd};
use super::generic_matvec::{generic_parallel_matvec, generic_parallel_matvec_into};
use super::types::QK_K;
use crate::error::{RealizarError, Result};
use std::borrow::Cow;

// ============================================================================

/// Default tile size for L2-aware tiled matmul
///
/// Chosen to fit in L2 cache while maximizing parallelism:
/// - Typical L2 size: 256KB-512KB
/// - Q4_K row size for hidden_dim=2560: ~1440 bytes
/// - 64 rows = ~92KB of weight data, plus activations
const DEFAULT_OUTPUT_TILE_SIZE: usize = 64;

/// Pad activations to super-block boundary when `in_dim % QK_K != 0`.
///
/// Quantized weights are stored with per-row padding to QK_K (256) element
/// boundaries. The fused dot product kernels expect activations to match
/// the padded length. This function zero-pads activations when needed,
/// returning a borrowed reference when no padding is required (zero-cost).
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

/// Fused Q4_K matrix-vector multiply with L2-aware tiling
///
/// Processes outputs in tiles to maximize L2 cache reuse.
/// Each tile loads weight data once and computes multiple outputs.
///
/// # Arguments
///
/// * `weight_data` - Raw Q4_K quantized weight data
/// * `activations` - Input activations [in_dim]
/// * `in_dim` - Input dimension (must be multiple of 256 for Q4_K)
/// * `out_dim` - Output dimension
/// * `tile_size` - Number of outputs to process per tile (default: 64)
///
/// # Returns
///
/// Output vector [out_dim]
///
/// # Errors
///
/// Returns error if dimensions don't match weight data
///
/// # Performance
///
/// - **L2-aware**: Tiles fit in L2 cache, reducing DRAM traffic
/// - **Fused**: Dequantize inline with dot product (8x bandwidth reduction)
/// - **SIMD**: Uses AVX2 when available for 4-8x compute speedup
#[allow(clippy::similar_names)]
pub fn fused_q4k_tiled_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    tile_size: Option<usize>,
) -> Result<Vec<f32>> {
    let tile_size = tile_size.unwrap_or(DEFAULT_OUTPUT_TILE_SIZE);

    // Calculate bytes per output row
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144; // Q4_K: 144 bytes per super-block

    // Validate dimensions
    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K weight data too small: need {} bytes for {}x{}, have {}",
                expected_weight_bytes,
                out_dim,
                in_dim,
                weight_data.len()
            ),
        });
    }

    if activations.len() != in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match in_dim {}",
                activations.len(),
                in_dim
            ),
        });
    }

    // GH-202 FIX: Pad activations to super-block boundary
    let padded_in_dim = super_blocks_per_row * QK_K;
    let acts = pad_activations(activations, padded_in_dim);

    let mut output = vec![0.0f32; out_dim];

    // Process outputs in tiles for L2 cache efficiency
    let num_tiles = out_dim.div_ceil(tile_size);

    for tile_idx in 0..num_tiles {
        let tile_start = tile_idx * tile_size;
        let tile_end = (tile_start + tile_size).min(out_dim);

        // Prefetch next tile's weight data (if available)
        #[cfg(target_arch = "x86_64")]
        if tile_idx + 1 < num_tiles {
            let next_tile_start = (tile_idx + 1) * tile_size;
            let next_row_start = next_tile_start * bytes_per_row;
            if next_row_start < weight_data.len() {
                // SAFETY: Prefetch is a hint, no memory safety requirements
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    use std::arch::x86_64::_MM_HINT_T0;
                    let ptr = weight_data.as_ptr().add(next_row_start);
                    _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
                }
            }
        }

        // Process tile: compute dot products for tile_start..tile_end
        for (idx, out_slot) in output[tile_start..tile_end].iter_mut().enumerate() {
            let o = tile_start + idx;
            let row_start = o * bytes_per_row;
            let row_end = row_start + bytes_per_row;
            let row_data = &weight_data[row_start..row_end];

            // Fused dequant + dot product
            *out_slot = fused_q4k_dot_simd(row_data, &acts)?;
        }
    }

    Ok(output)
}

// ============================================================================
// PARALLEL TILED MATRIX-VECTOR MULTIPLICATION (Phase 2 + 3)
// ============================================================================
//
// Per Blumofe & Leiserson [6] "Scheduling Multithreaded Computations by Work Stealing":
// - Work-stealing schedulers like rayon maximize CPU utilization
// - Each output row is independent → trivially parallelizable
// - Expected speedup: ~Nx on N-core systems for memory-bound workloads
// ============================================================================

/// Parallel fused Q4_K matrix-vector multiply with L2-aware tiling
///
/// Uses rayon parallel iterators for multi-core acceleration.
/// Per Valiant's BSP model [14], synchronization happens at tile boundaries.
///
/// # Performance
///
/// - **Multi-core**: Linear speedup up to memory bandwidth saturation
/// - **L2-aware**: Tiles fit in L2 cache
/// - **Fused**: 8x memory bandwidth reduction
/// - **SIMD**: AVX2 when available
/// - **Adaptive parallelism**: Sequential for small matrices, parallel for large (IMP-103)
/// - **Q8K activation quantization**: Pre-quantizes f32 activations to Q8_K once per matmul,
///   enabling integer-only inner loops (maddubs) for ~4-8x speedup (Refs realizar#96)
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
#[allow(clippy::similar_names)]
pub fn fused_q4k_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; out_dim];
    fused_q4k_parallel_matvec_into(weight_data, activations, in_dim, out_dim, &mut output)?;
    Ok(output)
}

/// Parallel fused Q4_K matrix-vector multiply - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
/// This avoids Vec allocation overhead that causes 30-40% performance loss.
///
/// Contract: cpu-q4k-activation-quant-v1.yaml (Refs realizar#96)
/// Pre-quantizes f32 activations to Q8_K format once per matmul call, then
/// delegates to `fused_q4k_q8k_parallel_matvec_into` for integer-only inner loops.
/// This eliminates per-element f32 dequant×multiply and uses maddubs_epi16 instead.
///
/// # Arguments
/// * `weight_data` - Raw Q4_K quantized weights [out_dim, in_dim]
/// * `activations` - Input activations [in_dim]
/// * `in_dim` - Input dimension (must match activations length)
/// * `out_dim` - Output dimension (must match output buffer length)
/// * `output` - Pre-allocated output buffer [out_dim]
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
/// - Output buffer length doesn't match out_dim
#[allow(clippy::similar_names)]
pub fn fused_q4k_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
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
    let padded_in_dim = super_blocks_per_row * QK_K;
    let acts = pad_activations(activations, padded_in_dim);

    // PMAT-305: Direct FP32 parallel matmul — skip Q8K quantization entirely.
    // PMAT-304 showed realizr IPC 1.60 vs llama.cpp 1.01: we burn too many
    // cycles on Q8K quantize + scale extraction. Direct FP32 dequant×multiply
    // eliminates Q8K overhead and matches llama.cpp's CPU dot product pattern.
    //
    // The direct path uses fused_q4k_dot_simd which dequants Q4K to FP32 and
    // multiplies with FP32 activations — same as ggml's scalar fallback.
    // On bandwidth-bound workloads, this is no slower than Q8K+maddubs because
    // the activation data fits in L1 cache regardless of format.
    // PMAT-305: Direct FP32 FALSIFIED (-17%, 25.4 vs 30.8 tok/s).
    // Q8K + maddubs (32 muls/insn) beats FP32 fmadd (8 muls/insn).
    // The Q8K quantize overhead is small vs 4x multiply throughput gain.
    let use_direct_fp32 = std::env::var("DIRECT_FP32_GEMV").as_deref() == Ok("1");

    if use_direct_fp32 {
        use rayon::prelude::*;
        const SB_BYTES: usize = 144;
        let bytes_per_row = super_blocks_per_row * SB_BYTES;

        output[..out_dim]
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_base = chunk_idx * 64;
                for (i, out) in chunk.iter_mut().enumerate() {
                    let row = row_base + i;
                    let row_start = row * bytes_per_row;
                    let row_data = &weight_data[row_start..row_start + bytes_per_row];
                    *out = super::fused_k::fused_q4k_dot_simd(row_data, &acts).unwrap_or(0.0);
                }
            });
        return Ok(());
    }

    // Fallback: Q8K quantize + integer dot product (DIRECT_FP32_GEMV=0)
    let num_superblocks = padded_in_dim / QK_K;
    const MAX_STACK_DIM: usize = 8960;
    const MAX_STACK_SB: usize = (MAX_STACK_DIM + 255) / 256;

    if padded_in_dim <= MAX_STACK_DIM {
        let mut scales = [0.0f32; MAX_STACK_SB];
        let mut quants = [0i8; MAX_STACK_DIM];
        super::quantize_activations_q8k_into(
            &acts,
            &mut scales[..num_superblocks],
            &mut quants[..padded_in_dim],
        )?;
        fused_q4k_q8k_parallel_matvec_into(
            weight_data,
            &scales[..num_superblocks],
            &quants[..padded_in_dim],
            in_dim,
            out_dim,
            output,
        )
    } else {
        let mut scales = vec![0.0f32; num_superblocks];
        let mut quants = vec![0i8; padded_in_dim];
        super::quantize_activations_q8k_into(&acts, &mut scales, &mut quants)?;
        fused_q4k_q8k_parallel_matvec_into(weight_data, &scales, &quants, in_dim, out_dim, output)
    }
}

/// PMAT-309: Pre-quantized Q8K matvec — skips Q8K quantize when caller provides it.
///
/// Use this when multiple matmuls share the same activation (e.g., O projection
/// after attention, FFN down after SwiGLU). The caller quantizes ONCE and passes
/// the Q8K data to each matmul call.
#[allow(dead_code)]
pub fn fused_q4k_preq8k_matvec_into(
    weight_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    q8k_bsums: &[i16],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144;

    if weight_data.len() < out_dim * bytes_per_row {
        return Err(RealizarError::InvalidShape {
            reason: "Weight data too small".to_string(),
        });
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        use rayon::prelude::*;
        let bpr = bytes_per_row;
        let nsb = super_blocks_per_row;
        let w_addr = weight_data.as_ptr() as usize;
        let sc_addr = q8k_scales.as_ptr() as usize;
        let qq_addr = q8k_quants.as_ptr() as usize;
        let bs_addr = q8k_bsums.as_ptr() as usize;

        output[..out_dim]
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let row_start = ci * 64;
                unsafe {
                    let w = w_addr as *const u8;
                    let sc = sc_addr as *const f32;
                    let qq = qq_addr as *const i8;
                    let bs = bs_addr as *const i16;
                    for (i, out) in chunk.iter_mut().enumerate() {
                        let row = row_start + i;
                        *out = super::fused_k::ggml_style_q4k_q8k_dot_avx2_raw(
                            w.add(row * bpr),
                            sc,
                            qq,
                            bs,
                            nsb,
                        );
                    }
                }
            });
        return Ok(());
    }

    // Fallback: use the existing parallel path
    fused_q4k_q8k_parallel_matvec_into(weight_data, q8k_scales, q8k_quants, in_dim, out_dim, output)
}

/// Quantize FP32 activations to Q8K format + compute bsums.
/// Returns (scales, quants, bsums) for use with fused_q4k_preq8k_matvec_into.
#[allow(dead_code)]
pub fn quantize_for_q4k_matvec(
    activations: &[f32],
    in_dim: usize,
) -> Result<(Vec<f32>, Vec<i8>, Vec<i16>)> {
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let padded_in_dim = super_blocks_per_row * QK_K;
    let num_superblocks = padded_in_dim / QK_K;
    let acts = pad_activations(activations, padded_in_dim);

    let mut scales = vec![0.0f32; num_superblocks];
    let mut quants = vec![0i8; padded_in_dim];
    super::quantize_activations_q8k_into(&acts, &mut scales, &mut quants)?;

    #[cfg(target_arch = "x86_64")]
    let bsums = unsafe { super::fused_k::precompute_q8k_bsums_i16(&quants, num_superblocks) };
    #[cfg(not(target_arch = "x86_64"))]
    let bsums = vec![0i16; num_superblocks * 16];

    Ok((scales, quants, bsums))
}

/// PMAT-297: Pre-allocated workspace for CPU Q8K activation quantization.
struct CpuQ8kWorkspace {
    scales: Vec<f32>,
    quants: Vec<i8>,
}

impl CpuQ8kWorkspace {
    const fn empty() -> Self {
        Self {
            scales: Vec::new(),
            quants: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, padded_in_dim: usize) {
        let num_superblocks = padded_in_dim / QK_K;
        if self.scales.len() < num_superblocks {
            self.scales.resize(num_superblocks, 0.0);
        }
        if self.quants.len() < padded_in_dim {
            self.quants.resize(padded_in_dim, 0);
        }
    }

    /// Borrow both buffers simultaneously (split borrow).
    fn buffers_mut(&mut self, num_sb: usize, padded_dim: usize) -> (&mut [f32], &mut [i8]) {
        (&mut self.scales[..num_sb], &mut self.quants[..padded_dim])
    }
}

/// Parallel fused Q5_K matrix-vector multiply
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
#[allow(clippy::similar_names)]
pub fn fused_q5k_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    // Contract: quantized-dot-product-v1.yaml — delegates to generic matvec
    generic_parallel_matvec::<Q5K>(
        weight_data,
        activations,
        in_dim,
        out_dim,
        fused_q5k_dot_simd,
    )
}

include!("q5k_q6k_matvec.rs");
include!("parallel_k_fused_q4k.rs");
