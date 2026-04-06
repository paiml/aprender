//! Quantization and dequantization for model weights
//!
//! Implements quantization formats used by GGUF models:
//! - `F16`: 16-bit IEEE 754 half-precision
//! - `Q4_0`: 4-bit quantization (block size 32)
//! - `Q4_1`: 4-bit with scale and min (block size 32)
//! - `Q5_0`: 5-bit quantization (block size 32)
//! - `Q5_1`: 5-bit with scale and min (block size 32)
//! - `Q8_0`: 8-bit quantization (block size 32)
//! - `Q4_K`: 4-bit K-quantization (super-block size 256)
//! - `Q5_K`: 5-bit K-quantization (super-block size 256)
//! - `Q6_K`: 6-bit K-quantization (super-block size 256)
//!
//! ## `Q4_0` Format
//!
//! `Q4_0` stores weights in blocks of 32 values:
//! - 1 float32 scale factor per block
//! - 16 bytes of 4-bit quantized values (2 values per byte)
//! - Dequantization: `value = scale * quantized_value`
//!
//! ## `Q8_0` Format
//!
//! `Q8_0` stores weights in blocks of 32 values:
//! - 1 float32 scale factor per block
//! - 32 int8 quantized values
//! - Dequantization: `value = scale * quantized_value`
//!
//! ## `Q4_K` Format
//!
//! `Q4_K` uses super-blocks of 256 values divided into 8 blocks of 32 values:
//! - 1 half-precision super-block scale (`d`)
//! - 1 half-precision super-block min (`dmin`)
//! - 12 bytes of 6-bit block scales (packed)
//! - 128 bytes of 4-bit quantized values
//! - Dequantization: `value = d * scale * quantized - dmin * min`
//! - Achieves 4.5 bits per weight with better quality than `Q4_0`
//!
//! ## `Q5_K` Format
//!
//! `Q5_K` uses super-blocks of 256 values divided into 8 blocks of 32 values:
//! - 1 half-precision super-block scale (`d`)
//! - 1 half-precision super-block min (`dmin`)
//! - 12 bytes of 6-bit block scales (packed)
//! - 32 bytes of high bits (1 bit per value for 5-bit quantization)
//! - 128 bytes of low 4-bit quantized values
//! - Dequantization: `value = d * scale * quantized - dmin * min`
//! - Achieves 5.5 bits per weight (higher quality than `Q4_K`)
//!
//! ## `Q6_K` Format
//!
//! `Q6_K` uses super-blocks of 256 values divided into 16 blocks of 16 values:
//! - 1 half-precision super-block scale (`d`)
//! - 16 bytes of 8-bit block scales
//! - 64 bytes of high 2 bits (2 bits per value for 6-bit quantization)
//! - 128 bytes of low 4-bit quantized values
//! - Dequantization: `value = d * scale * quantized`
//! - Achieves 6.5625 bits per weight (highest quality K-quant format)

use crate::error::{RealizarError, Result};

// ============================================================================
// Shattered submodules (PMAT-802)
// ============================================================================

pub mod activation;
pub mod bsum_precompute;
pub mod contract_tests;
pub mod dequant;
pub mod encode;
pub mod format_trait;
pub mod fused_gate_up;
pub mod fused_k;
pub mod fused_q5k_q6k;
pub(crate) mod gemv_pool;
pub mod generic_dot;
pub mod generic_matvec;
pub mod parallel_dequant;
pub mod parallel_k;
pub mod simd;
pub mod types;

// Re-export types from submodules (PMAT-802)
pub use types::{
    detect_simd_backend, DequantStats, Q4_0Block, Q4_KBlock, Q5_KBlock, Q6_KBlock, Q8KSuperBlock,
    Q8_0Block, SimdBackend, BLOCK_SIZE, QK_K,
};

// Re-export dequantization functions (PMAT-802)
pub use dequant::{
    dequantize_f16, dequantize_q2_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    f16_to_f32,
};

// Re-export fused K-quant operations (PMAT-802)
pub use fused_k::{fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd};
pub use fused_q5k_q6k::{
    fused_q4k_q8_dot, fused_q5k_dot, fused_q5k_dot_simd, fused_q6k_dot, fused_q6k_dot_simd,
};

// Re-export parallel K-quant operations (PMAT-802)
// LAYOUT-002: All kernels are ROW-MAJOR. No colmajor/auto aliases.
pub use parallel_k::{
    fused_q4k_parallel_matvec, fused_q4k_parallel_matvec_into, fused_q4k_q8k_ffn_up_gate_into,
    fused_q4k_q8k_parallel_matvec_into, fused_q4k_tiled_matvec, fused_q5k_parallel_matvec,
    fused_q5k_parallel_matvec_into, fused_q6k_parallel_matvec, fused_q6k_parallel_matvec_into,
};

// Re-export activation functions (PMAT-802)
pub use activation::{
    fused_rmsnorm_ffn_up_gate, fused_rmsnorm_q4_0_matmul, fused_swiglu_simd,
    quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_rmsnorm_q8_0_into, softmax_simd,
};

// Re-export parallel dequant operations (PMAT-802)
pub use parallel_dequant::{
    apply_rope_rotation_simd, dequantize_q4_k_parallel, dequantize_q4_k_simd,
    dequantize_q8_0_parallel, dequantize_q8_0_simd,
};

// Re-export SIMD utilities (for tests and internal use)
pub use simd::{extract_scale_min, extract_scale_min_from_slice, read_f16};

// Re-export format trait and generic kernels (Contract: quantized-dot-product-v1.yaml)
pub use format_trait::{Q4_0Fmt, Q8_0Fmt, QuantBlockFormat, QuantFamily, Q4K, Q5K, Q6K};
pub use generic_dot::{compute_bsums, generic_fused_dot_scalar};
pub use generic_matvec::{generic_parallel_matvec, generic_parallel_matvec_into};

// Re-export fused gate+up kernel (PMAT-FFN-FUSION)
pub use fused_gate_up::{
    fused_gate_up_q4k_into, fused_gate_up_q5k_into, fused_gate_up_q6k_into,
    generic_fused_gate_up_matvec_into,
};

// Re-export bsum precomputation (Contract: quantized-dot-product-v1.yaml, Step 3)
pub use bsum_precompute::{fused_q4k_q8k_parallel_matvec_with_bsums_into, precompute_q8k_bsums};

// Re-export encoding functions (Toyota Way: ONE source of truth)
// aprender imports these for format conversion - NEVER duplicates
pub use encode::{
    dequantize_q4_k_to_f32,
    dequantize_q5_k_to_f32,
    dequantize_q6_k_to_f32,
    // Q4_K
    quantize_q4_k,
    quantize_q4_k_matrix,
    // Q5_K
    quantize_q5_k,
    quantize_q5_k_matrix,
    // Q6_K
    quantize_q6_k,
    quantize_q6_k_matrix,
    // Transpose (LAYOUT-002)
    transpose_q4k_for_matmul,
    transpose_q5k_for_matmul,
    transpose_q6k_for_matmul,
    // Constants
    F16_MIN_NORMAL,
};

/// Pre-computed f16 to f32 lookup table (65536 entries = 256KB)
///
/// Eliminates per-block f16 conversion overhead in hot paths.
/// Per spec §4.1: f16 scale LUT should provide ~1.1x throughput improvement.
///
/// # Safety
/// The table is initialized once on first access and is immutable thereafter.
static F16_TO_F32_LUT: std::sync::LazyLock<Box<[f32; 65536]>> = std::sync::LazyLock::new(|| {
    let mut lut = Box::new([0.0f32; 65536]);
    for i in 0..65536u32 {
        lut[i as usize] = half::f16::from_bits(i as u16).to_f32();
    }
    lut
});

/// Fast f16 to f32 conversion using pre-computed LUT
///
/// Takes raw u16 bits (little-endian) and returns f32 value.
/// ~3x faster than half::f16::from_bits().to_f32() for hot paths.
#[inline]
pub(crate) fn f16_to_f32_lut(bits: u16) -> f32 {
    F16_TO_F32_LUT[bits as usize]
}

// BLOCK_SIZE, QK_K, Q4_0Block, Q8_0Block, Q8KSuperBlock moved to types.rs (PMAT-802)

/// Quantize f32 activations to Q8_K super-blocks (zero-allocation variant)
///
/// Pre-allocates output buffers for scales and quantized values.
/// Used for amortized quantization in hot inference path.
///
/// # Arguments
/// * `activations` - Input f32 values (must be multiple of 256)
/// * `scales` - Output scales buffer (len = activations.len() / 256)
/// * `quants` - Output int8 buffer (len = activations.len())
///
/// # Errors
/// Returns error if length is not a multiple of 256
pub fn quantize_activations_q8k_into(
    activations: &[f32],
    scales: &mut [f32],
    quants: &mut [i8],
) -> Result<()> {
    if !activations.len().is_multiple_of(256) {
        return Err(RealizarError::FormatError {
            reason: format!(
                "Q8_K quantization requires length multiple of 256, got {}",
                activations.len()
            ),
        });
    }

    let num_superblocks = activations.len() / 256;

    if scales.len() < num_superblocks {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Scales buffer too small: need {}, have {}",
                num_superblocks,
                scales.len()
            ),
        });
    }

    if quants.len() < activations.len() {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Quants buffer too small: need {}, have {}",
                activations.len(),
                quants.len()
            ),
        });
    }

    for (sb_idx, chunk) in activations.chunks_exact(256).enumerate() {
        Q8KSuperBlock::quantize_into(
            chunk,
            &mut scales[sb_idx],
            &mut quants[sb_idx * 256..(sb_idx + 1) * 256],
        );
    }

    Ok(())
}

/// Quantize a slice of f32 values to Q8_0 blocks
///
/// # Arguments
/// * `values` - F32 values (must be multiple of 32 in length)
///
/// # Returns
/// Vector of Q8_0Block, one per 32 values
///
/// # Errors
/// Returns error if length is not a multiple of 32
pub fn quantize_to_q8_blocks(values: &[f32]) -> Result<Vec<Q8_0Block>> {
    if !values.len().is_multiple_of(32) {
        return Err(RealizarError::FormatError {
            reason: format!(
                "Q8_0 quantization requires length multiple of 32, got {}",
                values.len()
            ),
        });
    }

    let blocks: Vec<Q8_0Block> = values
        .chunks_exact(32)
        .map(|chunk| {
            let arr: [f32; 32] = chunk.try_into().expect("chunk is exactly 32 elements");
            Q8_0Block::quantize(&arr)
        })
        .collect();

    Ok(blocks)
}

/// Dequantize Q8_0 blocks back to f32 values
pub fn dequantize_q8_blocks(blocks: &[Q8_0Block]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * 32);
    for block in blocks {
        output.extend_from_slice(&block.dequantize());
    }
    output
}

// Q4_KBlock, Q5_KBlock, Q6_KBlock moved to types.rs (PMAT-802)

/// PMAT-PERF-002: Pre-interleaved Q4_K weights for SIMD-friendly access
///
/// Weights reordered at load time to eliminate gather operations during inference.
/// This provides 2-4x speedup for Q4_K GEMV operations by enabling contiguous
/// SIMD loads instead of scattered nibble extraction.
///
/// # Layout
///
/// Original Q4_K layout (training-friendly):
/// ```text
/// Super-block: [d, dmin, scales[12], qs[128]]
/// qs layout: byte[i] contains value[2i] in low nibble, value[2i+1] in high nibble
/// ```
///
/// Interleaved layout (inference-friendly):
/// ```text
/// Super-block: [d, dmin, scales[12], qs_interleaved[128]]
/// qs_interleaved: values reordered for 32-byte aligned SIMD loads
/// After AVX2 256-bit load + nibble extraction, values are in processing order
/// ```
///
/// # Performance
///
/// - Before: Nibble extraction requires shift/mask per byte (32 ops for 64 values)
/// - After: Single SIMD load gets 32 contiguous values (1 op for 32 values)
/// - Expected speedup: 2-4x for GEMV kernel
///
/// # References
///
/// - Intel AVX-512 Guide: Contiguous loads 5x faster than VPGATHERDD
/// - llama.cpp: Pre-interleaved layout in ggml-quants.c
/// - CUTLASS: Tile-based weight layout for tensor cores
#[derive(Debug, Clone)]
pub struct InterleavedQ4K {
    /// Super-block scales (one per super-block, f32 from f16)
    pub d: Vec<f32>,
    /// Super-block mins (one per super-block, f32 from f16)
    pub dmin: Vec<f32>,
    /// Block scales (12 bytes per super-block, 6-bit packed)
    pub scales: Vec<u8>,
    /// Interleaved 4-bit quantized values
    /// Reordered so SIMD loads get contiguous values without gather
    pub qs: Vec<u8>,
    /// Number of super-blocks
    pub num_super_blocks: usize,
}

include!("product.rs");
include!("q4_0.rs");
include!("fused_q4_0_q8_0.rs");
include!("fused_q8_0_q8_0.rs");
