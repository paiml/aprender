//! Q4_K Dequantization-Fused GEMM Kernel
//!
//! Implements fused dequantization with matrix multiplication per GGML/llama.cpp methodology.
//!
//! ## Q4_K Super-block Layout (144 bytes for 256 values)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Offset 0-1: d (f16 super-block scale)                       │
//! │ Offset 2-3: dmin (f16 super-block min)                      │
//! │ Offset 4-15: scales (12 bytes, packed 6-bit scale+min × 8)  │
//! │ Offset 16-143: qs (128 bytes, 256 × 4-bit values packed)    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Sub-block Structure
//!
//! Each super-block contains 8 sub-blocks of 32 values:
//! - Sub-block b uses: scale_b (6-bit) and min_b (6-bit) from scales[12]
//! - Dequantization: val = d × scale_b × quant - dmin × min_b
//!
//! ## PARITY-041: Fused Q4_K GEMM
//!
//! This kernel fuses dequantization with GEMM to eliminate intermediate buffers:
//! - Memory bandwidth: 144 bytes → 256 values (vs 512 bytes if dequantized to f16)
//! - 3.5x memory bandwidth reduction

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use super::Kernel;
use crate::ptx::PtxKernel;

mod dot;
mod dp4a_gemm;
mod fp16_tensor;
mod fused;
mod fused_gemm;
pub mod fused_kv_scatter;
mod legacy;
mod nf4;
mod nf4_cpu;
mod nf4_tensor_core;
mod q4k;
mod q5k;
mod q6k;
mod q8;

pub use dot::{PackedDp4aQ4KQ8Kernel, Q4KQ8DotKernel};
pub use dp4a_gemm::Dp4aQ4KGemmKernel;
pub use fp16_tensor::{
    Fp16Q4KGemvKernel, InterleavedWmmaQ4KGemmKernel, MultiWarpTensorCoreQ4KGemmKernel,
    TensorCoreQ4KGemmKernel, W4a16WmmaQ4KGemmKernel,
};
pub use fused::{
    FusedGateUpQ4KGemvKernel, FusedNf4GateUpGemmKernel, FusedRmsNormGateUpSwigluQ4KKernel,
    FusedRmsNormNf4GemvKernel, FusedRmsNormQ4KGemvKernel,
};
pub use legacy::{Q4_0GemvKernel, Q4_1GemvKernel, Q5_0GemvKernel, Q8_0GemvKernel};
pub(crate) use nf4::nf4_register_lut_lookup;
pub use nf4::{Nf4GemmKernel, Nf4GemmTransposeKernel};
pub use nf4_cpu::{
    dequantize_nf4, pack_nf4_for_gpu, quantize_nf4, unpack_nf4_from_gpu, Nf4Quantized,
    NF4_BLOCK_BYTES, NF4_BLOCK_SIZE, NF4_LUT,
};
pub use nf4_tensor_core::Nf4TensorCoreGemmKernel;
pub use q4k::interleaved::repack_q4k_interleaved;
pub use q4k::w4a16::repack_q4k_w4a16;
pub use q4k::{
    BatchedHwDp4aQ4KGemvKernel, BatchedQ4KGemvKernel, ChunkedTiledQ4KGemvKernel,
    CoalescedQ4KGemvKernel, Dp4aQ4KGemvKernel, FusedFp32Q4KGemvKernel,
    FusedGateUpSwigluHwDp4aQ4KGemvKernel, FusedQKVHwDp4aQ4KGemvKernel, HalfWarpDp4aQ4KGemvKernel,
    InlineQ8Dp4aQ4KGemvKernel, MultiWarpVectorizedQ4KGemvKernel, MwvDp4aQ4KGemvKernel,
    Q4KDequantFp16Kernel, Q4KDequantKernel, Q4KGemvKernel, TiledQ4KGemvKernel,
    TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel, WideQ4KGemvKernel,
};
pub use q5k::{Q5KGemvKernel, Q5KKernel};
pub use q6k::{
    BatchedQ6KGemvKernel, CoalescedQ6KGemvKernel, Dp4aQ6KGemvKernel, HalfWarpDp4aQ6KGemvKernel,
    MultiWarpQ6KGemvKernel, Q6KDequantKernel, Q6KGemvKernel, Q6KKernel,
};
pub use q8::Q8QuantizeKernel;

/// Q4_K sub-block size (number of weights per sub-block)
const Q4K_BLOCK_SIZE: u32 = 32;
/// Q4_K super-block size (number of weights per super-block)
pub(crate) const Q4K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q4_K super-block (2 + 2 + 12 + 128 = 144 bytes)
pub(crate) const Q4K_SUPER_BLOCK_BYTES: u32 = 144;
/// Legacy: Bytes per simplified Q4_K block (for backwards compatibility)
const Q4K_BLOCK_BYTES: u32 = 18;

/// Q5_K super-block size (number of weights per super-block)
pub(crate) const Q5K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q5_K super-block (2 + 2 + 12 + 128 + 32 = 176 bytes)
/// Layout: d(2) + dmin(2) + scales(12) + qs(128) + qh(32)
pub(crate) const Q5K_SUPER_BLOCK_BYTES: u32 = 176;

/// Q6_K super-block size (number of weights per super-block)
pub(crate) const Q6K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q6_K super-block (128 + 64 + 16 + 2 = 210 bytes)
/// Layout: ql(128) + qh(64) + scales(16) + d(2)
pub(crate) const Q6K_SUPER_BLOCK_BYTES: u32 = 210;

/// Q8_0 block size (number of weights per block)
const Q8_0_BLOCK_SIZE: u32 = 32;
/// Bytes per Q8_0 block (2 + 32 = 34 bytes)
/// Layout: d(2 bytes, fp16) + qs[32] (32 int8 values)
const Q8_0_BLOCK_BYTES: u32 = 34;

/// Q5_0 block size (number of weights per block)
const Q5_0_BLOCK_SIZE: u32 = 32;
/// Bytes per Q5_0 block (2 + 4 + 16 = 22 bytes)
/// Layout: d(2 bytes, fp16) + qh(4 bytes, u32 with 32 high bits) + qs[16] (32 nibbles)
const Q5_0_BLOCK_BYTES: u32 = 22;

/// Q4_K format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4KFormat {
    /// Simplified format (32 values, 18 bytes) - legacy
    Simplified,
    /// Real GGML format (256 values, 144 bytes per super-block)
    GgmlSuperBlock,
}

/// Q4_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct QuantizeKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by super_block_size (256)
    pub k: u32,
    /// Tile size for output
    pub tile_size: u32,
    /// Quantization block size
    pub block_size: u32,
    /// Format variant (GGML super-block or simplified)
    pub format: Q4KFormat,
    /// M-dimension tiling factor (default 1 = serial, >1 = tiled weight reuse)
    pub tile_m: u32,
}

impl QuantizeKernel {
    /// Create a new Q4_K quantized GEMM kernel (simplified format for compatibility)
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
            block_size: Q4K_BLOCK_SIZE,
            format: Q4KFormat::Simplified,
            tile_m: 1,
        }
    }

    /// Create a Q4_K kernel using real GGML super-block format (PARITY-041)
    ///
    /// This is the correct format for GGUF model weights:
    /// - 256 values per super-block
    /// - 144 bytes per super-block (2+2+12+128)
    /// - 8 sub-blocks with 6-bit scale/min each
    #[must_use]
    pub fn ggml(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
            block_size: Q4K_SUPER_BLOCK_SIZE,
            format: Q4KFormat::GgmlSuperBlock,
            tile_m: 1,
        }
    }

    /// Create a tiled Q4_K GGML kernel (GH-182: weight reuse across M rows)
    ///
    /// Each thread computes `tile_m` output rows for one column, loading weights
    /// once and reusing across all rows. Reduces weight bandwidth by `tile_m`×.
    ///
    /// Grid: (ceil(N/blockDim.x), ceil(M/tile_m))
    #[must_use]
    pub fn ggml_tiled(m: u32, n: u32, k: u32, tile_m: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
            block_size: Q4K_SUPER_BLOCK_SIZE,
            format: Q4KFormat::GgmlSuperBlock,
            tile_m,
        }
    }

    /// Set output tile size
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Set M-dimension tiling factor
    #[must_use]
    pub const fn with_tile_m(mut self, tile_m: u32) -> Self {
        self.tile_m = tile_m;
        self
    }

    /// Get number of quantization blocks per row
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        self.k / self.block_size
    }

    /// Get number of super-blocks per row (for GGML format)
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        self.k / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for QuantizeKernel {
    fn name(&self) -> &str {
        match (self.format, self.tile_m > 1) {
            (Q4KFormat::Simplified, _) => "q4k_gemm_fused",
            (Q4KFormat::GgmlSuperBlock, false) => "q4k_gemm_ggml",
            (Q4KFormat::GgmlSuperBlock, true) => "q4k_gemm_ggml_tiled",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match (self.format, self.tile_m > 1) {
            (Q4KFormat::Simplified, _) => self.build_fused_gemm_simplified(),
            (Q4KFormat::GgmlSuperBlock, false) => self.build_fused_gemm_ggml(),
            (Q4KFormat::GgmlSuperBlock, true) => self.build_fused_gemm_ggml_tiled(),
        }
    }
}

// Tests (~2K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;
