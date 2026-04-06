//! Q4_K GEMV Kernels (Fused Dequantization Matrix-Vector Multiplication)
//!
//! Optimized kernels for M=1 matmuls (token generation critical path):
//! y = W * x where W is (N×K) in Q4_K format, x is (K), y is (N)
//!
//! ## Kernel Variants
//!
//! - `Q4KGemvKernel`: Basic warp-per-output GEMV
//! - `BatchedQ4KGemvKernel`: Multi-batch processing with register tiling
//! - `TiledQ4KGemvKernel`: Shared memory input caching
//! - `ChunkedTiledQ4KGemvKernel`: Overlapped compute + global read
//! - `CoalescedQ4KGemvKernel`: Multi-output per block with vectorized loads
//! - `Dp4aQ4KGemvKernel`: DP4A integer dot product acceleration
//! - `VectorizedQ4KGemvKernel`: Coalesced u32 loads for high bandwidth
//! - `TrueDp4aQ4KGemvKernel`: Full DP4A implementation with Q8 activations
//!
//! ## Memory Bandwidth
//!
//! Q4_K: 144 bytes per 256 values = 0.5625 bytes/value (vs 4 bytes for f32)
//! This is 7.1x more memory efficient than dequantize+GEMV approach.

mod basic;
mod batched;
mod coalesced;
mod dequant;
mod dp4a;
pub mod interleaved;
mod tiled;
pub mod w4a16;

pub use basic::Q4KGemvKernel;
pub use batched::BatchedQ4KGemvKernel;
pub use coalesced::{
    BatchedHwDp4aQ4KGemvKernel, CoalescedQ4KGemvKernel, FusedFp32Q4KGemvKernel,
    FusedGateUpSwigluHwDp4aQ4KGemvKernel, FusedQKVHwDp4aQ4KGemvKernel, HalfWarpDp4aQ4KGemvKernel,
    InlineQ8Dp4aQ4KGemvKernel, MultiWarpVectorizedQ4KGemvKernel, MwvDp4aQ4KGemvKernel,
    VectorizedQ4KGemvKernel, WideQ4KGemvKernel,
};
pub use dequant::{Q4KDequantFp16Kernel, Q4KDequantKernel};
pub use dp4a::{Dp4aQ4KGemvKernel, TrueDp4aQ4KGemvKernel};
pub use tiled::{ChunkedTiledQ4KGemvKernel, TiledQ4KGemvKernel};
