//! Element-wise GPU Kernels
//!
//! Simple element-wise operations for transformer forward passes.
//!
//! ## Available Kernels
//!
//! ### Residual Connection Kernels
//! - [`ResidualAddKernel`]: Element-wise addition for residual connections
//! - [`BatchedResidualAddKernel`]: Batched version processing M sequences
//! - [`FusedResidualRmsNormKernel`]: Fused residual add + RMSNorm
//!
//! ### Activation Kernels
//! - [`ReluKernel`]: ReLU activation
//! - [`SiluKernel`]: SiLU/Swish activation
//! - [`GeluKernel`]: GELU activation (approximate)
//! - [`ElementwiseMulKernel`]: Element-wise multiplication
//! - [`ScaleKernel`]: Scalar multiplication
//!
//! ### SwiGLU Kernels
//! - [`FusedSwigluKernel`]: Fused SiLU + multiply
//! - [`BatchedSwigluKernel`]: Batched SwiGLU
//!
//! ### KV Cache Kernels
//! - [`KvCacheScatterKernel`]: Scatter K/V to cache
//! - [`KvCacheScatterIndirectKernel`]: CUDA Graph compatible
//!
//! ### RoPE Kernels
//! - [`RopeKernel`]: Standard adjacent-pair RoPE
//! - [`RopeIndirectKernel`]: CUDA Graph compatible
//! - [`RopeNeoxKernel`]: NEOX-style (split halves)
//! - [`RopeNeoxIndirectKernel`]: NEOX + CUDA Graph
//! - [`BatchedRopeKernel`]: Multi-sequence batched RoPE
//! - [`PreciseRopeKernel`]: High-precision for theta=1M
//! - [`PreciseRopeIndirectKernel`]: Precise + CUDA Graph
//!
//! ### Transform Kernels
//! - [`TransposeKernel`]: Matrix transpose
//! - [`InterleavedToBatchedKernel`]: Layout conversion
//! - [`BatchedToInterleavedKernel`]: Layout conversion
//! - [`ExtractSingleHeadKernel`]: Extract one head
//! - [`CopySingleHeadKernel`]: Copy to head position
//! - [`BatchedTransposeKernel`]: Batched transpose
//! - [`BatchedScaleKernel`]: Batched scale
//! - [`BatchedSoftmaxKernel`]: Row-wise softmax
//!
//! # PAR-023: Async pipeline support
//!
//! These kernels are designed for GPU-resident execution without sync.

mod activations;
mod kv_cache;
mod residual;
mod rope;
mod swiglu;
mod transform;

#[cfg(test)]
mod rope_tests;

#[cfg(test)]
mod transform_tests;

// Re-export all kernel types
pub use activations::{ElementwiseMulKernel, GeluKernel, ReluKernel, ScaleKernel, SiluKernel};
pub use kv_cache::{KvCacheScatterIndirectKernel, KvCacheScatterKernel};
pub use residual::{
    BatchedFusedResidualRmsNormKernel, BatchedResidualAddKernel, FusedResidualRmsNormKernel,
    ResidualAddKernel,
};
pub use rope::{
    BatchedRopeBackwardKernel, BatchedRopeKernel, BatchedRopeNeoxBackwardKernel,
    BatchedRopeNeoxKernel, PreciseRopeIndirectKernel, PreciseRopeKernel, RopeIndirectKernel,
    RopeKernel, RopeNeoxIndirectKernel, RopeNeoxKernel,
};
pub use swiglu::{BatchedSwigluKernel, FusedSwigluKernel};
pub use transform::{
    BatchedScaleKernel, BatchedSoftmaxKernel, BatchedToInterleavedKernel, BatchedTransposeKernel,
    CastF32ToF16Kernel, CopySingleHeadKernel, ExtractSingleHeadKernel, InterleavedToBatchedKernel,
    TransposeKernel,
};
