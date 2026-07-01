//! RoPE (Rotary Position Embedding) Kernels
//!
//! GPU kernels for rotary position embeddings in transformer models.
//!
//! ## Kernel Variants
//!
//! - `RopeKernel`: Standard adjacent-pair RoPE
//! - `RopeIndirectKernel`: CUDA Graph compatible version
//! - `RopeNeoxKernel`: NEOX/GPT-NeoX style (split halves)
//! - `RopeNeoxIndirectKernel`: CUDA Graph compatible NEOX version
//! - `BatchedRopeKernel`: Multi-sequence batched RoPE
//! - `PreciseRopeKernel`: High-precision for theta=1M (Qwen2.5)
//! - `PreciseRopeIndirectKernel`: Precise + CUDA Graph compatible

mod batched;
mod neox;
mod precise;
mod standard;

pub use batched::{BatchedRopeBackwardKernel, BatchedRopeKernel};
pub use neox::{
    BatchedRopeNeoxBackwardKernel, BatchedRopeNeoxKernel, RopeNeoxIndirectKernel, RopeNeoxKernel,
};
pub use precise::{PreciseRopeIndirectKernel, PreciseRopeKernel};
pub use standard::{RopeIndirectKernel, RopeKernel};

#[cfg(test)]
mod tests;
