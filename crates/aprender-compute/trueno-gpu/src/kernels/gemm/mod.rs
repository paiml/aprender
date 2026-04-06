//! GEMM (General Matrix Multiply) Kernels
//!
//! Implements C = alpha * A @ B + beta * C with multiple variants:
//!
//! - **Basic**: Standard 2D GEMM (naive, tiled, tensor core)
//! - **Batched**: 3D batched GEMM for independent matrix multiplications
//! - **Batched4D**: 4D batched GEMM for multi-head attention

pub(crate) mod basic;
mod batched;
mod batched_4d;

pub use basic::{GemmConfig, GemmKernel};
pub use batched::{BatchedGemmConfig, BatchedGemmKernel};
pub use batched_4d::{Batched4DGemmConfig, Batched4DGemmKernel};
