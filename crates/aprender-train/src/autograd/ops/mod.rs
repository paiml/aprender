//! Autograd operations with backward passes
//!
//! This module provides differentiable operations for automatic differentiation.

mod activations;
mod attention;
mod basic;
#[cfg(test)]
mod correctness_tests;
pub(crate) mod matmul;
mod normalize;

// Re-export all public operations
pub use activations::{gelu, relu, softmax, swish};
pub use attention::{attention, attention_causal};
pub use basic::{add, add_scaled, mul, scale, sum};
// The same gate as the definition: `realizar` alone did not compile (found enabling
// entrenar/realizar from apr-cli's `training`, #3742).
#[cfg(all(feature = "realizar", feature = "cuda"))]
pub use matmul::pre_warm_realizador_gemm;
pub use matmul::{matmul, matmul_compute, matmul_nt, transpose, transpose_tracked};
#[cfg(feature = "gpu")]
pub use matmul::{suppress_per_op_wgpu, unsuppress_per_op_wgpu};
pub use normalize::layer_norm;
