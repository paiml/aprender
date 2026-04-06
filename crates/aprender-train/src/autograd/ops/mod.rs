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
pub use attention::attention;
pub use basic::{add, add_scaled, mul, scale, sum};
#[cfg(feature = "realizar")]
pub use matmul::pre_warm_realizador_gemm;
pub use matmul::{matmul, matmul_compute, matmul_nt, transpose, transpose_tracked};
#[cfg(feature = "gpu")]
pub use matmul::{suppress_per_op_wgpu, unsuppress_per_op_wgpu};
pub use normalize::layer_norm;
