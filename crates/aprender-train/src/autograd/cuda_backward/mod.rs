//! CUDA-accelerated backward kernels for autograd
//!
//! This module wraps trueno-gpu backward kernels for GPU-accelerated gradient computation.
//! Provides 10-100x speedup over CPU ndarray implementations.
//!
//! # Safety
//!
//! This module uses unsafe code for CUDA kernel launching, which is inherently unsafe.
//! The unsafe blocks are required for FFI calls to the CUDA driver API.
//!
//! # Architecture (SPEC-FT-001 v3.0.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_backward (this module)
//!             └── trueno-gpu/kernels/backward
//!                     └── PTX generation + CUDA driver
//! ```
//!
//! # Available Kernels
//!
//! - `relu_backward` - ReLU gradient: dL/dx = dL/dy * (x > 0)
//! - `gelu_backward` - GELU gradient with tanh approximation
//! - `silu_backward` - SiLU/Swish gradient
//! - `softmax_backward` - Softmax Jacobian-vector product
//! - `rms_norm_backward` - RMSNorm gradients for input and gamma
//! - `layer_norm_backward` - LayerNorm gradients for input, gamma, beta
//! - `gemm_backward_a` - Matrix multiply gradient w.r.t. A
//! - `gemm_backward_b` - Matrix multiply gradient w.r.t. B

mod cache;
mod elementwise;
mod gemm;
mod structured;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(crate) use cache::set_backward_cublas_stream;
pub use cache::{init_kernel_cache, pre_warm_lora_backward_kernels};
pub use elementwise::{gelu_backward, relu_backward, silu_backward};
pub use gemm::{
    gemm_backward_a, gemm_backward_a_accumulate, gemm_backward_a_fp16_dispatch,
    gemm_backward_a_fp16_dispatch_accumulate, gemm_backward_b,
};
pub use structured::{
    batched_softmax_backward, layer_norm_backward, rms_norm_backward, rms_norm_forward,
    softmax_backward,
};
