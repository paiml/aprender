//! CUDA-accelerated forward kernels for autograd
//!
//! This module wraps trueno-gpu forward kernels for GPU-accelerated forward passes.
//! Provides 10-100x speedup over CPU ndarray implementations.
//!
//! # Architecture (SPEC-FT-001 v3.0.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_forward (this module)
//!             └── trueno-gpu/kernels
//!                     └── PTX generation + CUDA driver
//! ```
//!
//! # Available Kernels
//!
//! - `relu_forward` - ReLU activation
//! - `softmax_forward` - Numerically stable softmax with warp shuffle
//! - `batched_softmax_forward` - Row-wise batched softmax for attention
//! - `layer_norm_forward` - Fused layer normalization
//! - `rms_norm_forward` - RMS normalization (LLaMA-style)
//! - `gelu_forward` - GELU activation
//! - `silu_forward` - SiLU/Swish activation
//! - `gemm_forward` - Matrix multiplication
//! - `batched_4d_gemm_forward` - 4D batched GEMM for multi-head attention
//! - `residual_add_forward` - Element-wise addition (residual connections)
//! - `elementwise_mul_forward` - Element-wise multiplication
//! - `scale_forward` - Scalar multiplication

mod activations;
pub mod bf16_cast;
mod cache;
mod elementwise;
mod matmul;
pub mod matmul_f16;
mod normalization;
#[cfg(test)]
mod tests;

pub use activations::{
    batched_softmax_forward, gelu_forward, relu_forward, silu_forward, softmax_forward,
};
pub use bf16_cast::{bf16_slice_to_f32, f32_slice_to_bf16};
#[cfg(feature = "cuda")]
pub use bf16_cast::{cast_bf16_to_f32_gpu, cast_f32_to_bf16_gpu, cast_f32_to_f16_gpu};
#[cfg(feature = "cuda")]
pub(crate) use cache::set_forward_cublas_stream;
pub use cache::{
    init_forward_kernel_cache, pre_warm_forward_kernels, pre_warm_lora_backward_kernels,
    set_cublas_workspace,
};
pub use elementwise::{
    batched_to_interleaved_forward, batched_transpose_forward, elementwise_mul_forward,
    expand_kv_heads, inplace_add_gpu, interleaved_to_batched_forward, residual_add_forward,
    scale_forward,
};
#[cfg(feature = "cuda")]
pub use matmul::gemm_forward_bf16;
pub use matmul::{
    batched_4d_gemm_forward, fused_swiglu_forward, gemm_forward, gemm_forward_bt,
    gemm_nf4_backward_a, gemm_nf4_forward, gemm_nf4_gate_up_forward, gemm_nf4_tc_backward_a,
    gemm_nf4_tc_forward,
};
#[cfg(feature = "cuda")]
pub(crate) use matmul::{
    cublas_gemm_backward_a, cublas_gemm_backward_a_accumulate, cublas_gemm_backward_b,
};
#[cfg(feature = "cuda")]
pub use matmul::{gemm_nf4_backward_a_cublas, gemm_nf4_dequant_cublas};
#[cfg(feature = "cuda")]
pub use matmul_f16::{gemm_f16_to_f32_backward_a, gemm_f16_to_f32_forward, gemm_forward_f16};
pub use normalization::{
    batched_rope_neox_backward, batched_rope_neox_forward, fused_residual_rmsnorm_forward,
    layer_norm_forward, per_head_rmsnorm_forward, rms_norm_forward, rope_neox_forward,
};
