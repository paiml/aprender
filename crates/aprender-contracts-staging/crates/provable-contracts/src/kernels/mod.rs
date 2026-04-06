//! Kernel implementations: scalar reference, AVX2 SIMD, and CUDA PTX.
//!
//! Each submodule provides three variants of its kernel:
//! - `fn {name}_scalar(...)` — Pure Rust scalar reference (ground truth)
//! - `unsafe fn {name}_avx2(...)` — AVX2 SIMD implementation
//! - `fn {name}_ptx() -> &'static str` — PTX assembly source string

// Kernel code naturally uses single-character math variable names (m, n, k, q, v, etc.),
// raw string hashes for PTX assembly, and unsafe intrinsics inside unsafe fns.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_raw_string_hashes,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::explicit_iter_loop,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::wildcard_imports,
    clippy::doc_markdown,
    unsafe_op_in_unsafe_fn
)]

pub mod ops;
pub mod ulp;

// Group A — Elementwise
pub mod activation;
pub mod silu_standalone;

// Group B — Normalization
pub mod batchnorm;
pub mod layernorm;
pub mod rmsnorm;
pub mod softmax;

// Group C — Gated + Positional + Loss
pub mod absolute_position;
pub mod bias_add;
pub mod cross_entropy;
pub mod dropout;
pub mod gelu;
pub mod rope;
pub mod swiglu;

// Group D — Matrix + Projection
pub mod attention;
pub mod flash_attention;
pub mod gqa;
pub mod linear;
pub mod matmul;
pub mod tied_embeddings;
pub mod transpose;

// Group E — Optimizer + Sequence + Classical ML + IO
pub mod adamw;
pub mod alibi;
pub mod cma_es;
pub mod conv1d;
pub mod embedding;
pub mod f16_convert;
pub mod gated_delta_net;
pub mod kmeans;
pub mod lbfgs;
pub mod pagerank;
pub mod sampling;
pub mod ssm;

#[cfg(kani)]
mod kani_proofs;

/// Backend selector for kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pure Rust scalar reference implementation.
    Scalar,
    /// x86-64 AVX2 SIMD implementation.
    Avx2,
    /// CUDA PTX kernel (returned as assembly source string).
    Ptx,
}
