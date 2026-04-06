#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::float_cmp,
        clippy::panic
    )
)]
//! N-dimensional tensor contractions — Einstein summation via TTGT.
//!
//! This crate provides a dense tensor type and Einstein summation (`einsum`)
//! that reduces tensor contractions to explicit loops, following the TTGT
//! (Transpose-Transpose-GEMM-Transpose) strategy for cuTENSOR parity.

pub mod einsum;
pub mod error;
pub mod tensor;

pub use einsum::{batch_matmul, einsum, einsum_nary, matmul, outer, trace};
pub use error::TensorError;
pub use tensor::Tensor;

#[cfg(test)]
mod tests;
