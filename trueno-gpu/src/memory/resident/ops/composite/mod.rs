//! Composite operations for GPU-resident tensors.
//!
//! Includes layer normalization, GELU activation, bias add, linear projection,
//! fused linear+GELU, and conv1d operations. Each synchronous operation has
//! a corresponding `_with_stream` variant for pipelined execution.
//!
//! ## Submodules
//!
//! - [`norm_activation`] - Layer normalization and GELU activation
//! - [`linear_bias`] - Bias add, linear projection, fused linear+GELU, and conv1d

#![allow(clippy::similar_names)]

mod linear_bias;
mod norm_activation;
