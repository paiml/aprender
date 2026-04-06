//! Helper functions for batched attention operations.
//!
//! Contains GPU kernel wrappers for layout conversion, transpose, GEMM,
//! scaling, softmax, and head extraction/copy operations.
//!
//! ## Submodules
//!
//! - [`layout`]: Layout conversions (interleaved <-> batched, transpose)
//! - [`compute`]: GEMM, scale, softmax, and head extraction/copy operations

mod compute;
mod layout;

#[cfg(feature = "cuda")]
pub(super) use compute::{
    batched_gemm, batched_scale_all, batched_softmax_all, copy_head_to_output, extract_single_head,
    transpose_matrix,
};
#[cfg(feature = "cuda")]
pub(super) use layout::{
    batched_to_interleaved_all, batched_transpose_all, interleaved_to_batched_all,
};
