//! Layout Transform and Batched Kernels
//!
//! GPU kernels for tensor layout transformations in multi-head attention.
//!
//! ## Transpose Kernels ([`transpose`])
//!
//! - [`TransposeKernel`]: Matrix transpose
//! - [`BatchedTransposeKernel`]: Transpose multiple matrices
//!
//! ## Layout Conversion Kernels ([`layout`])
//!
//! - [`InterleavedToBatchedKernel`]: Convert interleaved to batched layout
//! - [`BatchedToInterleavedKernel`]: Convert batched to interleaved layout
//! - [`ExtractSingleHeadKernel`]: Extract one head from interleaved tensor
//! - [`CopySingleHeadKernel`]: Copy to head position in interleaved tensor
//!
//! ## Element-wise Kernels ([`element_wise`])
//!
//! - [`BatchedScaleKernel`]: Scale all elements by scalar
//! - [`BatchedSoftmaxKernel`]: Row-wise softmax for attention

mod element_wise;
mod layout;
mod transpose;

pub use element_wise::{BatchedScaleKernel, BatchedSoftmaxKernel, CastF32ToF16Kernel};
pub use layout::{
    BatchedToInterleavedKernel, CopySingleHeadKernel, ExtractSingleHeadKernel,
    InterleavedToBatchedKernel,
};
pub use transpose::{BatchedTransposeKernel, TransposeKernel};
