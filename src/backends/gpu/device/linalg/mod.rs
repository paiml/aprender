//! GPU linear algebra operations
//!
//! Matrix multiplication, vector addition, dot product, and 2D convolution.

pub mod cached_matmul;
mod convolve2d;
mod dot;
mod matmul;
mod vec_ops;
pub mod wgsl_forward;
