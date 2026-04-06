//! Q4K x Q8 Dot Product Kernels
//!
//! This module contains kernels for computing dot products between Q4K quantized weights
//! and Q8 quantized activations using DP4A SIMD instructions.
//!
//! ## Kernels
//!
//! - [`Q4KQ8DotKernel`] - Basic Q4K x Q8 dot product using DP4A
//! - [`PackedDp4aQ4KQ8Kernel`] - Optimized packed DP4A version for 4 multiply-adds per instruction

mod packed_dp4a;
mod q4k_q8;

pub use packed_dp4a::PackedDp4aQ4KQ8Kernel;
pub use q4k_q8::Q4KQ8DotKernel;
