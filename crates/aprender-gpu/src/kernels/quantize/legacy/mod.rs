//! Legacy Quantization GEMV Kernels
//!
//! This module contains GEMV kernels for older quantization formats that are still
//! used in some models but are not the primary focus for optimization.
//!
//! ## Kernels
//!
//! - [`Q8_0GemvKernel`] - 8-bit quantization (32 int8 + fp16 scale)
//! - [`Q4_0GemvKernel`] - 4-bit centered quantization (nibble - 8)
//! - [`Q4_1GemvKernel`] - 4-bit affine quantization (d * nibble + m)
//! - [`Q5_0GemvKernel`] - 5-bit quantization with high bits

mod q4_0;
mod q4_1;
mod q5_0;
mod q8_0;

pub use q4_0::Q4_0GemvKernel;
pub use q4_1::Q4_1GemvKernel;
pub use q5_0::Q5_0GemvKernel;
pub use q8_0::Q8_0GemvKernel;

// Re-export parent constants for use by submodules via `super::`
use super::{Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
