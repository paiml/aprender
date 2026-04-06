//! DP4A-based Q4_K GEMV Kernels for 4x Instruction Reduction
//!
//! - `Dp4aQ4KGemvKernel`: Basic DP4A implementation
//! - `TrueDp4aQ4KGemvKernel`: Full DP4A with Q8 activations

mod basic;
mod vectorized;

pub use basic::Dp4aQ4KGemvKernel;
pub use vectorized::TrueDp4aQ4KGemvKernel;
