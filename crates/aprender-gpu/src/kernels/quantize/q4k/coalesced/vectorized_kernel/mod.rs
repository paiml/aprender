//! Vectorized Q4_K GEMV kernel with coalesced u32 loads (PAR-069)
//!
//! Achieves high memory bandwidth by loading weights as u32:
//! - Each thread loads 4 consecutive bytes (8 nibbles = 8 Q4 values)
//! - 32 threads x 4 bytes = 128 bytes per warp transaction (perfectly coalesced!)
//!
//! ## Submodules
//!
//! - [`build_ptx`]: PTX code generation for the kernel

mod build_ptx;

/// Vectorized Q4_K GEMV kernel with coalesced u32 loads (PAR-069)
///
/// Achieves high memory bandwidth by loading weights as u32:
/// - Each thread loads 4 consecutive bytes (8 nibbles = 8 Q4 values)
/// - 32 threads x 4 bytes = 128 bytes per warp transaction (perfectly coalesced!)
pub struct VectorizedQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl VectorizedQ4KGemvKernel {
    /// Create a new vectorized Q4_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}
