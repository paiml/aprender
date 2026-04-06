//! Tensor Core GEMM variants (simulated 16x16 and true WMMA FP16)
//!
//! ## Submodules
//!
//! - [`simulated`]: Simulated Tensor Core GEMM using 16x16 shared memory tiles
//! - [`wmma`]: True WMMA FP16 GEMM using hardware Tensor Core PTX intrinsics
//! - [`cta_wmma`]: CTA-level WMMA with 4 warps sharing 32×32 tiles
//! - [`cta64_wmma`]: CTA-level WMMA with 16 warps sharing 64×64 tiles (2× data reuse)

pub mod cta128_wmma;
pub mod cta64_wmma;
pub mod cta_wmma;
mod simulated;
mod wmma;
