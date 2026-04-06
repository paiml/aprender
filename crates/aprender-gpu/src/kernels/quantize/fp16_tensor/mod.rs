//! FP16 and Tensor Core Q4K Kernels
//!
//! High-performance quantized inference kernels optimized for memory bandwidth
//! and tensor core utilization.
//!
//! ## Kernels
//!
//! - [`Fp16Q4KGemvKernel`] - FP16 input/output Q4K GEMV with 4x bandwidth reduction
//! - [`TensorCoreQ4KGemmKernel`] - Tensor Core accelerated Q4K GEMM for batched decode
//! - [`MultiWarpTensorCoreQ4KGemmKernel`] - 4-warp WMMA Q4K GEMM (PMAT-045)
//! - [`InterleavedWmmaQ4KGemmKernel`] - Coalesced WMMA Q4K GEMM with interleaved weights (PMAT-091)

mod fp16_gemv;
mod interleaved_wmma_gemm;
mod mw_tensor_core_gemm;
mod tensor_core_gemm;
mod w4a16_wmma_gemm;

pub use fp16_gemv::Fp16Q4KGemvKernel;
pub use interleaved_wmma_gemm::InterleavedWmmaQ4KGemmKernel;
pub use mw_tensor_core_gemm::MultiWarpTensorCoreQ4KGemmKernel;
pub use tensor_core_gemm::TensorCoreQ4KGemmKernel;
pub use w4a16_wmma_gemm::W4a16WmmaQ4KGemmKernel;
