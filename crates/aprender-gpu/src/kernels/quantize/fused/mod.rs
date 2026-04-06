//! Fused Quantized GEMV Kernels
//!
//! These kernels fuse multiple operations to reduce memory bandwidth:
//! - FusedRmsNormQ4KGemvKernel: RMSNorm + Q4K GEMV in single pass
//! - FusedGateUpQ4KGemvKernel: Gate + Up projections sharing input load
//! - FusedRmsNormGateUpSwigluQ4KKernel: RMSNorm + Gate+Up + SwiGLU (3-way fusion)
//! - FusedRmsNormNf4GemvKernel: RMSNorm + NF4 GEMV for training (PMAT-475)
//! - FusedNf4GateUpGemmKernel: Gate + Up NF4 GEMM sharing input (PMAT-475)

mod gate_up_gemv;
mod nf4_gate_up;
mod nf4_rmsnorm_gemv;
mod rmsnorm_gate_up_swiglu;
mod rmsnorm_gemv;

pub use gate_up_gemv::FusedGateUpQ4KGemvKernel;
pub use nf4_gate_up::FusedNf4GateUpGemmKernel;
pub use nf4_rmsnorm_gemv::FusedRmsNormNf4GemvKernel;
pub use rmsnorm_gate_up_swiglu::FusedRmsNormGateUpSwigluQ4KKernel;
pub use rmsnorm_gemv::FusedRmsNormQ4KGemvKernel;
