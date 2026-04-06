//! PARITY-110: Modality × Style Test Matrix
//!
//! Popperian falsifiable tests for ALL inference modalities.
//! Each test MUST be capable of FAILING to have scientific value.
//!
//! ## Modalities
//! - CPU Scalar: Baseline, no SIMD/GPU
//! - SIMD (AVX2): SIMD-accelerated CPU
//! - WGPU (Vulkan): GPU via wgpu
//! - CUDA (RTX 4090): Native CUDA kernels
//!
//! ## Batch Sizes
//! - Single-shot (n=1)
//! - Batch-4
//! - Batch-32 (GPU threshold)
//! - Batch-64

pub mod common;
pub mod cpu_scalar;
pub mod cuda_integration;
pub mod cuda_rtx4090;
pub mod simd_avx2;
pub mod wgpu_vulkan;
