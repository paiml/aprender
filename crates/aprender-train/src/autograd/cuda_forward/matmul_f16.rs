//! FP16 cuBLAS GEMM operations for training
//!
//! Contract: fp16-cublas-gemm-v1.yaml (PMAT-458)
//!
//! FP16 GEMM uses tensor cores on sm_89+ (83 TFLOPS vs 2 TFLOPS SIMD).
//! All matrices are FP16 (CUDA_R_16F) with FP32 accumulation (CUBLAS_COMPUTE_32F).
//! Expected: ~2x throughput vs fp32 on memory-BW-bound workloads.
//!
//! # Safety
//!
//! Backward GEMMs use `gemm_f16()` which internally uses CUBLAS_COMPUTE_32F
//! (FP32 accumulation). This is safe for transposed backward GEMMs — unlike
//! TF32 tensor cores which produce NaN at gradient magnitude ~1e5 (ALB-076).

#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CublasHandle, CudaStream, GemmOp, GpuBuffer};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

/// FP16 cuBLAS GEMM forward: C[M,N] = A[M,K] @ B[K,N] using tensor cores
///
/// Contract: fp16-cublas-gemm-v1.yaml C-FP16GEMM-001 (PMAT-458)
/// All matrices are FP16 (CUDA_R_16F). Accumulation in FP32 (CUBLAS_COMPUTE_32F).
/// Tensor cores activated via CUBLAS_GEMM_DEFAULT_TENSOR_OP.
/// Expected: ~2x throughput vs fp32 on memory-BW-bound workloads (RTX 4060L).
#[cfg(feature = "cuda")]
pub fn gemm_forward_f16(
    a: &GpuBuffer<u16>,
    b: &GpuBuffer<u16>,
    c: &mut GpuBuffer<u16>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    let cublas = cache.cublas().ok_or_else(|| {
        CudaTensorError::KernelError("cuBLAS handle required for fp16 GEMM".to_string())
    })?;
    let _ = stream; // cuBLAS handle already bound to stream
    cublas
        .gemm_f16(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            a.as_ptr(),
            k as i32,
            0.0,
            c.as_ptr(),
            n as i32,
        )
        .map_err(|e| {
            CudaTensorError::KernelError(format!("cuBLAS fp16 GEMM forward failed: {e:?}"))
        })
}

/// FP16 cuBLAS backward A: grad_A[M,K] = grad_C[M,N] @ B[K,N]^T (tensor cores)
///
/// Contract: fp16-cublas-gemm-v1.yaml C-FP16GEMM-002 (PMAT-458)
/// Gradient GEMM uses fp16 for memory bandwidth savings. Gradient accumulation
/// should be promoted to fp32 in the caller to prevent underflow.
/// Note: trueno gemm_f16 uses CUBLAS_COMPUTE_32F (fp32 accumulation), which
/// is safe for transposed backward GEMMs (unlike TF32 per ALB-076).
#[cfg(feature = "cuda")]
pub(crate) fn cublas_gemm_backward_a_f16(
    cublas: &CublasHandle,
    grad_output: &GpuBuffer<u16>,
    b: &GpuBuffer<u16>,
    grad_a: &mut GpuBuffer<u16>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    cublas
        .gemm_f16(
            GemmOp::Trans,
            GemmOp::NoTrans,
            k as i32,
            m as i32,
            n as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            grad_output.as_ptr(),
            n as i32,
            0.0,
            grad_a.as_ptr(),
            k as i32,
        )
        .map_err(|e| CudaTensorError::KernelError(format!("cuBLAS fp16 backward_a failed: {e:?}")))
}

/// FP16 cuBLAS backward B: grad_B[K,N] = A[M,K]^T @ grad_C[M,N] (tensor cores)
///
/// Contract: fp16-cublas-gemm-v1.yaml C-FP16GEMM-002 (PMAT-458)
#[cfg(feature = "cuda")]
pub(crate) fn cublas_gemm_backward_b_f16(
    cublas: &CublasHandle,
    a: &GpuBuffer<u16>,
    grad_output: &GpuBuffer<u16>,
    grad_b: &mut GpuBuffer<u16>,
    m: u32,
    k: u32,
    n: u32,
) -> Result<()> {
    cublas
        .gemm_f16(
            GemmOp::NoTrans,
            GemmOp::Trans,
            n as i32,
            k as i32,
            m as i32,
            1.0,
            grad_output.as_ptr(),
            n as i32,
            a.as_ptr(),
            k as i32,
            0.0,
            grad_b.as_ptr(),
            n as i32,
        )
        .map_err(|e| CudaTensorError::KernelError(format!("cuBLAS fp16 backward_b failed: {e:?}")))
}

/// Mixed-precision backward_a: grad_A(fp32) = grad_C(fp16) @ B(fp16)^T (tensor cores)
///
/// Contract: C-FP16GEMM-002 (PMAT-472)
/// Enables dropping fp32 weights: backward uses fp16 weights with fp32 accumulation.
/// Cast grad_output fp32→fp16 at call site, pass fp16 weights, get fp32 grad_input.
#[cfg(feature = "cuda")]
pub fn gemm_f16_to_f32_backward_a(
    grad_output: &GpuBuffer<u16>,
    b: &GpuBuffer<u16>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    let cublas = cache.cublas().ok_or_else(|| {
        CudaTensorError::KernelError("cuBLAS handle required for fp16→fp32 backward".to_string())
    })?;
    let _ = stream;
    cublas
        .gemm_f16_to_f32(
            GemmOp::Trans,
            GemmOp::NoTrans,
            k as i32,
            m as i32,
            n as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            grad_output.as_ptr(),
            n as i32,
            0.0,
            grad_a.as_ptr(),
            k as i32,
        )
        .map_err(|e| {
            CudaTensorError::KernelError(format!("cuBLAS fp16→fp32 backward_a failed: {e:?}"))
        })
}

/// Mixed-precision GEMM: C(fp32) = A(fp16) @ B(fp16) using tensor cores
///
/// Contract: fp16-cublas-gemm-v1.yaml C-FP16GEMM-001 (PMAT-470)
/// A and B are FP16 (weights and activations cast to fp16). C is FP32.
/// Uses CUBLAS_COMPUTE_32F with CUBLAS_GEMM_DEFAULT_TENSOR_OP.
/// This is the "practical FP16 path": cast fp32 activations to fp16,
/// multiply by fp16 weights, produce fp32 output for the rest of the pipeline.
#[cfg(feature = "cuda")]
pub fn gemm_f16_to_f32_forward(
    a: &GpuBuffer<u16>,
    b: &GpuBuffer<u16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    let cublas = cache.cublas().ok_or_else(|| {
        CudaTensorError::KernelError("cuBLAS handle required for fp16→fp32 GEMM".to_string())
    })?;
    let _ = stream;
    cublas
        .gemm_f16_to_f32(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            a.as_ptr(),
            k as i32,
            0.0,
            c.as_ptr(),
            n as i32,
        )
        .map_err(|e| {
            CudaTensorError::KernelError(format!("cuBLAS fp16→fp32 GEMM forward failed: {e:?}"))
        })
}
