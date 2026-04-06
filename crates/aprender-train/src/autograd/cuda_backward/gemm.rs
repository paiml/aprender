#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::backward::{GemmBackwardAKernel, GemmBackwardBKernel};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::Kernel;

use super::super::cuda_tensor::{CudaTensorError, Result};
#[cfg(feature = "cuda")]
use super::cache::KERNEL_CACHE;

// cuBLAS backward dispatch (ALB-075)
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{cublas_gemm_backward_a, cublas_gemm_backward_b};

/// Tile size for backward GEMM kernels (C-TILE-BWD-001).
///
/// Must be divisible by 4 (unroll factor). Shared memory per block = 2 * TILE^2 * 4 bytes.
/// TILE=16: 2KB smem, 256 threads/block. Safe for all dimensions including LoRA rank=16.
const BACKWARD_TILE_SIZE: u32 = 16;

/// GEMM backward pass for matrix A on GPU (trueno#109: tiled)
///
/// Given C = A @ B, computes: grad_A = grad_C @ B^T
///
/// Uses tiled GEMM with shared memory (C-TILE-BWD-001) and 4x unrolled inner loop.
#[cfg(feature = "cuda")]
pub fn gemm_backward_a(
    grad_output: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    // ALB-075: cuBLAS SIMD fast path (6-14x faster than PTX)
    // ALB-076: Uses CUBLAS_DEFAULT_MATH (no tensor cores) for backward GEMMs.
    // trueno#170 fixed NaN corruption caused by tensor core algorithms (TF32)
    // on transposed GEMMs with gradient magnitudes ~1e5. Forward GEMMs remain
    // on tensor cores since NoTrans/NoTrans is unaffected.
    if let Some(cublas) = cache.cublas() {
        return cublas_gemm_backward_a(cublas, grad_output, b, grad_a, m, k, n);
    }

    let tile = BACKWARD_TILE_SIZE;
    // Kernel object needed for name(); cheap struct creation, PTX deferred.
    let kernel = GemmBackwardAKernel::tiled_unrolled(m, n, k, tile);
    let kernel_name = kernel.name();

    let key = format!("gemm_backward_a_{m}_{k}_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Tiled launch: block = (TILE, TILE), grid covers output grad_a[M, K]
    let smem = 2 * tile * tile * 4; // 2 tiles of f32
    let config = LaunchConfig {
        grid: (k.div_ceil(tile), m.div_ceil(tile), 1),
        block: (tile, tile, 1),
        shared_mem: smem,
    };

    let grad_out_ptr = grad_output.as_ptr();
    let b_ptr = b.as_ptr();
    let grad_a_ptr = grad_a.as_ptr();

    // PTX kernel signature: (grad_c_ptr, b_ptr, grad_a_ptr, m, n, k)
    // CRITICAL: must match param declaration order in GemmBackwardAKernel::build_ptx()
    let mut args: [*mut std::ffi::c_void; 6] = [
        &grad_out_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &grad_a_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM backward A launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GEMM backward pass for matrix B on GPU (trueno#109: tiled)
///
/// Given C = A @ B, computes: grad_B = A^T @ grad_C
///
/// Uses tiled GEMM with shared memory (C-TILE-BWD-002) and 4x unrolled inner loop.
#[cfg(feature = "cuda")]
pub fn gemm_backward_b(
    a: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_b: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    // ALB-075: cuBLAS SIMD fast path (6-14x faster than PTX)
    // ALB-076: Uses CUBLAS_DEFAULT_MATH (no tensor cores) for backward GEMMs.
    // trueno#170 fixed NaN corruption caused by tensor core algorithms (TF32)
    // on transposed GEMMs with gradient magnitudes ~1e5. Forward GEMMs remain
    // on tensor cores since NoTrans/NoTrans is unaffected.
    if let Some(cublas) = cache.cublas() {
        return cublas_gemm_backward_b(cublas, a, grad_output, grad_b, m, k, n);
    }

    let tile = BACKWARD_TILE_SIZE;
    // Kernel object needed for name(); cheap struct creation, PTX deferred.
    let kernel = GemmBackwardBKernel::tiled_unrolled(m, n, k, tile);
    let kernel_name = kernel.name();

    let key = format!("gemm_backward_b_{m}_{k}_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Tiled launch: block = (TILE, TILE), grid covers output grad_b[K, N]
    let smem = 2 * tile * tile * 4;
    let config = LaunchConfig {
        grid: (n.div_ceil(tile), k.div_ceil(tile), 1),
        block: (tile, tile, 1),
        shared_mem: smem,
    };

    let a_ptr = a.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_b_ptr = grad_b.as_ptr();

    // PTX kernel signature: (a_ptr, grad_c_ptr, grad_b_ptr, m, n, k)
    // CRITICAL: must match param declaration order in GemmBackwardBKernel::build_ptx()
    let mut args: [*mut std::ffi::c_void; 6] = [
        &a_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_b_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &n as *const _ as *mut _,
        &k as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("GEMM backward B launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GEMM backward A with accumulation: grad_A += grad_C @ B^T (PMAT-484)
///
/// Adds result into grad_a instead of overwriting. Used for fused Gate+Up backward
/// to eliminate the separate cuda_add_inplace kernel launch.
#[cfg(feature = "cuda")]
pub fn gemm_backward_a_accumulate(
    grad_output: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    _stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    // cuBLAS accumulate path (beta=1.0) — this is the only path that matters
    // in production since cuBLAS is always initialized for NF4 QLoRA training.
    if let Some(cublas) = cache.cublas() {
        return crate::autograd::cuda_forward::cublas_gemm_backward_a_accumulate(
            cublas,
            grad_output,
            b,
            grad_a,
            m,
            k,
            n,
        );
    }

    // No cuBLAS = no accumulation support. NF4 training requires cuBLAS.
    Err(CudaTensorError::KernelError(
        "gemm_backward_a_accumulate requires cuBLAS (NF4 training always has it)".to_string(),
    ))
}

/// FP16-aware backward A with accumulation (PMAT-484): grad_A += grad_C @ B^T
///
/// Same as gemm_backward_a_fp16_dispatch but accumulates into grad_a.
/// Used for fused Gate+Up backward to eliminate cuda_add_inplace.
#[cfg(feature = "cuda")]
pub fn gemm_backward_a_fp16_dispatch_accumulate(
    grad_output: &GpuBuffer<f32>,
    w_fp16: Option<&GpuBuffer<u16>>,
    w_fp32: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
    _ctx: &trueno_gpu::driver::CudaContext,
) -> Result<()> {
    // For fp16 path: compute into temp then add (cuBLAS fp16 doesn't easily support beta=1 mixed)
    // For fp32 path: use cuBLAS beta=1.0 directly
    if w_fp16.is_some() {
        // FP16: compute into temp, then accumulate
        let mut temp = GpuBuffer::<f32>::new(_ctx, (m * k) as usize)
            .map_err(|e| CudaTensorError::AllocationFailed(format!("fp16 accum temp: {e:?}")))?;
        gemm_backward_a_fp16_dispatch(
            grad_output,
            w_fp16,
            w_fp32,
            &mut temp,
            m,
            k,
            n,
            stream,
            _ctx,
        )?;
        crate::transformer::cuda_block::cuda_add_inplace(grad_a, &temp, (m * k) as usize, stream)?;
        Ok(())
    } else {
        gemm_backward_a_accumulate(grad_output, w_fp32, grad_a, m, k, n, stream)
    }
}

/// FP16-aware backward A dispatch (PMAT-472): uses fp16 weights when available.
///
/// If `w_fp16` is Some, casts grad_output to fp16 and uses tensor core GEMM
/// (fp16×fp16→fp32). Otherwise falls back to fp32. Eliminates fp32 weight
/// storage — freeing ~2.6 GB VRAM for GPU embeddings on yoga 8GB.
#[cfg(feature = "cuda")]
pub fn gemm_backward_a_fp16_dispatch(
    grad_output: &GpuBuffer<f32>,
    w_fp16: Option<&GpuBuffer<u16>>,
    w_fp32: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
    ctx: &trueno_gpu::driver::CudaContext,
) -> Result<()> {
    if let Some(w16) = w_fp16 {
        let elems = (m * n) as usize;
        let mut grad_f16 = GpuBuffer::<u16>::new(ctx, elems)
            .map_err(|e| CudaTensorError::AllocationFailed(format!("grad f16 cast: {e:?}")))?;
        crate::autograd::cuda_forward::cast_f32_to_f16_gpu(
            grad_output,
            &mut grad_f16,
            m * n,
            stream,
        )?;
        crate::autograd::cuda_forward::gemm_f16_to_f32_backward_a(
            &grad_f16, w16, grad_a, m, k, n, stream,
        )
    } else {
        gemm_backward_a(grad_output, w_fp32, grad_a, m, k, n, stream)
    }
}
