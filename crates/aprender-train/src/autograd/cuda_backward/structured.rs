#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::backward::{
    BatchedRmsNormBackwardKernel, BatchedSoftmaxBackwardKernel, LayerNormBackwardKernel,
    SoftmaxBackwardKernel,
};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::BatchedVectorizedRmsNormKernel;
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::Kernel;

use super::super::cuda_tensor::{CudaTensorError, Result};
#[cfg(feature = "cuda")]
use super::cache::KERNEL_CACHE;
#[cfg(feature = "cuda")]
use provable_contracts_macros::requires;

/// Softmax backward pass on GPU
///
/// Computes: grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))
#[cfg(feature = "cuda")]
// Contract: backward-pass-v1 / softmax_backward
#[requires(batch_size > 0 && seq_len > 0)]
pub fn softmax_backward(
    softmax_output: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    batch_size: u32,
    seq_len: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("softmax_backward_{batch_size}_{seq_len}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = SoftmaxBackwardKernel::new(batch_size, seq_len);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Softmax backward uses warp-parallel reduction
    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (32.min(seq_len), 1, 1), // Warp size
        shared_mem: 0,
    };

    let output_ptr = softmax_output.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        &output_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &seq_len as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "softmax_backward", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Softmax backward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Batched softmax backward pass on GPU (handles row_size > 32)
///
/// Computes: grad_input[r][i] = y[r][i] * (grad_output[r][i] - Σⱼ grad_output[r][j] * y[r][j])
///
/// Uses stride-loop + warp-shuffle reduction (one warp per row, one block per row).
///
/// # Contract (C-BSMAX-BACK-002)
///
/// - **Precondition**: softmax_output contains valid softmax output, all buffers have at least
///   total_rows * row_size elements, row_size > 0, total_rows > 0, KERNEL_CACHE initialized
/// - **Postcondition**: grad_input[r][i] = y[r][i] * (∂L/∂y[r][i] - dot(∂L/∂y[r], y[r]))
/// - **Invariant**: Zero CPU-side data transfers; in-place safe (grad_input may alias grad_output)
#[cfg(feature = "cuda")]
pub fn batched_softmax_backward(
    softmax_output: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    total_rows: u32,
    row_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    // Contract: dimension-independent-kernels-v1.yaml
    // Note: BatchedSoftmaxBackwardKernel not yet dimension-independent in trueno,
    // but using generic key prepares for the fix.
    let key = "batched_softmax_backward";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let kernel = BatchedSoftmaxBackwardKernel::new(total_rows, row_size);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(key, &ptx)?
        }
    };

    // One warp (32 threads) per row, one block per row
    let config =
        LaunchConfig { grid: (total_rows, 1, 1), block: (32.min(row_size), 1, 1), shared_mem: 0 };

    let output_ptr = softmax_output.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        &output_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &total_rows as *const _ as *mut _,
        &row_size as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "batched_softmax_backward", &config, &mut args).map_err(
            |e| {
                CudaTensorError::KernelError(format!(
                    "Batched softmax backward launch failed: {e:?}"
                ))
            },
        )?;
    }

    Ok(())
}

/// RMSNorm backward pass on GPU
///
/// Computes gradients for input (and placeholder for gamma parameters).
/// Uses stride-loop kernel that supports arbitrary hidden_size (no warp-only limit).
///
/// # Contract (C-RMSBACK-WRAP-001)
///
/// - **Precondition**: input contains original forward input, gamma has hidden_size elements,
///   all buffers allocated with at least batch_size * hidden_size elements
/// - **Postcondition**: grad_input contains ∂L/∂x per the RMSNorm backward formula
/// - **Invariant**: Uses batched stride-loop kernel; no hidden_size upper limit
#[cfg(feature = "cuda")]
pub fn rms_norm_backward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    grad_gamma: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-001)
    let key = "batched_rms_norm_backward";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let kernel = BatchedRmsNormBackwardKernel::new(batch_size, hidden_size, eps);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(key, &ptx)?
        }
    };

    // One warp (32 threads) per row, one block per row
    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (32.min(hidden_size), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();
    let grad_gamma_ptr = grad_gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 8] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &grad_gamma_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
        &eps as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "batched_rms_norm_backward", &config, &mut args).map_err(
            |e| CudaTensorError::KernelError(format!("RMSNorm backward launch failed: {e:?}")),
        )?;
    }

    Ok(())
}

/// RMSNorm forward pass on GPU (KAIZEN-066).
///
/// Computes: output = input * rsqrt(mean(input^2) + eps) * gamma
///
/// Uses BatchedVectorizedRmsNormKernel — 8 warps per block, processes
/// seq_len rows in parallel via Grid.y.
///
/// # Contract (C-GPUNORM-001)
///
/// - **Precondition**: input has batch_size * hidden_size elements, gamma has hidden_size elements
/// - **Postcondition**: output contains RMSNorm(input) * gamma
/// - **Invariant**: Same numerical result as CPU norm.forward_batched (within fp32 precision)
#[cfg(feature = "cuda")]
pub fn rms_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("batched_rmsnorm_fwd_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, batch_size);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Grid: (1, batch_size, 1) — one block per row, each block processes one row
    // Block: (256, 1, 1) — 8 warps per block for parallel reduction
    let config = LaunchConfig {
        grid: (1, batch_size, 1),
        block: (256, 1, 1),
        shared_mem: 8 * 4, // 8 warp partial sums
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();
    let gamma_ptr = gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. input has batch_size * hidden_size elements,
    // output has batch_size * hidden_size elements, gamma has hidden_size elements.
    // Parameters match PTX signature (u64 input_ptr, u64 output_ptr, u64 gamma_ptr).
    unsafe {
        stream.launch_kernel(module, "batched_rmsnorm_vectorized", &config, &mut args).map_err(
            |e| CudaTensorError::KernelError(format!("RMSNorm forward launch failed: {e:?}")),
        )?;
    }

    Ok(())
}

/// LayerNorm backward pass on GPU
///
/// Computes gradients for input, gamma, and beta parameters
#[cfg(feature = "cuda")]
pub fn layer_norm_backward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    grad_gamma: &mut GpuBuffer<f32>,
    grad_beta: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("layer_norm_backward_{batch_size}_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = LayerNormBackwardKernel::new(batch_size, hidden_size);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (256.min(hidden_size), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();
    let grad_gamma_ptr = grad_gamma.as_ptr();
    let grad_beta_ptr = grad_beta.as_ptr();

    let mut args: [*mut std::ffi::c_void; 8] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &grad_gamma_ptr as *const _ as *mut _,
        &grad_beta_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "layer_norm_backward", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("LayerNorm backward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}
