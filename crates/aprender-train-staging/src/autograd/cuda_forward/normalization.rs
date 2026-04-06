#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    BatchedRopeBackwardKernel, BatchedRopeKernel, BatchedVectorizedRmsNormKernel,
    FusedResidualRmsNormKernel, Kernel, LayerNormKernel, PerHeadRmsNormKernel, RopeNeoxKernel,
};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

/// Layer normalization forward pass on GPU
///
/// Computes: output = gamma * (input - mean) / sqrt(var + eps) + beta
#[cfg(feature = "cuda")]
pub fn layer_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    beta: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = LayerNormKernel::new(hidden_size);
    let kernel_name = kernel.name();

    let key = format!("layer_norm_forward_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
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
    let beta_ptr = beta.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 6] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &beta_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("LayerNorm forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// RMS normalization forward pass on GPU (LLaMA-style)
///
/// Computes: output = gamma * input / sqrt(mean(input^2) + eps)
///
/// Uses BatchedVectorizedRmsNormKernel: single kernel launch processes all
/// batch_size rows in parallel via grid.y = batch_size, 256 threads per block.
///
/// ALB-076: Previously launched one 32-thread kernel per row (2048 launches for
/// batch=4, seq=512). nsys profiling showed this was 97.1% of all GPU time.
/// Single batched launch eliminates 100K+ kernel launches per step.
#[cfg(feature = "cuda")]
pub fn rms_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, batch_size);

    let key = format!("batched_rmsnorm_fwd_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Grid: (1, batch_size, 1) — one block per row, all rows in parallel
    // Block: (256, 1, 1) — 8 warps per block for parallel reduction
    let config = LaunchConfig {
        grid: (1, batch_size, 1),
        block: (256, 1, 1),
        shared_mem: 8 * 4, // 8 warp partial sums (f32)
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

/// Per-head RMSNorm forward pass on GPU (ENT-270: QK-norm for Qwen3).
///
/// Applies RMSNorm independently to each attention head:
///   output[h] = input[h] / sqrt(mean(input[h]^2) + eps) * gamma
///
/// Input layout: `[num_heads * head_dim]` (single sequence position, interleaved).
/// Gamma: `[head_dim]` (shared across all heads).
///
/// For seq_len > 1, call once per position (loop in caller).
#[cfg(feature = "cuda")]
pub fn per_head_rmsnorm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    num_heads: u32,
    head_dim: u32,
    pos_offset: usize,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = PerHeadRmsNormKernel::new(head_dim, num_heads);

    let key = format!("per_head_rmsnorm_fwd_{head_dim}_{num_heads}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // One block per head, one warp (32 threads) per block
    let config = LaunchConfig { grid: (num_heads, 1, 1), block: (32, 1, 1), shared_mem: 0 };

    // Offset into the buffer for this position
    let stride = (num_heads * head_dim) as usize;
    let input_offset = pos_offset * stride;
    let output_offset = pos_offset * stride;

    // CUdeviceptr is u64 — use arithmetic, not pointer .add()
    let input_ptr = input.as_ptr() + (input_offset * std::mem::size_of::<f32>()) as u64;
    let output_ptr = output.as_ptr() + (output_offset * std::mem::size_of::<f32>()) as u64;
    let gamma_ptr = gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
    ];

    unsafe {
        stream.launch_kernel(module, "per_head_rmsnorm", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("PerHeadRmsNorm forward failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// RoPE (NeoX/half-rotation) forward pass on GPU (ENT-270).
///
/// Applies rotary position embeddings with half-rotation layout:
///   pairs at (i, i + half_dim) — required for Qwen/LLaMA models.
///
/// Input layout: `[num_heads * head_dim]` (single sequence position, interleaved).
///
/// For seq_len > 1, call once per position with the position index.
#[cfg(feature = "cuda")]
pub fn rope_neox_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    num_heads: u32,
    head_dim: u32,
    pos: u32,
    pos_offset: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = RopeNeoxKernel::new(num_heads, head_dim, theta);

    let key = format!("rope_neox_fwd_{num_heads}_{head_dim}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // One block per head, half_dim threads per block
    let config =
        LaunchConfig { grid: (num_heads, 1, 1), block: (head_dim / 2, 1, 1), shared_mem: 0 };

    // Offset into buffer for this position
    let stride = (num_heads * head_dim) as usize;
    let byte_offset = pos_offset * stride * std::mem::size_of::<f32>();

    // CUdeviceptr is u64 — use arithmetic, not pointer .add()
    let input_ptr = input.as_ptr() + byte_offset as u64;
    let output_ptr = output.as_ptr() + byte_offset as u64;

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &pos as *const _ as *mut _,
    ];

    unsafe {
        stream.launch_kernel(module, "rope_neox", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("RoPE NeoX forward failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Batched RoPE NeoX forward — processes all seq_len positions in a single kernel launch.
///
/// Replaces per-position `rope_neox_forward` loop to avoid ~2048 kernel launches per block.
/// Uses Grid(num_heads, seq_len, 1) with positions read from a GPU buffer.
///
/// Input layout: `[seq_len, num_heads * head_dim]` (interleaved).
#[cfg(feature = "cuda")]
pub fn batched_rope_neox_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    positions: &GpuBuffer<u32>,
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = BatchedRopeKernel::new(num_heads, head_dim, seq_len, theta);

    let key = format!("batched_rope_fwd_{num_heads}_{head_dim}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config =
        LaunchConfig { grid: (num_heads, seq_len, 1), block: (head_dim / 2, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();
    let positions_ptr = positions.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &positions_ptr as *const _ as *mut _,
    ];

    unsafe {
        stream.launch_kernel(module, "batched_rope", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Batched RoPE NeoX forward failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Batched RoPE NeoX backward — inverse rotation for gradient flow.
///
/// Applies R^T(-θ) to gradients so Q/K projection backward receives
/// correctly-framed gradients. Without this, dW_q and dW_k are computed
/// in the rotated coordinate frame, producing incorrect weight updates.
#[cfg(feature = "cuda")]
pub fn batched_rope_neox_backward(
    grad_input: &GpuBuffer<f32>,
    grad_output: &mut GpuBuffer<f32>,
    positions: &GpuBuffer<u32>,
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = BatchedRopeBackwardKernel::new(num_heads, head_dim, seq_len, theta);

    let key = format!("batched_rope_bwd_{num_heads}_{head_dim}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config =
        LaunchConfig { grid: (num_heads, seq_len, 1), block: (head_dim / 2, 1, 1), shared_mem: 0 };

    let input_ptr = grad_input.as_ptr();
    let output_ptr = grad_output.as_ptr();
    let positions_ptr = positions.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &positions_ptr as *const _ as *mut _,
    ];

    unsafe {
        stream.launch_kernel(module, "batched_rope_backward", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Batched RoPE NeoX backward failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Fused residual add + RMSNorm forward: output = RMSNorm(residual + input, gamma)
///
/// Contract: entrenar#321 — eliminates NaN cascade in layers 24-27 by fusing
/// the residual add with RMSNorm into a single kernel pass. The RMSNorm
/// normalization prevents activation explosion through the residual chain.
///
/// Saves the un-normalized residual sum in `residual_out` for backward pass.
///
/// # Parameters
/// - `residual`: Previous layer output (residual connection input)
/// - `input`: Current block output to add
/// - `residual_out`: Stores residual + input (for backward, can alias residual)
/// - `output`: RMSNorm(residual + input) * gamma
/// - `gamma`: Scale weights (hidden_size elements)
/// - `batch_size`: Number of rows (seq_len)
/// - `hidden_size`: Number of columns per row
#[cfg(feature = "cuda")]
pub fn fused_residual_rmsnorm_forward(
    residual: &GpuBuffer<f32>,
    input: &GpuBuffer<f32>,
    residual_out: &mut GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("fused_residual_rmsnorm_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = FusedResidualRmsNormKernel::new(hidden_size);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Grid: (1, batch_size, 1) — one block per row
    // Block: (32, 1, 1) — single warp for reduction
    let config = LaunchConfig { grid: (1, batch_size, 1), block: (32, 1, 1), shared_mem: 0 };

    let residual_ptr = residual.as_ptr();
    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();
    let gamma_ptr = gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &residual_ptr as *const _ as *mut _,
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
    ];

    // Also store the un-normalized residual sum for backward pass
    // The fused kernel writes residual+input to output before normalizing,
    // so we need to save it separately if residual_out != output
    if residual_out.as_ptr() != residual.as_ptr() {
        // First do the residual add into residual_out
        crate::autograd::cuda_forward::residual_add_forward(
            residual,
            input,
            residual_out,
            batch_size * hidden_size,
            stream,
        )?;
    }

    // Launch fused kernel: output = RMSNorm(residual + input) * gamma
    unsafe {
        stream.launch_kernel(module, "fused_residual_rmsnorm", &config, &mut args).map_err(
            |e| {
                CudaTensorError::KernelError(format!(
                    "Fused residual+RMSNorm forward failed: {e:?}"
                ))
            },
        )?;
    }

    Ok(())
}
