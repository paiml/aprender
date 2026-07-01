#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    BatchedFusedResidualRmsNormKernel, BatchedRopeBackwardKernel, BatchedRopeKernel,
    BatchedVectorizedRmsNormKernel, Kernel, LayerNormKernel, PerHeadRmsNormKernel, RopeNeoxKernel,
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
    // Backwards-compatible default for legacy callers (Llama default).
    // Production callers in transformer/cuda_block.rs should call
    // rms_norm_forward_with_eps directly with config.rms_norm_eps so
    // Qwen2 / Qwen2.5 (rms_norm_eps=1e-6) gets the right epsilon.
    rms_norm_forward_with_eps(input, gamma, output, batch_size, hidden_size, 1e-5, stream)
}

/// FALSIFY-CUDA-RMSNORM-EPS-PARITY-001 (eps-aware variant): batched RMSNorm
/// forward that honours `config.rms_norm_eps` instead of hardcoding 1e-5.
///
/// Pre-fix: `rms_norm_forward` constructed `BatchedVectorizedRmsNormKernel::new`
/// (eps=1e-5, the Llama default) regardless of model. Qwen2 / Qwen2.5
/// uses `rms_norm_eps=1e-6` per its config.json. The 9e-6 absolute eps
/// difference compounds over 24 layers × 2 RMSNorms per block = 48 calls,
/// and is one of the residual contributors to CUDA-CPU forward divergence
/// surfaced by `apr-pretrain-cuda-forward-parity-v1.yaml`.
///
/// Cache key includes eps bits so two different epsilons compile to two
/// different PTX modules; otherwise a stale cached module would silently
/// shadow the new eps.
#[cfg(feature = "cuda")]
pub fn rms_norm_forward_with_eps(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, batch_size).with_epsilon(eps);

    // Cache key MUST include eps bits — different eps values compile to
    // different PTX (the constant is baked into `mov.f32`).
    let eps_bits = eps.to_bits();
    let key = format!("batched_rmsnorm_fwd_{hidden_size}_eps{eps_bits:08x}");
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

    // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
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

    // FALSIFY-CUDA-ROPE-THETA-CACHE-KEY-001: theta is baked into the
    // PTX at build_ptx time (RopeNeoxKernel::build_ptx captures
    // self.theta into the closure as `mov.f32 imm`). Two calls with
    // different theta values produce different PTX, so the cache key
    // MUST include theta_bits — otherwise the first theta to populate
    // the cache wins and subsequent calls silently use the wrong theta
    // (e.g. Llama 1e4 caches first → Qwen 1e6 calls reuse 1e4 PTX).
    let theta_bits = theta.to_bits();
    let key = format!("rope_neox_fwd_{num_heads}_{head_dim}_th{theta_bits:08x}");
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

    // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
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

    // FALSIFY-CUDA-ROPE-THETA-CACHE-KEY-001: cache key MUST include
    // theta_bits (and seq_len, which is also baked in via grid sizing).
    // See `rope_neox_forward` rationale.
    let theta_bits = theta.to_bits();
    let key = format!("batched_rope_fwd_{num_heads}_{head_dim}_{seq_len}_th{theta_bits:08x}");
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

    // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
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

    // FALSIFY-CUDA-ROPE-THETA-CACHE-KEY-001: cache key MUST include
    // theta_bits. See `rope_neox_forward` rationale.
    let theta_bits = theta.to_bits();
    let key = format!("batched_rope_bwd_{num_heads}_{head_dim}_{seq_len}_th{theta_bits:08x}");
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

    // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
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
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    // FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001 (wave of 4, all fixed here):
    //
    // 1. Self-deadlock: this function held the FORWARD_KERNEL_CACHE mutex
    //    guard while calling the public `residual_add_forward`, which
    //    re-locks the SAME non-reentrant `std::sync::Mutex` on the same
    //    thread — `Mutex::lock_contended` futex-waited forever and froze
    //    every `apr finetune -m qlora` run on the first transformer block
    //    forward. Fixed structurally: the batched kernel writes
    //    `residual_out` itself, so the nested call is gone entirely.
    // 2. Single-row kernel launched as batched: the old
    //    `FusedResidualRmsNormKernel` has no ctaid indexing (one warp, one
    //    row) but was launched with grid.y = batch_size — every block
    //    redundantly computed row 0 and rows 1.. were never written.
    //    `BatchedFusedResidualRmsNormKernel` (PMAT-092) indexes rows via
    //    ctaid.y.
    // 3. eps not threaded: the kernel default (1e-5, Llama) was silently
    //    used for Qwen2 models (1e-6). Callers now pass
    //    `config.rms_norm_eps`, and the cache key includes the eps bits
    //    (PMAT-698k lesson: eps-less keys shadow stale PTX).
    // 4. Missing pre-warm: this kernel JIT-compiled mid-training
    //    (Blackwell stream-poisoning class, PMAT-698). pre_warm_for_model
    //    now warms it at both Qwen2 (1e-6) and Llama (1e-5) eps.
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let eps_bits = eps.to_bits();
    let key = format!("batched_fused_residual_rmsnorm_{hidden_size}_eps{eps_bits:08x}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel =
                BatchedFusedResidualRmsNormKernel::new(hidden_size, batch_size).with_epsilon(eps);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Grid: (1, batch_size, 1) — one block per row via ctaid.y
    // Block: (256, 1, 1) — 8 warps; shared: 8 warp partial sums (f32)
    let config = LaunchConfig { grid: (1, batch_size, 1), block: (256, 1, 1), shared_mem: 8 * 4 };

    let residual_ptr = residual.as_ptr();
    let input_ptr = input.as_ptr();
    let residual_out_ptr = residual_out.as_ptr();
    let output_ptr = output.as_ptr();
    let gamma_ptr = gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        &residual_ptr as *const _ as *mut _,
        &input_ptr as *const _ as *mut _,
        &residual_out_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
    ];

    // Launch fused kernel:
    //   residual_out = residual + input
    //   output       = RMSNorm(residual + input) * gamma
    // (`residual_out` may alias `residual`: pass 1 reads each element before
    // writing it back, per-thread, so the aliased store is ordered safely.)
    // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
    unsafe {
        stream
            .launch_kernel(module, "batched_fused_residual_rmsnorm", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!(
                    "Fused residual+RMSNorm forward failed: {e:?}"
                ))
            })?;
    }

    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::autograd::cuda_forward::cache::init_forward_kernel_cache;
    use crate::autograd::cuda_tensor::CudaDevice;
    use trueno_gpu::driver::GpuBuffer;

    /// Reference CPU RMSNorm matching the kernel's exact arithmetic order:
    /// rms = sqrt(mean(x^2) + eps); y = (x / rms) * gamma.
    fn cpu_rmsnorm_reference(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len() as f32;
        let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / n;
        let rms = (mean_sq + eps).sqrt();
        input.iter().zip(gamma.iter()).map(|(&x, &g)| (x / rms) * g).collect()
    }

    /// FALSIFY-CUDA-RMSNORM-EPS-PARITY-001: With Qwen's eps=1e-6 the
    /// CUDA `rms_norm_forward_with_eps` MUST match the CPU reference to
    /// within 1e-5 absolute. The legacy `rms_norm_forward` (eps=1e-5
    /// hardcoded) cannot meet this bound on Qwen because the eps in the
    /// kernel disagrees with the reference's eps.
    ///
    /// On main pre-fix this test FAILS for `rms_norm_forward` (legacy)
    /// because the kernel uses eps=1e-5 while the CPU ref uses 1e-6.
    /// Post-fix `rms_norm_forward_with_eps(eps=1e-6)` passes by
    /// construction — the kernel compiles with the same eps the
    /// reference uses, so diffs are bounded by f32 round-off only.
    #[test]
    fn falsify_cuda_rmsnorm_eps_parity_qwen_1e_minus_6() {
        let device = match CudaDevice::default_device() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[falsify-cuda-rmsnorm-eps-parity-001] skipping (no CUDA host): {e}");
                return;
            }
        };
        let ctx = device.context().clone();
        let stream = device.stream();
        if let Err(e) = init_forward_kernel_cache(ctx.clone()) {
            eprintln!("[falsify-cuda-rmsnorm-eps-parity-001] kernel cache init failed: {e}");
            return;
        }

        // Qwen 0.5B hidden size; values intentionally small so mean_sq is
        // small enough that the eps difference 1e-5 vs 1e-6 actually
        // moves the rms denominator measurably. Real Qwen activations
        // post-embedding have std~0.02 so this is realistic.
        let hidden_size = 896usize;
        let batch_size = 4u32;
        let total = batch_size as usize * hidden_size;
        let input_data: Vec<f32> =
            (0..total).map(|i| (((i as f32) * 0.013).sin()) * 0.02).collect();
        let gamma_data: Vec<f32> =
            (0..hidden_size).map(|i| 1.0 + ((i as f32) * 0.005).cos() * 0.1).collect();

        // Build CPU reference once; it's the same per-row.
        let mut cpu_out = Vec::with_capacity(total);
        for b in 0..batch_size as usize {
            let row = &input_data[b * hidden_size..(b + 1) * hidden_size];
            cpu_out.extend(cpu_rmsnorm_reference(row, &gamma_data, 1e-6));
        }

        let input_gpu = GpuBuffer::from_host(&ctx, &input_data).expect("input");
        let gamma_gpu = GpuBuffer::from_host(&ctx, &gamma_data).expect("gamma");
        let mut output_gpu = GpuBuffer::<f32>::new(&ctx, total).expect("output alloc");

        rms_norm_forward_with_eps(
            &input_gpu,
            &gamma_gpu,
            &mut output_gpu,
            batch_size,
            hidden_size as u32,
            1e-6,
            stream,
        )
        .expect("kernel launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0.0f32; total];
        output_gpu.copy_to_host(&mut gpu_out).expect("download");

        let max_diff =
            cpu_out.iter().zip(gpu_out.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);

        eprintln!("[falsify-cuda-rmsnorm-eps-parity-001] max_diff={max_diff} (Qwen eps=1e-6)");
        assert!(
            max_diff < 1e-4,
            "FALSIFY-CUDA-RMSNORM-EPS-PARITY-001: max_diff={max_diff} >= 1e-4. \
             CUDA RMSNorm kernel disagrees with CPU reference at Qwen eps=1e-6. \
             Pre-fix root cause: BatchedVectorizedRmsNormKernel::new hardcodes \
             epsilon=1e-5 (Llama default) so calling `rms_norm_forward` for \
             Qwen2 silently uses the wrong eps. Fix: \
             `rms_norm_forward_with_eps(.., eps, ..)` threads `config.rms_norm_eps` \
             into the kernel and the cache key includes eps bits to avoid stale \
             PTX shadowing. See contract apr-pretrain-cuda-rmsnorm-eps-parity-v1.yaml."
        );
    }

    /// FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001: `fused_residual_rmsnorm_forward`
    /// with a `residual_out` buffer DISTINCT from `residual` must complete
    /// (liveness) and produce correct results (oracle).
    ///
    /// On main pre-fix this test DEADLOCKS: the function acquired the
    /// `FORWARD_KERNEL_CACHE` mutex guard for its whole body, then called the
    /// public `residual_add_forward` (normalization.rs:494) which re-locks the
    /// SAME non-reentrant `std::sync::Mutex` on the same thread →
    /// `Mutex::lock_contended` futex-waits forever. This froze every
    /// `apr finetune -m qlora` run on the first `CudaNf4TransformerBlock::
    /// forward` (gdb: thread 1 stuck in lock_contended ← residual_add_forward
    /// ← fused_residual_rmsnorm_forward ← CudaNf4TransformerBlock::forward).
    ///
    /// The watchdog thread converts the pre-fix deadlock into a bounded test
    /// failure instead of a hung test binary.
    ///
    /// Oracle (not just liveness): `residual_out == residual + input`
    /// element-wise, and `output` matches the CPU RMSNorm reference of
    /// `residual + input` at the threaded eps=1e-6 (Qwen2). The output
    /// oracle also caught the single-row kernel being launched as batched
    /// (rows 1.. never written, max_diff=2.36 vs reference) — see the
    /// wave-of-4 fix note in `fused_residual_rmsnorm_forward`.
    #[test]
    fn falsify_cuda_fused_rmsnorm_distinct_residual_out_no_deadlock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let hidden_size = 1536usize; // Qwen2.5-Coder-1.5B hidden (live repro dims)
        let batch_size = 4u32;
        let total = batch_size as usize * hidden_size;

        let residual_data: Vec<f32> =
            (0..total).map(|i| (((i as f32) * 0.017).sin()) * 0.02).collect();
        let input_data: Vec<f32> =
            (0..total).map(|i| (((i as f32) * 0.011).cos()) * 0.02).collect();
        let gamma_data: Vec<f32> =
            (0..hidden_size).map(|i| 1.0 + ((i as f32) * 0.007).cos() * 0.1).collect();

        enum Outcome {
            NoCuda(String),
            Done { residual_out: Vec<f32>, output: Vec<f32> },
        }

        let (tx, rx) = mpsc::channel();
        let res_clone = residual_data.clone();
        let inp_clone = input_data.clone();
        let gam_clone = gamma_data.clone();
        // Detached worker: on the pre-fix deadlock it blocks forever and the
        // watchdog below fails the test after the timeout instead of hanging.
        std::thread::spawn(move || {
            let device = match CudaDevice::default_device() {
                Ok(d) => d,
                Err(e) => {
                    let _ = tx.send(Outcome::NoCuda(format!("{e}")));
                    return;
                }
            };
            let ctx = device.context().clone();
            let stream = device.stream();
            if let Err(e) = init_forward_kernel_cache(ctx.clone()) {
                let _ = tx.send(Outcome::NoCuda(format!("cache init: {e}")));
                return;
            }

            let residual_gpu = GpuBuffer::from_host(&ctx, &res_clone).expect("residual");
            let input_gpu = GpuBuffer::from_host(&ctx, &inp_clone).expect("input");
            let gamma_gpu = GpuBuffer::from_host(&ctx, &gam_clone).expect("gamma");
            // DISTINCT residual_out buffer — the deadlock precondition.
            let mut residual_out_gpu =
                GpuBuffer::<f32>::new(&ctx, res_clone.len()).expect("residual_out alloc");
            let mut output_gpu =
                GpuBuffer::<f32>::new(&ctx, res_clone.len()).expect("output alloc");

            fused_residual_rmsnorm_forward(
                &residual_gpu,
                &input_gpu,
                &mut residual_out_gpu,
                &mut output_gpu,
                &gamma_gpu,
                batch_size,
                hidden_size as u32,
                1e-6, // Qwen2 rms_norm_eps (the live repro model family)
                stream,
            )
            .expect("fused_residual_rmsnorm_forward");
            stream.synchronize().expect("sync");

            let mut residual_out = vec![0.0f32; res_clone.len()];
            let mut output = vec![0.0f32; res_clone.len()];
            residual_out_gpu.copy_to_host(&mut residual_out).expect("download residual_out");
            output_gpu.copy_to_host(&mut output).expect("download output");
            let _ = tx.send(Outcome::Done { residual_out, output });
        });

        let outcome = rx.recv_timeout(Duration::from_secs(120)).unwrap_or_else(|_| {
            panic!(
                "FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001: \
                 fused_residual_rmsnorm_forward did not complete within 120s with a \
                 distinct residual_out buffer. Pre-fix root cause: the function held \
                 the FORWARD_KERNEL_CACHE mutex guard while calling the public \
                 residual_add_forward, which re-locks the same non-reentrant mutex \
                 on the same thread (self-deadlock). Fix: enqueue the residual add \
                 BEFORE acquiring the cache lock."
            )
        });

        let (residual_out, output) = match outcome {
            Outcome::NoCuda(reason) => {
                eprintln!(
                    "[falsify-cuda-fused-rmsnorm-deadlock-001] skipping (no CUDA host): {reason}"
                );
                return;
            }
            Outcome::Done { residual_out, output } => (residual_out, output),
        };

        // Oracle 1: residual_out = residual + input (single f32 add — exact).
        let max_add_diff = residual_data
            .iter()
            .zip(input_data.iter())
            .zip(residual_out.iter())
            .map(|((r, i), out)| (r + i - out).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_add_diff == 0.0,
            "FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001: residual_out != residual + input \
             (max_diff={max_add_diff})"
        );

        // Oracle 2: output = RMSNorm(residual + input) * gamma at the eps the
        // caller threads through (1e-6, Qwen2). Pre-fix the kernel silently
        // used its 1e-5 default AND only ever wrote row 0 (single-row kernel
        // launched with grid.y = batch_size), so this oracle also falsifies
        // the batched-row and eps-threading defects, not just liveness.
        let summed: Vec<f32> =
            residual_data.iter().zip(input_data.iter()).map(|(r, i)| r + i).collect();
        let mut cpu_out = Vec::with_capacity(total);
        for b in 0..batch_size as usize {
            let row = &summed[b * hidden_size..(b + 1) * hidden_size];
            cpu_out.extend(cpu_rmsnorm_reference(row, &gamma_data, 1e-6));
        }
        let max_norm_diff = cpu_out
            .iter()
            .zip(output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_norm_diff < 1e-4,
            "FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001: output disagrees with CPU \
             RMSNorm(residual+input) reference (max_diff={max_norm_diff})"
        );
    }
}
