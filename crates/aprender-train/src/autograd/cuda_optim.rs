//! CUDA-accelerated optimizer kernels for autograd
//!
//! This module wraps trueno-gpu optimizer kernels for GPU-resident weight updates.
//! Eliminates CPU↔GPU synchronization by keeping all optimizer state on GPU.
//!
//! # Architecture (SPEC-FT-001 v3.1.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_optim (this module)
//!             └── trueno-gpu/kernels/optimizer
//!                     └── AdamWStepKernel, AdamStepKernel, GradientClipKernel
//! ```
//!
//! # Available Functions
//!
//! - `adamw_step_cuda` - Fused AdamW with weight decay
//! - `adam_step_cuda` - Vanilla Adam without weight decay
//! - `gradient_clip_cuda` - Apply gradient clipping scale
//! - `squared_sum_cuda` - GPU-side sum-of-squares for L2 norm (KAIZEN-049)

#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::backward::{FusedCausalCrossEntropyKernel, FusedCrossEntropyKernel};
use trueno_gpu::kernels::{
    AdamStepKernel, AdamWStepKernel, ClipScaleReduceKernel, GradientClipGpuScaleKernel,
    GradientClipKernel, Kernel, SquaredSumKernel,
};

use super::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for optimizer kernels
#[cfg(feature = "cuda")]
static OPTIM_KERNEL_CACHE: OnceLock<Mutex<OptimKernelCache>> = OnceLock::new();

/// Cache for compiled optimizer kernel modules
#[cfg(feature = "cuda")]
struct OptimKernelCache {
    ctx: std::sync::Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
    sm_target: String,
}

#[cfg(feature = "cuda")]
impl OptimKernelCache {
    fn new(ctx: std::sync::Arc<CudaContext>) -> Self {
        let sm_target = ctx.sm_target().unwrap_or_else(|_| "sm_70".to_string());
        Self { ctx, modules: HashMap::new(), sm_target }
    }

    fn sm_target(&self) -> &str {
        &self.sm_target
    }

    /// Look up a previously compiled module by key (KAIZEN-058).
    fn get_cached(&mut self, name: &str) -> Option<&mut CudaModule> {
        self.modules.get_mut(name)
    }

    fn get_or_compile(&mut self, name: &str, ptx: &str) -> Result<&mut CudaModule> {
        if !self.modules.contains_key(name) {
            let module = CudaModule::from_ptx(&self.ctx, ptx).map_err(|e| {
                CudaTensorError::KernelError(format!("Failed to compile {name}: {e:?}"))
            })?;
            self.modules.insert(name.to_string(), module);
        }
        Ok(self.modules.get_mut(name).expect("module was just inserted above"))
    }
}

/// Initialize optimizer kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_optim_kernel_cache(ctx: std::sync::Arc<CudaContext>) -> Result<()> {
    OPTIM_KERNEL_CACHE.get_or_init(|| Mutex::new(OptimKernelCache::new(ctx)));
    Ok(())
}

/// Pre-warm AdamW optimizer kernels for all trainable parameter sizes (ENT-153).
///
/// The kernel key is `adamw_step_{n}` where `n` is parameter count.
///
/// ## LoRA mode (NF4 QLoRA)
/// - `hidden * rank` for A_q, A_v
/// - `rank * q_dim` for B_q
/// - `rank * kv_hidden` for B_v
/// - `hidden_size` for norm weights
///
/// ## Full fp32 mode (non-NF4)
/// - `hidden * hidden` for w_q, w_o
/// - `hidden * kv_hidden` for w_k, w_v
/// - `hidden * intermediate` for w_gate, w_up, w_down
/// - `hidden_size` for norm weights
///
/// ## Classifier head (both modes)
/// - `num_classes * hidden_size` for classifier weight
/// - `num_classes` for classifier bias
///
/// Must JIT-compile before block upload fills VRAM (C-PREWARM-001).
#[cfg(feature = "cuda")]
pub fn pre_warm_lora_adamw_kernels(
    hidden_size: usize,
    q_dim: usize,
    kv_hidden_size: usize,
    lora_rank: usize,
    num_classes: usize,
    intermediate_size: usize,
    quantize_nf4: bool,
) -> Result<()> {
    if lora_rank == 0 {
        return Ok(());
    }

    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire optim kernel cache lock".to_string())
    })?;

    let target = cache.sm_target().to_string();

    let mut sizes: Vec<u32> = vec![
        (hidden_size * lora_rank) as u32,    // A_q, A_v
        (lora_rank * q_dim) as u32,          // B_q
        (lora_rank * kv_hidden_size) as u32, // B_v
        hidden_size as u32,                  // norm weights
    ];

    // Full fp32 weight sizes (non-NF4 mode: optimizer runs on all block weights)
    if !quantize_nf4 {
        sizes.push((hidden_size * hidden_size) as u32); // w_q, w_o
        sizes.push((hidden_size * kv_hidden_size) as u32); // w_k, w_v
        sizes.push((hidden_size * intermediate_size) as u32); // w_gate, w_up, w_down
    }

    // Classifier head sizes
    if num_classes > 0 {
        sizes.push((num_classes * hidden_size) as u32);
        sizes.push(num_classes as u32);
    }

    sizes.sort_unstable();
    sizes.dedup();

    for n in sizes {
        let kernel = AdamWStepKernel::new(n);
        let ptx = kernel.emit_ptx_for_target(&target);
        let key = format!("adamw_step_{n}");
        cache.get_or_compile(&key, &ptx)?;
    }

    Ok(())
}

/// Fused AdamW optimizer step on GPU
///
/// Performs in-place weight update with momentum, adaptive learning rate, and weight decay.
///
/// # Arguments
/// - `params`: weight tensor (updated in-place)
/// - `grads`: gradient tensor
/// - `m`: first moment state (updated in-place)
/// - `v`: second moment state (updated in-place)
/// - `lr`: learning rate
/// - `beta1`: first moment decay (typically 0.9)
/// - `beta2`: second moment decay (typically 0.999)
/// - `eps`: numerical stability (typically 1e-8)
/// - `weight_decay`: L2 penalty coefficient
/// - `step`: current step number (for bias adjust)
/// - `n`: number of parameters
/// - `stream`: CUDA stream
#[cfg(feature = "cuda")]
pub fn adamw_step_cuda(
    params: &mut GpuBuffer<f32>,
    grads: &GpuBuffer<f32>,
    m: &mut GpuBuffer<f32>,
    v: &mut GpuBuffer<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("adamw_step_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = AdamWStepKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    // Pre-compute bias adjust factors
    let bias_adjust1 = 1.0 / (1.0 - beta1.powi(step as i32));
    let bias_adjust2 = 1.0 / (1.0 - beta2.powi(step as i32));

    let params_ptr = params.as_ptr();
    let grads_ptr = grads.as_ptr();
    let m_ptr = m.as_ptr();
    let v_ptr = v.as_ptr();

    let mut args: [*mut std::ffi::c_void; 12] = [
        &params_ptr as *const _ as *mut _,
        &grads_ptr as *const _ as *mut _,
        &m_ptr as *const _ as *mut _,
        &v_ptr as *const _ as *mut _,
        &lr as *const _ as *mut _,
        &beta1 as *const _ as *mut _,
        &beta2 as *const _ as *mut _,
        &eps as *const _ as *mut _,
        &weight_decay as *const _ as *mut _,
        &bias_adjust1 as *const _ as *mut _,
        &bias_adjust2 as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "adamw_step", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("AdamW step launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Fused Adam optimizer step on GPU (no weight decay)
///
/// Same as `adamw_step_cuda` but without the decoupled weight decay term.
#[cfg(feature = "cuda")]
pub fn adam_step_cuda(
    params: &mut GpuBuffer<f32>,
    grads: &GpuBuffer<f32>,
    m: &mut GpuBuffer<f32>,
    v: &mut GpuBuffer<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("adam_step_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = AdamStepKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    // Pre-compute bias adjust factors
    let bias_adjust1 = 1.0 / (1.0 - beta1.powi(step as i32));
    let bias_adjust2 = 1.0 / (1.0 - beta2.powi(step as i32));

    let params_ptr = params.as_ptr();
    let grads_ptr = grads.as_ptr();
    let m_ptr = m.as_ptr();
    let v_ptr = v.as_ptr();

    let mut args: [*mut std::ffi::c_void; 11] = [
        &params_ptr as *const _ as *mut _,
        &grads_ptr as *const _ as *mut _,
        &m_ptr as *const _ as *mut _,
        &v_ptr as *const _ as *mut _,
        &lr as *const _ as *mut _,
        &beta1 as *const _ as *mut _,
        &beta2 as *const _ as *mut _,
        &eps as *const _ as *mut _,
        &bias_adjust1 as *const _ as *mut _,
        &bias_adjust2 as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "adam_step", &config, &mut args)
            .map_err(|e| CudaTensorError::KernelError(format!("Adam step launch failed: {e:?}")))?;
    }

    Ok(())
}

/// Apply gradient clipping on GPU
///
/// Scales gradients by a pre-computed factor to enforce maximum norm.
///
/// # Arguments
/// - `grads`: gradient tensor (updated in-place)
/// - `scale`: clipping scale factor (pre-computed as `min(1.0, max_norm / grad_norm)`)
/// - `n`: number of gradient elements
/// - `stream`: CUDA stream
///
/// # Usage
/// ```ignore
/// // Compute gradient norm on host
/// let grad_norm = compute_l2_norm(&grads);
/// let scale = (max_norm / grad_norm).min(1.0);
///
/// // Apply clipping on GPU
/// gradient_clip_cuda(&mut grads, scale, n, &stream)?;
/// ```
#[cfg(feature = "cuda")]
pub fn gradient_clip_cuda(
    grads: &mut GpuBuffer<f32>,
    scale: f32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    // Skip kernel launch if no clipping needed
    if (scale - 1.0).abs() < 1e-7 {
        return Ok(());
    }

    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gradient_clip_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = GradientClipKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let grads_ptr = grads.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] =
        [&grads_ptr as *const _ as *mut _, &scale as *const _ as *mut _, &n as *const _ as *mut _];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "gradient_clip", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Gradient clip launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// GPU-side sum-of-squares reduction (KAIZEN-049).
///
/// Computes `sum(input[i]^2)` entirely on GPU, returning only `num_blocks` partial sums
/// (~1KB) to host. Host finishes with f64 summation and sqrt for the L2 norm.
///
/// # Contract (C-SQSUM-002)
///
/// - **Precondition**: `n > 0`, `input` has at least `n` elements
/// - **Postcondition**: returned f32 = sqrt(sum(input[i]^2)) to within O(n × eps_f32)
/// - **Transfer**: ~1KB D2H instead of n×4 bytes (128MB for 32M elements)
///
/// # Errors
///
/// Returns `Err` if kernel cache not initialized, kernel compilation fails, or GPU transfer fails.
#[cfg(feature = "cuda")]
pub fn squared_sum_cuda(input: &GpuBuffer<f32>, n: u32, stream: &CudaStream) -> Result<f32> {
    let pending = squared_sum_launch_cuda(input, n, stream)?;
    stream
        .synchronize()
        .map_err(|e| CudaTensorError::KernelError(format!("Stream sync failed: {e:?}")))?;
    squared_sum_collect(&pending)
}

/// Launched but not-yet-collected squared sum reduction.
///
/// Holds the GPU buffer of partial sums until `squared_sum_collect` downloads them.
#[cfg(feature = "cuda")]
pub struct PendingSquaredSum {
    output: GpuBuffer<f32>,
    num_blocks: u32,
}

/// Launch a squared sum reduction kernel without synchronizing (KAIZEN-055).
///
/// Returns a `PendingSquaredSum` handle. The caller MUST call `stream.synchronize()`
/// before calling `squared_sum_collect()` on the handle.
///
/// This allows batching multiple reductions with a single sync point:
/// ```ignore
/// let p1 = squared_sum_launch_cuda(&buf1, n1, stream)?;
/// let p2 = squared_sum_launch_cuda(&buf2, n2, stream)?;
/// stream.synchronize()?;  // single sync for both
/// let norm1 = squared_sum_collect(&p1)?;
/// let norm2 = squared_sum_collect(&p2)?;
/// ```
#[cfg(feature = "cuda")]
pub fn squared_sum_launch_cuda(
    input: &GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<PendingSquaredSum> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = SquaredSumKernel::new(n);
    let num_blocks = kernel.num_blocks();

    // Clone ctx before mutable borrow via get_or_compile/get_cached
    let ctx = std::sync::Arc::clone(&cache.ctx);

    let key = format!("squared_sum_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Allocate output buffer for block partial sums (num_blocks × 4 bytes, typically ≤1KB)
    let output = GpuBuffer::<f32>::new(&ctx, num_blocks as usize).map_err(|e| {
        CudaTensorError::KernelError(format!("Failed to allocate squared_sum output: {e:?}"))
    })?;

    let config = LaunchConfig {
        grid: (num_blocks, 1, 1),
        block: (kernel.block_size(), 1, 1),
        shared_mem: 8 * 4, // 8 warp partials × 4 bytes
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. input has n elements, output has num_blocks elements,
    // parameters match PTX signature (u64 input_ptr, u64 output_ptr, u32 n).
    unsafe {
        stream.launch_kernel(module, "squared_sum_reduce", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Squared sum launch failed: {e:?}"))
        })?;
    }

    Ok(PendingSquaredSum { output, num_blocks })
}

/// Collect the result of a previously launched squared sum reduction (KAIZEN-055).
///
/// **Precondition**: `stream.synchronize()` must have been called after the launch.
#[cfg(feature = "cuda")]
pub fn squared_sum_collect(pending: &PendingSquaredSum) -> Result<f32> {
    let mut partials = vec![0.0f32; pending.num_blocks as usize];
    pending.output.copy_to_host(&mut partials).map_err(|e| {
        CudaTensorError::KernelError(format!("Failed to download partial sums: {e:?}"))
    })?;

    // Sum partials in f64 for precision, then sqrt for L2 norm
    let total: f64 = partials.iter().map(|&x| f64::from(x)).sum();
    Ok(total.sqrt() as f32)
}

/// Launch a squared sum reduction writing into a pre-allocated output buffer at offset (ALB-078).
///
/// Unlike `squared_sum_launch_cuda` which allocates a fresh output buffer, this writes
/// partial sums to `output_ptr + output_offset_elements * 4` in an existing buffer.
/// Used by the fused clip pipeline to collect all 9 groups' partials into one contiguous buffer.
///
/// # Returns
///
/// Number of blocks written (= number of f32 partial sums at the output offset).
#[cfg(feature = "cuda")]
pub fn squared_sum_launch_into(
    input: &GpuBuffer<f32>,
    n: u32,
    output_ptr: u64, // raw device pointer with offset already applied
    stream: &CudaStream,
) -> Result<u32> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = SquaredSumKernel::new(n);
    let num_blocks = kernel.num_blocks();

    let key = format!("squared_sum_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig {
        grid: (num_blocks, 1, 1),
        block: (kernel.block_size(), 1, 1),
        shared_mem: 8 * 4, // 8 warp partials × 4 bytes
    };

    let input_ptr = input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. input has n elements, output region has num_blocks elements,
    // parameters match PTX signature (u64 input_ptr, u64 output_ptr, u32 n).
    unsafe {
        stream.launch_kernel(module, "squared_sum_reduce", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Squared sum launch_into failed: {e:?}"))
        })?;
    }

    Ok(num_blocks)
}

/// Launch the clip scale reduction kernel on GPU (ALB-078).
///
/// Reads a contiguous buffer of squared-sum partial results, computes the global
/// L2 norm and clip scale entirely on GPU. Writes `[scale, norm]` to `output`.
///
/// # Contract (C-FUSEDCLIP-001)
///
/// - output[0] = min(1.0, max_norm / sqrt(sum(partials[0..total_n])))
/// - output[1] = sqrt(sum(partials[0..total_n]))
#[cfg(feature = "cuda")]
pub fn clip_scale_reduce_cuda(
    partials: &GpuBuffer<f32>,
    total_n: u32,
    max_norm: f32,
    output: &GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "clip_scale_reduce".to_string();
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = ClipScaleReduceKernel;
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Single CTA, single thread — partials buffer is tiny (~1800 f32 for 350M)
    let config = LaunchConfig { grid: (1, 1, 1), block: (1, 1, 1), shared_mem: 0 };

    let partials_ptr = partials.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &partials_ptr as *const _ as *mut _,
        &total_n as *const _ as *mut _,
        &max_norm as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
    ];

    // SAFETY: partials has total_n elements, output has 2 elements (scale + norm).
    unsafe {
        stream.launch_kernel(module, "clip_scale_reduce", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Clip scale reduce launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Apply gradient clipping with scale read from GPU memory (ALB-078).
///
/// Like `gradient_clip_cuda` but reads the scale from a GPU pointer instead of
/// a host f32. This avoids D2H transfer of the clip scale.
///
/// The kernel exits early if scale ≈ 1.0, avoiding unnecessary write bandwidth.
#[cfg(feature = "cuda")]
pub fn gradient_clip_gpu_scale_cuda(
    grads: &mut GpuBuffer<f32>,
    scale_ptr: u64, // device pointer to f32 scale value
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gradient_clip_gpu_scale_{n}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = GradientClipGpuScaleKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let grads_ptr = grads.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &grads_ptr as *const _ as *mut _,
        &scale_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: grads has n elements, scale_ptr points to valid f32 on GPU.
    unsafe {
        stream.launch_kernel(module, "gradient_clip_gpu_scale", &config, &mut args).map_err(
            |e| {
                CudaTensorError::KernelError(format!(
                    "Gradient clip GPU scale launch failed: {e:?}"
                ))
            },
        )?;
    }

    Ok(())
}

/// Pre-allocated state for fused gradient clipping pipeline (ALB-078).
///
/// Holds GPU buffers for the contiguous partial-sum collection and clip scale output.
/// Initialized once at trainer creation, reused every step.
#[cfg(feature = "cuda")]
pub struct FusedClipState {
    /// Contiguous buffer for all squared-sum partial results across 9 gradient groups
    pub partials_buf: GpuBuffer<f32>,
    /// Output buffer: [clip_scale, grad_norm] (2 × f32)
    pub scale_buf: GpuBuffer<f32>,
    /// Byte offset into partials_buf for each of the 9 gradient groups
    pub offsets: [u32; 9],
    /// Number of partial-sum blocks for each group
    pub num_blocks: [u32; 9],
    /// Total number of partial sums across all groups
    pub total_partials: u32,
}

#[cfg(feature = "cuda")]
impl FusedClipState {
    /// Create a new fused clip state for the given gradient buffer sizes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `grad_sizes` - sizes (in elements) of the 9 gradient buffers:
    ///   [w_q, w_k, w_v, w_o, gate, up, down, input_norm, post_attn_norm]
    pub fn new(ctx: &std::sync::Arc<CudaContext>, grad_sizes: &[u32; 9]) -> Result<Self> {
        let mut offsets = [0u32; 9];
        let mut num_blocks_arr = [0u32; 9];
        let mut total = 0u32;

        for (i, &n) in grad_sizes.iter().enumerate() {
            offsets[i] = total;
            let kernel = SquaredSumKernel::new(n);
            let nb = kernel.num_blocks();
            num_blocks_arr[i] = nb;
            total += nb;
        }

        let partials_buf = GpuBuffer::<f32>::new(ctx, total as usize).map_err(|e| {
            CudaTensorError::KernelError(format!("Failed to allocate partials buffer: {e:?}"))
        })?;

        let scale_buf = GpuBuffer::<f32>::new(ctx, 2).map_err(|e| {
            CudaTensorError::KernelError(format!("Failed to allocate scale buffer: {e:?}"))
        })?;

        Ok(Self {
            partials_buf,
            scale_buf,
            offsets,
            num_blocks: num_blocks_arr,
            total_partials: total,
        })
    }
}

/// Fused GPU cross-entropy loss + softmax backward, in-place (KAIZEN-050 + KAIZEN-052).
///
/// Computes cross-entropy loss and writes gradient **in-place** to the logits buffer,
/// eliminating both the logits D2H (77.8MB) + CPU softmax (40ms) + gradient H2D (77.8MB)
/// AND the separate gradient buffer allocation (77.8MB for Qwen3-4B).
///
/// # Returns
///
/// Scalar loss (averaged over seq_len). Gradient is written in-place to `logits_buf`.
///
/// # Contract (C-XENT-002, updated KAIZEN-052)
///
/// - **Precondition**: `logits_buf` has `seq_len * vocab_size` elements, targets in `[0, vocab_size)`
/// - **Postcondition**: `logits_buf[i] = (softmax - one_hot) * scale` (gradient, in-place)
/// - **Postcondition**: `loss = mean(-log(softmax[target]))`
/// - **Transfer**: H2D targets (seq_len×4 bytes) + D2H loss_partials (seq_len×4 bytes)
/// - **Allocation**: 0 bytes grad buffer (was 77.8MB before KAIZEN-052)
#[cfg(feature = "cuda")]
pub fn fused_cross_entropy_cuda(
    logits_buf: &mut GpuBuffer<f32>,
    target_ids: &[u32],
    seq_len: u32,
    vocab_size: u32,
    scale: f32,
    stream: &CudaStream,
) -> Result<f32> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = FusedCrossEntropyKernel::new(vocab_size);

    // Clone ctx before mutable borrow via get_or_compile/get_cached
    let ctx = std::sync::Arc::clone(&cache.ctx);

    let key = format!("fused_xent_{vocab_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Upload targets to GPU (seq_len × u32 = ~512 bytes for seq_len=128)
    let targets_u32: Vec<u32> = target_ids[..seq_len as usize].to_vec();
    let targets_gpu = GpuBuffer::<u32>::from_host(&ctx, &targets_u32)
        .map_err(|e| CudaTensorError::KernelError(format!("Failed to upload targets: {e:?}")))?;

    // KAIZEN-052: No grad_gpu allocation — gradient written in-place to logits_buf.

    // Allocate loss partials buffer (seq_len × f32 — downloaded for scalar average)
    let loss_gpu = GpuBuffer::<f32>::new(&ctx, seq_len as usize).map_err(|e| {
        CudaTensorError::KernelError(format!("Failed to allocate loss partials: {e:?}"))
    })?;

    // Shared memory: 72 bytes (8 warp maxes + global max + 8 warp sums + global sum)
    let config =
        LaunchConfig { grid: (seq_len, 1, 1), block: (kernel.block_size(), 1, 1), shared_mem: 72 };

    let logits_grad_ptr = logits_buf.as_ptr();
    let targets_ptr = targets_gpu.as_ptr();
    let loss_ptr = loss_gpu.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        &logits_grad_ptr as *const _ as *mut _,
        &targets_ptr as *const _ as *mut _,
        &loss_ptr as *const _ as *mut _,
        &vocab_size as *const _ as *mut _,
        &scale as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. logits_buf has seq_len*vocab_size elements
    // (read as logits, overwritten with gradients in-place). targets_gpu has seq_len u32
    // elements, loss_gpu has seq_len f32 elements. Parameters match PTX signature
    // (u64 logits_grad_ptr, u64 targets_ptr, u64 loss_ptr, u32 vocab_size, f32 scale).
    unsafe {
        stream.launch_kernel(module, "fused_cross_entropy", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Fused cross-entropy launch failed: {e:?}"))
        })?;
    }

    // Synchronize and download loss partials (~512 bytes)
    stream
        .synchronize()
        .map_err(|e| CudaTensorError::KernelError(format!("Stream sync failed: {e:?}")))?;

    let mut loss_partials = vec![0.0f32; seq_len as usize];
    loss_gpu.copy_to_host(&mut loss_partials).map_err(|e| {
        CudaTensorError::KernelError(format!("Failed to download loss partials: {e:?}"))
    })?;

    // Average loss across sequence positions (f64 for precision)
    let total_loss: f64 = loss_partials.iter().map(|&x| f64::from(x)).sum();
    let avg_loss = (total_loss / f64::from(seq_len)) as f32;

    Ok(avg_loss)
}

/// Fused GPU causal cross-entropy loss + softmax backward, in-place (KAIZEN-064).
///
/// Like `fused_cross_entropy_cuda` but with causal LM masking: only positions
/// `[loss_start, loss_end)` contribute to loss and gradient. Positions outside
/// this range get zero gradient. Eliminates:
/// - Logits D2H download (~296MB for Qwen3-4B)
/// - CPU softmax computation
/// - Gradient H2D upload (~296MB for Qwen3-4B)
///
/// # Contract (C-CAUSALXENT-001)
///
/// - **Precondition**: logits_buf has `seq_len * vocab_size` valid elements
/// - **Precondition**: `loss_start < loss_end <= seq_len`
/// - **Postcondition**: logits_buf overwritten with gradient (in-place)
/// - **Postcondition**: Returns average loss over `[loss_start, loss_end)` only
#[cfg(feature = "cuda")]
pub fn fused_causal_cross_entropy_cuda(
    logits_buf: &mut GpuBuffer<f32>,
    target_ids: &[u32],
    seq_len: u32,
    vocab_size: u32,
    loss_start: u32,
    loss_end: u32,
    scale: f32,
    stream: &CudaStream,
) -> Result<f32> {
    let cache = OPTIM_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = FusedCausalCrossEntropyKernel::new(vocab_size);

    let ctx = std::sync::Arc::clone(&cache.ctx);

    let key = format!("fused_causal_xent_{vocab_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Upload targets to GPU (seq_len × u32 = ~2KB for seq_len=512)
    let targets_u32: Vec<u32> = target_ids[..seq_len as usize].to_vec();
    let targets_gpu = GpuBuffer::<u32>::from_host(&ctx, &targets_u32)
        .map_err(|e| CudaTensorError::KernelError(format!("Failed to upload targets: {e:?}")))?;

    // Allocate loss partials buffer (seq_len × f32)
    let loss_gpu = GpuBuffer::<f32>::new(&ctx, seq_len as usize).map_err(|e| {
        CudaTensorError::KernelError(format!("Failed to allocate loss partials: {e:?}"))
    })?;

    // Shared memory: 72 bytes (same as base kernel)
    let config =
        LaunchConfig { grid: (seq_len, 1, 1), block: (kernel.block_size(), 1, 1), shared_mem: 72 };

    let logits_grad_ptr = logits_buf.as_ptr();
    let targets_ptr = targets_gpu.as_ptr();
    let loss_ptr = loss_gpu.as_ptr();

    let mut args: [*mut std::ffi::c_void; 7] = [
        &logits_grad_ptr as *const _ as *mut _,
        &targets_ptr as *const _ as *mut _,
        &loss_ptr as *const _ as *mut _,
        &vocab_size as *const _ as *mut _,
        &scale as *const _ as *mut _,
        &loss_start as *const _ as *mut _,
        &loss_end as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. logits_buf has seq_len*vocab_size elements
    // (read as logits, overwritten with gradients in-place). targets_gpu has seq_len u32
    // elements. loss_gpu has seq_len f32 elements. Parameters match PTX signature.
    // Positions outside [loss_start, loss_end) get zero gradient and zero loss.
    unsafe {
        stream.launch_kernel(module, "fused_causal_cross_entropy", &config, &mut args).map_err(
            |e| {
                CudaTensorError::KernelError(format!(
                    "Fused causal cross-entropy launch failed: {e:?}"
                ))
            },
        )?;
    }

    // Synchronize and download loss partials (~2KB)
    stream
        .synchronize()
        .map_err(|e| CudaTensorError::KernelError(format!("Stream sync failed: {e:?}")))?;

    let mut loss_partials = vec![0.0f32; seq_len as usize];
    loss_gpu.copy_to_host(&mut loss_partials).map_err(|e| {
        CudaTensorError::KernelError(format!("Failed to download loss partials: {e:?}"))
    })?;

    // Average loss across loss positions only (f64 for precision)
    let num_loss_tokens = loss_end.saturating_sub(loss_start) as usize;
    if num_loss_tokens == 0 {
        return Ok(0.0);
    }
    let total_loss: f64 =
        loss_partials[loss_start as usize..loss_end as usize].iter().map(|&x| f64::from(x)).sum();
    let avg_loss = (total_loss / num_loss_tokens as f64) as f32;

    Ok(avg_loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_optim_module_compiles() {
        // This test verifies the module compiles correctly
        // Actual CUDA tests require GPU hardware
        assert!(true);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_optim_kernel_cache_initialization() {
        use trueno_gpu::driver::cuda_available;

        if !cuda_available() {
            return;
        }

        let ctx = CudaContext::new(0).expect("operation should succeed");
        let ctx = std::sync::Arc::new(ctx);
        let result = init_optim_kernel_cache(ctx);
        assert!(result.is_ok());
    }

    /// Create a fresh GPU context for a test
    /// Note: Using fresh contexts per-test avoids CUDA driver state issues
    /// when running multiple tests sequentially
    #[cfg(feature = "cuda")]
    fn get_test_gpu_context() -> Option<std::sync::Arc<CudaContext>> {
        use trueno_gpu::driver::cuda_available;

        if cuda_available() {
            CudaContext::new(0).ok().map(std::sync::Arc::new)
        } else {
            None
        }
    }

    /// CPU reference implementation for AdamW step
    fn adamw_step_cpu(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let bias_adjust1 = 1.0 / (1.0 - beta1.powi(step as i32));
        let bias_adjust2 = 1.0 / (1.0 - beta2.powi(step as i32));

        for i in 0..params.len() {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            // Update biased second moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Compute bias-corrected estimates
            let m_hat = m[i] * bias_adjust1;
            let v_hat = v[i] * bias_adjust2;

            // AdamW update: weight decay is applied directly to params
            params[i] = params[i] * (1.0 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// CPU reference implementation for Adam step (no weight decay)
    fn adam_step_cpu(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: u32,
    ) {
        let bias_adjust1 = 1.0 / (1.0 - beta1.powi(step as i32));
        let bias_adjust2 = 1.0 / (1.0 - beta2.powi(step as i32));

        for i in 0..params.len() {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            // Update biased second moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Compute bias-corrected estimates
            let m_hat = m[i] * bias_adjust1;
            let v_hat = v[i] * bias_adjust2;

            // Adam update (no weight decay)
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// CPU reference implementation for gradient clipping
    fn gradient_clip_cpu(grads: &mut [f32], scale: f32) {
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adamw_step_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;
        let step = 1u32;

        // Initial values
        let mut params_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grads_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let mut m_data: Vec<f32> = vec![0.0; n as usize];
        let mut v_data: Vec<f32> = vec![0.0; n as usize];

        // CPU reference
        let mut cpu_params = params_data.clone();
        let mut cpu_m = m_data.clone();
        let mut cpu_v = v_data.clone();
        adamw_step_cpu(
            &mut cpu_params,
            &grads_data,
            &mut cpu_m,
            &mut cpu_v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        );

        // GPU execution
        let mut params =
            GpuBuffer::from_host(&ctx, &params_data).expect("operation should succeed");
        let grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");
        let mut m = GpuBuffer::from_host(&ctx, &m_data).expect("operation should succeed");
        let mut v = GpuBuffer::from_host(&ctx, &v_data).expect("operation should succeed");

        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n,
            &stream,
        )
        .expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        params.copy_to_host(&mut params_data).expect("operation should succeed");
        m.copy_to_host(&mut m_data).expect("operation should succeed");
        v.copy_to_host(&mut v_data).expect("operation should succeed");

        // Compare GPU vs CPU results
        for i in 0..n as usize {
            assert!(
                (params_data[i] - cpu_params[i]).abs() < 1e-4,
                "AdamW params mismatch at {i}: GPU={}, CPU={}",
                params_data[i],
                cpu_params[i]
            );
            assert!(
                (m_data[i] - cpu_m[i]).abs() < 1e-5,
                "AdamW m mismatch at {i}: GPU={}, CPU={}",
                m_data[i],
                cpu_m[i]
            );
            assert!(
                (v_data[i] - cpu_v[i]).abs() < 1e-5,
                "AdamW v mismatch at {i}: GPU={}, CPU={}",
                v_data[i],
                cpu_v[i]
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adamw_step_not_hardcoded() {
        // Mutation-killing test: verify params actually change
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let initial_params: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grads_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5]; // Non-zero gradients
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params =
            GpuBuffer::from_host(&ctx, &initial_params).expect("operation should succeed");
        let grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");
        let mut m = GpuBuffer::from_host(&ctx, &m_data).expect("operation should succeed");
        let mut v = GpuBuffer::from_host(&ctx, &v_data).expect("operation should succeed");

        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.01, // Larger LR to see effect
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
            n,
            &stream,
        )
        .expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        let mut result_params = vec![0.0f32; n as usize];
        params.copy_to_host(&mut result_params).expect("operation should succeed");

        // Kill mutant: params should have changed
        assert_ne!(result_params, initial_params, "mutant: AdamW params unchanged after step");
        // Verify params decreased (negative gradient update)
        for (i, (&new, &old)) in result_params.iter().zip(initial_params.iter()).enumerate() {
            assert!(new < old, "AdamW params[{i}] should decrease with positive gradients");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adamw_weight_decay() {
        // Test that weight decay is actually applied
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let params_data: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0]; // Large weights
        let grads_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0]; // Zero gradients
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params =
            GpuBuffer::from_host(&ctx, &params_data).expect("operation should succeed");
        let grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");
        let mut m = GpuBuffer::from_host(&ctx, &m_data).expect("operation should succeed");
        let mut v = GpuBuffer::from_host(&ctx, &v_data).expect("operation should succeed");

        // With zero gradients, only weight decay should affect params
        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.01, // LR
            0.9,
            0.999,
            1e-8,
            0.1, // High weight decay
            1,
            n,
            &stream,
        )
        .expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        let mut result = vec![0.0f32; n as usize];
        params.copy_to_host(&mut result).expect("operation should succeed");

        // With zero gradients, params should decay: p = p * (1 - lr * wd)
        let expected = 10.0 * (1.0 - 0.01 * 0.1);
        for (i, &p) in result.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-3,
                "Weight decay not applied correctly at {i}: got {p}, expected {expected}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adam_step_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let step = 1u32;

        // Initial values
        let mut params_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grads_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let mut m_data: Vec<f32> = vec![0.0; n as usize];
        let mut v_data: Vec<f32> = vec![0.0; n as usize];

        // CPU reference
        let mut cpu_params = params_data.clone();
        let mut cpu_m = m_data.clone();
        let mut cpu_v = v_data.clone();
        adam_step_cpu(
            &mut cpu_params,
            &grads_data,
            &mut cpu_m,
            &mut cpu_v,
            lr,
            beta1,
            beta2,
            eps,
            step,
        );

        // GPU execution
        let mut params =
            GpuBuffer::from_host(&ctx, &params_data).expect("operation should succeed");
        let grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");
        let mut m = GpuBuffer::from_host(&ctx, &m_data).expect("operation should succeed");
        let mut v = GpuBuffer::from_host(&ctx, &v_data).expect("operation should succeed");

        adam_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            lr,
            beta1,
            beta2,
            eps,
            step,
            n,
            &stream,
        )
        .expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        params.copy_to_host(&mut params_data).expect("operation should succeed");
        m.copy_to_host(&mut m_data).expect("operation should succeed");
        v.copy_to_host(&mut v_data).expect("operation should succeed");

        // Compare GPU vs CPU results
        for i in 0..n as usize {
            assert!(
                (params_data[i] - cpu_params[i]).abs() < 1e-4,
                "Adam params mismatch at {i}: GPU={}, CPU={}",
                params_data[i],
                cpu_params[i]
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adam_step_multiple_iterations() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let lr = 0.01f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut params_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grads_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params =
            GpuBuffer::from_host(&ctx, &params_data).expect("operation should succeed");
        let grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");
        let mut m = GpuBuffer::from_host(&ctx, &m_data).expect("operation should succeed");
        let mut v = GpuBuffer::from_host(&ctx, &v_data).expect("operation should succeed");

        // Run 10 steps
        for step in 1..=10 {
            adam_step_cuda(
                &mut params,
                &grads,
                &mut m,
                &mut v,
                lr,
                beta1,
                beta2,
                eps,
                step,
                n,
                &stream,
            )
            .expect("operation should succeed");
        }
        stream.synchronize().expect("operation should succeed");

        params.copy_to_host(&mut params_data).expect("operation should succeed");

        // Params should have decreased significantly after 10 steps
        for &p in &params_data {
            assert!(p < 1.0, "Params should decrease after multiple Adam steps");
            assert!(p > 0.0, "Params should remain positive");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_clip_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let grads_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        let scale = 0.5f32; // Scale down by half

        let mut grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");

        gradient_clip_cuda(&mut grads, scale, n, &stream).expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        let mut result = vec![0.0f32; n as usize];
        grads.copy_to_host(&mut result).expect("operation should succeed");

        // CPU reference
        let mut expected = grads_data.clone();
        gradient_clip_cpu(&mut expected, scale);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Gradient clip mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_clip_no_op() {
        // Test that scale=1.0 is a no-op
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let grads_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let scale = 1.0f32; // No scaling

        let mut grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");

        // This should be a no-op (kernel not even launched)
        gradient_clip_cuda(&mut grads, scale, n, &stream).expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        let mut result = vec![0.0f32; n as usize];
        grads.copy_to_host(&mut result).expect("operation should succeed");

        // Gradients should be unchanged
        for (i, (&got, &exp)) in result.iter().zip(grads_data.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "Gradient clip with scale=1 should not modify values at {i}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_clip_not_hardcoded() {
        // Mutation-killing test
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 4u32;
        let grads_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let scale = 0.1f32;

        let mut grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");

        gradient_clip_cuda(&mut grads, scale, n, &stream).expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        let mut result = vec![0.0f32; n as usize];
        grads.copy_to_host(&mut result).expect("operation should succeed");

        // Kill mutant: result should NOT equal original
        assert_ne!(result, grads_data, "mutant: gradient clip had no effect");

        // Verify scaled values
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_optimizer_large_scale() {
        // Test with larger parameter count
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).expect("operation should succeed");
        let stream = CudaStream::new(&ctx).expect("operation should succeed");

        let n = 1024u32;
        let params_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let grads_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params =
            GpuBuffer::from_host(&ctx, &params_data).expect("operation should succeed");
        let grads = GpuBuffer::from_host(&ctx, &grads_data).expect("operation should succeed");
        let mut m = GpuBuffer::from_host(&ctx, &m_data).expect("operation should succeed");
        let mut v = GpuBuffer::from_host(&ctx, &v_data).expect("operation should succeed");

        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.001,
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
            n,
            &stream,
        )
        .expect("operation should succeed");
        stream.synchronize().expect("operation should succeed");

        let mut result = vec![0.0f32; n as usize];
        params.copy_to_host(&mut result).expect("operation should succeed");

        // Verify no NaN or Inf
        assert!(
            !result.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Large-scale optimizer should not produce NaN/Inf"
        );
    }
}
