#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    BatchedToInterleavedKernel, BatchedTransposeKernel, ElementwiseMulKernel,
    InterleavedToBatchedKernel, Kernel, ResidualAddKernel, ScaleKernel,
};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

/// Residual addition forward pass on GPU
///
/// Computes: output[i] = a[i] + b[i] for i in [0, n)
///
/// # Contract (C-RESADD-001)
///
/// - **Precondition**: a.len() == b.len() == output.len() >= n, n > 0
/// - **Postcondition**: output[i] == a[i] + b[i] for all i in [0, n)
/// - **Invariant**: Zero CPU-side data transfers (no gpu_to_vec / vec_to_gpu)
#[cfg(feature = "cuda")]
pub fn residual_add_forward(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "residual_add_forward".to_string(); // PTX is n-independent (trueno#184)
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = ResidualAddKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &a_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "residual_add", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Residual add forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// In-place GPU buffer addition: dst[i] += src[i]
///
/// # Contract (C-IPADD-001)
///
/// - **Precondition**: dst.len() >= n, src.len() >= n, n > 0
/// - **Postcondition**: dst[i] == old_dst[i] + src[i] for all i in [0, n)
/// - **Invariant**: Zero CPU-side data transfers, zero stream synchronization
#[cfg(feature = "cuda")]
pub fn inplace_add_gpu(
    dst: &mut GpuBuffer<f32>,
    src: &GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "inplace_add".to_string(); // PTX is n-independent (trueno#184)
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = ResidualAddKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    // Use dst pointer for both input `a` and `output` — in-place accumulation.
    // SAFETY: ResidualAddKernel computes output[i] = a[i] + b[i] per thread.
    // With a == output (aliased), each thread reads dst[i], adds src[i], writes dst[i].
    // No inter-thread data dependency — safe for in-place operation.
    let dst_ptr = dst.as_ptr();
    let src_ptr = src.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &dst_ptr as *const _ as *mut _,
        &src_ptr as *const _ as *mut _,
        &dst_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: kernel launch with pre-validated device pointers and grid config;
    // both src and dst are valid CudaTensor buffers with length >= n elements.
    unsafe {
        stream.launch_kernel(module, "residual_add", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("In-place add launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Element-wise multiplication forward pass on GPU
///
/// Computes: output[i] = a[i] * b[i] for i in [0, n)
///
/// # Contract (C-ELMUL-001)
///
/// - **Precondition**: a.len() == b.len() == output.len() >= n, n > 0
/// - **Postcondition**: output[i] == a[i] * b[i] for all i in [0, n)
/// - **Invariant**: Zero CPU-side data transfers
#[cfg(feature = "cuda")]
pub fn elementwise_mul_forward(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "elementwise_mul_forward".to_string(); // PTX is n-independent (trueno#184)
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = ElementwiseMulKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &a_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "elementwise_mul", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Elementwise mul forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Scale forward pass on GPU
///
/// Computes: output[i] = input[i] * scale for i in [0, n)
///
/// # Contract (C-SCALE-001)
///
/// - **Precondition**: input.len() == output.len() >= n, n > 0
/// - **Postcondition**: output[i] == input[i] * scale for all i in [0, n)
/// - **Invariant**: Zero CPU-side data transfers; in-place aliasing allowed (output may == input)
#[cfg(feature = "cuda")]
pub fn scale_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    scale: f32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = "scale_forward".to_string(); // PTX is n-independent (trueno#184)
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let kernel = ScaleKernel::new(n);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &scale as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "scale", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Scale forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Convert interleaved to batched layout on GPU
///
/// Transforms: [seq_len, n_heads * head_dim] → [n_heads, seq_len, head_dim]
///
/// Used to prepare Q/K/V for batched multi-head attention GEMM.
///
/// # Contract (C-I2B-001)
///
/// - **Precondition**: input.len() >= seq_len * n_heads * head_dim, output.len() >= same
/// - **Postcondition**: output[h, s, d] = input[s, h * head_dim + d]
/// - **Invariant**: Zero CPU-side data transfers; total element count preserved
#[cfg(feature = "cuda")]
pub fn interleaved_to_batched_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let total = seq_len * n_heads * head_dim;
    // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
    // Use generic cache key — PTX is dimension-independent, one module handles all dims.
    let key = "interleaved_to_batched";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            // Constructor args don't matter — PTX is identical for any dimensions
            let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(key, &ptx)?
        }
    };

    let config =
        LaunchConfig { grid: (total.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    // Dimension-independent kernel: pass dims as runtime params
    let mut args: [*mut std::ffi::c_void; 6] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &seq_len as *const _ as *mut _,
        &n_heads as *const _ as *mut _,
        &head_dim as *const _ as *mut _,
        &total as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations.
    unsafe {
        stream.launch_kernel(module, "interleaved_to_batched", &config, &mut args).map_err(
            |e| {
                CudaTensorError::KernelError(format!("Interleaved-to-batched launch failed: {e:?}"))
            },
        )?;
    }

    Ok(())
}

/// Batched transpose on GPU
///
/// Transforms: [batch, rows, cols] → [batch, cols, rows]
///
/// Used for K^T in attention: [n_heads, seq_len, head_dim] → [n_heads, head_dim, seq_len]
///
/// # Contract (C-BTRANS-001)
///
/// - **Precondition**: input.len() >= batch * rows * cols, output.len() >= same
/// - **Postcondition**: output[b, j, i] = input[b, i, j]
/// - **Invariant**: Zero CPU-side data transfers; total element count preserved
#[cfg(feature = "cuda")]
pub fn batched_transpose_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch: u32,
    rows: u32,
    cols: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let total_per_batch = rows * cols;
    // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
    let key = "batched_transpose";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let kernel = BatchedTransposeKernel::new(batch, rows, cols);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(key, &ptx)?
        }
    };

    // Grid: (ceil(total_per_batch/256), 1, batch)
    let config = LaunchConfig {
        grid: (total_per_batch.div_ceil(256), 1, batch),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    // Dimension-independent kernel: pass dims as runtime params
    let mut args: [*mut std::ffi::c_void; 6] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &batch as *const _ as *mut _,
        &rows as *const _ as *mut _,
        &cols as *const _ as *mut _,
        &total_per_batch as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations.
    unsafe {
        stream.launch_kernel(module, "batched_transpose", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Batched transpose launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Convert batched to interleaved layout on GPU
///
/// Transforms: [n_heads, seq_len, head_dim] → [seq_len, n_heads * head_dim]
///
/// Used to convert attention output back to interleaved layout for output projection.
///
/// # Contract (C-B2I-001)
///
/// - **Precondition**: input.len() >= n_heads * seq_len * head_dim, output.len() >= same
/// - **Postcondition**: output[s, h * head_dim + d] = input[h, s, d]
/// - **Invariant**: Zero CPU-side data transfers; total element count preserved
#[cfg(feature = "cuda")]
pub fn batched_to_interleaved_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let total = seq_len * n_heads * head_dim;
    // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
    let key = "batched_to_interleaved";
    let module = match cache.get_cached(key) {
        Some(m) => m,
        None => {
            let kernel = BatchedToInterleavedKernel::new(seq_len, n_heads, head_dim);
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(key, &ptx)?
        }
    };

    let config =
        LaunchConfig { grid: (total.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    // Dimension-independent kernel: pass dims as runtime params
    let mut args: [*mut std::ffi::c_void; 6] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &seq_len as *const _ as *mut _,
        &n_heads as *const _ as *mut _,
        &head_dim as *const _ as *mut _,
        &total as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations.
    unsafe {
        stream.launch_kernel(module, "batched_to_interleaved", &config, &mut args).map_err(
            |e| {
                CudaTensorError::KernelError(format!("Batched-to-interleaved launch failed: {e:?}"))
            },
        )?;
    }

    Ok(())
}

/// Expand KV heads for grouped-query attention (GQA) on GPU
///
/// Replicates each KV head `heads_per_kv` times using D2D copies.
/// Transforms: [num_kv_heads, seq_len, head_dim] → [num_heads, seq_len, head_dim]
///
/// # Contract (C-GQAEXP-001)
///
/// - **Precondition**: src has at least num_kv_heads * elems_per_head elements,
///   dst has at least num_kv_heads * heads_per_kv * elems_per_head elements
/// - **Postcondition**: dst[h, :, :] = src[h / heads_per_kv, :, :] for all h in [0, num_heads)
/// - **Invariant**: Zero CPU-side data transfers (D2D only)
#[cfg(feature = "cuda")]
pub fn expand_kv_heads(
    src: &GpuBuffer<f32>,
    dst: &mut GpuBuffer<f32>,
    num_kv_heads: usize,
    heads_per_kv: usize,
    elems_per_head: usize,
    stream: &CudaStream,
) -> Result<()> {
    for kv_h in 0..num_kv_heads {
        let src_offset = kv_h * elems_per_head;
        for rep in 0..heads_per_kv {
            let dst_offset = (kv_h * heads_per_kv + rep) * elems_per_head;
            // SAFETY: Both buffers are valid GPU allocations with sufficient size.
            // The async D2D copy is ordered on the stream with prior kernel launches.
            unsafe {
                dst.copy_from_buffer_at_async(src, dst_offset, src_offset, elems_per_head, stream)
                    .map_err(|e| {
                        CudaTensorError::TransferFailed(format!(
                            "GQA head expansion D2D copy failed: {e}"
                        ))
                    })?;
            }
        }
    }
    Ok(())
}
