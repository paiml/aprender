//! Layout conversion helpers for batched attention operations.
//!
//! Functions for converting between interleaved and batched tensor layouts,
//! and batched transpose operations.

#[cfg(feature = "cuda")]
use super::super::super::cache::compile_lock_launch;
#[cfg(feature = "cuda")]
use super::super::super::GpuResidentTensor;
#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::Kernel;

/// Default CUDA workgroup size for batched attention kernels.
#[cfg(feature = "cuda")]
const CUDA_WORKGROUP_SIZE: u32 = 256;

/// Convert interleaved tensor to batched layout for all heads
#[cfg(feature = "cuda")]
pub(in super::super) fn interleaved_to_batched_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::InterleavedToBatchedKernel;

    let total_size = (seq_len * n_heads * head_dim) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!(
        "interleaved_to_batched:{}:{}:{}",
        seq_len, n_heads, head_dim
    );
    let stream = CudaStream::new(ctx)?;

    let threads = CUDA_WORKGROUP_SIZE;
    let blocks = (total_size as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    // The `interleaved_to_batched` PTX kernel declares SIX params
    // (input_ptr, output_ptr, seq_len, n_heads, head_dim, total_elems) and uses
    // every one of them. The args slice MUST supply all six — `cuLaunchKernel`
    // reads one pointer per declared param, so under-supplying makes the driver
    // dereference past the end of `args` (host-side SIGSEGV inside libcuda,
    // invisible to compute-sanitizer).
    let total_elems = total_size as u32;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(seq_len) as *mut _,
        std::ptr::addr_of!(n_heads) as *mut _,
        std::ptr::addr_of!(head_dim) as *mut _,
        std::ptr::addr_of!(total_elems) as *mut _,
    ];

    compile_lock_launch(
        ctx,
        &stream,
        &cache_key,
        &ptx,
        kernel.name(),
        &config,
        &mut args,
    )?;
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Transpose all matrices in batch using grid.z
#[cfg(feature = "cuda")]
pub(in super::super) fn batched_transpose_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedTransposeKernel;

    let total_size = (batch * rows * cols) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = BatchedTransposeKernel::new(batch, rows, cols);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("batched_transpose:{}:{}:{}", batch, rows, cols);
    let stream = CudaStream::new(ctx)?;

    let threads = CUDA_WORKGROUP_SIZE;
    let elems_per_batch = rows * cols;
    let blocks_x = (elems_per_batch + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks_x, 1, batch), // z-dimension for batch/heads
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    // The `batched_transpose` PTX kernel declares SIX params
    // (input_ptr, output_ptr, batch, rows, cols, total_per_batch) and uses
    // `total_per_batch` for its in-bounds guard and batch stride. The args slice
    // MUST supply all six — `cuLaunchKernel` reads one pointer per declared param,
    // so omitting `total_per_batch` makes the driver dereference past the end of
    // `args` (host-side SIGSEGV inside libcuda, invisible to compute-sanitizer).
    let total_per_batch = elems_per_batch;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
        std::ptr::addr_of!(total_per_batch) as *mut _,
    ];

    compile_lock_launch(
        ctx,
        &stream,
        &cache_key,
        &ptx,
        kernel.name(),
        &config,
        &mut args,
    )?;
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Convert batched tensor back to interleaved layout
#[cfg(feature = "cuda")]
pub(in super::super) fn batched_to_interleaved_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedToInterleavedKernel;

    let total_size = (seq_len * n_heads * head_dim) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = BatchedToInterleavedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!(
        "batched_to_interleaved:{}:{}:{}",
        seq_len, n_heads, head_dim
    );
    let stream = CudaStream::new(ctx)?;

    let threads = CUDA_WORKGROUP_SIZE;
    let blocks = (total_size as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    // The `batched_to_interleaved` PTX kernel declares SIX params
    // (input_ptr, output_ptr, seq_len, n_heads, head_dim, total_elems) and uses
    // every one of them. The args slice MUST supply all six — `cuLaunchKernel`
    // reads one pointer per declared param, so under-supplying makes the driver
    // dereference past the end of `args` (host-side SIGSEGV inside libcuda,
    // invisible to compute-sanitizer).
    let total_elems = total_size as u32;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(seq_len) as *mut _,
        std::ptr::addr_of!(n_heads) as *mut _,
        std::ptr::addr_of!(head_dim) as *mut _,
        std::ptr::addr_of!(total_elems) as *mut _,
    ];

    compile_lock_launch(
        ctx,
        &stream,
        &cache_key,
        &ptx,
        kernel.name(),
        &config,
        &mut args,
    )?;
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}
