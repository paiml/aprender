//! Element-wise operations for GPU-resident tensors.
//!
//! Includes softmax, add, scale, and layout transform operations.
//! Each operation has both synchronous and stream-based async variants.

#![allow(clippy::similar_names)]

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::{Kernel, LongRowSoftmaxKernel, ScaleKernel, SoftmaxKernel};

#[cfg(feature = "cuda")]
use super::super::cache::compile_lock_launch;
#[cfg(feature = "cuda")]
use super::super::GpuResidentTensor;

/// Compile (or fetch from cache), lock, launch, and return.
///
/// Thin wrapper around `cache::compile_lock_launch` for local call-site brevity.
#[cfg(feature = "cuda")]
fn launch_cached_kernel(
    ctx: &CudaContext,
    stream: &CudaStream,
    cache_key: &str,
    ptx: &str,
    kernel_name: &str,
    config: &LaunchConfig,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    compile_lock_launch(ctx, stream, cache_key, ptx, kernel_name, config, args)
}

#[cfg(feature = "cuda")]
impl GpuResidentTensor<f32> {
    /// Row-wise softmax (stays on GPU)
    ///
    /// Computes softmax along the last dimension.
    /// Result is a new GPU-resident tensor.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `seq_len` - Sequence length (number of rows)
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn softmax(&self, ctx: &CudaContext, seq_len: u32) -> Result<GpuResidentTensor<f32>> {
        let total_elements = self.len();
        let row_size = total_elements / (seq_len as usize);

        if total_elements % (seq_len as usize) != 0 {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Tensor size {} not divisible by seq_len {}",
                total_elements, seq_len
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, total_elements)?;

        // Choose kernel based on row size:
        // - row_size <= 32: warp shuffle softmax (1 warp per row)
        // - row_size > 32: long row softmax (multi-warp with grid-stride loops)
        let stream = CudaStream::new(ctx)?;
        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let row_size_val = row_size as u32;

        if row_size <= 32 {
            // Use warp shuffle softmax for short rows (cached)
            let kernel = SoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax:{}", row_size);

            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (32, 1, 1), // One warp per row
                shared_mem: 0,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            launch_cached_kernel(
                ctx,
                &stream,
                &cache_key,
                &ptx,
                kernel.name(),
                &config,
                &mut args,
            )?;
        } else {
            // Use long row softmax for rows > 32 elements (cached)
            let kernel = LongRowSoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax_long_row:{}", row_size);

            // 256 threads per block (8 warps), one block per row
            // Shared memory: 8 warp maxes + 8 warp sums + 2 global = 72 bytes
            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (256, 1, 1),
                shared_mem: 72,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            launch_cached_kernel(
                ctx,
                &stream,
                &cache_key,
                &ptx,
                kernel.name(),
                &config,
                &mut args,
            )?;
        }

        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Row-wise softmax with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `softmax` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `seq_len` - Sequence length (number of rows)
    /// * `stream` - Caller-provided CUDA stream
    pub fn softmax_with_stream(
        &self,
        ctx: &CudaContext,
        seq_len: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let total_elements = self.len();
        let row_size = total_elements / (seq_len as usize);

        if total_elements % (seq_len as usize) != 0 {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Tensor size {} not divisible by seq_len {}",
                total_elements, seq_len
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, total_elements)?;
        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let row_size_val = row_size as u32;

        if row_size <= 32 {
            // Use warp shuffle softmax for short rows (cached)
            let kernel = SoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax:{}", row_size);

            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (32, 1, 1), // One warp per row
                shared_mem: 0,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            launch_cached_kernel(
                ctx,
                stream,
                &cache_key,
                &ptx,
                kernel.name(),
                &config,
                &mut args,
            )?;
        } else {
            // Use long row softmax for rows > 32 elements (cached)
            let kernel = LongRowSoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax_long_row:{}", row_size);

            // 256 threads per block (8 warps), one block per row
            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (256, 1, 1),
                shared_mem: 72,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            launch_cached_kernel(
                ctx,
                stream,
                &cache_key,
                &ptx,
                kernel.name(),
                &config,
                &mut args,
            )?;
        }
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Element-wise add (stays on GPU)
    ///
    /// Computes C = A + B element-wise.
    /// Result is a new GPU-resident tensor.
    ///
    /// # Errors
    ///
    /// Returns error if sizes don't match or kernel fails.
    pub fn add(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
    ) -> Result<GpuResidentTensor<f32>> {
        if self.len() != other.len() {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }

        let n = self.len();

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, n)?;

        // Use simple add kernel via ResidualAddKernel (cached)
        use crate::kernels::ResidualAddKernel;
        let kernel = ResidualAddKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("residual_add:{}", n);
        let stream = CudaStream::new(ctx)?;

        // Configure launch
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        launch_cached_kernel(
            ctx,
            &stream,
            &cache_key,
            &ptx,
            kernel.name(),
            &config,
            &mut args,
        )?;
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Element-wise add with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `add` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `other` - Tensor to add
    /// * `stream` - Caller-provided CUDA stream
    pub fn add_with_stream(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        if self.len() != other.len() {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }

        let n = self.len();

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, n)?;

        // Use simple add kernel via ResidualAddKernel (cached)
        use crate::kernels::ResidualAddKernel;
        let kernel = ResidualAddKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("residual_add:{}", n);

        // Configure launch
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        launch_cached_kernel(
            ctx,
            stream,
            &cache_key,
            &ptx,
            kernel.name(),
            &config,
            &mut args,
        )?;
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Transform interleaved layout to head-first layout (for attention KV caches)
    ///
    /// Converts: [seq_len, n_heads * head_dim] -> [n_heads, seq_len, head_dim]
    ///
    /// This is the inverse of batched-to-interleaved and is used for preparing
    /// cross-attention K/V caches from encoder output projections.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `seq_len` - Sequence length (first dimension)
    /// * `n_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (n_heads * head_dim = d_model)
    /// * `stream` - Caller-provided CUDA stream
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match tensor size.
    pub fn interleaved_to_head_first(
        &self,
        ctx: &CudaContext,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let d_model = n_heads * head_dim;
        let total_elems = (seq_len * d_model) as usize;

        if self.len() != total_elems {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Tensor size {} doesn't match seq_len ({}) x d_model ({})",
                self.len(),
                seq_len,
                d_model
            )));
        }

        let output_buffer = GpuBuffer::new(ctx, total_elems)?;

        use crate::kernels::InterleavedToBatchedKernel;
        let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
        let ptx = kernel.emit_ptx();
        let cache_key = format!(
            "interleaved_to_batched:{}:{}:{}",
            seq_len, n_heads, head_dim
        );

        let threads = 256u32;
        let blocks = (total_elems as u32 + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();

        // The `interleaved_to_batched` PTX kernel declares SIX params
        // (input_ptr, output_ptr, seq_len, n_heads, head_dim, total_elems) and uses
        // every one of them. The args slice MUST supply all six — `cuLaunchKernel`
        // reads one pointer per declared param, so under-supplying makes the driver
        // dereference past the end of `args` (host-side SIGSEGV inside libcuda,
        // invisible to compute-sanitizer).
        let total_elems_u32 = total_elems as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(seq_len) as *mut _,
            std::ptr::addr_of!(n_heads) as *mut _,
            std::ptr::addr_of!(head_dim) as *mut _,
            std::ptr::addr_of!(total_elems_u32) as *mut _,
        ];

        launch_cached_kernel(
            ctx,
            stream,
            &cache_key,
            &ptx,
            kernel.name(),
            &config,
            &mut args,
        )?;
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Scale tensor by constant (stays on GPU)
    ///
    /// Computes B = A * scale element-wise.
    pub fn scale(&self, ctx: &CudaContext, scale: f32) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, n)?;

        // Use ScaleKernel (multiplies by scalar constant, cached)
        let kernel = ScaleKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("scale:{}", n);
        let stream = CudaStream::new(ctx)?;

        // Configure launch
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        // Prepare arguments (must match kernel params: input_ptr, output_ptr, scale, n)
        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(scale) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        launch_cached_kernel(
            ctx,
            &stream,
            &cache_key,
            &ptx,
            kernel.name(),
            &config,
            &mut args,
        )?;
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }
}
