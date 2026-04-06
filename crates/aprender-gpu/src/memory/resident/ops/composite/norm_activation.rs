//! Layer normalization and GELU activation for GPU-resident tensors.
//!
//! Each operation has a synchronous variant (creates its own stream, synchronizes)
//! and a `_with_stream` variant for pipelined execution / CUDA Graph capture.

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::Kernel;

#[cfg(feature = "cuda")]
use super::super::super::cache::compile_lock_launch;
#[cfg(feature = "cuda")]
use super::super::super::GpuResidentTensor;

#[cfg(feature = "cuda")]
impl GpuResidentTensor<f32> {
    /// Layer normalization (stays on GPU)
    ///
    /// Computes: output = (x - mean) / sqrt(var + eps) * gamma + beta
    ///
    /// # Arguments
    /// * `ctx` - CUDA context
    /// * `gamma` - Scale parameters [hidden_size]
    /// * `beta` - Shift parameters [hidden_size]
    /// * `hidden_size` - Dimension being normalized
    /// * `batch_size` - Number of rows to normalize
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn layer_norm(
        &self,
        ctx: &CudaContext,
        gamma: &GpuResidentTensor<f32>,
        beta: &GpuResidentTensor<f32>,
        hidden_size: u32,
        batch_size: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::LayerNormKernel;
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("layer_norm:{}", hidden_size);
        let stream = CudaStream::new(ctx)?;

        // Launch one warp per row - always use 32 threads for warp shuffle reduction
        // The kernel handles bounds checking internally for hidden_size < 32
        let threads = 32u32;
        let blocks = batch_size;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let gamma_ptr = gamma.as_ptr();
        let beta_ptr = beta.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(gamma_ptr) as *mut _,
            std::ptr::addr_of!(beta_ptr) as *mut _,
            std::ptr::addr_of!(hidden_size) as *mut _,
            std::ptr::addr_of!(batch_size) as *mut _,
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

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Layer normalization with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `layer_norm` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `gamma` - Scale parameters (hidden_size)
    /// * `beta` - Bias parameters (hidden_size)
    /// * `hidden_size` - Size of hidden dimension
    /// * `batch_size` - Number of rows (batch or seq_len)
    /// * `stream` - Caller-provided CUDA stream
    pub fn layer_norm_with_stream(
        &self,
        ctx: &CudaContext,
        gamma: &GpuResidentTensor<f32>,
        beta: &GpuResidentTensor<f32>,
        hidden_size: u32,
        batch_size: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::LayerNormKernel;
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("layer_norm:{}", hidden_size);

        // Launch one warp per row - always use 32 threads for warp shuffle reduction
        // The kernel handles bounds checking internally for hidden_size < 32
        let threads = 32u32;
        let blocks = batch_size;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let gamma_ptr = gamma.as_ptr();
        let beta_ptr = beta.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(gamma_ptr) as *mut _,
            std::ptr::addr_of!(beta_ptr) as *mut _,
            std::ptr::addr_of!(hidden_size) as *mut _,
            std::ptr::addr_of!(batch_size) as *mut _,
        ];

        compile_lock_launch(
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

    /// GELU activation (stays on GPU)
    ///
    /// Computes: output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn gelu(&self, ctx: &CudaContext) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::GeluKernel;
        let kernel = GeluKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("gelu:{}", n);
        let stream = CudaStream::new(ctx)?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
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

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// GELU activation with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `gelu` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `stream` - Caller-provided CUDA stream
    pub fn gelu_with_stream(
        &self,
        ctx: &CudaContext,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::GeluKernel;
        let kernel = GeluKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("gelu:{}", n);

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        compile_lock_launch(
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
}
