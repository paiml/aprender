//! Bias add, linear projection, fused linear+GELU, and conv1d for GPU-resident tensors.
//!
//! Each operation has a synchronous variant (creates its own stream, synchronizes)
//! and where applicable a `_with_stream` variant for pipelined execution.

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
    /// Bias add (stays on GPU)
    ///
    /// Computes: output[i] = input[i] + bias[i % bias_size]
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn bias_add(
        &self,
        ctx: &CudaContext,
        bias: &GpuResidentTensor<f32>,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let bias_size = bias.len();

        // WAPR-PERF-027 FIX: Create stream FIRST to ensure D2D copy and kernel use same stream
        // BiasActivationKernel is IN-PLACE: reads from output, adds bias, writes to output
        let stream = CudaStream::new(ctx)?;

        // Allocate output buffer and copy input data using SAME stream
        // Previously used clone() which ran on default stream - race condition with kernel!
        let mut output_buffer = GpuBuffer::new(ctx, n)?;
        // SAFETY: both buffers valid, stream will be synchronized before returning
        unsafe {
            output_buffer.copy_from_buffer_async(&self.buffer, &stream)?;
        }

        use crate::kernels::BiasActivationKernel;
        let kernel = BiasActivationKernel::new(n as u32, bias_size as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("bias_add:{}:{}", n, bias_size);

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig { grid: (blocks, 1, 1), block: (threads, 1, 1), shared_mem: 0 };

        let output_ptr = output_buffer.as_ptr();
        let bias_ptr = bias.as_ptr();
        let n_val = n as u32;

        // Kernel params: (output, bias, n) - kernel is in-place on output
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        compile_lock_launch(ctx, &stream, &cache_key, &ptx, kernel.name(), &config, &mut args)?;
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// WAPR-PERF-014: Bias add with external stream (no stream creation, no sync)
    pub fn bias_add_with_stream(
        &self,
        ctx: &CudaContext,
        bias: &GpuResidentTensor<f32>,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let bias_size = bias.len();

        let output_buffer = self.buffer.clone(ctx)?;

        use crate::kernels::BiasActivationKernel;
        let kernel = BiasActivationKernel::new(n as u32, bias_size as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("bias_add:{}:{}", n, bias_size);

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig { grid: (blocks, 1, 1), block: (threads, 1, 1), shared_mem: 0 };

        let output_ptr = output_buffer.as_ptr();
        let bias_ptr = bias.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        compile_lock_launch(ctx, stream, &cache_key, &ptx, kernel.name(), &config, &mut args)?;
        // NO SYNC - caller controls synchronization

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Linear projection: output = input @ weight + bias (stays on GPU)
    ///
    /// Weight is [in_features, out_features] row-major.
    /// Input is [batch_size * in_features] flattened.
    /// Output is [batch_size * out_features].
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn linear(
        &self,
        ctx: &CudaContext,
        weight: &GpuResidentTensor<f32>,
        bias: Option<&GpuResidentTensor<f32>>,
        batch_size: u32,
        in_features: u32,
        out_features: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        let debug = std::env::var("WHISPER_DEBUG_LINEAR").is_ok();
        if debug {
            eprintln!(
                "[DEBUG-LINEAR] input: len={}, batch={}, in_feat={}, out_feat={}",
                self.len(),
                batch_size,
                in_features,
                out_features
            );
            let inp = self.peek_host()?;
            eprintln!(
                "[DEBUG-LINEAR] input stats: mean={:.6}, max={:.6}",
                inp.iter().sum::<f32>() / inp.len() as f32,
                inp.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // matmul: [batch_size, in_features] @ [in_features, out_features] = [batch_size, out_features]
        let result = self.matmul(ctx, weight, batch_size, out_features, in_features)?;

        if debug {
            let res = result.peek_host()?;
            eprintln!(
                "[DEBUG-LINEAR] matmul result: len={}, mean={:.6}, max={:.6}",
                res.len(),
                res.iter().sum::<f32>() / res.len() as f32,
                res.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // Add bias if provided
        if let Some(b) = bias {
            let output = result.bias_add(ctx, b)?;
            if debug {
                let out = output.peek_host()?;
                eprintln!(
                    "[DEBUG-LINEAR] after bias_add: len={}, mean={:.6}, max={:.6}",
                    out.len(),
                    out.iter().sum::<f32>() / out.len() as f32,
                    out.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            }
            Ok(output)
        } else {
            Ok(result)
        }
    }

    /// Fused linear + GELU: output = GELU(input @ weight + bias) (WAPR-PERF-007)
    ///
    /// Combines GEMM + Bias + GELU into a single kernel launch, eliminating
    /// 2 kernel launches and associated memory traffic.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `weight` - Weight matrix [in_features, out_features]
    /// * `bias` - Bias vector [out_features]
    /// * `batch_size` - Number of rows in input
    /// * `in_features` - Input dimension (K)
    /// * `out_features` - Output dimension (N)
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn fused_linear_gelu(
        &self,
        ctx: &CudaContext,
        weight: &GpuResidentTensor<f32>,
        bias: &GpuResidentTensor<f32>,
        batch_size: u32,
        in_features: u32,
        out_features: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        use crate::kernels::FusedGemmBiasGeluKernel;

        let output_size = (batch_size * out_features) as usize;
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build and compile fused kernel (cached)
        let kernel = FusedGemmBiasGeluKernel::new(batch_size, out_features, in_features);
        let ptx = kernel.emit_ptx();
        let cache_key =
            format!("fused_gemm_bias_gelu:{}x{}x{}", batch_size, out_features, in_features);
        let stream = CudaStream::new(ctx)?;

        // Configure launch: 16x16 block, grid covers output matrix
        let block_size = 16u32;
        let grid_x = (out_features + block_size - 1) / block_size;
        let grid_y = (batch_size + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (block_size, block_size, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = weight.as_ptr();
        let bias_ptr = bias.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let m_val = batch_size;
        let n_val = out_features;
        let k_val = in_features;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(m_val) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
            std::ptr::addr_of!(k_val) as *mut _,
        ];

        // Launch fused kernel
        compile_lock_launch(ctx, &stream, &cache_key, &ptx, kernel.name(), &config, &mut args)?;
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// WAPR-PERF-012: GPU Conv1d with GELU activation
    ///
    /// Computes 1D convolution for Whisper audio frontend.
    /// Target: Move 588ms CPU conv to GPU (<50ms).
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `weight` - Weight tensor [out_channels, in_channels, kernel_size]
    /// * `bias` - Bias tensor [out_channels] (optional)
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Convolution kernel size
    /// * `stride` - Stride
    /// * `padding` - Padding
    /// * `seq_len` - Input sequence length
    ///
    /// # Returns
    ///
    /// Output tensor [out_seq_len, out_channels] with GELU applied
    pub fn conv1d(
        &self,
        ctx: &CudaContext,
        weight: &GpuResidentTensor<f32>,
        bias: Option<&GpuResidentTensor<f32>>,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
        seq_len: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        use crate::kernels::Conv1dKernel;

        // Calculate output sequence length
        let out_seq_len = (seq_len + 2 * padding - kernel_size) / stride + 1;
        let output_size = (out_seq_len * out_channels) as usize;

        // Validate input dimensions
        let expected_input = (seq_len * in_channels) as usize;
        if self.len() != expected_input {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Input has {} elements, expected {} ({}x{})",
                self.len(),
                expected_input,
                seq_len,
                in_channels
            )));
        }

        let expected_weight = (out_channels * in_channels * kernel_size) as usize;
        if weight.len() != expected_weight {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Weight has {} elements, expected {} ({}x{}x{})",
                weight.len(),
                expected_weight,
                out_channels,
                in_channels,
                kernel_size
            )));
        }

        // Allocate output buffer
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build kernel
        let kernel = Conv1dKernel::new(in_channels, out_channels, kernel_size, stride, padding);
        let cache_key = format!(
            "conv1d:{}:{}:{}:{}:{}",
            in_channels, out_channels, kernel_size, stride, padding
        );
        let ptx = kernel.emit_ptx();
        let stream = CudaStream::new(ctx)?;

        // Launch configuration
        let block_x = 32u32;
        let block_y = 8u32;
        let grid_x = (out_seq_len + block_x - 1) / block_x;
        let grid_y = (out_channels + block_y - 1) / block_y;

        let config =
            LaunchConfig { grid: (grid_x, grid_y, 1), block: (block_x, block_y, 1), shared_mem: 0 };

        // Prepare arguments
        let input_ptr = self.as_ptr();
        let weight_ptr = weight.as_ptr();
        let bias_ptr = bias.map_or(0_u64, |b| b.as_ptr());
        let output_ptr = output_buffer.as_ptr();
        let seq_len_val = seq_len;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(weight_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(seq_len_val) as *mut _,
        ];

        // Launch kernel
        compile_lock_launch(ctx, &stream, &cache_key, &ptx, kernel.name(), &config, &mut args)?;
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }
}
