//! GEMM (General Matrix Multiply) operations for GPU-resident tensors.
//!
//! Provides `matmul` and `matmul_with_stream` with automatic kernel selection:
//! - WMMA Tensor Cores for large matrices (m,n,k >= 64)
//! - Tiled unrolled for medium matrices (k >= 64)
//! - Naive for small matrices

#![allow(clippy::similar_names)]

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::{GemmKernel, Kernel};

#[cfg(feature = "cuda")]
use super::super::cache::compile_lock_launch;
#[cfg(feature = "cuda")]
use super::super::GpuResidentTensor;

#[cfg(feature = "cuda")]
impl GpuResidentTensor<f32> {
    /// Matrix multiply: C = A @ B (stays on GPU)
    ///
    /// Both tensors must be f32. Result is a new GPU-resident tensor.
    /// Does NOT transfer data to host.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `other` - Right-hand matrix
    /// * `m` - Rows of A
    /// * `n` - Columns of B
    /// * `k` - Columns of A / Rows of B
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn matmul(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        // Validate dimensions
        let expected_a = (m * k) as usize;
        let expected_b = (k * n) as usize;
        let output_size = (m * n) as usize;

        if self.len() != expected_a {
            return Err(crate::GpuError::InvalidParameter(format!(
                "A has {} elements, expected {} ({}x{})",
                self.len(),
                expected_a,
                m,
                k
            )));
        }
        if other.len() != expected_b {
            return Err(crate::GpuError::InvalidParameter(format!(
                "B has {} elements, expected {} ({}x{})",
                other.len(),
                expected_b,
                k,
                n
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build and compile GEMM kernel (cached)
        // WAPR-PERF-010: Use WMMA Tensor Cores for large matrices
        // Fixed: D → C accumulator copy for multi-tile K dimension
        let tile_size = 16u32;
        // WAPR-PERF-014: Allow disabling WMMA for precision debugging
        let force_fp32 = std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok();
        let use_wmma = !force_fp32 && k >= 64 && m >= 64 && n >= 64;
        let use_tiled = !use_wmma && k >= 64;

        let (kernel, cache_key, config) = if use_wmma {
            let kernel = GemmKernel::wmma_fp16(m, n, k);
            let key = format!("gemm_wmma_fp16:{}x{}x{}", m, n, k);
            // WMMA: one warp (32 threads) per 16x16 output tile
            let grid_x = (n + 15) / 16;
            let grid_y = (m + 15) / 16;
            // Shared memory: 2 FP16 tiles = 16*16*2*2 = 1024 bytes
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (32, 1, 1), // One warp
                shared_mem: 1024,
            };
            (kernel, key, cfg)
        } else if use_tiled {
            let kernel = GemmKernel::tiled_unrolled(m, n, k, tile_size);
            let key = format!("gemm_tiled_unrolled:{}x{}x{}", m, n, k);
            let grid_x = (n + tile_size - 1) / tile_size;
            let grid_y = (m + tile_size - 1) / tile_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (tile_size, tile_size, 1),
                shared_mem: tile_size * tile_size * 4 * 2,
            };
            (kernel, key, cfg)
        } else {
            let kernel = GemmKernel::naive(m, n, k);
            let key = format!("gemm_naive:{}x{}x{}", m, n, k);
            let block_size = 16u32;
            let grid_x = (n + block_size - 1) / block_size;
            let grid_y = (m + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (block_size, block_size, 1),
                shared_mem: 0,
            };
            (kernel, key, cfg)
        };

        let ptx = kernel.emit_ptx();
        let stream = CudaStream::new(ctx)?;

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let m_val = m;
        let n_val = n;
        let k_val = k;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(m_val) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
            std::ptr::addr_of!(k_val) as *mut _,
        ];

        compile_lock_launch(ctx, &stream, &cache_key, &ptx, kernel.name(), &config, &mut args)?;
        stream.synchronize()?;

        // Return result as GPU-resident tensor (no host transfer!)
        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// WAPR-PERF-014: Matrix multiply with external stream (no stream creation, no sync)
    ///
    /// Same as `matmul` but uses caller-provided stream and does NOT synchronize.
    /// Use this in tight loops to avoid 16+ stream creates/syncs per token.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `other` - Right-hand matrix
    /// * `m` - Rows of A
    /// * `n` - Columns of B
    /// * `k` - Columns of A / Rows of B
    /// * `stream` - Caller-provided CUDA stream (reuse across operations)
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn matmul_with_stream(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
        m: u32,
        n: u32,
        k: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        // Validate dimensions
        let expected_a = (m * k) as usize;
        let expected_b = (k * n) as usize;
        let output_size = (m * n) as usize;

        if self.len() != expected_a {
            return Err(crate::GpuError::InvalidParameter(format!(
                "A has {} elements, expected {} ({}x{})",
                self.len(),
                expected_a,
                m,
                k
            )));
        }
        if other.len() != expected_b {
            return Err(crate::GpuError::InvalidParameter(format!(
                "B has {} elements, expected {} ({}x{})",
                other.len(),
                expected_b,
                k,
                n
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build and compile GEMM kernel (cached)
        let tile_size = 16u32;
        // WAPR-PERF-014: Allow disabling WMMA for precision debugging
        let force_fp32 = std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok();
        let use_wmma = !force_fp32 && k >= 64 && m >= 64 && n >= 64;
        let use_tiled = !use_wmma && k >= 64;

        let (kernel, cache_key, config) = if use_wmma {
            let kernel = GemmKernel::wmma_fp16(m, n, k);
            let key = format!("gemm_wmma_fp16:{}x{}x{}", m, n, k);
            let grid_x = (n + 15) / 16;
            let grid_y = (m + 15) / 16;
            let cfg =
                LaunchConfig { grid: (grid_x, grid_y, 1), block: (32, 1, 1), shared_mem: 1024 };
            (kernel, key, cfg)
        } else if use_tiled {
            let kernel = GemmKernel::tiled_unrolled(m, n, k, tile_size);
            let key = format!("gemm_tiled_unrolled:{}x{}x{}", m, n, k);
            let grid_x = (n + tile_size - 1) / tile_size;
            let grid_y = (m + tile_size - 1) / tile_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (tile_size, tile_size, 1),
                shared_mem: tile_size * tile_size * 4 * 2,
            };
            (kernel, key, cfg)
        } else {
            let kernel = GemmKernel::naive(m, n, k);
            let key = format!("gemm_naive:{}x{}x{}", m, n, k);
            let block_size = 16u32;
            let grid_x = (n + block_size - 1) / block_size;
            let grid_y = (m + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (block_size, block_size, 1),
                shared_mem: 0,
            };
            (kernel, key, cfg)
        };

        let ptx = kernel.emit_ptx();

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let m_val = m;
        let n_val = n;
        let k_val = k;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(m_val) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
            std::ptr::addr_of!(k_val) as *mut _,
        ];

        // Launch kernel using caller's stream
        compile_lock_launch(ctx, stream, &cache_key, &ptx, kernel.name(), &config, &mut args)?;

        // NO SYNC - caller controls synchronization for pipelining
        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }
}
