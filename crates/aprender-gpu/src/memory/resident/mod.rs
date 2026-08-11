//! GPU-Resident Tensor (WAPR-PERF-004)
//!
//! Tensors that stay on GPU with transfer tracking to minimize host↔device traffic.
//!
//! ## Problem
//!
//! Standard approach: Each operation transfers data back to host
//! - matmul: GPU → CPU (for softmax)
//! - softmax: CPU → GPU (for next matmul)
//! - Result: ~150 transfers per encoder pass
//!
//! ## Solution
//!
//! GpuResidentTensor keeps data on device, only transfers when explicitly requested.
//! - Operations return new GpuResidentTensors (still on device)
//! - Only `.to_host()` triggers device→host transfer
//! - Transfer counters enable debugging and verification
//!
//! ## Citations
//!
//! - [Dao2022] FlashAttention: Fast and Memory-Efficient Exact Attention
//! - [Kwon2023] PagedAttention for LLM Serving with vLLM

// Sub-modules
mod attention;
mod cache;
mod ops;
mod stats;
mod weights;

#[cfg(test)]
mod ops_tests;

// Re-exports from submodules
#[cfg(feature = "cuda")]
pub use attention::{
    batched_multihead_attention, batched_multihead_attention_optimized, incremental_attention_gpu,
    incremental_attention_gpu_async, incremental_attention_gpu_with_stream, kv_cache_scatter_gpu,
};
pub use cache::{
    clear_kernel_cache, kernel_cache_exclusive, kernel_cache_hits, kernel_cache_misses,
    reset_kernel_cache_stats, KernelCacheExclusive,
};
pub use stats::{
    reset_transfer_counters, total_d2h_bytes, total_d2h_transfers, total_h2d_bytes,
    total_h2d_transfers, TransferStats,
};
#[cfg(feature = "cuda")]
pub use weights::{
    forward_encoder_block_gpu, GpuConvFrontendWeights, GpuDecoderBlockWeights, GpuDecoderConfig,
    GpuEncoderBlockWeights, GpuEncoderConfig, GpuKvCache,
};

// Internal access to submodule functions
#[cfg(feature = "cuda")]
use stats::{record_d2h_transfer, record_h2d_transfer};

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, GpuBuffer};
#[cfg(feature = "cuda")]
use crate::error::Result;

// ============================================================================
// GpuResidentTensor (CUDA-only)
// ============================================================================

#[cfg(feature = "cuda")]
/// A tensor that resides on GPU with transfer tracking
///
/// Unlike regular GpuBuffer, this tracks all transfers for debugging
/// and verification of GPU-resident pipelines.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::memory::resident::GpuResidentTensor;
///
/// let ctx = CudaContext::new(0)?;
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
///
/// // Upload data (1 H2D transfer)
/// let tensor = GpuResidentTensor::from_host(&ctx, &data)?;
/// assert_eq!(tensor.h2d_transfers(), 1);
///
/// // Operations stay on GPU
/// let doubled = tensor.scale(2.0)?; // No transfer!
/// assert_eq!(doubled.d2h_transfers(), 0);
///
/// // Only explicit download triggers transfer
/// let result = doubled.to_host()?;
/// assert_eq!(doubled.d2h_transfers(), 1);
/// ```
pub struct GpuResidentTensor<T: Copy> {
    /// Underlying GPU buffer
    pub(crate) buffer: GpuBuffer<T>,
    /// Number of host-to-device transfers for this tensor
    h2d_count: u64,
    /// Number of device-to-host transfers for this tensor
    d2h_count: u64,
    /// Number of kernel launches involving this tensor
    kernel_launches: u64,
    /// Whether this tensor is currently on device
    is_resident: bool,
}

#[cfg(feature = "cuda")]
impl<T: Copy> GpuResidentTensor<T> {
    /// Create a GPU-resident tensor from host data
    ///
    /// This uploads the data to GPU (1 H2D transfer).
    pub fn from_host(ctx: &CudaContext, data: &[T]) -> Result<Self> {
        let buffer = GpuBuffer::from_host(ctx, data)?;
        let bytes = data.len() * std::mem::size_of::<T>();

        // Track transfer
        record_h2d_transfer(bytes as u64);

        Ok(Self {
            buffer,
            h2d_count: 1,
            d2h_count: 0,
            kernel_launches: 0,
            is_resident: true,
        })
    }

    /// Create an uninitialized tensor on GPU
    ///
    /// The tensor has allocated memory but uninitialized contents.
    /// Use this for output buffers.
    pub fn new_uninit(ctx: &CudaContext, len: usize) -> Result<Self> {
        let buffer = GpuBuffer::new(ctx, len)?;

        Ok(Self {
            buffer,
            h2d_count: 0,
            d2h_count: 0,
            kernel_launches: 0,
            is_resident: true,
        })
    }

    /// Create from existing GPU buffer (internal constructor)
    ///
    /// Used when creating result tensors from GPU operations.
    /// Does NOT count as a transfer since data never left GPU.
    pub(crate) fn from_buffer_internal(buffer: GpuBuffer<T>, kernel_launches: u64) -> Self {
        Self {
            buffer,
            h2d_count: 0,
            d2h_count: 0,
            kernel_launches,
            is_resident: true,
        }
    }

    /// Download tensor to host memory
    ///
    /// This triggers 1 D2H transfer.
    pub fn to_host(&mut self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut result = vec![T::default(); self.buffer.len()];
        self.buffer.copy_to_host(&mut result)?;

        let bytes = result.len() * std::mem::size_of::<T>();

        // Track transfer
        self.d2h_count += 1;
        record_d2h_transfer(bytes as u64);

        Ok(result)
    }

    /// Peek at tensor data on host (debug only, no transfer tracking)
    ///
    /// This copies data to host without updating transfer counters.
    /// Use only for debugging to avoid affecting transfer statistics.
    pub fn peek_host(&self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut result = vec![T::default(); self.buffer.len()];
        self.buffer.copy_to_host(&mut result)?;
        Ok(result)
    }

    /// Check if tensor is currently resident on device
    #[must_use]
    pub const fn is_device_resident(&self) -> bool {
        self.is_resident
    }

    /// Get number of host-to-device transfers for this tensor
    #[must_use]
    pub const fn h2d_transfers(&self) -> u64 {
        self.h2d_count
    }

    /// Alias for h2d_transfers
    #[must_use]
    pub const fn host_to_device_transfers(&self) -> u64 {
        self.h2d_count
    }

    /// Get number of device-to-host transfers for this tensor
    #[must_use]
    pub const fn d2h_transfers(&self) -> u64 {
        self.d2h_count
    }

    /// Alias for d2h_transfers
    #[must_use]
    pub const fn device_to_host_transfers(&self) -> u64 {
        self.d2h_count
    }

    /// Get number of kernel launches involving this tensor
    #[must_use]
    pub const fn kernel_launches(&self) -> u64 {
        self.kernel_launches
    }

    /// Get tensor length (number of elements)
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if tensor is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.buffer.size_bytes()
    }

    /// Get underlying GPU buffer (for kernel operations)
    #[must_use]
    pub fn buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// Get mutable reference to underlying GPU buffer
    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buffer
    }

    /// Get device pointer
    #[must_use]
    pub fn as_ptr(&self) -> u64 {
        self.buffer.as_ptr()
    }

    /// Increment kernel launch counter (called by kernel executors)
    pub fn record_kernel_launch(&mut self) {
        self.kernel_launches += 1;
    }
}
#[cfg(test)]
mod tests;
