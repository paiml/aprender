#![allow(missing_docs)]
//! GPU backend using wgpu (Vulkan/Metal/DX12/WebGPU)
//!
//! This backend provides GPU-accelerated compute for large-scale operations.
//! It uses wgpu for cross-platform GPU access and WGSL compute shaders.
//!
//! # Performance
//!
//! GPU backend is optimal for very large workloads (>100K elements for reductions,
//! >1000×1000 for matrix operations) where transfer overhead is amortized.
//!
//! Expected speedups vs SIMD:
//! - Matrix multiplication (large): 10-50x
//! - Reductions (large): 5-20x
//!
//! # Architecture
//!
//! - Device initialization is lazy (first GPU operation)
//! - Compute shaders written in WGSL
//! - Asynchronous execution with pollster for blocking
//! - Automatic fallback to CPU if GPU unavailable
//!
//! # Memory Hierarchy Abstractions
//!
//! - [`TensorView`] - Structured view into GPU memory with shape/stride metadata
//! - [`PartitionView`] - Tiling strategy for efficient GPU work distribution
//!
//! Based on cuda-tile-behavior.md Section 3.2.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod batch;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod device;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod pool;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub mod shaders;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub mod runtime;

// Memory hierarchy abstractions (always available, no GPU feature required)
mod partition_view;
mod tensor_view;
mod tiled_reduction;

pub use partition_view::{PartitionView, TileInfo};
pub use tensor_view::{MemoryLayout, TensorView};
pub use tiled_reduction::{
    tiled_max_2d, tiled_min_2d, tiled_reduce_2d, tiled_reduce_partial, tiled_sum_2d, MaxOp, MinOp,
    ReduceOp, SumOp, TILE_SIZE,
};

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub use batch::{BufferId, GpuCommandBatch, PipelineCache};

// Export GpuDevice for both native and WASM GPU features
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub use device::GpuDevice;

// PMAT-778: process-global shared wgpu instance, reused by every crate-internal
// adapter/device enumeration so the broken freedreno ICD (GB10) is touched once.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub(crate) use device::shared_instance;

// PMAT-925: non-GLES backend mask, reused by every crate-internal
// `enumerate_adapters` call so the broken GLES/EGL adapter (SIGABRT-in-Drop on
// Linux/AMD-RADV) is never instantiated.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub(crate) use device::gpu_backends;

/// Re-export wgpu types for downstream crates that need to create persistent
/// GPU buffers (KAIZEN-015: GPU-resident weights).
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub use wgpu;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub use pool::GpuDevicePool;

/// PMAT-322: Cached matmul with persistent weight buffers for LLM inference.
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub use device::linalg::cached_matmul::GpuMatmulCache;

/// PMAT-324: WGSL transformer forward pass shaders.
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub use device::linalg::wgsl_forward::{QkvLoRA, WgslForwardPass};

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod backend_ops;

/// GPU backend for compute operations (native only, uses sync wrappers)
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[derive(Clone)]
pub struct GpuBackend {
    device: Option<GpuDevice>,
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
impl GpuBackend {
    /// Create a new GPU backend
    pub fn new() -> Self {
        Self { device: None }
    }

    /// Initialize GPU device (lazy)
    fn ensure_device(&mut self) -> Result<&GpuDevice, String> {
        if self.device.is_none() {
            self.device = Some(GpuDevice::new()?);
        }
        Ok(self.device.as_ref().expect("device initialized above"))
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        GpuDevice::is_available()
    }
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Stub implementation when GPU feature is disabled or on WASM
#[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
#[derive(Clone)]
pub struct GpuBackend;

#[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
impl GpuBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn is_available() -> bool {
        false
    }
}

#[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
impl Default for GpuBackend {
    fn default() -> Self {
        Self
    }
}

// Tests for stub implementation (when GPU feature is NOT enabled)
#[cfg(test)]
#[cfg(not(feature = "gpu"))]
mod stub_tests {
    use super::*;

    #[test]
    fn test_gpu_backend_stub_new() {
        let _backend = GpuBackend::new();
    }

    #[test]
    fn test_gpu_backend_stub_is_available() {
        assert!(!GpuBackend::is_available());
    }

    #[test]
    fn test_gpu_backend_stub_default() {
        let _ = GpuBackend;
    }

    #[test]
    fn test_gpu_backend_stub_clone() {
        let backend = GpuBackend::new();
        let _cloned = backend.clone();
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests_gpu;
