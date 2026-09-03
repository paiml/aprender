//! Multi-Backend Abstraction
//!
//! Provides a unified interface for different GPU backends:
//! - CUDA (NVIDIA) - Primary, uses PTX
//! - WGPU (WebGPU) - Cross-platform, uses WGSL (Vulkan/Metal/DX12/WebGPU)
//! - Metal (Apple) - shader source only; no dispatcher (see `metal_shaders`)
//! - Vulkan (cross-platform, future)

pub mod metal_shaders;

/// Backend trait for GPU operations
pub trait Backend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Get device count
    fn device_count(&self) -> usize;
}

/// CUDA backend (NVIDIA GPUs)
#[derive(Debug, Default)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            crate::driver::cuda_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn device_count(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            if self.is_available() {
                crate::driver::device_count().unwrap_or(0)
            } else {
                0
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
}

/// Metal backend (Apple GPUs) - placeholder
///
/// Reports unavailable on every platform. This crate contains Metal shader
/// source (`metal_shaders`) but no dispatcher: nothing here calls
/// `MTLDevice::newLibraryWithSource` or `MTLComputeCommandEncoder`. On macOS,
/// use the `wgpu` feature, which reaches Apple GPUs through wgpu's Metal
/// backend and does execute.
#[derive(Debug, Default)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "Metal"
    }

    fn is_available(&self) -> bool {
        false // No dispatcher; see struct docs.
    }

    fn device_count(&self) -> usize {
        0
    }
}

/// Vulkan backend (cross-platform) - placeholder
#[derive(Debug, Default)]
pub struct VulkanBackend;

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan"
    }

    fn is_available(&self) -> bool {
        false // Not implemented yet
    }

    fn device_count(&self) -> usize {
        0
    }
}

/// WGPU backend (WebGPU - cross-platform via wgpu crate)
///
/// Uses WGSL shading language, runs on:
/// - Vulkan (Linux, Windows, Android)
/// - Metal (macOS, iOS)
/// - DX12 (Windows)
/// - WebGPU (browsers via wasm)
#[derive(Debug, Default)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        "WGPU"
    }

    fn is_available(&self) -> bool {
        // Availability based on wgpu feature flag
        cfg!(feature = "wgpu")
    }

    fn device_count(&self) -> usize {
        // Returns 1 if wgpu is available, 0 otherwise (adapter enumeration not yet wired)
        usize::from(self.is_available())
    }
}

/// Detect best available backend
///
/// Priority order:
/// 1. CUDA (NVIDIA) - highest performance for NVIDIA GPUs
/// 2. WGPU - cross-platform fallback (Vulkan/Metal/DX12)
/// 3. Metal - Apple-specific (subset of WGPU)
/// 4. Vulkan - direct Vulkan (subset of WGPU)
#[must_use]
pub fn detect_backend() -> Box<dyn Backend> {
    let cuda = CudaBackend;
    if cuda.is_available() {
        return Box::new(cuda);
    }

    let wgpu = WgpuBackend;
    if wgpu.is_available() {
        return Box::new(wgpu);
    }

    let metal = MetalBackend;
    if metal.is_available() {
        return Box::new(metal);
    }

    let vulkan = VulkanBackend;
    if vulkan.is_available() {
        return Box::new(vulkan);
    }

    // Return CUDA as default (even if unavailable) for PTX generation
    Box::new(CudaBackend)
}

#[cfg(test)]
mod tests;
