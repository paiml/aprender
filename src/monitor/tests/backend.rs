//! H0-MON-10 through H0-MON-12: GpuBackend identification tests

use crate::monitor::*;

// =========================================================================
// H0-MON-10: GpuBackend identification
// =========================================================================

#[test]
fn h0_mon_10_backend_names() {
    assert_eq!(GpuBackend::Vulkan.name(), "Vulkan");
    assert_eq!(GpuBackend::Metal.name(), "Metal");
    assert_eq!(GpuBackend::Dx12.name(), "DirectX 12");
    assert_eq!(GpuBackend::Dx11.name(), "DirectX 11");
    assert_eq!(GpuBackend::WebGpu.name(), "WebGPU");
    assert_eq!(GpuBackend::Cuda.name(), "CUDA");
    assert_eq!(GpuBackend::OpenGl.name(), "OpenGL");
    assert_eq!(GpuBackend::Cpu.name(), "CPU");
}

#[test]
fn h0_mon_11_backend_is_gpu() {
    assert!(GpuBackend::Vulkan.is_gpu());
    assert!(GpuBackend::Metal.is_gpu());
    assert!(GpuBackend::Dx12.is_gpu());
    assert!(GpuBackend::Dx11.is_gpu());
    assert!(GpuBackend::WebGpu.is_gpu());
    assert!(GpuBackend::Cuda.is_gpu());
    assert!(GpuBackend::OpenGl.is_gpu());
    assert!(!GpuBackend::Cpu.is_gpu());
}

#[test]
fn h0_mon_12_backend_supports_compute() {
    assert!(GpuBackend::Vulkan.supports_compute());
    assert!(GpuBackend::Metal.supports_compute());
    assert!(GpuBackend::Dx12.supports_compute());
    assert!(GpuBackend::WebGpu.supports_compute());
    assert!(GpuBackend::Cuda.supports_compute());
    assert!(!GpuBackend::Dx11.supports_compute()); // DX11 compute shaders limited
    assert!(!GpuBackend::OpenGl.supports_compute());
    assert!(!GpuBackend::Cpu.supports_compute());
}
