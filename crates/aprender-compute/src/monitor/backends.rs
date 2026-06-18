//! wgpu and CUDA backend implementations for GPU monitoring

#[cfg(feature = "cuda-monitor")]
use super::GpuMemoryMetrics;
#[cfg(any(all(feature = "gpu", not(target_arch = "wasm32")), feature = "cuda-monitor"))]
use super::MonitorError;
#[cfg(any(all(feature = "gpu", not(target_arch = "wasm32")), feature = "cuda-monitor"))]
use super::{GpuBackend, GpuDeviceInfo, GpuVendor};

// ============================================================================
// wgpu Backend Implementation
// ============================================================================

/// Query device info from wgpu adapter
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub(crate) fn query_wgpu_device_info(device_index: u32) -> Result<GpuDeviceInfo, MonitorError> {
    use crate::backends::gpu::runtime;

    runtime::block_on(async {
        // PMAT-778: reuse the process-global instance (single freedreno enumeration).
        let instance = crate::backends::gpu::shared_instance();

        // Get all adapters (wgpu 27+ returns Vec directly)
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err(MonitorError::NoDevice);
        }

        let adapter =
            adapters.get(device_index as usize).ok_or(MonitorError::InvalidDevice(device_index))?;

        let info = adapter.get_info();

        // Map wgpu backend to our backend enum
        let backend = match info.backend {
            wgpu::Backend::Vulkan => GpuBackend::Vulkan,
            wgpu::Backend::Metal => GpuBackend::Metal,
            wgpu::Backend::Dx12 => GpuBackend::Dx12,
            wgpu::Backend::Gl => GpuBackend::OpenGl,
            wgpu::Backend::BrowserWebGpu => GpuBackend::WebGpu,
            wgpu::Backend::Noop => GpuBackend::Cpu,
        };

        // Map vendor ID
        let vendor = GpuVendor::from_vendor_id(info.vendor);

        // Get memory limits (rough estimate from adapter limits)
        let limits = adapter.limits();
        // Use max buffer size as a proxy for VRAM (not exact but gives an idea)
        let vram_estimate = limits.max_buffer_size;

        Ok(GpuDeviceInfo::new(device_index, info.name, vendor, backend)
            .with_vram(vram_estimate)
            .with_driver_version(format!("{:?}", info.driver_info)))
    })
}

/// Enumerate all wgpu devices
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub(crate) fn enumerate_wgpu_devices() -> Result<Vec<GpuDeviceInfo>, MonitorError> {
    use crate::backends::gpu::runtime;

    runtime::block_on(async {
        // PMAT-778: reuse the process-global instance (single freedreno enumeration).
        let instance = crate::backends::gpu::shared_instance();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err(MonitorError::NoDevice);
        }

        let mut devices = Vec::with_capacity(adapters.len());

        for (idx, adapter) in adapters.iter().enumerate() {
            let info: wgpu::AdapterInfo = adapter.get_info();

            let backend = match info.backend {
                wgpu::Backend::Vulkan => GpuBackend::Vulkan,
                wgpu::Backend::Metal => GpuBackend::Metal,
                wgpu::Backend::Dx12 => GpuBackend::Dx12,
                wgpu::Backend::Gl => GpuBackend::OpenGl,
                wgpu::Backend::BrowserWebGpu => GpuBackend::WebGpu,
                wgpu::Backend::Noop => GpuBackend::Cpu,
            };

            let vendor = GpuVendor::from_vendor_id(info.vendor);
            let limits = adapter.limits();

            devices.push(
                GpuDeviceInfo::new(idx as u32, info.name, vendor, backend)
                    .with_vram(limits.max_buffer_size)
                    .with_driver_version(format!("{:?}", info.driver_info)),
            );
        }

        Ok(devices)
    })
}

// ============================================================================
// CUDA Backend Implementation (TRUENO-SPEC-010 Section 8)
// ============================================================================

/// Query device info from native CUDA via trueno-gpu
///
/// This provides more accurate information than wgpu including:
/// - Actual device name (e.g., "NVIDIA GeForce RTX 4090")
/// - Accurate VRAM total from cuDeviceTotalMem
#[cfg(feature = "cuda-monitor")]
pub fn query_cuda_device_info(device_index: u32) -> Result<GpuDeviceInfo, MonitorError> {
    use trueno_gpu::CudaDeviceInfo;

    let cuda_info = CudaDeviceInfo::query(device_index)
        .map_err(|e| MonitorError::BackendInit(format!("CUDA query failed: {}", e)))?;

    Ok(GpuDeviceInfo::new(
        cuda_info.index,
        cuda_info.name,
        GpuVendor::Nvidia, // CUDA is NVIDIA-only
        GpuBackend::Cuda,
    )
    .with_vram(cuda_info.total_memory))
}

/// Enumerate all CUDA devices via trueno-gpu
#[cfg(feature = "cuda-monitor")]
pub fn enumerate_cuda_devices() -> Result<Vec<GpuDeviceInfo>, MonitorError> {
    use trueno_gpu::CudaDeviceInfo;

    let cuda_devices = CudaDeviceInfo::enumerate()
        .map_err(|e| MonitorError::BackendInit(format!("CUDA enumerate failed: {}", e)))?;

    Ok(cuda_devices
        .into_iter()
        .map(|cuda_info| {
            GpuDeviceInfo::new(cuda_info.index, cuda_info.name, GpuVendor::Nvidia, GpuBackend::Cuda)
                .with_vram(cuda_info.total_memory)
        })
        .collect())
}

/// Query real-time CUDA memory metrics
///
/// Returns current free/used VRAM from cuMemGetInfo.
#[cfg(feature = "cuda-monitor")]
pub fn query_cuda_memory(device_index: u32) -> Result<GpuMemoryMetrics, MonitorError> {
    use trueno_gpu::driver::CudaContext;
    use trueno_gpu::CudaMemoryInfo;

    let ctx = CudaContext::new(device_index as i32)
        .map_err(|e| MonitorError::BackendInit(format!("CUDA context failed: {}", e)))?;

    let mem = CudaMemoryInfo::query(&ctx)
        .map_err(|e| MonitorError::QueryFailed(format!("CUDA memory query failed: {}", e)))?;

    Ok(GpuMemoryMetrics::new(mem.total, mem.used(), mem.free))
}

/// Check if CUDA monitoring is available
#[cfg(feature = "cuda-monitor")]
#[must_use]
pub fn cuda_monitor_available() -> bool {
    trueno_gpu::cuda_monitoring_available()
}

/// Check if CUDA monitoring is available (stub when feature disabled)
#[cfg(not(feature = "cuda-monitor"))]
#[must_use]
pub fn cuda_monitor_available() -> bool {
    false
}
