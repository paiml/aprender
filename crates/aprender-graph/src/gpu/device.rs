//! GPU device initialization and management
//!
//! Handles wgpu device creation, adapter selection, and GPU resource lifecycle.

use thiserror::Error;
use wgpu::util::DeviceExt;

/// Platform-appropriate wgpu backend mask for adapter enumeration.
///
/// PMAT-927 (class follow-up to PMAT-925): `wgpu::Backends::all()` includes
/// [`wgpu::Backends::GL`], which on Linux hosts that expose both Vulkan and
/// GLES/EGL (notably the intel AMD-RADV cross-silicon baseline box) instantiates
/// a GLES adapter whose `EglContext::make_current` **panics inside `Drop`**. A
/// panic in a destructor during cleanup aborts the whole process with SIGABRT
/// ("panic in a destructor during cleanup"); standalone enumeration can also
/// spin/hang on the broken EGL path. The graph compute kernels themselves are
/// correct — the fragility is purely in adapter *enumeration* and the GLES
/// `Drop` path.
///
/// We return [`wgpu::Backends::PRIMARY`], which in wgpu 22 (this crate's pin) is
/// `VULKAN | METAL | DX12 | BROWSER_WEBGPU` and **excludes** `GL` (GL lives only
/// in `Backends::SECONDARY`). This keeps the real GPU on every platform — Vulkan
/// on Linux/AMD-RADV, Metal on Apple, DX12 on Windows — while guaranteeing the
/// broken GLES/EGL adapter is never created.
#[must_use]
pub const fn gpu_backends() -> wgpu::Backends {
    // PRIMARY = VULKAN | METAL | DX12 | BROWSER_WEBGPU (never GL/GLES).
    wgpu::Backends::PRIMARY
}

/// GPU device initialization errors
#[derive(Debug, Error)]
pub enum GpuDeviceError {
    /// No compatible GPU adapter found
    #[error("No compatible GPU adapter found")]
    NoAdapter,

    /// Failed to request GPU device
    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(String),

    /// GPU feature not supported
    #[error("GPU feature not supported: {0}")]
    UnsupportedFeature(String),
}

/// GPU device wrapper for graph operations
///
/// # Example
///
/// ```ignore
/// # use trueno_graph::gpu::GpuDevice;
/// let device = GpuDevice::new().await?;
/// assert!(device.is_available());
/// ```
#[derive(Debug)]
pub struct GpuDevice {
    #[allow(dead_code)]
    device: wgpu::Device,
    #[allow(dead_code)]
    queue: wgpu::Queue,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
}

impl GpuDevice {
    /// Check if GPU is available without creating a device
    ///
    /// This is useful for tests to skip gracefully when GPU is not available.
    pub async fn is_gpu_available() -> bool {
        Self::new().await.is_ok()
    }

    /// Initialize GPU device with default settings
    ///
    /// # Errors
    ///
    /// Returns `GpuDeviceError` if:
    /// - No compatible GPU adapter found
    /// - Device request fails
    /// - Required features not supported
    pub async fn new() -> Result<Self, GpuDeviceError> {
        // PMAT-927: use the non-GLES mask (see `gpu_backends`) instead of
        // Backends::all() so the broken GLES/EGL adapter (SIGABRT-in-Drop on
        // Linux/AMD-RADV) is never registered.
        Self::new_with_backend(gpu_backends()).await
    }

    /// Initialize GPU device with specific backend
    ///
    /// # Errors
    ///
    /// Returns `GpuDeviceError` if device initialization fails
    pub async fn new_with_backend(backends: wgpu::Backends) -> Result<Self, GpuDeviceError> {
        // Create wgpu instance
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..Default::default() });

        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuDeviceError::NoAdapter)?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("trueno-graph GPU device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuDeviceError::DeviceRequest(e.to_string()))?;

        Ok(Self { device, queue, adapter })
    }

    /// Check if GPU is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        true // If we constructed successfully, GPU is available
    }

    /// Get adapter info (GPU name, backend, etc.)
    #[must_use]
    pub fn info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Create GPU buffer with initial data
    ///
    /// # Errors
    ///
    /// Returns error if buffer creation fails (typically won't happen with wgpu)
    pub fn create_buffer_init(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, GpuDeviceError> {
        Ok(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        }))
    }

    /// Create empty GPU buffer
    ///
    /// # Errors
    ///
    /// Returns error if buffer creation fails
    pub fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, GpuDeviceError> {
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }

    /// Get device reference
    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference
    #[must_use]
    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PMAT-927 FALSIFIER: the wgpu adapter-enumeration backend mask used by
    /// aprender-graph MUST NOT contain GLES (`wgpu::Backends::GL`), and MUST
    /// contain the platform's real backend.
    ///
    /// RED on `Backends::all()` (contains GL → GLES/EGL adapter → SIGABRT-in-Drop
    /// on Linux/AMD-RADV). GREEN on `Backends::PRIMARY`. Host-independent: it
    /// inspects the bitmask, it does not create any adapter.
    #[test]
    fn test_gpu_backends_excludes_gles() {
        let mask = gpu_backends();

        // The whole point: GLES/EGL must never be enumerated.
        assert!(
            !mask.contains(wgpu::Backends::GL),
            "gpu_backends() must NOT include Backends::GL (GLES/EGL panics in Drop \
             on Linux/AMD-RADV → SIGABRT). mask = {mask:?}"
        );

        // The real GPU backend on each platform must still be present.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        assert!(
            mask.contains(wgpu::Backends::VULKAN),
            "gpu_backends() must include VULKAN on Linux (AMD-RADV/NVIDIA). mask = {mask:?}"
        );
        #[cfg(target_os = "macos")]
        assert!(
            mask.contains(wgpu::Backends::METAL),
            "gpu_backends() must include METAL on macOS (Apple Silicon). mask = {mask:?}"
        );
        #[cfg(target_os = "windows")]
        assert!(
            mask.contains(wgpu::Backends::VULKAN) || mask.contains(wgpu::Backends::DX12),
            "gpu_backends() must include VULKAN or DX12 on Windows. mask = {mask:?}"
        );
    }

    #[tokio::test]
    async fn test_gpu_device_creation() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_gpu_device_creation: GPU not available");
            return;
        }

        let device = GpuDevice::new().await;
        assert!(device.is_ok(), "Failed to create GPU device");

        let device = device.unwrap();
        assert!(device.is_available());
    }

    #[tokio::test]
    async fn test_gpu_adapter_info() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_gpu_adapter_info: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let info = device.info();

        // Basic sanity checks
        assert!(!info.name.is_empty(), "Adapter name should not be empty");
        println!("GPU: {info:?}");
    }

    #[tokio::test]
    async fn test_gpu_device_with_invalid_backend() {
        // Try to create device with no backends (should fail)
        let device = GpuDevice::new_with_backend(wgpu::Backends::empty()).await;
        assert!(device.is_err(), "Device creation should fail with empty backends");
    }

    #[test]
    fn test_gpu_device_error_display() {
        let err = GpuDeviceError::NoAdapter;
        assert_eq!(err.to_string(), "No compatible GPU adapter found");

        let err = GpuDeviceError::DeviceRequest("test error".to_string());
        assert_eq!(err.to_string(), "Failed to request GPU device: test error");
    }

    #[tokio::test]
    async fn test_gpu_device_queue() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_gpu_device_queue: GPU not available");
            return;
        }

        let gpu_device = GpuDevice::new().await.unwrap();
        let device = gpu_device.device();
        let queue = gpu_device.queue();

        // Test buffer operations using device and queue
        let test_data: Vec<u32> = vec![1, 2, 3, 4, 5];
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_buffer"),
            size: (test_data.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&test_data));
        queue.submit(std::iter::empty());

        // Verify device and queue are valid
        assert!(gpu_device.is_available());
    }

    #[tokio::test]
    async fn test_create_buffer_init() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_create_buffer_init: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let data: Vec<u32> = vec![1, 2, 3, 4];

        let buffer = device
            .create_buffer_init(
                "test_init",
                bytemuck::cast_slice(&data),
                wgpu::BufferUsages::STORAGE,
            )
            .unwrap();

        // Verify buffer was created
        assert_eq!(buffer.size(), (data.len() * 4) as u64);
    }

    #[tokio::test]
    async fn test_create_buffer() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_create_buffer: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create empty buffer
        let buffer = device
            .create_buffer(
                "test_buffer",
                1024,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
            .unwrap();

        assert_eq!(buffer.size(), 1024);
    }

    #[tokio::test]
    async fn test_different_buffer_usages() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_different_buffer_usages: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Storage buffer
        let storage = device.create_buffer("storage", 512, wgpu::BufferUsages::STORAGE).unwrap();
        assert_eq!(storage.size(), 512);

        // Uniform buffer
        let uniform = device.create_buffer("uniform", 256, wgpu::BufferUsages::UNIFORM).unwrap();
        assert_eq!(uniform.size(), 256);

        // Vertex buffer
        let vertex = device.create_buffer("vertex", 128, wgpu::BufferUsages::VERTEX).unwrap();
        assert_eq!(vertex.size(), 128);
    }
}
