//! GPU device initialization and management
//!
//! Handles wgpu device creation, adapter selection, and GPU resource lifecycle.

use thiserror::Error;
use wgpu::util::DeviceExt;

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
        Self::new_with_backend(wgpu::Backends::all()).await
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
