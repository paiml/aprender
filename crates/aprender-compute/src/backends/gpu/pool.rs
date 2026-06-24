//! Multi-GPU device pool for data-parallel workloads
//!
//! Provides [`GpuDevicePool`] for managing multiple GPU devices simultaneously.
//! Designed for data-parallel training where each GPU processes a shard of the
//! mini-batch independently.
//!
//! # Example
//!
//! ```rust,no_run
//! use trueno::backends::gpu::GpuDevicePool;
//!
//! // Open all available GPUs
//! let pool = GpuDevicePool::all()?;
//! println!("Found {} GPUs", pool.len());
//!
//! // Or select specific GPUs by index
//! let pool = GpuDevicePool::with_indices(&[0, 1])?;
//! # Ok::<(), String>(())
//! ```

use super::GpuDevice;

/// Pool of GPU devices for multi-GPU workloads
///
/// Thin wrapper over `Vec<GpuDevice>` that handles enumeration
/// and selective device opening.
pub struct GpuDevicePool {
    devices: Vec<GpuDevice>,
    indices: Vec<u32>,
}

impl GpuDevicePool {
    /// Open all available non-CPU GPU adapters
    ///
    /// Filters out adapters with `Noop` backend (CPU fallback).
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn all() -> Result<Self, String> {
        use super::runtime;
        runtime::block_on(Self::all_async())
    }

    /// Open all available non-CPU GPU adapters (async)
    pub async fn all_async() -> Result<Self, String> {
        // PMAT-778: reuse the process-global instance so the broken freedreno
        // ICD (GB10/aarch64) is enumerated exactly once, never concurrently.
        let instance = super::device::shared_instance();
        // PMAT-925: exclude GLES so the broken GLES/EGL adapter (SIGABRT-in-Drop
        // on Linux/AMD-RADV) is never instantiated (see `device::gpu_backends`).
        let adapters = instance.enumerate_adapters(super::device::gpu_backends());

        if adapters.is_empty() {
            return Err("No GPU adapters found".to_string());
        }

        // Filter out CPU/Noop backends
        let gpu_adapters: Vec<(usize, _)> = adapters
            .into_iter()
            .enumerate()
            .filter(|(_, adapter)| adapter.get_info().backend != wgpu::Backend::Noop)
            .collect();

        if gpu_adapters.is_empty() {
            return Err("No non-CPU GPU adapters found".to_string());
        }

        let mut devices = Vec::with_capacity(gpu_adapters.len());
        let mut indices = Vec::with_capacity(gpu_adapters.len());

        for (idx, adapter) in gpu_adapters {
            let mut limits = wgpu::Limits::default();
            limits.max_buffer_size = adapter.limits().max_buffer_size;
            limits.max_storage_buffer_binding_size =
                adapter.limits().max_storage_buffer_binding_size;

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some(&format!("Trueno GPU Device [{}]", idx)),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                    experimental_features: Default::default(),
                    trace: Default::default(),
                })
                .await
                .map_err(|e| format!("Failed to create device at index {}: {}", idx, e))?;

            devices.push(GpuDevice { device, queue });
            indices.push(idx as u32);
        }

        Ok(Self { devices, indices })
    }

    /// Open specific GPU adapters by index
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn with_indices(adapter_indices: &[u32]) -> Result<Self, String> {
        use super::runtime;
        runtime::block_on(Self::with_indices_async(adapter_indices))
    }

    /// Open specific GPU adapters by index (async)
    pub async fn with_indices_async(adapter_indices: &[u32]) -> Result<Self, String> {
        if adapter_indices.is_empty() {
            return Err("No adapter indices specified".to_string());
        }

        let mut devices = Vec::with_capacity(adapter_indices.len());
        let mut indices = Vec::with_capacity(adapter_indices.len());

        for &idx in adapter_indices {
            let device = GpuDevice::new_with_adapter_index_async(idx).await?;
            devices.push(device);
            indices.push(idx);
        }

        Ok(Self { devices, indices })
    }

    /// Number of devices in the pool
    #[must_use]
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Whether the pool is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// Get a device by pool position (0-based within this pool)
    #[must_use]
    pub fn get(&self, pool_index: usize) -> Option<&GpuDevice> {
        self.devices.get(pool_index)
    }

    /// Get the adapter index for a pool position
    #[must_use]
    pub fn adapter_index(&self, pool_index: usize) -> Option<u32> {
        self.indices.get(pool_index).copied()
    }

    /// Iterate over (adapter_index, device) pairs
    pub fn iter(&self) -> impl Iterator<Item = (u32, &GpuDevice)> {
        self.indices.iter().copied().zip(self.devices.iter())
    }

    /// Consume the pool and return the devices
    pub fn into_devices(self) -> Vec<GpuDevice> {
        self.devices
    }
}

#[cfg(all(test, feature = "gpu", not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_pool_len_matches_devices() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping pool test");
            return;
        }

        let pool = GpuDevicePool::all();
        match pool {
            Ok(p) => {
                assert!(!p.is_empty());
                assert!(p.len() > 0);
                assert!(p.get(0).is_some());
                assert!(p.adapter_index(0).is_some());
            }
            Err(e) => {
                eprintln!("Pool creation failed (expected on CPU-only): {}", e);
            }
        }
    }
}
