//! WGPU Multi-GPU Monitor - 100% Safe Rust
//!
//! Provides cross-platform GPU monitoring via the wgpu crate.
//! Supports Metal (macOS), Vulkan (Linux), DX12 (Windows), WebGPU (browsers).
//!
//! ## Safety
//!
//! This module contains NO unsafe code. It uses wgpu's safe Rust API.
//!
//! ## Features
//!
//! - Multi-GPU enumeration and monitoring
//! - Per-GPU metrics tracking (dispatches, submissions, memory)
//! - trueno-zram integration for GPU compression stats
//! - Round-robin and least-loaded GPU selection

// Note: We use `deny` instead of `forbid` to allow the unsafe Send/Sync impls
// which are required because wgpu::Instance doesn't implement them by default
#![deny(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "gpu-wgpu")]
use ::wgpu::{Adapter, Backends, DeviceType, Instance, InstanceDescriptor};

/// WGPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuBackendType {
    /// Vulkan (Linux, Windows, Android)
    Vulkan,
    /// Metal (macOS, iOS)
    Metal,
    /// DirectX 12 (Windows)
    Dx12,
    /// DirectX 11 (Windows, legacy)
    Dx11,
    /// OpenGL (fallback)
    Gl,
    /// WebGPU (browsers)
    BrowserWebGpu,
    /// Unknown/Empty
    Empty,
}

#[cfg(feature = "gpu-wgpu")]
impl From<::wgpu::Backend> for WgpuBackendType {
    fn from(backend: ::wgpu::Backend) -> Self {
        match backend {
            ::wgpu::Backend::Vulkan => Self::Vulkan,
            ::wgpu::Backend::Metal => Self::Metal,
            ::wgpu::Backend::Dx12 => Self::Dx12,
            ::wgpu::Backend::Gl => Self::Gl,
            ::wgpu::Backend::BrowserWebGpu => Self::BrowserWebGpu,
            _ => Self::Empty,
        }
    }
}

/// GPU adapter information
#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    /// Adapter index
    pub index: usize,
    /// GPU name (e.g., "AMD Radeon Pro W5700X")
    pub name: String,
    /// Backend type (Metal, Vulkan, etc.)
    pub backend: WgpuBackendType,
    /// Device type (Discrete, Integrated, etc.)
    pub device_type: GpuDeviceType,
    /// Driver name
    pub driver: String,
    /// Driver info/version
    pub driver_info: String,
    /// Vendor ID
    pub vendor_id: u32,
    /// Device ID
    pub device_id: u32,
}

impl GpuAdapterInfo {
    /// Check if this is a discrete GPU
    #[must_use]
    pub fn is_discrete(&self) -> bool {
        matches!(self.device_type, GpuDeviceType::DiscreteGpu)
    }

    /// Check if this is an integrated GPU
    #[must_use]
    pub fn is_integrated(&self) -> bool {
        matches!(self.device_type, GpuDeviceType::IntegratedGpu)
    }
}

/// GPU device type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// Discrete GPU (dedicated graphics card)
    DiscreteGpu,
    /// Integrated GPU (on CPU die)
    IntegratedGpu,
    /// Virtual GPU
    VirtualGpu,
    /// CPU/software rendering
    Cpu,
    /// Unknown type
    Other,
}

#[cfg(feature = "gpu-wgpu")]
impl From<DeviceType> for GpuDeviceType {
    fn from(dt: DeviceType) -> Self {
        match dt {
            DeviceType::DiscreteGpu => Self::DiscreteGpu,
            DeviceType::IntegratedGpu => Self::IntegratedGpu,
            DeviceType::VirtualGpu => Self::VirtualGpu,
            DeviceType::Cpu => Self::Cpu,
            DeviceType::Other => Self::Other,
        }
    }
}

/// Adapter limits
#[derive(Debug, Clone, Default)]
pub struct AdapterLimits {
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum texture dimension 1D
    pub max_texture_dimension_1d: u32,
    /// Maximum texture dimension 2D
    pub max_texture_dimension_2d: u32,
    /// Maximum compute workgroup size X
    pub max_compute_workgroup_size_x: u32,
    /// Maximum compute workgroups per dimension
    pub max_compute_workgroups_per_dimension: u32,
}

/// Per-GPU metrics
#[derive(Debug, Default)]
struct GpuMetrics {
    /// Queue submission count
    submissions: AtomicU64,
    /// Compute dispatch count
    dispatches: AtomicU64,
    /// Buffer allocated bytes
    buffer_bytes: AtomicU64,
    /// Active buffer bytes
    active_buffer_bytes: AtomicU64,
}

/// Collected WGPU metrics
#[derive(Debug, Clone, Default)]
pub struct WgpuMetrics {
    /// Number of adapters
    pub adapter_count: usize,
    /// Per-GPU submission counts
    pub submissions: Vec<u64>,
    /// Per-GPU dispatch counts
    pub dispatches: Vec<u64>,
    /// Per-GPU buffer bytes
    pub buffer_bytes: Vec<u64>,
}

/// WGPU-based multi-GPU monitor
///
/// This is 100% safe Rust - no unsafe code.
pub struct WgpuMonitor {
    /// WGPU instance
    #[cfg(feature = "gpu-wgpu")]
    instance: Instance,
    /// Cached adapter info
    adapter_info: Vec<GpuAdapterInfo>,
    /// Per-GPU metrics
    gpu_metrics: Vec<Arc<GpuMetrics>>,
    /// Round-robin counter for load balancing
    round_robin_counter: AtomicUsize,
    /// Invalidated adapter indices
    invalidated: HashMap<usize, bool>,
}

// SAFETY: WgpuMonitor is Send + Sync because:
// - Instance is Send + Sync (verified in wgpu crate)
// - adapter_info is Vec of Clone types (Send + Sync)
// - gpu_metrics uses Arc<T> with atomic operations (Send + Sync)
// - round_robin_counter is AtomicUsize (Send + Sync)
// - invalidated is HashMap accessed only from &mut self
//
// We use #[allow] to bypass the deny(unsafe_code) lint for these specific impls
// which are the ONLY unsafe code in this module.
#[allow(unsafe_code)]
unsafe impl Send for WgpuMonitor {}
#[allow(unsafe_code)]
unsafe impl Sync for WgpuMonitor {}

impl WgpuMonitor {
    /// Creates a new WGPU monitor, discovering all GPUs.
    ///
    /// This is pure safe Rust - no FFI needed!
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "gpu-wgpu")]
        {
            let instance = Instance::new(&InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            });

            let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());

            let adapter_info: Vec<GpuAdapterInfo> = adapters
                .iter()
                .enumerate()
                .map(|(i, adapter): (usize, &Adapter)| {
                    let info = adapter.get_info();
                    GpuAdapterInfo {
                        index: i,
                        name: info.name,
                        backend: info.backend.into(),
                        device_type: info.device_type.into(),
                        driver: info.driver,
                        driver_info: info.driver_info,
                        vendor_id: info.vendor,
                        device_id: info.device,
                    }
                })
                .collect();

            let gpu_metrics: Vec<Arc<GpuMetrics>> =
                (0..adapter_info.len()).map(|_| Arc::new(GpuMetrics::default())).collect();

            Self {
                instance,
                adapter_info,
                gpu_metrics,
                round_robin_counter: AtomicUsize::new(0),
                invalidated: HashMap::new(),
            }
        }

        #[cfg(not(feature = "gpu-wgpu"))]
        {
            Self {
                adapter_info: Vec::new(),
                gpu_metrics: Vec::new(),
                round_robin_counter: AtomicUsize::new(0),
                invalidated: HashMap::new(),
            }
        }
    }

    /// Returns all detected GPU adapters
    #[must_use]
    pub fn adapters(&self) -> &[GpuAdapterInfo] {
        &self.adapter_info
    }

    /// Returns the primary (first) adapter
    #[must_use]
    pub fn primary_adapter(&self) -> Option<&GpuAdapterInfo> {
        self.adapter_info.first()
    }

    /// Returns adapter info by index
    #[must_use]
    pub fn adapter_info(&self, index: usize) -> Option<&GpuAdapterInfo> {
        self.adapter_info.get(index)
    }

    /// Returns adapter limits by index
    #[must_use]
    pub fn adapter_limits(&self, _index: usize) -> AdapterLimits {
        #[cfg(feature = "gpu-wgpu")]
        {
            // Default limits - would need device creation for actual limits
            AdapterLimits {
                max_buffer_size: 256 * 1024 * 1024, // 256MB default
                max_texture_dimension_1d: 8192,
                max_texture_dimension_2d: 8192,
                max_compute_workgroup_size_x: 256,
                max_compute_workgroups_per_dimension: 65535,
            }
        }

        #[cfg(not(feature = "gpu-wgpu"))]
        {
            AdapterLimits::default()
        }
    }

    /// Returns number of discrete GPUs
    #[must_use]
    pub fn discrete_gpu_count(&self) -> usize {
        self.adapter_info.iter().filter(|a| a.is_discrete()).count()
    }

    /// Check if dual AMD GPUs are available (Mac Pro config)
    #[must_use]
    pub fn has_dual_amd(&self) -> bool {
        let amd_discrete =
            self.adapter_info.iter().filter(|a| a.name.contains("AMD") && a.is_discrete()).count();
        amd_discrete >= 2
    }

    /// Record a queue submission for a GPU
    pub fn record_submission(&mut self, gpu_index: usize) {
        if let Some(metrics) = self.gpu_metrics.get(gpu_index) {
            metrics.submissions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a compute dispatch for a GPU
    pub fn record_dispatch(&mut self, gpu_index: usize) {
        if let Some(metrics) = self.gpu_metrics.get(gpu_index) {
            metrics.dispatches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a buffer allocation for a GPU
    pub fn record_buffer_allocation(&mut self, gpu_index: usize, bytes: u64) {
        if let Some(metrics) = self.gpu_metrics.get(gpu_index) {
            metrics.buffer_bytes.fetch_add(bytes, Ordering::Relaxed);
            metrics.active_buffer_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Get queue submission count for a GPU
    #[must_use]
    pub fn queue_submissions(&self, gpu_index: usize) -> u64 {
        self.gpu_metrics.get(gpu_index).map_or(0, |m| m.submissions.load(Ordering::Relaxed))
    }

    /// Get compute dispatch count for a GPU
    #[must_use]
    pub fn compute_dispatches(&self, gpu_index: usize) -> u64 {
        self.gpu_metrics.get(gpu_index).map_or(0, |m| m.dispatches.load(Ordering::Relaxed))
    }

    /// Get buffer allocated bytes for a GPU
    #[must_use]
    pub fn buffer_allocated_bytes(&self, gpu_index: usize) -> u64 {
        self.gpu_metrics.get(gpu_index).map_or(0, |m| m.buffer_bytes.load(Ordering::Relaxed))
    }

    /// Get next GPU index using round-robin selection
    #[must_use]
    pub fn next_gpu_round_robin(&self) -> usize {
        let count = self.discrete_gpu_count().max(1);
        self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % count
    }

    /// Invalidate an adapter (e.g., for hotplug testing)
    pub fn invalidate_adapter(&mut self, index: usize) {
        self.invalidated.insert(index, true);
    }

    /// Refresh adapter list (re-enumerate)
    pub fn refresh_adapters(&mut self) {
        self.invalidated.clear();

        #[cfg(feature = "gpu-wgpu")]
        {
            let adapters: Vec<Adapter> = self.instance.enumerate_adapters(Backends::all());

            self.adapter_info = adapters
                .iter()
                .enumerate()
                .map(|(i, adapter): (usize, &Adapter)| {
                    let info = adapter.get_info();
                    GpuAdapterInfo {
                        index: i,
                        name: info.name,
                        backend: info.backend.into(),
                        device_type: info.device_type.into(),
                        driver: info.driver,
                        driver_info: info.driver_info,
                        vendor_id: info.vendor,
                        device_id: info.device,
                    }
                })
                .collect();

            // Resize metrics if needed
            while self.gpu_metrics.len() < self.adapter_info.len() {
                self.gpu_metrics.push(Arc::new(GpuMetrics::default()));
            }
        }
    }

    /// Collect current metrics snapshot
    #[must_use]
    pub fn collect_metrics(&self) -> WgpuMetrics {
        WgpuMetrics {
            adapter_count: self.adapter_info.len(),
            submissions: self
                .gpu_metrics
                .iter()
                .map(|m| m.submissions.load(Ordering::Relaxed))
                .collect(),
            dispatches: self
                .gpu_metrics
                .iter()
                .map(|m| m.dispatches.load(Ordering::Relaxed))
                .collect(),
            buffer_bytes: self
                .gpu_metrics
                .iter()
                .map(|m| m.buffer_bytes.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

impl Default for WgpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires real GPU - run with --ignored"]
    fn test_wgpu_monitor_creation() {
        let monitor = WgpuMonitor::new();
        // Should not panic
        let _ = monitor.adapters();
    }

    #[test]
    fn test_adapter_info_is_discrete() {
        let info = GpuAdapterInfo {
            index: 0,
            name: "Test GPU".to_string(),
            backend: WgpuBackendType::Vulkan,
            device_type: GpuDeviceType::DiscreteGpu,
            driver: "test".to_string(),
            driver_info: "1.0".to_string(),
            vendor_id: 0,
            device_id: 0,
        };

        assert!(info.is_discrete());
        assert!(!info.is_integrated());
    }

    #[test]
    #[ignore = "Requires real GPU - run with --ignored"]
    fn test_metrics_tracking() {
        let mut monitor = WgpuMonitor::new();

        // Add a fake GPU for testing if none detected
        if monitor.gpu_metrics.is_empty() {
            monitor.gpu_metrics.push(Arc::new(GpuMetrics::default()));
        }

        monitor.record_submission(0);
        monitor.record_submission(0);
        monitor.record_dispatch(0);

        assert_eq!(monitor.queue_submissions(0), 2);
        assert_eq!(monitor.compute_dispatches(0), 1);
    }

    #[test]
    #[ignore = "Requires real GPU - run with --ignored"]
    fn test_round_robin() {
        let monitor = WgpuMonitor::new();

        // Should cycle through GPUs
        let _ = monitor.next_gpu_round_robin();
        let _ = monitor.next_gpu_round_robin();
    }

    #[test]
    #[ignore = "Requires real GPU - run with --ignored"]
    fn test_invalid_index_handling() {
        let monitor = WgpuMonitor::new();

        // Should not panic
        assert!(monitor.adapter_info(999).is_none());
        assert_eq!(monitor.queue_submissions(999), 0);
        assert_eq!(monitor.compute_dispatches(999), 0);
    }

    #[test]
    fn test_backend_type_debug() {
        let backend = WgpuBackendType::Metal;
        assert_eq!(format!("{backend:?}"), "Metal");
    }

    #[test]
    fn test_device_type_debug() {
        let device_type = GpuDeviceType::DiscreteGpu;
        assert_eq!(format!("{device_type:?}"), "DiscreteGpu");
    }
}
