//! GPU Monitoring, Tracing, and Visualization (TRUENO-SPEC-010)
//!
//! This module provides comprehensive GPU monitoring capabilities:
//! - Device discovery and information
//! - Real-time metrics collection
//! - Cross-platform support (wgpu + optional CUDA)
//!
//! # Design Philosophy
//!
//! **Dual-Backend Architecture**: Supports both wgpu (cross-platform) and
//! trueno-gpu (native CUDA) for maximum flexibility.
//!
//! # References
//!
//! - Nickolls et al. (2008): GPU parallel computing model
//! - Gregg (2016): Flame graph visualization
//! - btop: NVML/ROCm SMI patterns
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::monitor::{GpuDeviceInfo, GpuMonitor};
//!
//! // Query device information
//! let info = GpuDeviceInfo::query()?;
//! println!("GPU: {}", info.name);
//! println!("VRAM: {} MB", info.vram_total / (1024 * 1024));
//!
//! // Start background monitoring
//! let monitor = GpuMonitor::new(0)?;
//! let metrics = monitor.collect()?;
//! println!("VRAM used: {} / {}", metrics.memory.used, metrics.memory.total);
//! ```

pub mod backends;
mod compute_device;
mod config;
mod gpu_monitor;
mod memory;
mod metrics;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types at the module level
pub use backends::cuda_monitor_available;
#[cfg(feature = "cuda-monitor")]
pub use backends::{enumerate_cuda_devices, query_cuda_device_info, query_cuda_memory};
pub use compute_device::{ComputeDevice, DeviceId, DeviceType};
pub use config::{MonitorConfig, MonitorError};
pub use gpu_monitor::GpuMonitor;
pub use memory::GpuMemoryMetrics;
pub use metrics::{
    GpuClockMetrics, GpuMetrics, GpuPcieMetrics, GpuPowerMetrics, GpuThermalMetrics, GpuUtilization,
};
pub use types::{GpuBackend, GpuDeviceInfo, GpuVendor};
