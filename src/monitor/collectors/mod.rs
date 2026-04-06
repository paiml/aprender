//! Metric collectors for system and ML workload monitoring.
//!
//! This module provides collectors for gathering metrics from various sources:
//!
//! - **System**: CPU, memory, disk, network, processes, sensors, battery
//! - **GPU**: NVIDIA (via NVML), AMD (via ROCm SMI)
//! - **Stack**: realizar, entrenar, trueno-zram, repartir

// Core system collectors
pub mod battery;
pub mod battery_sensors_simd;
pub mod cpu;
pub mod cpu_simd;
pub mod disk;
pub mod disk_simd;
pub mod gpu_simd;
pub mod memory;
pub mod memory_simd;
pub mod network;
pub mod network_simd;
pub mod process;
pub mod process_simd;
pub mod sensors;

pub use battery::BatteryCollector;
pub use battery_sensors_simd::SimdBatterySensorsCollector;
pub use cpu::{CpuCollector, CpuFrequency, LoadAverage};
pub use cpu_simd::SimdCpuCollector;
pub use disk::DiskCollector;
pub use disk_simd::SimdDiskCollector;
pub use gpu_simd::{GpuMetricsSoA, SimdGpuHistory};
pub use memory::MemoryCollector;
pub use memory_simd::SimdMemoryCollector;
pub use network::NetworkCollector;
pub use network_simd::SimdNetworkCollector;
pub use process::ProcessCollector;
pub use process_simd::SimdProcessCollector;
pub use sensors::SensorCollector;

// GPU collectors (feature-gated)
#[cfg(feature = "monitor-nvidia")]
#[cfg_attr(docsrs, doc(cfg(feature = "monitor-nvidia")))]
pub mod gpu_nvidia;

#[cfg(feature = "monitor-nvidia")]
pub use gpu_nvidia::{GpuInfo, NvidiaGpuCollector};

// AMD GPU (always compiled, dynamically loads librocm_smi64.so at runtime)
#[cfg(target_os = "linux")]
pub mod gpu_amd;

#[cfg(target_os = "linux")]
pub use gpu_amd::{AmdGpuCollector, AmdGpuInfo};

// Apple GPU (macOS only)
#[cfg(target_os = "macos")]
pub mod gpu_apple;

#[cfg(target_os = "macos")]
pub use gpu_apple::{AppleGpuCollector, AppleGpuInfo};

// Stack collectors (feature-gated)
#[cfg(feature = "monitor-stack")]
pub mod stack;
