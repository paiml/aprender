//! NVIDIA GPU metrics collector via NVML.
//!
//! Uses the nvml-wrapper crate to collect GPU metrics from NVIDIA GPUs.
//!
//! ## Feature Flag
//!
//! Requires `monitor-nvidia` feature to be enabled.
//!
//! ## Metrics Collected
//!
//! - GPU utilization percentage
//! - Memory utilization percentage
//! - Memory used/total
//! - GPU temperature
//! - Power draw (watts)
//! - GPU clock speed (MHz)
//! - Memory clock speed (MHz)
//! - Fan speed percentage
//! - PCIe throughput (optional)

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::ring_buffer::RingBuffer;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::Nvml;
use std::time::Duration;

/// Information about a single GPU.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU index.
    pub index: u32,
    /// GPU name/model.
    pub name: String,
    /// GPU utilization percentage (0-100).
    pub gpu_util: f64,
    /// Memory utilization percentage (0-100).
    pub mem_util: f64,
    /// Memory used in bytes.
    pub mem_used: u64,
    /// Memory total in bytes.
    pub mem_total: u64,
    /// GPU temperature in Celsius.
    pub temperature: f64,
    /// Power draw in milliwatts.
    pub power_mw: u32,
    /// Power limit in milliwatts.
    pub power_limit_mw: u32,
    /// GPU clock speed in MHz.
    pub gpu_clock_mhz: u32,
    /// Memory clock speed in MHz.
    pub mem_clock_mhz: u32,
    /// Fan speed percentage (0-100), if available.
    pub fan_speed: Option<u32>,
    /// PCIe TX throughput in KB/s, if measured.
    pub pcie_tx_kbps: Option<u32>,
    /// PCIe RX throughput in KB/s, if measured.
    pub pcie_rx_kbps: Option<u32>,
}

/// Collector for NVIDIA GPU metrics via NVML.
#[derive(Debug)]
pub struct NvidiaGpuCollector {
    /// NVML instance.
    nvml: Option<Nvml>,
    /// Number of GPUs detected.
    gpu_count: u32,
    /// GPU utilization history per GPU.
    gpu_history: Vec<RingBuffer<f64>>,
    /// Memory utilization history per GPU.
    mem_history: Vec<RingBuffer<f64>>,
    /// Temperature history per GPU.
    temp_history: Vec<RingBuffer<f64>>,
    /// Power history per GPU (normalized 0-1).
    power_history: Vec<RingBuffer<f64>>,
    /// Whether to measure PCIe throughput (can impact performance).
    measure_pcie: bool,
    /// Cached GPU info.
    gpus: Vec<GpuInfo>,
}

impl NvidiaGpuCollector {
    /// Creates a new NVIDIA GPU collector.
    #[must_use]
    pub fn new() -> Self {
        let nvml = Nvml::init().ok();
        let gpu_count = nvml.as_ref().and_then(|n| n.device_count().ok()).unwrap_or(0);

        let mut gpu_history = Vec::with_capacity(gpu_count as usize);
        let mut mem_history = Vec::with_capacity(gpu_count as usize);
        let mut temp_history = Vec::with_capacity(gpu_count as usize);
        let mut power_history = Vec::with_capacity(gpu_count as usize);

        for _ in 0..gpu_count {
            gpu_history.push(RingBuffer::new(300));
            mem_history.push(RingBuffer::new(300));
            temp_history.push(RingBuffer::new(300));
            power_history.push(RingBuffer::new(300));
        }

        Self {
            nvml,
            gpu_count,
            gpu_history,
            mem_history,
            temp_history,
            power_history,
            measure_pcie: false,
            gpus: Vec::new(),
        }
    }

    /// Enables PCIe throughput measurement.
    ///
    /// Note: This can have a small performance impact on the GPU.
    pub fn enable_pcie_measurement(&mut self) {
        self.measure_pcie = true;
    }

    /// Returns the number of GPUs detected.
    #[must_use]
    pub fn gpu_count(&self) -> u32 {
        self.gpu_count
    }

    /// Returns GPU utilization history for a specific GPU.
    #[must_use]
    pub fn gpu_history(&self, index: usize) -> Option<&RingBuffer<f64>> {
        self.gpu_history.get(index)
    }

    /// Returns memory utilization history for a specific GPU.
    #[must_use]
    pub fn mem_history(&self, index: usize) -> Option<&RingBuffer<f64>> {
        self.mem_history.get(index)
    }

    /// Returns temperature history for a specific GPU.
    #[must_use]
    pub fn temp_history(&self, index: usize) -> Option<&RingBuffer<f64>> {
        self.temp_history.get(index)
    }

    /// Returns power history for a specific GPU (normalized 0-1).
    #[must_use]
    pub fn power_history(&self, index: usize) -> Option<&RingBuffer<f64>> {
        self.power_history.get(index)
    }

    /// Returns cached GPU information.
    #[must_use]
    pub fn gpus(&self) -> &[GpuInfo] {
        &self.gpus
    }

    /// Collects metrics from all GPUs.
    fn collect_all(&mut self) -> Result<Vec<GpuInfo>> {
        let nvml = self.nvml.as_ref().ok_or_else(|| MonitorError::CollectionFailed {
            collector: "nvidia_gpu",
            message: "NVML not initialized".to_string(),
        })?;

        let mut gpus = Vec::with_capacity(self.gpu_count as usize);

        for i in 0..self.gpu_count {
            let device = nvml.device_by_index(i).map_err(|e| MonitorError::CollectionFailed {
                collector: "nvidia_gpu",
                message: format!("Failed to get GPU {i}: {e}"),
            })?;

            let name = device.name().unwrap_or_else(|_| format!("GPU {i}"));

            // Utilization
            let (gpu_util, mem_util) = device
                .utilization_rates()
                .map(|u| (f64::from(u.gpu), f64::from(u.memory)))
                .unwrap_or((0.0, 0.0));

            // Memory
            let (mem_used, mem_total) =
                device.memory_info().map(|m| (m.used, m.total)).unwrap_or((0, 1));

            // Temperature
            let temperature =
                device.temperature(TemperatureSensor::Gpu).map(f64::from).unwrap_or(0.0);

            // Power
            let power_mw = device.power_usage().unwrap_or(0);
            let power_limit_mw = device.enforced_power_limit().unwrap_or(1);

            // Clocks
            let gpu_clock_mhz = device
                .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)
                .unwrap_or(0);
            let mem_clock_mhz =
                device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory).unwrap_or(0);

            // Fan speed (may not be available on all GPUs)
            let fan_speed = device.fan_speed(0).ok();

            // PCIe throughput (optional, can impact performance)
            let (pcie_tx_kbps, pcie_rx_kbps) = if self.measure_pcie {
                let tx = device
                    .pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Send)
                    .ok();
                let rx = device
                    .pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Receive)
                    .ok();
                (tx, rx)
            } else {
                (None, None)
            };

            gpus.push(GpuInfo {
                index: i,
                name,
                gpu_util,
                mem_util,
                mem_used,
                mem_total,
                temperature,
                power_mw,
                power_limit_mw,
                gpu_clock_mhz,
                mem_clock_mhz,
                fan_speed,
                pcie_tx_kbps,
                pcie_rx_kbps,
            });
        }

        Ok(gpus)
    }
}

impl Default for NvidiaGpuCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for NvidiaGpuCollector {
    fn id(&self) -> &'static str {
        "nvidia_gpu"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let gpus = self.collect_all()?;
        let mut metrics = Metrics::new();

        metrics.insert("gpu.count", MetricValue::Counter(u64::from(self.gpu_count)));

        for (i, gpu) in gpus.iter().enumerate() {
            let prefix = format!("gpu.{i}");

            // Utilization
            metrics.insert(format!("{prefix}.util"), gpu.gpu_util);
            metrics.insert(format!("{prefix}.mem_util"), gpu.mem_util);

            // Memory
            metrics.insert(format!("{prefix}.mem_used"), MetricValue::Counter(gpu.mem_used));
            metrics.insert(format!("{prefix}.mem_total"), MetricValue::Counter(gpu.mem_total));

            // Temperature
            metrics.insert(format!("{prefix}.temp"), gpu.temperature);

            // Power
            metrics.insert(
                format!("{prefix}.power_mw"),
                MetricValue::Counter(u64::from(gpu.power_mw)),
            );
            metrics.insert(
                format!("{prefix}.power_limit_mw"),
                MetricValue::Counter(u64::from(gpu.power_limit_mw)),
            );

            // Clocks
            metrics.insert(
                format!("{prefix}.gpu_clock_mhz"),
                MetricValue::Counter(u64::from(gpu.gpu_clock_mhz)),
            );
            metrics.insert(
                format!("{prefix}.mem_clock_mhz"),
                MetricValue::Counter(u64::from(gpu.mem_clock_mhz)),
            );

            // Fan speed
            if let Some(fan) = gpu.fan_speed {
                metrics.insert(format!("{prefix}.fan_speed"), f64::from(fan));
            }

            // PCIe throughput
            if let Some(tx) = gpu.pcie_tx_kbps {
                metrics
                    .insert(format!("{prefix}.pcie_tx_kbps"), MetricValue::Counter(u64::from(tx)));
            }
            if let Some(rx) = gpu.pcie_rx_kbps {
                metrics
                    .insert(format!("{prefix}.pcie_rx_kbps"), MetricValue::Counter(u64::from(rx)));
            }

            // Update history
            if let Some(history) = self.gpu_history.get_mut(i) {
                history.push(gpu.gpu_util / 100.0);
            }
            if let Some(history) = self.mem_history.get_mut(i) {
                history.push(gpu.mem_util / 100.0);
            }
            if let Some(history) = self.temp_history.get_mut(i) {
                history.push(gpu.temperature);
            }
            if let Some(history) = self.power_history.get_mut(i) {
                let power_norm = if gpu.power_limit_mw > 0 {
                    f64::from(gpu.power_mw) / f64::from(gpu.power_limit_mw)
                } else {
                    0.0
                };
                history.push(power_norm);
            }
        }

        self.gpus = gpus;
        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        self.nvml.is_some() && self.gpu_count > 0
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "NVIDIA GPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvidia_collector_new() {
        let collector = NvidiaGpuCollector::new();
        // May or may not have GPUs, but shouldn't panic
        let _ = collector.gpu_count();
    }

    #[test]
    fn test_nvidia_collector_pcie_toggle() {
        let mut collector = NvidiaGpuCollector::new();
        assert!(!collector.measure_pcie);
        collector.enable_pcie_measurement();
        assert!(collector.measure_pcie);
    }

    #[test]
    fn test_nvidia_collector_history_bounds() {
        let collector = NvidiaGpuCollector::new();
        // Accessing out of bounds should return None
        assert!(collector.gpu_history(999).is_none());
        assert!(collector.mem_history(999).is_none());
        assert!(collector.temp_history(999).is_none());
        assert!(collector.power_history(999).is_none());
    }

    #[test]
    fn test_gpu_info_struct() {
        let info = GpuInfo {
            index: 0,
            name: "Test GPU".to_string(),
            gpu_util: 50.0,
            mem_util: 30.0,
            mem_used: 4 * 1024 * 1024 * 1024,
            mem_total: 8 * 1024 * 1024 * 1024,
            temperature: 65.0,
            power_mw: 150_000,
            power_limit_mw: 250_000,
            gpu_clock_mhz: 1800,
            mem_clock_mhz: 7000,
            fan_speed: Some(45),
            pcie_tx_kbps: Some(1000),
            pcie_rx_kbps: Some(2000),
        };

        assert_eq!(info.index, 0);
        assert_eq!(info.gpu_util, 50.0);
        assert_eq!(info.fan_speed, Some(45));
    }
}
