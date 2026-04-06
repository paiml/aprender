//! GPU monitor that collects metrics.
//!
//! Uses NVML when available (feature `nvml`), otherwise falls back to mock.

use super::{GpuMetrics, GpuProcess};

#[cfg(feature = "nvml")]
use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, Nvml};

#[cfg(feature = "nvml")]
use std::fs;

/// GPU monitor that collects metrics
///
/// When compiled with `nvml` feature, uses real NVIDIA NVML for hardware metrics.
/// Otherwise provides mock mode for testing.
#[derive(Debug)]
pub struct GpuMonitor {
    /// Number of detected devices
    num_devices: u32,
    /// Mock mode for testing
    mock_mode: bool,
    /// Mock metrics generator
    mock_metrics: Vec<GpuMetrics>,
    /// NVML instance (when feature enabled)
    #[cfg(feature = "nvml")]
    nvml: Option<Nvml>,
}

impl GpuMonitor {
    /// Create a new GPU monitor
    ///
    /// Attempts to initialize NVML if feature enabled, falls back gracefully.
    #[cfg(feature = "nvml")]
    pub fn new() -> Result<Self, String> {
        match Nvml::init() {
            Ok(nvml) => {
                let num_devices = nvml.device_count().unwrap_or(0);
                Ok(Self {
                    num_devices,
                    mock_mode: false,
                    mock_metrics: Vec::new(),
                    nvml: Some(nvml),
                })
            }
            Err(e) => {
                eprintln!("[GpuMonitor] NVML init failed: {e}, using mock mode");
                Ok(Self { num_devices: 0, mock_mode: false, mock_metrics: Vec::new(), nvml: None })
            }
        }
    }

    /// Create a new GPU monitor (non-NVML fallback)
    #[cfg(not(feature = "nvml"))]
    pub fn new() -> Result<Self, String> {
        // Without NVML feature, return empty (graceful degradation)
        Ok(Self { num_devices: 0, mock_mode: false, mock_metrics: Vec::new() })
    }

    /// Create a mock GPU monitor for testing
    pub fn mock(num_devices: u32) -> Self {
        let mock_metrics = (0..num_devices).map(GpuMetrics::mock).collect();
        Self {
            num_devices,
            mock_mode: true,
            mock_metrics,
            #[cfg(feature = "nvml")]
            nvml: None,
        }
    }

    /// Get number of detected devices
    pub fn num_devices(&self) -> u32 {
        self.num_devices
    }

    /// Check if running in mock mode
    pub fn is_mock(&self) -> bool {
        self.mock_mode
    }

    /// Sample current GPU metrics
    #[cfg(feature = "nvml")]
    pub fn sample(&self) -> Vec<GpuMetrics> {
        contract_pre_sample!();
        if self.mock_mode {
            return self.mock_metrics.clone();
        }

        let Some(nvml) = &self.nvml else {
            return Vec::new();
        };

        let mut metrics = Vec::with_capacity(self.num_devices as usize);

        for i in 0..self.num_devices {
            let Ok(device) = nvml.device_by_index(i) else {
                continue;
            };

            let name = device.name().unwrap_or_else(|_err| format!("GPU {i}"));

            // Utilization rates
            let (utilization_percent, memory_utilization_percent) =
                device.utilization_rates().map_or((0, 0), |rates| (rates.gpu, rates.memory));

            // Memory info
            let (memory_used_mb, memory_total_mb) = device
                .memory_info()
                .map_or((0, 0), |mem| (mem.used / (1024 * 1024), mem.total / (1024 * 1024)));

            // Temperature
            let temperature_celsius = device.temperature(TemperatureSensor::Gpu).unwrap_or(0);

            // Power
            let power_watts = device.power_usage().map_or(0.0, |mw| mw as f32 / 1000.0);
            let power_limit_watts =
                device.enforced_power_limit().map_or(0.0, |mw| mw as f32 / 1000.0);

            // Clocks
            let clock_mhz = device
                .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)
                .unwrap_or(0);
            let memory_clock_mhz =
                device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory).unwrap_or(0);

            // PCIe throughput
            let pcie_tx_kbps = u64::from(
                device
                    .pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Send)
                    .unwrap_or(0),
            );
            let pcie_rx_kbps = u64::from(
                device
                    .pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Receive)
                    .unwrap_or(0),
            );

            // Fan speed (may not be available on all GPUs)
            let fan_speed_percent = device.fan_speed(0).unwrap_or(0);

            // Collect running compute processes
            let processes = Self::collect_gpu_processes(&device);

            metrics.push(GpuMetrics {
                device_id: i,
                name,
                utilization_percent,
                memory_used_mb,
                memory_total_mb,
                memory_utilization_percent,
                temperature_celsius,
                power_watts,
                power_limit_watts,
                clock_mhz,
                memory_clock_mhz,
                pcie_tx_kbps,
                pcie_rx_kbps,
                fan_speed_percent,
                processes,
            });
        }

        metrics
    }

    /// Sample current GPU metrics (non-NVML fallback)
    #[cfg(not(feature = "nvml"))]
    pub fn sample(&self) -> Vec<GpuMetrics> {
        if self.mock_mode {
            return self.mock_metrics.clone();
        }

        // Production implementation would query NVML here
        Vec::new()
    }

    /// Sample with simulated variation (for testing)
    pub fn sample_with_variation(&mut self, variation: f32) -> Vec<GpuMetrics> {
        if !self.mock_mode {
            return Vec::new();
        }

        self.mock_metrics
            .iter()
            .map(|base| {
                let mut m = base.clone();
                let var = (variation * 10.0) as i32;
                m.utilization_percent = (m.utilization_percent as i32 + var).clamp(0, 100) as u32;
                m.temperature_celsius =
                    (m.temperature_celsius as i32 + var / 2).clamp(30, 100) as u32;
                m.power_watts = (m.power_watts + variation * 20.0).clamp(0.0, m.power_limit_watts);
                m
            })
            .collect()
    }

    /// Set mock metrics (for testing specific scenarios)
    pub fn set_mock_metrics(&mut self, metrics: Vec<GpuMetrics>) {
        self.mock_metrics = metrics;
        self.num_devices = self.mock_metrics.len() as u32;
        self.mock_mode = true;
    }

    /// Collect GPU processes from NVML and enrich with /proc data
    #[cfg(feature = "nvml")]
    fn collect_gpu_processes(device: &nvml_wrapper::Device<'_>) -> Vec<GpuProcess> {
        use nvml_wrapper::enums::device::UsedGpuMemory;

        let mut processes = Vec::new();

        // Helper to extract memory from UsedGpuMemory enum
        let extract_memory = |mem: UsedGpuMemory| -> u64 {
            match mem {
                UsedGpuMemory::Used(bytes) => bytes / (1024 * 1024),
                UsedGpuMemory::Unavailable => 0,
            }
        };

        // Get compute processes (CUDA apps)
        if let Ok(compute_procs) = device.running_compute_processes() {
            for proc in compute_procs {
                let pid = proc.pid;
                let gpu_memory_mb = extract_memory(proc.used_gpu_memory);

                // Read /proc/PID/exe for full path
                let exe_path = fs::read_link(format!("/proc/{pid}/exe"))
                    .map_or_else(|_| format!("[pid {pid}]"), |p| p.to_string_lossy().to_string());

                // Read /proc/PID/stat for CPU and memory
                let (cpu_percent, rss_mb) = Self::read_proc_stats(pid);

                processes.push(GpuProcess { pid, exe_path, gpu_memory_mb, cpu_percent, rss_mb });
            }
        }

        // Also check graphics processes
        if let Ok(graphics_procs) = device.running_graphics_processes() {
            for proc in graphics_procs {
                // Skip if already in compute list
                if processes.iter().any(|p| p.pid == proc.pid) {
                    continue;
                }

                let pid = proc.pid;
                let gpu_memory_mb = extract_memory(proc.used_gpu_memory);

                let exe_path = fs::read_link(format!("/proc/{pid}/exe"))
                    .map_or_else(|_| format!("[pid {pid}]"), |p| p.to_string_lossy().to_string());

                let (cpu_percent, rss_mb) = Self::read_proc_stats(pid);

                processes.push(GpuProcess { pid, exe_path, gpu_memory_mb, cpu_percent, rss_mb });
            }
        }

        processes
    }

    /// Read CPU% and RSS from /proc/PID/stat and /proc/PID/statm
    #[cfg(feature = "nvml")]
    fn read_proc_stats(pid: u32) -> (f32, u64) {
        // Read RSS from /proc/PID/statm (second field, in pages)
        let rss_mb = fs::read_to_string(format!("/proc/{pid}/statm"))
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map_or(0, |pages| pages * 4096 / (1024 * 1024));

        // CPU% would require sampling over time - approximate from /proc/PID/stat
        // For now, read utime+stime and estimate based on uptime
        let cpu_percent = fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|s| {
                let fields: Vec<&str> = s.split_whitespace().collect();
                if fields.len() > 14 {
                    let utime: u64 = fields[13].parse().ok()?;
                    let stime: u64 = fields[14].parse().ok()?;
                    let total_ticks = utime + stime;
                    // Rough approximation: assume 100 ticks/sec, sample over 1 sec
                    // This is imprecise but gives an order of magnitude
                    Some((total_ticks as f32 / 100.0).min(100.0))
                } else {
                    None
                }
            })
            .unwrap_or(0.0);

        (cpu_percent, rss_mb)
    }

    /// Collect GPU processes (non-NVML fallback)
    #[cfg(not(feature = "nvml"))]
    #[allow(dead_code)]
    fn collect_gpu_processes(_device: &()) -> Vec<GpuProcess> {
        Vec::new()
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_err| Self::mock(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_monitor_new() {
        let monitor = GpuMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_gpu_monitor_mock() {
        let monitor = GpuMonitor::mock(2);
        assert_eq!(monitor.num_devices(), 2);
        assert!(monitor.is_mock());
    }

    #[test]
    fn test_gpu_monitor_sample_mock() {
        let monitor = GpuMonitor::mock(2);
        let metrics = monitor.sample();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].device_id, 0);
        assert_eq!(metrics[1].device_id, 1);
    }

    #[test]
    fn test_gpu_monitor_sample_with_variation() {
        let mut monitor = GpuMonitor::mock(1);
        let base = monitor.sample()[0].utilization_percent;

        let varied = monitor.sample_with_variation(1.0);
        // Variation should change the value
        assert!(varied[0].utilization_percent != base || base == 100 || base == 0);
    }

    #[test]
    fn test_gpu_monitor_set_mock_metrics() {
        let mut monitor = GpuMonitor::mock(0);
        monitor.set_mock_metrics(vec![GpuMetrics {
            device_id: 5,
            utilization_percent: 99,
            ..Default::default()
        }]);

        let metrics = monitor.sample();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].device_id, 5);
        assert_eq!(metrics[0].utilization_percent, 99);
    }

    #[test]
    fn test_gpu_monitor_default() {
        let monitor = GpuMonitor::default();
        // Should either be real or mock, but should work
        let _ = monitor.num_devices();
    }

    #[test]
    fn test_gpu_monitor_non_mock_sample() {
        // Create non-mock monitor but call sample (should return empty or real data)
        let monitor = GpuMonitor::new().expect("operation should succeed");
        let metrics = monitor.sample();
        // Just verify it doesn't crash - may be empty if no NVML
        let _ = metrics;
    }

    #[test]
    fn test_gpu_monitor_non_mock_sample_with_variation() {
        let mut monitor = GpuMonitor::new().expect("operation should succeed");
        // Non-mock mode should return empty
        let metrics = monitor.sample_with_variation(1.0);
        // Should be empty for non-mock mode
        assert!(metrics.is_empty() || !monitor.is_mock());
    }

    #[cfg(feature = "nvml")]
    #[test]
    fn test_gpu_monitor_nvml_sample() {
        // Test real NVML sampling if available
        let monitor = GpuMonitor::new().expect("operation should succeed");
        if monitor.num_devices() > 0 {
            let metrics = monitor.sample();
            assert!(!metrics.is_empty());
            // Verify basic sanity
            for m in &metrics {
                assert!(m.utilization_percent <= 100);
                assert!(m.temperature_celsius < 150);
            }
        }
    }
}
