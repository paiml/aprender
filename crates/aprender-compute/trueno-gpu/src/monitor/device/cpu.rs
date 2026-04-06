//! CPU compute device implementation (TRUENO-SPEC-020)
//!
//! CPU monitoring via `/proc` and `/sys` filesystem on Linux.

use super::{ComputeDevice, DeviceId, DeviceType};
use crate::GpuError;

// ============================================================================
// CPU Device Implementation
// ============================================================================

/// CPU compute device using sysinfo
#[derive(Debug)]
pub struct CpuDevice {
    name: String,
    core_count: u32,
    total_memory: u64,
    // Cached metrics (updated on refresh)
    cpu_usage: f64,
    memory_used: u64,
    temperature: Option<f64>,
}

impl CpuDevice {
    /// Create a new CPU device monitor
    #[must_use]
    pub fn new() -> Self {
        // Get CPU info from /proc/cpuinfo on Linux
        let name = Self::read_cpu_name().unwrap_or_else(|| "Unknown CPU".to_string());
        let core_count = Self::read_core_count();
        let total_memory = Self::read_total_memory();

        Self { name, core_count, total_memory, cpu_usage: 0.0, memory_used: 0, temperature: None }
    }

    fn read_cpu_name() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            for line in content.lines() {
                if line.starts_with("model name") {
                    return line.split(':').nth(1).map(|s| s.trim().to_string());
                }
            }
        }
        None
    }

    fn read_core_count() -> u32 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                return content.lines().filter(|line| line.starts_with("processor")).count() as u32;
            }
        }
        // Fallback
        std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1)
    }

    fn read_total_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        // Parse "MemTotal:       32847868 kB"
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }
        0
    }

    fn read_memory_used() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total = 0u64;
                let mut available = 0u64;

                for line in content.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if line.starts_with("MemTotal:") {
                            total = parts[1].parse().unwrap_or(0) * 1024;
                        } else if line.starts_with("MemAvailable:") {
                            available = parts[1].parse().unwrap_or(0) * 1024;
                        }
                    }
                }
                return total.saturating_sub(available);
            }
        }
        0
    }

    fn read_cpu_usage() -> f64 {
        #[cfg(target_os = "linux")]
        {
            // Read /proc/stat for CPU usage
            // This is a simplified version - real implementation would track deltas
            if let Ok(content) = std::fs::read_to_string("/proc/stat") {
                for line in content.lines() {
                    if line.starts_with("cpu ") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 5 {
                            let user: u64 = parts[1].parse().unwrap_or(0);
                            let nice: u64 = parts[2].parse().unwrap_or(0);
                            let system: u64 = parts[3].parse().unwrap_or(0);
                            let idle: u64 = parts[4].parse().unwrap_or(0);

                            let total = user + nice + system + idle;
                            let busy = user + nice + system;
                            if total > 0 {
                                return (busy as f64 / total as f64) * 100.0;
                            }
                        }
                    }
                }
            }
        }
        0.0
    }

    fn read_temperature() -> Option<f64> {
        #[cfg(target_os = "linux")]
        {
            // Try hwmon thermal zones
            if let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") {
                for entry in entries.flatten() {
                    let temp_path = entry.path().join("temp1_input");
                    if let Ok(content) = std::fs::read_to_string(&temp_path) {
                        if let Ok(millidegrees) = content.trim().parse::<i64>() {
                            return Some(millidegrees as f64 / 1000.0);
                        }
                    }
                }
            }
            // Fallback to thermal_zone
            if let Ok(content) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
                if let Ok(millidegrees) = content.trim().parse::<i64>() {
                    return Some(millidegrees as f64 / 1000.0);
                }
            }
        }
        None
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDevice for CpuDevice {
    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn compute_utilization(&self) -> Result<f64, GpuError> {
        Ok(self.cpu_usage)
    }

    fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
        // Read current CPU frequency
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            {
                if let Ok(khz) = content.trim().parse::<u64>() {
                    return Ok((khz / 1000) as u32);
                }
            }
        }
        Err(GpuError::NotSupported("CPU frequency not available".to_string()))
    }

    fn compute_temperature_c(&self) -> Result<f64, GpuError> {
        self.temperature
            .ok_or_else(|| GpuError::NotSupported("CPU temperature not available".to_string()))
    }

    fn compute_power_watts(&self) -> Result<f64, GpuError> {
        // CPU power estimation based on TDP and utilization
        // This is a rough estimate - RAPL provides better data on supported CPUs
        Err(GpuError::NotSupported("CPU power not available".to_string()))
    }

    fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
        Err(GpuError::NotSupported("CPU power limit not available".to_string()))
    }

    fn memory_used_bytes(&self) -> Result<u64, GpuError> {
        Ok(self.memory_used)
    }

    fn memory_total_bytes(&self) -> Result<u64, GpuError> {
        Ok(self.total_memory)
    }

    fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> {
        // Would need memory controller stats - not easily available
        Err(GpuError::NotSupported("Memory bandwidth not available".to_string()))
    }

    fn compute_unit_count(&self) -> u32 {
        self.core_count
    }

    fn active_compute_units(&self) -> Result<u32, GpuError> {
        // All cores are typically active
        Ok(self.core_count)
    }

    fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported("CPU has no PCIe metrics".to_string()))
    }

    fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported("CPU has no PCIe metrics".to_string()))
    }

    fn pcie_generation(&self) -> u8 {
        0 // N/A for CPU
    }

    fn pcie_width(&self) -> u8 {
        0 // N/A for CPU
    }

    fn refresh(&mut self) -> Result<(), GpuError> {
        self.cpu_usage = Self::read_cpu_usage();
        self.memory_used = Self::read_memory_used();
        self.temperature = Self::read_temperature();
        Ok(())
    }
}
