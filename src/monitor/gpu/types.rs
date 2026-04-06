//! GPU metric types for monitoring.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Mock GPU hardware constants (modelled after NVIDIA RTX 4090)
// ---------------------------------------------------------------------------

/// Total VRAM in megabytes for the mock GPU device
const MOCK_GPU_MEMORY_TOTAL_MB: u64 = 24576;
/// Typical board power draw in watts
const MOCK_GPU_POWER_WATTS: f32 = 250.0;
/// Maximum power limit in watts
const MOCK_GPU_POWER_LIMIT_WATTS: f32 = 450.0;
/// Core clock frequency in MHz
const MOCK_GPU_CLOCK_MHZ: u32 = 2100;
/// Memory clock frequency in MHz
const MOCK_GPU_MEMORY_CLOCK_MHZ: u32 = 10_000;
/// PCIe receive throughput in KB/s
const MOCK_GPU_PCIE_RX_KBPS: u64 = 2000;
/// Placeholder PID for the mock training process
const MOCK_PROCESS_PID: u32 = 12345;

/// Process using GPU resources
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuProcess {
    /// Process ID
    pub pid: u32,
    /// Full path to executable
    pub exe_path: String,
    /// GPU memory used by this process in MB
    pub gpu_memory_mb: u64,
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// Resident set size (RSS) in MB
    pub rss_mb: u64,
}

/// GPU metrics snapshot (inspired by btop's GPU visualization)
///
/// Reference: btop `src/btop_shared.hpp` lines 130-171
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Device index
    pub device_id: u32,
    /// GPU name (e.g., "RTX 4090")
    pub name: String,
    /// GPU compute utilization (0-100%)
    pub utilization_percent: u32,
    /// Used VRAM in MB
    pub memory_used_mb: u64,
    /// Total VRAM in MB
    pub memory_total_mb: u64,
    /// Memory utilization (0-100%)
    pub memory_utilization_percent: u32,
    /// GPU temperature in Celsius
    pub temperature_celsius: u32,
    /// Current power draw in watts
    pub power_watts: f32,
    /// Power limit in watts
    pub power_limit_watts: f32,
    /// Graphics clock in MHz
    pub clock_mhz: u32,
    /// Memory clock in MHz
    pub memory_clock_mhz: u32,
    /// PCIe transmit throughput in KB/s
    pub pcie_tx_kbps: u64,
    /// PCIe receive throughput in KB/s
    pub pcie_rx_kbps: u64,
    /// Fan speed percentage (0-100%)
    pub fan_speed_percent: u32,
    /// Processes using this GPU
    pub processes: Vec<GpuProcess>,
}

impl GpuMetrics {
    /// Create mock metrics for testing
    pub fn mock(device_id: u32) -> Self {
        Self {
            device_id,
            name: format!("Mock GPU {device_id}"),
            utilization_percent: 75,
            memory_used_mb: 8192,
            memory_total_mb: MOCK_GPU_MEMORY_TOTAL_MB,
            memory_utilization_percent: 33,
            temperature_celsius: 65,
            power_watts: MOCK_GPU_POWER_WATTS,
            power_limit_watts: MOCK_GPU_POWER_LIMIT_WATTS,
            clock_mhz: MOCK_GPU_CLOCK_MHZ,
            memory_clock_mhz: MOCK_GPU_MEMORY_CLOCK_MHZ,
            pcie_tx_kbps: 1000,
            pcie_rx_kbps: MOCK_GPU_PCIE_RX_KBPS,
            fan_speed_percent: 50,
            processes: vec![GpuProcess {
                pid: MOCK_PROCESS_PID,
                exe_path: "/usr/bin/mock_training".to_string(),
                gpu_memory_mb: 4096,
                cpu_percent: 95.0,
                rss_mb: 2048,
            }],
        }
    }

    /// Calculate memory utilization percentage
    pub fn memory_percent(&self) -> f64 {
        if self.memory_total_mb == 0 {
            return 0.0;
        }
        self.memory_used_mb as f64 / self.memory_total_mb as f64 * 100.0
    }

    /// Calculate power utilization percentage
    pub fn power_percent(&self) -> f64 {
        if self.power_limit_watts <= 0.0 {
            return 0.0;
        }
        f64::from(self.power_watts) / f64::from(self.power_limit_watts) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_metrics_mock() {
        let m = GpuMetrics::mock(0);
        assert_eq!(m.device_id, 0);
        assert!(!m.name.is_empty());
        assert!(m.utilization_percent <= 100);
    }

    #[test]
    fn test_gpu_metrics_memory_percent() {
        let mut m = GpuMetrics::mock(0);
        m.memory_used_mb = 8000;
        m.memory_total_mb = 16000;
        assert!((m.memory_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_metrics_memory_percent_zero_total() {
        let mut m = GpuMetrics::mock(0);
        m.memory_total_mb = 0;
        assert!((m.memory_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_metrics_power_percent() {
        let mut m = GpuMetrics::mock(0);
        m.power_watts = 225.0;
        m.power_limit_watts = 450.0;
        assert!((m.power_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_metrics_power_percent_zero_limit() {
        let mut m = GpuMetrics::mock(0);
        m.power_limit_watts = 0.0;
        assert!((m.power_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_metrics_default() {
        let m = GpuMetrics::default();
        assert_eq!(m.device_id, 0);
        assert!(m.name.is_empty());
        assert_eq!(m.utilization_percent, 0);
    }

    #[test]
    fn test_gpu_metrics_clone() {
        let metrics = GpuMetrics::mock(0);
        let cloned = metrics.clone();
        assert_eq!(metrics.device_id, cloned.device_id);
        assert_eq!(metrics.name, cloned.name);
    }

    #[test]
    fn test_gpu_metrics_serde() {
        let metrics = GpuMetrics::mock(0);
        let json = serde_json::to_string(&metrics).expect("JSON serialization should succeed");
        let parsed: GpuMetrics =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(metrics.device_id, parsed.device_id);
        assert_eq!(metrics.utilization_percent, parsed.utilization_percent);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_memory_percent_bounds(used in 0u64..100000, total in 1u64..100000) {
            let m = GpuMetrics {
                memory_used_mb: used,
                memory_total_mb: total,
                ..Default::default()
            };
            let percent = m.memory_percent();
            prop_assert!(percent >= 0.0);
            // Can be > 100 if used > total (which is invalid but shouldn't crash)
        }

        #[test]
        fn prop_power_percent_bounds(power in 0.0f32..1000.0, limit in 0.1f32..1000.0) {
            let m = GpuMetrics {
                power_watts: power,
                power_limit_watts: limit,
                ..Default::default()
            };
            let percent = m.power_percent();
            prop_assert!(percent >= 0.0);
        }
    }
}
