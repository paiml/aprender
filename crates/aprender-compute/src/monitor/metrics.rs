//! GPU metrics structs (TRUENO-SPEC-010 Section 4)

use std::time::{Duration, Instant};

use super::GpuMemoryMetrics;

// ============================================================================
// GPU Utilization Metrics (TRUENO-SPEC-010 Section 4.1.1)
// ============================================================================

/// GPU utilization metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuUtilization {
    /// GPU compute utilization (0-100%)
    pub gpu_percent: u32,
    /// Memory controller utilization (0-100%)
    pub memory_percent: u32,
    /// Video encoder utilization (0-100%), if available
    pub encoder_percent: Option<u32>,
    /// Video decoder utilization (0-100%), if available
    pub decoder_percent: Option<u32>,
}

// ============================================================================
// GPU Thermal Metrics (TRUENO-SPEC-010 Section 4.1.3)
// ============================================================================

/// GPU thermal metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuThermalMetrics {
    /// Current temperature in Celsius
    pub temperature_celsius: u32,
    /// Shutdown threshold temperature, if available
    pub temperature_threshold_shutdown: Option<u32>,
    /// Fan speed percentage (0-100), if available
    pub fan_speed_percent: Option<u32>,
}

impl GpuThermalMetrics {
    /// Check if temperature is in safe range (< 80°C)
    #[must_use]
    pub const fn is_safe(&self) -> bool {
        self.temperature_celsius < 80
    }

    /// Check if temperature is critical (>= 90°C)
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        self.temperature_celsius >= 90
    }

    /// Get thermal status string
    #[must_use]
    pub const fn status(&self) -> &'static str {
        match self.temperature_celsius {
            0..=50 => "COOL",
            51..=70 => "WARM",
            71..=85 => "HOT",
            86.. => "CRITICAL",
        }
    }
}

// ============================================================================
// GPU Power Metrics (TRUENO-SPEC-010 Section 4.1.3)
// ============================================================================

/// GPU power metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuPowerMetrics {
    /// Current power draw in watts
    pub power_draw_watts: f32,
    /// Power limit (TDP) in watts
    pub power_limit_watts: f32,
    /// Power state (P-state, 0 = highest performance)
    pub power_state: u32,
}

impl GpuPowerMetrics {
    /// Calculate power usage percentage
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.power_limit_watts <= 0.0 {
            0.0
        } else {
            (self.power_draw_watts as f64 / self.power_limit_watts as f64) * 100.0
        }
    }
}

// ============================================================================
// GPU Clock Metrics (TRUENO-SPEC-010 Section 4.1.4)
// ============================================================================

/// GPU clock metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuClockMetrics {
    /// Graphics/shader clock in MHz
    pub graphics_mhz: u32,
    /// Memory clock in MHz
    pub memory_mhz: u32,
    /// SM clock in MHz (NVIDIA), if available
    pub sm_mhz: Option<u32>,
}

// ============================================================================
// GPU PCIe Metrics (TRUENO-SPEC-010 Section 4.1.5)
// ============================================================================

/// GPU PCIe metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuPcieMetrics {
    /// PCIe TX throughput in bytes/sec
    pub tx_bytes_per_sec: u64,
    /// PCIe RX throughput in bytes/sec
    pub rx_bytes_per_sec: u64,
    /// PCIe link generation (1-5)
    pub link_gen: u32,
    /// PCIe link width (lanes)
    pub link_width: u32,
}

// ============================================================================
// Combined GPU Metrics Snapshot (TRUENO-SPEC-010 Section 4.2)
// ============================================================================

/// Complete GPU metrics snapshot
///
/// Contains all available metrics at a point in time.
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Device index
    pub device_index: u32,
    /// Memory metrics
    pub memory: GpuMemoryMetrics,
    /// Utilization metrics
    pub utilization: GpuUtilization,
    /// Thermal metrics (if available)
    pub thermal: Option<GpuThermalMetrics>,
    /// Power metrics (if available)
    pub power: Option<GpuPowerMetrics>,
    /// Clock metrics (if available)
    pub clocks: Option<GpuClockMetrics>,
    /// PCIe metrics (if available)
    pub pcie: Option<GpuPcieMetrics>,
}

impl GpuMetrics {
    /// Create a new metrics snapshot with only memory info
    #[must_use]
    pub fn new(device_index: u32, memory: GpuMemoryMetrics) -> Self {
        Self {
            timestamp: Instant::now(),
            device_index,
            memory,
            utilization: GpuUtilization::default(),
            thermal: None,
            power: None,
            clocks: None,
            pcie: None,
        }
    }

    /// Age of this snapshot
    #[must_use]
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}
