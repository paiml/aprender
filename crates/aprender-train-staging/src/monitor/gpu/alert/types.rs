//! GPU alert type definitions.

use serde::{Deserialize, Serialize};

/// GPU alert types for Andon system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAlert {
    /// Temperature exceeds threshold (thermal throttling imminent)
    ThermalThrottling { device: u32, temp: u32, threshold: u32 },
    /// VRAM usage exceeds threshold (OOM imminent)
    MemoryPressure { device: u32, used_percent: u32, threshold: u32 },
    /// Power draw exceeds threshold percentage of limit
    PowerLimit { device: u32, power_percent: u32, threshold: u32 },
    /// GPU utilization dropped to zero (possible hang)
    GpuIdle { device: u32, duration_secs: u32 },
}

impl GpuAlert {
    /// Get severity level (0-100, higher is more severe)
    pub fn severity(&self) -> u32 {
        match self {
            GpuAlert::ThermalThrottling { temp, threshold, .. } => {
                let excess = temp.saturating_sub(*threshold);
                50 + excess.min(50)
            }
            GpuAlert::MemoryPressure { used_percent, .. } => {
                if *used_percent >= 99 {
                    100
                } else if *used_percent >= 95 {
                    80
                } else {
                    60
                }
            }
            GpuAlert::PowerLimit { power_percent, .. } => {
                if *power_percent >= 100 {
                    70
                } else {
                    50
                }
            }
            GpuAlert::GpuIdle { duration_secs, .. } => {
                if *duration_secs > 60 {
                    90
                } else if *duration_secs > 30 {
                    70
                } else {
                    40
                }
            }
        }
    }

    /// Get human-readable description
    pub fn message(&self) -> String {
        match self {
            GpuAlert::ThermalThrottling { device, temp, threshold } => {
                format!("GPU {device} thermal throttling: {temp}°C exceeds {threshold}°C threshold")
            }
            GpuAlert::MemoryPressure { device, used_percent, threshold } => {
                format!(
                    "GPU {device} memory pressure: {used_percent}% exceeds {threshold}% threshold"
                )
            }
            GpuAlert::PowerLimit { device, power_percent, threshold } => {
                format!("GPU {device} power limit: {power_percent}% exceeds {threshold}% threshold")
            }
            GpuAlert::GpuIdle { device, duration_secs } => {
                format!("GPU {device} idle for {duration_secs} seconds (possible hang)")
            }
        }
    }
}
