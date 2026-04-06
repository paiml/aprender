//! Andon threshold configuration.

use serde::{Deserialize, Serialize};

/// Andon thresholds for GPU alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndonThresholds {
    /// Temperature warning threshold in Celsius
    pub thermal_warning: u32,
    /// Temperature critical threshold in Celsius
    pub thermal_critical: u32,
    /// Memory usage warning threshold (0-100%)
    pub memory_warning: u32,
    /// Memory usage critical threshold (0-100%)
    pub memory_critical: u32,
    /// Power usage warning threshold (0-100% of limit)
    pub power_warning: u32,
    /// Idle duration threshold in seconds
    pub idle_threshold_secs: u32,
}

impl Default for AndonThresholds {
    fn default() -> Self {
        Self {
            thermal_warning: 80,
            thermal_critical: 90,
            memory_warning: 90,
            memory_critical: 95,
            power_warning: 95,
            idle_threshold_secs: 30,
        }
    }
}
