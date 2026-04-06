//! GPU thermal monitoring and threshold evaluation.
//!
//! Pure functions for evaluating GPU temperature readings against
//! configurable thresholds.

use serde::{Deserialize, Serialize};

use crate::supervisor::strategy::HealthAction;

/// Thermal threshold configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Temperature (°C) below which the GPU is considered healthy.
    pub healthy_below_c: u32,
    /// Temperature (°C) at or above which the workload should be throttled.
    pub throttle_at_c: u32,
    /// Temperature (°C) at or above which the GPU should be shut down.
    pub shutdown_at_c: u32,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self { healthy_below_c: 80, throttle_at_c: 85, shutdown_at_c: 95 }
    }
}

/// Evaluate a temperature reading against the given thresholds.
#[must_use]
pub fn evaluate_thermal(config: &ThermalConfig, temp_c: u32) -> HealthAction {
    if temp_c >= config.shutdown_at_c {
        HealthAction::Shutdown
    } else if temp_c >= config.throttle_at_c {
        HealthAction::Throttle
    } else {
        HealthAction::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_below_threshold() {
        let cfg = ThermalConfig::default();
        assert_eq!(evaluate_thermal(&cfg, 70), HealthAction::Healthy);
    }

    #[test]
    fn throttle_at_threshold() {
        let cfg = ThermalConfig::default();
        assert_eq!(evaluate_thermal(&cfg, 85), HealthAction::Throttle);
    }

    #[test]
    fn shutdown_at_threshold() {
        let cfg = ThermalConfig::default();
        assert_eq!(evaluate_thermal(&cfg, 95), HealthAction::Shutdown);
    }

    #[test]
    fn shutdown_above_threshold() {
        let cfg = ThermalConfig::default();
        assert_eq!(evaluate_thermal(&cfg, 110), HealthAction::Shutdown);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn thermal_monotonic(temp in 0u32..200) {
            let cfg = ThermalConfig {
                healthy_below_c: 80,
                throttle_at_c: 85,
                shutdown_at_c: 95,
            };
            let action = evaluate_thermal(&cfg, temp);
            match action {
                HealthAction::Shutdown => prop_assert!(temp >= 95),
                HealthAction::Throttle => prop_assert!((85..95).contains(&temp)),
                HealthAction::Healthy => prop_assert!(temp < 85),
                HealthAction::RestartWorker => prop_assert!(false, "unexpected RestartWorker"),
            }
        }
    }
}
