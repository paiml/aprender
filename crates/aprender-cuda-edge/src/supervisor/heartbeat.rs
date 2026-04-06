//! GPU health monitoring via heartbeat protocol.
//!
//! [`GpuHealthMonitor`] tracks heartbeat misses and temperature thresholds
//! to determine the appropriate [`HealthAction`] for each worker.

use serde::{Deserialize, Serialize};

use super::strategy::HealthAction;
use super::worker::HeartbeatStatus;

/// Monitors GPU worker health and decides on corrective actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHealthMonitor {
    /// Maximum heartbeat misses before restarting the worker.
    pub max_missed: u32,
    /// Temperature (°C) above which the worker should be throttled.
    pub throttle_temp_c: u32,
    /// Temperature (°C) above which the worker should be shut down.
    pub shutdown_temp_c: u32,
}

impl GpuHealthMonitor {
    /// Create a new health monitor with the specified thresholds.
    #[must_use]
    pub fn new(max_missed: u32, throttle_temp_c: u32, shutdown_temp_c: u32) -> Self {
        Self {
            max_missed,
            throttle_temp_c,
            shutdown_temp_c,
        }
    }

    /// Create a builder for configuring a health monitor.
    #[must_use]
    pub fn builder() -> GpuHealthMonitorBuilder {
        GpuHealthMonitorBuilder::default()
    }

    /// Evaluate a heartbeat status and return the appropriate action.
    #[must_use]
    pub fn check_status(&self, status: HeartbeatStatus) -> HealthAction {
        match status {
            HeartbeatStatus::MissedBeats(n) if n >= self.max_missed => HealthAction::RestartWorker,
            HeartbeatStatus::Dead => HealthAction::Shutdown,
            // Alive or MissedBeats below threshold — still healthy
            HeartbeatStatus::Alive | HeartbeatStatus::MissedBeats(_) => HealthAction::Healthy,
        }
    }

    /// Evaluate a temperature reading and return the appropriate action.
    #[must_use]
    pub fn check_temperature(&self, temp_c: u32) -> HealthAction {
        if temp_c >= self.shutdown_temp_c {
            HealthAction::Shutdown
        } else if temp_c >= self.throttle_temp_c {
            HealthAction::Throttle
        } else {
            HealthAction::Healthy
        }
    }
}

impl Default for GpuHealthMonitor {
    fn default() -> Self {
        Self {
            max_missed: 3,
            throttle_temp_c: 85,
            shutdown_temp_c: 95,
        }
    }
}

/// Builder for [`GpuHealthMonitor`].
#[derive(Debug, Default)]
pub struct GpuHealthMonitorBuilder {
    max_missed: Option<u32>,
    throttle_temp_c: Option<u32>,
    shutdown_temp_c: Option<u32>,
}

impl GpuHealthMonitorBuilder {
    /// Set the maximum missed heartbeats before restart.
    #[must_use]
    pub fn max_missed(mut self, n: u32) -> Self {
        self.max_missed = Some(n);
        self
    }

    /// Set the throttle temperature threshold in °C.
    #[must_use]
    pub fn throttle_temp(mut self, temp_c: u32) -> Self {
        self.throttle_temp_c = Some(temp_c);
        self
    }

    /// Set the shutdown temperature threshold in °C.
    #[must_use]
    pub fn shutdown_temp(mut self, temp_c: u32) -> Self {
        self.shutdown_temp_c = Some(temp_c);
        self
    }

    /// Build the health monitor.
    #[must_use]
    pub fn build(self) -> GpuHealthMonitor {
        let defaults = GpuHealthMonitor::default();
        GpuHealthMonitor {
            max_missed: self.max_missed.unwrap_or(defaults.max_missed),
            throttle_temp_c: self.throttle_temp_c.unwrap_or(defaults.throttle_temp_c),
            shutdown_temp_c: self.shutdown_temp_c.unwrap_or(defaults.shutdown_temp_c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alive_is_healthy() {
        let monitor = GpuHealthMonitor::default();
        assert_eq!(
            monitor.check_status(HeartbeatStatus::Alive),
            HealthAction::Healthy
        );
    }

    #[test]
    fn missed_below_threshold_is_healthy() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);
        assert_eq!(
            monitor.check_status(HeartbeatStatus::MissedBeats(2)),
            HealthAction::Healthy
        );
    }

    #[test]
    fn missed_at_threshold_triggers_restart() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);
        assert_eq!(
            monitor.check_status(HeartbeatStatus::MissedBeats(3)),
            HealthAction::RestartWorker
        );
    }

    #[test]
    fn missed_above_threshold_triggers_restart() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);
        assert_eq!(
            monitor.check_status(HeartbeatStatus::MissedBeats(10)),
            HealthAction::RestartWorker
        );
    }

    #[test]
    fn dead_triggers_shutdown() {
        let monitor = GpuHealthMonitor::default();
        assert_eq!(
            monitor.check_status(HeartbeatStatus::Dead),
            HealthAction::Shutdown
        );
    }

    #[test]
    fn temperature_below_throttle_is_healthy() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);
        assert_eq!(monitor.check_temperature(70), HealthAction::Healthy);
    }

    #[test]
    fn temperature_at_throttle_triggers_throttle() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);
        assert_eq!(monitor.check_temperature(85), HealthAction::Throttle);
    }

    #[test]
    fn temperature_at_shutdown_triggers_shutdown() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);
        assert_eq!(monitor.check_temperature(95), HealthAction::Shutdown);
    }

    #[test]
    fn builder_applies_custom_values() {
        let monitor = GpuHealthMonitor::builder()
            .max_missed(5)
            .throttle_temp(80)
            .shutdown_temp(90)
            .build();
        assert_eq!(monitor.max_missed, 5);
        assert_eq!(monitor.throttle_temp_c, 80);
        assert_eq!(monitor.shutdown_temp_c, 90);
    }

    #[test]
    fn builder_uses_defaults_for_unset() {
        let monitor = GpuHealthMonitor::builder().max_missed(10).build();
        assert_eq!(monitor.max_missed, 10);
        assert_eq!(monitor.throttle_temp_c, 85); // default
        assert_eq!(monitor.shutdown_temp_c, 95); // default
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn missed_beats_monotonic(n in 0u32..100, threshold in 1u32..50) {
            let monitor = GpuHealthMonitor::new(threshold, 85, 95);
            let action = monitor.check_status(HeartbeatStatus::MissedBeats(n));
            if n >= threshold {
                prop_assert_eq!(action, HealthAction::RestartWorker);
            } else {
                prop_assert_eq!(action, HealthAction::Healthy);
            }
        }
    }
}
