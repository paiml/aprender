//! GPU Andon system for alert management.

use super::thresholds::AndonThresholds;
use super::types::GpuAlert;
use crate::monitor::gpu::GpuMetrics;

/// GPU Andon system for alert management
#[derive(Debug)]
pub struct GpuAndonSystem {
    /// Alert thresholds
    thresholds: AndonThresholds,
    /// Active alerts
    alerts: Vec<GpuAlert>,
    /// Idle tracking per device (device_id -> consecutive idle samples)
    idle_samples: Vec<u32>,
    /// Sample interval in seconds (for idle calculation)
    sample_interval_secs: u32,
}

impl GpuAndonSystem {
    /// Create a new Andon system with default thresholds
    pub fn new() -> Self {
        Self::with_thresholds(AndonThresholds::default())
    }

    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: AndonThresholds) -> Self {
        Self { thresholds, alerts: Vec::new(), idle_samples: Vec::new(), sample_interval_secs: 1 }
    }

    /// Set sample interval for idle calculation
    pub fn set_sample_interval(&mut self, secs: u32) {
        self.sample_interval_secs = secs;
    }

    /// Check metrics and generate alerts
    pub fn check(&mut self, metrics: &[GpuMetrics]) -> Vec<GpuAlert> {
        self.alerts.clear();

        // Ensure idle_samples is sized correctly
        if self.idle_samples.len() < metrics.len() {
            self.idle_samples.resize(metrics.len(), 0);
        }

        for m in metrics {
            // Thermal check
            if m.temperature_celsius >= self.thresholds.thermal_warning {
                self.alerts.push(GpuAlert::ThermalThrottling {
                    device: m.device_id,
                    temp: m.temperature_celsius,
                    threshold: self.thresholds.thermal_warning,
                });
            }

            // Memory pressure check
            let mem_percent = m.memory_percent() as u32;
            if mem_percent >= self.thresholds.memory_warning {
                self.alerts.push(GpuAlert::MemoryPressure {
                    device: m.device_id,
                    used_percent: mem_percent,
                    threshold: self.thresholds.memory_warning,
                });
            }

            // Power limit check
            let power_percent = m.power_percent() as u32;
            if power_percent >= self.thresholds.power_warning {
                self.alerts.push(GpuAlert::PowerLimit {
                    device: m.device_id,
                    power_percent,
                    threshold: self.thresholds.power_warning,
                });
            }

            // Idle check
            let device_idx = m.device_id as usize;
            if device_idx < self.idle_samples.len() {
                if m.utilization_percent == 0 {
                    self.idle_samples[device_idx] += 1;
                    let idle_secs = self.idle_samples[device_idx] * self.sample_interval_secs;
                    if idle_secs >= self.thresholds.idle_threshold_secs {
                        self.alerts.push(GpuAlert::GpuIdle {
                            device: m.device_id,
                            duration_secs: idle_secs,
                        });
                    }
                } else {
                    self.idle_samples[device_idx] = 0;
                }
            }
        }

        self.alerts.clone()
    }

    /// Get current alerts
    pub fn alerts(&self) -> &[GpuAlert] {
        &self.alerts
    }

    /// Get thresholds
    pub fn thresholds(&self) -> &AndonThresholds {
        &self.thresholds
    }

    /// Check if any critical alerts are active
    pub fn has_critical_alerts(&self) -> bool {
        self.alerts.iter().any(|a| a.severity() >= 80)
    }
}

impl Default for GpuAndonSystem {
    fn default() -> Self {
        Self::new()
    }
}
