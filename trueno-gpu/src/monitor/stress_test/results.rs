//! Stress test results, metrics, and reporting.
//!
//! Contains [`StressMetrics`], [`StressTestReport`], [`StressTestVerdict`],
//! and [`StressTestState`].

use std::time::{Duration, Instant};

use super::config::StressTestConfig;

// ============================================================================
// Stress Test Metrics
// ============================================================================

/// Stress test metrics collected during execution
#[derive(Debug, Clone, Default)]
pub struct StressMetrics {
    // Peak values
    /// Peak CPU utilization
    pub peak_cpu_utilization: f64,
    /// Peak GPU utilization
    pub peak_gpu_utilization: f64,
    /// Peak memory utilization
    pub peak_memory_utilization: f64,
    /// Peak temperature in Celsius
    pub peak_temperature_c: f64,
    /// Peak power in Watts
    pub peak_power_watts: f64,
    /// Peak PCIe bandwidth in GB/s
    pub peak_pcie_bandwidth_gbps: f64,

    // Throttling events
    /// Number of thermal throttle events
    pub thermal_throttle_count: u32,
    /// Number of power throttle events
    pub power_throttle_count: u32,
    /// Number of memory pressure events
    pub memory_pressure_events: u32,

    // Errors
    /// GPU errors
    pub gpu_errors: Vec<String>,
    /// Memory errors
    pub memory_errors: Vec<String>,
    /// Transfer errors
    pub transfer_errors: Vec<String>,

    // Performance
    /// Baseline FLOPS (before stress)
    pub baseline_flops: f64,
    /// Achieved FLOPS (during stress)
    pub achieved_flops: f64,
    /// Performance degradation percentage
    pub performance_degradation_pct: f64,

    // Timing
    /// Actual duration
    pub duration_actual: Duration,
    /// Number of samples collected
    pub sample_count: u32,
}

impl StressMetrics {
    /// Create new metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update peak values from current sample
    pub fn update_peaks(
        &mut self,
        cpu_util: f64,
        gpu_util: f64,
        mem_util: f64,
        temp_c: f64,
        power_w: f64,
        pcie_gbps: f64,
    ) {
        self.peak_cpu_utilization = self.peak_cpu_utilization.max(cpu_util);
        self.peak_gpu_utilization = self.peak_gpu_utilization.max(gpu_util);
        self.peak_memory_utilization = self.peak_memory_utilization.max(mem_util);
        self.peak_temperature_c = self.peak_temperature_c.max(temp_c);
        self.peak_power_watts = self.peak_power_watts.max(power_w);
        self.peak_pcie_bandwidth_gbps = self.peak_pcie_bandwidth_gbps.max(pcie_gbps);
        self.sample_count += 1;
    }

    /// Record a thermal throttle event
    pub fn record_thermal_throttle(&mut self) {
        self.thermal_throttle_count += 1;
    }

    /// Record a power throttle event
    pub fn record_power_throttle(&mut self) {
        self.power_throttle_count += 1;
    }

    /// Record a memory pressure event
    pub fn record_memory_pressure(&mut self) {
        self.memory_pressure_events += 1;
    }

    /// Add a GPU error
    pub fn add_gpu_error(&mut self, error: impl Into<String>) {
        self.gpu_errors.push(error.into());
    }

    /// Add a memory error
    pub fn add_memory_error(&mut self, error: impl Into<String>) {
        self.memory_errors.push(error.into());
    }

    /// Add a transfer error
    pub fn add_transfer_error(&mut self, error: impl Into<String>) {
        self.transfer_errors.push(error.into());
    }

    /// Calculate performance degradation
    pub fn calculate_degradation(&mut self) {
        if self.baseline_flops > 0.0 {
            let diff = self.baseline_flops - self.achieved_flops;
            self.performance_degradation_pct = (diff / self.baseline_flops) * 100.0;
        }
    }

    /// Check if there were any errors
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.gpu_errors.is_empty()
            || !self.memory_errors.is_empty()
            || !self.transfer_errors.is_empty()
    }

    /// Get total error count
    #[must_use]
    pub fn total_errors(&self) -> usize {
        self.gpu_errors.len() + self.memory_errors.len() + self.transfer_errors.len()
    }
}

// ============================================================================
// Stress Test Report
// ============================================================================

/// Stress test report
#[derive(Debug, Clone)]
pub struct StressTestReport {
    /// Test configuration
    pub config: StressTestConfig,
    /// Collected metrics
    pub metrics: StressMetrics,
    /// Actual test duration
    pub duration_actual: Duration,
    /// Overall verdict
    pub verdict: StressTestVerdict,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Report timestamp
    pub timestamp: Instant,
}

impl StressTestReport {
    /// Create a new report
    #[must_use]
    pub fn new(config: StressTestConfig, metrics: StressMetrics, duration: Duration) -> Self {
        let verdict = Self::calculate_verdict(&metrics);
        let recommendations = Self::generate_recommendations(&metrics, verdict);

        Self {
            config,
            metrics,
            duration_actual: duration,
            verdict,
            recommendations,
            timestamp: Instant::now(),
        }
    }

    fn calculate_verdict(metrics: &StressMetrics) -> StressTestVerdict {
        // Critical failures
        if metrics.has_errors() {
            return StressTestVerdict::Fail;
        }

        // Check for severe issues
        if metrics.thermal_throttle_count > 10 {
            return StressTestVerdict::Fail;
        }
        if metrics.peak_temperature_c > 95.0 {
            return StressTestVerdict::Fail;
        }
        if metrics.performance_degradation_pct > 50.0 {
            return StressTestVerdict::Fail;
        }

        // Minor issues
        if metrics.thermal_throttle_count > 0
            || metrics.power_throttle_count > 0
            || metrics.memory_pressure_events > 0
        {
            return StressTestVerdict::PassWithNotes;
        }

        StressTestVerdict::Pass
    }

    fn generate_recommendations(
        metrics: &StressMetrics,
        verdict: StressTestVerdict,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if metrics.peak_temperature_c > 85.0 {
            recs.push("Consider improving cooling - peak temperature exceeded 85°C".to_string());
        }

        if metrics.thermal_throttle_count > 0 {
            recs.push(format!(
                "Thermal throttling detected {} times - reduce workload or improve cooling",
                metrics.thermal_throttle_count
            ));
        }

        if metrics.power_throttle_count > 0 {
            recs.push(format!(
                "Power throttling detected {} times - check power supply capacity",
                metrics.power_throttle_count
            ));
        }

        if metrics.memory_pressure_events > 0 {
            recs.push(format!(
                "Memory pressure detected {} times - consider reducing parallel jobs",
                metrics.memory_pressure_events
            ));
        }

        if metrics.performance_degradation_pct > 10.0 {
            recs.push(format!(
                "Performance degraded by {:.1}% under load - investigate bottlenecks",
                metrics.performance_degradation_pct
            ));
        }

        if verdict == StressTestVerdict::Pass && recs.is_empty() {
            recs.push("System passed all stress tests - no issues detected".to_string());
        }

        recs
    }

    /// Export report to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        // Simplified JSON format
        format!(
            r#"{{
  "verdict": "{}",
  "duration_seconds": {:.1},
  "peak_cpu_pct": {:.1},
  "peak_gpu_pct": {:.1},
  "peak_memory_pct": {:.1},
  "peak_temp_c": {:.1},
  "peak_power_w": {:.1},
  "thermal_throttles": {},
  "power_throttles": {},
  "memory_pressure_events": {},
  "total_errors": {},
  "performance_degradation_pct": {:.1},
  "recommendations": {:?}
}}"#,
            self.verdict,
            self.duration_actual.as_secs_f64(),
            self.metrics.peak_cpu_utilization,
            self.metrics.peak_gpu_utilization,
            self.metrics.peak_memory_utilization,
            self.metrics.peak_temperature_c,
            self.metrics.peak_power_watts,
            self.metrics.thermal_throttle_count,
            self.metrics.power_throttle_count,
            self.metrics.memory_pressure_events,
            self.metrics.total_errors(),
            self.metrics.performance_degradation_pct,
            self.recommendations
        )
    }
}

/// Stress test verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressTestVerdict {
    /// All tests passed
    Pass,
    /// Passed with minor notes
    PassWithNotes,
    /// Failed
    Fail,
}

impl std::fmt::Display for StressTestVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::PassWithNotes => write!(f, "PASS_WITH_NOTES"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

// ============================================================================
// Stress Test Runner State
// ============================================================================

/// Stress test runner state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressTestState {
    /// Not started
    Idle,
    /// Ramping up intensity
    RampUp,
    /// Running at full intensity
    Running,
    /// Cooling down
    CoolDown,
    /// Completed
    Completed,
    /// Failed/aborted
    Aborted,
}

impl std::fmt::Display for StressTestState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::RampUp => write!(f, "Ramp-Up"),
            Self::Running => write!(f, "Running"),
            Self::CoolDown => write!(f, "Cool-Down"),
            Self::Completed => write!(f, "Completed"),
            Self::Aborted => write!(f, "Aborted"),
        }
    }
}
