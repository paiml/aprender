//! Server deployment efficiency metrics.

use serde::{Deserialize, Serialize};

/// Server deployment efficiency metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerEfficiency {
    /// Training/inference throughput in samples per second
    pub throughput_samples_per_sec: f64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
}

impl ServerEfficiency {
    /// Create new server efficiency metrics
    pub fn new(
        throughput_samples_per_sec: f64,
        gpu_utilization_percent: f64,
        memory_bandwidth_gbps: f64,
    ) -> Self {
        Self {
            throughput_samples_per_sec,
            gpu_utilization_percent: gpu_utilization_percent.clamp(0.0, 100.0),
            memory_bandwidth_gbps,
        }
    }

    /// Check if GPU is being efficiently utilized (>70%)
    pub fn is_gpu_efficient(&self) -> bool {
        self.gpu_utilization_percent >= 70.0
    }

    /// Check if GPU is underutilized (<50%)
    pub fn is_gpu_underutilized(&self) -> bool {
        self.gpu_utilization_percent < 50.0
    }

    /// Calculate throughput efficiency (samples per second per % GPU util)
    pub fn throughput_efficiency(&self) -> f64 {
        if self.gpu_utilization_percent > 0.0 {
            self.throughput_samples_per_sec / self.gpu_utilization_percent
        } else {
            0.0
        }
    }

    /// Estimate maximum throughput at 100% utilization
    pub fn estimated_max_throughput(&self) -> f64 {
        if self.gpu_utilization_percent > 0.0 {
            self.throughput_samples_per_sec * (100.0 / self.gpu_utilization_percent)
        } else {
            0.0
        }
    }

    /// Check if memory bandwidth might be a bottleneck
    pub fn memory_bound(&self, expected_bandwidth_gbps: f64) -> bool {
        self.memory_bandwidth_gbps > expected_bandwidth_gbps * 0.9
    }
}

impl Default for ServerEfficiency {
    fn default() -> Self {
        Self {
            throughput_samples_per_sec: 0.0,
            gpu_utilization_percent: 0.0,
            memory_bandwidth_gbps: 0.0,
        }
    }
}
