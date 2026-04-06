//! Metric and resource snapshot types for dashboard display.

use serde::{Deserialize, Serialize};

use super::Trend;
use crate::storage::MetricPoint;

/// A snapshot of metric values for dashboard display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Metric key (e.g., "loss", "accuracy")
    pub key: String,
    /// Time-value pairs: (timestamp_ms, value)
    pub values: Vec<(u64, f64)>,
    /// Current trend direction
    pub trend: Trend,
}

impl MetricSnapshot {
    /// Create a new metric snapshot.
    pub fn new(key: impl Into<String>, values: Vec<(u64, f64)>) -> Self {
        let trend = Trend::from_values(&values.iter().map(|(_, v)| *v).collect::<Vec<_>>());
        Self { key: key.into(), values, trend }
    }

    /// Create from metric points.
    pub fn from_points(key: impl Into<String>, points: &[MetricPoint]) -> Self {
        let values: Vec<(u64, f64)> = points
            .iter()
            .map(|p| {
                let ts = p.timestamp.timestamp_millis() as u64;
                (ts, p.value)
            })
            .collect();
        Self::new(key, values)
    }

    /// Get the latest value.
    pub fn latest(&self) -> Option<f64> {
        self.values.last().map(|(_, v)| *v)
    }

    /// Get the minimum value.
    pub fn min(&self) -> Option<f64> {
        self.values.iter().map(|(_, v)| *v).reduce(f64::min)
    }

    /// Get the maximum value.
    pub fn max(&self) -> Option<f64> {
        self.values.iter().map(|(_, v)| *v).reduce(f64::max)
    }

    /// Get the mean value.
    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().map(|(_, v)| *v).sum::<f64>() / self.values.len() as f64)
    }

    /// Check if metric is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get number of data points.
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Resource usage snapshot for dashboard display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// GPU utilization (0.0 to 1.0)
    pub gpu_util: f64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_util: f64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// Total memory in bytes
    pub memory_total: u64,
    /// GPU memory used in bytes (optional)
    pub gpu_memory_used: Option<u64>,
    /// Total GPU memory in bytes (optional)
    pub gpu_memory_total: Option<u64>,
}

impl Default for ResourceSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSnapshot {
    /// Create a new resource snapshot with zero values.
    pub fn new() -> Self {
        Self {
            gpu_util: 0.0,
            cpu_util: 0.0,
            memory_used: 0,
            memory_total: 0,
            gpu_memory_used: None,
            gpu_memory_total: None,
        }
    }

    /// Set GPU utilization.
    pub fn with_gpu_util(mut self, util: f64) -> Self {
        self.gpu_util = util.clamp(0.0, 1.0);
        self
    }

    /// Set CPU utilization.
    pub fn with_cpu_util(mut self, util: f64) -> Self {
        self.cpu_util = util.clamp(0.0, 1.0);
        self
    }

    /// Set memory usage.
    pub fn with_memory(mut self, used: u64, total: u64) -> Self {
        self.memory_used = used;
        self.memory_total = total;
        self
    }

    /// Set GPU memory usage.
    pub fn with_gpu_memory(mut self, used: u64, total: u64) -> Self {
        self.gpu_memory_used = Some(used);
        self.gpu_memory_total = Some(total);
        self
    }

    /// Get memory utilization as a fraction.
    pub fn memory_util(&self) -> f64 {
        if self.memory_total == 0 {
            return 0.0;
        }
        self.memory_used as f64 / self.memory_total as f64
    }

    /// Get GPU memory utilization as a fraction.
    pub fn gpu_memory_util(&self) -> Option<f64> {
        match (self.gpu_memory_used, self.gpu_memory_total) {
            (Some(used), Some(total)) if total > 0 => Some(used as f64 / total as f64),
            _ => None,
        }
    }
}
