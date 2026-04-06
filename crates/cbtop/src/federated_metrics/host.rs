//! Host state, aggregated metrics, and configuration for federated metrics.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::crdt::GCounter;

/// Host state in the federation
#[derive(Debug, Clone)]
pub struct FederatedHost {
    /// Host identifier
    pub host_id: String,
    /// Last seen timestamp
    pub last_seen: Instant,
    /// Sample count from this host
    pub sample_count: GCounter,
    /// Health status (0.0 = dead, 1.0 = healthy)
    pub health: f64,
    /// Current sampling rate (samples per second)
    pub sampling_rate: f64,
    /// Network latency to this host (milliseconds)
    pub latency_ms: f64,
    /// Logical clock value
    pub logical_clock: u64,
}

impl FederatedHost {
    /// Create a new federated host
    pub fn new(host_id: impl Into<String>) -> Self {
        Self {
            host_id: host_id.into(),
            last_seen: Instant::now(),
            sample_count: GCounter::new(),
            health: 1.0,
            sampling_rate: 100.0, // Default 100 Hz
            latency_ms: 0.0,
            logical_clock: 0,
        }
    }

    /// Update last seen time
    pub fn touch(&mut self) {
        self.last_seen = Instant::now();
    }

    /// Check if host is stale (not seen recently)
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_seen.elapsed() > timeout
    }

    /// Increment logical clock
    pub fn tick(&mut self) -> u64 {
        self.logical_clock += 1;
        self.logical_clock
    }

    /// Update logical clock from received message
    pub fn sync_clock(&mut self, received_time: u64) {
        self.logical_clock = self.logical_clock.max(received_time) + 1;
    }
}

/// Aggregated metrics across federation
#[derive(Debug, Clone, Default)]
pub struct AggregatedMetrics {
    /// Metric name
    pub metric_name: String,
    /// All sample values
    pub values: Vec<f64>,
    /// Per-host sample counts
    pub host_counts: HashMap<String, usize>,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Sum for mean calculation
    pub sum: f64,
}

impl AggregatedMetrics {
    /// Create new aggregated metrics
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            metric_name: metric_name.into(),
            values: Vec::new(),
            host_counts: HashMap::new(),
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
        }
    }

    /// Add a sample
    pub fn add_sample(&mut self, host_id: &str, value: f64) {
        self.values.push(value);
        *self.host_counts.entry(host_id.to_string()).or_insert(0) += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
    }

    /// Get mean value
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get percentile value
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get p50
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Get p95
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// Get p99
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// Detect skewed host (significantly slower)
    pub fn detect_skew(&self, threshold_percent: f64) -> Vec<String> {
        let mean = self.mean();
        if mean == 0.0 {
            return Vec::new();
        }

        let mut skewed = Vec::new();
        for (host_id, count) in &self.host_counts {
            // Calculate host-specific mean (simplified: use count as proxy)
            let expected_count = self.values.len() / self.host_counts.len().max(1);
            let deviation =
                ((*count as f64 - expected_count as f64) / expected_count as f64).abs() * 100.0;

            if deviation > threshold_percent {
                skewed.push(host_id.clone());
            }
        }
        skewed
    }
}

/// Configuration for federation
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Maximum clock drift tolerance (milliseconds)
    pub max_clock_drift_ms: i64,
    /// Host timeout before considered stale
    pub host_timeout: Duration,
    /// Memory limit per federation (bytes)
    pub memory_limit_bytes: usize,
    /// Default sampling rate (Hz)
    pub default_sampling_rate: f64,
    /// Skew detection threshold (percent)
    pub skew_threshold_percent: f64,
    /// Partition recovery timeout
    pub partition_recovery_timeout: Duration,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            max_clock_drift_ms: 100,
            host_timeout: Duration::from_secs(30),
            memory_limit_bytes: 100 * 1024 * 1024, // 100MB
            default_sampling_rate: 100.0,
            skew_threshold_percent: 40.0,
            partition_recovery_timeout: Duration::from_secs(30),
        }
    }
}
