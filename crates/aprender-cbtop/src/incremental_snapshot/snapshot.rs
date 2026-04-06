//! Profile and delta snapshot types.

use std::collections::HashMap;

use super::types::{DeltaMetric, MetricData, RetentionTier};

/// A profile snapshot
#[derive(Debug, Clone)]
pub struct ProfileSnapshot {
    /// Snapshot index
    pub index: usize,
    /// Timestamp when snapshot was taken
    pub timestamp_ns: u64,
    /// Workload fingerprint for identification
    pub workload_fingerprint: String,
    /// Metric data
    pub metrics: HashMap<String, MetricData>,
    /// Snapshot metadata
    pub metadata: HashMap<String, String>,
    /// CRC32 checksum
    pub checksum: u32,
}

impl ProfileSnapshot {
    /// Create a new snapshot
    pub fn new(index: usize) -> Self {
        Self {
            index,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            workload_fingerprint: String::new(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            checksum: 0,
        }
    }

    /// Add metric data
    pub fn add_metric(&mut self, metric: MetricData) {
        self.metrics.insert(metric.name.clone(), metric);
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<&MetricData> {
        self.metrics.get(name)
    }

    /// Set workload fingerprint
    pub fn set_fingerprint(&mut self, fingerprint: impl Into<String>) {
        self.workload_fingerprint = fingerprint.into();
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        let metrics_size: usize = self.metrics.values().map(|m| m.size_bytes()).sum();
        let metadata_size: usize = self.metadata.iter().map(|(k, v)| k.len() + v.len()).sum();

        24 + self.workload_fingerprint.len() + metrics_size + metadata_size
    }

    /// Compute checksum
    pub fn compute_checksum(&mut self) {
        // Simple CRC32-like checksum
        let mut hash: u32 = 0;

        hash = hash.wrapping_add(self.index as u32);
        hash = hash.wrapping_mul(31);
        hash = hash.wrapping_add((self.timestamp_ns & 0xFFFFFFFF) as u32);

        // Sort keys for deterministic iteration
        let mut keys: Vec<_> = self.metrics.keys().collect();
        keys.sort();

        for name in keys {
            if let Some(metric) = self.metrics.get(name) {
                for c in name.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(c as u32);
                }
                for &val in &metric.values {
                    // Include both halves of f64 bits
                    let bits = val.to_bits();
                    hash = hash.wrapping_mul(31).wrapping_add((bits >> 32) as u32);
                    hash = hash.wrapping_mul(31).wrapping_add((bits & 0xFFFFFFFF) as u32);
                }
            }
        }

        self.checksum = hash;
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        let mut copy = self.clone();
        copy.compute_checksum();
        copy.checksum == self.checksum
    }
}

/// Delta-encoded snapshot
#[derive(Debug, Clone)]
pub struct DeltaSnapshot {
    /// Snapshot index
    pub index: usize,
    /// Base snapshot index
    pub base_index: usize,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Workload fingerprint
    pub workload_fingerprint: String,
    /// Delta metrics
    pub deltas: HashMap<String, DeltaMetric>,
    /// New metrics (not in base)
    pub new_metrics: HashMap<String, MetricData>,
    /// Removed metric names
    pub removed_metrics: Vec<String>,
    /// Checksum
    pub checksum: u32,
}

impl DeltaSnapshot {
    /// Create from two snapshots
    pub fn from_diff(base: &ProfileSnapshot, current: &ProfileSnapshot) -> Self {
        let mut deltas = HashMap::new();
        let mut new_metrics = HashMap::new();
        let mut removed_metrics = Vec::new();

        // Find deltas and new metrics
        for (name, metric) in &current.metrics {
            if let Some(base_metric) = base.metrics.get(name) {
                let delta = metric.delta_from(base_metric);
                // Only store delta if it's smaller than full metric
                if delta.size_bytes() < metric.size_bytes() {
                    deltas.insert(name.clone(), delta);
                } else {
                    new_metrics.insert(name.clone(), metric.clone());
                }
            } else {
                new_metrics.insert(name.clone(), metric.clone());
            }
        }

        // Find removed metrics
        for name in base.metrics.keys() {
            if !current.metrics.contains_key(name) {
                removed_metrics.push(name.clone());
            }
        }

        Self {
            index: current.index,
            base_index: base.index,
            timestamp_ns: current.timestamp_ns,
            workload_fingerprint: current.workload_fingerprint.clone(),
            deltas,
            new_metrics,
            removed_metrics,
            checksum: 0,
        }
    }

    /// Get compressed size
    pub fn size_bytes(&self) -> usize {
        let delta_size: usize = self.deltas.values().map(|d| d.size_bytes()).sum();
        let new_size: usize = self.new_metrics.values().map(|m| m.size_bytes()).sum();
        let removed_size: usize = self.removed_metrics.iter().map(|s| s.len()).sum();

        24 + self.workload_fingerprint.len() + delta_size + new_size + removed_size
    }

    /// Apply delta to base snapshot
    pub fn apply_to(&self, base: &ProfileSnapshot) -> ProfileSnapshot {
        let mut result = ProfileSnapshot::new(self.index);
        result.timestamp_ns = self.timestamp_ns;
        result.workload_fingerprint = self.workload_fingerprint.clone();

        // Copy base metrics and apply deltas
        for (name, metric) in &base.metrics {
            if self.removed_metrics.contains(name) {
                continue;
            }

            if let Some(delta) = self.deltas.get(name) {
                let reconstructed = metric.apply_delta(delta);
                result.metrics.insert(name.clone(), reconstructed);
            } else {
                result.metrics.insert(name.clone(), metric.clone());
            }
        }

        // Add new metrics
        for (name, metric) in &self.new_metrics {
            result.metrics.insert(name.clone(), metric.clone());
        }

        result.compute_checksum();
        result
    }
}

/// Index entry for fast lookup
#[derive(Debug, Clone)]
pub struct SnapshotIndex {
    /// Snapshot index
    pub index: usize,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Workload fingerprint
    pub fingerprint: String,
    /// Retention tier
    pub tier: RetentionTier,
    /// Offset in storage
    pub offset: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Is delta encoded
    pub is_delta: bool,
    /// Base index (if delta)
    pub base_index: Option<usize>,
}
