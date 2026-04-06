//! Federated Metrics Aggregation (PMAT-048)
//!
//! Multi-host metrics aggregation with CRDT-based merging for distributed profiling.
//!
//! # Design
//!
//! - CRDT (Conflict-free Replicated Data Types) for partition-tolerant merging
//! - Adaptive sampling based on network bandwidth
//! - Automatic topology detection and health-based routing
//! - Skew detection for identifying degraded nodes
//!
//! # Falsification (FKR-049)
//!
//! H₀: Federated aggregation cannot maintain correctness across network partitions
//! Test: Simulate 3-host cluster with partition, verify CRDT convergence

mod crdt;
mod host;
mod types;

pub use crdt::{GCounter, LwwRegister, OrSet};
pub use host::{AggregatedMetrics, FederatedHost, FederationConfig};
pub use types::{FederatedError, FederatedResult, MetricSample, SampleId};

use std::collections::{HashMap, HashSet};

/// Federated metrics aggregator
#[derive(Debug)]
pub struct MetricsFederation {
    /// Federation configuration
    config: FederationConfig,
    /// Local host ID
    local_host_id: String,
    /// Known hosts in federation
    hosts: HashMap<String, FederatedHost>,
    /// Active hosts (OR-Set for partition tolerance)
    active_hosts: OrSet<String>,
    /// Collected samples (keyed by sample ID to prevent duplicates)
    samples: HashMap<SampleId, MetricSample>,
    /// Total sample counter
    total_samples: GCounter,
    /// Aggregated metrics by name
    aggregated: HashMap<String, AggregatedMetrics>,
    /// Current memory usage estimate
    memory_usage: usize,
    /// Logical clock for this node
    logical_clock: u64,
    /// Sample sequence counter
    sequence: u64,
}

impl MetricsFederation {
    /// Create a new federation
    pub fn new(local_host_id: impl Into<String>, config: FederationConfig) -> Self {
        let host_id = local_host_id.into();
        let mut hosts = HashMap::new();
        hosts.insert(host_id.clone(), FederatedHost::new(&host_id));

        let mut active_hosts = OrSet::new();
        active_hosts.add(host_id.clone(), format!("{}-0", host_id));

        Self {
            config,
            local_host_id: host_id,
            hosts,
            active_hosts,
            samples: HashMap::new(),
            total_samples: GCounter::new(),
            aggregated: HashMap::new(),
            memory_usage: 0,
            logical_clock: 0,
            sequence: 0,
        }
    }

    /// Add a host to the federation
    pub fn add_host(&mut self, host_id: impl Into<String>) {
        let id = host_id.into();
        if !self.hosts.contains_key(&id) {
            self.hosts.insert(id.clone(), FederatedHost::new(&id));
            let tick = self.tick();
            let local_id = self.local_host_id.clone();
            let tag = format!("{}-{}", local_id, tick);
            self.active_hosts.add(id, tag);
        }
    }

    /// Remove a host from the federation
    pub fn remove_host(&mut self, host_id: &str) {
        self.active_hosts.remove(&host_id.to_string());
    }

    /// Get active host count
    pub fn active_host_count(&self) -> usize {
        self.active_hosts.elements().len()
    }

    /// Increment logical clock
    fn tick(&mut self) -> u64 {
        self.logical_clock += 1;
        self.logical_clock
    }

    /// Record a local metric sample
    pub fn record(
        &mut self,
        metric_name: impl Into<String>,
        value: f64,
    ) -> FederatedResult<SampleId> {
        let time = self.tick();
        self.sequence += 1;

        let sample =
            MetricSample::new(&self.local_host_id, time, self.sequence, metric_name, value);

        self.add_sample(sample)
    }

    /// Add a sample (local or remote)
    pub fn add_sample(&mut self, sample: MetricSample) -> FederatedResult<SampleId> {
        // Check for duplicates (idempotent)
        if self.samples.contains_key(&sample.id) {
            return Ok(sample.id);
        }

        // Check memory limit
        let sample_size = std::mem::size_of::<MetricSample>() + sample.metric_name.len();
        if self.memory_usage + sample_size > self.config.memory_limit_bytes {
            return Err(FederatedError::MemoryLimitExceeded {
                used_bytes: self.memory_usage,
                limit_bytes: self.config.memory_limit_bytes,
            });
        }

        // Update host state
        if let Some(host) = self.hosts.get_mut(&sample.id.host_id) {
            host.touch();
            host.sample_count.increment(&sample.id.host_id, 1);
            host.sync_clock(sample.id.logical_time);
        }

        // Update aggregation
        let agg = self
            .aggregated
            .entry(sample.metric_name.clone())
            .or_insert_with(|| AggregatedMetrics::new(&sample.metric_name));
        agg.add_sample(&sample.id.host_id, sample.value);

        // Update totals
        self.total_samples.increment(&sample.id.host_id, 1);
        self.memory_usage += sample_size;

        let id = sample.id.clone();
        self.samples.insert(sample.id.clone(), sample);

        Ok(id)
    }

    /// Merge samples from another federation node
    pub fn merge(&mut self, other: &MetricsFederation) -> FederatedResult<usize> {
        let mut merged_count = 0;

        // Merge active hosts
        self.active_hosts.merge(&other.active_hosts);

        // Merge samples
        for (id, sample) in &other.samples {
            if !self.samples.contains_key(id) {
                self.add_sample(sample.clone())?;
                merged_count += 1;
            }
        }

        // Merge counters
        self.total_samples.merge(&other.total_samples);

        // Sync logical clock
        self.logical_clock = self.logical_clock.max(other.logical_clock) + 1;

        Ok(merged_count)
    }

    /// Get aggregated metrics for a metric name
    pub fn get_aggregated(&self, metric_name: &str) -> Option<&AggregatedMetrics> {
        self.aggregated.get(metric_name)
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<&String> {
        self.aggregated.keys().collect()
    }

    /// Detect hosts with performance skew
    pub fn detect_skewed_hosts(&self) -> Vec<String> {
        let mut skewed = HashSet::new();

        for agg in self.aggregated.values() {
            for host in agg.detect_skew(self.config.skew_threshold_percent) {
                skewed.insert(host);
            }
        }

        skewed.into_iter().collect()
    }

    /// Get hosts that are stale (not seen recently)
    pub fn get_stale_hosts(&self) -> Vec<String> {
        self.hosts
            .iter()
            .filter(|(_, host)| host.is_stale(self.config.host_timeout))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Update host health based on sample count
    pub fn update_health(&mut self) {
        let total = self.total_samples.value() as f64;
        if total == 0.0 {
            return;
        }

        let host_count = self.hosts.len() as f64;
        let expected_per_host = total / host_count;

        for (host_id, host) in &mut self.hosts {
            let count = self.total_samples.host_count(host_id) as f64;
            if expected_per_host > 0.0 {
                // Health is ratio of actual to expected samples
                host.health = (count / expected_per_host).min(1.0);
            }
        }
    }

    /// Adapt sampling rates based on network conditions
    pub fn adapt_sampling_rates(&mut self, bandwidth_mbps: f64) {
        // Lower sampling rate for higher latency hosts
        let base_rate = self.config.default_sampling_rate;

        for host in self.hosts.values_mut() {
            // Reduce rate proportionally to latency
            let latency_factor = 1.0 / (1.0 + host.latency_ms / 100.0);
            // Reduce rate based on bandwidth
            let bandwidth_factor = (bandwidth_mbps / 100.0).min(1.0);

            host.sampling_rate = base_rate * latency_factor * bandwidth_factor;
        }
    }

    /// Get total sample count
    pub fn total_samples(&self) -> u64 {
        self.total_samples.value()
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Get configuration
    pub fn config(&self) -> &FederationConfig {
        &self.config
    }

    /// Get host by ID
    pub fn get_host(&self, host_id: &str) -> Option<&FederatedHost> {
        self.hosts.get(host_id)
    }
}

/// Default maximum clock drift (milliseconds)
pub const DEFAULT_MAX_CLOCK_DRIFT_MS: i64 = 100;

/// Default memory limit (bytes)
pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 100 * 1024 * 1024;

/// Default skew threshold (percent)
pub const DEFAULT_SKEW_THRESHOLD_PERCENT: f64 = 40.0;

#[cfg(test)]
mod tests;
