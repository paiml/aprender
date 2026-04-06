//! Metric Record - time-series metrics for runs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Metric Record represents a single metric data point.
///
/// Designed for time-series storage, where metrics are ordered by step
/// and can be efficiently queried by `run_id` and key.
///
/// ## Time-Series Optimization
///
/// Metrics are stored with:
/// - `run_id` + `key` as the partition key for efficient filtering
/// - `step` as the sort key for time-series ordering
/// - `timestamp` for wall-clock time correlation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricRecord {
    run_id: String,
    key: String,
    step: u64,
    value: f64,
    timestamp: DateTime<Utc>,
}

impl MetricRecord {
    /// Create a new metric record.
    ///
    /// # Arguments
    ///
    /// * `run_id` - ID of the parent run
    /// * `key` - Metric name/key (e.g., "loss", "accuracy")
    /// * `step` - Training step or epoch number
    /// * `value` - Metric value
    ///
    /// # Returns
    ///
    /// A new `MetricRecord` with the current timestamp.
    #[must_use]
    pub fn new(run_id: impl Into<String>, key: impl Into<String>, step: u64, value: f64) -> Self {
        Self { run_id: run_id.into(), key: key.into(), step, value, timestamp: Utc::now() }
    }

    /// Create a builder for constructing a metric record with optional fields.
    #[must_use]
    pub fn builder(
        run_id: impl Into<String>,
        key: impl Into<String>,
        step: u64,
        value: f64,
    ) -> MetricRecordBuilder {
        MetricRecordBuilder::new(run_id, key, step, value)
    }

    /// Get the run ID.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get the metric key/name.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Get the step/epoch number.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Get the metric value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Get the timestamp when the metric was recorded.
    #[must_use]
    pub const fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

/// Builder for `MetricRecord`.
#[derive(Debug)]
pub struct MetricRecordBuilder {
    run_id: String,
    key: String,
    step: u64,
    value: f64,
    timestamp: DateTime<Utc>,
}

impl MetricRecordBuilder {
    /// Create a new builder with required fields.
    #[must_use]
    pub fn new(run_id: impl Into<String>, key: impl Into<String>, step: u64, value: f64) -> Self {
        Self { run_id: run_id.into(), key: key.into(), step, value, timestamp: Utc::now() }
    }

    /// Set a custom timestamp.
    #[must_use]
    pub const fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Build the `MetricRecord`.
    #[must_use]
    pub fn build(self) -> MetricRecord {
        MetricRecord {
            run_id: self.run_id,
            key: self.key,
            step: self.step,
            value: self.value,
            timestamp: self.timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_record_new() {
        let metric = MetricRecord::new("run-1", "loss", 0, 0.5);
        assert_eq!(metric.run_id(), "run-1");
        assert_eq!(metric.key(), "loss");
        assert_eq!(metric.step(), 0);
        assert!((metric.value() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_record_ordering() {
        let m1 = MetricRecord::new("run-1", "loss", 0, 1.0);
        let m2 = MetricRecord::new("run-1", "loss", 1, 0.9);
        assert!(m1.step() < m2.step());
    }
}
