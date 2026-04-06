//! Metrics storage trait definitions

use super::error::StorageResult;
use crate::monitor::{Metric, MetricRecord, MetricStats};

/// Metrics storage backend trait
pub trait MetricsStore: Send + Sync {
    /// Write a batch of metric records
    fn write_batch(&mut self, records: &[MetricRecord]) -> StorageResult<()>;

    /// Query metrics by name within a time range
    fn query_range(
        &self,
        metric: &Metric,
        start_ts: u64,
        end_ts: u64,
    ) -> StorageResult<Vec<MetricRecord>>;

    /// Get all records for a metric
    fn query_all(&self, metric: &Metric) -> StorageResult<Vec<MetricRecord>>;

    /// Get summary statistics for a metric
    fn query_stats(&self, metric: &Metric) -> StorageResult<Option<MetricStats>>;

    /// Get total record count
    fn count(&self) -> StorageResult<usize>;

    /// Flush pending writes
    fn flush(&mut self) -> StorageResult<()>;
}
