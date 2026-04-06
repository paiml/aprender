//! In-memory metrics store implementation

use super::error::StorageResult;
use super::traits::MetricsStore;
use crate::monitor::{Metric, MetricRecord, MetricStats};

/// In-memory metrics store (always available, no feature flag)
#[derive(Debug, Default)]
pub struct InMemoryStore {
    pub(crate) records: Vec<MetricRecord>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    /// Get all records
    pub fn all_records(&self) -> &[MetricRecord] {
        &self.records
    }
}

impl MetricsStore for InMemoryStore {
    fn write_batch(&mut self, records: &[MetricRecord]) -> StorageResult<()> {
        self.records.extend(records.iter().cloned());
        Ok(())
    }

    fn query_range(
        &self,
        metric: &Metric,
        start_ts: u64,
        end_ts: u64,
    ) -> StorageResult<Vec<MetricRecord>> {
        Ok(self
            .records
            .iter()
            .filter(|r| &r.metric == metric && r.timestamp >= start_ts && r.timestamp <= end_ts)
            .cloned()
            .collect())
    }

    fn query_all(&self, metric: &Metric) -> StorageResult<Vec<MetricRecord>> {
        Ok(self.records.iter().filter(|r| &r.metric == metric).cloned().collect())
    }

    fn query_stats(&self, metric: &Metric) -> StorageResult<Option<MetricStats>> {
        let values: Vec<f64> =
            self.records.iter().filter(|r| &r.metric == metric).map(|r| r.value).collect();

        if values.is_empty() {
            return Ok(None);
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let variance = if count > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let std = variance.sqrt();

        let has_nan = values.iter().any(|v| v.is_nan());
        let has_inf = values.iter().any(|v| v.is_infinite());

        Ok(Some(MetricStats { count, mean, std, min, max, sum, has_nan, has_inf }))
    }

    fn count(&self) -> StorageResult<usize> {
        Ok(self.records.len())
    }

    fn flush(&mut self) -> StorageResult<()> {
        Ok(())
    }
}
