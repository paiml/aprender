//! JSON file-based metrics store implementation

use std::path::Path;

use super::error::{StorageError, StorageResult};
use super::in_memory::InMemoryStore;
use super::traits::MetricsStore;
use crate::monitor::{Metric, MetricRecord, MetricStats};

/// JSON file-based metrics store
pub struct JsonFileStore {
    path: std::path::PathBuf,
    records: Vec<MetricRecord>,
    dirty: bool,
}

impl JsonFileStore {
    /// Create or open a JSON file store
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        let path = path.as_ref().to_path_buf();
        let records = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content)
                .map_err(|e| StorageError::Serialization(e.to_string()))?
        } else {
            Vec::new()
        };

        Ok(Self { path, records, dirty: false })
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl MetricsStore for JsonFileStore {
    fn write_batch(&mut self, records: &[MetricRecord]) -> StorageResult<()> {
        self.records.extend(records.iter().cloned());
        self.dirty = true;
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
        // Reuse InMemoryStore logic
        let mem_store = InMemoryStore { records: self.records.clone() };
        mem_store.query_stats(metric)
    }

    fn count(&self) -> StorageResult<usize> {
        Ok(self.records.len())
    }

    fn flush(&mut self) -> StorageResult<()> {
        if self.dirty {
            let json = serde_json::to_string_pretty(&self.records)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            std::fs::write(&self.path, json)?;
            self.dirty = false;
        }
        Ok(())
    }
}

impl Drop for JsonFileStore {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
