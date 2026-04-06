//! Query filtering and configuration for snapshot storage.

use std::time::Duration;

use super::snapshot::SnapshotIndex;

/// Query filter for snapshots
#[derive(Debug, Clone, Default)]
pub struct SnapshotQuery {
    /// Start timestamp (inclusive)
    pub start_ns: Option<u64>,
    /// End timestamp (inclusive)
    pub end_ns: Option<u64>,
    /// Workload fingerprint filter
    pub fingerprint: Option<String>,
    /// Metric name filter
    pub metric_name: Option<String>,
    /// Maximum results
    pub limit: Option<usize>,
}

impl SnapshotQuery {
    /// Create new query
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by time range
    pub fn time_range(mut self, start_ns: u64, end_ns: u64) -> Self {
        self.start_ns = Some(start_ns);
        self.end_ns = Some(end_ns);
        self
    }

    /// Filter by fingerprint
    pub fn fingerprint(mut self, fp: impl Into<String>) -> Self {
        self.fingerprint = Some(fp.into());
        self
    }

    /// Filter by metric name
    pub fn metric(mut self, name: impl Into<String>) -> Self {
        self.metric_name = Some(name.into());
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Check if index matches query
    pub fn matches(&self, idx: &SnapshotIndex) -> bool {
        if let Some(start) = self.start_ns {
            if idx.timestamp_ns < start {
                return false;
            }
        }
        if let Some(end) = self.end_ns {
            if idx.timestamp_ns > end {
                return false;
            }
        }
        if let Some(ref fp) = self.fingerprint {
            if &idx.fingerprint != fp {
                return false;
            }
        }
        true
    }
}

/// Configuration for snapshot storage
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Maximum memory for query operations
    pub max_query_memory_bytes: usize,
    /// Keyframe interval (full snapshot every N deltas)
    pub keyframe_interval: usize,
    /// Enable checksum verification
    pub verify_checksums: bool,
    /// Raw tier max age
    pub raw_max_age: Duration,
    /// Compressed tier max age
    pub compressed_max_age: Duration,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_query_memory_bytes: 50 * 1024 * 1024, // 50MB
            keyframe_interval: 10,
            verify_checksums: true,
            raw_max_age: Duration::from_secs(24 * 60 * 60),
            compressed_max_age: Duration::from_secs(30 * 24 * 60 * 60),
        }
    }
}
