//! Incremental Profile Snapshots (PMAT-050)
//!
//! Time-series profile compression with differential storage and streaming decompression.
//!
//! # Design
//!
//! - Delta compression using XOR encoding for consecutive snapshots
//! - Tiered retention: raw (1 day) -> compressed (30 days) -> archive (1 year)
//! - Streaming decompression with bounded memory
//! - Index by timestamp and workload fingerprint for fast queries
//!
//! # Falsification (FKR-051)
//!
//! H0: Incremental snapshots cannot achieve <5% compression ratio
//! Test: Store 100 sequential snapshots, verify disk usage <5% of raw

mod query;
mod snapshot;
mod store;
mod types;

pub use query::{SnapshotConfig, SnapshotQuery};
pub use snapshot::{DeltaSnapshot, ProfileSnapshot, SnapshotIndex};
pub use store::IncrementalSnapshotStore;
pub use types::{DeltaMetric, MetricData, RetentionTier, SnapshotError, SnapshotResult};

/// Default keyframe interval
pub const DEFAULT_KEYFRAME_INTERVAL: usize = 10;

/// Default max query memory (bytes)
pub const DEFAULT_MAX_QUERY_MEMORY: usize = 50 * 1024 * 1024;

#[cfg(test)]
mod tests;
